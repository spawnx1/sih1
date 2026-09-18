from __future__ import annotations

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

import config
from graphx.candidates import generate_candidates
from graphx.hetero_graph import build_heterogeneous_graph
from graphx.node_features import build_node_feature_matrix
from ml.features import FeatureContext
from ml.train import build_m3_events


SEED = config.SEED
HIDDEN = 64
EPOCHS = 5
LR = 5e-4
WEIGHT_DECAY = 1e-4
NEGATIVES_PER_POSITIVE = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class HeteroEmbedder(nn.Module):

    def __init__(self, account_in, atm_in, hidden=64):
        super().__init__()

        self.account_proj = nn.Linear(account_in, hidden)
        self.atm_proj = nn.Linear(atm_in, hidden)

        def make_relations():
            return {
                ("account", "transfer", "account"):
                    SAGEConv(hidden, hidden),

                ("account", "transfer_rev", "account"):
                    SAGEConv(hidden, hidden),

                ("account", "cashout", "atm"):
                    SAGEConv((hidden, hidden), hidden),

                ("atm", "cashout_rev", "account"):
                    SAGEConv((hidden, hidden), hidden),
            }

        self.conv1 = HeteroConv(
            make_relations(),
            aggr="sum"
        )

        self.conv2 = HeteroConv(
            make_relations(),
            aggr="sum"
        )

    def forward(self, data):

        x_dict = {
            "account": F.relu(
                self.account_proj(data["account"].x)
            ),
            "atm": F.relu(
                self.atm_proj(data["atm"].x)
            ),
        }

        x_dict = self.conv1(
            x_dict,
            data.edge_index_dict
        )

        x_dict = {
            k: F.relu(v)
            for k, v in x_dict.items()
        }

        x_dict = self.conv2(
            x_dict,
            data.edge_index_dict
        )

        return {
            k: F.normalize(v, p=2, dim=1)
            for k, v in x_dict.items()
        }


def build_atm_features(data):

    n = int(data["atm"].num_nodes)

    x = torch.zeros(
        (n, 2),
        dtype=torch.float32
    )

    rel = data[
        "account",
        "cashout",
        "atm"
    ]

    if rel.edge_index.numel() == 0:
        return x

    atm_idx = rel.edge_index[1]

    degree = torch.bincount(
        atm_idx,
        minlength=n
    ).float()

    amount = rel.edge_attr[:, 0].float()

    total = torch.zeros(n)

    total.scatter_add_(
        0,
        atm_idx,
        amount
    )

    x[:, 0] = torch.log1p(degree)
    x[:, 1] = torch.log1p(total)

    return x


def build_snapshot(ctx, snapshot):

    data, account_to_idx, atm_to_idx = (
        build_heterogeneous_graph(
            ctx,
            as_of=snapshot,
            observed_only=True
        )
    )

    account_x = build_node_feature_matrix(
        ctx,
        account_to_idx,
        snapshot,
    )

    data["account"].x = account_x.float()
    data["atm"].x = build_atm_features(data)

    transfer = data[
        "account",
        "transfer",
        "account"
    ]

    data[
        "account",
        "transfer_rev",
        "account"
    ].edge_index = transfer.edge_index.flip(0)

    cashout = data[
        "account",
        "cashout",
        "atm"
    ]

    data[
        "atm",
        "cashout_rev",
        "account"
    ].edge_index = cashout.edge_index.flip(0)

    return data, account_to_idx, atm_to_idx


def make_examples(ctx, events):

    examples = []

    for ev in events:

        if not ev["is_tainted"]:
            continue

        true_atm = str(ev["terminal"])

        if true_atm == "None":
            continue

        candidates = [
            str(x)
            for x in generate_candidates(
                ev["account"],
                ev["as_of"],
                ctx
            )
        ]

        if true_atm not in candidates:
            continue

        examples.append({
            "account": str(ev["account"]),
            "snapshot": pd.Timestamp(ev["as_of"]),
            "true_atm": true_atm,
            "candidates": candidates
        })

    return examples


def select_windows(examples, max_windows=10):

    days = sorted(
        set(
            pd.Timestamp(x["snapshot"]).normalize()
            for x in examples
        )
    )

    if len(days) <= max_windows:
        return days

    indices = np.linspace(
        0,
        len(days) - 1,
        max_windows,
        dtype=int
    )

    return [days[i] for i in indices]


def select_examples_from_windows(examples, windows):

    selected = set(windows)

    return [
        x for x in examples
        if pd.Timestamp(x["snapshot"]).normalize() in selected
    ]


def attach_graph_snapshot(examples):

    result = []

    for x in examples:

        y = dict(x)

        # Graph is frozen at the beginning of the day.
        # Therefore transactions later that day cannot leak
        # into the graph representation.

        y["graph_snapshot"] = (
            pd.Timestamp(x["snapshot"]).normalize()
        )

        result.append(y)

    return result

def train_model(ctx, examples):

    windows = select_windows(
        examples,
        max_windows=10
    )

    examples = select_examples_from_windows(
        examples,
        windows
    )

    examples = attach_graph_snapshot(
        examples
    )

    print()
    print(
        "Selected training windows:",
        len(windows)
    )

    print(
        "Training events in windows:",
        len(examples)
    )

    first = examples[0]

    data, _, _ = build_snapshot(
        ctx,
        first["graph_snapshot"]
    )

    account_in = data["account"].x.shape[1]
    atm_in = data["atm"].x.shape[1]

    model = HeteroEmbedder(
        account_in,
        atm_in,
        HIDDEN
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    print()
    print("WINDOWED FAST MODEL")
    print("Account features:", account_in)
    print("ATM features:", atm_in)
    print("Hidden:", HIDDEN)
    print("Epochs:", EPOCHS)

    cache = {}

    for epoch in range(1, EPOCHS + 1):

        random.shuffle(examples)

        model.train()

        total_loss = 0.0
        used = 0

        for ex in examples:

            key = str(ex["graph_snapshot"])

            if key not in cache:

                cache[key] = build_snapshot(
                    ctx,
                    ex["graph_snapshot"]
                )

            data, account_map, atm_map = cache[key]

            account = ex["account"]
            true_atm = ex["true_atm"]

            if account not in account_map:
                continue

            if true_atm not in atm_map:
                continue

            data_device = data.to(DEVICE)

            z = model(data_device)

            account_z = z["account"][
                account_map[account]
            ]

            true_z = z["atm"][
                atm_map[true_atm]
            ]

            negatives = [
                x for x in ex["candidates"]
                if x != true_atm and x in atm_map
            ]

            if not negatives:
                continue

            random.shuffle(negatives)

            negatives = negatives[
                :NEGATIVES_PER_POSITIVE
            ]

            neg_idx = torch.tensor(
                [atm_map[x] for x in negatives],
                dtype=torch.long,
                device=DEVICE
            )

            neg_z = z["atm"][neg_idx]

            positive = torch.sum(
                account_z * true_z
            )

            negative_scores = torch.matmul(
                neg_z,
                account_z
            )

            scores = torch.cat([
                positive.view(1),
                negative_scores
            ])

            targets = torch.zeros(
                scores.shape[0],
                dtype=torch.float32,
                device=DEVICE
            )

            targets[0] = 1.0

            loss = F.binary_cross_entropy_with_logits(
                scores,
                targets
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            total_loss += float(
                loss.detach()
            )

            used += 1

        if epoch == 1 or epoch == EPOCHS:

            print(
                f"Epoch {epoch:02d} "
                f"Loss={total_loss / max(used,1):.6f} "
                f"Examples={used:,}"
            )

    return model


@torch.no_grad()
@torch.no_grad()
def rank_events(ctx, model, examples):

    model.eval()

    windows = select_windows(
        examples,
        max_windows=10
    )

    examples = select_examples_from_windows(
        examples,
        windows
    )

    examples = attach_graph_snapshot(
        examples
    )

    print()
    print(
        "Selected evaluation windows:",
        len(windows)
    )

    print(
        "Evaluation events in windows:",
        len(examples)
    )

    cache = {}
    results = []

    for ex in examples:

        key = str(ex["graph_snapshot"])

        if key not in cache:

            cache[key] = build_snapshot(
                ctx,
                ex["graph_snapshot"]
            )

        data, account_map, atm_map = cache[key]

        account = ex["account"]
        true_atm = ex["true_atm"]

        if account not in account_map:
            continue

        data_device = data.to(DEVICE)

        z = model(data_device)

        account_z = z["account"][
            account_map[account]
        ]

        candidates = [
            atm for atm in ex["candidates"]
            if atm in atm_map
        ]

        if not candidates:
            continue

        idx = torch.tensor(
            [atm_map[x] for x in candidates],
            dtype=torch.long,
            device=DEVICE
        )

        scores = torch.matmul(
            z["atm"][idx],
            account_z
        )

        order = torch.argsort(
            scores,
            descending=True
        )

        ranked = [
            candidates[i]
            for i in order.cpu().tolist()
        ]

        if true_atm in ranked:
            rank = ranked.index(true_atm) + 1
        else:
            rank = 999999

        results.append({
            "account": account,
            "snapshot": ex["snapshot"],
            "true_atm": true_atm,
            "candidate_count": len(ranked),
            "rank": rank
        })

    return pd.DataFrame(results)


def candidate_baseline(examples):

    results = []

    for ex in examples:

        candidates = list(ex["candidates"])
        true_atm = ex["true_atm"]

        if true_atm in candidates:
            rank = candidates.index(true_atm) + 1
        else:
            rank = 999999

        results.append({
            "account": ex["account"],
            "snapshot": ex["snapshot"],
            "true_atm": true_atm,
            "candidate_count": len(candidates),
            "rank": rank
        })

    return pd.DataFrame(results)


def report_comparison(candidate_df, gnn_df):

    print()
    print("=" * 70)
    print("SPRINT 9B - CANDIDATE VS GRAPH GNN")
    print("=" * 70)

    print(
        f"Candidate events : {len(candidate_df):,}"
    )

    print(
        f"GNN events       : {len(gnn_df):,}"
    )

    print()
    print("CANDIDATE BASELINE")

    for k in [1, 3, 5, 10, 50]:

        recall = (
            candidate_df["rank"] <= k
        ).mean()

        print(
            f"Top-{k:<2} Recall : {recall:.4f}"
        )

    reciprocal = np.where(
        candidate_df["rank"] < 999999,
        1.0 / candidate_df["rank"],
        0.0
    )

    print(
        f"MRR           : {reciprocal.mean():.4f}"
    )

    print()
    print("GRAPH GNN")

    for k in [1, 3, 5, 10, 50]:

        recall = (
            gnn_df["rank"] <= k
        ).mean()

        print(
            f"Top-{k:<2} Recall : {recall:.4f}"
        )

    reciprocal = np.where(
        gnn_df["rank"] < 999999,
        1.0 / gnn_df["rank"],
        0.0
    )

    print(
        f"MRR           : {reciprocal.mean():.4f}"
    )


def report(df):

    print()
    print("=" * 70)
    print("SPRINT 9B FAST - GRAPH ATM RANKING")
    print("=" * 70)

    print(
        f"Evaluated events : {len(df):,}"
    )

    if df.empty:
        print("NO VALID TEST EVENTS")
        return

    for k in [1, 3, 5, 10, 50]:

        recall = (
            df["rank"] <= k
        ).mean()

        print(
            f"Top-{k:<2} Recall      : "
            f"{recall:.4f}"
        )

    reciprocal = np.where(
        df["rank"] < 999999,
        1.0 / df["rank"],
        0.0
    )

    print(
        f"MRR              : "
        f"{reciprocal.mean():.4f}"
    )

    valid = df.loc[
        df["rank"] < 999999,
        "rank"
    ]

    if len(valid):
        print(
            f"Median rank      : "
            f"{valid.median():.1f}"
        )
    else:
        print("Median rank      : N/A")



def fusion_score(tabular_score, graph_score, graph_weight=0.30):

    tabular_score = np.asarray(
        tabular_score,
        dtype=float
    )

    graph_score = np.asarray(
        graph_score,
        dtype=float
    )

    return (
        (1.0 - graph_weight) * tabular_score
        + graph_weight * graph_score
    )


def run_fusion_demo():

    print()
    print("=" * 70)
    print("SPRINT 10 - TABULAR + GRAPH FUSION")
    print("=" * 70)

    print()
    print("Architecture:")
    print("  Tabular ML risk")
    print("        +")
    print("  Graph risk")
    print("        ?")
    print("  Fusion risk")

    # Demonstration using the frozen Sprint 9B ranking results.
    #
    # Rank is converted into a normalized graph-risk signal:
    # rank 1 = strongest graph signal
    # rank 50 = weakest signal inside candidate set.
    #
    # This is a ranking-derived graph signal, not a trained
    # probability. It is intentionally kept separate from the
    # existing tabular model.

    candidate = np.array([
        0.3333,
        0.6111,
        0.6481,
        0.8704,
        1.0000
    ])

    graph = np.array([
        0.0556,
        0.0926,
        0.1111,
        0.2963,
        1.0000
    ])

    for w in [0.10, 0.20, 0.30, 0.40, 0.50]:

        fused = fusion_score(
            candidate,
            graph,
            graph_weight=w
        )

        print()
        print(
            f"Graph weight = {w:.1f}"
        )

        print(
            "Fused values:",
            np.round(fused, 4)
        )

    print()
    print("IMPORTANT:")
    print("This is the Sprint 10 fusion pipeline prototype.")
    print("It does NOT claim improved predictive accuracy.")
    print("Final fusion must be trained/evaluated on event-level labels.")

def main():

    print("=" * 70)
    print("SPRINT 9B - FAST GRAPH-BASED ATM RANKER")
    print("=" * 70)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print()
    print("BUILDING EVENTS...")

    ctx = FeatureContext()

    events = build_m3_events(
        ctx,
        SEED,
        n_legit=3000
    )

    print(
        f"Tainted events          : "
        f"{sum(e['is_tainted'] for e in events):,}"
    )

    examples = make_examples(
        ctx,
        events
    )

    print(
        f"Candidate-covered events: "
        f"{len(examples):,}"
    )

    examples = sorted(
        examples,
        key=lambda x: x["snapshot"]
    )

    cutoff_index = int(
        len(examples) * 0.70
    )

    cutoff = examples[
        cutoff_index
    ]["snapshot"]

    train = [
        x for x in examples
        if x["snapshot"] < cutoff
    ]

    test = [
        x for x in examples
        if x["snapshot"] >= cutoff
    ]

    print()
    print("TEMPORAL SPLIT")
    print("Cutoff:", cutoff)
    print("Train events:", len(train))
    print("Test events:", len(test))

    # Use the same selected temporal snapshots for both
    # candidate baseline and GNN evaluation.

    test_windows = select_windows(
        test,
        max_windows=10
    )

    test_eval = select_examples_from_windows(
        test,
        test_windows
    )

    print()
    print(
        "Selected test windows:",
        len(test_windows)
    )

    print(
        "Events in selected test windows:",
        len(test_eval)
    )

    print()
    print("CANDIDATE BASELINE...")

    candidate_df = candidate_baseline(
        test_eval
    )

    model = train_model(
        ctx,
        train
    )

    print()
    print("EVALUATING GRAPH GNN...")

    result = rank_events(
        ctx,
        model,
        test_eval
    )

    report(result)

    report_comparison(
        candidate_df,
        result
    )

    print()
    print("SPRINT 9B FAST COMPLETE")

    run_fusion_demo()

    print()
    print("SPRINT 10 FUSION PROTOTYPE COMPLETE")


if __name__ == "__main__":
    main()
