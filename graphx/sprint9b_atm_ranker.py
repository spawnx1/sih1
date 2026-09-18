# SPRINT 9B - GRAPH-BASED ATM RANKER

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

        relations = {
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
            relations,
            aggr="sum"
        )

        self.conv2 = HeteroConv(
            relations,
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


def train_model(ctx, examples):

    examples = examples[:250]
    first = examples[0]

    data, _, _ = build_snapshot(
        ctx,
        first["snapshot"]
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
    print("MODEL")
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

            key = str(ex["snapshot"])

            if key not in cache:
                cache[key] = build_snapshot(
                    ctx,
                    ex["snapshot"]
                )

            data, account_map, atm_map = cache[key]

            account = ex["account"]
            true_atm = ex["true_atm"]

            if account not in account_map:
                continue

            if true_atm not in atm_map:
                continue

            data = data.to(DEVICE)

            z = model(data)

            account_z = z["account"][
                account_map[account]
            ]

            true_z = z["atm"][
                atm_map[true_atm]
            ]

            positive = torch.sum(
                account_z * true_z
            )

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

        if epoch == 1 or epoch % 5 == 0:

            avg = (
                total_loss /
                max(used, 1)
            )

            print(
                f"Epoch {epoch:02d} "
                f"Loss={avg:.6f} "
                f"Examples={used:,}"
            )

    return model


@torch.no_grad()
def rank_events(ctx, model, examples):

    model.eval()

    cache = {}

    results = []

    for ex in examples:

        key = str(ex["snapshot"])

        if key not in cache:
            cache[key] = build_snapshot(
                ctx,
                ex["snapshot"]
            )

        data, account_map, atm_map = cache[key]

        account = ex["account"]
        true_atm = ex["true_atm"]

        if account not in account_map:
            continue

        data = data.to(DEVICE)

        z = model(data)

        account_z = z["account"][
            account_map[account]
        ]

        scored = []

        for atm in ex["candidates"]:

            if atm not in atm_map:
                continue

            score = torch.sum(
                account_z *
                z["atm"][atm_map[atm]]
            ).item()

            scored.append(
                (atm, score)
            )

        scored.sort(
            key=lambda x: x[1],
            reverse=True
        )

        ranked = [
            atm for atm, _ in scored
        ]

        if true_atm in ranked:
            rank = (
                ranked.index(true_atm) +
                1
            )
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


def report(df):

    print()
    print("=" * 70)
    print("SPRINT 9B - GRAPH ATM RANKING RESULTS")
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


def main():

    print("=" * 70)
    print("SPRINT 9B - GRAPH-BASED ATM RANKER")
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

    ctx = FeatureContext(
        config.DATA_DIR
    )

    print()
    print("BUILDING EVENTS...")

    events = build_m3_events(
        ctx,
        seed=config.SEED,
        n_legit=3000
    )

    tainted = [
        e for e in events
        if e["is_tainted"]
    ]

    examples = make_examples(
        ctx,
        tainted
    )

    print(
        f"Tainted events          : "
        f"{len(tainted):,}"
    )

    print(
        f"Candidate-covered events: "
        f"{len(examples):,}"
    )

    times = pd.Series([
        e["snapshot"]
        for e in examples
    ])

    cut = times.quantile(0.70)

    train = [
        e for e in examples
        if e["snapshot"] < cut
    ]

    test = [
        e for e in examples
        if e["snapshot"] >= cut
    ]

    print()
    print("TEMPORAL SPLIT")
    print("Cutoff:", cut)
    print("Train events:", len(train))
    print("Test events:", len(test))

    model = train_model(
        ctx,
        train
    )

    print()
    baseline = candidate_baseline(ctx, test)
    report_baseline(baseline)

    print("EVALUATING...")

    test = test[:50]
    print(f"Evaluation events: {len(test)}")
    result = rank_events(
        ctx,
        model,
        test
    )

    report(result)

    print()
    print("SPRINT 9B COMPLETE")


def candidate_baseline(ctx, examples):
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
            "rank": rank,
        })

    return pd.DataFrame(results)


def report_baseline(df):
    ranks = df["rank"].to_numpy()

    print()
    print("=" * 70)
    print("SPRINT 9B - CANDIDATE ORDER BASELINE")
    print("=" * 70)
    print(f"Evaluated events : {len(df):,}")

    for k in [1, 3, 5, 10, 50]:
        recall = (ranks <= k).mean()
        print(f"Top-{k:<2} Recall      : {recall:.4f}")

    valid = ranks[ranks < 999999]

    if len(valid):
        print(f"MRR              : {(1.0 / valid).mean():.4f}")
        print(f"Median rank      : {float(pd.Series(valid).median()):.1f}")
    else:
        print("MRR              : 0.0000")
        print("Median rank      : N/A")

if __name__ == "__main__":
    main()




