"""
Sprint 7D - Temporal Heterogeneous GNN

Temporal snapshots + heterogeneous financial relations.

Relations:
    Account -> Account : transfer
    Account -> ATM     : cashout
    ATM -> Account     : cashout_rev

The reverse cash-out relation is used so that ATM/cash-out
information can propagate back into the Account representation.

Prediction:
    Future cash-out within the next 60 minutes.

No future transactions are included in the graph at snapshot T.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from ml.features import FeatureContext
from graphx.hetero_graph import build_heterogeneous_graph
from graphx.node_features import build_node_feature_matrix
from graphx.temporal_activity_dataset import (
    build_activity_buckets,
    choose_activity_snapshots,
    MAX_SNAPSHOTS,
    HORIZON_MIN,
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

HIDDEN = 64
EPOCHS = 20
LR = 0.0005
WEIGHT_DECAY = 1e-4


# ============================================================
# MODEL
# ============================================================

class TemporalHeteroGNN(nn.Module):

    def __init__(self, account_in, atm_in, hidden=64):

        super().__init__()

        relations1 = {
            ("account", "transfer", "account"):
                SAGEConv(
                    (account_in, account_in),
                    hidden,
                ),

            ("account", "transfer_rev", "account"):
                SAGEConv(
                    (account_in, account_in),
                    hidden,
                ),

            ("account", "cashout", "atm"):
                SAGEConv(
                    (account_in, atm_in),
                    hidden,
                ),

            ("atm", "cashout_rev", "account"):
                SAGEConv(
                    (atm_in, account_in),
                    hidden,
                ),
        }

        relations2 = {
            ("account", "transfer", "account"):
                SAGEConv(
                    (hidden, hidden),
                    hidden,
                ),

            ("account", "transfer_rev", "account"):
                SAGEConv(
                    (hidden, hidden),
                    hidden,
                ),

            ("account", "cashout", "atm"):
                SAGEConv(
                    (hidden, hidden),
                    hidden,
                ),

            ("atm", "cashout_rev", "account"):
                SAGEConv(
                    (hidden, hidden),
                    hidden,
                ),
        }

        self.conv1 = HeteroConv(
            relations1,
            aggr="sum",
        )

        self.conv2 = HeteroConv(
            relations2,
            aggr="sum",
        )

        self.out = nn.Linear(hidden, 1)

    def forward(self, x_dict, edge_index_dict):

        x = self.conv1(
            x_dict,
            edge_index_dict,
        )

        x = {
            k: F.relu(v)
            for k, v in x.items()
        }

        x = self.conv2(
            x,
            edge_index_dict,
        )

        x = {
            k: F.relu(v)
            for k, v in x.items()
        }

        return self.out(
            x["account"]
        ).squeeze(-1)


# ============================================================
# FUTURE LABELS
# ============================================================

def future_cashout_labels(
    ctx,
    snapshot,
    account_ids,
):

    snapshot = np.datetime64(snapshot)

    end = (
        snapshot
        + np.timedelta64(HORIZON_MIN, "m")
    )

    account_to_idx = {
        str(account): i
        for i, account in enumerate(account_ids)
    }

    labels = np.zeros(
        len(account_ids),
        dtype=np.float32,
    )

    for account, positions in ctx.src_pos.items():

        key = str(account)

        if key not in account_to_idx:
            continue

        for pos in positions:

            ts = ctx.ts[pos]

            if not (
                snapshot < ts <= end
            ):
                continue

            if not ctx.is_cashout[pos]:
                continue

            labels[
                account_to_idx[key]
            ] = 1.0

            break

    return torch.tensor(
        labels,
        dtype=torch.float32,
    )


# ============================================================
# ATM FEATURES
# ============================================================

def build_atm_features(data):

    num_atms = int(
        data["atm"].num_nodes
    )

    features = torch.zeros(
        (num_atms, 2),
        dtype=torch.float32,
    )

    edge_index = data[
        "account",
        "cashout",
        "atm",
    ].edge_index

    edge_attr = data[
        "account",
        "cashout",
        "atm",
    ].edge_attr

    if edge_index.numel() == 0:
        features[:, 0] = 0.0
        features[:, 1] = 0.0
        return features

    atm_index = edge_index[1]

    degree = torch.bincount(
        atm_index,
        minlength=num_atms,
    ).float()

    amount = edge_attr[:, 0].float()

    total_amount = torch.zeros(
        num_atms,
        dtype=torch.float32,
    )

    total_amount.scatter_add_(
        0,
        atm_index,
        amount,
    )

    features[:, 0] = torch.log1p(
        degree
    )

    features[:, 1] = torch.log1p(
        total_amount
    )

    return features


# ============================================================
# BUILD ONE TEMPORAL HETERO SNAPSHOT
# ============================================================

def build_snapshot(ctx, snapshot):

    data, account_to_idx, atm_to_idx = (
        build_heterogeneous_graph(
            ctx,
            as_of=snapshot,
            observed_only=True,
        )
    )

    account_x = build_node_feature_matrix(
        ctx,
        account_to_idx,
        snapshot,
        trail=None,
    )

    atm_x = build_atm_features(
        data
    )

    data["account"].x = account_x
    data["atm"].x = atm_x

    # --------------------------------------------------------
    # Reverse transfer relation
    # --------------------------------------------------------

    transfer = data[
        "account",
        "transfer",
        "account",
    ]

    data[
        "account",
        "transfer_rev",
        "account",
    ].edge_index = transfer.edge_index.flip(0)

    data[
        "account",
        "transfer_rev",
        "account",
    ].edge_attr = transfer.edge_attr

    data[
        "account",
        "transfer_rev",
        "account",
    ].edge_time = transfer.edge_time

    # --------------------------------------------------------
    # Reverse cash-out relation
    # --------------------------------------------------------

    cashout = data[
        "account",
        "cashout",
        "atm",
    ]

    data[
        "atm",
        "cashout_rev",
        "account",
    ].edge_index = cashout.edge_index.flip(0)

    data[
        "atm",
        "cashout_rev",
        "account",
    ].edge_attr = cashout.edge_attr

    data[
        "atm",
        "cashout_rev",
        "account",
    ].edge_time = cashout.edge_time

    labels = future_cashout_labels(
        ctx,
        snapshot,
        data.account_id,
    )

    return data, labels


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_snapshots(
    snapshots,
    train_count,
):

    train_x = torch.cat(
        [
            snapshots[i][0]["account"].x
            for i in range(train_count)
        ],
        dim=0,
    )

    mean = train_x.mean(
        dim=0,
        keepdim=True,
    )

    std = train_x.std(
        dim=0,
        keepdim=True,
    ).clamp_min(1e-6)

    for data, _ in snapshots:

        data["account"].x = (
            data["account"].x - mean
        ) / std

    return mean, std


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SPRINT 7D - TEMPORAL HETEROGENEOUS GNN")
    print("=" * 70)

    print(
        "Device:",
        DEVICE,
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    ctx = FeatureContext()

    print()
    print("BUILDING ACTIVITY SNAPSHOTS...")

    buckets = build_activity_buckets(ctx)

    snapshot_times = choose_activity_snapshots(
        ctx,
        buckets,
        MAX_SNAPSHOTS,
    )

    snapshots = []

    for i, snapshot in enumerate(
        snapshot_times,
        1,
    ):

        data, labels = build_snapshot(
            ctx,
            snapshot,
        )

        nodes = int(
            data["account"].num_nodes
        )

        transfers = int(
            data[
                "account",
                "transfer",
                "account",
            ].edge_index.shape[1]
        )

        cashouts = int(
            data[
                "account",
                "cashout",
                "atm",
            ].edge_index.shape[1]
        )

        positives = int(
            labels.sum()
        )

        print(
            f"[{i:02d}/{len(snapshot_times)}] "
            f"{str(snapshot)[:16]} | "
            f"Nodes: {nodes} | "
            f"Transfers: {transfers} | "
            f"Cash-outs: {cashouts} | "
            f"Positives: {positives}"
        )

        if nodes > 0:
            snapshots.append(
                (data, labels)
            )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    split = int(
        len(snapshots) * 0.70
    )

    train_snapshots = snapshots[
        :split
    ]

    test_snapshots = snapshots[
        split:
    ]

    train_pos = sum(
        int(y.sum())
        for _, y in train_snapshots
    )

    train_total = sum(
        len(y)
        for _, y in train_snapshots
    )

    test_pos = sum(
        int(y.sum())
        for _, y in test_snapshots
    )

    test_total = sum(
        len(y)
        for _, y in test_snapshots
    )

    print()
    print("TEMPORAL SPLIT")
    print(
        "Train snapshots:",
        len(train_snapshots),
    )
    print(
        "Test snapshots:",
        len(test_snapshots),
    )
    print(
        "Train positives:",
        train_pos,
    )
    print(
        "Train negatives:",
        train_total - train_pos,
    )
    print(
        "Test positives:",
        test_pos,
    )
    print(
        "Test negatives:",
        test_total - test_pos,
    )

    # --------------------------------------------------------
    # Normalize using train snapshots only
    # --------------------------------------------------------

    normalize_snapshots(
        snapshots,
        split,
    )

    account_in = train_snapshots[
        0
    ][0]["account"].x.shape[1]

    atm_in = train_snapshots[
        0
    ][0]["atm"].x.shape[1]

    print()
    print("MODEL")
    print(
        "Account features:",
        account_in,
    )
    print(
        "ATM features:",
        atm_in,
    )
    print(
        "Hidden dimensions:",
        HIDDEN,
    )

    model = TemporalHeteroGNN(
        account_in=account_in,
        atm_in=atm_in,
        hidden=HIDDEN,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    pos_weight = min(
        train_total / max(train_pos, 1),
        20.0,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight,
            device=DEVICE,
        )
    )

    print(
        "Positive weight:",
        round(pos_weight, 4),
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print("TRAINING")

    model.train()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        total_loss = 0.0

        for data, labels in train_snapshots:

            data = data.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(
                data.x_dict,
                data.edge_index_dict,
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            total_loss += loss.item()

        if (
            epoch == 1
            or epoch % 5 == 0
        ):

            avg_loss = (
                total_loss
                / max(len(train_snapshots), 1)
            )

            print(
                f"Epoch {epoch:02d} "
                f"Loss={avg_loss:.6f}"
            )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        brier_score_loss,
    )

    model.eval()

    all_probs = []
    all_y = []

    with torch.no_grad():

        for data, labels in test_snapshots:

            data = data.to(DEVICE)

            logits = model(
                data.x_dict,
                data.edge_index_dict,
            )

            probs = torch.sigmoid(
                logits
            ).cpu()

            all_probs.append(
                probs
            )

            all_y.append(
                labels
            )

    probs = torch.cat(
        all_probs
    ).numpy()

    y = torch.cat(
        all_y
    ).numpy()

    roc = roc_auc_score(
        y,
        probs,
    )

    pr = average_precision_score(
        y,
        probs,
    )

    brier = brier_score_loss(
        y,
        probs,
    )

    print()
    print("EVALUATION")
    print()
    print("TEMPORAL HETEROGENEOUS GNN RESULTS")
    print(
        f"ROC-AUC : {roc:.4f}"
    )
    print(
        f"PR-AUC  : {pr:.4f}"
    )
    print(
        f"Brier   : {brier:.4f}"
    )
    print(
        "Test positives:",
        int(y.sum()),
    )
    print(
        "Test negatives:",
        int((y == 0).sum()),
    )

    print()
    print("=" * 70)
    print("SPRINT 7D COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
