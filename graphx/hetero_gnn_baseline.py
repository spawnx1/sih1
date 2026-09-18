"""
Sprint 7C - Heterogeneous GNN Baseline

Account -> Account : transfer
Account -> ATM     : cashout

Task:
Predict whether an account will perform a cash-out
within the next 60 minutes.

IMPORTANT:
- Graph uses only transactions before snapshot time.
- Labels use future transactions after snapshot time.
- Labels are never graph features.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, SAGEConv

from ml.features import FeatureContext
from graphx.hetero_graph import build_heterogeneous_graph
from graphx.node_features import build_node_feature_matrix


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HORIZON_MIN = 60

TRAIN_SNAPSHOT = "2026-07-14T15:50:00"
TEST_SNAPSHOT = "2026-08-21T13:20:00"


# ============================================================
# HETEROGENEOUS GRAPH SAGE
# ============================================================

class HeteroGraphSAGE(nn.Module):

    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()

        self.conv1 = HeteroConv(
            {
                ("account", "transfer", "account"):
                    SAGEConv(
                        (in_channels, in_channels),
                        hidden_channels
                    ),

                ("account", "cashout", "atm"):
                    SAGEConv(
                        (in_channels, 1),
                        hidden_channels
                    ),
            },
            aggr="sum",
        )

        self.conv2 = HeteroConv(
            {
                ("account", "transfer", "account"):
                    SAGEConv(
                        (hidden_channels, hidden_channels),
                        hidden_channels
                    ),

                ("account", "cashout", "atm"):
                    SAGEConv(
                        (hidden_channels, hidden_channels),
                        hidden_channels
                    ),
            },
            aggr="sum",
        )

        self.account_out = nn.Linear(hidden_channels, 1)

    def forward(self, x_dict, edge_index_dict):

        x_dict = self.conv1(
            x_dict,
            edge_index_dict
        )

        x_dict = {
            key: F.relu(value)
            for key, value in x_dict.items()
        }

        x_dict = self.conv2(
            x_dict,
            edge_index_dict
        )

        x_dict = {
            key: F.relu(value)
            for key, value in x_dict.items()
        }

        return self.account_out(
            x_dict["account"]
        ).squeeze(-1)


# ============================================================
# FUTURE CASH-OUT LABELS
# ============================================================

def future_cashout_labels(
    ctx,
    snapshot,
    account_ids,
):
    snapshot64 = np.datetime64(snapshot)

    end = (
        snapshot64
        + np.timedelta64(HORIZON_MIN, "m")
    )

    labels = np.zeros(
        len(account_ids),
        dtype=np.float32
    )

    account_to_idx = {
        str(account): i
        for i, account in enumerate(account_ids)
    }

    for account, positions in ctx.src_pos.items():

        account_key = str(account)

        if account_key not in account_to_idx:
            continue

        for pos in positions:

            ts = ctx.ts[pos]

            if not (
                snapshot64 < ts <= end
            ):
                continue

            if not ctx.is_cashout[pos]:
                continue

            labels[
                account_to_idx[account_key]
            ] = 1.0

            break

    return torch.tensor(
        labels,
        dtype=torch.float32
    )


# ============================================================
# BUILD GRAPH + FEATURES
# ============================================================

def build_training_graph(ctx, snapshot):

    data, account_to_idx, atm_to_idx = (
        build_heterogeneous_graph(
            ctx,
            as_of=snapshot,
            observed_only=True,
        )
    )

    # Existing 8 graph features.
    x_account = build_node_feature_matrix(
        ctx,
        account_to_idx,
        snapshot,
        trail=None,
    )

    # ATM structural feature:
    # simple constant feature for baseline.
    x_atm = torch.ones(
        data["atm"].num_nodes,
        1,
        dtype=torch.float32,
    )

    data["account"].x = x_account
    data["atm"].x = x_atm

    labels = future_cashout_labels(
        ctx,
        snapshot,
        data["account_id"],
    )

    return data, labels


# ============================================================
# METRICS
# ============================================================

def evaluate_binary(y_true, logits):

    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        brier_score_loss,
    )

    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y = y_true.detach().cpu().numpy()

    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")

    roc = roc_auc_score(y, probs)
    pr = average_precision_score(y, probs)
    brier = brier_score_loss(y, probs)

    return roc, pr, brier


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SPRINT 7C - HETEROGENEOUS GNN BASELINE")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    ctx = FeatureContext()

    train_snapshot = np.datetime64(
        TRAIN_SNAPSHOT
    )

    test_snapshot = np.datetime64(
        TEST_SNAPSHOT
    )

    print()
    print("BUILDING TRAIN GRAPH...")

    train_data, train_y = build_training_graph(
        ctx,
        train_snapshot,
    )

    print(
        "Train accounts:",
        train_data["account"].num_nodes
    )

    print(
        "Train ATMs:",
        train_data["atm"].num_nodes
    )

    print(
        "Train transfers:",
        train_data[
            "account",
            "transfer",
            "account"
        ].edge_index.shape[1]
    )

    print(
        "Train cash-outs:",
        train_data[
            "account",
            "cashout",
            "atm"
        ].edge_index.shape[1]
    )

    print(
        "Train positives:",
        int(train_y.sum())
    )

    print()
    print("BUILDING TEST GRAPH...")

    test_data, test_y = build_training_graph(
        ctx,
        test_snapshot,
    )

    print(
        "Test accounts:",
        test_data["account"].num_nodes
    )

    print(
        "Test ATMs:",
        test_data["atm"].num_nodes
    )

    print(
        "Test transfers:",
        test_data[
            "account",
            "transfer",
            "account"
        ].edge_index.shape[1]
    )

    print(
        "Test cash-outs:",
        test_data[
            "account",
            "cashout",
            "atm"
        ].edge_index.shape[1]
    )

    print(
        "Test positives:",
        int(test_y.sum())
    )

    # --------------------------------------------------------
    # Align feature dimensions
    # --------------------------------------------------------

    in_channels = train_data["account"].x.shape[1]

    print()
    print("MODEL")
    print("Account input features:", in_channels)
    print("Hidden dimensions:", 64)

    model = HeteroGraphSAGE(
        in_channels=in_channels,
        hidden_channels=64,
    ).to(DEVICE)

    train_data = train_data.to(DEVICE)
    test_data = test_data.to(DEVICE)
    train_y = train_y.to(DEVICE)
    test_y = test_y.to(DEVICE)

    # --------------------------------------------------------
    # Normalize account features using TRAIN graph only
    # --------------------------------------------------------

    mean = train_data["account"].x.mean(
        dim=0,
        keepdim=True,
    )

    std = train_data["account"].x.std(
        dim=0,
        keepdim=True,
    ).clamp_min(1e-6)

    train_data["account"].x = (
        train_data["account"].x - mean
    ) / std

    test_data["account"].x = (
        test_data["account"].x - mean
    ) / std

    # --------------------------------------------------------
    # ATM feature stays constant
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    positives = train_y.sum().item()
    negatives = len(train_y) - positives

    pos_weight = (
        negatives / max(positives, 1)
    )

    pos_weight = min(
        pos_weight,
        20.0,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight,
            device=DEVICE,
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0005,
        weight_decay=1e-4,
    )

    print(
        "Positive weight:",
        round(pos_weight, 4)
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print("TRAINING")

    model.train()

    for epoch in range(1, 31):

        optimizer.zero_grad()

        logits = model(
            train_data.x_dict,
            train_data.edge_index_dict,
        )

        loss = criterion(
            logits,
            train_y,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        if (
            epoch == 1
            or epoch % 5 == 0
        ):
            print(
                f"Epoch {epoch:02d} "
                f"Loss={loss.item():.6f}"
            )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print()
    print("EVALUATION")

    model.eval()

    with torch.no_grad():

        test_logits = model(
            test_data.x_dict,
            test_data.edge_index_dict,
        )

    roc, pr, brier = evaluate_binary(
        test_y,
        test_logits,
    )

    print()
    print("HETEROGENEOUS GNN RESULTS")
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
        int(test_y.sum())
    )

    print(
        "Test negatives:",
        int((test_y == 0).sum())
    )

    print()
    print("=" * 70)
    print("SPRINT 7C COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
