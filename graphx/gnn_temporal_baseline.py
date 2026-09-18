"""
Sprint 4A — Temporal GraphSAGE Baseline

Purpose:
    Verify that GraphSAGE can use historical graph information
    to predict cash-out activity in the next 60 minutes.

IMPORTANT:
    Features use only transactions BEFORE snapshot.
    Labels use transactions AFTER snapshot.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score, average_precision_score

import config
from ml.features import FeatureContext
from graphx.pyg_data import build_transaction_graph
from graphx.node_features import attach_node_features


HORIZON_MIN = 60


class GraphSAGE(nn.Module):

    def __init__(self, in_channels, hidden_channels=64):

        super().__init__()

        self.conv1 = SAGEConv(
            in_channels,
            hidden_channels
        )

        self.conv2 = SAGEConv(
            hidden_channels,
            hidden_channels
        )

        self.out = nn.Linear(
            hidden_channels,
            1
        )

    def forward(self, x, edge_index):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=0.2,
            training=self.training
        )

        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        return self.out(x).squeeze(-1)


def build_labels(
    ctx,
    id_to_node,
    snapshot,
    horizon_min=60
):

    snapshot_np = np.datetime64(snapshot)

    horizon_end = (
        snapshot_np
        + np.timedelta64(
            horizon_min,
            "m"
        )
    )

    positions = np.flatnonzero(
        (ctx.ts > snapshot_np)
        & (ctx.ts <= horizon_end)
        & ctx.is_cashout
    )

    future_cashout_accounts = set()

    for pos in positions:

        account_id = ctx.src[pos]

        future_cashout_accounts.add(
            account_id
        )

    labels = np.zeros(
        len(id_to_node),
        dtype=np.float32
    )

    for account_id, node_id in id_to_node.items():

        if account_id in future_cashout_accounts:

            labels[node_id] = 1.0

    return labels


def main():

    print("=" * 70)
    print("SPRINT 4A — TEMPORAL GRAPHSAGE BASELINE")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    ctx = FeatureContext(
        config.DATA_DIR
    )

    print(
        "Transactions:",
        len(ctx.ts)
    )

    # --------------------------------------------------------
    # Snapshot at 70% of timeline
    # --------------------------------------------------------

    snapshot_index = int(
        len(ctx.ts) * 0.70
    )

    snapshot_np = ctx.ts[
        snapshot_index
    ]

    snapshot = snapshot_np.astype(
        "datetime64[us]"
    ).tolist()

    print(
        "\nSnapshot:",
        snapshot
    )

    print(
        "Prediction horizon:",
        HORIZON_MIN,
        "minutes"
    )

    # --------------------------------------------------------
    # Historical graph
    # --------------------------------------------------------

    print("\nBuilding graph...")

    data, id_to_node, node_ids = (
        build_transaction_graph(
            ctx,
            snapshot,
            observed_only=True
        )
    )

    print(
        "Nodes:",
        data.num_nodes
    )

    print(
        "Edges:",
        data.num_edges
    )

    # --------------------------------------------------------
    # Historical features
    # --------------------------------------------------------

    data = attach_node_features(
        data,
        ctx,
        id_to_node,
        snapshot
    )

    print(
        "Features:",
        tuple(data.x.shape)
    )

    # --------------------------------------------------------
    # Future labels
    # --------------------------------------------------------

    labels_np = build_labels(
        ctx,
        id_to_node,
        snapshot,
        HORIZON_MIN
    )

    positive_count = int(
        labels_np.sum()
    )

    print(
        "Positive nodes:",
        positive_count
    )

    print(
        "Positive rate:",
        f"{labels_np.mean():.4f}"
    )

    if positive_count == 0:

        raise RuntimeError(
            "No positive labels found."
        )

    # --------------------------------------------------------
    # Tensor
    # --------------------------------------------------------

    data = data.to(device)

    y = torch.tensor(
        labels_np,
        dtype=torch.float32,
        device=device
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = GraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=64
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nTraining...")

    model.train()

    for epoch in range(1, 51):

        optimizer.zero_grad()

        logits = model(
            data.x,
            data.edge_index
        )

        loss = F.binary_cross_entropy_with_logits(
            logits,
            y
        )

        loss.backward()

        optimizer.step()

        if epoch == 1 or epoch % 10 == 0:

            print(
                f"Epoch {epoch:02d} "
                f"Loss={loss.item():.4f}"
            )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        logits = model(
            data.x,
            data.edge_index
        )

        probabilities = (
            torch.sigmoid(logits)
            .cpu()
            .numpy()
        )

    roc_auc = roc_auc_score(
        labels_np,
        probabilities
    )

    pr_auc = average_precision_score(
        labels_np,
        probabilities
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEMPORAL GRAPHSAGE RESULTS")
    print("=" * 70)

    print(
        f"ROC-AUC : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC  : {pr_auc:.4f}"
    )

    print(
        "Nodes   :",
        data.num_nodes
    )

    print(
        "Edges   :",
        data.num_edges
    )

    print("=" * 70)


if __name__ == "__main__":
    main()