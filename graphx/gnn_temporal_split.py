"""
Sprint 4B — Temporal Train/Test GraphSAGE

Train:
    Graph at 60% of the timeline.

Test:
    Later graph at 70% of the timeline.

Target:
    Cash-out within the next 60 minutes.

Features:
    Transactions strictly before each snapshot.

Labels:
    Future transactions after each snapshot.
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

TRAIN_FRACTION = 0.60
TEST_FRACTION = 0.70


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


def get_snapshot(ctx, fraction):

    index = int(
        len(ctx.ts) * fraction
    )

    timestamp = ctx.ts[index]

    return timestamp.astype(
        "datetime64[us]"
    ).tolist()


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

    cashout_accounts = set()

    for pos in positions:

        account_id = ctx.src[pos]

        cashout_accounts.add(
            account_id
        )

    labels = np.zeros(
        len(id_to_node),
        dtype=np.float32
    )

    for account_id, node_id in id_to_node.items():

        if account_id in cashout_accounts:

            labels[node_id] = 1.0

    return labels


def build_snapshot(
    ctx,
    snapshot
):

    data, id_to_node, node_ids = (
        build_transaction_graph(
            ctx,
            snapshot,
            observed_only=True
        )
    )

    data = attach_node_features(
        data,
        ctx,
        id_to_node,
        snapshot
    )

    return (
        data,
        id_to_node,
        node_ids
    )


def main():

    print("=" * 70)
    print("SPRINT 4B — TEMPORAL TRAIN / TEST GRAPHSAGE")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nDevice:",
        device
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    ctx = FeatureContext(
        config.DATA_DIR
    )

    print(
        "Transactions:",
        len(ctx.ts)
    )

    # --------------------------------------------------------
    # Get temporal snapshots
    # --------------------------------------------------------

    train_snapshot = get_snapshot(
        ctx,
        TRAIN_FRACTION
    )

    test_snapshot = get_snapshot(
        ctx,
        TEST_FRACTION
    )

    print(
        "\nTRAIN SNAPSHOT:",
        train_snapshot
    )

    print(
        "TEST SNAPSHOT :",
        test_snapshot
    )

    print(
        "HORIZON:",
        HORIZON_MIN,
        "minutes"
    )

    # --------------------------------------------------------
    # TRAIN GRAPH
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("BUILDING TRAIN GRAPH")
    print("-" * 70)

    train_data, train_mapping, _ = (
        build_snapshot(
            ctx,
            train_snapshot
        )
    )

    print(
        "Train nodes:",
        train_data.num_nodes
    )

    print(
        "Train edges:",
        train_data.num_edges
    )

    print(
        "Train features:",
        tuple(train_data.x.shape)
    )

    # --------------------------------------------------------
    # TRAIN LABELS
    # --------------------------------------------------------

    train_labels = build_labels(
        ctx,
        train_mapping,
        train_snapshot,
        HORIZON_MIN
    )

    train_positive = int(
        train_labels.sum()
    )

    print(
        "Train positives:",
        train_positive
    )

    print(
        "Train prevalence:",
        f"{train_labels.mean():.4f}"
    )

    # --------------------------------------------------------
    # TEST GRAPH
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("BUILDING TEST GRAPH")
    print("-" * 70)

    test_data, test_mapping, _ = (
        build_snapshot(
            ctx,
            test_snapshot
        )
    )

    print(
        "Test nodes:",
        test_data.num_nodes
    )

    print(
        "Test edges:",
        test_data.num_edges
    )

    print(
        "Test features:",
        tuple(test_data.x.shape)
    )

    # --------------------------------------------------------
    # TEST LABELS
    # --------------------------------------------------------

    test_labels = build_labels(
        ctx,
        test_mapping,
        test_snapshot,
        HORIZON_MIN
    )

    test_positive = int(
        test_labels.sum()
    )

    print(
        "Test positives:",
        test_positive
    )

    print(
        "Test prevalence:",
        f"{test_labels.mean():.4f}"
    )

    if train_positive == 0:

        raise RuntimeError(
            "Training has zero positive examples."
        )

    if test_positive == 0:

        raise RuntimeError(
            "Testing has zero positive examples."
        )

    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    train_data = train_data.to(device)

    y_train = torch.tensor(
        train_labels,
        dtype=torch.float32,
        device=device
    )

    model = GraphSAGE(
        in_channels=train_data.x.shape[1],
        hidden_channels=64
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    # --------------------------------------------------------
    # Class weighting
    # --------------------------------------------------------

    negatives = (
        len(train_labels)
        - train_positive
    )

    pos_weight_value = (
        negatives
        / train_positive
    )

    pos_weight = torch.tensor(
        pos_weight_value,
        dtype=torch.float32,
        device=device
    )

    print(
        "\nPositive weight:",
        f"{pos_weight_value:.4f}"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("TRAINING")
    print("-" * 70)

    model.train()

    for epoch in range(1, 51):

        optimizer.zero_grad()

        logits = model(
            train_data.x,
            train_data.edge_index
        )

        loss = F.binary_cross_entropy_with_logits(
            logits,
            y_train,
            pos_weight=pos_weight
        )

        loss.backward()

        optimizer.step()

        if epoch == 1 or epoch % 10 == 0:

            print(
                f"Epoch {epoch:02d} "
                f"Loss={loss.item():.4f}"
            )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("TESTING ON LATER SNAPSHOT")
    print("-" * 70)

    test_data = test_data.to(device)

    model.eval()

    with torch.no_grad():

        test_logits = model(
            test_data.x,
            test_data.edge_index
        )

        test_probabilities = (
            torch.sigmoid(test_logits)
            .cpu()
            .numpy()
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    roc_auc = roc_auc_score(
        test_labels,
        test_probabilities
    )

    pr_auc = average_precision_score(
        test_labels,
        test_probabilities
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEMPORAL GRAPHSAGE RESULTS")
    print("=" * 70)

    print(
        "Horizon          :",
        f"{HORIZON_MIN} minutes"
    )

    print(
        "Train snapshot   :",
        train_snapshot
    )

    print(
        "Test snapshot    :",
        test_snapshot
    )

    print(
        "Train nodes      :",
        train_data.num_nodes
    )

    print(
        "Test nodes       :",
        test_data.num_nodes
    )

    print(
        "Train positives  :",
        train_positive
    )

    print(
        "Test positives   :",
        test_positive
    )

    print(
        "Test prevalence  :",
        f"{test_labels.mean():.4f}"
    )

    print(
        f"ROC-AUC          : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC           : {pr_auc:.4f}"
    )

    print("=" * 70)

    print(
        "\nSPRINT 4B COMPLETE"
    )


if __name__ == "__main__":
    main()
