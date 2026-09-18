"""
graphx/gnn_baseline.py

Sprint 4 — First Graph Neural Network

Purpose:
    Train a simple GraphSAGE model on the validated
    transaction graph.

Important:
    - Uses only information available before as_of for features.
    - Does not use label columns as node features.
    - Does not modify the existing graph construction.
    - Future transactions are used ONLY to create prediction labels.
    - This is a first GNN baseline, not the final TGNN.
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


class GraphSAGE(nn.Module):
    """
    Simple 2-layer GraphSAGE.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
    ):
        super().__init__()

        self.conv1 = SAGEConv(
            in_channels,
            hidden_channels,
        )

        self.conv2 = SAGEConv(
            hidden_channels,
            hidden_channels,
        )

        self.out = nn.Linear(
            hidden_channels,
            1,
        )

    def forward(self, x, edge_index):

        x = self.conv1(
            x,
            edge_index,
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=0.2,
            training=self.training,
        )

        x = self.conv2(
            x,
            edge_index,
        )

        x = F.relu(x)

        return self.out(x).squeeze(-1)


def main():

    print("=" * 60)
    print("SPRINT 4 — GRAPH SAGE BASELINE")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n[1] Device")
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # ---------------------------------------------------------
    # 2. Load context
    # ---------------------------------------------------------

    print("\n[2] Loading FeatureContext...")

    ctx = FeatureContext(config.DATA_DIR)

    # ---------------------------------------------------------
    # 3. Choose snapshot
    # ---------------------------------------------------------

    snapshot_index = int(len(ctx.ts) * 0.70)

    snapshot_np = ctx.ts[snapshot_index]

    # Convert NumPy datetime64 -> Python datetime
    snapshot = snapshot_np.astype(
        "datetime64[us]"
    ).tolist()

    print("Snapshot:", snapshot)

    # ---------------------------------------------------------
    # 4. Build validated temporal graph
    # ---------------------------------------------------------

    print("\n[3] Building graph...")

    data, id_to_node, node_to_id = build_transaction_graph(
        ctx,
        snapshot,
        observed_only=True,
    )

    print("Nodes:", data.num_nodes)
    print("Edges:", data.num_edges)

    # ---------------------------------------------------------
    # 5. Attach node features
    # ---------------------------------------------------------

    print("\n[4] Building node features...")

    data = attach_node_features(
        data,
        ctx,
        id_to_node,
        snapshot,
    )

    print(
        "Node feature shape:",
        tuple(data.x.shape),
    )

    # ---------------------------------------------------------
    # 6. Move graph to GPU/CPU
    # ---------------------------------------------------------

    data = data.to(device)

    # ---------------------------------------------------------
    # 7. Create FUTURE cash-out labels
    # ---------------------------------------------------------

    print("\n[5] Building node labels...")

    snapshot_np = np.datetime64(snapshot)

    # IMPORTANT:
    #
    # Features:
    #     Only transactions before snapshot.
    #
    # Labels:
    #     Future cash-outs after snapshot.
    #
    # Future transactions are therefore used only as the
    # prediction target and never as model input.

    future_cashout_accounts = set()

    future_positions = np.flatnonzero(
        (ctx.ts > snapshot_np)
        & ctx.is_cashout
    )

    for pos in future_positions:

        account_id = ctx.src[pos]

        future_cashout_accounts.add(
            account_id
        )

    labels = [
        1 if account_id in future_cashout_accounts else 0
        for account_id in id_to_node
    ]

    y = torch.tensor(
        labels,
        dtype=torch.float32,
        device=device,
    )

    print(
        "Positive nodes:",
        int(y.sum()),
    )

    print(
        "Total nodes:",
        len(y),
    )

    print(
        "Positive rate:",
        float(y.mean()),
    )

    # ---------------------------------------------------------
    # Safety check
    # ---------------------------------------------------------

    if int(y.sum()) == 0:

        raise RuntimeError(
            "No positive future cash-out labels were found. "
            "GNN training cannot proceed."
        )

    if int(y.sum()) == len(y):

        raise RuntimeError(
            "All nodes are positive. "
            "GNN training cannot proceed."
        )

    # ---------------------------------------------------------
    # 8. Create model
    # ---------------------------------------------------------

    print("\n[6] Creating GraphSAGE model...")

    model = GraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=64,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
    )

    # ---------------------------------------------------------
    # 9. Train
    # ---------------------------------------------------------

    print("\n[7] Training...")

    model.train()

    for epoch in range(1, 51):

        optimizer.zero_grad()

        logits = model(
            data.x,
            data.edge_index,
        )

        loss = F.binary_cross_entropy_with_logits(
            logits,
            y,
        )

        loss.backward()

        optimizer.step()

        if epoch == 1 or epoch % 10 == 0:

            print(
                f"Epoch {epoch:02d} "
                f"Loss={loss.item():.4f}"
            )

    # ---------------------------------------------------------
    # 10. Evaluation
    # ---------------------------------------------------------

    print("\n[8] Evaluation...")

    model.eval()

    with torch.no_grad():

        logits = model(
            data.x,
            data.edge_index,
        )

        probabilities = torch.sigmoid(
            logits
        ).cpu().numpy()

    y_np = y.cpu().numpy()

    roc_auc = roc_auc_score(
        y_np,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_np,
        probabilities,
    )

    # ---------------------------------------------------------
    # 11. Results
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("GRAPH SAGE RESULTS")
    print("=" * 60)

    print(
        f"ROC-AUC : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC  : {pr_auc:.4f}"
    )

    print(
        f"Nodes   : {data.num_nodes}"
    )

    print(
        f"Edges   : {data.num_edges}"
    )

    # ---------------------------------------------------------
    # 12. Sprint result
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("SPRINT 4 RESULT")
    print("=" * 60)

    print(
        "GraphSAGE training completed."
    )


if __name__ == "__main__":
    main()