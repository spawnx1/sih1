from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

from ml.features import FeatureContext
from graphx.pyg_data import build_transaction_graph
from graphx.node_features import attach_node_features


class TemporalGraphSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.out = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        return self.out(x).squeeze(-1)


def build_snapshot(ctx, snapshot_time):
    data, id_to_node, node_ids = build_transaction_graph(
        ctx,
        as_of=snapshot_time,
        observed_only=True,
    )

    data = attach_node_features(
        data,
        ctx,
        id_to_node,
        snapshot_time,
        trail=None,
    )

    return data, id_to_node, node_ids


def future_cashout_labels(ctx, snapshot_time, node_ids, horizon_min=60):
    horizon_end = snapshot_time + np.timedelta64(horizon_min, "m")

    labels = np.zeros(len(node_ids), dtype=np.float32)

    for i, account_id in enumerate(node_ids):
        positions = ctx.src_pos.get(account_id)

        if positions is None:
            continue

        for pos in positions:
            ts = ctx.ts[pos]

            if ts <= np.datetime64(snapshot_time):
                continue

            if ts > np.datetime64(horizon_end):
                break

            if ctx.is_cashout[pos]:
                labels[i] = 1.0
                break

    return torch.tensor(labels, dtype=torch.float32)


def main():
    print("=" * 70)
    print("SPRINT 4B - TEMPORAL GRAPHSAGE")
    print("=" * 70)

    ctx = FeatureContext()

    snapshot_time = ctx.ts[len(ctx.ts) // 2] \
        .astype("datetime64[us]").tolist()

    data, id_to_node, node_ids = build_snapshot(
        ctx,
        snapshot_time,
    )

    labels = future_cashout_labels(
        ctx,
        snapshot_time,
        node_ids,
        horizon_min=60,
    )

    print("Snapshot time:", snapshot_time)
    print("Nodes:", data.num_nodes)
    print("Edges:", data.num_edges)
    print("Features:", tuple(data.x.shape))
    print("Positive labels:", int(labels.sum().item()))
    print("Negative labels:", int(len(labels) - labels.sum().item()))

    print("=" * 70)
    print("SPRINT 4B FOUNDATION: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()

