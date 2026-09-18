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


def make_snapshot_times(
    ctx,
    interval_min=15,
    max_snapshots=20,
):
    start = ctx.ts[0].astype("datetime64[m]")
    end = ctx.ts[-1].astype("datetime64[m]")

    interval = np.timedelta64(interval_min, "m")

    times = []
    current = start

    while current <= end:
        times.append(current.astype("datetime64[us]"))
        current = current + interval

    if len(times) > max_snapshots:
        indices = np.linspace(
            0,
            len(times) - 1,
            max_snapshots,
            dtype=int,
        )

        times = [times[i] for i in indices]

    return times


def build_examples(
    ctx,
    snapshot_times,
    horizon_min=60,
):
    examples = []

    print()
    print("=" * 70)
    print("BUILDING TEMPORAL GRAPH DATASET")
    print("=" * 70)

    for index, snapshot_time in enumerate(
        snapshot_times,
        start=1,
    ):
        print(
            f"[{index:02d}/{len(snapshot_times)}] "
            f"Snapshot: {snapshot_time}"
        )

        data, id_to_node, node_ids = build_snapshot(
            ctx,
            snapshot_time,
        )

        labels = future_cashout_labels(
            ctx,
            snapshot_time,
            node_ids,
            horizon_min=horizon_min,
        )

        positive_count = int(labels.sum().item())

        examples.append(
            {
                "time": snapshot_time,
                "data": data,
                "labels": labels,
                "positive_count": positive_count,
            }
        )

        print(f"    Nodes: {data.num_nodes}")
        print(f"    Edges: {data.num_edges}")
        print(f"    Positives: {positive_count}")

    return examples

def main():

    print("=" * 70)
    print("SPRINT 4B - TEMPORAL DATASET BUILDER")
    print("=" * 70)

    ctx = FeatureContext()

    snapshot_times = make_snapshot_times(
        ctx,
        interval_min=15,
        max_snapshots=20,
    )

    print(
        f"Total snapshots: {len(snapshot_times)}"
    )

    examples = build_examples(
        ctx,
        snapshot_times,
        horizon_min=60,
    )

    split_index = int(
        len(examples) * 0.70
    )

    train_examples = examples[:split_index]
    test_examples = examples[split_index:]

    train_positive = sum(
        x["positive_count"]
        for x in train_examples
    )

    test_positive = sum(
        x["positive_count"]
        for x in test_examples
    )

    train_nodes = sum(
        x["data"].num_nodes
        for x in train_examples
    )

    test_nodes = sum(
        x["data"].num_nodes
        for x in test_examples
    )

    print()
    print("=" * 70)
    print("TEMPORAL TRAIN / TEST SPLIT")
    print("=" * 70)

    print(f"Train snapshots: {len(train_examples)}")
    print(f"Test snapshots : {len(test_examples)}")
    print(f"Train nodes: {train_nodes}")
    print(f"Test nodes : {test_nodes}")
    print(f"Train positives: {train_positive}")
    print(f"Test positives : {test_positive}")

    print()
    print("TRAIN PERIOD:")
    print(
        f"  {train_examples[0]['time']}"
        f" -> "
        f"{train_examples[-1]['time']}"
    )

    print("TEST PERIOD:")
    print(
        f"  {test_examples[0]['time']}"
        f" -> "
        f"{test_examples[-1]['time']}"
    )

    print()
    print("=" * 70)

    if train_positive == 0:
        print("TEMPORAL DATASET: FAIL")

    elif test_positive == 0:
        print(
            "TEMPORAL DATASET: FAIL - "
            "TEST HAS NO POSITIVES"
        )

    else:
        print("TEMPORAL DATASET: PASS")

    print("=" * 70)


if __name__ == "__main__":
    main()



