import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from ml.features import FeatureContext
from graphx.gnn_temporal_examples import (
    make_snapshot_times,
    build_snapshot,
    future_cashout_labels,
)


# ============================================================
# TEMPORAL MESSAGE PASSING
# ============================================================

class TemporalMessagePassing(MessagePassing):
    """
    Message passing that explicitly uses:

        source node features
        transaction amount
        transaction age

    Transaction age = snapshot time - transaction time.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="mean")

        self.edge_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.message_mlp = nn.Sequential(
            nn.Linear(in_channels + 32, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

        self.self_linear = nn.Linear(in_channels, out_channels)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        edge_time,
        snapshot_timestamp,
    ):
        # Unix seconds
        snapshot_seconds = float(snapshot_timestamp)

        # Transaction age in minutes
        age_minutes = (
            snapshot_seconds - edge_time.float()
        ) / 60.0

        age_minutes = torch.clamp(age_minutes, min=0.0)

        # Log transforms reduce extreme transaction values
        amount = torch.log1p(
            torch.clamp(edge_attr[:, 0], min=0.0)
        )

        age = torch.log1p(age_minutes)

        temporal_edge_features = torch.stack(
            [amount, age],
            dim=1,
        )

        encoded_edges = self.edge_encoder(
            temporal_edge_features
        )

        propagated = self.propagate(
            edge_index,
            x=x,
            encoded_edges=encoded_edges,
        )

        return (
            self.self_linear(x)
            + propagated
        )

    def message(self, x_j, encoded_edges):
        message_input = torch.cat(
            [x_j, encoded_edges],
            dim=1,
        )

        return self.message_mlp(message_input)


# ============================================================
# TGNN MODEL
# ============================================================

class TemporalGraphSAGE(nn.Module):

    def __init__(self, in_channels, hidden=64):
        super().__init__()

        self.conv1 = TemporalMessagePassing(
            in_channels,
            hidden,
        )

        self.conv2 = TemporalMessagePassing(
            hidden,
            hidden,
        )

        self.output = nn.Linear(hidden, 1)

    def forward(
        self,
        data,
        snapshot_timestamp,
    ):

        x = self.conv1(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.edge_time,
            snapshot_timestamp,
        )

        x = torch.relu(x)

        x = self.conv2(
            x,
            data.edge_index,
            data.edge_attr,
            data.edge_time,
            snapshot_timestamp,
        )

        x = torch.relu(x)

        return self.output(x).squeeze(-1)


# ============================================================
# HELPERS
# ============================================================

def snapshot_timestamp_seconds(snapshot_time):
    return (
        np.datetime64(snapshot_time)
        - np.datetime64("1970-01-01T00:00:00")
    ) / np.timedelta64(1, "s")


def prepare_snapshot(ctx, snapshot_time):

    data, id_to_node, _ = build_snapshot(
        ctx,
        snapshot_time,
    )

    if data.num_nodes == 0:
        return data, None

    labels = future_cashout_labels(
        ctx,
        snapshot_time,
        list(id_to_node.keys()),
        horizon_min=60,
    )

    y = torch.tensor(
        labels,
        dtype=torch.float32,
    )

    return data, y


def normalize_snapshots(train_data, test_data):

    valid_train = [
        d.x for d, y in train_data
        if d.num_nodes > 0
    ]

    all_train_x = torch.cat(
        valid_train,
        dim=0,
    )

    mean = all_train_x.mean(dim=0)
    std = all_train_x.std(dim=0)

    std = torch.where(
        std < 1e-6,
        torch.ones_like(std),
        std,
    )

    for data, _ in train_data:
        if data.num_nodes > 0:
            data.x = (
                data.x - mean
            ) / std

    for data, _ in test_data:
        if data.num_nodes > 0:
            data.x = (
                data.x - mean
            ) / std

    return mean, std


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SPRINT 5 - TEMPORAL GNN")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    ctx = FeatureContext()

    # SAME snapshot protocol as Temporal GraphSAGE
    snapshot_times = make_snapshot_times(
        ctx,
        interval_min=15,
        max_snapshots=20,
    )

    print()
    print("Building chronological snapshots...")

    examples = []

    for i, snapshot_time in enumerate(
        snapshot_times,
        start=1,
    ):

        data, y = prepare_snapshot(
            ctx,
            snapshot_time,
        )

        positives = (
            int(y.sum().item())
            if y is not None
            else 0
        )

        print(
            f"[{i:02d}/{len(snapshot_times)}] "
            f"{snapshot_time} | "
            f"Nodes: {data.num_nodes} | "
            f"Edges: {data.num_edges} | "
            f"Positives: {positives}"
        )

        examples.append(
            (
                snapshot_time,
                data,
                y,
            )
        )

    # Keep exactly the same chronological split.
    split = int(
        len(examples) * 0.70
    )

    train_examples = examples[:split]
    test_examples = examples[split:]

    print()
    print("TEMPORAL SPLIT")
    print(
        "Train snapshots:",
        len(train_examples),
    )
    print(
        "Test snapshots :",
        len(test_examples),
    )

    train_data = [
        (data, y)
        for _, data, y in train_examples
        if data.num_nodes > 0
    ]

    test_data = [
        (data, y)
        for _, data, y in test_examples
        if data.num_nodes > 0
    ]

    train_pos = sum(
        int(y.sum().item())
        for _, y in train_data
    )

    train_total = sum(
        y.numel()
        for _, y in train_data
    )

    test_pos = sum(
        int(y.sum().item())
        for _, y in test_data
    )

    test_total = sum(
        y.numel()
        for _, y in test_data
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
    # Normalize using TRAINING graphs only
    # --------------------------------------------------------

    normalize_snapshots(
        train_data,
        test_data,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = TemporalGraphSAGE(
        in_channels=8,
        hidden=64,
    ).to(device)

    # Same stabilization strategy as the GraphSAGE baseline.
    raw_weight = (
        (train_total - train_pos)
        / max(train_pos, 1)
    )

    pos_weight_value = min(
        raw_weight,
        20.0,
    )

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=device,
    )

    print()
    print("MODEL")
    print("Input features: 8")
    print("Hidden dimensions: 64")
    print(
        "Raw positive weight:",
        round(raw_weight, 4),
    )
    print(
        "Used positive weight:",
        pos_weight_value,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0005,
        weight_decay=1e-4,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print("TRAINING")

    for epoch in range(1, 31):

        model.train()

        optimizer.zero_grad()

        total_loss = torch.tensor(
            0.0,
            device=device,
        )

        graph_count = 0

        for snapshot_time, data, y in train_examples:

            if data.num_nodes == 0:
                continue

            data = data.to(device)
            y = y.to(device)

            timestamp = snapshot_timestamp_seconds(
                snapshot_time
            )

            logits = model(
                data,
                timestamp,
            )

            loss = criterion(
                logits,
                y,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}"
                )

            loss.backward()

            graph_count += 1
            total_loss = (
                total_loss + loss.detach()
            )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        avg_loss = (
            total_loss.item()
            / max(graph_count, 1)
        )

        if (
            epoch == 1
            or epoch % 5 == 0
        ):
            print(
                f"Epoch {epoch:02d} "
                f"Loss={avg_loss:.6f}"
            )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print()
    print("EVALUATION")

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():

        for snapshot_time, data, y in test_examples:

            if data.num_nodes == 0:
                continue

            data = data.to(device)
            y = y.to(device)

            timestamp = snapshot_timestamp_seconds(
                snapshot_time
            )

            logits = model(
                data,
                timestamp,
            )

            probs = torch.sigmoid(
                logits
            )

            if not torch.isfinite(
                probs
            ).all():
                raise RuntimeError(
                    "Non-finite predictions detected"
                )

            all_probs.extend(
                probs.cpu().numpy()
            )

            all_labels.extend(
                y.cpu().numpy()
            )

    y_true = np.asarray(
        all_labels,
        dtype=np.float32,
    )

    y_prob = np.asarray(
        all_probs,
        dtype=np.float64,
    )

    roc = roc_auc_score(
        y_true,
        y_prob,
    )

    pr = average_precision_score(
        y_true,
        y_prob,
    )

    brier = brier_score_loss(
        y_true,
        y_prob,
    )

    print("=" * 70)
    print("TEMPORAL GNN RESULTS")
    print("=" * 70)
    print(f"ROC-AUC : {roc:.4f}")
    print(f"PR-AUC  : {pr:.4f}")
    print(f"Brier   : {brier:.4f}")
    print(
        "Test positives:",
        int(y_true.sum()),
    )
    print(
        "Test negatives:",
        int(len(y_true) - y_true.sum()),
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
