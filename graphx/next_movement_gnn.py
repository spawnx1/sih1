"""
Sprint 8 - Next-Movement Prediction

Predict the next action of an account from a temporal
heterogeneous financial graph.

Classes:
    0 = ATM
    1 = POS
    2 = onward_transfer
    3 = dormant

Graph:
    Account -> Account : transfer
    Account -> ATM     : cashout
    reverse relations are included for message passing.

Graph contains only history before snapshot T.
The next-action label is determined from future transactions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, SAGEConv

from ml.features import FeatureContext
from graphx.hetero_graph import build_heterogeneous_graph
from graphx.node_features import build_node_feature_matrix
from graphx.temporal_activity_dataset import (
    build_activity_buckets,
    choose_activity_snapshots,
    MAX_SNAPSHOTS,
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = 4
HIDDEN = 64
EPOCHS = 25
LR = 0.0005
WEIGHT_DECAY = 1e-4

CLASS_NAMES = [
    "ATM",
    "POS",
    "onward_transfer",
    "dormant",
]


# ============================================================
# MODEL
# ============================================================

class NextMovementGNN(nn.Module):

    def __init__(
        self,
        account_in,
        atm_in,
        hidden=64,
    ):
        super().__init__()

        self.conv1 = HeteroConv(
            {
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
            },
            aggr="sum",
        )

        self.conv2 = HeteroConv(
            {
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
            },
            aggr="sum",
        )

        self.out = nn.Linear(
            hidden,
            NUM_CLASSES,
        )

    def forward(
        self,
        x_dict,
        edge_index_dict,
    ):

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
        )


# ============================================================
# ATM FEATURES
# ============================================================

def build_atm_features(data):

    n = int(
        data["atm"].num_nodes
    )

    x = torch.zeros(
        (n, 2),
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
        return x

    atm = edge_index[1]

    degree = torch.bincount(
        atm,
        minlength=n,
    ).float()

    amounts = edge_attr[:, 0].float()

    total = torch.zeros(
        n,
        dtype=torch.float32,
    )

    total.scatter_add_(
        0,
        atm,
        amounts,
    )

    x[:, 0] = torch.log1p(degree)
    x[:, 1] = torch.log1p(total)

    return x


# ============================================================
# NEXT ACTION LABEL
# ============================================================

def next_action_labels(
    ctx,
    snapshot,
    account_ids,
):

    snapshot = np.datetime64(snapshot)

    account_to_idx = {
        str(a): i
        for i, a in enumerate(account_ids)
    }

    # Default = dormant
    labels = np.full(
        len(account_ids),
        3,
        dtype=np.int64,
    )

    first_event = {}

    for account, positions in ctx.src_pos.items():

        key = str(account)

        if key not in account_to_idx:
            continue

        for pos in positions:

            ts = ctx.ts[pos]

            if ts <= snapshot:
                continue

            channel = str(
                ctx.channel[pos]
            ).lower()

            if channel in (
                "atm",
                "cashout",
                "cash_out",
                "withdrawal",
                "atm_wdl",
            ):
                action = 0

            elif channel in (
                "pos",
                "point_of_sale",
            ):
                action = 1

            else:
                action = 2

            if (
                key not in first_event
                or ts < first_event[key][0]
            ):
                first_event[key] = (
                    ts,
                    action,
                )

    for account, (_, action) in first_event.items():

        labels[
            account_to_idx[account]
        ] = action

    return torch.tensor(
        labels,
        dtype=torch.long,
    )


# ============================================================
# BUILD SNAPSHOT
# ============================================================

def build_snapshot(
    ctx,
    snapshot,
):

    data, account_to_idx, atm_to_idx = (
        build_heterogeneous_graph(
            ctx,
            as_of=snapshot,
            observed_only=True,
        )
    )

    data["account"].x = (
        build_node_feature_matrix(
            ctx,
            account_to_idx,
            snapshot,
            trail=None,
        )
    )

    data["atm"].x = build_atm_features(
        data
    )

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

    labels = next_action_labels(
        ctx,
        snapshot,
        data.account_id,
    )

    return data, labels


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SPRINT 8 - NEXT-MOVEMENT PREDICTION")
    print("=" * 70)

    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    ctx = FeatureContext()

    print()
    print("BUILDING TEMPORAL SNAPSHOTS...")

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

        if data["account"].num_nodes == 0:
            continue

        counts = torch.bincount(
            labels,
            minlength=NUM_CLASSES,
        )

        print(
            f"[{i:02d}/{len(snapshot_times)}] "
            f"{str(snapshot)[:16]} | "
            f"Accounts: "
            f"{data['account'].num_nodes} | "
            f"ATM: "
            f"{data['atm'].num_nodes} | "
            f"ATM={int(counts[0])} "
            f"POS={int(counts[1])} "
            f"TRANSFER={int(counts[2])} "
            f"DORMANT={int(counts[3])}"
        )

        snapshots.append(
            (data, labels)
        )

    split = int(
        len(snapshots) * 0.70
    )

    train = snapshots[:split]
    test = snapshots[split:]

    print()
    print("TEMPORAL SPLIT")
    print("Train snapshots:", len(train))
    print("Test snapshots :", len(test))

    train_counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    test_counts = torch.zeros(
        NUM_CLASSES,
        dtype=torch.long,
    )

    for _, y in train:
        train_counts += torch.bincount(
            y,
            minlength=NUM_CLASSES,
        )

    for _, y in test:
        test_counts += torch.bincount(
            y,
            minlength=NUM_CLASSES,
        )

    print()
    print("TRAIN CLASS COUNTS")

    for i, name in enumerate(CLASS_NAMES):
        print(
            f"{name:16s}: "
            f"{int(train_counts[i])}"
        )

    print()
    print("TEST CLASS COUNTS")

    for i, name in enumerate(CLASS_NAMES):
        print(
            f"{name:16s}: "
            f"{int(test_counts[i])}"
        )

    # --------------------------------------------------------
    # Feature normalization using train only
    # --------------------------------------------------------

    train_x = torch.cat(
        [
            d["account"].x
            for d, _ in train
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

    account_in = train[0][
        0
    ]["account"].x.shape[1]

    atm_in = train[0][
        0
    ]["atm"].x.shape[1]

    print()
    print("MODEL")
    print("Account features:", account_in)
    print("ATM features:", atm_in)
    print("Hidden:", HIDDEN)
    print("Classes:", NUM_CLASSES)

    model = NextMovementGNN(
        account_in,
        atm_in,
        HIDDEN,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    total = train_counts.sum().item()

    weights = []

    for count in train_counts:

        c = max(
            int(count),
            1,
        )

        weights.append(
            total / (
                NUM_CLASSES * c
            )
        )

    weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    print(
        "Class weights:",
        [
            round(float(w), 3)
            for w in weights
        ],
    )

    criterion = nn.CrossEntropyLoss(
        weight=weights,
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

        for data, labels in train:

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

            avg = (
                total_loss
                / max(len(train), 1)
            )

            print(
                f"Epoch {epoch:02d} "
                f"Loss={avg:.6f}"
            )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        classification_report,
        confusion_matrix,
    )

    model.eval()

    all_y = []
    all_pred = []

    with torch.no_grad():

        for data, labels in test:

            data = data.to(DEVICE)

            logits = model(
                data.x_dict,
                data.edge_index_dict,
            )

            pred = torch.argmax(
                logits,
                dim=1,
            ).cpu()

            all_pred.append(pred)
            all_y.append(labels)

    y_true = torch.cat(
        all_y
    ).numpy()

    y_pred = torch.cat(
        all_pred
    ).numpy()

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    print()
    print("EVALUATION")
    print()
    print("NEXT-MOVEMENT RESULTS")
    print(
        f"Accuracy           : {accuracy:.4f}"
    )
    print(
        f"Balanced Accuracy  : {balanced:.4f}"
    )
    print(
        f"Macro F1           : {macro_f1:.4f}"
    )

    print()
    print("CONFUSION MATRIX")
    print(
        "Rows = actual, columns = predicted"
    )

    print(cm)

    print()
    print("CLASSIFICATION REPORT")

    print(
        classification_report(y_true, y_pred, labels=list(range(NUM_CLASSES)), target_names=CLASS_NAMES, zero_division=0)
    )

    print()
    print("=" * 70)
    print("SPRINT 8 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()




