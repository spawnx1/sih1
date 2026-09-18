from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

from graphx.gnn_temporal_examples import (
    TemporalGraphSAGE,
    make_snapshot_times,
    build_examples,
)

from ml.features import FeatureContext


def prepare_examples(examples):
    return [
        x for x in examples
        if x["data"].num_nodes > 0
    ]


def fit_feature_scaler(train_examples):

    all_x = torch.cat(
        [
            x["data"].x
            for x in train_examples
        ],
        dim=0,
    )

    mean = all_x.mean(dim=0)

    std = all_x.std(dim=0)

    std = torch.where(
        std < 1e-6,
        torch.ones_like(std),
        std,
    )

    return mean, std


def normalize_examples(
    examples,
    mean,
    std,
):

    normalized = []

    for example in examples:

        data = example["data"].clone()

        data.x = (
            data.x - mean
        ) / std

        if not torch.isfinite(
            data.x
        ).all():

            raise RuntimeError(
                "NaN/Inf after feature normalization."
            )

        normalized.append(
            {
                "time": example["time"],
                "data": data,
                "labels": example["labels"],
                "positive_count": example["positive_count"],
            }
        )

    return normalized


def main():

    print("=" * 70)
    print("SPRINT 4B - TEMPORAL GRAPHSAGE TRAINING")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    ctx = FeatureContext()

    snapshot_times = make_snapshot_times(
        ctx,
        interval_min=15,
        max_snapshots=20,
    )

    examples = build_examples(
        ctx,
        snapshot_times,
        horizon_min=60,
    )

    examples = prepare_examples(
        examples
    )

    split_index = int(
        len(examples) * 0.70
    )

    train_examples = examples[
        :split_index
    ]

    test_examples = examples[
        split_index:
    ]

    train_labels = np.concatenate(
        [
            x["labels"].numpy()
            for x in train_examples
        ]
    )

    test_labels = np.concatenate(
        [
            x["labels"].numpy()
            for x in test_examples
        ]
    )

    train_positive = float(
        train_labels.sum()
    )

    train_negative = float(
        len(train_labels)
        - train_positive
    )

    test_positive = int(
        test_labels.sum()
    )

    test_negative = int(
        len(test_labels)
        - test_positive
    )

    print()
    print("=" * 70)
    print("DATASET")
    print("=" * 70)

    print(
        "Train snapshots:",
        len(train_examples)
    )

    print(
        "Test snapshots:",
        len(test_examples)
    )

    print(
        "Train positives:",
        int(train_positive)
    )

    print(
        "Train negatives:",
        int(train_negative)
    )

    print(
        "Test positives:",
        test_positive
    )

    print(
        "Test negatives:",
        test_negative
    )

    if train_positive == 0:
        raise RuntimeError(
            "No positive training examples."
        )

    if test_positive == 0:
        raise RuntimeError(
            "No positive test examples."
        )

    print()
    print("=" * 70)
    print("FEATURE NORMALIZATION")
    print("=" * 70)

    mean, std = fit_feature_scaler(
        train_examples
    )

    print(
        "Feature means:",
        mean.numpy()
    )

    print(
        "Feature std:",
        std.numpy()
    )

    train_examples = normalize_examples(
        train_examples,
        mean,
        std,
    )

    test_examples = normalize_examples(
        test_examples,
        mean,
        std,
    )

    print(
        "Train features: FINITE"
    )

    print(
        "Test features: FINITE"
    )

    input_channels = (
        train_examples[0]["data"]
        .x.shape[1]
    )

    model = TemporalGraphSAGE(
        in_channels=input_channels,
        hidden_channels=64,
    ).to(device)

    raw_pos_weight = (
        train_negative
        / train_positive
    )

    pos_weight_value = min(
        raw_pos_weight,
        20.0,
    )

    pos_weight = torch.tensor(
        pos_weight_value,
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0005,
        weight_decay=1e-4,
    )

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    print(
        "Input features:",
        input_channels
    )

    print(
        "Raw positive weight:",
        round(raw_pos_weight, 4)
    )

    print(
        "Used positive weight:",
        round(pos_weight_value, 4)
    )

    for epoch in range(1, 31):

        model.train()

        total_loss = 0.0

        for example in train_examples:

            data = example["data"].to(device)

            labels = (
                example["labels"]
                .to(device)
            )

            optimizer.zero_grad()

            logits = model(
                data.x,
                data.edge_index,
            )

            if not torch.isfinite(
                logits
            ).all():

                raise RuntimeError(
                    "Model produced NaN/Inf logits."
                )

            loss = criterion(
                logits,
                labels,
            )

            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    "Loss became NaN/Inf."
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss
            / len(train_examples)
        )

        if (
            epoch == 1
            or epoch % 5 == 0
        ):

            print(
                f"Epoch {epoch:02d} "
                f"Loss={average_loss:.6f}"
            )

    print()
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)

    model.eval()

    y_true = []
    y_prob = []

    with torch.no_grad():

        for example in test_examples:

            data = example["data"].to(
                device
            )

            logits = model(
                data.x,
                data.edge_index,
            )

            probabilities = (
                torch.sigmoid(logits)
                .cpu()
                .numpy()
            )

            y_true.append(
                example["labels"]
                .numpy()
            )

            y_prob.append(
                probabilities
            )

    y_true = np.concatenate(
        y_true
    )

    y_prob = np.concatenate(
        y_prob
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob,
    )

    pr_auc = average_precision_score(
        y_true,
        y_prob,
    )

    brier = brier_score_loss(
        y_true,
        y_prob,
    )

    print()
    print("=" * 70)
    print("TEMPORAL GRAPHSAGE RESULTS")
    print("=" * 70)

    print(
        f"ROC-AUC : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC  : {pr_auc:.4f}"
    )

    print(
        f"Brier   : {brier:.4f}"
    )

    print(
        "Test positives:",
        test_positive
    )

    print(
        "Test negatives:",
        test_negative
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
