

"""
graphx/graph_baseline.py

Graph Feature Baseline.

Purpose:
    Measure the predictive value of the existing engineered
    graph features before introducing a neural GNN.

Model:
    StandardScaler + LogisticRegression

Features:
    The eight BASE_FEATURES from graphx.features.

Evaluation:
    Uses the same chronological split methodology as M1.

This module does NOT:
    - train a GNN
    - modify the existing M1 model
    - create labels
    - use future transactions as features
"""

from __future__ import annotations

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

import config

from graphx.features import BASE_FEATURES, MuleScorer
from ml.features import FeatureContext
from ml.panel import build_panel
from ml.train import chronological_split


def train_graph_baseline(train, test):
    """
    Train and evaluate a logistic regression using only graph features.
    """

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train[BASE_FEATURES]
    )

    X_test = scaler.transform(
        test[BASE_FEATURES]
    )

    y_train = train["y"].to_numpy()
    y_test = test["y"].to_numpy()

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=config.SEED,
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_test, probabilities)
        ),
        "brier": float(
            brier_score_loss(y_test, probabilities)
        ),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_pos_rate": float(y_train.mean()),
        "test_pos_rate": float(y_test.mean()),
    }

    return model, scaler, metrics


def main():

    print("============================================================")
    print("GRAPH FEATURE BASELINE")
    print("============================================================")

    # ------------------------------------------------------------
    # 1. Build feature context
    # ------------------------------------------------------------

    print("\n[1] Loading FeatureContext...")

    ctx = FeatureContext(config.DATA_DIR)

    ctx.mule_scorer = MuleScorer().fit(ctx)

    # ------------------------------------------------------------
    # 2. Build the same panel used by M1
    # ------------------------------------------------------------

    print("[2] Building panel...")

    panel = build_panel(
        ctx,
        n_legit_base=1800,
        verbose=True,
    )

    # ------------------------------------------------------------
    # 3. Verify required graph features exist
    # ------------------------------------------------------------

    print("\n[3] Checking graph features...")

    missing = [
        feature
        for feature in BASE_FEATURES
        if feature not in panel.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing graph features: {missing}"
        )

    print("Graph features: OK")

    print("\nFeatures used:")

    for feature in BASE_FEATURES:
        print(f"  - {feature}")

    # ------------------------------------------------------------
    # 4. Chronological split
    # ------------------------------------------------------------

    print("\n[4] Chronological split...")

    train, test, cutoff = chronological_split(panel)

    print(f"Cutoff: {cutoff}")
    print(f"Train rows: {len(train)}")
    print(f"Test rows:  {len(test)}")

    print(
        f"Train positive rate: {train['y'].mean():.4f}"
    )

    print(
        f"Test positive rate:  {test['y'].mean():.4f}"
    )

    # ------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------

    print("\n[5] Training Graph Logistic Regression...")

    model, scaler, metrics = train_graph_baseline(
        train,
        test,
    )

    # ------------------------------------------------------------
    # 6. Results
    # ------------------------------------------------------------

    print("\n============================================================")
    print("GRAPH BASELINE RESULTS")
    print("============================================================")

    print(
        f"ROC-AUC : {metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC  : {metrics['pr_auc']:.4f}"
    )

    print(
        f"Brier   : {metrics['brier']:.4f}"
    )

    print(
        f"Train   : {metrics['n_train']}"
    )

    print(
        f"Test    : {metrics['n_test']}"
    )

    # ------------------------------------------------------------
    # 7. Individual feature audit
    # ------------------------------------------------------------

    print("\n============================================================")
    print("GRAPH FEATURE AUDIT")
    print("============================================================")

    y = test["y"].to_numpy()

    results = []

    for feature in BASE_FEATURES:

        x = test[feature].to_numpy(dtype=float)

        if np.unique(x).size < 2:
            continue

        auc = roc_auc_score(y, x)

        results.append(
            (
                feature,
                max(auc, 1.0 - auc),
            )
        )

    results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    for feature, auc in results:

        print(
            f"{feature:30s} "
            f"AUC={auc:.4f}"
        )

    print("\n============================================================")
    print("RESULT")
    print("============================================================")

    print("GRAPH FEATURE BASELINE: PASS")


if __name__ == "__main__":
    main()