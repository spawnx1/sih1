"""
Basic integrity tests for as-of node features.
"""

from datetime import datetime

import torch

import config

from ml.features import FeatureContext
from graphx.pyg_data import build_transaction_graph
from graphx.node_features import attach_node_features
from graphx.features import BASE_FEATURES


def main():

    print("Loading FeatureContext...")
    ctx = FeatureContext(config.DATA_DIR)

    print("Building graph...")
    data, node_to_id, id_to_node = build_transaction_graph(
        ctx,
        observed_only=True,
    )

    # ------------------------------------------------------------
    # Pick a real timestamp from the transaction data.
    # ------------------------------------------------------------

    as_of = ctx.ts[len(ctx.ts) // 2]

    # Convert numpy datetime64 -> Python datetime
    as_of = (
        as_of.astype("datetime64[us]")
        .astype(datetime)
    )

    print()
    print("Snapshot:", as_of)

    # ------------------------------------------------------------
    # Build node features
    # ------------------------------------------------------------

    data = attach_node_features(
        data=data,
        ctx=ctx,
        id_to_node=id_to_node,
        as_of=as_of,
    )

    print()
    print("========== NODE FEATURE SUMMARY ==========")

    print("Nodes:", data.num_nodes)
    print("Features:", len(BASE_FEATURES))
    print("data.x shape:", tuple(data.x.shape))

    print()
    print("Feature names:")

    for i, name in enumerate(BASE_FEATURES):
        print(f"  {i}: {name}")

    # ------------------------------------------------------------
    # Integrity checks
    # ------------------------------------------------------------

    print()
    print("========== INTEGRITY CHECKS ==========")

    assert data.x.dim() == 2
    print("X dimensions: OK")

    assert data.x.shape[0] == data.num_nodes
    print("Node alignment: OK")

    assert data.x.shape[1] == len(BASE_FEATURES)
    print("Feature count: OK")

    assert data.x.dtype == torch.float32
    print("Data type: OK")

    assert not torch.isnan(data.x).any()
    print("NaN check: OK")

    assert not torch.isinf(data.x).any()
    print("Inf check: OK")

    # ------------------------------------------------------------
    # Display a few rows
    # ------------------------------------------------------------

    print()
    print("========== SAMPLE FEATURES ==========")

    for i in range(min(5, data.num_nodes)):

        account_id = id_to_node[i]
        values = data.x[i].tolist()

        print()
        print("Account:", account_id)

        for name, value in zip(BASE_FEATURES, values):
            print(f"  {name}: {value:.4f}")

    print()
    print("========== RESULT ==========")
    print("NODE FEATURES: PASS")


if __name__ == "__main__":
    main()