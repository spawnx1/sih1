"""
Test temporal/as-of correctness of the PyG transaction graph.
"""

import numpy as np

import config

from ml.features import FeatureContext
from graphx.pyg_data import build_transaction_graph


def main():

    print("Loading FeatureContext...")
    ctx = FeatureContext(config.DATA_DIR)

    # ------------------------------------------------------------
    # Choose two timestamps from the actual transaction timeline.
    # ------------------------------------------------------------

    t1 = ctx.ts[len(ctx.ts) // 3]
    t2 = ctx.ts[(len(ctx.ts) * 2) // 3]

    print()
    print("T1:", t1)
    print("T2:", t2)

    assert t1 < t2

    # ------------------------------------------------------------
    # Build two historical snapshots.
    # ------------------------------------------------------------

    print()
    print("Building snapshot T1...")

    data1, _, _ = build_transaction_graph(
        ctx,
        as_of=t1.astype("datetime64[us]").astype(object),
        observed_only=True,
    )

    print("Building snapshot T2...")

    data2, _, _ = build_transaction_graph(
        ctx,
        as_of=t2.astype("datetime64[us]").astype(object),
        observed_only=True,
    )

    # ------------------------------------------------------------
    # Basic summary
    # ------------------------------------------------------------

    print()
    print("========== SNAPSHOT SUMMARY ==========")

    print("T1 nodes:", data1.num_nodes)
    print("T1 edges:", data1.edge_index.shape[1])

    print("T2 nodes:", data2.num_nodes)
    print("T2 edges:", data2.edge_index.shape[1])

    # ------------------------------------------------------------
    # Temporal integrity.
    #
    # Every edge in snapshot T must satisfy:
    #
    #       edge_time < T
    #
    # ------------------------------------------------------------

    print()
    print("========== TEMPORAL INTEGRITY ==========")

    t1_seconds = (
        t1.astype("datetime64[s]")
        .astype(np.int64)
    )

    t2_seconds = (
        t2.astype("datetime64[s]")
        .astype(np.int64)
    )

    assert data1.edge_time.numel() == 0 or bool(
        (data1.edge_time < t1_seconds).all()
    )

    print("T1 future-edge check: PASS")

    assert data2.edge_time.numel() == 0 or bool(
        (data2.edge_time < t2_seconds).all()
    )

    print("T2 future-edge check: PASS")

    # ------------------------------------------------------------
    # Later snapshot should not contain fewer historical edges
    # than earlier snapshot.
    # ------------------------------------------------------------

    print()
    print("========== MONOTONICITY CHECK ==========")

    assert data2.edge_index.shape[1] >= data1.edge_index.shape[1]

    print("Historical edge count monotonicity: PASS")

    # ------------------------------------------------------------
    # Result
    # ------------------------------------------------------------

    print()
    print("========== RESULT ==========")
    print("TEMPORAL GRAPH: PASS")


if __name__ == "__main__":
    main()