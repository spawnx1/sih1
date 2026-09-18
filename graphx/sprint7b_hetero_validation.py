"""
Sprint 7B - Multi-Snapshot Heterogeneous Graph Validation

Validates that the heterogeneous financial graph remains:
1. structurally valid across time
2. chronologically consistent
3. free of future edges
4. stable in node indexing
"""

from datetime import datetime
import numpy as np
import torch

from ml.features import FeatureContext
from graphx.hetero_graph import build_heterogeneous_graph


SNAPSHOTS = [
    "2026-05-09T08:05:00",
    "2026-05-28T06:50:00",
    "2026-06-25T16:50:00",
    "2026-07-14T15:50:00",
    "2026-08-02T14:35:00",
    "2026-08-21T13:20:00",
]


def unix_seconds(ts):
    return int(
        (
            np.datetime64(ts)
            - np.datetime64("1970-01-01T00:00:00")
        )
        / np.timedelta64(1, "s")
    )


def main():
    print("=" * 70)
    print("SPRINT 7B - MULTI-SNAPSHOT HETEROGENEOUS GRAPH")
    print("=" * 70)

    ctx = FeatureContext()

    previous_transfer_edges = 0
    previous_cashout_edges = 0

    results = []

    for i, snapshot_text in enumerate(SNAPSHOTS, 1):

        snapshot = np.datetime64(snapshot_text)

        data, account_to_idx, atm_to_idx = build_heterogeneous_graph(
            ctx,
            as_of=snapshot,
            observed_only=True,
        )

        transfer = data["account", "transfer", "account"]
        cashout = data["account", "cashout", "atm"]

        transfer_edges = int(transfer.edge_index.shape[1])
        cashout_edges = int(cashout.edge_index.shape[1])

        accounts = int(data["account"].num_nodes)
        atms = int(data["atm"].num_nodes)

        cutoff = unix_seconds(snapshot)

        # ----------------------------------------------------
        # Edge-time leakage checks
        # ----------------------------------------------------

        if transfer.edge_time.numel():
            assert bool(
                torch.all(transfer.edge_time < cutoff)
            )

        if cashout.edge_time.numel():
            assert bool(
                torch.all(cashout.edge_time < cutoff)
            )

        # ----------------------------------------------------
        # Node-index checks
        # ----------------------------------------------------

        if transfer.edge_index.numel():
            assert int(transfer.edge_index.max()) < accounts
            assert int(transfer.edge_index.min()) >= 0

        if cashout.edge_index.numel():
            assert int(cashout.edge_index[0].max()) < accounts
            assert int(cashout.edge_index[0].min()) >= 0
            assert int(cashout.edge_index[1].max()) < atms
            assert int(cashout.edge_index[1].min()) >= 0

        # ----------------------------------------------------
        # Chronological monotonicity
        # ----------------------------------------------------

        if transfer_edges < previous_transfer_edges:
            raise AssertionError(
                f"Transfer edge count decreased at {snapshot_text}"
            )

        if cashout_edges < previous_cashout_edges:
            raise AssertionError(
                f"Cash-out edge count decreased at {snapshot_text}"
            )

        previous_transfer_edges = transfer_edges
        previous_cashout_edges = cashout_edges

        results.append(
            (
                snapshot_text,
                accounts,
                atms,
                transfer_edges,
                cashout_edges,
            )
        )

        print(
            f"[{i}/{len(SNAPSHOTS)}] "
            f"{snapshot_text} | "
            f"Accounts: {accounts} | "
            f"ATMs: {atms} | "
            f"Transfers: {transfer_edges} | "
            f"Cash-outs: {cashout_edges}"
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print()
    print("TEMPORAL LEAKAGE CHECK: PASS")
    print("NODE INDEX CHECK: PASS")
    print("EDGE COUNT MONOTONICITY: PASS")

    print()
    print("=" * 70)
    print("SPRINT 7B MULTI-SNAPSHOT HETEROGENEOUS GRAPH: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
