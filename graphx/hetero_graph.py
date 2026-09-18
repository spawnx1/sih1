from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import HeteroData

from ml.features import FeatureContext


def build_heterogeneous_graph(
    ctx: FeatureContext,
    as_of=None,
    observed_only=True,
):
    """
    Build a temporal heterogeneous financial graph.

    Node types:
        account
        atm

    Edge types:
        account -> account : transfer
        account -> atm     : cash-out

    Only transactions with ts < as_of are included.
    """

    # --------------------------------------------------------
    # Temporal cutoff
    # --------------------------------------------------------

    if as_of is None:
        cutoff = np.datetime64(
            ctx.ts[-1],
            "ns",
        )
    else:
        cutoff = np.datetime64(
            as_of,
            "ns",
        )

    mask = ctx.ts < cutoff

    if observed_only:
        mask &= ctx.observed

    positions = np.flatnonzero(mask)

    # --------------------------------------------------------
    # Containers
    # --------------------------------------------------------

    account_ids = set()
    atm_ids = set()

    transfer_positions = []
    cashout_positions = []

    # --------------------------------------------------------
    # Scan historical transactions
    # --------------------------------------------------------

    for pos in positions:

        src = ctx.src[pos]
        dst = ctx.dst[pos]
        channel = ctx.channel[pos]
        atm_id = ctx.atm_id[pos]

        account_ids.add(src)

        # Account -> Account
        #
        # Cash-out transactions are handled separately.
        if not ctx.is_cashout[pos]:

            if dst is not None and dst != "":
                account_ids.add(dst)

                transfer_positions.append(
                    pos
                )

        # Account -> ATM
        else:

            if (
                atm_id is not None
                and atm_id != ""
            ):
                atm_ids.add(atm_id)

                cashout_positions.append(
                    pos
                )

    # --------------------------------------------------------
    # --------------------------------------------------------
    # Stable node ordering
    # --------------------------------------------------------

    account_ids = sorted(str(x) for x in account_ids)
    atm_ids = sorted(str(x) for x in atm_ids)

    account_to_idx = {
        account_id: i
        for i, account_id
        in enumerate(account_ids)
    }

    atm_to_idx = {
        atm_id: i
        for i, atm_id
        in enumerate(atm_ids)
    }
    # --------------------------------------------------------
    # Create HeteroData
    # --------------------------------------------------------

    data = HeteroData()

    data["account"].num_nodes = len(
        account_ids
    )

    data["atm"].num_nodes = len(
        atm_ids
    )

    # --------------------------------------------------------
    # Account -> Account transfer edges
    # --------------------------------------------------------

    transfer_src = []
    transfer_dst = []
    transfer_amount = []
    transfer_time = []
    transfer_txn_ids = []

    for pos in transfer_positions:

        src = ctx.src[pos]
        dst = ctx.dst[pos]

        if (
            src not in account_to_idx
            or dst not in account_to_idx
        ):
            continue

        transfer_src.append(
            account_to_idx[src]
        )

        transfer_dst.append(
            account_to_idx[dst]
        )

        transfer_amount.append(
            float(ctx.amount[pos])
        )

        transfer_time.append(
            ctx.ts[pos]
        )

        transfer_txn_ids.append(
            str(ctx.txn_id[pos])
        )

    if transfer_src:

        data[
            "account",
            "transfer",
            "account",
        ].edge_index = torch.tensor(
            [
                transfer_src,
                transfer_dst,
            ],
            dtype=torch.long,
        )

        data[
            "account",
            "transfer",
            "account",
        ].edge_attr = torch.tensor(
            transfer_amount,
            dtype=torch.float32,
        ).view(-1, 1)

        transfer_seconds = np.array(
            [
                (
                    np.datetime64(t)
                    - np.datetime64(
                        "1970-01-01T00:00:00"
                    )
                )
                / np.timedelta64(1, "s")
                for t in transfer_time
            ],
            dtype=np.float64,
        )

        data[
            "account",
            "transfer",
            "account",
        ].edge_time = torch.tensor(
            transfer_seconds,
            dtype=torch.long,
        )

    else:

        data[
            "account",
            "transfer",
            "account",
        ].edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

        data[
            "account",
            "transfer",
            "account",
        ].edge_attr = torch.empty(
            (0, 1),
            dtype=torch.float32,
        )

        data[
            "account",
            "transfer",
            "account",
        ].edge_time = torch.empty(
            (0,),
            dtype=torch.long,
        )

    # --------------------------------------------------------
    # Account -> ATM cash-out edges
    # --------------------------------------------------------

    cash_src = []
    cash_dst = []
    cash_amount = []
    cash_time = []
    cash_txn_ids = []

    for pos in cashout_positions:

        src = ctx.src[pos]
        atm_id = ctx.atm_id[pos]

        if (
            src not in account_to_idx
            or atm_id not in atm_to_idx
        ):
            continue

        cash_src.append(
            account_to_idx[src]
        )

        cash_dst.append(
            atm_to_idx[atm_id]
        )

        cash_amount.append(
            float(ctx.amount[pos])
        )

        cash_time.append(
            ctx.ts[pos]
        )

        cash_txn_ids.append(
            str(ctx.txn_id[pos])
        )

    if cash_src:

        data[
            "account",
            "cashout",
            "atm",
        ].edge_index = torch.tensor(
            [
                cash_src,
                cash_dst,
            ],
            dtype=torch.long,
        )

        data[
            "account",
            "cashout",
            "atm",
        ].edge_attr = torch.tensor(
            cash_amount,
            dtype=torch.float32,
        ).view(-1, 1)

        cash_seconds = np.array(
            [
                (
                    np.datetime64(t)
                    - np.datetime64(
                        "1970-01-01T00:00:00"
                    )
                )
                / np.timedelta64(1, "s")
                for t in cash_time
            ],
            dtype=np.float64,
        )

        data[
            "account",
            "cashout",
            "atm",
        ].edge_time = torch.tensor(
            cash_seconds,
            dtype=torch.long,
        )

    else:

        data[
            "account",
            "cashout",
            "atm",
        ].edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

        data[
            "account",
            "cashout",
            "atm",
        ].edge_attr = torch.empty(
            (0, 1),
            dtype=torch.float32,
        )

        data[
            "account",
            "cashout",
            "atm",
        ].edge_time = torch.empty(
            (0,),
            dtype=torch.long,
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    data.account_id = account_ids
    data.atm_id = atm_ids

    data.transfer_txn_id = transfer_txn_ids
    data.cashout_txn_id = cash_txn_ids

    data.snapshot_time = cutoff

    return (
        data,
        account_to_idx,
        atm_to_idx,
    )


def main():

    print("=" * 70)
    print("SPRINT 7A - HETEROGENEOUS FINANCIAL GRAPH")
    print("=" * 70)

    ctx = FeatureContext()

    snapshot = np.datetime64(
        "2026-07-14T12:00:00"
    )

    print(
        "Snapshot:",
        snapshot,
    )

    data, account_to_idx, atm_to_idx = (
        build_heterogeneous_graph(
            ctx,
            as_of=snapshot,
            observed_only=True,
        )
    )

    print()
    print("NODE TYPES")
    print(
        "Accounts:",
        data["account"].num_nodes,
    )
    print(
        "ATMs:",
        data["atm"].num_nodes,
    )

    print()
    print("EDGE TYPES")

    transfer = data[
        "account",
        "transfer",
        "account",
    ]

    cashout = data[
        "account",
        "cashout",
        "atm",
    ]

    print(
        "Account -> Account:",
        transfer.edge_index.shape[1],
    )

    print(
        "Account -> ATM:",
        cashout.edge_index.shape[1],
    )

    print()
    print("EDGE ATTRIBUTES")

    print(
        "Transfer edge_attr:",
        tuple(transfer.edge_attr.shape),
    )

    print(
        "Transfer edge_time:",
        tuple(transfer.edge_time.shape),
    )

    print(
        "Cash-out edge_attr:",
        tuple(cashout.edge_attr.shape),
    )

    print(
        "Cash-out edge_time:",
        tuple(cashout.edge_time.shape),
    )

    # --------------------------------------------------------
    # Integrity checks
    # --------------------------------------------------------

    assert (
        transfer.edge_index.shape[1]
        == transfer.edge_attr.shape[0]
    )

    assert (
        transfer.edge_index.shape[1]
        == transfer.edge_time.shape[0]
    )

    assert (
        cashout.edge_index.shape[1]
        == cashout.edge_attr.shape[0]
    )

    assert (
        cashout.edge_index.shape[1]
        == cashout.edge_time.shape[0]
    )

    # Account indices
    if transfer.edge_index.numel():

        assert int(
            transfer.edge_index.max()
        ) < data["account"].num_nodes

        assert int(
            transfer.edge_index.min()
        ) >= 0

    # ATM indices
    if cashout.edge_index.numel():

        assert int(
            cashout.edge_index[1].max()
        ) < data["atm"].num_nodes

        assert int(
            cashout.edge_index[1].min()
        ) >= 0

    # --------------------------------------------------------
    # Temporal leakage check
    # --------------------------------------------------------

    cutoff_seconds = int(
        (
            snapshot
            - np.datetime64(
                "1970-01-01T00:00:00"
            )
        )
        / np.timedelta64(1, "s")
    )

    if transfer.edge_time.numel():

        assert bool(
            torch.all(
                transfer.edge_time
                < cutoff_seconds
            )
        )

    if cashout.edge_time.numel():

        assert bool(
            torch.all(
                cashout.edge_time
                < cutoff_seconds
            )
        )

    print()
    print("NODE INDEX CHECK: PASS")
    print("EDGE ATTRIBUTE CHECK: PASS")
    print("TEMPORAL CUTOFF CHECK: PASS")

    print()
    print("=" * 70)
    print("SPRINT 7A HETEROGENEOUS GRAPH: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()




