"""
graphx/pyg_data.py

Convert the existing transaction data from FeatureContext
into a PyTorch Geometric graph.

Responsibilities:
    - map account IDs -> integer node IDs
    - create edge_index
    - preserve transaction attributes
    - preserve timestamps for future temporal models
    - create node mapping

This module DOES NOT:
    - train models
    - create labels
    - generate candidates
    - reconstruct trails
    - modify FeatureContext
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import torch
from torch_geometric.data import Data

from ml.features import FeatureContext


def build_transaction_graph(
    ctx: FeatureContext,
    as_of: datetime | None = None,
    observed_only: bool = True,
) -> tuple[Data, dict[str, int], list[str]]:
    """
    Build a PyTorch Geometric transaction graph.

    Nodes:
        Accounts

    Edges:
        Account-to-account transactions.

    Returns:
        data:
            PyTorch Geometric Data object.

        node_to_id:
            Mapping from account_id -> integer node index.

        id_to_node:
            Mapping from integer node index -> account_id.
    """

    # ============================================================
    # 1. Select transactions
    # ============================================================

    mask = np.ones(len(ctx.tx), dtype=bool)

    # Only transactions strictly before as_of are visible.
    if as_of is not None:
        mask &= ctx.ts < np.datetime64(as_of)

    # Investigator-visible graph:
    # only transactions confirmed/observed by the bank.
    if observed_only:
        mask &= ctx.observed

    positions = np.flatnonzero(mask)

    # ============================================================
    # 2. Collect valid account nodes
    # ============================================================

    src_accounts = ctx.src[positions]
    dst_accounts = ctx.dst[positions]

    accounts = {
        account
        for account in src_accounts
        if isinstance(account, str) and account
    }

    accounts.update(
        account
        for account in dst_accounts
        if isinstance(account, str) and account
    )

    accounts = sorted(accounts)

    # Account ID -> integer node index
    node_to_id = {
        account_id: idx
        for idx, account_id in enumerate(accounts)
    }

    # Integer node index -> account ID
    id_to_node = accounts

    # ============================================================
    # 3. Convert transactions into graph edges
    # ============================================================

    src_nodes = []
    dst_nodes = []
    valid_positions = []

    for p in positions:

        src_account = ctx.src[p]
        dst_account = ctx.dst[p]

        if not isinstance(src_account, str) or not src_account:
            continue

        if src_account not in node_to_id:
            continue

        # Cash-out transactions may have no destination account.
        # They are excluded from this first account-only graph.
        if not isinstance(dst_account, str) or not dst_account:
            continue

        if dst_account not in node_to_id:
            continue

        src_nodes.append(node_to_id[src_account])
        dst_nodes.append(node_to_id[dst_account])
        valid_positions.append(p)

    # ============================================================
    # 4. Build edge_index
    # ============================================================

    if valid_positions:

        edge_index = torch.tensor(
            [src_nodes, dst_nodes],
            dtype=torch.long,
        )

    else:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

    # ============================================================
    # 5. Build edge attributes
    # ============================================================
    #
    # Model edge attributes:
    #
    #   0 -> transaction amount
    #   1 -> observed flag
    #
    # Timestamp is NOT placed inside edge_attr.
    #
    # It is stored separately as edge_time below so that temporal
    # models can use it without feeding raw Unix timestamps into
    # the neural network.
    # ============================================================

    if valid_positions:

        amounts = ctx.amount[valid_positions].astype(np.float32)

        observed = (
            ctx.observed[valid_positions]
            .astype(np.float32)
        )

        edge_attr = torch.tensor(
            np.column_stack(
                [
                    amounts,
                    observed,
                ]
            ),
            dtype=torch.float32,
        )

    else:

        edge_attr = torch.empty(
            (0, 2),
            dtype=torch.float32,
        )

    # ============================================================
    # 6. Create PyTorch Geometric Data object
    # ============================================================

    data = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(accounts),
    )

    # ============================================================
    # 7. Preserve transaction timestamps separately
    # ============================================================

    if valid_positions:

        data.edge_time = torch.tensor(
            ctx.ts[valid_positions]
            .astype("datetime64[s]")
            .astype(np.int64),
            dtype=torch.long,
        )

    else:

        data.edge_time = torch.empty(
            (0,),
            dtype=torch.long,
        )

    # ============================================================
    # 8. Preserve transaction IDs
    # ============================================================

    if valid_positions:

        data.txn_id = [
            str(ctx.txn_id[p])
            for p in valid_positions
        ]

    else:

        data.txn_id = []

    # ============================================================
    # 9. Return graph + mappings
    # ============================================================

    return data, node_to_id, id_to_node