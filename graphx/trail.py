"""
graphx/trail.py -- time-respecting money-flow reconstruction.

expand_trail(seed_txn_id, as_of, ctx) does a breadth-first walk from the seed
transaction, following ONLY edges where parent.ts < edge.ts < as_of, capped at
MAX_HOPS and at edges worth >= MIN_EDGE_FRACTION of the seed amount. Tainted
money is attributed proportionally through each split.

R1 lives or dies here: the graph is rebuilt fresh at every as_of. There is NO
precomputed whole-corpus graph -- that would be the first and worst leak. The
`ts < as_of` filter is delegated to FeatureContext.out_edges_between so it is
done exactly once, the same way everywhere.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime

import pandas as pd

import config
from contracts import Trail, TrailEdge, TrailNode


def _py(ts) -> datetime:
    return pd.Timestamp(ts).to_pydatetime()


def expand_trail(seed_txn_id: str, as_of: datetime, ctx) -> Trail:
    if seed_txn_id not in ctx.txn_pos:
        raise KeyError(f"unknown seed txn {seed_txn_id}")
    sp = ctx.txn_pos[seed_txn_id]
    seed_amt = float(ctx.amount[sp])
    seed_ts = _py(ctx.ts[sp])
    victim = ctx.src[sp]
    first_hop_acc = ctx.dst[sp]

    min_edge = config.MIN_EDGE_FRACTION * seed_amt

    # node state keyed by account
    state: dict[str, dict] = {}

    def touch(acc, hop, ts):
        if acc not in state:
            state[acc] = {"hop": hop, "in": 0.0, "out": 0.0, "first_ts": ts}
        else:
            state[acc]["hop"] = min(state[acc]["hop"], hop)

    edges: list[TrailEdge] = []

    # seed edge: victim -> first mule, fully tainted
    touch(victim, 0, seed_ts)
    touch(first_hop_acc, 1, seed_ts)
    state[victim]["out"] += seed_amt
    state[first_hop_acc]["in"] += seed_amt
    edges.append(TrailEdge(
        txn_id=seed_txn_id, src_account=victim, dst_account=first_hop_acc,
        amount=seed_amt, tainted_amount=seed_amt, ts=seed_ts,
        channel=str(ctx.channel[sp]), hop=1, observed=bool(ctx.observed[sp]),
        atm_id=None,
    ))

    # BFS: (account, hop, arrival_ts, tainted_held)
    q: deque = deque()
    q.append((first_hop_acc, 1, seed_ts, seed_amt))
    visited_expand: set[str] = set()

    while q:
        acc, hop, t_in, tainted_held = q.popleft()
        if hop >= config.MAX_HOPS or acc in visited_expand:
            continue
        visited_expand.add(acc)

        out_pos = ctx.out_edges_between(acc, t_in, as_of)
        if len(out_pos) == 0:
            continue
        # qualifying edges (>= 5% of seed)
        amts = ctx.amount[out_pos]
        keep = amts >= min_edge
        out_pos, amts = out_pos[keep], amts[keep]
        if len(out_pos) == 0:
            continue
        s_out = float(amts.sum())

        for p, amt in zip(out_pos, amts):
            amt = float(amt)
            # proportional attribution, capped at the edge amount
            if s_out > 0:
                tainted_e = min(amt, tainted_held * amt / s_out)
            else:
                tainted_e = 0.0
            if tainted_e < min_edge:
                continue
            e_ts = _py(ctx.ts[p])
            dst = ctx.dst[p]
            is_cashout = bool(ctx.is_cashout[p])
            state[acc]["out"] += tainted_e
            edges.append(TrailEdge(
                txn_id=str(ctx.txn_id[p]),
                src_account=acc,
                dst_account=None if is_cashout else dst,
                amount=amt,
                tainted_amount=round(tainted_e, 2),
                ts=e_ts,
                channel=str(ctx.channel[p]),
                hop=hop + 1,
                observed=bool(ctx.observed[p]),
                atm_id=str(ctx.atm_id[p]) if is_cashout else None,
            ))
            if not is_cashout and dst is not None:
                touch(dst, hop + 1, e_ts)
                state[dst]["in"] += tainted_e
                q.append((dst, hop + 1, e_ts, tainted_e))

    # assemble nodes; a leaf holds funds (received > forwarded/cashed) at as_of
    nodes: list[TrailNode] = []
    leaf_accounts: list[str] = []
    for acc, st in state.items():
        residual = st["in"] - st["out"]
        is_leaf = (acc != victim) and (residual > min_edge)
        if is_leaf:
            leaf_accounts.append(acc)
        nodes.append(TrailNode(
            account_id=acc,
            hop_from_victim=st["hop"],
            tainted_amount=round(max(st["in"], st["out"]), 2),
            is_victim=(acc == victim),
            is_leaf=is_leaf,
        ))

    return Trail(
        ack_no="",  # filled by the API from the complaint
        seed_txn_id=seed_txn_id,
        as_of=as_of,
        total_tainted=round(seed_amt, 2),
        nodes=nodes,
        edges=edges,
        leaf_accounts=leaf_accounts,
    )
