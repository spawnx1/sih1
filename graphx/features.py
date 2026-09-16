"""
graphx/features.py -- nine graph features per node, plus a mule-node score.

The nine features (all as-of correct -- computed only from ts < as_of, visible
edges):

  1 hop_from_victim          (from the trail; large if the node isn't in it)
  2 in_deg_1h                distinct senders in the last hour
  3 out_deg_1h               distinct recipients (+cash-outs) in the last hour
  4 dispersion_ratio         out-amount / in-amount over the last hour
  5 flow_concentration       Herfindahl over out-edge amounts (last hour)
  6 ring_size                size of the account's mule ring (0 if none)
  7 ring_prior_cashout_rate  historical cash-out rate across ring members
  8 n_tainted_predecessors   in-trail in-degree (tainted edges into the node)
  9 mule_score               logistic regression over features 1-8

The mule score is a FEATURE, not a headline model (it is one column in the M1
vector). It is fit by MuleScorer.fit over a train-period account snapshot.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

import config

BASE_FEATURES = [
    "hop_from_victim", "in_deg_1h", "out_deg_1h", "dispersion_ratio",
    "flow_concentration", "ring_size", "ring_prior_cashout_rate",
    "n_tainted_predecessors",
]
GRAPH_FEATURES = BASE_FEATURES + ["mule_score"]

_NO_TRAIL_HOP = config.MAX_HOPS + 5


def compute_graph_features(account_id: str, as_of: datetime, ctx,
                           trail=None, window_min: int = 60) -> dict:
    win_lo = as_of - timedelta(minutes=window_min)

    in_pos = ctx.in_positions(account_id, as_of)
    out_pos = ctx.out_positions(account_id, as_of)
    in_win = in_pos[ctx.ts[in_pos] >= np.datetime64(win_lo)] if len(in_pos) else in_pos
    out_win = out_pos[ctx.ts[out_pos] >= np.datetime64(win_lo)] if len(out_pos) else out_pos

    # degrees over the last hour
    in_srcs = {s for s in ctx.src[in_win] if s is not None}
    out_dsts = {d for d in ctx.dst[out_win] if d is not None}
    n_cashout_win = int(ctx.is_cashout[out_win].sum()) if len(out_win) else 0
    in_deg_1h = len(in_srcs)
    out_deg_1h = len(out_dsts) + n_cashout_win

    in_amt = float(ctx.amount[in_win].sum()) if len(in_win) else 0.0
    out_amt = float(ctx.amount[out_win].sum()) if len(out_win) else 0.0
    dispersion_ratio = out_amt / (in_amt + 1.0)

    if len(out_win):
        amts = ctx.amount[out_win].astype(float)
        tot = amts.sum()
        flow_concentration = float(((amts / tot) ** 2).sum()) if tot > 0 else 0.0
    else:
        flow_concentration = 0.0

    # ring features
    members = ctx.ring_members(account_id)
    ring_size = len(members) + 1 if members else 0
    if members:
        rates = []
        for m in members:
            mo = ctx.out_positions(m, as_of)
            if len(mo) == 0:
                continue
            co = int(ctx.is_cashout[mo].sum())
            rates.append(co / len(mo))
        ring_prior_cashout_rate = float(np.mean(rates)) if rates else 0.0
    else:
        ring_prior_cashout_rate = 0.0

    # trail-derived features
    hop_from_victim = _NO_TRAIL_HOP
    n_tainted_predecessors = 0
    if trail is not None:
        for node in trail.nodes:
            if node.account_id == account_id:
                hop_from_victim = node.hop_from_victim
                break
        n_tainted_predecessors = sum(
            1 for e in trail.edges if e.dst_account == account_id
        )

    return {
        "hop_from_victim": float(hop_from_victim),
        "in_deg_1h": float(in_deg_1h),
        "out_deg_1h": float(out_deg_1h),
        "dispersion_ratio": float(dispersion_ratio),
        "flow_concentration": float(flow_concentration),
        "ring_size": float(ring_size),
        "ring_prior_cashout_rate": float(ring_prior_cashout_rate),
        "n_tainted_predecessors": float(n_tainted_predecessors),
    }


class MuleScorer:
    """Small logistic regression producing the `mule_score` feature."""

    def __init__(self):
        self.model = None
        self.scaler = None

    def fit(self, ctx, n_sample: int = 3000, seed: int = 26184):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(seed)
        accts = ctx.accounts["account_id"].to_numpy()
        labels = ctx.accounts["label_is_mule"].to_numpy()
        idx = rng.choice(len(accts), size=min(n_sample, len(accts)), replace=False)
        # snapshot at the train cutoff (historical, as-of correct)
        as_of = _ts_from_int(ctx.train_cutoff)

        X, y = [], []
        for i in idx:
            f = compute_graph_features(str(accts[i]), as_of, ctx, trail=None)
            X.append([f[k] for k in BASE_FEATURES])
            y.append(int(labels[i]))
        X = np.array(X)
        y = np.array(y)
        self.scaler = StandardScaler().fit(X)
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.model.fit(self.scaler.transform(X), y)
        return self

    def score(self, features: dict) -> float:
        if self.model is None:
            return 0.0
        x = np.array([[features[k] for k in BASE_FEATURES]])
        return float(self.model.predict_proba(self.scaler.transform(x))[0, 1])


def _ts_from_int(int_ns: float):
    import pandas as pd
    return pd.Timestamp(int(int_ns)).floor("s").to_pydatetime()
