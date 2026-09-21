"""
ml/features.py -- the as-of feature store.

FeatureContext pre-sorts transactions by ts and builds per-account index arrays
so that `hist(account, as_of)` is a np.searchsorted cut, not a dataframe scan.
Everything downstream (graph, candidates, panel, models, API) rides on one
FeatureContext instance.

Two hard rules are enforced structurally here:

  R1 (as-of): every feature reads only transactions with ts < as_of. The cut is
      done once, centrally, by `_before`.
  R2 (label_) / R3 (internal): `feature_columns()` drops any column whose name
      starts with `label_` or is in config.INTERNAL_COLUMNS, so a label can
      never reach the model even if someone adds one to a feature dict.

Investigator visibility (partial observability): feature/graph reads use only
`observed == True` edges (the bank replied). Ground-truth LABELS, built in
ml/panel.py, may use all edges -- in hindsight we know the debit happened even
if that edge was never confirmed at the time.
"""
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd

import config

CASHOUT_CHANNELS = {"ATM_WDL", "POS"}


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """R2 + R3: strip every ground-truth and generator-internal column."""
    keep = [
        c for c in df.columns
        if not c.startswith(config.LABEL_PREFIX) and c not in config.INTERNAL_COLUMNS
    ]
    return df[keep]


class FeatureContext:
    def __init__(self, data_dir: str | None = None):
        self.data_dir = data_dir or config.DATA_DIR
        self.atms = pd.read_parquet(os.path.join(self.data_dir, "atms.parquet"))
        self.accounts = pd.read_parquet(os.path.join(self.data_dir, "accounts.parquet"))
        tx = pd.read_parquet(os.path.join(self.data_dir, "transactions.parquet"))
        tx = tx.sort_values("ts", kind="stable").reset_index(drop=True)
        self.tx = tx

        # column arrays (fast, no per-call dataframe access)
        self.ts = tx["ts"].to_numpy(dtype="datetime64[ns]")
        self.src = tx["src_account"].to_numpy(dtype=object)
        self.dst = tx["dst_account"].to_numpy(dtype=object)
        self.amount = tx["amount"].to_numpy(dtype=float)
        self.channel = tx["channel"].to_numpy(dtype=object)
        self.atm_id = tx["atm_id"].to_numpy(dtype=object)
        self.observed = tx["observed"].to_numpy(dtype=bool)
        self.is_cashout = np.isin(self.channel, list(CASHOUT_CHANNELS))
        # labels kept for panel/priors ONLY -- never exposed via feature dicts
        self._label_tainted = tx["label_is_tainted"].to_numpy(dtype=bool)
        self._label_case = tx["label_case_id"].to_numpy(dtype=object)

        # per-account position arrays (global positions, ts-ascending)
        self.src_pos = {k: np.asarray(v) for k, v in
                        tx.groupby("src_account").indices.items()}
        self.dst_pos = {k: np.asarray(v) for k, v in
                        tx.groupby("dst_account").indices.items()}
        self.txn_pos = {t: i for i, t in enumerate(tx["txn_id"].to_numpy())}
        self.txn_id = tx["txn_id"].to_numpy(dtype=object)

        # account lookups
        self.acc = self.accounts.set_index("account_id")
        self._ring_members: dict[str, list[str]] = {}
        for rid, grp in self.accounts.dropna(subset=["label_ring_id"]).groupby("label_ring_id"):
            self._ring_members[rid] = list(grp["account_id"])

        # atm lookups
        self.atm = self.atms.set_index("atm_id")
        self._atm_lat = self.atm["lat"].to_dict()
        self._atm_lon = self.atm["lon"].to_dict()
        self._atm_cell = self.atm["cell"].to_dict()
        self._atm_cell_fine = self.atm["cell_fine"].to_dict()
        self._atm_district = self.atm["district"].to_dict()
        self._atm_pin = self.atm["pin"].to_dict()
        self._atm_limit = self.atm["per_txn_limit"].to_dict()
        self._atm_operator = self.atm["operator"].to_dict()
        self._atm_standalone = self.atm["is_standalone"].to_dict()
        self._atm_hourly = {k: list(v) for k, v in self.atm["hourly_weight"].items()}
        self._acc_branch = self.accounts.set_index("account_id")[["branch_lat", "branch_lon"]]

        # ---- train-period-only priors (historical, not per-row labels) -----
        self.train_cutoff = np.quantile(self.ts.astype("int64"), 0.70)
        self._build_priors()

        # transient binding for graph/case features (set per panel row)
        self._trail = None
        self._case = None
        self.mule_scorer = None

    def bind(self, trail=None, case=None):
        """Attach the current case's trail + complaint so graph/case feature
        functions keep the uniform f(account_id, as_of, ctx) signature (R1)."""
        self._trail = trail
        self._case = case
        return self

    # ---- priors --------------------------------------------------------
    def _build_priors(self):
        train = (self.ts.astype("int64") < self.train_cutoff)
        co = self.is_cashout & train
        atm_ids = self.atm_id[co]
        tainted = self._label_tainted[co]
        # fraud rate per ATM (Laplace-smoothed), train period only
        total = pd.Series(atm_ids).value_counts()
        fraud = pd.Series(atm_ids[tainted]).value_counts()
        rate = {}
        for a in total.index:
            f = float(fraud.get(a, 0))
            n = float(total[a])
            rate[a] = (f + 0.5) / (n + 2.0)
        self._atm_fraud_rate = rate
        self._global_fraud_rate = (tainted.sum() + 0.5) / (co.sum() + 2.0)
        # district hotspot prior = share of fraud cash-outs by district
        d_series = pd.Series([self._atm_district.get(a) for a in atm_ids[tainted]])
        d_counts = d_series.value_counts(normalize=True)
        self._district_hotspot = {d: float(d_counts.get(d, 0.0)) for d in config.DISTRICTS}

    def atm_fraud_rate(self, atm_id: str) -> float:
        return self._atm_fraud_rate.get(atm_id, self._global_fraud_rate)

    def district_hotspot_prior(self, district: str) -> float:
        return self._district_hotspot.get(district, 0.0)

    # ---- atom lookups --------------------------------------------------
    def atm_info(self, atm_id: str) -> dict:
        return {
            "lat": self._atm_lat[atm_id],
            "lon": self._atm_lon[atm_id],
            "cell": self._atm_cell[atm_id],
            "cell_fine": self._atm_cell_fine[atm_id],
            "district": self._atm_district[atm_id],
            "per_txn_limit": self._atm_limit[atm_id],
        }

    def account_info(self, account_id: str) -> dict:
        if account_id not in self.acc.index:
            return {}
        r = self.acc.loc[account_id]
        return {
            "kyc_lat": float(r["kyc_lat"]),
            "kyc_lon": float(r["kyc_lon"]),
            "kyc_district": r["kyc_district"],
            "kyc_pin": r["kyc_pin"],
            "open_date": r["open_date"],
            "ring_id": r["label_ring_id"],
            "bank_code": r["bank_code"],
        }

    def ring_members(self, account_id: str) -> list[str]:
        info = self.account_info(account_id)
        rid = info.get("ring_id")
        if rid is None or (isinstance(rid, float) and np.isnan(rid)):
            return []
        return [m for m in self._ring_members.get(rid, []) if m != account_id]

    # ---- as-of cuts ----------------------------------------------------
    @staticmethod
    def _as64(as_of: datetime) -> np.datetime64:
        return np.datetime64(as_of)

    def _before(self, positions: np.ndarray, as_of: datetime,
                visible: bool = True) -> np.ndarray:
        """Positions with ts < as_of (R1). Optionally observed-only."""
        if positions is None or len(positions) == 0:
            return np.empty(0, dtype=int)
        tvals = self.ts[positions]
        cut = np.searchsorted(tvals, self._as64(as_of), side="left")
        pos = positions[:cut]
        if visible and len(pos):
            pos = pos[self.observed[pos]]
        return pos

    def out_positions(self, account_id: str, as_of: datetime,
                      visible: bool = True) -> np.ndarray:
        return self._before(self.src_pos.get(account_id), as_of, visible)

    def in_positions(self, account_id: str, as_of: datetime,
                     visible: bool = True) -> np.ndarray:
        return self._before(self.dst_pos.get(account_id), as_of, visible)

    def out_edges_between(self, account_id: str, after_ts: datetime,
                          before_ts: datetime, visible: bool = True) -> np.ndarray:
        """Out-edges strictly between after_ts and before_ts (trail expansion)."""
        pos = self.out_positions(account_id, before_ts, visible)
        if len(pos) == 0:
            return pos
        return pos[self.ts[pos] > self._as64(after_ts)]

    def withdrawal_positions(self, account_id: str, as_of: datetime,
                             days: int | None = None,
                             visible: bool = True) -> np.ndarray:
        """Cash-out rows debiting this account before as_of (src == account)."""
        pos = self.out_positions(account_id, as_of, visible)
        if len(pos) == 0:
            return pos
        pos = pos[self.is_cashout[pos]]
        if days is not None and len(pos):
            lo = self._as64(as_of) - np.timedelta64(days, "D")
            pos = pos[self.ts[pos] >= lo]
        return pos

    def withdrawal_atms(self, account_id: str, as_of: datetime,
                        days: int | None = None) -> list[str]:
        pos = self.withdrawal_positions(account_id, as_of, days)
        return [a for a in self.atm_id[pos] if a is not None]

    def withdrawal_centroid(self, account_id: str, as_of: datetime,
                            days: int | None = None):
        atms = self.withdrawal_atms(account_id, as_of, days)
        if not atms:
            return None
        lats = np.array([self._atm_lat[a] for a in atms])
        lons = np.array([self._atm_lon[a] for a in atms])
        return float(lats.mean()), float(lons.mean())

    # ---- ground-truth cash-out events (for panel labels; uses ALL edges) --
    def next_cashout_ts(self, account_id: str, after: datetime,
                        within_min: int) -> bool:
        """TRUE label: does a cash-out debit this account in (after, after+within]?
        Uses all edges incl. unobserved -- this is hindsight ground truth."""
        pos = self.src_pos.get(account_id)
        if pos is None or len(pos) == 0:
            return False
        pos = pos[self.is_cashout[pos]]
        if len(pos) == 0:
            return False
        lo = self._as64(after)
        hi = lo + np.timedelta64(int(within_min), "m")
        t = self.ts[pos]
        return bool(np.any((t > lo) & (t <= hi)))

    # ---- feature-matrix hygiene ---------------------------------------
    @staticmethod
    def feature_columns(df: pd.DataFrame) -> pd.DataFrame:
        return drop_leaky_columns(df)


# ==========================================================================
# Feature functions -- R1: EVERY one has signature f(account_id, as_of, ctx)
# and reads only ts < as_of internally. Graph/case features read ctx._trail /
# ctx._case, bound via ctx.bind(). tests/test_leakage.py checks this suite.
# ==========================================================================
from datetime import timedelta  # noqa: E402

from graphx.features import (GRAPH_FEATURES,  # noqa: E402
                             compute_graph_features)

FRAUD_CATEGORY_CODE = {
    "none": 0, "fast_fanout": 1, "layered": 2, "ring_reuse": 3, "pos_out": 4,
    "split_hold": 5, "cross_district": 6, "no_cashout": 7, "smish_burst": 8,
}


def _win(pos, as_of, minutes, ctx):
    if len(pos) == 0:
        return pos
    lo = np.datetime64(as_of - timedelta(minutes=minutes))
    return pos[ctx.ts[pos] >= lo]


def _amt(pos, ctx):
    return float(ctx.amount[pos].sum()) if len(pos) else 0.0


# ---- velocity -------------------------------------------------------------
def feat_taint_amt(a, as_of, ctx):
    if ctx._trail is None:
        return 0.0
    for n in ctx._trail.nodes:
        if n.account_id == a:
            return float(n.tainted_amount)
    return 0.0


def feat_mins_since_taint(a, as_of, ctx):
    if ctx._trail is None:
        return 9999.0
    arr = [e.ts for e in ctx._trail.edges if e.dst_account == a]
    if not arr:
        return 9999.0
    first = min(arr)
    return max(0.0, (as_of - first).total_seconds() / 60.0)


def _pct_forwarded(a, as_of, ctx, minutes):
    inp = _win(ctx.in_positions(a, as_of), as_of, minutes, ctx)
    outp = _win(ctx.out_positions(a, as_of), as_of, minutes, ctx)
    i, o = _amt(inp, ctx), _amt(outp, ctx)
    return float(min(1.0, o / (i + 1.0)))


def feat_pct_forwarded_5m(a, as_of, ctx): return _pct_forwarded(a, as_of, ctx, 5)
def feat_pct_forwarded_10m(a, as_of, ctx): return _pct_forwarded(a, as_of, ctx, 10)
def feat_pct_forwarded_30m(a, as_of, ctx): return _pct_forwarded(a, as_of, ctx, 30)


def _n_tx(a, as_of, ctx, minutes):
    ip = _win(ctx.in_positions(a, as_of), as_of, minutes, ctx)
    op = _win(ctx.out_positions(a, as_of), as_of, minutes, ctx)
    return float(len(ip) + len(op))


def feat_n_tx_5m(a, as_of, ctx): return _n_tx(a, as_of, ctx, 5)
def feat_n_tx_10m(a, as_of, ctx): return _n_tx(a, as_of, ctx, 10)
def feat_n_tx_30m(a, as_of, ctx): return _n_tx(a, as_of, ctx, 30)


def feat_mins_since_last_tx(a, as_of, ctx):
    pos = np.concatenate([ctx.in_positions(a, as_of), ctx.out_positions(a, as_of)])
    if len(pos) == 0:
        return 9999.0
    last = ctx.ts[pos].max()
    return max(0.0, (np.datetime64(as_of) - last) / np.timedelta64(1, "m"))


def feat_inflow_outflow_ratio(a, as_of, ctx):
    inp = _win(ctx.in_positions(a, as_of), as_of, 1440, ctx)
    outp = _win(ctx.out_positions(a, as_of), as_of, 1440, ctx)
    return float((_amt(inp, ctx) + 1.0) / (_amt(outp, ctx) + 1.0))


def feat_largest_debit_ratio(a, as_of, ctx):
    outp = _win(ctx.out_positions(a, as_of), as_of, 1440, ctx)
    if len(outp) == 0:
        return 0.0
    amts = ctx.amount[outp]
    return float(amts.max() / (amts.sum() + 1.0))


def feat_round_amount_flag(a, as_of, ctx):
    inp = ctx.in_positions(a, as_of)
    if len(inp) == 0:
        return 0.0
    last_amt = float(ctx.amount[inp][-1])
    return 1.0 if (last_amt % 1000 == 0 and last_amt >= 5000) else 0.0


# ---- account --------------------------------------------------------------
def feat_account_age_days(a, as_of, ctx):
    info = ctx.account_info(a)
    if not info:
        return 0.0
    od = pd.Timestamp(info["open_date"]).to_pydatetime()
    return max(0.0, (as_of - od).total_seconds() / 86400.0)


def feat_mean_txn_amt_90d(a, as_of, ctx):
    pos = np.concatenate([ctx.in_positions(a, as_of), ctx.out_positions(a, as_of)])
    pos = _win(pos, as_of, 90 * 1440, ctx)
    return float(ctx.amount[pos].mean()) if len(pos) else 0.0


def feat_amt_zscore_vs_history(a, as_of, ctx):
    pos = np.concatenate([ctx.in_positions(a, as_of), ctx.out_positions(a, as_of)])
    if len(pos) < 3:
        return 0.0
    amts = ctx.amount[pos]
    mu, sd = amts[:-1].mean(), amts[:-1].std()
    if sd < 1e-6:
        return 0.0
    return float((amts[-1] - mu) / sd)


def feat_dormancy_break_days(a, as_of, ctx):
    pos = np.concatenate([ctx.in_positions(a, as_of), ctx.out_positions(a, as_of)])
    if len(pos) < 2:
        return 0.0
    t = np.sort(ctx.ts[pos])
    gap = (t[-1] - t[-2]) / np.timedelta64(1, "D")
    return float(gap)


def feat_n_new_counterparties_24h(a, as_of, ctx):
    op = ctx.out_positions(a, as_of)
    if len(op) == 0:
        return 0.0
    recent = _win(op, as_of, 1440, ctx)
    older = op[~np.isin(op, recent)]
    seen = {d for d in ctx.dst[older] if d is not None}
    new = {d for d in ctx.dst[recent] if d is not None and d not in seen}
    return float(len(new))


# ---- graph (nine) ---------------------------------------------------------
def _graph(a, as_of, ctx):
    g = compute_graph_features(a, as_of, ctx, trail=ctx._trail)
    g["mule_score"] = ctx.mule_scorer.score(g) if ctx.mule_scorer is not None else 0.0
    return g


def feat_hop_from_victim(a, as_of, ctx): return _graph(a, as_of, ctx)["hop_from_victim"]
def feat_in_deg_1h(a, as_of, ctx): return _graph(a, as_of, ctx)["in_deg_1h"]
def feat_out_deg_1h(a, as_of, ctx): return _graph(a, as_of, ctx)["out_deg_1h"]
def feat_dispersion_ratio(a, as_of, ctx): return _graph(a, as_of, ctx)["dispersion_ratio"]
def feat_flow_concentration(a, as_of, ctx): return _graph(a, as_of, ctx)["flow_concentration"]
def feat_ring_size(a, as_of, ctx): return _graph(a, as_of, ctx)["ring_size"]
def feat_ring_prior_cashout_rate(a, as_of, ctx): return _graph(a, as_of, ctx)["ring_prior_cashout_rate"]
def feat_n_tainted_predecessors(a, as_of, ctx): return _graph(a, as_of, ctx)["n_tainted_predecessors"]
def feat_mule_score(a, as_of, ctx): return _graph(a, as_of, ctx)["mule_score"]


# ---- behavioural geo (historical footprint only, R3) ----------------------
def feat_prior_atm_count_90d(a, as_of, ctx):
    return float(len(ctx.withdrawal_positions(a, as_of, days=90)))


def feat_n_distinct_atms_90d(a, as_of, ctx):
    return float(len(set(ctx.withdrawal_atms(a, as_of, days=90))))


def feat_radius_of_gyration_km(a, as_of, ctx):
    atms = ctx.withdrawal_atms(a, as_of, days=90)
    if len(atms) < 2:
        return 0.0
    lats = np.array([ctx._atm_lat[x] for x in atms])
    lons = np.array([ctx._atm_lon[x] for x in atms])
    clat, clon = lats.mean(), lons.mean()
    d = haversine_km(clat, clon, lats, lons)
    return float(np.sqrt((d ** 2).mean()))


def feat_dist_kyc_to_centroid_km(a, as_of, ctx):
    info = ctx.account_info(a)
    cen = ctx.withdrawal_centroid(a, as_of, days=90)
    if not info or cen is None:
        return 0.0
    return float(haversine_km(info["kyc_lat"], info["kyc_lon"], cen[0], cen[1]))


def feat_modal_withdrawal_hour(a, as_of, ctx):
    pos = ctx.withdrawal_positions(a, as_of, days=90)
    if len(pos) == 0:
        return 12.0
    hours = pd.DatetimeIndex(ctx.ts[pos]).hour
    return float(pd.Series(hours).mode().iloc[0])


def feat_hour_sin(a, as_of, ctx): return float(np.sin(2 * np.pi * as_of.hour / 24))
def feat_hour_cos(a, as_of, ctx): return float(np.cos(2 * np.pi * as_of.hour / 24))
def feat_dow(a, as_of, ctx): return float(as_of.weekday())


# ---- case -----------------------------------------------------------------
def feat_mins_complaint_to_now(a, as_of, ctx):
    if ctx._case is None:
        return 0.0
    filed = pd.Timestamp(ctx._case["filed_ts"]).to_pydatetime()
    return (as_of - filed).total_seconds() / 60.0


def feat_seed_amount(a, as_of, ctx):
    if ctx._case is None:
        return 0.0
    return float(ctx._case["disputed_amount"])


def feat_fraud_category(a, as_of, ctx):
    if ctx._case is None:
        return float(FRAUD_CATEGORY_CODE["none"])
    return float(FRAUD_CATEGORY_CODE.get(ctx._case["fraud_category"], 0))


def feat_data_completeness_score(a, as_of, ctx):
    if ctx._trail is None:
        pos = np.concatenate([ctx.in_positions(a, as_of, visible=False),
                              ctx.out_positions(a, as_of, visible=False)])
        pos = _win(pos, as_of, 1440, ctx)
        if len(pos) == 0:
            return 1.0
        return float(ctx.observed[pos].mean())
    if not ctx._trail.edges:
        return 1.0
    return float(np.mean([1.0 if e.observed else 0.0 for e in ctx._trail.edges]))


# ---- registry -------------------------------------------------------------
FEATURE_FUNCS = {
    name[5:]: fn
    for name, fn in list(globals().items())
    if name.startswith("feat_") and callable(fn)
}


def build_feature_vector(account_id: str, as_of, ctx) -> dict:
    """All features for one (account, as_of). Graph features computed once."""
    g = _graph(account_id, as_of, ctx)
    out = {}
    for name, fn in FEATURE_FUNCS.items():
        if name in GRAPH_FEATURES:
            out[name] = g[name]
        else:
            out[name] = fn(account_id, as_of, ctx)
    return out


def feature_names() -> list[str]:
    return sorted(FEATURE_FUNCS.keys())


# ==========================================================================
# M2 channel label + M3 location-pair features
# ==========================================================================
CHANNEL_CLASSES = ["ATM", "POS", "onward_transfer", "dormant"]

LOCATION_FEATURES = [
    "distance_hist_km", "distance_kyc_km", "distance_branch_km", "rank_by_dist",
    "prior_use_count", "ring_use_count", "atm_fraud_rate", "hour_activity",
    "amount_fits_limit", "is_standalone", "operator_match", "district_hotspot_prior",
]

# loose map from account bank_code to the ATM operator label it most resembles
_BANK_TO_OP = {"SBIN": "SBI", "HDFC": "HDFC", "PUNB": "PUNB", "BARB": "BARB",
               "UTIB": "AXIS", "ICIC": "HDFC", "CNRB": "SBI", "IOBA": "SBI"}


def next_action_class(ctx: FeatureContext, account: str, as_of, window_min: int = 60) -> str:
    """M2 ground-truth label: the account's next action within the window.
    Uses ALL edges (hindsight). ATM/POS = cash-out channel; a transfer =
    onward_transfer; nothing = dormant."""
    pos = ctx.src_pos.get(account)
    if pos is None or len(pos) == 0:
        return "dormant"
    lo = np.datetime64(as_of)
    hi = lo + np.timedelta64(int(window_min), "m")
    t = ctx.ts[pos]
    m = (t > lo) & (t <= hi)
    pos = pos[m]
    if len(pos) == 0:
        return "dormant"
    first = pos[np.argmin(ctx.ts[pos])]
    if ctx.is_cashout[first]:
        return "ATM" if ctx.channel[first] == "ATM_WDL" else "POS"
    return "onward_transfer"


def build_location_matrix(account: str, as_of, candidates: list[str], ctx: FeatureContext,
                          amount: float = 0.0) -> pd.DataFrame:
    """One row per candidate terminal, with the M3 ranking features (all as-of)."""
    if not candidates:
        return pd.DataFrame(columns=LOCATION_FEATURES + ["atm_id"])
    info = ctx.account_info(account)
    kyc = (info.get("kyc_lat", 0.0), info.get("kyc_lon", 0.0)) if info else (0.0, 0.0)
    if account in ctx._acc_branch.index:
        br = ctx._acc_branch.loc[account]
        branch = (float(br["branch_lat"]), float(br["branch_lon"]))
    else:
        branch = kyc
    cen = ctx.withdrawal_centroid(account, as_of, days=90) or kyc

    # own decayed prior-use counts + modal operator
    wpos = ctx.withdrawal_positions(account, as_of, days=180)
    own_counts: dict[str, float] = {}
    op_counts: dict[str, int] = {}
    now = np.datetime64(as_of)
    for p in wpos:
        a = ctx.atm_id[p]
        if a is None:
            continue
        age_days = (now - ctx.ts[p]) / np.timedelta64(1, "D")
        own_counts[a] = own_counts.get(a, 0.0) + float(np.exp(-age_days / 30.0))
        op = ctx._atm_operator.get(a)
        op_counts[op] = op_counts.get(op, 0) + 1
    modal_op = max(op_counts, key=op_counts.get) if op_counts else None
    bank_op = _BANK_TO_OP.get(info.get("bank_code")) if info else None

    # ring member usage counts
    ring_counts: dict[str, int] = {}
    for m in ctx.ring_members(account):
        for a in ctx.withdrawal_atms(m, as_of, days=180):
            ring_counts[a] = ring_counts.get(a, 0) + 1

    rows = []
    dist_kyc_all = []
    for a in candidates:
        ai = ctx.atm_info(a)
        d_kyc = float(haversine_km(kyc[0], kyc[1], ai["lat"], ai["lon"]))
        dist_kyc_all.append(d_kyc)
        rows.append({
            "atm_id": a,
            "distance_hist_km": float(haversine_km(cen[0], cen[1], ai["lat"], ai["lon"])),
            "distance_kyc_km": d_kyc,
            "distance_branch_km": float(haversine_km(branch[0], branch[1], ai["lat"], ai["lon"])),
            "prior_use_count": own_counts.get(a, 0.0),
            "ring_use_count": float(ring_counts.get(a, 0)),
            "atm_fraud_rate": ctx.atm_fraud_rate(a),
            "hour_activity": float(ctx._atm_hourly[a][as_of.hour]),
            "amount_fits_limit": 1.0 if amount <= ai["per_txn_limit"] else 0.0,
            "is_standalone": 1.0 if ctx._atm_standalone.get(a) else 0.0,
            "operator_match": 1.0 if (modal_op == ctx._atm_operator.get(a)
                                      or bank_op == ctx._atm_operator.get(a)) else 0.0,
            "district_hotspot_prior": ctx.district_hotspot_prior(ai["district"]),
        })
    df = pd.DataFrame(rows)
    order = np.argsort(np.argsort(dist_kyc_all))  # 0 = nearest
    df["rank_by_dist"] = order.astype(float)
    return df
