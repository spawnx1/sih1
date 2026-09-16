"""
ml/panel.py -- training row construction for the discrete-time hazard.

Design principle that keeps the metrics honest: EVERY row is the same question --
"an account received money a few minutes ago; will a cash-out debit it within
the horizon?" -- so no feature can separate the label just by being present.

  * trail rows  -- accounts inside a complaint's money trail, sampled at several
                   offsets after they received the tainted funds. Small offsets
                   land before the cash-out (POSITIVE); large offsets land after
                   it (NEGATIVE). Onward-transfer forwarders and no_cashout
                   accounts are natural hard negatives.
  * legit rows  -- legitimate accounts given a SELF-TRAIL from their own largest
                   recent inflow (a pseudo-seed). They get the same trail-derived
                   features (taint_amt, hop, ...) computed the same way, at
                   matched times of day. Labels are whatever is true (a salary
                   sweep that withdraws soon is a genuine positive).

Each base row is STACKED across horizons [10,20,30,45,60]. Labels use ALL edges
(hindsight ground truth); features use only observed edges with ts < as_of.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from graphx.trail import expand_trail
from ml.features import FeatureContext, build_feature_vector

TRAIL_OFFSETS_MIN = (2, 6, 12, 25, 45)
LEGIT_OFFSETS_MIN = (2, 6, 12, 25)


def _cashout_within(ctx, account, after, within_min, tainted_only=False):
    pos = ctx.src_pos.get(account)
    if pos is None or len(pos) == 0:
        return False
    pos = pos[ctx.is_cashout[pos]]
    if tainted_only:
        pos = pos[ctx._label_tainted[pos]]
    if len(pos) == 0:
        return False
    lo = np.datetime64(after)
    hi = lo + np.timedelta64(int(within_min), "m")
    t = ctx.ts[pos]
    return bool(np.any((t > lo) & (t <= hi)))


def _emit(base_rows, ctx, account, as_of, seed_txn, case, is_trail, group_id):
    trail = expand_trail(seed_txn, as_of, ctx) if seed_txn is not None else None
    ctx.bind(trail=trail, case=case)
    feats = build_feature_vector(account, as_of, ctx)
    base_rows.append({
        "account_id": account, "as_of": as_of, "is_trail": int(is_trail),
        "group_id": group_id, "_feats": feats,
    })


def build_panel(ctx: FeatureContext, seed: int = config.SEED,
                n_legit_base: int = 1600, verbose: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    complaints = pd.read_parquet(f"{ctx.data_dir}/complaints.parquet")
    base_rows: list[dict] = []

    # ---- trail rows -----------------------------------------------------
    for gi, (_, case) in enumerate(complaints.iterrows()):
        case_id = case["seed_txn_id"]
        if case_id not in ctx.txn_pos:
            continue
        seed_ts = pd.Timestamp(ctx.ts[ctx.txn_pos[case_id]]).to_pydatetime()
        case_pos = np.where(ctx._label_case == case_id)[0]
        if len(case_pos) == 0:
            continue

        # every account that RECEIVED tainted funds, with its arrival time
        arrivals: dict[str, object] = {ctx.dst[ctx.txn_pos[case_id]]: seed_ts}
        for p in case_pos:
            if not ctx.is_cashout[p] and ctx.dst[p] is not None:
                t = pd.Timestamp(ctx.ts[p]).to_pydatetime()
                a = ctx.dst[p]
                if a not in arrivals or t < arrivals[a]:
                    arrivals[a] = t

        for acc, arr in arrivals.items():
            if acc is None:
                continue
            for off in TRAIL_OFFSETS_MIN:
                as_of = arr + pd.Timedelta(minutes=off).to_pytimedelta()
                _emit(base_rows, ctx, acc, as_of, case_id, case, True, gi)

    if verbose:
        print(f"[panel] trail base rows: {len(base_rows)}")

    # ---- legit self-trail rows (matched times) --------------------------
    legit_accts = ctx.accounts.loc[~ctx.accounts["label_is_mule"], "account_id"].to_numpy()
    n_added, attempts, gi = 0, 0, 10_000_000
    while n_added < n_legit_base and attempts < n_legit_base * 8:
        attempts += 1
        acc = str(rng.choice(legit_accts))
        inp = ctx.dst_pos.get(acc)
        if inp is None or len(inp) < 2:
            continue
        # exclude cash deposits (null payer) -- the pseudo-seed needs a real src
        inp = inp[np.array([not pd.isna(ctx.src[i]) for i in inp], dtype=bool)]
        if len(inp) < 2:
            continue
        # pick a sizeable inflow as the pseudo-seed
        amts = ctx.amount[inp]
        big = inp[amts >= np.quantile(amts, 0.6)]
        p = int(rng.choice(big if len(big) else inp))
        pseudo_seed = str(ctx.txn_id[p])
        arr = pd.Timestamp(ctx.ts[p]).to_pydatetime()
        for off in LEGIT_OFFSETS_MIN:
            as_of = arr + pd.Timedelta(minutes=off).to_pytimedelta()
            _emit(base_rows, ctx, acc, as_of, pseudo_seed, None, False, gi)
        gi += 1
        n_added += 1

    if verbose:
        print(f"[panel] + legit self-trail base rows: {n_added} "
              f"(total {len(base_rows)})")

    # ---- stack across horizons -----------------------------------------
    rows: list[dict] = []
    for br in base_rows:
        acc, as_of = br["account_id"], br["as_of"]
        for h in config.HORIZONS:
            y = _cashout_within(ctx, acc, as_of, h)
            y_tainted = _cashout_within(ctx, acc, as_of, h, tainted_only=True)
            rows.append({
                **br["_feats"], "horizon_min": float(h),
                "y": int(y), "label_is_tainted": int(y_tainted),
                "account_id": acc, "as_of": as_of,
                "is_trail": br["is_trail"], "group_id": br["group_id"],
            })
    df = pd.DataFrame(rows)
    if verbose:
        print(f"[panel] stacked rows: {len(df)}  positives: {df['y'].mean():.3f}  "
              f"(trail rows: {df['is_trail'].mean():.3f})")
    return df


if __name__ == "__main__":
    ctx = FeatureContext(config.DATA_DIR)
    from graphx.features import MuleScorer
    ctx.mule_scorer = MuleScorer().fit(ctx)
    df = build_panel(ctx, n_legit_base=800)
    print(df.groupby("horizon_min")["y"].mean())
    print("trail positives:", df[df.is_trail == 1]["y"].mean(),
          " legit positives:", df[df.is_trail == 0]["y"].mean())
