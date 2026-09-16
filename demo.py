"""
demo.py -- headless end-to-end demo (used by `make demo`).

Regenerates nothing. Loads the trained models, replays ONE scripted complaint
minute-by-minute from the seed transaction, and prints the timeline to stdout:
the money moving, the hazard rising, the forecast cell, the freeze
recommendation, and finally predicted-vs-actual with a lead time in minutes.

This is the fallback that makes the demo work even if the browser dies on stage.

Run:  python demo.py            (or: make demo)
"""
from __future__ import annotations

from datetime import timedelta

import joblib
import numpy as np
import pandas as pd

import config
from graphx.trail import expand_trail
from ml.features import FeatureContext
from ml.predict import (load_models, predict_cashout, predict_channel,
                        predict_location)

BAR = "=" * 72


def pick_case(ctx):
    """A vivid case: fast reporting + a tainted cash-out that is still in the
    future at filing (so there is a lead time to show)."""
    complaints = pd.read_parquet(f"{config.DATA_DIR}/complaints.parquet")
    best = None
    for _, c in complaints.iterrows():
        sid = c["seed_txn_id"]
        if sid not in ctx.txn_pos:
            continue
        cpos = np.where((ctx._label_case == sid) & ctx.is_cashout & ctx._label_tainted)[0]
        if len(cpos) == 0:
            continue
        filed = pd.Timestamp(c["filed_ts"]).to_pydatetime()
        future = cpos[ctx.ts[cpos] > np.datetime64(filed)]
        if len(future) and c["fraud_category"] in ("fast_fanout", "ring_reuse", "layered"):
            score = c["disputed_amount"]
            if best is None or score > best[1]:
                best = (c, score)
    return best[0] if best else complaints.iloc[0]


def main():
    print(BAR)
    print("SIH26184  Cash-out Forecast  --  headless demo")
    print(BAR)
    ctx = FeatureContext(config.DATA_DIR)
    ctx.mule_scorer = joblib.load(f"{config.MODEL_DIR}/mule_scorer.joblib")
    models = load_models()

    case = pick_case(ctx)
    sid = case["seed_txn_id"]
    seed_ts = pd.Timestamp(ctx.ts[ctx.txn_pos[sid]]).to_pydatetime()
    filed = pd.Timestamp(case["filed_ts"]).to_pydatetime()
    print(f"\nComplaint {case['ack_no']}  ({case['fraud_category']})")
    print(f"  victim {case['victim_account']}  disputed {case['disputed_amount']:,.0f}")
    print(f"  seed txn {sid} at {seed_ts}")
    print(f"  complaint filed at {filed}  "
          f"(reporting delay {case['reporting_delay_min']:.0f} min)")

    # actual tainted cash-outs of this case
    cpos = np.where((ctx._label_case == sid) & ctx.is_cashout & ctx._label_tainted)[0]
    actual = [(pd.Timestamp(ctx.ts[p]).to_pydatetime(), str(ctx.src[p]), str(ctx.atm_id[p]))
              for p in cpos]
    actual.sort()
    print(f"\n  ACTUAL cash-outs in this case ({len(actual)}):")
    for t, m, a in actual:
        ai = ctx.atm_info(a)
        print(f"    {t.strftime('%H:%M:%S')}  {m}  ->  {a}  "
              f"(cell {ai['cell'][:8]}.. {ai['district']})")

    print(f"\n{BAR}\n  TIMELINE  (replaying from the seed transaction)\n{BAR}")

    def most_imminent(trail, as_of):
        ctx.bind(trail=trail, case=case)
        received = [n.account_id for n in trail.nodes
                    if not n.is_victim and n.tainted_amount > 0]
        best = None
        for a in received:
            cp = predict_cashout(a, as_of, ctx, models)
            if best is None or cp.p_10m > best[1].p_10m:
                best = (a, cp)
        return best

    first_flag = None
    for t in (1, 3, 5, 8, 12, 20, 30, 45):
        as_of = seed_ts + timedelta(minutes=t)
        trail = expand_trail(sid, as_of, ctx)
        mi = most_imminent(trail, as_of)
        if mi is None:
            continue
        acc, cp = mi
        loc = predict_location(acc, as_of, ctx, models,
                               amount=next((n.tainted_amount for n in trail.nodes
                                            if n.account_id == acc), 0.0))
        tier = ("RED" if cp.p_10m >= config.TIER_RED_P else
                "AMBER" if cp.p_10m >= config.TIER_AMBER_P else "GREY")
        if tier in ("RED", "AMBER") and first_flag is None:
            first_flag = (as_of, acc)
        filed_mark = "  <-- complaint filed" if abs((as_of - filed).total_seconds()) < 90 else ""
        gran = loc.granularity
        where = (loc.terminals[0].atm_id if gran == "terminal" and loc.terminals
                 else (loc.cells[0].cell[:8] + ".." if gran == "cell" and loc.cells
                       else (loc.districts[0].district if loc.districts else "?")))
        print(f"  t+{t:>2}m {as_of.strftime('%H:%M:%S')}  acct {acc}  "
              f"P<=10m={cp.p_10m:.2f}  [{tier:5s}]  forecast {gran}:{where}{filed_mark}")

    # ---- predicted vs actual + lead time ----
    print(f"\n{BAR}\n  PREDICTED vs ACTUAL\n{BAR}")
    # choose a target that actually cashes out; forecast just before it does
    if first_flag and any(m == first_flag[1] for (_, m, _) in actual):
        acc = first_flag[1]
        t_co = min(t for (t, m, _) in actual if m == acc)
    else:
        t_co, acc, _a = actual[0]
    ref = max(seed_ts + timedelta(minutes=1), t_co - timedelta(minutes=5))
    trail = expand_trail(sid, ref, ctx)
    ctx.bind(trail=trail, case=case)
    loc = predict_location(acc, ref, ctx, models)
    pred_cells = [c.cell for c in loc.cells[:3]]
    actual_for_acc = [(t, a) for (t, m, a) in actual if m == acc]
    if actual_for_acc:
        t_co, a_true = actual_for_acc[0]
        true_cell = ctx.atm_info(a_true)["cell"]
        hit = "YES" if true_cell in pred_cells else "no"
        print(f"  target account : {acc}")
        print(f"  forecast level : {loc.granularity} (abstain@{loc.abstained_at})")
        print(f"  forecast cells : {[c[:8]+'..' for c in pred_cells]}")
        print(f"  actual terminal: {a_true}  (cell {true_cell[:8]}..)")
        print(f"  true cell in top-3 forecast? {hit}")
        lead = (t_co - ref).total_seconds() / 60.0
        print(f"\n  first flagged at : {ref.strftime('%H:%M:%S')} "
              f"({'AMBER/RED' if first_flag else 'filing'})")
        print(f"  actual cash-out  : {t_co.strftime('%H:%M:%S')}")
        print(f"  LEAD TIME        : {lead:+.1f} minutes "
              f"({'ahead of the debit' if lead > 0 else 'too late -- money already gone'})")
    else:
        print(f"  (target {acc} did not itself cash out in this case)")
    print(BAR)


if __name__ == "__main__":
    main()
