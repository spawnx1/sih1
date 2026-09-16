"""
ml/predict.py -- inference: M1 hazard curve, M2 channel, M3 hierarchical
location WITH abstention (R5).

The location head never names a single ATM on a flat distribution. It aggregates
terminal probabilities up to cells and districts and emits the finest level it
can defend:

    if max(p_terminal) >= 0.22 and entropy <= 2.60:  emit TERMINAL (top 5)
    elif max(p_cell) >= 0.30:                         emit CELL, abstain@terminal
    else:                                             emit DISTRICT only

"The system knows what it doesn't know" is the point -- abstaining is designed
behaviour, surfaced via LocationPrediction.abstained_at / abstain_reason.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

import config
from contracts import (CashoutPrediction, CellPrediction, ChannelPrediction,
                       DistrictPrediction, HazardPoint, LocationCandidate,
                       LocationPrediction)
from graphx.candidates import generate_candidates_detailed
from ml.features import (CHANNEL_CLASSES, LOCATION_FEATURES, FeatureContext,
                         build_feature_vector, build_location_matrix)


def load_models(model_dir: str | None = None) -> dict:
    d = model_dir or config.MODEL_DIR
    return {
        "m1": joblib.load(f"{d}/m1.joblib"),
        "m2": joblib.load(f"{d}/m2.joblib"),
        "m3": joblib.load(f"{d}/m3.joblib"),
        "m1_features": joblib.load(f"{d}/m1_features.joblib"),
        "m2_features": joblib.load(f"{d}/m2_features.joblib"),
        "mule_scorer": joblib.load(f"{d}/mule_scorer.joblib"),
    }


def _softmax_norm(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, None)
    return p / p.sum()


# --------------------------------------------------------------------------
# M1 -- hazard curve (probability + timing from one model)
# --------------------------------------------------------------------------
def predict_cashout(account: str, as_of, ctx: FeatureContext, models: dict,
                    fv: dict | None = None) -> CashoutPrediction:
    fv = fv or build_feature_vector(account, as_of, ctx)
    feats = models["m1_features"]
    base = {k: fv.get(k, 0.0) for k in feats if k != "horizon_min"}
    X = pd.DataFrame([{**base, "horizon_min": float(h)} for h in config.HORIZONS])[feats]
    p = models["m1"].predict_proba(X)[:, 1]
    pc = np.maximum.accumulate(p)  # cumulative hazard is non-decreasing

    hazard = [HazardPoint(horizon_min=h, p_cumulative=round(float(pc[i]), 4))
              for i, h in enumerate(config.HORIZONS)]
    # steepest rise in cumulative probability -> the action window
    prev_p, prev_h, best = 0.0, 0, (0, config.HORIZONS[0], -1.0)
    for i, h in enumerate(config.HORIZONS):
        slope = (pc[i] - prev_p) / max(1, (h - prev_h))
        if slope > best[2]:
            best = (prev_h, h, slope)
        prev_p, prev_h = pc[i], h
    window = (int(best[0]), int(best[1]))

    p10 = float(pc[0])
    severity = ("imminent" if p10 >= 0.60 else "soon" if p10 >= 0.30
                else "later" if p10 >= 0.10 else "unlikely")
    return CashoutPrediction(
        account_id=account,
        p_10m=round(float(pc[0]), 4), p_20m=round(float(pc[1]), 4),
        p_30m=round(float(pc[2]), 4), p_45m=round(float(pc[3]), 4),
        p_60m=round(float(pc[4]), 4),
        window_minutes=window, hazard=hazard, severity=severity)


# --------------------------------------------------------------------------
# M2 -- channel
# --------------------------------------------------------------------------
def predict_channel(account: str, as_of, ctx: FeatureContext, models: dict,
                    fv: dict | None = None) -> ChannelPrediction:
    fv = fv or build_feature_vector(account, as_of, ctx)
    X = pd.DataFrame([fv])[models["m2_features"]]
    probs = models["m2"].predict_proba(X)[0]
    d = {c: float(probs[i]) for i, c in enumerate(CHANNEL_CLASSES)}
    return ChannelPrediction(ATM=round(d["ATM"], 4), POS=round(d["POS"], 4),
                             onward_transfer=round(d["onward_transfer"], 4),
                             dormant=round(d["dormant"], 4))


# --------------------------------------------------------------------------
# M3 -- hierarchical location with abstention
# --------------------------------------------------------------------------
def predict_location(account: str, as_of, ctx: FeatureContext, models: dict,
                     amount: float = 0.0) -> LocationPrediction:
    cands, sources, _ = generate_candidates_detailed(account, as_of, ctx)
    info = ctx.account_info(account)
    fallback_d = info.get("kyc_district") if info else list(config.DISTRICTS)[0]

    if not cands:
        dc = config.DISTRICT_CENTROIDS[fallback_d]
        return LocationPrediction(
            account_id=account, granularity="district", abstained_at="cell",
            abstain_reason="no candidate terminals from history/KYC/ring",
            districts=[DistrictPrediction(district=fallback_d, lat=dc[0], lon=dc[1], p=1.0)],
            cells=[], terminals=[], entropy=0.0)

    M = build_location_matrix(account, as_of, cands, ctx, amount)
    p_term = _softmax_norm(models["m3"].predict_proba(M[LOCATION_FEATURES])[:, 1])
    H = float(-np.sum(p_term * np.log(p_term + 1e-12)))

    # per-terminal
    terminals = []
    for i, a in enumerate(M["atm_id"]):
        ai = ctx.atm_info(a)
        terminals.append(LocationCandidate(
            atm_id=a, lat=ai["lat"], lon=ai["lon"], district=ai["district"],
            cell=ai["cell"], cell_fine=ai["cell_fine"], p=round(float(p_term[i]), 4),
            sources=sources.get(a, []),
            distance_km=round(float(M["distance_kyc_km"].iloc[i]), 3)))
    terminals.sort(key=lambda t: t.p, reverse=True)

    # aggregate up
    cell_p, cell_meta, dist_p = {}, {}, {}
    for t in terminals:
        cell_p[t.cell] = cell_p.get(t.cell, 0.0) + t.p
        cell_meta.setdefault(t.cell, (t.lat, t.lon, t.district))
        dist_p[t.district] = dist_p.get(t.district, 0.0) + t.p
    cells = [CellPrediction(cell=c, lat=cell_meta[c][0], lon=cell_meta[c][1],
                            district=cell_meta[c][2], p=round(v, 4),
                            n_terminals=sum(1 for t in terminals if t.cell == c))
             for c, v in sorted(cell_p.items(), key=lambda kv: kv[1], reverse=True)]
    districts = [DistrictPrediction(district=d,
                                    lat=config.DISTRICT_CENTROIDS[d][0],
                                    lon=config.DISTRICT_CENTROIDS[d][1], p=round(v, 4))
                 for d, v in sorted(dist_p.items(), key=lambda kv: kv[1], reverse=True)]

    max_pt = terminals[0].p
    max_pc = cells[0].p if cells else 0.0

    if max_pt >= config.ABSTAIN_MIN_P_TERMINAL and H <= config.ABSTAIN_MAX_ENTROPY:
        granularity, abstained_at, reason = "terminal", None, None
    elif max_pc >= config.CELL_MIN_P:
        granularity, abstained_at = "cell", "terminal"
        reason = (f"top terminal p={max_pt:.2f} (<{config.ABSTAIN_MIN_P_TERMINAL}) "
                  f"or entropy={H:.2f} (>{config.ABSTAIN_MAX_ENTROPY}); "
                  f"confident only to the ~5 km cell")
    else:
        granularity, abstained_at = "district", "cell"
        reason = (f"top cell p={max_pc:.2f} (<{config.CELL_MIN_P}); "
                  f"distribution too flat to name a cell")

    return LocationPrediction(
        account_id=account, granularity=granularity, abstained_at=abstained_at,
        abstain_reason=reason, districts=districts, cells=cells[:8],
        terminals=terminals[:5], entropy=round(H, 3))


# --------------------------------------------------------------------------
# quick abstention-coverage demo (Stage 5)
# --------------------------------------------------------------------------
def _coverage_demo():
    ctx = FeatureContext(config.DATA_DIR)
    ctx.mule_scorer = joblib.load(f"{config.MODEL_DIR}/mule_scorer.joblib")
    models = load_models()
    co = np.where(ctx.is_cashout & ctx._label_tainted)[0]
    rng = np.random.default_rng(config.SEED)
    sample = rng.choice(co, size=min(400, len(co)), replace=False)
    levels = {"terminal": 0, "cell": 0, "district": 0}
    correct = {"terminal": 0, "cell": 0}
    for p in sample:
        acc = ctx.src[p]
        if acc is None or pd.isna(acc):
            continue
        as_of = pd.Timestamp(ctx.ts[p]).to_pydatetime() - pd.Timedelta(minutes=5).to_pytimedelta()
        loc = predict_location(str(acc), as_of, ctx, models, float(ctx.amount[p]))
        levels[loc.granularity] += 1
        true_term = str(ctx.atm_id[p])
        if loc.granularity == "terminal":
            correct["terminal"] += int(true_term in [t.atm_id for t in loc.terminals[:1]])
        if loc.granularity in ("terminal", "cell") and loc.cells:
            correct["cell"] += int(ctx._atm_cell[true_term] == loc.cells[0].cell)
    n = sum(levels.values())
    print("=== Abstention coverage over", n, "fraud cash-outs ===")
    print(f"  emitted TERMINAL: {levels['terminal']} ({levels['terminal']/n:.0%})  "
          f"top-1 correct when emitted: {correct['terminal']}/{levels['terminal']}")
    print(f"  abstained to CELL: {levels['cell']} ({levels['cell']/n:.0%})")
    print(f"  abstained to DISTRICT: {levels['district']} ({levels['district']/n:.0%})")


if __name__ == "__main__":
    _coverage_demo()
