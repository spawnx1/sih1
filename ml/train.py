"""
ml/train.py -- M1 (hazard), M2 (channel), M3 (location ranker).

M1  one XGBoost binary classifier, `horizon_min` stacked as a feature -> a
    hazard curve at inference (probability AND timing from one model).
M2  XGBoost multi:softprob over {ATM, POS, onward_transfer, dormant} -- lets the
    system say "this money is going to keep moving, don't send anyone yet."
M3  the SIMPLE location ranker: a binary XGBClassifier over (event, candidate)
    pairs, softmax-normalised within each event group at inference. It calibrates
    naturally and is easy to debug; rank:pairwise is left as a later option.

Run:  python -m ml.train   (Stage 4 + Stage 5 acceptance, then saves all models)
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from xgboost import XGBClassifier

import config
from graphx.candidates import generate_candidates
from graphx.features import MuleScorer
from ml.features import (CHANNEL_CLASSES, LOCATION_FEATURES, FeatureContext,
                         build_location_matrix, feature_names, haversine_km,
                         next_action_class)
from ml.panel import build_panel

M1_EXCLUDE = {"seed_amount", "fraud_category", "mins_complaint_to_now"}
M1_FEATURES = [f for f in feature_names() if f not in M1_EXCLUDE] + ["horizon_min"]
M2_FEATURES = [f for f in feature_names() if f not in M1_EXCLUDE]


def load_ctx(data_dir: str) -> FeatureContext:
    ctx = FeatureContext(data_dir)
    ctx.mule_scorer = MuleScorer().fit(ctx)
    return ctx


def chronological_split(df, q=0.70, by="as_of", account_col="account_id"):
    cut = df[by].quantile(q)
    train = df[df[by] < cut]
    seen = set(train[account_col]) if account_col in df else set()
    test = df[(df[by] >= cut) & (~df[account_col].isin(seen))] if account_col in df \
        else df[df[by] >= cut]
    return train, test, cut


def _xgb_binary():
    return XGBClassifier(
        n_estimators=350, max_depth=5, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.85, min_child_weight=5.0, reg_lambda=2.0,
        eval_metric="logloss", tree_method="hist", random_state=config.SEED, n_jobs=0)


def single_feature_audit(test, features):
    y = test["y"].to_numpy()
    rows = []
    for f in features:
        x = test[f].to_numpy(dtype=float)
        if np.unique(x).size < 2:
            continue
        try:
            auc = roc_auc_score(y, x)
        except ValueError:
            continue
        rows.append({"feature": f, "univariate_auc": max(auc, 1 - auc)})
    return pd.DataFrame(rows).sort_values("univariate_auc", ascending=False)


# --------------------------------------------------------------------------
# M3 event frames
# --------------------------------------------------------------------------
def build_m3_events(ctx, seed=config.SEED, n_legit=3000):
    rng = np.random.default_rng(seed)
    co = np.where(ctx.is_cashout)[0]
    tainted = co[ctx._label_tainted[co]]
    legit = co[~ctx._label_tainted[co]]
    legit = rng.choice(legit, size=min(n_legit, len(legit)), replace=False)
    events = []
    for p in np.concatenate([tainted, legit]):
        acc = ctx.src[p]
        if acc is None or pd.isna(acc):
            continue
        t_co = pd.Timestamp(ctx.ts[p]).to_pydatetime()
        as_of = t_co - pd.Timedelta(minutes=float(rng.uniform(2, 15))).to_pytimedelta()
        events.append({
            "account": str(acc), "as_of": as_of, "terminal": str(ctx.atm_id[p]),
            "amount": float(ctx.amount[p]), "is_tainted": bool(ctx._label_tainted[p]),
            "has_hist": len(ctx.withdrawal_atms(str(acc), as_of, days=90)) > 0,
        })
    return events


def build_m3_frame(ctx, events):
    frames = []
    for gi, ev in enumerate(events):
        cands = generate_candidates(ev["account"], ev["as_of"], ctx)
        if ev["terminal"] not in cands:
            continue  # no positive in group -> not useful for training/ranking
        m = build_location_matrix(ev["account"], ev["as_of"], cands, ctx, ev["amount"])
        m["label"] = (m["atm_id"] == ev["terminal"]).astype(int)
        m["group"] = gi
        m["as_of"] = ev["as_of"]
        m["true_term"] = ev["terminal"]
        m["has_hist"] = ev["has_hist"]
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


def _softmax_norm(p):
    p = np.clip(p, 1e-9, None)
    return p / p.sum()


def location_metrics(ctx, model, test_frame):
    """Top-k / cell / district / haversine, stratified by prior ATM history."""
    def blank():
        return {"n": 0, "t1": 0, "t3": 0, "t5": 0, "c1": 0, "c3": 0, "d1": 0,
                "km": []}
    agg = {"all": blank(), "hist": blank(), "nohist": blank()}

    for gid, g in test_frame.groupby("group"):
        p = model.predict_proba(g[LOCATION_FEATURES])[:, 1]
        p = _softmax_norm(p)
        atms = g["atm_id"].to_numpy()
        true_term = g["true_term"].iloc[0]
        order = np.argsort(p)[::-1]
        ranked = atms[order]
        # cell / district aggregation
        cells = np.array([ctx._atm_cell[a] for a in atms])
        dists = np.array([ctx._atm_district[a] for a in atms])
        pc = pd.Series(p).groupby(cells).sum().sort_values(ascending=False)
        pd_ = pd.Series(p).groupby(dists).sum().sort_values(ascending=False)
        true_cell = ctx._atm_cell[true_term]
        true_dist = ctx._atm_district[true_term]
        # haversine of top-1 terminal vs true
        top = ranked[0]
        km = float(haversine_km(ctx._atm_lat[top], ctx._atm_lon[top],
                                ctx._atm_lat[true_term], ctx._atm_lon[true_term]))
        buckets = ["all", "hist" if g["has_hist"].iloc[0] else "nohist"]
        for b in buckets:
            s = agg[b]
            s["n"] += 1
            s["t1"] += int(true_term in ranked[:1])
            s["t3"] += int(true_term in ranked[:3])
            s["t5"] += int(true_term in ranked[:5])
            s["c1"] += int(true_cell in list(pc.index[:1]))
            s["c3"] += int(true_cell in list(pc.index[:3]))
            s["d1"] += int(true_dist in list(pd_.index[:1]))
            s["km"].append(km)
    return agg


def _print_loc(agg):
    print(f"{'stratum':8s} {'n':>5s} {'T@1':>6s} {'T@3':>6s} {'T@5':>6s} "
          f"{'C@1':>6s} {'C@3':>6s} {'D@1':>6s} {'medkm':>7s}")
    for b in ("all", "hist", "nohist"):
        s = agg[b]
        n = max(1, s["n"])
        med = np.median(s["km"]) if s["km"] else float("nan")
        print(f"{b:8s} {s['n']:5d} {s['t1']/n:6.3f} {s['t3']/n:6.3f} {s['t5']/n:6.3f} "
              f"{s['c1']/n:6.3f} {s['c3']/n:6.3f} {s['d1']/n:6.3f} {med:7.2f}")


def main():
    ctx = load_ctx(config.DATA_DIR)

    # ==================== M1 ====================
    print("=== M1: discrete-time cash-out hazard ===")
    df = build_panel(ctx, n_legit_base=1800)
    tr, te, cut = chronological_split(df)
    print(f"[split] cut={pd.Timestamp(cut)}  train={len(tr)} test={len(te)} "
          f"(test positives={te['y'].mean():.3f})")
    m1 = _xgb_binary().fit(tr[M1_FEATURES], tr["y"])
    p = m1.predict_proba(te[M1_FEATURES])[:, 1]
    y = te["y"].to_numpy()
    print(f"[M1] ROC-AUC={roc_auc_score(y,p):.3f}  PR-AUC={average_precision_score(y,p):.3f}  "
          f"Brier={brier_score_loss(y,p):.3f}")
    if roc_auc_score(y, p) > 0.95:
        print("!! AUC > 0.95 -- leak. Widen the generator (R4).")
    audit = single_feature_audit(te, M1_FEATURES)
    top = audit.head(3).to_dict("records")
    print("   single-feature audit top-3:",
          ", ".join(f"{r['feature']}={r['univariate_auc']:.2f}" for r in top),
          "(>0.85 = artefact)")
    m1_full = _xgb_binary().fit(df[M1_FEATURES], df["y"])  # final model on all data

    # ==================== M2 ====================
    print("\n=== M2: channel classifier {ATM, POS, onward_transfer, dormant} ===")
    base = df[df["horizon_min"] == config.HORIZONS[0]].copy()
    base["chan"] = [next_action_class(ctx, a, t) for a, t in
                    zip(base["account_id"], base["as_of"])]
    cls_idx = {c: i for i, c in enumerate(CHANNEL_CLASSES)}
    base["chan_y"] = base["chan"].map(cls_idx)
    btr, bte, _ = chronological_split(base)
    m2 = XGBClassifier(objective="multi:softprob", num_class=4, n_estimators=300,
                       max_depth=5, learning_rate=0.05, subsample=0.85,
                       colsample_bytree=0.85, eval_metric="mlogloss",
                       tree_method="hist", random_state=config.SEED, n_jobs=0)
    m2.fit(btr[M2_FEATURES], btr["chan_y"])
    acc = (m2.predict(bte[M2_FEATURES]) == bte["chan_y"].to_numpy()).mean()
    print(f"[M2] test accuracy={acc:.3f}  class balance="
          + ", ".join(f"{c}:{(base['chan']==c).mean():.2f}" for c in CHANNEL_CLASSES))
    m2_full = XGBClassifier(objective="multi:softprob", num_class=4, n_estimators=300,
                            max_depth=5, learning_rate=0.05, subsample=0.85,
                            colsample_bytree=0.85, eval_metric="mlogloss",
                            tree_method="hist", random_state=config.SEED, n_jobs=0)
    m2_full.fit(base[M2_FEATURES], base["chan_y"])

    # ==================== M3 ====================
    print("\n=== M3: location ranker (simple binary + softmax within group) ===")
    events = build_m3_events(ctx)
    frame = build_m3_frame(ctx, events)
    # chronological cut on event time; each group has a single as_of so groups
    # stay intact (this is Split A for location; Stage 6 adds the unseen-terminal
    # split C).
    cut3 = frame["as_of"].quantile(0.70)
    ev_tr = frame[frame["as_of"] < cut3]
    ev_te = frame[frame["as_of"] >= cut3]
    m3 = _xgb_binary().fit(ev_tr[LOCATION_FEATURES], ev_tr["label"])
    print(f"[M3] events: train groups={ev_tr['group'].nunique()} "
          f"test groups={ev_te['group'].nunique()}")
    agg = location_metrics(ctx, m3, ev_te)
    print("candidate recall@50 is the ceiling (Stage 3): ~0.82 overall\n")
    _print_loc(agg)
    m3_full = _xgb_binary().fit(frame[LOCATION_FEATURES], frame["label"])

    # ==================== save ====================
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(m1_full, f"{config.MODEL_DIR}/m1.joblib")
    joblib.dump(m2_full, f"{config.MODEL_DIR}/m2.joblib")
    joblib.dump(m3_full, f"{config.MODEL_DIR}/m3.joblib")
    joblib.dump(M1_FEATURES, f"{config.MODEL_DIR}/m1_features.joblib")
    joblib.dump(M2_FEATURES, f"{config.MODEL_DIR}/m2_features.joblib")
    joblib.dump(LOCATION_FEATURES, f"{config.MODEL_DIR}/m3_features.joblib")
    joblib.dump(ctx.mule_scorer, f"{config.MODEL_DIR}/mule_scorer.joblib")
    print(f"\n[saved] m1/m2/m3 + feature lists + mule_scorer -> {config.MODEL_DIR}")


if __name__ == "__main__":
    main()
