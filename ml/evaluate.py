"""
ml/evaluate.py -- the honest scorecard.

Four splits, each strictly harder than the last:
  A  Chronological     -- global 70th-percentile time cut, no account straddles.
  B  Unseen account    -- 20% of accounts held out entirely ("isn't it just
                          memorising that mule X uses ATM Y?").
  C  Unseen terminal   -- 15% of ATMs held out (location generalisation).
  D  Sealed shifted    -- the --sealed corpus, whose parameters the model never
                          saw (different hotspots, slower forwarding, a fraud
                          typology absent from training).

Four baselines to beat, all reported:
  B1 rule engine   inflow>=25k AND >70% forwarded in 10m AND age<90d  (why not rules?)
  B2 logistic regression
  B3 nearest ATM to KYC          (location)
  B4 most-frequent ATM           (location)

Also: reliability diagram, coverage-accuracy curve for the abstention
thresholds, and the North Star -- median LEAD TIME in minutes.

Writes reports/metrics.json and reports/*.png.

Run:  python -m ml.evaluate
"""
from __future__ import annotations

import json
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss,  # noqa: E402
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler  # noqa: E402

import config  # noqa: E402
from graphx.candidates import generate_candidates  # noqa: E402
from graphx.features import MuleScorer  # noqa: E402
from ml.features import (LOCATION_FEATURES, FeatureContext,  # noqa: E402
                         build_location_matrix)
from ml.panel import build_panel  # noqa: E402
from ml.predict import load_models, predict_location  # noqa: E402
from ml.train import (M1_FEATURES, _xgb_binary, build_m3_events,  # noqa: E402
                      build_m3_frame, location_metrics)


def _m1_scores(model, test):
    p = model.predict_proba(test[M1_FEATURES])[:, 1]
    y = test["y"].to_numpy()
    return {"roc_auc": round(float(roc_auc_score(y, p)), 4),
            "pr_auc": round(float(average_precision_score(y, p)), 4),
            "brier": round(float(brier_score_loss(y, p)), 4),
            "n_test": int(len(test)), "test_pos_rate": round(float(y.mean()), 4)}


def _loc_summary(agg, bucket="all"):
    s = agg[bucket]
    n = max(1, s["n"])
    return {"n": s["n"], "terminal_top1": round(s["t1"] / n, 4),
            "terminal_top3": round(s["t3"] / n, 4), "terminal_top5": round(s["t5"] / n, 4),
            "cell_top1": round(s["c1"] / n, 4), "cell_top3": round(s["c3"] / n, 4),
            "district_top1": round(s["d1"] / n, 4),
            "median_km": round(float(np.median(s["km"])) if s["km"] else float("nan"), 3)}


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------
def baseline_rule(test):
    """B1: inflow>=25k AND >70% forwarded in 10m AND account age<90d."""
    pred = ((test["taint_amt"] >= 25000) & (test["pct_forwarded_10m"] > 0.70)
            & (test["account_age_days"] < 90)).astype(int).to_numpy()
    y = test["y"].to_numpy()
    return {"roc_auc": round(float(roc_auc_score(y, pred)), 4),
            "pr_auc": round(float(average_precision_score(y, pred)), 4)}


def baseline_logreg(train, test):
    """B2: logistic regression on the same features."""
    sc = StandardScaler().fit(train[M1_FEATURES])
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(sc.transform(train[M1_FEATURES]), train["y"])
    p = lr.predict_proba(sc.transform(test[M1_FEATURES]))[:, 1]
    y = test["y"].to_numpy()
    return {"roc_auc": round(float(roc_auc_score(y, p)), 4),
            "pr_auc": round(float(average_precision_score(y, p)), 4)}


def baseline_location(ctx, test_frame):
    """B3 nearest-to-KYC and B4 most-frequent-ATM top-1 accuracy."""
    b3, b4, n = 0, 0, 0
    for _, g in test_frame.groupby("group"):
        true = g["true_term"].iloc[0]
        near = g.loc[g["distance_kyc_km"].idxmin(), "atm_id"]
        freq = g.loc[g["prior_use_count"].idxmax(), "atm_id"]
        b3 += int(near == true)
        b4 += int(freq == true)
        n += 1
    n = max(1, n)
    return {"B3_nearest_kyc_top1": round(b3 / n, 4),
            "B4_most_frequent_top1": round(b4 / n, 4), "n": n}


# --------------------------------------------------------------------------
# lead time -- the North Star
# --------------------------------------------------------------------------
def lead_time(ctx, models, amber=config.TIER_AMBER_P):
    """At each complaint's filed_ts, would the system have flagged the mule that
    then cashes out -- and how many minutes AHEAD? Negative/missed cases (the
    complaint arrived after the money was already gone) are reported honestly."""
    from ml.predict import predict_cashout
    from graphx.trail import expand_trail
    complaints = pd.read_parquet(f"{ctx.data_dir}/complaints.parquet")
    leads, missed, flagged = [], 0, 0
    for _, case in complaints.iterrows():
        sid = case["seed_txn_id"]
        if sid not in ctx.txn_pos:
            continue
        filed = pd.Timestamp(case["filed_ts"]).to_pydatetime()
        # tainted cash-outs of this case
        cpos = np.where((ctx._label_case == sid) & ctx.is_cashout & ctx._label_tainted)[0]
        if len(cpos) == 0:
            continue
        future = cpos[ctx.ts[cpos] > np.datetime64(filed)]
        if len(future) == 0:
            missed += 1
            continue
        # earliest future cash-out and the mule doing it
        p0 = future[np.argmin(ctx.ts[future])]
        mule = ctx.src[p0]
        t_co = pd.Timestamp(ctx.ts[p0]).to_pydatetime()
        ctx.bind(trail=expand_trail(sid, filed, ctx), case=case)
        cp = predict_cashout(str(mule), filed, ctx, models)
        if max(cp.p_10m, cp.p_20m, cp.p_30m) >= amber:
            flagged += 1
            leads.append((t_co - filed).total_seconds() / 60.0)
    return {"median_lead_min": round(float(np.median(leads)), 1) if leads else None,
            "mean_lead_min": round(float(np.mean(leads)), 1) if leads else None,
            "n_flagged": flagged, "n_missed_already_cashed": missed,
            "n_cases": int(len(complaints)), "_leads": leads}


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def plot_reliability(model, test, path):
    p = model.predict_proba(test[M1_FEATURES])[:, 1]
    frac, mean = calibration_curve(test["y"], p, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", label="perfect")
    plt.plot(mean, frac, "o-", label="M1")
    plt.xlabel("mean predicted probability")
    plt.ylabel("observed frequency")
    plt.title("M1 reliability diagram (Split A)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def plot_coverage_accuracy(ctx, models, test_frame, path):
    thresholds = np.linspace(0.05, 0.6, 12)
    # precompute per-group top terminal prob + correctness
    tops = []
    for _, g in test_frame.groupby("group"):
        p = models["m3"].predict_proba(g[LOCATION_FEATURES])[:, 1]
        p = np.clip(p, 1e-9, None)
        p = p / p.sum()
        j = int(np.argmax(p))
        tops.append((float(p[j]), g["atm_id"].to_numpy()[j] == g["true_term"].iloc[0]))
    tops = np.array([(a, float(b)) for a, b in tops])
    cov, acc = [], []
    for th in thresholds:
        emit = tops[:, 0] >= th
        cov.append(emit.mean())
        acc.append(tops[emit, 1].mean() if emit.any() else np.nan)
    plt.figure(figsize=(6, 4))
    plt.plot(cov, acc, "o-")
    plt.xlabel("coverage (fraction where a terminal is named)")
    plt.ylabel("top-1 accuracy among emitted")
    plt.title("Location coverage vs accuracy (abstention)")
    for th, c, a in zip(thresholds, cov, acc):
        if not np.isnan(a):
            plt.annotate(f"{th:.2f}", (c, a), fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def plot_lead_hist(leads, path):
    plt.figure(figsize=(6, 4))
    if leads:
        plt.hist(leads, bins=20, color="#3a7", edgecolor="k")
        plt.axvline(np.median(leads), color="r", ls="--",
                    label=f"median {np.median(leads):.0f} min")
        plt.legend()
    plt.xlabel("lead time (minutes before cash-out)")
    plt.ylabel("cases")
    plt.title("Lead time at complaint filing (North Star)")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def main():
    print("Building train context + panels (this takes a minute) ...")
    ctx = FeatureContext(config.DATA_DIR)
    ctx.mule_scorer = MuleScorer().fit(ctx)
    panel = build_panel(ctx, n_legit_base=1800, verbose=False)
    events = build_m3_events(ctx)
    frame = build_m3_frame(ctx, events)
    rng = np.random.default_rng(config.SEED)

    metrics: dict = {"splits": {}, "baselines": {}, "notes": {}}

    # ---- Split A: chronological ----------------------------------------
    cutA = panel["as_of"].quantile(0.70)
    trA = panel[panel["as_of"] < cutA]
    seenA = set(trA["account_id"])
    teA = panel[(panel["as_of"] >= cutA) & (~panel["account_id"].isin(seenA))]
    m1A = _xgb_binary().fit(trA[M1_FEATURES], trA["y"])
    metrics["splits"]["A_chronological"] = {"m1": _m1_scores(m1A, teA)}
    cut3 = frame["as_of"].quantile(0.70)
    fr_trA, fr_teA = frame[frame["as_of"] < cut3], frame[frame["as_of"] >= cut3]
    m3A = _xgb_binary().fit(fr_trA[LOCATION_FEATURES], fr_trA["label"])
    metrics["splits"]["A_chronological"]["location"] = _loc_summary(location_metrics(ctx, m3A, fr_teA))

    # ---- Split B: unseen account ---------------------------------------
    accts = panel["account_id"].unique()
    hold = set(rng.choice(accts, size=int(0.2 * len(accts)), replace=False))
    trB = panel[~panel["account_id"].isin(hold)]
    teB = panel[panel["account_id"].isin(hold)]
    m1B = _xgb_binary().fit(trB[M1_FEATURES], trB["y"])
    metrics["splits"]["B_unseen_account"] = {"m1": _m1_scores(m1B, teB)}
    fr_trB = frame[~frame["true_term"].isin([])]  # location: split by account
    ev_acc = np.array([events[g]["account"] for g in frame["group"]])
    frame_acc = frame.assign(_acc=ev_acc)
    fr_trB = frame_acc[~frame_acc["_acc"].isin(hold)]
    fr_teB = frame_acc[frame_acc["_acc"].isin(hold)]
    if fr_teB["group"].nunique() > 5:
        m3B = _xgb_binary().fit(fr_trB[LOCATION_FEATURES], fr_trB["label"])
        metrics["splits"]["B_unseen_account"]["location"] = _loc_summary(location_metrics(ctx, m3B, fr_teB))

    # ---- Split C: unseen terminal (location) ---------------------------
    all_atms = ctx.atms["atm_id"].to_numpy()
    hold_atm = set(rng.choice(all_atms, size=int(0.15 * len(all_atms)), replace=False))
    fr_trC = frame[~frame["true_term"].isin(hold_atm)]
    fr_teC = frame[frame["true_term"].isin(hold_atm)]
    if fr_teC["group"].nunique() > 5:
        m3C = _xgb_binary().fit(fr_trC[LOCATION_FEATURES], fr_trC["label"])
        metrics["splits"]["C_unseen_terminal"] = {
            "location": _loc_summary(location_metrics(ctx, m3C, fr_teC))}

    # ---- Split D: sealed shifted corpus --------------------------------
    print("Scoring the sealed corpus (Split D) ...")
    models_full = load_models()  # trained on the FULL train corpus
    ctxD = FeatureContext(config.DATA_SEALED_DIR)
    ctxD.mule_scorer = models_full["mule_scorer"]
    panelD = build_panel(ctxD, n_legit_base=1200, verbose=False)
    metrics["splits"]["D_sealed"] = {"m1": _m1_scores(models_full["m1"], panelD)}
    eventsD = build_m3_events(ctxD, seed=config.SEED + 5, n_legit=1500)
    frameD = build_m3_frame(ctxD, eventsD)

    class _Wrap:  # location_metrics expects a model with predict_proba
        def __init__(self, m): self.m = m
        def predict_proba(self, X): return self.m.predict_proba(X)
    metrics["splits"]["D_sealed"]["location"] = _loc_summary(
        location_metrics(ctxD, models_full["m3"], frameD))

    # ---- baselines (on Split A) ----------------------------------------
    metrics["baselines"]["B1_rule_engine"] = baseline_rule(teA)
    metrics["baselines"]["B2_logreg"] = baseline_logreg(trA, teA)
    metrics["baselines"].update({"location": baseline_location(ctx, fr_teA)})
    metrics["baselines"]["M1_for_comparison"] = {"roc_auc": metrics["splits"]["A_chronological"]["m1"]["roc_auc"]}
    metrics["baselines"]["M3_for_comparison"] = {
        "terminal_top1": metrics["splits"]["A_chronological"]["location"]["terminal_top1"]}

    # ---- lead time (North Star) ----------------------------------------
    print("Computing lead time ...")
    lt = lead_time(ctx, models_full)
    leads = lt.pop("_leads")
    metrics["lead_time"] = lt

    # ---- candidate recall (ceiling) ------------------------------------
    metrics["notes"]["candidate_recall_at_50"] = "see graphx.candidates (0.80-0.92 band)"

    # ---- plots ----------------------------------------------------------
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    plot_reliability(m1A, teA, f"{config.REPORTS_DIR}/reliability.png")
    plot_coverage_accuracy(ctx, {"m3": m3A}, fr_teA, f"{config.REPORTS_DIR}/coverage_accuracy.png")
    plot_lead_hist(leads, f"{config.REPORTS_DIR}/lead_time.png")

    with open(f"{config.REPORTS_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- print degradation table ---------------------------------------
    print("\n=== M1 ROC-AUC degradation A -> D (honest, degrading is the point) ===")
    for k in ("A_chronological", "B_unseen_account", "D_sealed"):
        s = metrics["splits"].get(k, {}).get("m1")
        if s:
            print(f"  {k:20s} AUC={s['roc_auc']:.3f}  PR-AUC={s['pr_auc']:.3f}  "
                  f"Brier={s['brier']:.3f}")
    print("\n=== Location terminal Top-1 across splits ===")
    for k in ("A_chronological", "B_unseen_account", "C_unseen_terminal", "D_sealed"):
        s = metrics["splits"].get(k, {}).get("location")
        if s:
            print(f"  {k:20s} top1={s['terminal_top1']:.3f} cell1={s['cell_top1']:.3f} "
                  f"med_km={s['median_km']}")
    print("\n=== Baselines (Split A) ===")
    print(f"  B1 rule     AUC={metrics['baselines']['B1_rule_engine']['roc_auc']:.3f}")
    print(f"  B2 logreg   AUC={metrics['baselines']['B2_logreg']['roc_auc']:.3f}")
    print(f"  M1          AUC={metrics['baselines']['M1_for_comparison']['roc_auc']:.3f}")
    print(f"  B3 near-KYC top1={metrics['baselines']['location']['B3_nearest_kyc_top1']:.3f}  "
          f"B4 freq top1={metrics['baselines']['location']['B4_most_frequent_top1']:.3f}  "
          f"M3 top1={metrics['baselines']['M3_for_comparison']['terminal_top1']:.3f}")
    print(f"\n=== North Star: lead time ===\n  {lt}")
    print(f"\n[saved] reports/metrics.json + reliability.png, coverage_accuracy.png, lead_time.png")


if __name__ == "__main__":
    main()
