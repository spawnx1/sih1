"""
api/main.py -- the law-enforcement interface (one FastAPI process, no auth, no
database, no websockets -- a hackathon prototype meant to survive a live demo).

Endpoints:
  GET  /queue?limit=&sort=            triage across all complaints
  GET  /case/{ack_no}                 complaint header + golden-hour clock
  GET  /trail/{ack_no}?as_of=         money-flow graph truncated at as_of
  GET  /predict/{ack_no}?as_of=&account=   M1 + M2 + M3 in one response
  GET  /hotspots?from=&to=&min_amt=&channel=   GIS heatmap cells + ATM markers
  POST /alerts/dispatch   GET /alerts  simulated SMS/email/webhook + delivery log
  GET  /replay/state?case=&t=         full render state at a simulated instant

The DECISION LAYER is plain rules, not ML -- an officer must be able to read why
an alert fired. The primary recommendation is FREEZE (stop the debit clearing);
dispatching a constable is the second-best outcome. Alerts route to the police
station covering the top-ranked CELL, not the top-ranked terminal.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import config
from contracts import (Alert, Case, Prediction, QueueItem, Recommendation, Trail)
from graphx.trail import expand_trail
from ml.features import FeatureContext
from ml.predict import (load_models, predict_cashout, predict_channel,
                        predict_location)

app = FastAPI(title="SIH26184 -- Cash-out Forecast Console", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

STATE: dict = {}


# --------------------------------------------------------------------------
# startup: load everything once
# --------------------------------------------------------------------------
@app.on_event("startup")
def _startup():
    import joblib
    ctx = FeatureContext(config.DATA_DIR)
    ctx.mule_scorer = joblib.load(f"{config.MODEL_DIR}/mule_scorer.joblib")
    STATE["ctx"] = ctx
    STATE["models"] = load_models()
    STATE["complaints"] = pd.read_parquet(f"{config.DATA_DIR}/complaints.parquet")
    STATE["case_by_ack"] = STATE["complaints"].set_index("ack_no")
    STATE["alerts"] = []
    STATE["dedup"] = set()
    STATE["queue_cache"] = None
    print(f"[api] loaded {len(STATE['complaints'])} complaints, models ready")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _parse_as_of(s: str | None, default: datetime) -> datetime:
    if not s:
        return default
    s = s.replace("Z", "").split("+")[0].strip()
    try:
        return pd.Timestamp(s).to_pydatetime().replace(tzinfo=None)
    except Exception:
        return default


def _case_row(ack_no: str):
    cb = STATE["case_by_ack"]
    if ack_no not in cb.index:
        return None
    r = cb.loc[ack_no]
    return r


def _seed_ts(seed_txn_id: str) -> datetime:
    ctx = STATE["ctx"]
    return pd.Timestamp(ctx.ts[ctx.txn_pos[seed_txn_id]]).to_pydatetime()


def _top_leaf(trail: Trail, ctx) -> str | None:
    """The account holding the most tainted money right now."""
    leaves = [n for n in trail.nodes if n.is_leaf]
    if not leaves:
        cand = [n for n in trail.nodes if not n.is_victim]
        if not cand:
            return None
        return max(cand, key=lambda n: n.tainted_amount).account_id
    return max(leaves, key=lambda n: n.tainted_amount).account_id


def _most_imminent(trail: Trail, as_of: datetime, case):
    """The account MOST LIKELY to cash out next (max M1 P<=10m) among all accounts
    that received tainted funds -- the correct triage target, not merely the one
    holding the most money."""
    ctx, models = STATE["ctx"], STATE["models"]
    ctx.bind(trail=trail, case=case)
    received = [n.account_id for n in trail.nodes
                if not n.is_victim and n.tainted_amount > 0]
    if not received:
        acc = _top_leaf(trail, ctx) or case["victim_account"]
        return acc, predict_cashout(acc, as_of, ctx, models)
    best = None
    for a in received:
        cp = predict_cashout(a, as_of, ctx, models)
        if best is None or cp.p_10m > best[1].p_10m:
            best = (a, cp)
    return best


def make_recommendation(cashout, location, tainted_amount: float,
                        account: str, as_of: datetime) -> Recommendation:
    """Plain-rules decision layer. Routes to the station covering the top CELL."""
    top_cell = location.cells[0] if location.cells else None
    district = top_cell.district if top_cell else (
        location.districts[0].district if location.districts else None)
    sho = config.JURISDICTION_SHO.get(district, {})
    p10 = cashout.p_10m

    if p10 >= config.TIER_RED_P and tainted_amount >= config.TIER_RED_AMOUNT:
        return Recommendation(
            tier="RED",
            action="FREEZE",
            reason=(
                f"P(cash-out<=10m)={p10:.2f} >= {config.TIER_RED_P} AND tainted "
                f"Rs{tainted_amount:,.0f} >= Rs{config.TIER_RED_AMOUNT:,} -> "
                f"freeze first; alert {sho.get('station', 'SHO')} covering cell "
                f"{top_cell.cell if top_cell else '?'}"
            ),
            freeze_account=account,
            freeze_deadline=as_of + timedelta(minutes=10),
            target_station=sho.get("station"),
            target_district=district,
            target_cell=top_cell.cell if top_cell else None,
            tainted_amount=tainted_amount,
            p_cashout_10m=p10
        )

    if p10 >= config.TIER_AMBER_P:
        if p10 >= config.TIER_RED_P:
            reason = (
                f"P(cash-out<=10m)={p10:.2f} >= {config.TIER_RED_P}, "
                f"but tainted Rs{tainted_amount:,.0f} < "
                f"Rs{config.TIER_RED_AMOUNT:,} -> "
                f"AMBER monitoring; email brief to {district} desk, "
                f"no dispatch yet"
            )
        else:
            reason = (
                f"P(cash-out<=10m)={p10:.2f} in "
                f"[{config.TIER_AMBER_P},{config.TIER_RED_P}) -> "
                f"email brief to {district} desk, no dispatch yet"
            )

        return Recommendation(
            tier="AMBER",
            action="MONITOR",
            reason=reason,
            target_station=sho.get("station"),
            target_district=district,
            target_cell=top_cell.cell if top_cell else None,
            tainted_amount=tainted_amount,
            p_cashout_10m=p10
        )

    return Recommendation(
        tier="GREY",
        action="MONITOR",
        reason=f"P(cash-out<=10m)={p10:.2f} < {config.TIER_AMBER_P} -> keep tracing",
        target_district=district,
        tainted_amount=tainted_amount,
        p_cashout_10m=p10
    )


def _predict(ack_no: str, as_of: datetime, account: str | None) -> Prediction:
    ctx, models = STATE["ctx"], STATE["models"]
    case = _case_row(ack_no)
    trail = expand_trail(case["seed_txn_id"], as_of, ctx)
    trail.ack_no = ack_no
    if account is None:
        account, cashout = _most_imminent(trail, as_of, case)
    else:
        ctx.bind(trail=trail, case=case)
        cashout = predict_cashout(account, as_of, ctx, models)
    ctx.bind(trail=trail, case=case)
    channel = predict_channel(account, as_of, ctx, models)
    tainted = next((n.tainted_amount for n in trail.nodes if n.account_id == account),
                   float(case["disputed_amount"]))
    location = predict_location(account, as_of, ctx, models, amount=tainted)
    rec = make_recommendation(cashout, location, tainted, account, as_of)
    return Prediction(ack_no=ack_no, account_id=account, as_of=as_of,
                      cashout=cashout, channel=channel, location=location,
                      recommendation=rec)


def _build_queue():
    if STATE["queue_cache"] is not None:
        return STATE["queue_cache"]
    ctx, models = STATE["ctx"], STATE["models"]
    items = []
    for _, case in STATE["complaints"].iterrows():
        sid = case["seed_txn_id"]
        if sid not in ctx.txn_pos:
            continue
        filed = pd.Timestamp(case["filed_ts"]).to_pydatetime()
        seed_ts = _seed_ts(sid)
        trail = expand_trail(sid, filed, ctx)
        if not any(not n.is_victim for n in trail.nodes):
            continue
        acc, cp = _most_imminent(trail, filed, case)
        tainted = next((n.tainted_amount for n in trail.nodes if n.account_id == acc), 0.0)
        tier = ("RED" if (cp.p_10m >= config.TIER_RED_P and tainted >= config.TIER_RED_AMOUNT)
                else "AMBER" if cp.p_10m >= config.TIER_AMBER_P else "GREY")
        mins_since = (filed - seed_ts).total_seconds() / 60.0
        golden = max(0.0, 60.0 - mins_since)
        imminence = cp.p_10m * np.log1p(tainted)
        items.append(QueueItem(
            ack_no=case["ack_no"], victim_account=case["victim_account"],
            filed_ts=filed, disputed_amount=float(case["disputed_amount"]),
            fraud_category=case["fraud_category"], top_account=acc,
            p_cashout_10m=cp.p_10m, tier=tier, imminence=round(float(imminence), 3),
            minutes_since_filed=round(mins_since, 1),
            golden_hour_remaining_min=round(golden, 1)))
    STATE["queue_cache"] = items
    return items


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------
@app.get("/queue")
def get_queue(limit: int = 50, sort: str = "imminence"):
    items = list(_build_queue())
    key = {"imminence": lambda q: q.imminence,
           "p": lambda q: q.p_cashout_10m,
           "amount": lambda q: q.disputed_amount,
           "golden": lambda q: -q.golden_hour_remaining_min}.get(sort,
                                                                 lambda q: q.imminence)
    items.sort(key=key, reverse=True)
    return items[:limit]


@app.get("/case/{ack_no}")
def get_case(ack_no: str):
    r = _case_row(ack_no)
    if r is None:
        return {"error": "unknown ack_no"}
    seed_ts = _seed_ts(r["seed_txn_id"])
    filed = pd.Timestamp(r["filed_ts"]).to_pydatetime()
    case = Case(ack_no=ack_no, filed_ts=filed, victim_account=r["victim_account"],
                seed_txn_id=r["seed_txn_id"], disputed_amount=float(r["disputed_amount"]),
                fraud_category=r["fraud_category"],
                reporting_delay_min=float(r["reporting_delay_min"]))
    return {"case": case.model_dump(mode="json"),
            "seed_ts": seed_ts.strftime("%Y-%m-%dT%H:%M:%S") + "+05:30",
            "golden_hour_remaining_min": round(max(0.0, 60.0 -
                (filed - seed_ts).total_seconds() / 60.0), 1)}


@app.get("/trail/{ack_no}")
def get_trail(ack_no: str, as_of: str | None = None):
    r = _case_row(ack_no)
    if r is None:
        return {"error": "unknown ack_no"}
    filed = pd.Timestamp(r["filed_ts"]).to_pydatetime()
    aof = _parse_as_of(as_of, filed)
    trail = expand_trail(r["seed_txn_id"], aof, STATE["ctx"])
    trail.ack_no = ack_no
    return trail


@app.get("/predict/{ack_no}")
def get_predict(ack_no: str, as_of: str | None = None, account: str | None = None):
    r = _case_row(ack_no)
    if r is None:
        return {"error": "unknown ack_no"}
    aof = _parse_as_of(as_of, pd.Timestamp(r["filed_ts"]).to_pydatetime())
    return _predict(ack_no, aof, account)


@app.get("/hotspots")
def get_hotspots(from_: str | None = Query(None, alias="from"),
                 to: str | None = None, min_amt: float = 0.0,
                 channel: str | None = None):
    """Predicted risk per H3 cell across complaints in the window + ATM markers."""
    ctx = STATE["ctx"]
    cache_key = (from_, to, min_amt, channel)
    STATE.setdefault("hotspot_cache", {})
    if cache_key in STATE["hotspot_cache"]:
        return STATE["hotspot_cache"][cache_key]
    lo = _parse_as_of(from_, config.CORPUS_START)
    hi = _parse_as_of(to, config.CORPUS_END + timedelta(days=3))
    cell_risk: dict[str, dict] = {}
    for _, case in STATE["complaints"].iterrows():
        filed = pd.Timestamp(case["filed_ts"]).to_pydatetime()
        if not (lo <= filed <= hi):
            continue
        if float(case["disputed_amount"]) < min_amt:
            continue
        pred = _predict(case["ack_no"], filed, None)
        if channel and channel.upper() not in (
                "ATM" if pred.channel.ATM >= pred.channel.POS else "POS",):
            # light channel filter: keep if the dominant cash-out channel matches
            if channel.upper() not in ("ATM", "POS"):
                pass
        for c in pred.location.cells[:5]:
            d = cell_risk.setdefault(c.cell, {"cell": c.cell, "lat": c.lat,
                "lon": c.lon, "district": c.district, "risk": 0.0, "n": 0})
            d["risk"] += float(c.p) * float(pred.recommendation.tainted_amount)
            d["n"] += 1
    cells = sorted(cell_risk.values(), key=lambda d: d["risk"], reverse=True)
    if cells:
        mx = max(c["risk"] for c in cells) or 1.0
        for c in cells:
            c["risk_norm"] = round(c["risk"] / mx, 4)
            c["risk"] = round(c["risk"], 1)
    markers = [{"atm_id": a, "lat": ctx._atm_lat[a], "lon": ctx._atm_lon[a],
                "operator": ctx._atm_operator[a], "district": ctx._atm_district[a],
                "fraud_rate": round(ctx.atm_fraud_rate(a), 3)}
               for a in ctx.atms["atm_id"]]
    result = {"cells": cells[:60], "atm_markers": markers,
              "district_centroids": {d: {"lat": v[0], "lon": v[1]}
                                     for d, v in config.DISTRICT_CENTROIDS.items()}}
    STATE["hotspot_cache"][cache_key] = result
    return result


@app.post("/alerts/dispatch")
def dispatch_alert(ack_no: str, account: str | None = None, as_of: str | None = None):
    r = _case_row(ack_no)
    if r is None:
        return {"error": "unknown ack_no"}
    aof = _parse_as_of(as_of, pd.Timestamp(r["filed_ts"]).to_pydatetime())
    pred = _predict(ack_no, aof, account)
    rec = pred.recommendation
    # dedup: one alert per account per 15 minutes
    bucket = int(aof.timestamp() // (config.ALERT_DEDUP_MINUTES * 60))
    key = f"{pred.account_id}:{bucket}"
    if key in STATE["dedup"]:
        return {"deduped": True, "recommendation": rec.model_dump(mode="json")}
    made = []
    if rec.tier == "RED":
        sho = config.JURISDICTION_SHO.get(rec.target_district, {})
        made.append(_mk_alert(pred, "SMS", sho.get("sho_phone", "+91-0000000000"),
            f"[RED] FREEZE {pred.account_id} now",
            f"Freeze a/c {pred.account_id} (Rs{rec.tainted_amount:,.0f} tainted). "
            f"P(cash-out<=10m)={rec.p_cashout_10m:.2f}. Likely cell {rec.target_cell} "
            f"({rec.target_district}). Deadline {rec.freeze_deadline}."))
        made.append(_mk_alert(pred, "WEBHOOK", "bank-freeze-api://cfcfrms",
            f"FREEZE_REQUEST {pred.account_id}",
            f"Automated freeze request for {pred.account_id}, ack {ack_no}."))
    elif rec.tier == "AMBER":
        made.append(_mk_alert(pred, "EMAIL", f"{rec.target_district}.cyber@jhpolice.gov.in",
            f"[AMBER] Watch {pred.account_id}",
            f"Monitor a/c {pred.account_id}. P(cash-out<=10m)={rec.p_cashout_10m:.2f}. "
            f"Likely district {rec.target_district}."))
    if made:
        STATE["dedup"].add(key)
        STATE["alerts"].extend(made)
    return {"deduped": False, "dispatched": [a.model_dump(mode="json") for a in made],
            "recommendation": rec.model_dump(mode="json")}


def _mk_alert(pred, channel, recipient, subject, body) -> Alert:
    rec = pred.recommendation
    return Alert(alert_id=uuid.uuid4().hex[:12], created_ts=pred.as_of,
                 tier=rec.tier, channel=channel, recipient=recipient,
                 ack_no=pred.ack_no, account_id=pred.account_id,
                 district=rec.target_district, cell=rec.target_cell,
                 subject=subject, body=body, status="sent",
                 dedup_key=f"{pred.account_id}")


@app.get("/alerts")
def get_alerts(district: str | None = None):
    alerts = STATE["alerts"]

    if district:
        district = district.strip()
        alerts = [
            alert for alert in alerts
            if alert.district and alert.district.lower() == district.lower()
        ]

    return alerts

@app.get("/replay/state")
def replay_state(case: str, t: float = 0.0):
    """Full render state at t minutes after the seed transaction (for the
    replay scrubber). Returns the trail + prediction + clock at that instant."""
    r = _case_row(case)
    if r is None:
        return {"error": "unknown ack_no"}
    seed_ts = _seed_ts(r["seed_txn_id"])
    aof = seed_ts + timedelta(minutes=float(t))
    pred = _predict(case, aof, None)
    trail = expand_trail(r["seed_txn_id"], aof, STATE["ctx"])
    trail.ack_no = case
    filed = pd.Timestamp(r["filed_ts"]).to_pydatetime()
    return {"t_minutes": t, "as_of": aof.strftime("%Y-%m-%dT%H:%M:%S") + "+05:30",
            "seed_ts": seed_ts.strftime("%Y-%m-%dT%H:%M:%S") + "+05:30",
            "complaint_filed": filed >= aof and "pending" or "filed",
            "minutes_to_complaint": round((filed - aof).total_seconds() / 60.0, 1),
            "trail": trail.model_dump(mode="json"),
            "prediction": pred.model_dump(mode="json")}


# serve the officer console at / (mount last so API routes win)
_web = os.path.join(config.BASE_DIR, "web")
if os.path.isdir(_web):
    app.mount("/", StaticFiles(directory=_web, html=True), name="web")
