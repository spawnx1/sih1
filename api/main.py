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

import hashlib
import os
import threading
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
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


@app.middleware("http")
async def _no_cache(request, call_next):
    # never let the browser cache the console -- a stale cached index.html was
    # serving old JS and hiding new features. Always fetch fresh on refresh.
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-store"
    return resp


STATE: dict = {}
# FeatureContext.bind() stores the current case on the shared context, so two forecasts
# running at once on FastAPI's thread pool would read each other's trail. One at a time.
_CTX_LOCK = threading.RLock()


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
    with _CTX_LOCK:
        return _predict_locked(ack_no, as_of, account)


def _predict_locked(ack_no: str, as_of: datetime, account: str | None) -> Prediction:
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
    if STATE["queue_cache"] is None:
        with _CTX_LOCK:                      # build once, never interleaved with a forecast
            if STATE["queue_cache"] is None:
                STATE["queue_cache"] = _compute_queue()
    return STATE["queue_cache"]


def _compute_queue():
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
        raise HTTPException(status_code=404, detail=f"unknown ack_no: {ack_no}")
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
        raise HTTPException(status_code=404, detail=f"unknown ack_no: {ack_no}")
    filed = pd.Timestamp(r["filed_ts"]).to_pydatetime()
    aof = _parse_as_of(as_of, filed)
    trail = expand_trail(r["seed_txn_id"], aof, STATE["ctx"])
    trail.ack_no = ack_no
    return trail


@app.get("/predict/{ack_no}")
def get_predict(ack_no: str, as_of: str | None = None, account: str | None = None):
    r = _case_row(ack_no)
    if r is None:
        raise HTTPException(status_code=404, detail=f"unknown ack_no: {ack_no}")
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
        if channel:
            dominant = "ATM" if pred.channel.ATM >= pred.channel.POS else "POS"
            if channel.strip().upper() != dominant:
                continue
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
                "pin": str(ctx._atm_pin[a]),
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
        raise HTTPException(status_code=404, detail=f"unknown ack_no: {ack_no}")
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
            f"[RED][URGENT] FREEZE {pred.account_id} now",
            f"URGENT - FREEZE ACTION REQUIRED. Ack {ack_no}. Account {pred.account_id} "
            f"holds Rs{rec.tainted_amount:,.0f} tainted funds. "
            f"P(cash-out<=10m)={rec.p_cashout_10m:.2f}. Likely cash-out cell "
            f"{rec.target_cell} ({rec.target_district}). Freeze deadline {rec.freeze_deadline}."))
        made.append(_mk_alert(pred, "WEBHOOK", "bank-freeze-api://cfcfrms",
            f"FREEZE_REQUEST {pred.account_id}",
            f"tier=RED action=FREEZE ack_no={ack_no} account_id={pred.account_id} "
            f"tainted_amount={rec.tainted_amount:,.0f} p_cashout_10m={rec.p_cashout_10m:.2f} "
            f"district={rec.target_district} target_cell={rec.target_cell} "
            f"freeze_deadline={rec.freeze_deadline}"))
    elif rec.tier == "AMBER":
        made.append(_mk_alert(pred, "EMAIL", f"{rec.target_district}.cyber@jhpolice.gov.in",
            f"[AMBER] Monitor {pred.account_id}",
            f"MONITOR ONLY - no freeze or dispatch requested. Ack {ack_no}. "
            f"Account {pred.account_id} holds Rs{rec.tainted_amount:,.0f} tainted funds. "
            f"P(cash-out<=10m)={rec.p_cashout_10m:.2f}. District {rec.target_district}, "
            f"cell {rec.target_cell}. Action: MONITOR - continue tracing, no immediate dispatch."))
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

@app.get("/explain/{ack_no}")
def explain_case(ack_no: str, as_of: str | None = None):
    """Return deterministic explanation facts for the officer console."""
    r = _case_row(ack_no)
    if r is None:
        raise HTTPException(status_code=404, detail=f"unknown ack_no: {ack_no}")

    aof = _parse_as_of(as_of, pd.Timestamp(r["filed_ts"]).to_pydatetime())
    pred = _predict(ack_no, aof, None)
    rec = pred.recommendation

    dominant_channel = (
        "ATM" if pred.channel.ATM >= pred.channel.POS else "POS"
    )

    top_cell = pred.location.cells[0] if pred.location.cells else None
    location_district = top_cell.district if top_cell else None

    reasons = [
        f"P(cash-out<=10m)={pred.cashout.p_10m:.2f}",
        f"tainted_amount=Rs{rec.tainted_amount:,.0f}",
        f"dominant_channel={dominant_channel}",
        f"location={location_district}",
        f"granularity={pred.location.granularity}",
        f"decision={rec.tier}/{rec.action}",
    ]

    return {
        "ack_no": pred.ack_no,
        "account_id": pred.account_id,
        "as_of": pred.as_of,
        "reasons": reasons,
        "prediction": pred.model_dump(mode="json"),
        "recommendation": rec.model_dump(mode="json"),
        "narrative": (
            f"Case {pred.ack_no}: account {pred.account_id} has "
            f"P(cash-out<=10m)={pred.cashout.p_10m:.2f} with "
            f"Rs{rec.tainted_amount:,.0f} tainted funds. "
            f"Dominant channel is {dominant_channel}, and the predicted "
            f"location is {location_district}. "
            f"Decision: {rec.tier} / {rec.action}."
        ),
    }

@app.get("/replay/state")

@app.get("/replay/state")
def replay_state(case: str, t: float = 0.0):
    """Full render state at t minutes after the seed transaction (for the
    replay scrubber). Returns the trail + prediction + clock at that instant."""
    r = _case_row(case)
    if r is None:
        raise HTTPException(status_code=404, detail=f"unknown ack_no: {case}")
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


# --------------------------------------------------------------------------
# TICKET CENTRE -- SQLite-backed incident lifecycle. Additive: nothing above
# depends on it, the forecast demo runs with or without it. Stores status,
# assignment, analyst notes, resolution + outcome, and an append-only audit
# timeline. The RESOLVED outcome (frozen / recovered / cashed-out) is also the
# ground-truth feedback loop the models can later be retrained on.
# --------------------------------------------------------------------------
import sqlite3
from fastapi import Body
from fastapi.responses import Response

TICKET_DB = os.path.join(config.DATA_DIR, "tickets.db")
_TSTATUS = ["NEW", "ASSIGNED", "INVESTIGATING", "FREEZE_SENT", "RESOLVED", "CLOSED"]
_TOUTCOME = ["FROZEN", "RECOVERED", "CASHED_OUT", "FALSE_POSITIVE"]
_OPEN = ("NEW", "ASSIGNED", "INVESTIGATING", "FREEZE_SENT")


def _priority(tier: str) -> str:
    return {"RED": "P1", "AMBER": "P2"}.get(tier, "P3")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _tdb():
    con = sqlite3.connect(TICKET_DB, timeout=8)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")      # concurrent readers + one writer
    con.execute("PRAGMA busy_timeout=8000")     # wait instead of erroring on a lock
    con.execute("""CREATE TABLE IF NOT EXISTS tickets(
        ack_no TEXT PRIMARY KEY, status TEXT DEFAULT 'NEW', assignee TEXT,
        priority TEXT, tier TEXT, fraud_category TEXT, disputed_amount REAL,
        outcome TEXT, recovered_amount REAL DEFAULT 0, created TEXT, updated TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS ticket_events(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ack_no TEXT, ts TEXT,
        actor TEXT, kind TEXT, detail TEXT)""")
    return con


def _log(con, ack, kind, detail, actor="system"):
    con.execute("INSERT INTO ticket_events(ack_no,ts,actor,kind,detail) VALUES(?,?,?,?,?)",
                (ack, _now(), actor, kind, detail))


def _seed_tickets(con):
    """One ticket per queued case, created once (race-safe via INSERT OR IGNORE).
    Analyst-owned fields are never touched afterwards."""
    if STATE.get("tickets_seeded"):
        return
    for q in _build_queue():
        cur = con.execute(
            """INSERT OR IGNORE INTO tickets(ack_no,status,priority,tier,fraud_category,disputed_amount,created,updated)
               VALUES(?,?,?,?,?,?,?,?)""",
            (q.ack_no, "NEW", _priority(q.tier), q.tier, q.fraud_category, q.disputed_amount, _now(), _now()))
        if cur.rowcount:
            _log(con, q.ack_no, "created", f"Ticket opened at {_priority(q.tier)} ({q.tier})")
    con.commit()
    STATE["tickets_seeded"] = True


def _fetch_ticket(con, ack_no):
    r = con.execute("SELECT * FROM tickets WHERE ack_no=?", (ack_no,)).fetchone()
    if not r:
        return None
    ev = [dict(e) for e in con.execute(
        "SELECT * FROM ticket_events WHERE ack_no=? ORDER BY id DESC", (ack_no,))]
    d = dict(r)
    d["title"] = f"{d['fraud_category']} — {d['ack_no']}"
    d["timeline"] = ev
    return d


@app.get("/tickets/summary")
def ticket_summary():
    con = _tdb(); _seed_tickets(con)
    rows = [dict(r) for r in con.execute("SELECT * FROM tickets")]; con.close()
    return {"total": len(rows),
            "open": sum(1 for r in rows if r["status"] in _OPEN),
            "p1_open": sum(1 for r in rows if r["priority"] == "P1" and r["status"] in _OPEN),
            "assigned": sum(1 for r in rows if r["assignee"]),
            "resolved": sum(1 for r in rows if r["status"] in ("RESOLVED", "CLOSED")),
            "recovered": round(sum((r["recovered_amount"] or 0) for r in rows), 2)}


@app.get("/tickets.csv")
def tickets_csv():
    con = _tdb(); _seed_tickets(con)
    rows = [dict(r) for r in con.execute("SELECT * FROM tickets")]; con.close()
    cols = ["ack_no", "priority", "tier", "fraud_category", "status", "assignee",
            "outcome", "recovered_amount", "disputed_amount", "created", "updated"]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r.get(c, "") if r.get(c) is not None else "") for c in cols))
    return Response("\n".join(lines), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=incidents.csv"})


@app.get("/tickets")
def list_tickets(status: str | None = None, assignee: str | None = None, q: str | None = None):
    con = _tdb(); _seed_tickets(con)
    rows = [_fetch_ticket(con, r["ack_no"]) for r in con.execute("SELECT ack_no FROM tickets")]
    con.close()
    if status:
        rows = [r for r in rows if r["status"] == status]
    if assignee:
        rows = [r for r in rows if (r["assignee"] or "") == assignee]
    if q:
        ql = q.lower()
        rows = [r for r in rows if ql in (r["ack_no"] + " " + (r["fraud_category"] or "") + " " + (r["assignee"] or "")).lower()]
    order = {"P1": 0, "P2": 1, "P3": 2}
    rows.sort(key=lambda r: (r["status"] in ("RESOLVED", "CLOSED"),
                             order.get(r["priority"], 9), -(r["disputed_amount"] or 0)))
    return rows


@app.get("/tickets/{ack_no}")
def get_ticket(ack_no: str):
    con = _tdb(); _seed_tickets(con)
    d = _fetch_ticket(con, ack_no); con.close()
    if not d:
        raise HTTPException(status_code=404, detail=f"no ticket {ack_no}")
    return d


@app.patch("/tickets/{ack_no}")
def update_ticket(ack_no: str, status: str | None = None,
                  assignee: str | None = None, actor: str = "analyst"):
    if status and status not in _TSTATUS:
        raise HTTPException(status_code=400, detail=f"bad status; use {_TSTATUS}")
    con = _tdb(); _seed_tickets(con)
    r = con.execute("SELECT * FROM tickets WHERE ack_no=?", (ack_no,)).fetchone()
    if not r:
        con.close(); raise HTTPException(status_code=404, detail="no ticket")
    if status and status != r["status"]:
        con.execute("UPDATE tickets SET status=?,updated=? WHERE ack_no=?", (status, _now(), ack_no))
        _log(con, ack_no, "status", f"{r['status']} → {status}", actor)
    if assignee is not None and assignee != (r["assignee"] or ""):
        con.execute("""UPDATE tickets SET assignee=?, updated=?,
                       status=CASE WHEN status='NEW' THEN 'ASSIGNED' ELSE status END
                       WHERE ack_no=?""", (assignee or None, _now(), ack_no))
        _log(con, ack_no, "assign", f"Assigned to {assignee or 'Unassigned'}", actor)
    con.commit()
    d = _fetch_ticket(con, ack_no); con.close()
    return d


@app.post("/tickets/{ack_no}/note")
def add_note(ack_no: str, text: str = Body(..., embed=True), actor: str = Body("analyst", embed=True)):
    con = _tdb(); _seed_tickets(con)
    if not con.execute("SELECT 1 FROM tickets WHERE ack_no=?", (ack_no,)).fetchone():
        con.close(); raise HTTPException(status_code=404, detail="no ticket")
    _log(con, ack_no, "note", text, actor)
    con.execute("""UPDATE tickets SET updated=?,
                   status=CASE WHEN status='NEW' THEN 'INVESTIGATING' ELSE status END
                   WHERE ack_no=?""", (_now(), ack_no))
    con.commit()
    d = _fetch_ticket(con, ack_no); con.close()
    return d


@app.post("/tickets/{ack_no}/resolve")
def resolve_ticket(ack_no: str, outcome: str = Body(..., embed=True),
                   recovered_amount: float = Body(0.0, embed=True),
                   note: str | None = Body(None, embed=True), actor: str = Body("analyst", embed=True)):
    if outcome not in _TOUTCOME:
        raise HTTPException(status_code=400, detail=f"bad outcome; use {_TOUTCOME}")
    con = _tdb(); _seed_tickets(con)
    if not con.execute("SELECT 1 FROM tickets WHERE ack_no=?", (ack_no,)).fetchone():
        con.close(); raise HTTPException(status_code=404, detail="no ticket")
    final = "CLOSED" if outcome in ("CASHED_OUT", "FALSE_POSITIVE") else "RESOLVED"
    con.execute("UPDATE tickets SET status=?,outcome=?,recovered_amount=?,updated=? WHERE ack_no=?",
                (final, outcome, recovered_amount, _now(), ack_no))
    detail = f"{final}: {outcome}" + (f" · ₹{recovered_amount:,.0f} recovered" if recovered_amount else "")
    _log(con, ack_no, "resolve", detail, actor)
    if note:
        _log(con, ack_no, "note", note, actor)
    con.commit()
    d = _fetch_ticket(con, ack_no); con.close()
    return d


# ==========================================================================
# CASE STUDIO -- generate a SYNTHETIC demo case on demand (generator.case).
# Everything returned is clearly synthetic; ground_truth is a SEPARATE key the
# front-end only reveals after the models have "predicted" (it is never fed in).
# ==========================================================================
@app.get("/studio/scenarios")
def studio_scenarios():
    from generator.case import SCENARIOS
    return {"scenarios": list(SCENARIOS)}


@app.get("/studio/generate")
def studio_generate(scenario: str = "cashout", seed: int | None = None,
                    accounts: int = 40):
    from generator.case import SCENARIOS, generate_case
    if scenario not in SCENARIOS:
        raise HTTPException(status_code=400, detail=f"bad scenario; use {list(SCENARIOS)}")
    accounts = max(6, min(400, int(accounts)))
    case = generate_case(scenario, seed=seed, accounts=accounts, case_number=seed)
    # model input = everything except ground_truth; ground_truth returned alongside
    # for the reveal step ONLY. The two are separate objects, never merged.
    model_case = {k: v for k, v in case.items() if k != "ground_truth"}
    return {"input": model_case, "ground_truth": case["ground_truth"]}


# ==========================================================================
# PERSISTENT MULE / ACCOUNT PROFILE  (audit req #1 + #2)
# A read-only entity view: ONE account_id, all of its activity across the whole
# corpus, and every fraud case it is linked to -- so an investigator can see that
# "Mule A -> Victim 1 / 2 / 3" and "Mule A -> Mule B -> Mule C" are the SAME
# entity. Assembled entirely from the existing FeatureContext + ticket store;
# no new data model, no second case system.
# ==========================================================================
def _iso_ts(t) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%S")


# deterministic, memorable codename for an account (stable across restarts)
_CODEWORDS = ["VIPER", "HERON", "JACKAL", "ORACLE", "RAVEN", "COBRA", "FALCON",
              "WRAITH", "HYDRA", "LYNX", "OSPREY", "MAMBA", "KESTREL", "NOMAD",
              "CIPHER", "PHANTOM", "BASILISK", "MAGPIE", "SABLE", "ONYX", "VULTURE",
              "SCARAB", "TALON", "GRYPHON", "JAGUAR", "ADDER", "HORNET", "SPECTRE",
              "CARACAL", "IBIS", "RONIN", "MANTIS", "DRAKE", "CRANE", "PYTHON",
              "SHRIKE", "GOSHAWK", "MARLIN", "COYOTE", "SERAPH"]


def _codename(account_id: str) -> str:
    h = int(hashlib.md5(account_id.encode()).hexdigest(), 16)
    word = _CODEWORDS[h % len(_CODEWORDS)]
    digits = "".join(c for c in account_id if c.isdigit())
    suffix = digits[-4:] if digits else f"{h % 9999:04d}"
    return f"{word}-{suffix}"


@app.get("/codenames")
def codenames(ids: str = ""):
    """Stable aliases (the same ones the Mule Network shows) for a comma-separated id list."""
    return {a: _codename(a) for a in (x.strip() for x in ids.split(",")[:500]) if a}


@app.get("/account/{account_id}")
def account_profile(account_id: str, limit: int = 250):
    ctx = STATE["ctx"]
    out_pos = list(ctx.src_pos.get(account_id, []))   # this account is the sender
    in_pos = list(ctx.dst_pos.get(account_id, []))    # this account is the receiver
    known = account_id in ctx.acc.index
    if not out_pos and not in_pos and not known:
        raise HTTPException(status_code=404, detail=f"unknown account: {account_id}")

    events = [(int(p), "out") for p in out_pos] + [(int(p), "in") for p in in_pos]
    events.sort(key=lambda e: ctx.ts[e[0]])           # chronological

    total_in = round(float(sum(ctx.amount[p] for p in in_pos)), 2)
    total_out = round(float(sum(ctx.amount[p] for p in out_pos)), 2)
    counterparties: dict[str, dict] = {}
    atms_used: set[str] = set()
    cases: dict[str, dict] = {}                        # ack_no -> linkage info
    tx_rows = []

    for p, direction in events:
        other = ctx.dst[p] if direction == "out" else ctx.src[p]
        other = None if (other is None or pd.isna(other)) else str(other)
        atm = None if (ctx.atm_id[p] is None or pd.isna(ctx.atm_id[p])) else str(ctx.atm_id[p])
        cashout = bool(ctx.is_cashout[p])
        tainted = bool(ctx._label_tainted[p])
        cid = ctx._label_case[p]
        cid = None if (cid is None or pd.isna(cid)) else str(cid)
        amt = round(float(ctx.amount[p]), 2)
        if other:
            c = counterparties.setdefault(other, {"account": other, "n": 0, "amount": 0.0})
            c["n"] += 1; c["amount"] = round(c["amount"] + amt, 2)
        if cashout and atm:
            atms_used.add(atm)
        if tainted and cid:
            cs = cases.setdefault(cid, {"ack_no": cid, "n_txns": 0, "first_ts": _iso_ts(ctx.ts[p])})
            cs["n_txns"] += 1
        tx_rows.append({"txn_id": str(ctx.txn_id[p]), "ts": _iso_ts(ctx.ts[p]),
                        "direction": direction, "counterparty": other, "amount": amt,
                        "channel": str(ctx.channel[p]), "atm_id": atm,
                        "cashout": cashout, "tainted": tainted, "case_id": cid})

    # linked cases are keyed by the fraud's SEED transaction (label_case_id);
    # resolve each to its filed complaint (ack_no) + role + any existing ticket.
    if "case_by_seed" not in STATE:
        STATE["case_by_seed"] = STATE["complaints"].set_index("seed_txn_id")
    sb = STATE["case_by_seed"]
    acks = list({str(sb.loc[s]["ack_no"]) for s in cases if s in sb.index})
    ticket_status = {}
    if acks:
        try:
            con = _tdb()
            qs = ",".join("?" * len(acks))
            for row in con.execute(f"SELECT ack_no,status,priority FROM tickets WHERE ack_no IN ({qs})", acks):
                ticket_status[row[0]] = {"status": row[1], "priority": row[2]}
            con.close()
        except Exception:
            pass
    linked_cases = []
    for seed, cs in cases.items():
        comp = sb.loc[seed] if seed in sb.index else None
        ack = str(comp["ack_no"]) if comp is not None else None
        victim = str(comp["victim_account"]) if comp is not None else None
        role = "VICTIM" if (victim and victim == account_id) else "MULE"
        linked_cases.append({
            "case_ref": ack or seed, "ack_no": ack, "reported": comp is not None,
            "role": role, "victim_account": victim,
            "disputed_amount": round(float(comp["disputed_amount"]), 2) if comp is not None else None,
            "fraud_category": str(comp["fraud_category"]) if comp is not None else None,
            "filed_ts": _iso_ts(comp["filed_ts"]) if comp is not None else cs["first_ts"],
            "n_txns_here": cs["n_txns"],
            "ticket": ticket_status.get(ack) if ack else None})
    linked_cases.sort(key=lambda c: c["filed_ts"])
    victims = sorted({c["victim_account"] for c in linked_cases
                      if c["role"] == "MULE" and c["victim_account"]})

    info = ctx.account_info(account_id)
    ring = [m for m in ctx.ring_members(account_id) if m != account_id]
    tainted_in = round(float(sum(ctx.amount[p] for p in in_pos if ctx._label_tainted[p])), 2)

    top_links = sorted(counterparties.values(), key=lambda c: c["amount"], reverse=True)[:10]
    return {
        "account_id": account_id,
        "codename": _codename(account_id),
        "role": _archetype(ctx, account_id),   # rule-based flow role, same as the Mule Network
        "known_account": known,
        "kyc": {"district": info.get("kyc_district"), "pin": info.get("kyc_pin"),
                "lat": info.get("kyc_lat"), "lon": info.get("kyc_lon"),
                "bank_code": info.get("bank_code"),
                "open_date": _iso_ts(info["open_date"]) if info.get("open_date") is not None else None,
                "ring_id": (None if info.get("ring_id") is None or pd.isna(info.get("ring_id")) else str(info.get("ring_id")))},
        "stats": {
            "n_transactions": len(tx_rows),
            "n_incoming": len(in_pos), "n_outgoing": len(out_pos),
            "total_inflow": total_in, "total_outflow": total_out,
            "tainted_inflow": tainted_in,
            "unique_counterparties": len(counterparties),
            "distinct_atms": len(atms_used),
            "n_linked_cases": len(linked_cases),
            "first_activity": tx_rows[0]["ts"] if tx_rows else None,
            "last_activity": tx_rows[-1]["ts"] if tx_rows else None},
        "linked_cases": linked_cases,          # the SAME entity across many complaints
        "victims": victims,                    # everyone this mule helped defraud
        "ring_members": ring,
        "top_counterparties": top_links,
        "transactions": list(reversed(tx_rows[-limit:])),   # newest first, capped
    }


# ==========================================================================
# MULE NETWORK  -- the codenamed entity graph. Nodes = the busiest mule accounts
# (ranked by how many fraud cases they appear in), grouped by BEHAVIOUR archetype,
# connected where they collaborate (shared case / direct transfer / same ring).
# Click a node -> that mule's profile (/account/{id}). Cached after first build.
# ==========================================================================
def _archetype(ctx, acc: str) -> str:
    """Behavioural role from the account's own flow shape."""
    out_pos = ctx.src_pos.get(acc, [])
    in_pos = ctx.dst_pos.get(acc, [])
    n_out = len(out_pos)
    n_out_co = int(sum(bool(ctx.is_cashout[p]) for p in out_pos))
    in_cp = len({str(ctx.src[p]) for p in in_pos if not pd.isna(ctx.src[p])})
    out_cp = len({str(ctx.dst[p]) for p in out_pos if not pd.isna(ctx.dst[p])})
    co_ratio = n_out_co / max(1, n_out)
    if co_ratio >= 0.6:
        return "CASHER"          # pulls the money out as cash
    if out_cp > in_cp and out_cp >= 4:
        return "DISTRIBUTOR"     # fans money out to many accounts
    if in_cp >= out_cp and in_cp >= 4:
        return "COLLECTOR"       # gathers money from many sources
    return "RELAY"               # simple pass-through


@app.get("/network")
def mule_network(limit: int = 48):
    limit = max(8, min(120, int(limit)))
    if STATE.get("network_cache", {}).get("limit") == limit:
        return STATE["network_cache"]["data"]
    ctx = STATE["ctx"]
    tx = ctx.tx
    t = tx[tx["label_is_tainted"] & tx["label_case_id"].notna()]
    recv = t.dropna(subset=["dst_account"])
    mcases = recv.groupby("dst_account")["label_case_id"].apply(lambda s: set(s))
    top = list(mcases.apply(len).sort_values(ascending=False).index[:limit])
    topset = set(top)

    nodes = []
    for a in top:
        in_pos = ctx.dst_pos.get(a, [])
        tainted_in = float(sum(ctx.amount[p] for p in in_pos if ctx._label_tainted[p]))
        nodes.append({"id": a, "codename": _codename(a), "group": _archetype(ctx, a),
                      "n_cases": len(mcases[a]), "tainted_in": round(tainted_in, 2),
                      "district": ctx.account_info(a).get("kyc_district")})

    edges: dict = {}
    def _add(a, b, kind, w=1):
        if a == b:
            return
        key = tuple(sorted((a, b)))
        e = edges.setdefault(key, {"source": key[0], "target": key[1], "weight": 0, "kinds": set()})
        e["weight"] += w
        e["kinds"].add(kind)

    # shared fraud case = a working cell
    for i, a in enumerate(top):
        for b in top[i + 1:]:
            sh = len(mcases[a] & mcases[b])
            if sh:
                _add(a, b, "cell", sh)
    # direct tainted transfer between two ranked mules
    mm = t[(t["src_account"].isin(topset)) & (t["dst_account"].isin(topset))]
    for (s, d) in mm.groupby(["src_account", "dst_account"]).groups:
        _add(str(s), str(d), "transfer", 1)
    # same ring
    ring_of = {a: (ctx.acc.loc[a, "label_ring_id"] if a in ctx.acc.index else None) for a in top}
    for i, a in enumerate(top):
        for b in top[i + 1:]:
            ra = ring_of[a]
            if ra is not None and not pd.isna(ra) and ra == ring_of[b]:
                _add(a, b, "ring", 2)

    elist = [{"source": e["source"], "target": e["target"], "weight": min(e["weight"], 6),
              "kind": ("ring" if "ring" in e["kinds"] else "transfer" if "transfer" in e["kinds"] else "cell")}
             for e in edges.values()]
    data = {"nodes": nodes, "edges": elist,
            "groups": [{"key": "CASHER", "label": "Cash-out specialist"},
                       {"key": "DISTRIBUTOR", "label": "Fan-out distributor"},
                       {"key": "COLLECTOR", "label": "Collector / aggregator"},
                       {"key": "RELAY", "label": "Pass-through relay"}]}
    STATE["network_cache"] = {"limit": limit, "data": data}
    return data


# serve the officer console at / (mount last so API routes win)
_web = os.path.join(config.BASE_DIR, "web")
if os.path.isdir(_web):
    app.mount("/", StaticFiles(directory=_web, html=True), name="web")
