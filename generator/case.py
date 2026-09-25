"""
generator/case.py -- terminal-based SYNTHETIC cyber-fraud case generator.

    python -m generator.case --scenario cashout --seed 42
    python -m generator.case --scenario complex --seed 42 --accounts 100
    python -m generator.case --scenario mule_chain --output cases/case_001.json

EVERYTHING here is synthetic. No real accounts, people, banks or locations.
The generator manufactures a leakage-safe case: an OBSERVED money-flow (victim ->
mules -> a holding account) that the model may use, and a HIDDEN cash-out event
(account / cell / time / amount) kept in a separate ground_truth object that must
never enter model input. The models predict the cash-out; ground_truth only
scores WHERE / WHEN / WHO.

Design mirrors the existing corpus semantics (a transaction table where a cash-out
has dst_account=null and an atm/cell set; ground truth carried under label_* keys)
so a generated case can feed the ML + Graph + GNN/TGN pipeline unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timedelta

# --------------------------------------------------------------------------
# synthetic vocabularies -- nothing here maps to a real place or institution
# --------------------------------------------------------------------------
TRANSFER_CHANNELS = ["UPI", "IMPS", "NEFT"]
CASHOUT_CHANNEL = "ATM"
DISTRICTS = ["DISTRICT_A", "DISTRICT_B", "DISTRICT_C", "DISTRICT_D"]
# 12 synthetic ~5km cells, 3 per district
CELLS = [f"CELL_{i:03d}" for i in range(1, 13)]
CELL_DISTRICT = {c: DISTRICTS[i // 3] for i, c in enumerate(CELLS)}
BASE_DAY = datetime(2026, 9, 22, 10, 0, 0)  # synthetic incident clock (IST)

SCENARIOS = ("simple", "branching", "mule_chain", "complex", "cashout")


# --------------------------------------------------------------------------
# scenario topologies -- abstract directed flow over account indices.
# Returns (edges, holders) where edges = [(src_idx, dst_idx, hop)] in flow order
# and holders = indices still holding funds at as_of (cash-out candidates).
# Victim is always index 0.
# --------------------------------------------------------------------------
def _topology(scenario: str, rng: random.Random):
    if scenario == "simple":
        edges = [(0, 1, 1), (1, 2, 2)]
        holders = [2]
    elif scenario == "mule_chain":
        edges = [(0, 1, 1), (1, 2, 2), (2, 3, 3), (3, 4, 4)]
        holders = [4]
    elif scenario == "branching":
        edges = [(0, 1, 1), (1, 2, 2), (1, 3, 2)]
        holders = [2, 3]
    elif scenario == "complex":
        edges = [(0, 1, 1),
                 (1, 2, 2), (1, 3, 2),          # branch
                 (2, 4, 3), (3, 4, 3),          # converge into 4
                 (4, 5, 4), (4, 6, 4)]          # branch again -> two candidates
        holders = [5, 6]
    else:  # "cashout" -- a representative moderate flow with one branch
        edges = [(0, 1, 1), (1, 2, 2), (2, 3, 3), (2, 4, 3)]
        holders = [3, 4]
    return edges, holders


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _acc(i: int) -> str:
    return f"ACC_SYN_{i:03d}"


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _hhmm(dt: datetime) -> str:
    return dt.strftime("%H:%M")


# --------------------------------------------------------------------------
# core generation
# --------------------------------------------------------------------------
def generate_case(scenario: str = "cashout", seed: int | None = None,
                  accounts: int = 40, case_number: int | None = None) -> dict:
    """Build one synthetic case. Deterministic for a given (scenario, seed).

    Returns a dict with model-facing sections and a SEPARATE `ground_truth` key.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
    rng = random.Random(seed if seed is not None else random.randrange(1 << 30))
    case_no = case_number if case_number is not None else (seed if seed is not None else rng.randrange(1, 9999))
    case_id = f"CASE-SYN-{case_no % 10000:04d}"

    edges, holders = _topology(scenario, rng)
    flow_ids = sorted({i for e in edges for i in e[:2]} | set(holders))
    n_flow = len(flow_ids)

    # ---- accounts: flow participants + benign decoys (graph noise) ----
    # hidden true node-type; NOT trivially separable (attributes overlap).
    accounts_rows = {}
    node_labels = {}

    def make_account(idx: int, kind: str):
        # mules skew freshly-opened, but plenty of normals are fresh too -> overlap
        if kind in ("possible_mule", "cashout_account", "suspicious"):
            age_days = rng.randint(6, 240)
        elif kind == "victim":
            age_days = rng.randint(400, 2600)
        else:  # normal
            age_days = rng.randint(20, 2600)
        cell = rng.choice(CELLS)
        aid = _acc(idx)
        accounts_rows[aid] = {
            "account_id": aid,
            "open_date": _iso(BASE_DAY - timedelta(days=age_days, minutes=rng.randint(0, 600))),
            "account_age_days": age_days,
            "home_cell": cell,
            "home_district": CELL_DISTRICT[cell],
            "bank_code": rng.choice(["BANK_S1", "BANK_S2", "BANK_S3", "BANK_S4"]),
        }
        node_labels[aid] = kind

    # classify flow accounts
    intermediaries = {i for e in edges for i in e[:2]} - {0} - set(holders)
    make_account(0, "victim")
    for i in sorted(intermediaries):
        make_account(i, "possible_mule")
    for i in holders:
        make_account(i, "possible_mule")  # true cash-out one is upgraded below

    # decoys
    decoy_start = max(flow_ids) + 1
    n_decoy = max(0, accounts - n_flow)
    decoy_ids = list(range(decoy_start, decoy_start + n_decoy))
    for i in decoy_ids:
        make_account(i, "normal")

    # ---- OBSERVED layering transactions (ts <= as_of) ----
    txns = []            # observed, model-visible
    tno = [0]

    def add_txn(src, dst, amount, ts, channel, hop, tainted, atm=None, cell=None):
        tno[0] += 1
        txns.append({
            "transaction_id": f"TXN_SYN_{tno[0]:04d}",
            "timestamp": _iso(ts),
            "source_account": src,
            "destination_account": dst,          # None for a cash-out
            "amount": round(amount, 2),
            "transaction_type": "cash_out" if atm else "transfer",
            "channel": channel,
            "cell": cell,                         # only set for a cash-out
            "atm_id": atm,
            "hop": hop,
            "_tainted": tainted,                  # kept out of model files (leak guard)
        })

    amount0 = rng.choice([85000, 92000, 78000, 120000, 65000])
    # ---- money-conserving amounts: at a branch the balance is SPLIT (not copied);
    # each hop skims a little. Process sources in flow order so inflows are known. ----
    out_by_src = {}
    for (s, d, hop) in edges:
        out_by_src.setdefault(s, []).append((s, d, hop))
    amt_at = {0: amount0}
    edge_amt = {}
    for s in sorted(out_by_src):                    # indices increase along the flow
        avail = amt_at.get(s, 0.0)
        outs = out_by_src[s]
        if len(outs) == 1:
            fr = [1.0]
        else:                                       # random split summing to 1
            parts = [rng.uniform(0.35, 0.65) for _ in outs]
            tot = sum(parts)
            fr = [p / tot for p in parts]
        skim = rng.uniform(0.04, 0.11)
        for (s, d, hop), f in zip(outs, fr):
            amt = avail * (1 - skim) * f
            edge_amt[(s, d, hop)] = amt
            amt_at[d] = amt_at.get(d, 0.0) + amt
    # emit transactions in chronological (hop) order, minutes apart
    t = BASE_DAY + timedelta(minutes=rng.randint(1, 3))
    last_ts = t
    for (s, d, hop) in edges:
        t = t + timedelta(minutes=rng.randint(2, 5))
        ch = TRANSFER_CHANNELS[min(hop - 1, len(TRANSFER_CHANNELS) - 1)]
        add_txn(_acc(s), _acc(d), edge_amt[(s, d, hop)], t, ch, hop, tainted=True)
        last_ts = max(last_ts, t)

    # benign decoy transactions among normals (same window/cells) -> noise
    for _ in range(int(n_decoy * 1.4)):
        if len(decoy_ids) < 2:
            break
        a, b = rng.sample(decoy_ids, 2)
        tt = BASE_DAY + timedelta(minutes=rng.randint(-40, 8))
        add_txn(_acc(a), _acc(b), rng.uniform(500, 40000), tt,
                rng.choice(TRANSFER_CHANNELS), hop=0, tainted=False)

    # ---- as_of = the prediction moment (just after the money settles) ----
    as_of = last_ts + timedelta(minutes=2)

    # keep only observed <= as_of for model input; sort chronologically
    obs_txns = sorted([x for x in txns if x["timestamp"] <= _iso(as_of)],
                      key=lambda x: x["timestamp"])

    # ---- HIDDEN cash-out (ground truth; ts > as_of) ----
    true_holder = rng.choice(holders)
    true_acc = _acc(true_holder)
    node_labels[true_acc] = "cashout_account"
    # cash-out cell: usually the holder's home cell, sometimes an adjacent one
    home_cell = accounts_rows[true_acc]["home_cell"]
    if rng.random() < 0.25:
        true_cell = rng.choice([c for c in CELLS if CELL_DISTRICT[c] == CELL_DISTRICT[home_cell]])
    else:
        true_cell = home_cell
    cashout_amt = round(amt_at.get(true_holder, amount0) * rng.uniform(0.9, 0.99), 2)
    cashout_time = as_of + timedelta(minutes=rng.randint(3, 12))  # positive lead time
    cashout_node = f"CASHOUT_SYN_{rng.randint(1, 99):02d}"

    # candidate cash-out cells (space the WHERE prediction ranks over)
    cand = {accounts_rows[_acc(h)]["home_cell"] for h in holders}
    cand.add(true_cell)
    while len(cand) < 3:
        cand.add(rng.choice(CELLS))
    candidate_cells = sorted(cand)

    # ---- per-account ML features (as-of, observed only; NO future) ----
    features = _features(obs_txns, accounts_rows, as_of)

    # ---- GNN nodes/edges (observed only, temporal order preserved) ----
    nodes = []
    involved = sorted({x["source_account"] for x in obs_txns} |
                      {x["destination_account"] for x in obs_txns if x["destination_account"]})
    for aid in involved:
        f = features[aid]
        nodes.append({
            "node_id": aid,
            "is_victim": int(aid == _acc(0)),      # the ONLY known role (from the complaint)
            "home_cell": accounts_rows[aid]["home_cell"],
            "account_age_days": accounts_rows[aid]["account_age_days"],
            "total_inflow": f["total_inflow"],
            "total_outflow": f["total_outflow"],
            "n_incoming": f["number_of_incoming_transactions"],
            "n_outgoing": f["number_of_outgoing_transactions"],
            "unique_counterparties": f["unique_counterparties"],
            "velocity": f["velocity"],
            "network_degree": f["network_degree"],
        })
    edges_out = [{
        "source_node": x["source_account"],
        "destination_node": x["destination_account"],
        "timestamp": x["timestamp"],
        "transaction_amount": x["amount"],
        "transaction_type": x["transaction_type"],
        "channel": x["channel"],
    } for x in obs_txns if x["destination_account"]]

    # ---- timelines: an OBSERVED one (model-safe) and a FULL one (eval side, with
    # the hidden cash-out) that lives ONLY inside ground_truth. ----
    obs_timeline = sorted(
        [{"timestamp": x["timestamp"], "event": "transfer",
          "src": x["source_account"], "dst": x["destination_account"],
          "amount": x["amount"], "channel": x["channel"]} for x in obs_txns],
        key=lambda e: e["timestamp"])
    cashout_event = {"timestamp": _iso(cashout_time), "event": "CASH_OUT",
                     "src": true_acc, "dst": cashout_node, "amount": cashout_amt,
                     "channel": CASHOUT_CHANNEL, "cell": true_cell}
    full_timeline = sorted(obs_timeline + [cashout_event], key=lambda e: e["timestamp"])

    # strip the internal taint flag from model-facing transactions
    model_txns = [{k: v for k, v in x.items() if not k.startswith("_")} for x in obs_txns]

    # counts for the terminal banner
    mule_like = [a for a, k in node_labels.items()
                 if k in ("possible_mule", "cashout_account") and a in involved]

    case = {
        "case_id": case_id,
        "incident_type": "cyber_fraud",
        "scenario": scenario,
        "seed": seed,
        "generated_by": "generator.case (SYNTHETIC -- not real data)",
        "as_of": _iso(as_of),
        "initial_complaint": {
            "victim_account": _acc(0),
            "disputed_amount": round(amount0, 2),
            "reported_at": _iso(as_of),
            "fraud_category": rng.choice(["upi_fraud", "kyc_scam", "investment_scam",
                                          "job_scam", "refund_scam"]),
        },
        "victim_account": _acc(0),
        "accounts": [accounts_rows[a] for a in sorted(accounts_rows)],
        "transactions": model_txns,          # OBSERVED, model input
        "locations": {"cells": CELLS, "districts": DISTRICTS,
                      "candidate_cashout_cells": candidate_cells},
        "features": features,
        "graph": {"nodes": nodes, "edges": edges_out},
        "timeline": obs_timeline,          # OBSERVED only -- model-safe
        "stats": {
            "n_transactions": len(model_txns),
            "n_accounts": len(involved),
            "n_potential_mules": len(mule_like),
            "candidate_cells": candidate_cells,
        },
        # ------ SEPARATE. Never merge into model input. ------
        "ground_truth": {
            "true_cashout_account": true_acc,
            "true_cashout_cell": true_cell,
            "true_cashout_time": _iso(cashout_time),
            "cashout_amount": cashout_amt,
            "lead_time_min": round((cashout_time - as_of).total_seconds() / 60, 1),
            "flow_path": [_acc(i) for i in _victim_path(edges, true_holder)],
            "node_labels": {a: node_labels[a] for a in involved},
            "full_timeline": full_timeline,   # observed + the hidden cash-out (eval/replay)
        },
    }
    return case


def _victim_path(edges, target):
    """Reconstruct the hop path victim(0) -> target over the flow edges."""
    pred = {}
    for s, d, _h in edges:
        pred.setdefault(d, s)
    path, cur, guard = [target], target, 0
    while cur != 0 and cur in pred and guard < 50:
        cur = pred[cur]
        path.append(cur)
        guard += 1
    return list(reversed(path))


def _features(obs_txns, accounts_rows, as_of):
    """As-of per-account features from OBSERVED transactions only (no leakage)."""
    feats = {}
    by_acc = {}
    for x in obs_txns:
        by_acc.setdefault(x["source_account"], []).append(("out", x))
        if x["destination_account"]:
            by_acc.setdefault(x["destination_account"], []).append(("in", x))
    for aid, evs in by_acc.items():
        ins = [x for d, x in evs if d == "in"]
        outs = [x for d, x in evs if d == "out"]
        tstamps = sorted(x["timestamp"] for _d, x in evs)
        counterparties = set()
        for d, x in evs:
            other = x["destination_account"] if d == "out" else x["source_account"]
            if other:
                counterparties.add(other)
        total_in = sum(x["amount"] for x in ins)
        total_out = sum(x["amount"] for x in outs)
        first = datetime.fromisoformat(tstamps[0])
        last = datetime.fromisoformat(tstamps[-1])
        active_min = max(1.0, (last - first).total_seconds() / 60)
        age = accounts_rows.get(aid, {}).get("account_age_days", 0)
        feats[aid] = {
            "transaction_amount": round(total_in + total_out, 2),
            "transaction_frequency": round(len(evs) / active_min, 4),
            "time_since_previous_transaction_min": round((as_of - last).total_seconds() / 60, 2),
            "number_of_incoming_transactions": len(ins),
            "number_of_outgoing_transactions": len(outs),
            "total_inflow": round(total_in, 2),
            "total_outflow": round(total_out, 2),
            "account_age_days": age,
            "unique_counterparties": len(counterparties),
            "velocity": round((total_in + total_out) / active_min, 2),
            "network_degree": len(counterparties),
            "cell": accounts_rows.get(aid, {}).get("home_cell"),
        }
    return feats


# --------------------------------------------------------------------------
# cinematic terminal render
# --------------------------------------------------------------------------
class C:
    def __init__(self, on): self.on = on
    def _w(self, s, c): return f"\033[{c}m{s}\033[0m" if self.on else s
    def dim(self, s): return self._w(s, "2;37")
    def cyan(self, s): return self._w(s, "1;36")
    def amber(self, s): return self._w(s, "1;33")
    def red(self, s): return self._w(s, "1;31")
    def green(self, s): return self._w(s, "1;32")
    def white(self, s): return self._w(s, "1;37")


def _flow_tree(case, col: C) -> list[str]:
    """ASCII money-flow tree from the victim over the OBSERVED transfers."""
    adj = {}
    for e in case["graph"]["edges"]:
        adj.setdefault(e["source_node"], []).append(e["destination_node"])
    truth_path = set(case["ground_truth"]["flow_path"])  # for highlight only (render side)
    lines = []
    seen = set()

    def walk(node, prefix, is_last, top=False):
        if top:
            lines.append(col.red("VICTIM  ") + col.dim(node))
        seen.add(node)
        kids = [k for k in adj.get(node, []) if k not in seen]
        for i, k in enumerate(kids):
            last = i == len(kids) - 1
            branch = "└──▶ " if last else "├──▶ "
            single = "  ↓  " if len(kids) == 1 else branch
            conn = single if len(kids) == 1 else branch
            tag = col.amber(k) if k in truth_path else col.white(k)
            lines.append(prefix + col.dim(conn) + tag)
            walk(k, prefix + ("     " if last else col.dim("│    ")), last)

    root = case["victim_account"]
    walk(root, "", True, top=True)
    # the cash-out is a forecast target -> shown as a hidden terminal node
    lines.append("")
    lines.append("  " + col.dim("↓  (forecast)"))
    lines.append("  " + col.red("◆ CASH-OUT") + col.dim("   [ predicted — not in model input ]"))
    return lines


def print_case(case, color=True, reveal=True):
    col = C(color)
    bar = col.cyan("=" * 54)
    p = print
    p(bar)
    p(col.cyan("  SYNTHETIC CYBERCRIME CASE") + col.dim("   (demo data — not real)"))
    p(bar)
    p("")
    p(col.dim("  Case ID       ") + col.white(case["case_id"]) +
      col.dim("      scenario ") + col.amber(case["scenario"]))
    p(col.dim("  Incident      ") + col.white(case["incident_type"]))
    p(col.dim("  Initial amt   ") + col.amber(f"₹{case['initial_complaint']['disputed_amount']:,.0f}"))
    p(col.dim("  As-of (now)   ") + col.white(case["as_of"].replace("T", "  ")))
    p("")
    p(col.dim("  MONEY FLOW"))
    p("")
    for ln in _flow_tree(case, col):
        p("    " + ln)
    p("")
    s = case["stats"]
    p(col.dim("  Transactions            ") + col.white(str(s["n_transactions"])))
    p(col.dim("  Accounts in flow        ") + col.white(str(s["n_accounts"])))
    p(col.dim("  Potential mule accounts ") + col.amber(str(s["n_potential_mules"])))
    p(col.dim("  Candidate cash-out cells"))
    for c in s["candidate_cells"]:
        p("      " + col.cyan(c) + col.dim("  " + CELL_DISTRICT[c]))
    p("")
    p(bar)
    if reveal:
        gt = case["ground_truth"]
        p("")
        p(col.red("  [ GROUND TRUTH — HIDDEN FROM MODEL ]"))
        p(col.dim("  cash-out cell   ") + col.green(gt["true_cashout_cell"]))
        p(col.dim("  cash-out time   ") + col.green(gt["true_cashout_time"].replace("T", "  ")))
        p(col.dim("  cash-out acct   ") + col.green(gt["true_cashout_account"]))
        p(col.dim("  amount          ") + col.green(f"₹{gt['cashout_amount']:,.0f}"))
        p(col.dim("  lead time       ") + col.green(f"{gt['lead_time_min']} min"))
        p("")


# --------------------------------------------------------------------------
# file bundle writers
# --------------------------------------------------------------------------
def _write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_bundle(case, out_dir):
    """Write the case bundle. ground_truth.json is kept SEPARATE from model input."""
    d = os.path.join(out_dir, case["case_id"])
    os.makedirs(d, exist_ok=True)

    model_case = {k: v for k, v in case.items() if k != "ground_truth"}
    with open(os.path.join(d, "case.json"), "w", encoding="utf-8") as f:
        json.dump(model_case, f, indent=2)
    with open(os.path.join(d, "ground_truth.json"), "w", encoding="utf-8") as f:
        json.dump(case["ground_truth"], f, indent=2)

    _write_csv(os.path.join(d, "transactions.csv"), case["transactions"],
               ["transaction_id", "timestamp", "source_account", "destination_account",
                "amount", "transaction_type", "channel", "cell", "atm_id", "hop"])
    _write_csv(os.path.join(d, "accounts.csv"), case["accounts"],
               ["account_id", "open_date", "account_age_days", "home_cell",
                "home_district", "bank_code"])
    _write_csv(os.path.join(d, "nodes.csv"), case["graph"]["nodes"],
               list(case["graph"]["nodes"][0].keys()) if case["graph"]["nodes"] else ["node_id"])
    _write_csv(os.path.join(d, "edges.csv"), case["graph"]["edges"],
               ["source_node", "destination_node", "timestamp",
                "transaction_amount", "transaction_type", "channel"])
    _write_csv(os.path.join(d, "timeline.csv"), case["timeline"],
               ["timestamp", "event", "src", "dst", "amount", "channel", "cell"])
    return d


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None):
    try:                                   # Windows consoles default to cp1252; we emit ₹ and box glyphs
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        prog="generator.case",
        description="Generate a SYNTHETIC cyber-fraud case (demo/test only).")
    ap.add_argument("--scenario", default="cashout", choices=SCENARIOS)
    ap.add_argument("--seed", type=int, default=None, help="deterministic when set")
    ap.add_argument("--accounts", type=int, default=40, help="account pool size (adds benign noise)")
    ap.add_argument("--output", default=None, help="also write case.json to this path")
    ap.add_argument("--out-dir", default="cases", help="directory for the case bundle")
    ap.add_argument("--no-bundle", action="store_true", help="print only; write no files")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--hide-truth", action="store_true", help="do not print ground truth")
    args = ap.parse_args(argv)

    case = generate_case(args.scenario, seed=args.seed, accounts=args.accounts,
                         case_number=args.seed)
    color = (not args.no_color) and sys.stdout.isatty()
    print_case(case, color=color, reveal=not args.hide_truth)

    if not args.no_bundle:
        d = write_bundle(case, args.out_dir)
        print(("\033[2;37m" if color else "") + f"  wrote bundle -> {d}" +
              ("\033[0m" if color else ""))
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in case.items() if k != "ground_truth"}, f, indent=2)
        print(("\033[2;37m" if color else "") + f"  wrote case.json -> {args.output}" +
              ("\033[0m" if color else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
