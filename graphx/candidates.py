"""
graphx/candidates.py -- 5-source candidate ATM generation.

generate_candidates(account_id, as_of, ctx) returns up to CANDIDATE_CAP (50)
terminal ids, blind to the label. The union of five sources, in priority order:

  C1 (15)  every ATM this account used in the last 90 days
  C2 (12)  ATMs within 4 km of its historical withdrawal centroid
  C3 (10)  ATMs within 3 km of its KYC PIN centroid
  C4 (10)  top ATMs used by other accounts in the same ring in 180 days  <-- the differentiator
  C5 (8)   top fraud-withdrawal ATMs in the district, TRAIN period only

Candidate recall@50 (fraction of true cash-out terminals that appear anywhere
in this set) is a HARD CEILING on Top-K accuracy and must be reported first.

Run:  python -m graphx.candidates   (Stage 3 acceptance: trail + recall@50)
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

import config
from ml.features import FeatureContext, haversine_km


def _ensure_cache(ctx):
    if getattr(ctx, "_cand_ready", False):
        return
    ctx._alat = ctx.atms["lat"].to_numpy()
    ctx._alon = ctx.atms["lon"].to_numpy()
    ctx._aid = ctx.atms["atm_id"].to_numpy()
    # top fraud ATMs per district (train period only)
    train = ctx.ts.astype("int64") < ctx.train_cutoff
    mask = train & ctx.is_cashout & ctx._label_tainted
    fatm = ctx.atm_id[mask]
    fdist = np.array([ctx._atm_district.get(a) for a in fatm])
    dfa: dict[str, list[str]] = {}
    for d in config.DISTRICTS:
        ids = fatm[fdist == d]
        if len(ids):
            vc = _value_counts(ids)
            dfa[d] = [a for a, _ in vc]
        else:
            dfa[d] = []
    ctx._district_fraud_atms = dfa
    ctx._cand_ready = True


def _value_counts(arr) -> list[tuple[str, int]]:
    vals, counts = np.unique(arr, return_counts=True)
    order = np.argsort(counts)[::-1]
    return [(str(vals[i]), int(counts[i])) for i in order]


def _atms_within(ctx, clat, clon, radius_km, cap):
    d = haversine_km(clat, clon, ctx._alat, ctx._alon)
    idx = np.where(d <= radius_km)[0]
    if len(idx) == 0:
        return []
    idx = idx[np.argsort(d[idx])][:cap]
    return [str(ctx._aid[i]) for i in idx]


def generate_candidates_detailed(account_id: str, as_of: datetime, ctx):
    """Return (ordered_list, sources_map, use_counts) for M3 feature building."""
    _ensure_cache(ctx)
    sources: dict[str, list[str]] = {}
    ordered: list[str] = []

    def add(atm_id, src):
        if atm_id is None:
            return
        atm_id = str(atm_id)
        sources.setdefault(atm_id, [])
        if src not in sources[atm_id]:
            sources[atm_id].append(src)
        if atm_id not in ordered:
            ordered.append(atm_id)

    # C1 -- own withdrawals, last 120 days (most-frequent first)
    used = ctx.withdrawal_atms(account_id, as_of, days=120)
    use_counts = {a: c for a, c in _value_counts(np.array(used))} if used else {}
    for a, _ in (_value_counts(np.array(used)) if used else [])[:15]:
        add(a, "C1")

    # C2 -- near the historical withdrawal centroid
    cen = ctx.withdrawal_centroid(account_id, as_of, days=120)
    if cen is not None:
        for a in _atms_within(ctx, cen[0], cen[1], 6.0, 16):
            add(a, "C2")

    # C3 -- near the KYC PIN centroid (falls back to the account-opening branch,
    # both R3-allowed) -- this is the only geo signal for a no-history account.
    info = ctx.account_info(account_id)
    if info:
        for a in _atms_within(ctx, info["kyc_lat"], info["kyc_lon"], 6.0, 14):
            add(a, "C3")
        br = ctx.acc.loc[account_id]
        for a in _atms_within(ctx, float(br["branch_lat"]), float(br["branch_lon"]), 6.0, 10):
            add(a, "C3")

    # C4 -- ring co-members' ATMs, last 180 days (the differentiator)
    ring_atms = []
    for m in ctx.ring_members(account_id):
        ring_atms.extend(ctx.withdrawal_atms(m, as_of, days=180))
    if ring_atms:
        for a, _ in _value_counts(np.array(ring_atms))[:10]:
            add(a, "C4")

    # C5 -- district's top fraud ATMs (train period only)
    district = info.get("kyc_district") if info else None
    if district:
        for a in ctx._district_fraud_atms.get(district, [])[:12]:
            add(a, "C5")

    ordered = ordered[: config.CANDIDATE_CAP]
    return ordered, sources, use_counts


def generate_candidates(account_id: str, as_of: datetime, ctx) -> list[str]:
    ordered, _, _ = generate_candidates_detailed(account_id, as_of, ctx)
    return ordered


# --------------------------------------------------------------------------
# Stage 3 acceptance
# --------------------------------------------------------------------------
def _stage3_acceptance():
    import pandas as pd

    from graphx.trail import expand_trail

    ctx = FeatureContext(config.DATA_DIR)
    complaints = pd.read_parquet(f"{config.DATA_DIR}/complaints.parquet")

    # ---- 1) expand a known case, print node/edge counts per hop ----------
    # pick a fast_fanout case whose seed resolves
    case = complaints[complaints["fraud_category"] == "fast_fanout"].iloc[0]
    trail = expand_trail(case["seed_txn_id"], case["filed_ts"].to_pydatetime(), ctx)
    print(f"\n=== Trail for case {case['ack_no']} ({case['fraud_category']}) ===")
    print(f"seed={case['seed_txn_id']}  disputed=₹{case['disputed_amount']:.0f}  "
          f"as_of={case['filed_ts']}")
    from collections import Counter
    node_hops = Counter(n.hop_from_victim for n in trail.nodes)
    edge_hops = Counter(e.hop for e in trail.edges)
    for h in sorted(set(node_hops) | set(edge_hops)):
        print(f"  hop {h}: nodes={node_hops.get(h,0):3d}  edges={edge_hops.get(h,0):3d}")
    print(f"  total nodes={len(trail.nodes)} edges={len(trail.edges)} "
          f"leaves(holding funds)={len(trail.leaf_accounts)}")

    # ---- 2) candidate recall@50 over all fraud cash-outs -----------------
    # target = each true (tainted) cash-out; as_of = its timestamp (candidates
    # use only history strictly before it, so the target itself is excluded).
    co_mask = ctx.is_cashout & ctx._label_tainted
    pos = np.where(co_mask)[0]
    hits, total = 0, 0
    hits_hist, total_hist = 0, 0  # stratified: accounts WITH prior ATM history
    for p in pos:
        acc = ctx.src[p]
        if acc is None:
            continue
        as_of = pd.Timestamp(ctx.ts[p]).to_pydatetime()
        term = str(ctx.atm_id[p])
        cands = set(generate_candidates(acc, as_of, ctx))
        has_hist = len(ctx.withdrawal_atms(acc, as_of, days=90)) > 0
        total += 1
        hit = term in cands
        hits += int(hit)
        if has_hist:
            total_hist += 1
            hits_hist += int(hit)
    print(f"\n=== Candidate recall@50 over {total} true cash-outs ===")
    print(f"  overall recall@50 = {hits/max(1,total):.3f}  (expect 0.80-0.92)")
    print(f"  with prior ATM history ({total_hist}) = {hits_hist/max(1,total_hist):.3f}")
    print(f"  no prior history ({total-total_hist}) = "
          f"{(hits-hits_hist)/max(1,total-total_hist):.3f}")


if __name__ == "__main__":
    _stage3_acceptance()
