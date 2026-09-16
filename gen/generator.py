"""
gen/generator.py -- the synthetic corpus.

Produces four tables over a 180-day window ending 2026-08-31 (naive IST):

    accounts.parquet       shipped account table (NO internal columns)
    _accounts_internal.parquet   archetype + home_* (generator-only, R3)
    transactions.parquet   ~250k rows, cash-outs encoded inline
    complaints.parquet     ~600 cybercrime complaints

The generator's whole job is to be HARD. If a model trained on this scores
> 0.95 AUC, this file has a leak -- widen the distributions, don't celebrate.

Run:
    python -m gen.generator            # training corpus  -> data/
    python -m gen.generator --sealed   # shifted sealed corpus -> data_sealed/

Cash-out encoding (the ONLY cash-out representation, no separate table):
    channel in (ATM_WDL, POS)  AND  dst_account IS NULL  AND  atm_id non-null
"""
from __future__ import annotations

import argparse
import os
from datetime import timedelta

import numpy as np
import pandas as pd

import config
from gen.params import Params

CASHOUT_CHANNELS = {"ATM_WDL", "POS"}
BANK_CODES = ["SBIN", "HDFC", "PUNB", "BARB", "UTIB", "ICIC", "CNRB", "IOBA"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised haversine. Any arg may be scalar or ndarray."""
    r = 6371.0088
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def round_amount(x: float, step: int = 500) -> float:
    return float(max(step, round(x / step) * step))


def scatter_point(rng, clat, clon, radius_shape=1.2, radius_scale=1.6):
    r_km = rng.gamma(radius_shape, radius_scale)
    theta = rng.uniform(0, 2 * np.pi)
    dlat = (r_km * np.cos(theta)) / 111.32
    dlon = (r_km * np.sin(theta)) / (111.32 * np.cos(np.radians(clat)))
    return clat + dlat, clon + dlon


# --------------------------------------------------------------------------
# generator
# --------------------------------------------------------------------------
class Generator:
    def __init__(self, params: Params):
        self.p = params
        self.rng = np.random.default_rng(params.seed)
        self.atms = pd.read_parquet(os.path.join(config.DATA_DIR, "atms.parquet"))
        # arrays for fast candidate maths
        self.a_lat = self.atms["lat"].to_numpy()
        self.a_lon = self.atms["lon"].to_numpy()
        self.a_id = self.atms["atm_id"].to_numpy()
        self.a_cell = self.atms["cell"].to_numpy()
        self.a_district = self.atms["district"].to_numpy()
        self.a_attract = self.atms["_attract"].to_numpy()
        self.a_limit = self.atms["per_txn_limit"].to_numpy()
        self.a_hourly = list(self.atms["hourly_weight"])
        self.idx_by_district = {
            d: np.where(self.a_district == d)[0] for d in config.DISTRICTS
        }
        self.idx_by_cell: dict[str, np.ndarray] = {}
        for c in np.unique(self.a_cell):
            self.idx_by_cell[c] = np.where(self.a_cell == c)[0]
        self._txn_seq = 0

    # ---- ids -----------------------------------------------------------
    def _txn_id(self) -> str:
        self._txn_seq += 1
        return f"TXN_{self._txn_seq:08d}"

    # ---- per-account ATM preference (the honest-Top-1 machinery) --------
    def _preference(self, home_lat, home_lon):
        d = haversine_km(home_lat, home_lon, self.a_lat, self.a_lon)
        k = min(self.p.pref_n_nearest, len(d))
        idx = np.argpartition(d, k - 1)[:k]
        dd = d[idx]
        w = np.exp(-dd / self.p.pref_decay_km) * (0.6 + self.a_attract[idx])
        w = w / w.sum()
        alpha = np.maximum(self.p.pref_dirichlet_alpha * w / w.mean(), 1e-3)
        pvec = self.rng.dirichlet(alpha)
        order = np.argsort(pvec)[::-1]
        idx, pvec = idx[order], pvec[order]
        return idx, pvec

    def _pick_from_pref(self, pref):
        idx, pvec = pref
        j = self.rng.choice(len(idx), p=pvec)
        return int(idx[j])

    def _pick_in_cell(self, pref, cell):
        idx, pvec = pref
        mask = self.a_cell[idx] == cell
        if mask.any():
            sub_idx, sub_p = idx[mask], pvec[mask]
            sub_p = sub_p / sub_p.sum()
            j = self.rng.choice(len(sub_idx), p=sub_p)
            return int(sub_idx[j])
        pool = self.idx_by_cell.get(cell)
        if pool is not None and len(pool):
            return int(self.rng.choice(pool))
        return self._pick_from_pref(pref)

    def _pick_in_district(self, district, exclude_cell=None):
        pool = self.idx_by_district[district]
        if exclude_cell is not None:
            pool = pool[self.a_cell[pool] != exclude_cell]
        if len(pool) == 0:
            pool = self.idx_by_district[district]
        return int(self.rng.choice(pool))

    def _pick_other_district(self, home_district):
        others = [d for d in config.DISTRICTS if d != home_district]
        d = self.rng.choice(others)
        return int(self.rng.choice(self.idx_by_district[d]))

    # ---- accounts ------------------------------------------------------
    def build_accounts(self):
        p, rng = self.p, self.rng
        n = p.n_accounts
        acc = []
        internal = []
        legit_arch = list(p.legit_archetype_mix)
        legit_w = np.array(list(p.legit_archetype_mix.values()))
        legit_w /= legit_w.sum()
        mule_arch = list(p.mule_archetype_mix)
        mule_w = np.array(list(p.mule_archetype_mix.values()))
        mule_w /= mule_w.sum()
        hot_d = list(p.mule_hotspot_mix)
        hot_w = np.array(list(p.mule_hotspot_mix.values()))
        hot_w /= hot_w.sum()
        leg_d = list(p.legit_district_mix)
        leg_w = np.array(list(p.legit_district_mix.values()))
        leg_w /= leg_w.sum()

        is_mule = rng.random(n) < p.mule_frac
        # rings: collect MULE_RING accounts into groups of ring_size
        ring_members: list[int] = []

        self.pref: list = [None] * n
        self.expected_cell = [None] * n
        self.home_district = [None] * n
        self.behaves_legit = np.zeros(n, dtype=bool)

        for i in range(n):
            if is_mule[i]:
                archetype = rng.choice(mule_arch, p=mule_w)
                district = rng.choice(hot_d, p=hot_w)
            else:
                archetype = rng.choice(legit_arch, p=legit_w)
                district = rng.choice(leg_d, p=leg_w)
            clat, clon = config.DISTRICTS[district]["lat"], config.DISTRICTS[district]["lon"]
            home_lat, home_lon = scatter_point(rng, clat, clon)
            import h3

            home_cell = h3.latlng_to_cell(home_lat, home_lon, config.H3_RES_CELL)
            # People bank near where they live: KYC + opening branch sit a short
            # offset from home (not independently across the district). This is
            # what makes the KYC PIN centroid a usable prior (R3-allowed).
            kyc_lat, kyc_lon = scatter_point(rng, home_lat, home_lon, 1.0, 0.9)
            branch_lat, branch_lon = scatter_point(rng, home_lat, home_lon, 1.0, 0.7)
            open_days_ago = int(rng.integers(30, 2000))
            open_date = (config.CORPUS_END - timedelta(days=open_days_ago)).date()

            pref = self._preference(home_lat, home_lon)
            self.pref[i] = pref
            self.expected_cell[i] = self.a_cell[pref[0][0]]
            self.home_district[i] = district
            if is_mule[i] and rng.random() < p.mule_as_legit_frac:
                self.behaves_legit[i] = True

            aid = f"ACC_{i:05d}"
            acc.append(
                {
                    "account_id": aid,
                    "bank_code": rng.choice(BANK_CODES),
                    "open_date": pd.Timestamp(open_date),
                    "kyc_pin": str(814000 + int(rng.integers(0, 20000))),
                    "kyc_district": district,
                    "kyc_lat": round(float(kyc_lat), 6),
                    "kyc_lon": round(float(kyc_lon), 6),
                    "branch_lat": round(float(branch_lat), 6),
                    "branch_lon": round(float(branch_lon), 6),
                    "label_is_mule": bool(is_mule[i]),
                    "label_ring_id": None,  # filled below
                }
            )
            internal.append(
                {
                    "account_id": aid,
                    "archetype": archetype,
                    "home_lat": round(float(home_lat), 6),
                    "home_lon": round(float(home_lon), 6),
                    "home_cell": home_cell,
                }
            )
            if archetype == "MULE_RING":
                ring_members.append(i)

        # ---- assign rings & shared cash-out geography ------------------
        # Rings are district-LOCAL: a real mule ring works one area, so all its
        # members share both a district and a cash-out cell (this also makes C4
        # -- ring co-members' ATMs -- a genuine signal rather than noise).
        self.ring_of: dict[int, str] = {}
        self.ring_members: dict[str, list[int]] = {}
        self.ring_pref: dict[str, tuple] = {}
        by_district: dict[str, list[int]] = {d: [] for d in config.DISTRICTS}
        for i in ring_members:
            by_district[self.home_district[i]].append(i)
        rid = 0
        district_chunks: list[list[int]] = []
        for d, members_d in by_district.items():
            rng.shuffle(members_d)
            for start in range(0, len(members_d) - p.ring_size + 1, p.ring_size):
                district_chunks.append(members_d[start : start + p.ring_size])
        for members in district_chunks:
            ring_id = f"RING_{rid:04d}"
            rid += 1
            # shared preference computed from the FIRST member's home
            anchor = internal[members[0]]
            shared_pref = self._preference(anchor["home_lat"], anchor["home_lon"])
            self.ring_pref[ring_id] = shared_pref
            self.ring_members[ring_id] = members
            for m in members:
                self.ring_of[m] = ring_id
                acc[m]["label_ring_id"] = ring_id
                self.pref[m] = shared_pref  # rings share cash-out geography
                self.expected_cell[m] = self.a_cell[shared_pref[0][0]]

        # ---- label noise: flip 2-3% of label_is_mule ------------------
        flip = rng.random(n) < p.label_noise
        for i in np.where(flip)[0]:
            acc[i]["label_is_mule"] = not acc[i]["label_is_mule"]

        self.accounts = pd.DataFrame(acc)
        self.accounts_internal = pd.DataFrame(internal)
        self.is_mule = is_mule  # TRUE archetype-based flag (pre-noise) for gen use
        self.n = n
        # index pools
        self.mule_idx = np.where(is_mule)[0]
        self.legit_idx = np.where(~is_mule)[0]
        return self.accounts

    # ---- timing helpers ------------------------------------------------
    def _jitter(self, ts):
        lo, hi = self.p.ts_jitter_sec
        return ts + timedelta(seconds=float(self.rng.uniform(-hi, hi)))

    def _channel_batch_lag(self, channel):
        if channel in ("NEFT", "RTGS"):
            lo, hi = self.p.neft_batch_lag_min
            return timedelta(minutes=float(self.rng.uniform(lo, hi)))
        return timedelta(0)

    def _random_ts(self):
        day = int(self.rng.integers(0, config.CORPUS_DAYS))
        base = config.CORPUS_START + timedelta(days=day)
        hour = int(self.rng.choice(24, p=self._daytime_profile()))
        return base + timedelta(hours=hour, minutes=float(self.rng.uniform(0, 60)))

    _daytime_cache = None

    def _daytime_profile(self):
        if self._daytime_cache is None:
            h = np.arange(24)
            w = np.exp(-0.5 * ((h - 13) / 4.0) ** 2) + 0.05
            w[0:6] *= 0.15
            self._daytime_cache = w / w.sum()
        return self._daytime_cache

    def _withdrawal_ts(self, atm_i):
        day = int(self.rng.integers(0, config.CORPUS_DAYS))
        base = config.CORPUS_START + timedelta(days=day)
        hw = np.asarray(self.a_hourly[atm_i], dtype=float)
        hw = hw / hw.sum()
        hour = int(self.rng.choice(24, p=hw))
        return base + timedelta(hours=hour, minutes=float(self.rng.uniform(0, 60)))

    def _mk(self, ts, src, dst, amount, channel, atm_i=None, tainted=False,
            case_id=None, hop=-1, dst_vpa=None):
        """Build one transaction row, enforcing the cash-out invariant."""
        is_cashout = channel in CASHOUT_CHANNELS
        if is_cashout:
            assert dst is None and atm_i is not None
            atm_id = str(self.a_id[atm_i])
        else:
            assert dst is not None
            atm_id = None
        observed = bool(self.rng.random() >= self.p.unobserved_rate)
        return {
            "txn_id": self._txn_id(),
            "ts": self._jitter(ts),
            "src_account": src,
            "dst_account": dst,
            "amount": round(float(amount), 2),
            "channel": channel,
            "dst_vpa": dst_vpa,
            "atm_id": atm_id,
            "observed": observed,
            "label_is_tainted": tainted,
            "label_case_id": case_id,
            "label_hop": hop,
        }

    # ---- background (legit) activity -----------------------------------
    def build_background(self):
        rng, p = self.rng, self.p
        rows: list[dict] = []
        internal = self.accounts_internal
        # choose which legit accounts get a hard-negative look-alike burst
        hardneg = np.zeros(self.n, dtype=bool)
        legit = self.legit_idx
        hn_sel = legit[rng.random(len(legit)) < p.hard_negative_frac]
        hardneg[hn_sel] = True
        hn_styles = [
            "salary_sweep", "smb_cash_cycle", "family_two_tranche",
            "dormant_payout", "emergency", "round_transfer",
        ]

        for i in range(self.n):
            aid = f"ACC_{i:05d}"
            arch = internal.iloc[i]["archetype"]
            pref = self.pref[i]
            n_tx = int(rng.poisson(26)) + 4
            for _ in range(n_tx):
                r = rng.random()
                if r < 0.20:  # incoming credit
                    ch = rng.choice(["NEFT", "IMPS", "UPI"], p=[0.4, 0.35, 0.25])
                    amt = round_amount(rng.lognormal(9.4, 0.8), 100)
                    ts = self._random_ts() + self._channel_batch_lag(ch)
                    rows.append(self._mk(ts, f"EXT_{int(rng.integers(0,4000)):04d}",
                                         aid, amt, ch, hop=-1))
                elif r < 0.60:  # outgoing UPI/IMPS payment to a payee
                    ch = rng.choice(["UPI", "IMPS", "NEFT"], p=[0.7, 0.2, 0.1])
                    amt = round_amount(rng.lognormal(8.2, 0.9), 50)
                    ts = self._random_ts() + self._channel_batch_lag(ch)
                    rows.append(self._mk(ts, aid, f"PAYEE_{int(rng.integers(0,6000)):04d}",
                                         amt, ch, dst_vpa=f"user{int(rng.integers(0,9999))}@ybl"))
                elif r < 0.90:  # ATM cash withdrawal (a legit cash-out)
                    atm_i = self._pick_from_pref(pref)
                    limit = int(self.a_limit[atm_i])
                    amt = round_amount(min(rng.lognormal(8.6, 0.5), limit), 500)
                    ts = self._withdrawal_ts(atm_i)
                    rows.append(self._mk(ts, aid, None, amt, "ATM_WDL", atm_i=atm_i))
                elif r < 0.96:  # POS cash-out
                    atm_i = self._pick_from_pref(pref)
                    limit = int(self.a_limit[atm_i])
                    amt = round_amount(min(rng.lognormal(8.1, 0.6), limit), 100)
                    ts = self._withdrawal_ts(atm_i)
                    rows.append(self._mk(ts, aid, None, amt, "POS", atm_i=atm_i))
                else:  # cash deposit (SMB / FARMER mostly)
                    amt = round_amount(rng.lognormal(9.0, 0.7), 100)
                    ts = self._random_ts()
                    rows.append(self._mk(ts, None, aid, amt, "CASH_DEP", hop=-1))

            # hard-negative burst: a legit account that LOOKS like a mule
            if hardneg[i]:
                self._hard_negative_burst(rows, i, aid, pref,
                                          rng.choice(hn_styles))

        self.bg_rows = rows
        return rows

    def _hard_negative_burst(self, rows, i, aid, pref, style):
        """Six legit patterns that mimic mule velocity/shape (no fraud label)."""
        rng = self.rng
        t0 = self._random_ts()
        atm_i = self._pick_from_pref(pref)
        limit = int(self.a_limit[atm_i])
        if style == "salary_sweep":
            credit = round_amount(rng.lognormal(10.6, 0.4), 1000)  # big salary
            rows.append(self._mk(t0, "EXT_SALARY", aid, credit, "NEFT"))
            t1 = t0 + timedelta(minutes=float(rng.uniform(8, 45)))
            rows.append(self._mk(t1, aid, None, round_amount(min(credit * 0.6, limit), 500),
                                 "ATM_WDL", atm_i=atm_i))
        elif style == "smb_cash_cycle":
            for _ in range(int(rng.integers(2, 4))):
                credit = round_amount(rng.lognormal(9.8, 0.5), 500)
                tt = t0 + timedelta(hours=float(rng.uniform(0, 6)))
                rows.append(self._mk(tt, f"MERCH_{int(rng.integers(0,999)):03d}", aid,
                                     credit, "UPI"))
                rows.append(self._mk(tt + timedelta(minutes=float(rng.uniform(5, 30))),
                                     aid, None, round_amount(min(credit, limit), 500),
                                     "ATM_WDL", atm_i=atm_i))
        elif style == "family_two_tranche":
            credit = round_amount(rng.lognormal(10.0, 0.4), 1000)
            rows.append(self._mk(t0, "EXT_FAMILY", aid, credit, "IMPS"))
            for frac in (0.5, 0.5):
                tt = t0 + timedelta(minutes=float(rng.uniform(10, 40)))
                rows.append(self._mk(tt, aid, None, round_amount(min(credit * frac, limit), 500),
                                     "ATM_WDL", atm_i=atm_i))
        elif style == "dormant_payout":
            credit = round_amount(rng.lognormal(11.0, 0.3), 1000)  # insurance
            rows.append(self._mk(t0, "EXT_INSURER", aid, credit, "NEFT"))
            tt = t0 + timedelta(hours=float(rng.uniform(1, 20)))
            rows.append(self._mk(tt, aid, None, round_amount(min(limit, credit * 0.4), 500),
                                 "ATM_WDL", atm_i=atm_i))
        elif style == "emergency":
            credit = round_amount(rng.lognormal(10.4, 0.5), 1000)
            rows.append(self._mk(t0, "EXT_FAMILY", aid, credit, "IMPS"))
            for _ in range(int(rng.integers(2, 5))):
                tt = t0 + timedelta(minutes=float(rng.uniform(5, 60)))
                rows.append(self._mk(tt, aid, None, round_amount(min(limit, credit * 0.3), 500),
                                     "ATM_WDL", atm_i=atm_i))
        else:  # round_transfer
            amt = round_amount(rng.lognormal(10.0, 0.3), 5000)  # suspiciously round
            rows.append(self._mk(t0, "EXT_0001", aid, amt, "IMPS"))
            rows.append(self._mk(t0 + timedelta(minutes=float(rng.uniform(3, 20))),
                                 aid, f"PAYEE_{int(rng.integers(0,6000)):04d}", amt * 0.9, "IMPS"))

    # ---- fraud chains --------------------------------------------------
    def _delay(self, mule_i):
        p, rng = self.p, self.rng
        if self.behaves_legit[mule_i]:
            return timedelta(minutes=float(rng.gamma(p.legit_delay_shape, p.legit_delay_scale)))
        return timedelta(minutes=float(rng.gamma(p.mule_delay_shape, p.mule_delay_scale)))

    def _mule_cashout_atm(self, mule_i, typology):
        """Cash-out terminal for a mule, with the 58/18/17/7 location noise."""
        p, rng = self.p, self.rng
        district = self.home_district[mule_i]
        exp_cell = self.expected_cell[mule_i]
        if typology in ("cross_district",):
            return self._pick_other_district(district)
        if self.behaves_legit[mule_i]:
            return self._pick_from_pref(self.pref[mule_i])
        u = rng.random()
        if u < p.loc_noise_expected:
            return self._pick_in_cell(self.pref[mule_i], exp_cell)
        elif u < p.loc_noise_expected + p.loc_noise_ring:
            ring = self.ring_of.get(mule_i)
            if ring is not None:
                return self._pick_from_pref(self.ring_pref[ring])
            return self._pick_in_cell(self.pref[mule_i], exp_cell)
        elif u < p.loc_noise_expected + p.loc_noise_ring + p.loc_noise_district:
            return self._pick_in_district(district, exclude_cell=exp_cell)
        else:
            return self._pick_other_district(district)

    def _cashout(self, rows, mule_i, amount, ts, typology, case_id, hop, channel="ATM_WDL"):
        atm_i = self._mule_cashout_atm(mule_i, typology)
        limit = int(self.a_limit[atm_i])
        amt = round_amount(min(amount, limit), 500)
        # src is the withdrawing mule (the debited account); dst stays NULL --
        # that (plus atm_id) is what encodes a cash-out.
        rows.append(self._mk(ts, f"ACC_{mule_i:05d}", None, amt, channel, atm_i=atm_i,
                             tainted=True, case_id=case_id, hop=hop))

    def build_fraud(self):
        rng, p = self.rng, self.p
        rows: list[dict] = []
        complaints: list[dict] = []
        typ_names = list(p.fraud_typology_mix)
        typ_w = np.array(list(p.fraud_typology_mix.values()))
        typ_w /= typ_w.sum()
        used_ack = set()

        for _ in range(p.n_complaints):
            typology = rng.choice(typ_names, p=typ_w)
            victim = int(rng.choice(self.legit_idx))
            v_id = f"ACC_{victim:05d}"
            amount = round_amount(rng.lognormal(10.9, 0.7), 1000)  # ~₹15k-₹300k
            t0 = self._random_ts()
            # seed edge: victim -> first mule
            m1 = int(rng.choice(self.mule_idx))
            seed_ch = rng.choice(["UPI", "IMPS"], p=[0.6, 0.4])
            seed = self._mk(t0, v_id, f"ACC_{m1:05d}", amount, seed_ch,
                            tainted=True, case_id=None, hop=1,
                            dst_vpa=f"mule{m1}@upi")
            rows.append(seed)
            seed_txn_id = seed["txn_id"]
            case_id = seed_txn_id  # case keyed by its seed txn
            seed["label_case_id"] = case_id

            self._grow_chain(rows, typology, case_id, m1, amount, seed["ts"])

            # complaint filed after a lognormal reporting delay
            delay_min = float(rng.lognormal(p.report_delay_lognorm_mu,
                                            p.report_delay_lognorm_sigma))
            filed = seed["ts"] + timedelta(minutes=delay_min)
            while True:
                ack = "".join(str(int(d)) for d in rng.integers(0, 10, size=14))
                if ack not in used_ack:
                    used_ack.add(ack)
                    break
            complaints.append(
                {
                    "ack_no": ack,
                    "filed_ts": filed,
                    "victim_account": v_id,
                    "seed_txn_id": seed_txn_id,
                    "disputed_amount": amount,
                    "fraud_category": typology,
                    "reporting_delay_min": round(delay_min, 1),
                }
            )

        self.fraud_rows = rows
        self.complaints = pd.DataFrame(complaints)
        return rows

    def _grow_chain(self, rows, typology, case_id, m1, amount, t_seed):
        """Typology-specific layering & cash-out from first mule m1."""
        rng, p = self.rng, self.p

        def forward_frac():
            return float(rng.uniform(0.70, 0.98))

        if typology in ("fast_fanout", "pos_out", "smish_burst"):
            ch = "POS" if typology == "pos_out" else "ATM_WDL"
            k = int(rng.integers(3, 7)) if typology != "smish_burst" else int(rng.integers(6, 11))
            legs = [int(rng.choice(self.mule_idx)) for _ in range(k)]
            share = amount / k
            for leg in legs:
                t_recv = t_seed + self._delay(m1)
                rows.append(self._mk(t_recv, f"ACC_{m1:05d}", f"ACC_{leg:05d}",
                                     share, rng.choice(["IMPS", "UPI"]),
                                     tainted=True, case_id=case_id, hop=2))
                # 22% of recipients are innocent downstream (do nothing)
                if rng.random() < p.innocent_downstream_frac:
                    continue
                t_out = t_recv + self._delay(leg)
                co_ch = "POS" if (typology in ("pos_out",) or
                                  (typology == "smish_burst" and rng.random() < 0.6)) else ch
                self._cashout(rows, leg, share * forward_frac(), t_out,
                              typology, case_id, hop=3, channel=co_ch)

        elif typology == "layered":
            cur, cur_amt, t_cur, hop = m1, amount, t_seed, 1
            while hop < config.MAX_HOPS:
                nxt = int(rng.choice(self.mule_idx))
                t_cur = t_cur + self._delay(cur)
                cur_amt = cur_amt * forward_frac()
                hop += 1
                rows.append(self._mk(t_cur, f"ACC_{cur:05d}", f"ACC_{nxt:05d}",
                                     cur_amt, rng.choice(["IMPS", "NEFT", "UPI"]),
                                     tainted=True, case_id=case_id, hop=hop))
                cur = nxt
                if rng.random() < 0.45:
                    break
            if rng.random() >= p.innocent_downstream_frac:
                self._cashout(rows, cur, cur_amt, t_cur + self._delay(cur),
                              typology, case_id, hop=hop + 1)

        elif typology == "ring_reuse":
            ring = self.ring_of.get(m1)
            members = self.ring_members.get(ring, [m1]) if ring else [m1]
            share = amount / max(1, len(members))
            for leg in members:
                t_recv = t_seed + self._delay(m1)
                if leg != m1:
                    rows.append(self._mk(t_recv, f"ACC_{m1:05d}", f"ACC_{leg:05d}",
                                         share, "IMPS", tainted=True,
                                         case_id=case_id, hop=2))
                self._cashout(rows, leg, share * forward_frac(),
                              t_recv + self._delay(leg), typology, case_id, hop=3)

        elif typology == "split_hold":
            # one leg cashed now, one leg held then cashed within the window
            t_recv = t_seed + self._delay(m1)
            self._cashout(rows, m1, amount * 0.5, t_recv, typology, case_id, hop=2)
            held_leg = int(rng.choice(self.mule_idx))
            t_hold = t_recv + timedelta(minutes=float(rng.uniform(20, 90)))
            rows.append(self._mk(t_hold, f"ACC_{m1:05d}", f"ACC_{held_leg:05d}",
                                 amount * 0.45, "IMPS", tainted=True,
                                 case_id=case_id, hop=2))
            self._cashout(rows, held_leg, amount * 0.4,
                          t_hold + self._delay(held_leg), typology, case_id, hop=3)

        elif typology == "cross_district":
            t_out = t_seed + self._delay(m1)
            self._cashout(rows, m1, amount * forward_frac(), t_out,
                          typology, case_id, hop=2)  # forced other-district

        elif typology == "no_cashout":
            # money keeps moving as onward transfers, never cashed out
            cur, cur_amt, t_cur, hop = m1, amount, t_seed, 1
            for _ in range(int(rng.integers(1, 3))):
                nxt = int(rng.choice(self.mule_idx))
                t_cur = t_cur + self._delay(cur)
                cur_amt *= forward_frac()
                hop += 1
                rows.append(self._mk(t_cur, f"ACC_{cur:05d}", f"ACC_{nxt:05d}",
                                     cur_amt, rng.choice(["IMPS", "NEFT"]),
                                     tainted=True, case_id=case_id, hop=hop))
                cur = nxt

    # ---- assemble ------------------------------------------------------
    def build_transactions(self):
        rows = list(self.bg_rows) + list(self.fraud_rows)
        df = pd.DataFrame(rows)
        df = df.sort_values("ts", kind="stable").reset_index(drop=True)
        self.transactions = df
        return df


# --------------------------------------------------------------------------
# validation & driver
# --------------------------------------------------------------------------
def validate(atms, accounts, transactions, complaints):
    print("\n--- validation ---")
    atm_ids = set(atms["atm_id"])
    tx = transactions
    is_cash = tx["channel"].isin(CASHOUT_CHANNELS)

    # atm_id exists in ATM table
    used = set(tx.loc[tx["atm_id"].notna(), "atm_id"])
    missing = used - atm_ids
    assert not missing, f"{len(missing)} atm_ids not in ATM table"

    # atm_id non-null iff cash-out channel
    assert (tx["atm_id"].notna() == is_cash).all(), "atm_id/cash-out mismatch"
    # dst_account null iff cash-out
    assert (tx["dst_account"].isna() == is_cash).all(), "dst_account/cash-out mismatch"

    # every seed_txn resolves and filed_ts >= seed.ts
    txn_by_id = tx.set_index("txn_id")["ts"]
    for _, c in complaints.iterrows():
        assert c["seed_txn_id"] in txn_by_id.index, f"seed {c['seed_txn_id']} missing"
        assert c["filed_ts"] >= txn_by_id[c["seed_txn_id"]], "filed before seed"

    # cash-out split
    co = tx[is_cash]
    fraud_co = int(co["label_is_tainted"].sum())
    legit_co = int((~co["label_is_tainted"]).sum())
    print(f"cash-outs: total={len(co)}  legit={legit_co}  fraud={fraud_co}  "
          f"ratio legit/fraud={legit_co / max(1, fraud_co):.1f}x")
    assert legit_co > fraud_co * 3, "fraud cash-outs not a small minority -- too easy"
    print("validation OK")


def run(params: Params):
    out = config.data_dir(sealed=params.is_sealed)
    os.makedirs(out, exist_ok=True)
    g = Generator(params)
    print(f"[gen] {'SEALED' if params.is_sealed else 'TRAIN'} corpus -> {out}")
    g.build_accounts()
    g.build_background()
    g.build_fraud()
    tx = g.build_transactions()

    # ship accounts WITHOUT internal columns (R3)
    g.accounts.to_parquet(os.path.join(out, "accounts.parquet"), index=False)
    g.accounts_internal.to_parquet(os.path.join(out, "_accounts_internal.parquet"),
                                   index=False)
    tx.to_parquet(os.path.join(out, "transactions.parquet"), index=False)
    g.complaints.to_parquet(os.path.join(out, "complaints.parquet"), index=False)
    # atms shared: copy reference into sealed dir too for self-containment
    g.atms.to_parquet(os.path.join(out, "atms.parquet"), index=False)

    n_mule = int(g.accounts["label_is_mule"].sum())
    n_ring = g.accounts["label_ring_id"].nunique(dropna=True)
    unobs = 1.0 - tx["observed"].mean()
    print(f"accounts={len(g.accounts)}  mules={n_mule}  rings={n_ring}")
    print(f"transactions={len(tx)}  complaints={len(g.complaints)}")
    print(f"unobserved rate={unobs:.3f}")
    print("typology mix:\n" + g.complaints["fraud_category"].value_counts().to_string())
    validate(g.atms, g.accounts, tx, g.complaints)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sealed", action="store_true", help="build the shifted sealed corpus")
    args = ap.parse_args()
    params = Params.sealed() if args.sealed else Params()
    run(params)


if __name__ == "__main__":
    main()
