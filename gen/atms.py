"""
gen/atms.py -- the ATM / cash-out terminal table.

~700 terminals scattered radially around the four district centroids so they
cluster in town centres with a highway tail (Gamma radius). Everything here is
SYNTHETIC. To swap in real terminals later, run the Overpass query in the TODO
below against OpenStreetMap and map the fields.

Run:  python -m gen.atms
Output: data/atms.parquet  (shared by both the train and sealed corpora)
"""
from __future__ import annotations

import math

import h3
import numpy as np
import pandas as pd

import config

# TODO(real-data): replace the synthetic scatter with an OpenStreetMap pull.
# Overpass query (one district shown; loop over the four):
#   [out:json][timeout:90];
#   area["name"="Pune"]["boundary"="administrative"]->.a;
#   ( node["amenity"="atm"](area.a); node["amenity"="bank"]["atm"="yes"](area.a); );
#   out body;
# Map node lat/lon -> lat/lon; operator tag -> operator; addr:postcode -> pin.

# RBI cash-dispenser / WLA deployment share (approximate, normalised below).
OPERATOR_SHARE = {
    "SBI": 0.28,
    "INDICASH": 0.16,   # Tata Indicash (white-label)
    "HITACHI": 0.12,    # Hitachi Money Spot (white-label)
    "HDFC": 0.10,
    "PUNB": 0.09,
    "BARB": 0.09,
    "AXIS": 0.08,
    "AGS": 0.08,        # AGS Transact (white-label)
}

# Rough PIN prefix per district (Jharkhand, stored as string -- leading zeros
# are meaningless here but the column MUST stay string so nobody int-casts it).
PIN_BASE = {
    "Pune City": 411001,
    "Pimpri-Chinchwad": 411018,
    "Hinjawadi": 411057,
    "Hadapsar": 411028,
}

# How many terminals per district (bigger cities get more).
COUNT_PER_DISTRICT = {
    "Pune City": 220,
    "Pimpri-Chinchwad": 210,
    "Hinjawadi": 120,
    "Hadapsar": 150,
}

PER_TXN_LIMITS = [10000, 20000, 25000]
PER_TXN_LIMIT_W = [0.35, 0.45, 0.20]


def _hourly_weight(rng: np.random.Generator, night_active: bool) -> list[float]:
    """24-element weight vector, bimodal around 11:30 and 18:30 IST."""
    hours = np.arange(24)
    # Two Gaussian bumps (late morning + early evening) + small floor.
    w = (
        np.exp(-0.5 * ((hours - 11.5) / 2.3) ** 2)
        + 0.95 * np.exp(-0.5 * ((hours - 18.5) / 2.1) ** 2)
        + 0.03
    )
    if not night_active:
        # Suppress 00:00-05:59 for non-24h machines.
        w[0:6] *= 0.05
    # Small per-ATM jitter so no two machines are identical.
    w *= rng.uniform(0.9, 1.1, size=24)
    w = w / w.sum()
    return [round(float(x), 6) for x in w]


def _scatter(rng: np.random.Generator, clat: float, clon: float) -> tuple[float, float]:
    """Radial offset with a Gamma(1.7, 2.1) km radius -> town-centre cluster
    with a long highway tail."""
    r_km = rng.gamma(shape=1.7, scale=2.1)
    theta = rng.uniform(0.0, 2.0 * math.pi)
    dlat = (r_km * math.cos(theta)) / 111.32
    dlon = (r_km * math.sin(theta)) / (111.32 * math.cos(math.radians(clat)))
    return clat + dlat, clon + dlon


def build_atms() -> pd.DataFrame:
    rng = np.random.default_rng(config.SEED)
    ops = list(OPERATOR_SHARE.keys())
    op_p = np.array(list(OPERATOR_SHARE.values()), dtype=float)
    op_p = op_p / op_p.sum()

    rows: list[dict] = []
    for district, meta in config.DISTRICTS.items():
        clat, clon = meta["lat"], meta["lon"]
        code = meta["code"]
        n = COUNT_PER_DISTRICT[district]
        for i in range(1, n + 1):
            lat, lon = _scatter(rng, clat, clon)
            night = bool(rng.random() < 0.55)
            pin = str(PIN_BASE[district] + int(rng.integers(0, 60)))
            rows.append(
                {
                    "atm_id": f"ATM_{code}_{i:04d}",
                    "lat": round(float(lat), 6),
                    "lon": round(float(lon), 6),
                    "operator": str(rng.choice(ops, p=op_p)),
                    "district": district,
                    "state": config.STATE,
                    "pin": pin,  # string on purpose
                    "cell": h3.latlng_to_cell(lat, lon, config.H3_RES_CELL),
                    "cell_fine": h3.latlng_to_cell(lat, lon, config.H3_RES_FINE),
                    "per_txn_limit": int(
                        rng.choice(PER_TXN_LIMITS, p=PER_TXN_LIMIT_W)
                    ),
                    "is_standalone": bool(rng.random() < 0.40),
                    "night_active": night,
                    "hourly_weight": _hourly_weight(rng, night),
                    # Generator-only structural attractiveness (footfall proxy).
                    "_attract": round(float(rng.beta(2, 5)), 4),
                }
            )
    df = pd.DataFrame(rows)
    return df


def main() -> None:
    import os

    df = build_atms()
    path = os.path.join(config.DATA_DIR, "atms.parquet")
    df.to_parquet(path, index=False)
    print(f"[atms] wrote {len(df)} terminals -> {path}")
    print(df.groupby("district").size().to_string())
    print("operators:\n" + df["operator"].value_counts().to_string())
    print(
        f"standalone={df['is_standalone'].mean():.2f} "
        f"night_active={df['night_active'].mean():.2f} "
        f"distinct cells(res7)={df['cell'].nunique()}"
    )


if __name__ == "__main__":
    main()
