"""
config.py -- all tunables, district centroids, and decision thresholds.

This is the single source of truth for constants shared across the whole
pipeline (generator, features, models, API). Nothing here is secret: the
generator's *parameters* live in gen/params.py and must NOT be read by the ML
side. This file is safe for everyone.

All timestamps in this project are naive `datetime` objects representing IST
(see R6). Do not introduce tz-aware datetimes.
"""
from __future__ import annotations

from datetime import datetime

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 26184  # SIH problem statement number, used as the master RNG seed

# --------------------------------------------------------------------------
# Geography -- four real districts in Jharkhand, the well-documented
# "Jamtara belt" cybercrime cash-out region. Centroids are (lat, lon).
# --------------------------------------------------------------------------
STATE = "Maharashtra"

DISTRICTS: dict[str, dict] = {
    "Pune City":        {"lat": 18.5204, "lon": 73.8567, "code": "PNC"},
    "Pimpri-Chinchwad": {"lat": 18.6279, "lon": 73.8009, "code": "PCH"},
    "Hinjawadi":        {"lat": 18.5913, "lon": 73.7389, "code": "HJW"},
    "Hadapsar":         {"lat": 18.5089, "lon": 73.9260, "code": "HDP"},
}

DISTRICT_CENTROIDS: dict[str, tuple[float, float]] = {
    name: (d["lat"], d["lon"]) for name, d in DISTRICTS.items()
}

# Police station covering each district's main cash-out cluster. Alerts route
# here (to the SHO), keyed by district -- the officer-facing routing table.
JURISDICTION_SHO: dict[str, dict] = {
    "Pune City":        {"station": "Pune City Cyber PS", "sho_phone": "+91-9430000201"},
    "Pimpri-Chinchwad": {"station": "Pimpri-Chinchwad Cyber PS", "sho_phone": "+91-9430000202"},
    "Hinjawadi":        {"station": "Hinjawadi IT-Park PS", "sho_phone": "+91-9430000203"},
    "Hadapsar":         {"station": "Hadapsar Cyber PS", "sho_phone": "+91-9430000204"},
}

# --------------------------------------------------------------------------
# Time horizons (minutes). M1 is a discrete-time hazard model scored at each
# of these horizons to build a hazard curve. See R4/ml/train.py.
# --------------------------------------------------------------------------
HORIZONS: list[int] = [10, 20, 30, 45, 60]

# --------------------------------------------------------------------------
# Money-flow graph reconstruction (graphx/trail.py)
# --------------------------------------------------------------------------
MAX_HOPS = 4                # cap on BFS depth from the seed transaction
MIN_EDGE_FRACTION = 0.05    # ignore edges worth < 5% of the seed amount
CANDIDATE_CAP = 50          # max ATM candidates per account (graphx/candidates.py)

# --------------------------------------------------------------------------
# H3 spatial resolution (R5). v4 API: h3.latlng_to_cell(lat, lon, res).
# res 7 ~ 5 km^2 operational cell; res 8 ~ 0.5 km^2 fine tier.
# --------------------------------------------------------------------------
H3_RES_CELL = 7
H3_RES_FINE = 8

# --------------------------------------------------------------------------
# Abstention thresholds (R5). The system refuses to name a single terminal
# when the terminal distribution is too flat to justify it.
# --------------------------------------------------------------------------
ABSTAIN_MIN_P_TERMINAL = 0.22  # need >= this on the top terminal to emit one
ABSTAIN_MAX_ENTROPY = 2.60     # nats; above this the distribution is too flat
CELL_MIN_P = 0.30              # need >= this on top cell to emit a cell

# --------------------------------------------------------------------------
# Decision tiers (plain rules, NOT ML -- an officer must read why an alert
# fired). See api/main.py decision layer.
# --------------------------------------------------------------------------
TIER_RED_P = 0.60          # P(cash-out <= 10m) at/above this -> RED
# Amount floor for a FREEZE. Set to a realistic per-account figure: after
# fan-out/layering the money is split across mules, so a single account rarely
# holds the full disputed sum -- but a 60%+ imminent cash-out of even ~15k is
# worth a freeze (freezing is cheap; letting it clear is not).
TIER_RED_AMOUNT = 5000     # AND tainted amount >= this (rupees) -> RED
TIER_AMBER_P = 0.30        # middle band -> AMBER
ALERT_DEDUP_MINUTES = 15   # one alert per account per 15 minutes

# --------------------------------------------------------------------------
# Columns that must NEVER reach the model. The feature loader drops these on
# sight (belt-and-braces alongside the `label_` prefix rule, R2/R3).
# home_lat/home_lon are the generator's ground-truth home location.
# --------------------------------------------------------------------------
INTERNAL_COLUMNS = {"archetype", "home_lat", "home_lon", "home_cell"}
LABEL_PREFIX = "label_"

# --------------------------------------------------------------------------
# Corpus time window. 180-day window ending 2026-08-31 (naive IST).
# --------------------------------------------------------------------------
CORPUS_END = datetime(2026, 8, 31, 23, 59, 59)
CORPUS_DAYS = 180
CORPUS_START = datetime(2026, 3, 4, 0, 0, 0)  # CORPUS_END - 180 days (approx)

# --------------------------------------------------------------------------
# Filesystem layout. Data and model artefacts live under these dirs.
# --------------------------------------------------------------------------
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_SEALED_DIR = os.path.join(BASE_DIR, "data_sealed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

for _d in (DATA_DIR, DATA_SEALED_DIR, MODEL_DIR, REPORTS_DIR):
    os.makedirs(_d, exist_ok=True)


def data_dir(sealed: bool = False) -> str:
    """Return the data directory for the training corpus or the sealed one."""
    return DATA_SEALED_DIR if sealed else DATA_DIR
