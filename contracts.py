"""
contracts.py -- the Pydantic data contracts for the whole system.

FREEZE THIS FIRST. Every other module (generator, models, API, web console)
codes against these shapes. If a shape needs to change, change it here and let
the type checker find every call site.

R6 (naive IST timestamps): datetimes are stored naive and are understood to be
IST everywhere internally. The ONLY place +05:30 appears is JSON serialisation,
handled by the `IST` annotated type below -- so an API consumer sees a correct
offset while the pipeline never juggles tz-aware objects.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, PlainSerializer

# --------------------------------------------------------------------------
# Naive-IST datetime that serialises to JSON with a +05:30 offset appended.
# Internally it is a plain naive datetime (R6).
# --------------------------------------------------------------------------
def _to_ist_string(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + "+05:30"


IST = Annotated[datetime, PlainSerializer(_to_ist_string, return_type=str, when_used="json")]

Tier = Literal["RED", "AMBER", "GREY"]
Granularity = Literal["terminal", "cell", "district"]


# ==========================================================================
# Complaint / case
# ==========================================================================
class Case(BaseModel):
    ack_no: str = Field(..., description="14-digit CFCFRMS acknowledgement number")
    filed_ts: IST
    victim_account: str
    seed_txn_id: str = Field(..., description="the one disputed transaction")
    disputed_amount: float
    fraud_category: str
    reporting_delay_min: float = Field(..., description="minutes from fraud to filing")
    status: str = "open"


# ==========================================================================
# Money-flow trail (graphx/trail.py)
# ==========================================================================
class TrailNode(BaseModel):
    account_id: str
    hop_from_victim: int
    tainted_amount: float = Field(..., description="stolen money attributed to this node")
    is_victim: bool = False
    is_leaf: bool = Field(False, description="currently holds funds -- a forecast target")
    mule_score: Optional[float] = None
    lat: Optional[float] = Field(None, description="account KYC centroid, for plotting on the map")
    lon: Optional[float] = None


class TrailEdge(BaseModel):
    txn_id: str
    src_account: str
    dst_account: Optional[str] = Field(None, description="null iff this edge is a cash-out")
    amount: float
    tainted_amount: float
    ts: IST
    channel: str
    hop: int
    observed: bool = True
    atm_id: Optional[str] = None


class Trail(BaseModel):
    ack_no: str
    seed_txn_id: str
    as_of: IST
    total_tainted: float
    nodes: list[TrailNode]
    edges: list[TrailEdge]
    leaf_accounts: list[str] = Field(default_factory=list)


# ==========================================================================
# M1 -- cash-out hazard (probability + timing from one model)
# ==========================================================================
class HazardPoint(BaseModel):
    horizon_min: int
    p_cumulative: float = Field(..., description="P(cash-out within (t, t+horizon])")


class CashoutPrediction(BaseModel):
    account_id: str
    p_10m: float
    p_20m: float
    p_30m: float
    p_45m: float
    p_60m: float
    window_minutes: tuple[int, int] = Field(
        ..., description="steepest-rise interval read off the hazard curve"
    )
    hazard: list[HazardPoint]
    severity: str = Field(..., description="imminent | soon | later | unlikely")


# ==========================================================================
# M2 -- channel (exactly four classes)
# ==========================================================================
class ChannelPrediction(BaseModel):
    ATM: float
    POS: float
    onward_transfer: float
    dormant: float


# ==========================================================================
# M3 -- hierarchical location with abstention (R5)
# ==========================================================================
class LocationCandidate(BaseModel):
    atm_id: str
    lat: float
    lon: float
    district: str
    cell: str
    cell_fine: str
    p: float = Field(..., description="calibrated probability this terminal is used")
    sources: list[str] = Field(default_factory=list, description="C1..C5 that proposed it")
    distance_km: Optional[float] = None


class CellPrediction(BaseModel):
    cell: str
    lat: float
    lon: float
    district: str
    p: float
    n_terminals: int = 0


class DistrictPrediction(BaseModel):
    district: str
    lat: float
    lon: float
    p: float


class LocationPrediction(BaseModel):
    account_id: str
    granularity: Granularity = Field(
        ..., description="the finest level actually emitted"
    )
    # These two fields carry the design: when the distribution is too flat the
    # system refuses to name a terminal and says so, rather than guessing.
    abstained_at: Optional[Granularity] = Field(
        None, description="level at which the system refused to be more precise"
    )
    abstain_reason: Optional[str] = None

    districts: list[DistrictPrediction] = Field(default_factory=list)
    cells: list[CellPrediction] = Field(default_factory=list)
    terminals: list[LocationCandidate] = Field(default_factory=list)

    entropy: float = Field(..., description="entropy (nats) over the terminal distribution")
    candidate_recall_note: Optional[str] = None


# ==========================================================================
# Decision layer (plain rules, not ML)
# ==========================================================================
class Recommendation(BaseModel):
    tier: Tier
    action: str = Field(..., description="FREEZE | DISPATCH | MONITOR")
    reason: str = Field(..., description="human-readable rule trace")
    freeze_account: Optional[str] = None
    freeze_deadline: Optional[IST] = None
    target_station: Optional[str] = None
    target_district: Optional[str] = None
    target_cell: Optional[str] = None
    tainted_amount: float = 0.0
    p_cashout_10m: float = 0.0


# ==========================================================================
# Top-level prediction object -- what /predict returns
# ==========================================================================
class Prediction(BaseModel):
    ack_no: str
    account_id: str
    as_of: IST
    cashout: CashoutPrediction
    channel: ChannelPrediction
    location: LocationPrediction
    recommendation: Recommendation


# ==========================================================================
# Alerting (simulated SMS / email / webhook)
# ==========================================================================
class Alert(BaseModel):
    alert_id: str
    created_ts: IST
    tier: Tier
    channel: Literal["SMS", "EMAIL", "WEBHOOK"]
    recipient: str
    ack_no: str
    account_id: str
    district: Optional[str] = None
    cell: Optional[str] = None
    subject: str
    body: str
    status: Literal["queued", "sent", "failed"] = "queued"
    dedup_key: str


# ==========================================================================
# Triage queue
# ==========================================================================
class QueueItem(BaseModel):
    ack_no: str
    victim_account: str
    filed_ts: IST
    disputed_amount: float
    fraud_category: str
    top_account: Optional[str] = Field(None, description="account most at risk right now")
    p_cashout_10m: float = 0.0
    tier: Tier = "GREY"
    imminence: float = Field(0.0, description="sort key: higher = act sooner")
    minutes_since_filed: float = 0.0
    golden_hour_remaining_min: float = Field(
        0.0, description="minutes left in the 60-min golden hour"
    )
