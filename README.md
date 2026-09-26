# SIH26184 — Predictive Analytics for Cybercrime Cash-out Forecasting

**Ministry of Home Affairs / I4C, CIS Division.** A base prototype for a
4-person student team to build on. It runs end to end: a cybercrime complaint
naming a victim and one disputed transaction arrives; the system reconstructs
where the money went, then forecasts — for each account now holding stolen
funds — **whether** it will be converted to cash, **when**, through which
**channel**, and **where**. The operational output is a time-bounded **freeze
recommendation** plus a **ranked geographic forecast**.

> All data is **synthetic by design** (see *Known limitations*). Nothing here
> touches real financial records.

## The four I4C deliverables

1. **Predictive analytics engine** — M1 (cash-out hazard), M2 (channel), M3
   (location hotspots). `ml/`
2. **GIS risk heatmap dashboard with filtering** — inline-SVG map, no tiles.
   `web/index.html`, `GET /hotspots`
3. **Law-enforcement interface** — triage queue, per-case intelligence,
   freeze/dispatch actions. `api/main.py`, `web/`
4. **Alert & notification system (SMS / email / API)** — simulated, with a
   delivery log. `POST /alerts/dispatch`, `GET /alerts`

## Quick start (Windows / PowerShell)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy scikit-learn xgboost networkx fastapi uvicorn pydantic pyarrow h3 matplotlib pytest
```

Then, from the `sih26184/` folder:

```bash
python -m gen.atms                 # ATM table          -> data/atms.parquet
python -m gen.generator            # training corpus    -> data/
python -m gen.generator --sealed   # sealed test corpus -> data_sealed/
python -m ml.train                 # fit M1, M2, M3     -> models/
python -m ml.evaluate              # 4 splits + plots   -> reports/
python -m pytest tests/ -q         # leakage suite (5 tests)
python demo.py                     # headless end-to-end replay
python -m uvicorn api.main:app --port 8000   # then open http://localhost:8000
```

Or with `make`: `make all && make test && make demo && make api`.

## New here? Read the manual

**[MANUAL.md](MANUAL.md)** explains the whole project in plain language (no ML
background needed): what it does, every screen, a glossary, the numbers for
judges, and a 3-minute demo script. Start there.

## What each package does

| path | role |
|---|---|
| `config.py` | all tunables, district centroids, thresholds |
| `contracts.py` | Pydantic models — the frozen data contract everything codes against |
| `gen/atms.py` | ~700 synthetic ATM terminals (h3 res-7/8 cells) |
| `gen/params.py` | every generator dial; `.sealed()` = the shifted test corpus. **The ML side must not import this.** |
| `gen/generator.py` | the synthetic corpus (accounts, ~250k transactions, complaints) |
| `graphx/trail.py` | time-respecting BFS money-flow reconstruction |
| `graphx/features.py` | nine graph features + a logistic mule-node score |
| `graphx/candidates.py` | 5-source candidate ATM generation (C1–C5) |
| `ml/features.py` | as-of feature store (`searchsorted` cuts) + all feature functions |
| `ml/panel.py` | `(account, as_of)` training-row construction |
| `ml/train.py` | M1 (hazard), M2 (channel), M3 (location ranker) |
| `ml/predict.py` | inference → `Prediction`, with hierarchical **abstention** |
| `ml/evaluate.py` | 4 splits, 4 baselines, reliability & coverage plots, lead time |
| `api/main.py` | FastAPI console backend + plain-rules decision layer |
| `web/index.html` | single-page officer console (queue • heatmap • case • replay) |
| `tests/test_leakage.py` | mechanical leakage guarantees (R1/R2/R3) |

## Models: what runs in the product vs. what is an offline experiment

| Layer | What exists in the code | Runs in the console / API? |
|---|---|---|
| **Traditional ML** | M1 cash-out timing, M2 channel, M3 ATM/cell ranker, all XGBoost (`ml/train.py`) | **Yes**, every prediction |
| **Graph-based analysis** | money-trail tracing (`graphx/trail.py`), nine graph features + a logistic-regression mule score (`graphx/features.py`), candidate ATM generation C1-C5 (`graphx/candidates.py`) | **Yes**, feeds M1/M2/M3 |
| **GNN** | GraphSAGE and heterogeneous (account + ATM) GraphSAGE baselines (`graphx/gnn_baseline.py`, `graphx/hetero_gnn_baseline.py`) | **No**, offline experiments; need PyTorch + PyG. Also offline: a graph ATM ranker (`graphx/sprint9b_atm_ranker.py`, did not beat the candidate baseline) and a fusion prototype (`graphx/sprint10_fusion.py`) |
| **Temporal graph model** | snapshot-based temporal GraphSAGE with transaction-age edge features (`graphx/tgnn_baseline.py`, `graphx/tgnn_activity_train.py`, `graphx/temporal_hetero_gnn.py`, `graphx/next_movement_gnn.py`). This is a *temporal GNN*, not a TGN (there is no per-node memory module) | **No**, offline experiments; recorded ROC-AUC 0.51-0.79 (best: temporal heterogeneous GNN, 0.79, measured on an earlier version of the data), below M1's 0.913 |
| **Clustering** | **Not implemented.** The Mule Network roles (casher, distributor, collector, relay) come from fixed if/else thresholds in `api/main.py:_archetype`, a rule-based grouping, not a clustering algorithm | n/a |
| **Risk / decision layer** | plain rules: RED / AMBER / GREY tier + FREEZE / MONITOR (`api/main.py:make_recommendation`), location abstention (`ml/predict.py`) | **Yes** |

The benchmark numbers in `graphx/sprint12_final_validation.py` are typed into
the script from earlier runs; running it does not recompute them.

## Pipeline: from data to the investigator

```
DATA                 gen/generator.py, gen/atms.py (synthetic corpus + complaints)
  ↓
PREPROCESSING        ml/features.py FeatureContext (as-of index, label_* columns dropped)
  ↓
FEATURE ENGINEERING  ml/features.py + graphx/trail.py + graphx/features.py + graphx/candidates.py
  ↓
ML / GRAPH MODEL     ml/predict.py: M1, M2, M3 (XGBoost) on the graph + account features
  ↓
RISK / PREDICTION    probabilities per horizon, channel mix, ranked cells/ATMs (with abstention)
  ↓
EXPLANATION          GET /explain, the "Why" panel and the evidence list in web/index.html
  ↓
ALERT                api/main.py make_recommendation (RED/AMBER/GREY) + POST /alerts/dispatch
  ↓
INVESTIGATOR         web/index.html console, Case centre (tickets), Mule Network
```

The GNN / temporal GNN experiments sit outside this pipeline today.

## How each model works (input → process → output)

**Money-trail tracing** (`graphx/trail.py`)
`victim's disputed transaction + transfers seen so far` → follow the money forward hop by hop, only along transfers that happened after the money arrived → `the accounts now holding stolen money`.
*Plain words: it follows the money from the victim through each account it was sent to.*

**Graph features + mule score** (`graphx/features.py`)
`an account's recent transfers in and out` → count senders/receivers, how fast money leaves, how concentrated the outflow is, ring history; a logistic regression turns these into one score → `nine graph features, used as inputs to M1/M2`.
*Plain words: it describes how an account behaves inside the money network. The score is an input to the models, not a verdict that the account is a mule.*

**M1 cash-out timing** (XGBoost, `ml/train.py`)
`account + graph features at the moment of prediction` → one classifier asked "cash-out within 10 / 20 / 30 / 45 / 60 min?" → `a probability for each horizon (the hazard curve)`.
*Plain words: how likely the money is to be withdrawn as cash soon, and roughly when.*

**M2 channel** (XGBoost multi-class)
`the same features` → pick between ATM, POS, onward transfer, dormant → `a probability for each channel`.
*Plain words: whether the money is likely to leave as cash, or keep moving to another account.*

**M3 location ranker** (XGBoost, with abstention)
`candidate ATMs from the account's past withdrawals, its ring, its KYC area and known fraud ATMs` → score each candidate, then report at the level it is confident about (terminal, ~5 km² cell or district) → `a ranked list of likely areas`.
*Plain words: where the cash is most likely to be withdrawn. It is a ranked estimate, not a guarantee.*

**Decision rules** (`api/main.py`)
`M1 probability of cash-out in 10 min + amount of stolen money held` → fixed thresholds → `RED (freeze now) / AMBER (watch) / GREY (keep tracing)` and the police station to alert.
*Plain words: the risk tier is a rule on top of the probability, so a RED tier is a priority for review, not proof of crime.*

**GNN (offline experiment)** (`graphx/gnn_baseline.py`, `graphx/hetero_gnn_baseline.py`)
`transaction graph built from transfers before a cut-off time` → GraphSAGE passes information between connected accounts (and ATMs) → `a per-account risk of cash-out in the next 60 min`.
*Plain words: a graph model that looks at relationships between connected accounts. Not used by the console.*

**Temporal GNN (offline experiment)** (`graphx/tgnn_baseline.py` and related)
`graph snapshots at several times + each transfer's amount and age` → message passing that weights recent transfers differently from old ones → `a per-account risk of cash-out, or the next movement`.
*Plain words: a graph model that also considers how the money-flow network changes over time. Not used by the console; its best recorded score (0.79) is below M1.*

## The non-negotiable rules, and how they are enforced

- **R1 as-of** — every feature is `f(account_id, as_of, ctx)` and reads only
  `ts < as_of`. `tests/test_leakage.py` rebuilds a *physically truncated* corpus
  and asserts bit-identical feature values.
- **R2 `label_` prefix** — `drop_leaky_columns` strips every `label_*` column, so
  a label can never reach a model.
- **R3 no live location** — only KYC PIN centroid, opening branch, and the
  account's *historical* withdrawal footprint. `home_lat/lon` are generator-only
  and never shipped (`_accounts_internal.parquet`).
- **R4 honest metrics** — target ROC-AUC 0.86–0.92; a single-feature audit flags
  anything > 0.85 alone (the strongest single feature scores 0.69). We hit
  **0.913** (chronological); the generator was widened until this was true, not
  tuned upward.
- **R5 hierarchical abstention** — district → ~5 km² cell → terminal; the system
  refuses to name a terminal on a flat distribution (`abstained_at`,
  `abstain_reason`).
- **R6 naive IST** — all datetimes are naive IST internally; `+05:30` is appended
  only at JSON serialisation.

## Results you can reproduce (`reports/metrics.json`)

| split | M1 ROC-AUC | location terminal Top-1 | cell Top-1 |
|---|---|---|---|
| A chronological | 0.913 | 0.210 | 0.65 |
| B unseen account | 0.902 | 0.180 | 0.64 |
| C unseen terminal | — | 0.168 | 0.65 |
| D sealed (shifted) | **0.885** | 0.207 | 0.62 |

- **Candidate recall@50 = 0.79** (the hard ceiling on Top-K; just under the 0.80 we aim for).
- **M2 channel accuracy = 0.922** (chronological). Note ~77% of rows are "dormant", so always guessing "dormant" would already score ~0.77.
- **Baselines**: a hand-written rule (inflow ≥ ₹25k AND >70% forwarded in 10 min
  AND account < 90 days old) never fires on the test cases, so it scores ROC-AUC
  **0.500** (no better than chance); logistic regression 0.834; M1 0.913.
  Location: nearest-KYC Top-1 0.057, most-frequent-ATM 0.203, vs M3 0.210.
- **North Star — lead time**: ~8 min median (8.3) *when the system can act*, but in
  **~70% of cases (420 of 600) the money was already cashed out before the
  complaint was even filed** (reporting delay ≫ forwarding delay). That honest
  number is the case for proactive freezing and faster reporting.
- **Data**: 8,000 accounts, ~248k transactions, 600 complaints.

## Known limitations

- **ATM coordinates are synthetic**, scattered around real district centroids
  pending an OpenStreetMap pull. `gen/atms.py` contains the exact Overpass query
  to swap in real `amenity=atm` nodes.
- **All training data is synthetic by design.** Real CFCFRMS money-mule chains
  are personal financial data protected under the **DPDP Act 2023**; they cannot
  be used to train an open prototype. The generator is deliberately *hard*
  (overlapping mule/legit delays, six mule-look-alike legitimate patterns,
  partial observability, label noise) so the modelling lessons transfer.
- **Not a licensed forecast of any real person or account.** Outputs are
  decision-support for authorised investigators, not evidence.
- The location ranker is the *simple* binary-classifier version (softmax within
  group); `rank:pairwise` is left as a documented next step for the team.

## Roadmap

**Current:** an investigator enters a complaint and the console shows the traced
money trail, M1/M2/M3 forecasts, a RED/AMBER/GREY tier, the reasons and a
simulated alert, all on synthetic Pune data.

**What we have built (implemented and used by the console)**
- Synthetic corpus + sealed shifted test corpus; a command-line synthetic case generator (`python -m generator.case`)
- Money-trail tracing, graph features, candidate ATM generation
- M1 / M2 / M3 XGBoost models with abstention; leakage test suite
- Decision rules, simulated SMS / email / API alerts, Case centre tickets, mule profile, Mule Network view

**What we are improving (exists in the repo, being tested offline, not in the console)**
- GNN and temporal GNN experiments in `graphx/` (so far they do not beat M1)
- More realistic generated cases (amounts, timing, legitimate look-alikes)

**Future (not built yet)**
- Real ATM locations from OpenStreetMap (`gen/atms.py` has the query)
- Pairwise (`rank:pairwise`) location ranker
- Wiring a graph model into the API, only if it beats M1 on the sealed split
- A real clustering step for mule roles, if needed

🤖 Generated with [Claude Code](https://claude.com/claude-code)
