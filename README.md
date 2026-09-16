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
  anything > 0.85 alone. We hit **0.913** (chronological); the generator was
  widened until this was true, not tuned upward.
- **R5 hierarchical abstention** — district → ~5 km² cell → terminal; the system
  refuses to name a terminal on a flat distribution (`abstained_at`,
  `abstain_reason`).
- **R6 naive IST** — all datetimes are naive IST internally; `+05:30` is appended
  only at JSON serialisation.

## Results you can reproduce (`reports/metrics.json`)

| split | M1 ROC-AUC | location terminal Top-1 | cell Top-1 |
|---|---|---|---|
| A chronological | 0.913 | 0.207 | 0.69 |
| B unseen account | 0.916 | 0.267 | 0.73 |
| C unseen terminal | — | 0.146 | 0.68 |
| D sealed (shifted) | **0.879** | 0.210 | 0.67 |

- **Candidate recall@50 = 0.82** (the hard ceiling on Top-K).
- **Baselines**: rule engine ROC-AUC **0.500** (useless — the answer to "why not
  rules?"), logistic regression 0.840, vs M1 0.913. Location: nearest-KYC Top-1
  0.057, most-frequent-ATM 0.198, vs M3 0.207.
- **North Star — lead time**: ~12 min median *when the system can act*, but in
  **~66% of cases the money was already cashed out before the complaint was even
  filed** (reporting delay ≫ forwarding delay). That honest number is the case
  for proactive freezing and faster reporting.

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

🤖 Generated with [Claude Code](https://claude.com/claude-code)
