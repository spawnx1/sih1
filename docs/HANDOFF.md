# PROJECT HANDOFF — for a new Claude taking over "Mule Connection" (SIH26184)

> Paste this whole file to the new Claude as the first message. It is written **to that new assistant**.
> Goal: let you pick up mid-stream and keep working exactly as the previous Claude did.

---

## 0. Read this first (tone + rules)

- The user is a student on a Smart India Hackathon team. Be **direct, decisive, and verify your work in the browser** — don't just claim things work, show proof (screenshots / curl output). They value honesty highly (they once caught fake academic citations and were right to).
- Use **respectful language** in everything you write, including visible thinking. If the user is ever crude out of frustration, don't mirror it and don't lecture — just keep helping.
- **Act, don't over-plan.** When you have enough to act, act. Keep changes **additive**; don't rebuild or refactor working code. The user explicitly hates unnecessary work / duplicate systems.
- The user iterates a LOT on visuals. Expect "make it more mature / simpler / more cinematic" loops. Build, screenshot, adjust.

---

## 1. What the project is

**SIH26184** — a predictive **cybercrime cash-out forecasting** system for the Ministry of Home Affairs / I4C. Product/demo name the user prefers: **"Mule Connection."**

The idea in one line: *a fraud complaint comes in → the system reconstructs where the stolen money went → forecasts, for each mule account still holding it, whether/when/how/where it will be pulled out as cash → so police can freeze the account before the "golden hour" closes.*

Everything is **synthetic data by design** (DPDP Act 2023 — real mule data can't be used). The generator is deliberately hard so the modelling lessons transfer.

---

## 2. Where things live (IMPORTANT paths)

- Repo root (git): `C:\Users\Vaibhav\Downloads\cl` — git remote is **`spawnx1/sih1`**, default branch `main`, work branch `master`.
- **The actual project is the subfolder `sih26184/`.** (The parent `cl` repo also contains an UNRELATED project called **NEXUS** — a RAG/document-Q&A app. Do NOT confuse NEXUS with this project.)
- Working dir you'll usually want: `C:\Users\Vaibhav\Downloads\cl\sih26184`
- Python venv: `sih26184\.venv` (Python 3.12). **Always call `.venv\Scripts\python.exe`** — the system `python` lacks the deps (uvicorn etc.).
- Node v24 is installed. `imageio-ffmpeg` is installed in the venv (bundles its own ffmpeg 7.1 binary — there is **no** system ffmpeg).

### Environment gotchas (Windows)
- The console is **cp1252** — printing `₹` or box glyphs crashes with `UnicodeEncodeError`. Prefix Python one-offs with `export PYTHONIOENCODING=utf-8` (Git Bash) or `sys.stdout.reconfigure(encoding="utf-8")` in scripts.
- Shell is Git Bash **and** PowerShell (both available). Prefer absolute paths.
- To free port 8000 before restarting: `netstat -ano | grep ':8000' | grep LISTENING | awk '{print $5}'` then `powershell -Command "Stop-Process -Id <PID> -Force"`.

---

## 3. How to run it

```bash
cd C:/Users/Vaibhav/Downloads/cl/sih26184
# one-time build (data + models) is already done — data/ and models/ exist
./.venv/Scripts/python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --log-level warning
# open http://127.0.0.1:8000
```
Uvicorn is **not** run with `--reload`, so **restart it after any `api/` change**. Frontend (`web/index.html`) is served with `no-store`, so a browser refresh picks up JS/CSS edits immediately.

To verify in-browser you have the built-in browser tools (`mcp__Claude_Browser__*`): `navigate` to the URL, `computer`/`screenshot`, `javascript_tool` (pass `tabId:"seed"` — the app tab). Prefer `read_page`/`find` over blind clicking.

---

## 4. Architecture & stack

One FastAPI process, built from offline artefacts (Parquet + trained models), serving one single-file console. No DB server except a small SQLite for tickets. No cloud.

```
gen/ (synthetic world) → data/*.parquet ; ml.train → models/*.joblib      [OFFLINE, already built]
api/main.py (FastAPI): FeatureContext(in-RAM) → per request:
   complaint → graphx/trail (as-of BFS) → graphx/candidates (C1–C5) →
   ml/features (as-of vectors) → ml/predict (M1/M2/M3 + abstention) →
   plain-rules decision layer → Prediction/Recommendation
   + SQLite tickets, /studio (generator), StaticFiles → web/index.html
web/index.html: vanilla JS + Leaflet + d3. Views: Command Centre · Case view ·
   Case Centre · Mule Network · Case Studio. Base/Advanced mode switch.
```

**Stack:** Python 3.12 · pandas/numpy/pyarrow · scikit-learn + **XGBoost** · networkx · **h3** · FastAPI/uvicorn/pydantic · SQLite · vanilla HTML/CSS/JS + Leaflet.js + d3.js · (PyTorch + PyTorch-Geometric = a *research-only* GNN track, NOT shipping).

What runs in the product vs. what is an offline experiment (GNN, temporal GNN, fusion) is listed in the **"Models"** table in `README.md`. Keep that table in sync; never describe the `graphx/*gnn*` track as live. There is no clustering step; Mule Network roles are rule-based (`api/main.py:_archetype`).

### Key files
`config.py` (constants, 4 Pune districts, thresholds) · `contracts.py` (Pydantic) · `gen/{params,atms,generator}.py` · `graphx/{trail,candidates,features}.py` + `graphx/*gnn*` (research) · `ml/{features,panel,train,predict,evaluate}.py` · `api/main.py` · `web/index.html` (~2,300 lines) · `generator/case.py` (per-case synthetic generator) · `tests/test_leakage.py`.

---

## 5. The models & the numbers (memorise)

- **M1 hazard** (will it cash out & when): `XGBClassifier` binary, horizon stacked as a feature → cumulative hazard curve over 10/20/30/45/60 min.
- **M2 channel** (how): `XGBClassifier` multi:softprob over ATM/POS/onward_transfer/dormant.
- **M3 location** (where): `XGBClassifier` binary over (event, candidate-ATM) pairs, softmax-in-group, with **hierarchical abstention** (district→cell→terminal — it refuses to name one ATM on a flat distribution).
- **mule_score**: a small **`LogisticRegression`** (binary mule vs not) over 8 graph features — it's a *feature* fed into M1, not a headline model. (User noted logistic "isn't the best" — an upgrade to boosted/GNN is a possible future task, but keep the R4 audit in mind: any single feature scoring >0.85 alone is treated as a leak.)

**Metrics (held-out, from `reports/metrics.json`):** M1 ROC-AUC **0.913** (chronological), 0.916 (unseen account), **0.879** (sealed/shifted — proves it generalises); PR-AUC & Brier reported too. Location terminal top-1 ~0.21, cell top-1 ~0.69, median error ~0.7 km. Candidate recall@50 ~0.82. Baselines: rules 0.500, logistic 0.840. Lead time median **11.8 min**, but **396/600** cases were already cashed out before the complaint was filed (the honest reality that motivates proactive freezing).

### The six non-negotiable rules (this is the credibility story)
R1 **as-of** (features read only `ts < as_of`; enforced by a bit-for-bit leakage test) · R2 **label_ prefix** dropped · R3 **no live location** (only KYC + historical footprint) · R4 **honest metrics** (target 0.86–0.92; audit any single feature >0.85) · R5 **abstention** · R6 **naive IST** (`+05:30` only at JSON serialisation).

---

## 6. What has been built (feature inventory)

All of this is **implemented and verified working** unless marked otherwise.

**Backend (`api/main.py`) endpoints:** `/queue`, `/case/{ack}`, `/trail/{ack}`, `/predict/{ack}`, `/hotspots`, `/alerts/dispatch`, `/alerts`, `/explain/{ack}`, `/replay/state`, `/tickets*` (Case Centre SQLite lifecycle), `/studio/{scenarios,generate}` (synthetic case generator), **`/account/{id}`** (mule profile — added this session), **`/network`** (mule network graph — added this session). Helper `_codename(account_id)` gives stable evocative codenames (e.g. `NOMAD-2405`).

**Frontend (`web/index.html`) views/features:**
- **Command Centre** (overview: KPI tiles + metro risk map).
- **Case view**: the live **money-flow map** (Leaflet). NOTE: the connecting victim→mule→ATM "waypoint" lines are drawn in a **screen-space SVG overlay `#flowlay`** via `renderRoutes()` using `latLngToContainerPoint`, NOT Leaflet polylines (Leaflet mis-positioned them under a big render padding). Lines are trimmed to stop at each node's circle edge. Plus the money-mule network graph (`muleGraphSVG`), the forecast panels, freeze recommendation, ranked target-ATM menu, and the **cinematic Replay** (`togglePlay` — letterbox, timecode, "⚠ CASH-OUT WINDOW"). The money-mule graph nodes are clickable → open the mule profile.
- **Case Centre**: SOAR-style ticket lifecycle (assign/status/notes/resolve/audit, CSV export).
- **Mule Network** (`showNetwork`, d3-force): the NEXUS-style glowing entity graph — 54 codenamed mule nodes, coloured by behaviour archetype (CASHER=amber, DISTRIBUTOR=teal, COLLECTOR=violet), sized by #cases, linked by shared-case/transfer/ring edges; hover-highlight + tooltip, drag, zoom, legend toggles, **click a node → its case file**.
- **Mule Profile modal** (`openMule` → `/account/{id}`): one persistent entity showing codename, KYC, all cross-corpus transactions, **linked cases across many complaints** (e.g. ACC_02405 / NOMAD-2405 = 11 victims), chain-following counterparties, ring, and Open-case buttons into the existing tickets. This solved the mentor's #1 ask (persistent mule identity).
- **Base ⇄ Advanced mode switch** (header, `setMode`, persisted in localStorage): Base = the SAME advanced console with plain-English `.bx` notes layered on each panel + a guide banner, and the densest analyst-only panels (Case management, "Why", Technical details, Case Studio, voice picker) hidden via `.bhide`/`#navStudio`. Advanced = untouched full console.
- **Case Studio** (`showStudio`): cinematic synthetic-case generator UI (letterbox, act beats, graph reveal, ground-truth reveal), driven by `generator/case.py` + `/studio/generate`.

**Standalone generator:** `generator/case.py` — deterministic (`--seed`), leakage-safe per-case synthetic bundles (`cases/<id>/case.json, transactions.csv, nodes.csv, edges.csv, accounts.csv, timeline.csv, ground_truth.json`) with a cinematic terminal print. `python -m generator.case --scenario complex --seed 42`.

**Deliverables produced (in `docs/`):**
- `complete-manual.pdf` (51pp) + `field-guide.pdf` (14pp) — built from HTML via Edge headless print-to-PDF, page numbers stamped with reportlab/pypdf.
- `TRAILER.md` — full 60s Apple-style trailer production plan (grounded in real features).
- **`mule_connection.mp4`** — a real 67s 1080p H.264 + AAC cinematic explainer, rendered frame-by-frame by **`scripts/make_trailer.py`** (Pillow + numpy → imageio-ffmpeg, generated soundscape muxed). Re-run: `./.venv/Scripts/python.exe scripts/make_trailer.py`.

**Current UI aesthetic (WIP, user-driven):** currently **Michroma** (display/`--sans`) + **Share Tech Mono** (`--mono`) + **Inter** (`--body`), muted steel-teal palette (`--cyan:#4fadba`), glows dialled down. The user has bounced between "more mature / less neon" and "more cinematic (Watch Dogs / Resident Evil)". There are 7 exported UI-direction mockups + a "Leon/RE tactical" and an "ops-terminal" concept the user was choosing between (see conversation). Treat the exact theme as **not finalised**.

---

## 7. The mentor's 8 requirements — audit status

The user's mentor gave 8 requirements; a full audit was done (see conversation). Status:
1. **Persistent mule identity** — ✅ DONE this session (`/account/{id}` + Mule Network + profile modal).
2. **Mule history / investigation cases** — ✅ DONE (profile links to existing tickets; no second case system).
3. **Base mode** — ✅ (implemented; user still refining how simple).
4. **Advanced mode** — ✅ (the full console).
5. **Map visualization** — ✅ (Leaflet money-flow map); tiny TODO: add a visible "synthetic/demo locations" label.
6. **Live transaction demonstration** — ⚠️ mostly (Case Studio + Replay); optional: push a Studio case live into the queue + auto-create ticket.
7. **Investigation ticket/case generation** — ⚠️ tickets exist & are mule-linked; optional: on-demand "create case for this account" button.
8. **Mature professional UI** — ⚠️ in progress (theme not finalised; see §6).

---

## 8. State of the git repo

**Almost nothing from this session is committed.** The user kept deferring commits. Large uncommitted work includes: the whole map/theme overhaul, Base/Advanced mode, the mule profile + Mule Network + `/account` + `/network` + codenames, the generator (`generator/case.py`), Case Studio, `docs/` (manuals, TRAILER.md, mp4), `scripts/make_trailer.py`. `cases/` and `data/tickets.db` are gitignored. **Ask before committing/pushing** (outward-facing), and use the attribution lines the harness specifies. Do a `git status` first to see the true current state (this handoff is a snapshot).

---

## 9. How to continue "the same conversation"

Open with something like: *"I've read the handoff — the project's in `sih26184/`, server runs via `.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000`, and we just shipped the persistent mule identity + Mule Network graph + a rendered mp4 trailer. What do you want to tackle next?"* Then, before editing, **run the server, open the app, and screenshot the current state** so you're grounded in reality (things may have changed since this snapshot).

**Likely next tasks the user may want:** finalise the UI theme (mature vs cinematic); requirement #8 polish; requirement #5/6/7 optional bits; upgrade the mule-score model; centre/curate the Mule Network layout; a phone/tablet layout; commit + push the session's work; more trailer/marketing assets.

**Do first, every session:** `git status`, start the server, screenshot the app. **Never** rebuild working features; keep it additive; verify in-browser; be honest about what works.
