# Project Manual — Cash-out Forecasting (SIH26184)

A plain-language guide to what this project is, how it works, and how to explain
it to your team and to the judges. **No machine-learning background needed to
read this.**

---

## 1. The one-paragraph version

When someone is scammed online, the fraudster doesn't keep the money — they
quickly move it through a chain of "mule" bank accounts and then **pull it out as
cash** at an ATM, usually within minutes. Once it's cash, it's gone. Our system
takes a fraud complaint, **traces where the money went**, and then **predicts,
for each account still holding the stolen money: will it be cashed out? when?
through what channel? and at roughly which ATM area?** The goal is to **freeze
the account before the cash is withdrawn** — and if that's not possible, to tell
police *where* to look.

---

## 2. The problem, in real terms

- A victim files a complaint (on the national cybercrime portal) naming **one
  disputed transaction** — e.g. "₹50,000 left my account to some UPI ID."
- The fraudster has already **split and forwarded** that money through several
  mule accounts (this is called *layering*).
- Each mule **withdraws cash** at an ATM in their local area.
- **The average complaint is filed ~110 minutes after the fraud** — by then, in
  most cases, the cash is already withdrawn. The window to act is tiny.

**So the winning move is not "catch them at the ATM." It's "freeze the account
before the ATM."** Our system is built around that idea.

---

## 3. The big idea (say this out loud in the demo)

> "We don't just detect fraud after the fact. We **forecast the cash-out** —
> whether, when, and where the stolen money will become cash — so police can
> **freeze the account in time**, and if they can't, know **which area** to
> watch."

---

## 4. How it works — 5 steps

```
   1. COMPLAINT           2. TRACE                3. FORECAST
   ┌──────────┐          ┌──────────────┐        ┌────────────────────────┐
   │ 1 victim │  ─────▶  │ follow the   │ ─────▶ │ for each account still  │
   │ 1 txn    │          │ money hop by │        │ holding stolen money:   │
   └──────────┘          │ hop (a graph)│        │  • WILL it be cashed?   │
                         └──────────────┘        │  • WHEN? (next 10–60m)  │
                                                 │  • HOW? (ATM/POS/...)   │
                                                 │  • WHERE? (district→    │
                                                 │    ~5 km cell→terminal) │
                                                 └───────────┬────────────┘
                                                             ▼
   5. ALERT                          4. DECIDE
   ┌───────────────────┐            ┌──────────────────────────────┐
   │ SMS / email / API │  ◀──────── │ plain rules:                 │
   │ to the right      │            │  RED   → FREEZE now + alert   │
   │ police station    │            │  AMBER → watch / email brief  │
   └───────────────────┘            │  GREY  → keep tracing         │
                                    └──────────────────────────────┘
```

---

## 5. The four things we built (these are the four required deliverables)

| # | Deliverable | Where it is |
|---|---|---|
| 1 | **Prediction engine** (the AI that forecasts will / when / how / where) | `ml/` folder |
| 2 | **GIS risk heat-map** (the map that shades the high-risk areas) | the middle panel of the web console |
| 3 | **Police interface** (the queue of cases + what to do about each) | the web console + `api/` |
| 4 | **Alert system** (simulated SMS / email / API messages, with a log) | "Dispatch alert" button + `/alerts` |

---

## 6. The three AI models, in plain words

Think of them as three questions asked about **one account that just received
stolen money**:

- **M1 — "Will it be cashed out, and when?"**
  Gives a probability for the next 10, 20, 30, 45, 60 minutes. We call this rising
  curve the **hazard curve**. From it we read the most likely **time window**.

- **M2 — "How will the money leave?"**
  Four options: **ATM** withdrawal, **POS** (card-machine cash), **onward
  transfer** (it'll keep moving — *don't send anyone yet*), or **dormant**
  (nothing right now). This stops police from rushing to an ATM when the money is
  actually about to move again.

- **M3 — "Where?"**
  Ranks likely ATMs. Crucially, it answers at the **level it can defend**:
  - if confident → a specific **terminal** (one ATM),
  - if less sure → a **~5 km² cell** (a neighbourhood),
  - if unsure → just the **district**.
  This is called **abstention** — *the system says "I only know the
  neighbourhood, not the exact ATM"* instead of pretending. Judges love this
  because it's honest.

**The same three models as input → process → output:**

| Model | Input | Process | Output |
|---|---|---|---|
| M1 | the account's transactions and position in the money trail, up to *now* | XGBoost (traditional ML) | probability of cash-out in the next 10-60 min |
| M2 | the same information | XGBoost | chance of ATM / POS / onward transfer / dormant |
| M3 | candidate ATMs (past withdrawals, ring, KYC area, known fraud ATMs) | XGBoost ranker + abstention | ranked areas: terminal, ~5 km² cell or district |
| Rules | M1's 10-min probability + stolen amount held | fixed thresholds | RED / AMBER / GREY tier and the police station to alert |

**Graph analysis (used by the console).** Before the models run, the system
follows the money from the victim through each account it was sent to (the
*trail*) and describes how each account behaves in that network (how many
senders, how fast money leaves). These graph facts are inputs to M1 and M2.

**Graph neural networks (offline experiments, not in the console).** The
`graphx/` folder also has GNN code:
- **GNN** (technical: GraphSAGE): *a graph model that analyses relationships
  between connected accounts.*
- **Temporal GNN** (technical: snapshot temporal GraphSAGE; not a TGN): *a graph
  model that also considers how the money-flow network changes over time.*

These are research experiments. Their best recorded score (0.79) is below M1
(0.913), so the console does not use them. Do not present them as live.

**Clustering.** The project does not run a clustering algorithm. The "roles" on
the Mule Network screen (cash-out specialist, distributor, collector, relay) come
from simple fixed rules about how money flows in and out of each account.

---

## 7. The screen, panel by panel (this is what confused you)

The console has **three columns** and a **player bar** at the bottom.

**LEFT — "Triage queue".** The list of active complaints, most urgent at the
top. Each card shows:
- a coloured tag: **RED / AMBER / GREY** = how urgent (see glossary),
- the fraud type and amount,
- **"golden-hour left"** = minutes remaining in the first hour after the fraud
  (the best window to freeze),
- a blue bar = overall urgency score.
👉 **Click a card to load that case on the right.**

**MIDDLE — "GIS risk heat-map".** A map of the four districts. Each district is
a **shaded region** — the darker/redder it is, the more forecast cash-out risk
there. The **dots** are the specific predicted ATM areas ("cells"), sized and
coloured by risk; the top ones are **numbered**. The box on the left lists the
**Top forecast cells**. Filters on top let you narrow by date / amount / district.

**RIGHT — "Case detail".** Everything about the selected case:
- a coloured **recommendation** box: the action (FREEZE / MONITOR) and *why*,
- the **target account** and how much stolen money it's holding,
- the **hazard curve** (M1) — the will/when,
- the **channel** breakdown (M2) — the how,
- the **predicted location** (M3) — the where, with the abstention note,
- the **money-flow trail** — a small diagram of the money hopping between
  accounts (blue = victim, red = account holding cash, grey = pass-through).

**BOTTOM — replay player.** Press **Play ×20** to *re-run a real case in fast
motion*: the clock ticks, the money moves, the hazard rises, the forecast
updates live. This is your best demo moment.

---

## 8. Glossary (define these when you present)

| Term | Plain meaning |
|---|---|
| **Complaint / ack_no** | A filed fraud report; `ack_no` is its 14-digit ID. |
| **Seed transaction** | The one disputed payment the victim reported. |
| **Mule account** | A (often rented) bank account used to move stolen money. |
| **Trail** | The chain of accounts the money passed through. |
| **Cash-out** | The moment stolen money becomes physical cash (ATM withdrawal or POS card-cash). Written "cash-out" as a noun, "cash out" as a verb. |
| **Cell** | A ~5 km² map tile (an "H3" hexagon) — a neighbourhood-sized area. |
| **Hazard curve** | The rising probability of cash-out over the next hour. |
| **Tier RED/AMBER/GREY** | RED = freeze now (high chance + big money); AMBER = watch; GREY = keep tracing. |
| **Abstention** | The model refusing to name one ATM when it isn't sure, and giving the neighbourhood instead. |
| **Golden hour** | The first 60 minutes after the fraud — best chance to freeze. |
| **Lead time** | How many minutes *before* the cash-out we raised the alarm. Positive = we were early. |
| **Cash withdrawal** | One specific cash-out at an ATM (channel `ATM_WDL`). |
| **Suspected mule account** | An account the system has flagged for review because of how money moves through it. A flag is a lead, **not proof** that the account holder is a criminal. |
| **Detection vs prediction** | *Detection* = recognising what already happened (the trail). *Prediction* = estimating what may happen next (cash-out, channel, location). |
| **Probability vs risk tier** | *Probability* = the model's estimated chance (e.g. 0.66 = 66%). *Risk tier* (RED/AMBER/GREY) = a fixed rule on top of that probability and the amount, used to prioritise work. |
| **Predicted location** | A ranked estimate of where cash may be withdrawn, not a guarantee. The system abstains when unsure. |

---

## 9. Why the project is credible (the engineering story judges respect)

Because our data is generated by us, the easy trap is a model that "cheats" and
scores a fake 99% accuracy. We deliberately avoided that:

- **No time-travel ("as-of" rule).** Every prediction uses *only* information
  that existed *before* the moment of prediction. We have automated tests that
  re-check this and fail the build if any feature peeks into the future.
- **No leaking the answer.** The true labels are physically dropped before the
  model sees the data.
- **No live GPS / phone location.** Police can't get a suspect's live location in
  10 minutes, so we don't use it — only the account's *past* ATM habits and its
  registered (KYC) area.
- **Honest scores.** We *targeted* a ROC-AUC of 0.86–0.92 (not higher) and got
  **0.91**. If a model scores above 0.95 we treat it as a bug and hunt the leak.
- **We beat the simple alternatives.** A hand-written rule never flags a single
  test case (so it is no better than chance, 0.50); a basic statistical model
  (logistic regression) scores 0.83; ours scores **0.91**. That is the answer to
  "why not just use rules?"

---

## 10. The numbers that matter (memorise these)

- **ROC-AUC 0.91** forecasting *whether* an account cashes out (1.0 = perfect
  ranking of risky over safe accounts, 0.5 = guessing).
- Holds at **0.885** on a *totally different* ("sealed") dataset the model never
  saw — proof it generalises, not memorises.
- **79%** of the time the real ATM is inside our shortlist of candidate ATMs.
- **~0.7 km** median error when we do name a location.
- **A simple rule never fires (0.50); basic statistics 0.83; our model 0.91.**
- **Reality check:** in ~70% of complaints the money was *already cashed out
  before the complaint was even filed.* When we *can* act, we get **~8 minutes**
  of lead time. This is the honest case for faster reporting + auto-freeze.

---

## 11. How to run it (for the team)

From the `sih26184` folder, with Python installed:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy scikit-learn xgboost networkx fastapi uvicorn pydantic pyarrow h3 matplotlib pytest
```

Then, one time, build the data and train the models:

```bash
python -m gen.atms
python -m gen.generator
python -m gen.generator --sealed
python -m ml.train
python -m ml.evaluate
```

Run it:

```bash
python demo.py                                # prints one case start-to-finish
python -m uvicorn api.main:app --port 8000    # then open http://localhost:8000
```

(Or just `make all`, `make demo`, `make api`.)

---

## 12. How to demo it live (3-minute script)

1. **Open the console.** "This is the officer's screen. Left = complaints,
   sorted by urgency."
2. **Click the top RED case.** "The system traced the money to this account,
   which is holding ₹X of stolen funds."
3. **Point at the hazard curve.** "It predicts a 66% chance of cash-out in the
   next 10 minutes."
4. **Point at the map + location.** "Most likely in *this* neighbourhood — and
   notice it *abstains* from naming a single ATM because it isn't certain. Honest
   AI."
5. **Point at the recommendation.** "So the action is **FREEZE**, and an alert
   goes to *this* police station."
6. **Hit "Replay simulation".** "Here's the same case replayed in fast motion — watch the
   money move and the risk rise in real time."
7. **Close with the number:** "0.91 ROC-AUC, where a simple rule catches nothing,
   and it tells us when it doesn't know."

If the browser dies on stage, run `python demo.py` — it prints the whole story
to the terminal.

---

## 13. What's real vs. what's simulated (be upfront)

- **All data is synthetic** (made by our generator). Real money-mule data is
  personal financial information protected by the **DPDP Act 2023** and can't be
  used in an open prototype. Our generator is deliberately *hard* (mules and
  honest people look similar) so the lessons still transfer.
- **ATM locations are synthetic**, scattered around four real Pune-area
  districts (Pune City, Pimpri-Chinchwad, Hinjawadi, Hadapsar). The code has the exact OpenStreetMap query to drop in real ATMs.
- **Alerts are simulated** — we build and log the SMS/email/API payloads; we
  don't actually send them.
- **GNN / temporal GNN code is experimental** — it lives in `graphx/` and is not
  used by the console. See the model table in `README.md`.
- **Outputs are decision support**, not evidence. A RED tier means "review and
  consider freezing first", not "this person is guilty".

---

## 14. Who does what (suggested split for a 4-person team)

- **Person A — Data & realism:** `gen/` (make the synthetic world harder / add
  real ATMs from OpenStreetMap).
- **Person B — Models:** `ml/` (improve M1/M2/M3, try the pairwise location
  ranker, tune features).
- **Person C — Backend & rules:** `api/` (the decision layer, alert routing, new
  endpoints).
- **Person D — Frontend & story:** `web/` (the console, the demo, the pitch).

Everyone should be able to run `make demo` and explain Section 6.

---

*Questions this manual should let you answer without opening the code: what the
system does, why it's needed, what each screen shows, why it's trustworthy, and
what to say to a judge.*
