# Data Requirements

What data this project needs, the exact schema, and why common public
fraud datasets (like the Kaggle card-fraud set) **do not fit**. Read this before
spending time sourcing or cleaning any dataset.

---

## The one thing that matters most

This system **follows money between accounts and forecasts the cash-out.** So the
data must be a **money-movement graph**, not a list of purchases. Concretely,
every dataset we use MUST have:

1. **Account-to-account edges** — each transfer has a `src_account` **and** a
   `dst_account`, so money can be traced hop by hop.
2. **Cash-out rows** — the moment money becomes cash, encoded as:
   > `channel ∈ {ATM_WDL, POS}`  **AND**  `dst_account IS NULL`  **AND**  `atm_id` is set.

If a dataset has no destination *account* (only a merchant), it cannot build the
trail, and **none of M1/M2/M3 can run.** No amount of cleaning fixes that.

---

## The four tables (code-accurate schema)

Types: `ts`/`open_date`/`filed_ts` are **naive datetimes representing IST** (R6).
"Required" = the pipeline breaks without it. "Optional" = improves realism/accuracy.

### 1. `accounts.parquet`
| column | type | required | notes |
|---|---|---|---|
| `account_id` | str | ✅ | primary key, e.g. `ACC_00042` |
| `bank_code` | str | ✅ | used to tell same-bank vs inter-bank transfers |
| `open_date` | datetime | ✅ | for account-age features |
| `kyc_pin` | str | ✅ | **string** (leading zeros matter) |
| `kyc_district` | str | ✅ | jurisdiction routing |
| `kyc_lat`, `kyc_lon` | float | ✅ | KYC PIN centroid (an R3-allowed location prior) |
| `branch_lat`, `branch_lon` | float | ⭐ optional | account-opening branch; helps location model |
| `label_is_mule` | bool | 🔖 label | **training only** |
| `label_ring_id` | str/null | 🔖 label | **training only**; groups mules that share cash-out geography |

> Generator-internal file `_accounts_internal.parquet` (`archetype`, `home_lat`,
> `home_lon`, `home_cell`) is **never** given to the model (R3). If real data ever
> contains a true "home location", it stays out of features too.

### 2. `transactions.parquet`  ← the heart of the dataset
| column | type | required | notes |
|---|---|---|---|
| `txn_id` | str | ✅ | primary key |
| `ts` | datetime (IST) | ✅ | drives the entire as-of / time-respecting logic |
| `src_account` | str/null | ✅ | debited account (null only for cash-in like `CASH_DEP`) |
| `dst_account` | str/null | ✅ | credited account; **null iff this is a cash-out** |
| `amount` | float | ✅ | |
| `channel` | str | ✅ | one of `UPI, IMPS, NEFT, RTGS, ATM_WDL, POS, AEPS, CASH_DEP` |
| `atm_id` | str/null | ✅ | set **iff** cash-out channel; must exist in `atms` |
| `dst_vpa` | str/null | ⭐ optional | UPI handle; enables the UPI branch/playbook |
| `observed` | bool | ⭐ optional | did the bank confirm this edge? (models partial visibility) |
| `label_is_tainted` | bool | 🔖 label | **training only**: is this stolen money? |
| `label_case_id` | str/null | 🔖 label | **training only**: which complaint this edge belongs to |
| `label_hop` | int | 🔖 label | **training only**: hop # from the victim |

**Invariant (must hold for every row):**
`atm_id` non-null ⟺ `channel ∈ {ATM_WDL, POS}` ⟺ `dst_account` is null.

### 3. `complaints.parquet`
| column | type | required | notes |
|---|---|---|---|
| `ack_no` | str | ✅ | 14-digit acknowledgement number |
| `filed_ts` | datetime (IST) | ✅ | when the complaint was filed (`filed_ts ≥ seed.ts`) |
| `victim_account` | str | ✅ | |
| `seed_txn_id` | str | ✅ | the one disputed transaction; must resolve in `transactions` |
| `disputed_amount` | float | ✅ | |
| `fraud_category` | str | ✅ | typology, e.g. `fast_fanout, layered, ring_reuse, …` |
| `reporting_delay_min` | float | ⭐ optional | minutes from fraud to filing (drives the lead-time story) |

### 4. `atms.parquet`
| column | type | required | notes |
|---|---|---|---|
| `atm_id` | str | ✅ | primary key, referenced by cash-out transactions |
| `lat`, `lon` | float | ✅ | terminal location |
| `district` | str | ✅ | |
| `cell` | str | ✅ | H3 res-7 (~5 km²) cell — the operational forecast unit |
| `operator` | str | ⭐ optional | SBI/HDFC/… — used by M3 |
| `pin` | str | ⭐ optional | **string** |
| `cell_fine` | str | ⭐ optional | H3 res-8 (~0.5 km²) |
| `per_txn_limit` | int | ⭐ optional | 10000/20000/25000 — feeds "amount fits limit" |
| `is_standalone`, `night_active`, `hourly_weight` | — | ⭐ optional | improve M3 |

**So: is this all we need? Yes.** The ✅ columns are the complete minimum to run
the whole pipeline; ⭐ columns raise realism/accuracy; 🔖 labels are for training
and evaluation only.

---

## Why the Kaggle "Credit Card Fraud" set does NOT fit

`fraudTest.csv` (columns: `cc_num, merchant, category, amt, first, last, gender,
street, city, state, zip, lat, long, city_pop, job, dob, merch_lat, merch_long,
is_fraud`) is **card-purchase fraud detection** — a different problem.

| We need | That dataset has | |
|---|---|---|
| account → account transfers | card → **merchant** purchases | ❌ |
| mule accounts / laundering chains | individual shoppers | ❌ |
| cash-out at ATMs (`atm_id`) | merchant purchases | ❌ |
| channels (UPI/IMPS/NEFT/ATM_WDL) | merchant `category` | ❌ |
| complaints (victim + seed txn) | — | ❌ |
| ATM locations | cardholder-home & merchant lat/long | ❌ |

With no destination **account**, there is no money trail — so M1 (will/when),
M2 (UPI/bank/ATM routing), and M3 (where) cannot run. **Do not clean or adapt it;
it answers a different question.** Using it would mean abandoning the actual
problem statement (forecasting cash-out *location*) for generic transaction
classification.

---

## Where real/realistic data can come from

- **Primary: our synthetic generator (`gen/`).** Real money-mule chains are not
  public — genuine CFCFRMS/I4C data is personal financial data protected under
  the **DPDP Act 2023** and cannot be used in an open prototype. The generator is
  deliberately *hard* (overlapping mule/legit behaviour, look-alike legitimate
  bursts, partial observability, label noise) so the modelling lessons transfer.
  This is a strength in the pitch, not a weakness.
- **Real ATM locations (do this):** swap the synthetic ATM scatter for real
  `amenity=atm` nodes from **OpenStreetMap** — the exact Overpass query is already
  in `gen/atms.py`. Zero legal risk, instant realism.
- **Optional external cross-check** (only to show models transfer — never the main
  data): **PaySim** (Kaggle "Synthetic Financial Datasets for Fraud Detection":
  has `TRANSFER` + `CASH_OUT`, `nameOrig → nameDest`, `isFraud`) or **IBM AMLSim**
  (simulated money-laundering *networks*). These have account-to-account edges and
  cash-out, so they're relevant — but they still lack ATM geo, UPI, and
  complaints, so map them in only as a secondary sanity check.

---

## What the data teammate should actually do
1. Pull **real OSM ATMs** for the four districts (Overpass query in `gen/atms.py`).
2. Make the generator harder/more realistic and re-verify honest AUC
   (`python -m ml.train`; AUC must stay **0.86–0.92**, never >0.95).
3. If time: map **PaySim/AMLSim** into the `transactions`/`accounts` schema above
   as an external transfer→cash-out cross-check.
4. **Do not** invest further in the card-purchase dataset.
