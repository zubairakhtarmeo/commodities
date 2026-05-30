# PROCUREMENT ENGINE SPECIFICATION

**Source of Truth:** `data/strategy/strategies.xlsx`  
**Date Documented:** 2026-05-29  
**Status:** Phase 2C — Analysis Only. No code has been written or modified.

---

## Table of Contents

1. [Workbook Architecture](#1-workbook-architecture)
2. [Raw Material Sheet Logic](#2-raw-material-sheet-logic)
3. [Consumption Sheet Logic](#3-consumption-sheet-logic)
4. [Market Inputs Sheet Logic](#4-market-inputs-sheet-logic)
5. [Strategy Sheet Logic](#5-strategy-sheet-logic)
6. [Formula Catalog](#6-formula-catalog)
7. [Organization Mapping](#7-organization-mapping)
8. [Data Flow Diagram](#8-data-flow-diagram)
9. [Implementation Blueprint](#9-implementation-blueprint)
10. [Risks, Ambiguities, and Open Questions](#10-risks-ambiguities-and-open-questions)

---

## 1. Workbook Architecture

### 1.1 Sheet Inventory

| Sheet | Dimensions | Role | Inputs From | Consumed By |
|---|---|---|---|---|
| Raw Material | A1:F194 | Oracle inventory snapshot + per-org aggregation | Oracle ERP extract | Strategy (stock values), Dashboard (total) |
| Consumption | A1:I27014 | Oracle transaction register (monthly) | Oracle ERP extract | Strategy (daily consumption rates) |
| Market Inputs | A1:E46 | Market prices, forecasts, cost-of-carry parameters | Manual + forecasting model | Strategy (all market params), Dashboard (snapshot) |
| Strategy | A1:T26 | Procurement decisions and arbitrage calculations | Raw Material + Consumption + Market Inputs | Dashboard (KPIs + coverage table) |
| Dashboard | A1:Q38 | Executive summary — no original calculations | Strategy + Raw Material + Market Inputs | End users / management reporting |

### 1.2 Dependency Chain

```
Oracle ERP                Oracle ERP
(Inventory)              (Transactions)        External Markets
     │                        │                      │
     ▼                        ▼                      ▼
Raw Material            Consumption           Market Inputs
(Stock levels)          (Daily rates)         (Prices + params)
     │                        │                      │
     └────────────────────────┴──────────────────────┘
                              │
                              ▼
                           Strategy
                    (Shortfall + arbitrage)
                              │
                              ▼
                          Dashboard
                      (KPIs + coverage)
```

### 1.3 Update Cadence

| Sheet | Updated | Trigger |
|---|---|---|
| Raw Material | Monthly | Oracle stock report download (end of month) |
| Consumption | Monthly | Oracle transaction register download (end of month) |
| Market Inputs Section A | Daily | Market open / treasury update |
| Market Inputs Section B | Monthly or on-demand | Forecasting model run |
| Market Inputs Sections C+D | Automatic | Formula-driven from Sections A+B |
| Strategy | Automatic | Formula-driven from all upstream sheets |
| Dashboard | Automatic | Formula-driven from Strategy |

---

## 2. Raw Material Sheet Logic

### 2.1 Purpose

Holds the monthly Oracle inventory snapshot for all raw material items across all spinning units. The lower section (rows 186–194) aggregates this detail data into a per-organisation, per-category summary table that the Strategy sheet reads directly.

### 2.2 Sheet Layout

| Zone | Rows | Content |
|---|---|---|
| Title | 1–2 | Report title, stock date, item class |
| Column headers | 5 | Sr#, Item Code, Description, Org Name, Primary Qty (Kgs), Category |
| Detail data | 6–181 | Individual item-level stock records (176 rows in current extract) |
| Blank separator | 182–185 | Empty |
| Summary headers | 186 | Org Name, Cotton, Fiber, Stretch Fiber, Cotton Waste, Total Inventory |
| Summary data | 187–193 | One row per organisation — SUMPRODUCT aggregations |
| Grand total | 194 | SUM of all org rows |

### 2.3 Detail Columns (Rows 6–181)

| Column | Name | Type | Mandatory | Purpose |
|---|---|---|---|---|
| A | Sr# | Integer | No | Serial number — no business logic |
| B | Item Code | String | Yes | Oracle item code; prefix determines category (e.g. `COT-100020`) |
| C | Description | String | No | Human-readable item name |
| D | Org Name | String | Yes | Spinning unit — drives the summary row grouping |
| E | Primary Qty (Kgs) | Float | Yes | Stock on hand in kilograms — the only numeric measure |
| F | Category | String | Yes | Material category — drives the summary column grouping |

### 2.4 Category Values Observed in Workbook

| Category Label (in F column) | Meaning |
|---|---|
| `Cotton` | All cotton varieties (local, imported, organic, recycled) |
| `Fiber` | Non-cotton fibers (viscose, polyester, bamboo, silk, flax) |
| `Stretch Fiber` | Elastane / spandex blends |
| `Cotton - Waste` | Cotton waste / by-product (note: hyphen in label) |

> **Critical:** The category label for waste material is `"Cotton - Waste"` with a hyphen-space pattern. Any external system writing to this sheet must use this exact string.

### 2.5 Summary Table Formulas (Rows 187–193)

The summary is built entirely from SUMPRODUCT formulas. The data range is hardcoded to rows 6–181.

**Cotton stock per org (Column B):**
```excel
=SUMPRODUCT(($D$6:$D$181=A187)*($F$6:$F$181="Cotton")*$E$6:$E$181)
```
Plain English: Sum of Primary Qty for all rows where Org Name matches this row's org AND Category is "Cotton".

**Fiber stock per org (Column C):**
```excel
=SUMPRODUCT(($D$6:$D$181=A187)*(($F$6:$F$181="Fiber")+($F$6:$F$181="Stretch Fiber"))*$E$6:$E$181)
```
Plain English: Sum of Primary Qty where Org Name matches AND Category is either "Fiber" OR "Stretch Fiber".

> **Important:** The Fiber column aggregates both `"Fiber"` and `"Stretch Fiber"` rows. Stretch Fiber is therefore included in both the Fiber total (column C) and the Stretch Fiber column (column D). This is the intended design — the Fiber column represents "all non-cotton fiber available for spinning", and Stretch Fiber is additionally broken out separately for informational purposes.

**Stretch Fiber stock per org (Column D):**
```excel
=SUMPRODUCT(($D$6:$D$181=A187)*($F$6:$F$181="Stretch Fiber")*$E$6:$E$181)
```
Plain English: Sum of Primary Qty where Org Name matches AND Category is exactly "Stretch Fiber".

**Cotton Waste stock per org (Column E):**
```excel
=SUMPRODUCT(($D$6:$D$181=A187)*($F$6:$F$181="Cotton - Waste")*$E$6:$E$181)
```
Plain English: Sum of Primary Qty where Org Name matches AND Category is "Cotton - Waste".

**Total Inventory per org (Column F):**
```excel
=SUM(B187:E187)
```
Plain English: Cotton + Fiber + Stretch Fiber + Cotton Waste for this org.

> Note: Because Fiber (C) already includes Stretch Fiber, and Stretch Fiber (D) is also shown separately, a row's Total Inventory double-counts Stretch Fiber. Whether this is intentional or a design choice to match a specific business definition needs confirmation (see Risk #1).

**Grand total row (Row 194):**
```excel
=SUM(B187:B193)   [repeats for each column]
```

### 2.6 Current Summary Values (as of 31-Mar-2026 extract)

| Org Name | Cotton (Kgs) | Fiber (Kgs) | Stretch Fiber (Kgs) | Cotton Waste (Kgs) | Total Inventory (Kgs) |
|---|---|---|---|---|---|
| MSL - Fibres U3 | 758,172 | 287,447 | 19,264 | 0 | 1,064,883 |
| MSM - Spinning U1 | 1,664,729 | 326,631 | 59,937 | 52 | 2,051,350 |
| MTM - Spinning U1 | 423,797 | 170,867 | 0 | 0 | 594,664 |
| MTM - Spinning U2 | 1,690,207 | 338 | 0 | 0 | 1,690,546 |
| MTM - Spinning U3 | 168,455 | 102,567 | 54,979 | 0 | 326,001 |
| MTM - Spinning U5 | 192,126 | 327,549 | 0 | 0 | 519,675 |
| MTM - Spinning U6 | 976,610 | 161,879 | 10,492 | 0 | 1,148,981 |
| **TOTAL** | **5,874,096** | **1,377,279** | **144,672** | **52** | **7,396,099** |

### 2.7 How Strategy Reads Raw Material

Strategy reads specific cells from the summary table by row/column position — not by named ranges. The mapping is:

| Summary Row | Org | Cotton cell | Fiber cell |
|---|---|---|---|
| 187 | MSL - Fibres U3 | B187 | C187 |
| 188 | MSM - Spinning U1 | B188 | C188 |
| 189 | MTM - Spinning U1 | B189 | C189 |
| 190 | MTM - Spinning U2 | B190 | C190 |
| 191 | MTM - Spinning U3 | B191 | C191 |
| 192 | MTM - Spinning U5 | B192 | C192 |
| 193 | MTM - Spinning U6 | B193 | C193 |

If rows are inserted or deleted in the detail data section (rows 6–181), or if new orgs are added to the summary section (rows 187–193), all Strategy D-column references will break silently.

---

## 3. Consumption Sheet Logic

### 3.1 Purpose

Holds the complete Oracle transaction register for the reporting period — one row per inventory issue transaction. This sheet is pure input data with no formula cells. All consumption aggregation happens in the Strategy sheet via cross-sheet SUMPRODUCT formulas.

### 3.2 Sheet Layout

| Zone | Rows | Content |
|---|---|---|
| Title | 1 | Report title |
| Period description | 2 | `"April 2026 (01-Apr to 30-Apr) | Transaction Type: Direct Organization Transfer"` |
| Blank | 3 | Empty |
| Column headers | 4 | 8 column names |
| Transaction data | 5–27014 | 27,010 individual transaction rows |

### 3.3 Columns

| Column | Name | Type | Mandatory | Purpose in Strategy |
|---|---|---|---|---|
| A | Item Code | String | Yes | Not directly used in Strategy formulas |
| B | Item Desc | String | No | Not used in Strategy formulas |
| C | Org Name | String | Yes | Filter key in Strategy SUMPRODUCT formulas |
| D | Lot Number | String | No | Not used in Strategy formulas |
| E | Subinventory | String | No | Not used in Strategy formulas (all rows are RM-COTTON or RM-FIBER) |
| F | Txn Type | String | No | All rows = `"Direct Organization Transfer"` in this extract |
| G | Primary Qty (Kgs) | Float | Yes | Core quantity — **all values are negative** |
| H | Date | DateTime | No | Not used in Strategy formulas |
| I | Category | String | Yes | Filter key in Strategy SUMPRODUCT formulas |

### 3.4 Sign Convention

All 27,010 transaction rows have a **negative** Primary Qty. This is Oracle's standard representation for inventory issues — when cotton is transferred out of raw material storage to production, Oracle records a negative quantity. The Strategy formulas apply `ABS()` to recover the positive consumption amount.

```
Oracle Primary Qty (negative) → ABS() → Positive consumption quantity
Example: -210.85714 Kgs → 210.85714 Kgs consumed
```

### 3.5 How Monthly Consumption Is Derived

The Strategy sheet does not use any pre-aggregated total from the Consumption sheet. Instead, it performs live SUMPRODUCT calculations across all 27,010 rows for each org-category combination, then divides by the consumption period (30 days) to get a daily rate.

**Template formula (from Strategy E9):**
```excel
=SUMPRODUCT(
    (Consumption!$C$5:$C$27014 = "MTM - Spinning U6") *
    (Consumption!$I$5:$I$27014 = "Cotton") *
    ABS(Consumption!$G$5:$G$27014)
) / $K$3
```

This formula:
1. Creates a boolean array matching rows where Org Name = the target org
2. Creates a boolean array matching rows where Category = the target category
3. Multiplies both boolean arrays by the absolute quantity array — effectively filtering and summing in one operation
4. Divides by K3 (= 30, the consumption period in days) to get the daily rate

**Business meaning:** Total kilograms of [category] consumed by [org] during April 2026, divided by 30, equals average daily consumption rate in Kgs/day.

### 3.6 What Columns Are Mandatory vs Optional

| Column | Status | Reason |
|---|---|---|
| C — Org Name | **Mandatory** | Primary filter in every Strategy SUMPRODUCT formula |
| G — Primary Qty | **Mandatory** | The quantity value being summed |
| I — Category | **Mandatory** | Secondary filter in every Strategy SUMPRODUCT formula |
| A — Item Code | Optional (for Strategy) | Not referenced in any Strategy formula |
| B — Item Desc | Optional | Not referenced |
| D — Lot Number | Optional | Not referenced |
| E — Subinventory | Optional (for Strategy) | Not referenced in Strategy; used for data quality validation only |
| F — Txn Type | Optional (for Strategy) | The current extract contains only one type; if multiple types are added, this column becomes a required filter |
| H — Date | Optional (for Strategy) | Not referenced in Strategy; required for audit trail and future multi-period support |

### 3.7 Current Period Data

- **Period:** April 2026 (01-Apr-2026 to 30-Apr-2026)
- **Transaction type:** Direct Organization Transfer (100% of rows)
- **Row count:** 27,010 transactions
- **All transaction dates:** 31-Mar-2026 (confirmed from first rows — note: dates are end-of-month Oracle posting date, not actual issue date)

---

## 4. Market Inputs Sheet Logic

### 4.1 Purpose

The single configurable parameter store for all market-facing inputs. Divided into four distinct sections. Sections A and B are data inputs (manual or automated). Sections C and D are calculation and configuration layers.

### 4.2 Section A — Current Market Prices (Rows 5–13)

All cells in this section are **manually entered**. They are the primary targets for market_inputs.py automation.

| Cell | Field | Current Value | Unit | Source | Update Frequency | Automation Candidate |
|---|---|---|---|---|---|---|
| B6 | ICE Cotton Spot Proxy | 82.75 | c/lb | ICE futures market | Daily | Yes — Yahoo Finance CT=F |
| B7 | ICE Cotton Jul'26 Futures | 84.19 | c/lb | ICE CTN26 | Daily | Yes — Yahoo Finance CTN26 |
| B8 | ICE Cotton Dec'26 Futures | 84.56 | c/lb | ICE CTZ26 | Daily | Yes — Yahoo Finance CTZ26 |
| B9 | Cotlook A Index | 92.05 | c/lb | Cotlook | Daily/Weekly | Partial — Cotlook requires subscription |
| B10 | Cycle Low Reference | 60.9 | c/lb | ICE Feb-2026 | Monthly | No — historical reference point, manual decision |
| B11 | PSF Asia Benchmark | 908 | USD/MT | External PSF market | Weekly | Yes — SunSirs scraper |
| B12 | SBP Policy Rate | 0.105 | decimal p.a. | SBP Pakistan | After MPC | Yes — SBP website scraper |
| B13 | USD/PKR Assumption | 281 | PKR/USD | Treasury | Daily | Yes — open.er-api.com |

> **Critical:** B12 stores SBP rate as a **decimal** (0.105 = 10.5%). The market_inputs.py fetcher returns percentage (10.5). Automation must divide by 100 before writing to this cell.

> **Important:** B13 (USD/PKR) is labeled "Treasury Assumption" — it is a treasury-approved rate used for cost calculations, which may differ from the live market rate. The source is treasury, not an automated FX feed. Automation of this field requires a treasury confirmation workflow.

### 4.3 Section B — Predictive Model Forecasts (Rows 16–21)

These are forecast outputs from the ML forecasting model. Currently populated with **placeholder values** marked `"v2.1-draft"`. They must be replaced by actual model output.

| Row | Horizon | Cotton (c/lb) | PSF (USD/MT) | Confidence | Model Version |
|---|---|---|---|---|---|
| 17 | 30-day (Jun'26) | 86.5 | 925 | 72% | v2.1-draft |
| 18 | 60-day (Jul'26) | 89.2 | 940 | 65% | v2.1-draft |
| 19 | 90-day (Aug'26) | 91.0 | 955 | 58% | v2.1-draft |
| 20 | 120-day (Sep'26) | 88.75 | 935 | 52% | v2.1-draft |
| 21 | 180-day (Nov'26) | 85.4 | 920 | 45% | v2.1-draft |

**Which forecasts are consumed by downstream sheets:**

| Cell | Consumed By | Purpose |
|---|---|---|
| B17 (30d Cotton) | Market Inputs B45 | Predicted put strike = ROUNDDOWN(B17 × 0.85, 0) |
| B18 (60d Cotton) | Strategy M3; Market Inputs B32, B44, B46 | 60d forecast is the primary forward planning horizon |
| C17 (30d PSF) | Strategy M3 | PSF 30-day forecast for delay-saving calculation |
| C18 (60d PSF) | Dashboard H10 | Display only |

### 4.4 Section C — Cost-of-Carry Model (Rows 24–32)

This section computes the theoretical fair value of cotton futures and measures market mispricing. All cells except B28 are formula-driven.

**Parameter table:**

| Cell | Field | Value | Unit | Formula | Notes |
|---|---|---|---|---|---|
| B25 | Risk-free rate | 0.105 | decimal p.a. | `=B12` | Mirrors SBP rate from Section A |
| B26 | Storage cost | 0.03 | decimal p.a. | Hardcoded | Internal assumption — 3% p.a. |
| B27 | Insurance & logistics | 0.0125 | decimal p.a. | Hardcoded | Internal assumption — 1.25% p.a. |
| B28 | Days to Jul'26 expiry | 66 | days | Manual | Must be updated monthly — "04-May to Jul window" |
| B29 | Implied convenience yield | 0.05209 | decimal p.a. | `=B25+B26+B27-LN(B7/B6)/(B28/365)` | Derived from market data |
| B30 | Theoretical Jul'26 futures | 84.19 | c/lb | `=B6*EXP((B25+B26+B27-B29)*(B28/365))` | Fair value |
| B31 | Market vs Theoretical | 0.00 | c/lb | `=B7-B30` | Mispricing signal |
| B32 | Predicted 60d vs Theoretical | 5.01 | c/lb | `=B18-B30` | Model-based forward signal |

**Formula explanations:**

**Implied Convenience Yield (B29):**
```
y = r + u + w − ln(F/S) / T

Where:
  r = risk-free rate (SBP, 10.5% = 0.105)
  u = storage cost (3% = 0.03)
  w = insurance & logistics (1.25% = 0.0125)
  F = market futures price (Jul'26 = 84.19 c/lb)
  S = spot proxy price (82.75 c/lb)
  T = time to expiry in years (66/365 = 0.18082)

y = 0.105 + 0.03 + 0.0125 − ln(84.19/82.75) / (66/365)
y = 0.1475 − ln(1.01740) / 0.18082
y = 0.1475 − 0.01725 / 0.18082
y = 0.1475 − 0.09540
y = 0.05209  (5.209% p.a.)
```

**Business meaning:** The convenience yield represents the benefit of holding physical cotton versus a futures contract. A positive value means the physical commodity carries a net benefit (supply tightness, production certainty, etc.) even after carrying costs.

**Theoretical Futures Price (B30):**
```
F_theoretical = S × exp((r + u + w − y) × T)

Substituting y = r + u + w − ln(F/S)/T:
  (r + u + w − y) = ln(F/S)/T
  F_theoretical = S × exp(ln(F/S)/T × T)
                = S × exp(ln(F/S))
                = S × (F/S)
                = F  [the market futures price]

Result: 84.19 c/lb (equals market futures, as expected mathematically)
```

**Business meaning:** When the theoretical price equals the market price, the market is fairly priced. The formula is retained explicitly so any deviation from fair value (caused by model input changes) becomes immediately visible.

**Market vs Theoretical (B31):**
```
Mispricing = Market Futures − Theoretical = 84.19 − 84.19 = 0.00 c/lb
```
Currently zero because convenience yield is derived from market prices. The signal becomes non-zero when the model forecasts deviate from the market.

### 4.5 Section D — Hedge Instrument Parameters (Rows 35–46)

| Cell | Field | Value | Unit | Formula | Update Rule |
|---|---|---|---|---|---|
| B36 | ICE contract size | 50,000 | lbs | Hardcoded | Never — standard exchange specification |
| B37 | Kg to lbs conversion | 2.20462 | lbs/kg | Hardcoded | Never — physical constant |
| B38 | Call strike (current) | 87 | c/lb | Manual | Each strategy review |
| B39 | Put strike (collar floor) | 75 | c/lb | Manual | Each strategy review |
| B40 | Collar call cap | 90 | c/lb | Manual | Each strategy review |
| B41 | Call premium estimate | 2.5 | c/lb | Manual | Use broker quote before execution |
| B42 | Min stock policy | 45 | days | Manual | Board-approved — change requires board approval |
| B43 | Consumption period | 30 | days | Manual | Updated monthly to match Oracle report period |
| B44 | Predicted call strike | 90 | c/lb | `=ROUNDUP(B18,0)` | 60d forecast rounded up to nearest cent |
| B45 | Predicted put strike | 73 | c/lb | `=ROUNDDOWN(B17×0.85,0)` | 30d forecast × 0.85, rounded down |
| B46 | Predicted collar cap | 94 | c/lb | `=ROUNDUP(B18×1.05,0)` | 60d forecast × 1.05, rounded up |

**Business meaning of predicted strikes:**
- Predicted call strike (B44): The system recommends a call option strike at the rounded 60-day forecast — buying protection against cotton rising above the model's expected price.
- Predicted put strike (B45): The collar floor is set 15% below the 30-day forecast — protecting against a price collapse while retaining upside.
- Predicted collar cap (B46): The collar ceiling is 5% above the 60-day forecast — caps the premium cost of the collar structure.

---

## 5. Strategy Sheet Logic

### 5.1 Overview

The Strategy sheet is the procurement decision engine. It operates in two sections:

- **Section A (rows 7–15):** Cotton procurement — physical buying + ICE futures arbitrage
- **Section B (rows 16–24):** Fiber (PSF) procurement — physical POs only, no futures market

The sheet evaluates each organisation independently, computes its stock cover, determines any shortfall against the 45-day policy, and outputs procurement actions.

### 5.2 Parameter Row (Row 3)

Row 3 collects all required parameters from Market Inputs into a single reference row, so all calculation rows reference `$A$3`, `$B$3`, etc. — never Market Inputs directly (except through row 3).

| Cell | Header | Value | Source |
|---|---|---|---|
| A3 | Spot Price (c/lb) | 82.75 | `='Market Inputs'!B6` |
| B3 | R-Free Rate (p.a.) | 0.105 | `='Market Inputs'!B12` |
| C3 | Storage Cost (p.a.) | 0.03 | `='Market Inputs'!B26` |
| D3 | Ins & Logistics (p.a.) | 0.0125 | `='Market Inputs'!B27` |
| E3 | Days to Expiry | 66 | `='Market Inputs'!B28` |
| F3 | Contract Size (lbs) | 50,000 | `='Market Inputs'!B36` |
| G3 | Kg→Lbs Conversion | 2.20462 | `='Market Inputs'!B37` |
| H3 | Call Strike (c/lb) | 87 | `='Market Inputs'!B38` |
| I3 | Put Strike (c/lb) | 75 | `='Market Inputs'!B39` |
| J3 | Min Cover (days) | 45 | `='Market Inputs'!B42` |
| K3 | Consumption Period (days) | 30 | `='Market Inputs'!B43` |
| L3 | PSF Now (USD/MT) | 908 | `='Market Inputs'!B11` |
| M3 | PSF 30d Forecast (USD/MT) | 925 | `='Market Inputs'!C17` |
| N3 | Theoretical Futures (c/lb) | 84.987 | `=A3*EXP((B3+C3+D3)*(E3/365))` |
| O3 | Market Futures (c/lb) | 84.19 | `='Market Inputs'!B7` |
| P3 | Mispricing (c/lb) | -0.797 | `=O3-N3` |
| Q3 | Arb/Contract (USD) | 398.37 | `=($N$3-$O$3)*$F$3/100` |

> **Note on N3 (Theoretical Futures):** Strategy row 3 computes theoretical futures as `S × exp((r+u+w) × T)` — without subtracting the convenience yield. This differs from Market Inputs B30 which uses `S × exp((r+u+w-y) × T)`. With the convenience yield deducted, B30 equals the market price. Without it, N3 produces a slightly higher theoretical price (84.987 vs 84.19), showing the market as underpriced by 0.797 c/lb. The Strategy sheet's theoretical calculation is intentionally conservative — it assumes the full cost-of-carry without a convenience yield offset, to create a more cautious hedge trigger.

### 5.3 Column Definitions (Row 6 Headers)

| Col | Header | Calculation |
|---|---|---|
| A | Step | Action label (A1, A2, ... B1, B2, ...) |
| B | Org Unit | Organisation name |
| C | Action | Procurement decision text |
| D | Stock Before (Kgs) | Raw Material stock (from summary table) |
| E | Daily Rate (Kgs/day) | Total monthly consumption ÷ 30 |
| F | 45-Day Need (Kgs) | Daily Rate × 45 |
| G | Shortfall (Kgs) | MAX(0, 45-Day Need − Stock Before) |
| H | Spot Price (c/lb) | Reference to A3 |
| I | Theoretical Futures (c/lb) | Reference to N3 |
| J | Market Futures (c/lb) | Reference to O3 |
| K | Mispricing (c/lb) | Reference to P3 |
| L | Call Strike / Put Strike | Text label from B38 and B39 |
| M | Physical Qty (Kgs) | = Shortfall (same as G) |
| N | Futures Contracts | CEILING(Shortfall × lbs_per_kg / contract_size, 1) |
| O | Physical Cost (USD) | Physical Qty × lbs_per_kg × spot_price / 100 |
| P | Futures Hedge Cost | Contracts × contract_size × theoretical_futures / 100 |
| Q | Arbitrage Profit (USD) | Contracts × contract_size × mispricing / 100 |
| R | Stock After (Kgs) | Stock Before + Physical Qty Procured |
| S | Days Cover After | Stock After ÷ Daily Rate |
| T | Explanation | Dynamic text string (presentational only) |

---

### 5.4 The Ten Core Formulas

---

#### Formula 1 — Daily Consumption Rate

**Excel Formula (example — MTM - Spinning U6, Cotton):**
```excel
=SUMPRODUCT(
    (Consumption!$C$5:$C$27014="MTM - Spinning U6") *
    (Consumption!$I$5:$I$27014="Cotton") *
    ABS(Consumption!$G$5:$G$27014)
) / $K$3
```

**Plain English:** For each transaction in the Consumption sheet where the Org Name equals this org AND the Category equals "Cotton", sum the absolute value of the quantity. Divide by 30 (the consumption period) to get the average daily rate.

**Business Purpose:** Establishes the org's actual rate of raw material consumption from Oracle ERP data. This is the denominator for all days-cover calculations. It is recalculated automatically every month when the Consumption sheet is refreshed.

**Unit:** Kilograms per day (Kgs/day)

**Current values:**

| Org | Cotton Kgs/day | Fiber Kgs/day |
|---|---|---|
| MTM - Spinning U5 | 9,461.5 | 8,042.1 |
| MTM - Spinning U6 | 35,954.3 | 13,457.5 |
| MSM - Spinning U1 | 56,327.5 | 8,684.9 |
| MTM - Spinning U2 | 58,448.2 | 798.0 |
| MTM - Spinning U1 | 7,865.4 | 19,108.0 |
| MTM - Spinning U3 | 1,042.6 | 4,703.9 |
| MSL - Fibres U3 | 0 (no April data) | 0 (no April data) |

---

#### Formula 2 — Days Cover (Current)

**Excel Formula:**
```excel
= Stock_Before / Daily_Rate
```
*Derived from S-column formula:* `=ROUND(R{row}/E{row}, 1)` applied to pre-procurement stock.

**Plain English:** How many days the current stock will last at the current daily consumption rate.

**Business Purpose:** Measures urgency. If Days Cover is below 45, procurement action is required. The urgency label in column C is manually assigned based on this value:
- < 1 day → EMERGENCY
- 1–15 days → CRITICAL
- 15–30 days → LOW
- 30–44 days → below policy
- 45+ days → NO ACTION

**Unit:** Days

---

#### Formula 3 — 45-Day Requirement

**Excel Formula:**
```excel
=E{row} * $J$3
```
Where `$J$3` = 45 (board-approved minimum stock policy).

**Plain English:** The minimum quantity of raw material that must be on hand to meet 45 days of production at the current daily rate.

**Business Purpose:** The 45-day policy is a board-approved buffer to protect against supply chain disruptions. It is the absolute minimum — the system flags any org falling below this threshold.

**Unit:** Kilograms

---

#### Formula 4 — Shortfall

**Excel Formula:**
```excel
=MAX(0, F{row} - D{row})
```
Where `F` = 45-Day Need and `D` = Stock Before.

**Plain English:** The quantity of raw material that must be procured to bring the org up to exactly 45 days of stock cover. If the org already has more than 45 days of stock, shortfall is zero (never negative).

**Business Purpose:** This is the exact procurement quantity required by policy. The system buys precisely this amount — no more, no less — to restore 45-day cover.

**Unit:** Kilograms

---

#### Formula 5 — Procurement Quantity

**Excel Formula:**
```excel
=G{row}   [for active procurement rows]
=0        [for NO ACTION rows]
```

**Plain English:** The physical quantity of raw material to purchase. Equals the shortfall for all orgs below 45-day cover. Zero for orgs that are already at or above policy.

**Business Purpose:** This is the purchase order quantity. For cotton, it also equals the underlying quantity for the futures hedge. For fiber, it equals the fixed-price PO quantity.

**Unit:** Kilograms

---

#### Formula 6 — Futures Contracts (Cotton Only)

**Excel Formula:**
```excel
=CEILING(G{row} * $G$3 / $F$3, 1)
```
Where:
- `G{row}` = shortfall in Kgs
- `$G$3` = 2.20462 (lbs per kg)
- `$F$3` = 50,000 (lbs per ICE contract)

**Plain English:** Convert the cotton shortfall from kilograms to pounds, then divide by the ICE standard contract size of 50,000 lbs. Round up to the nearest whole contract (CEILING).

**Business Purpose:** ICE cotton futures are standardised at 50,000 lbs per contract. The org cannot buy fractional contracts. Rounding up ensures the hedge covers the full shortfall. The small overhang (rounding excess) is accepted as immaterial.

**Unit:** Number of contracts (integer, always rounded up)

**Example (MTM - Spinning U6):**
```
Shortfall = 641,334 Kgs
× 2.20462 lbs/kg = 1,413,641 lbs
÷ 50,000 lbs/contract = 28.27 contracts
CEILING(28.27, 1) = 29 contracts
```

---

#### Formula 7 — Physical Cost (USD)

**Excel Formula:**
```excel
=(M{row} * $G$3 * $A$3 / 100)
```
Where:
- `M{row}` = physical quantity in Kgs
- `$G$3` = 2.20462 (lbs per kg)
- `$A$3` = 82.75 (spot price in cents/lb)
- `/ 100` = converts cents to dollars

**Plain English:** The total dollar cost of buying the required cotton at the current spot price. Converts kg to lbs, applies the spot price (cents), then converts cents to dollars.

**Business Purpose:** The actual cash outflow for the physical cotton purchase. This is the primary cost line in the procurement budget.

**Unit:** USD

**Example (MSM - Spinning U1):**
```
870,010 Kgs × 2.20462 lbs/kg × 82.75 c/lb ÷ 100 = $1,587,179 USD
```

---

#### Formula 8 — Futures Hedge Cost (Notional, Cotton Only)

**Excel Formula:**
```excel
=(N{row} * $F$3 * $N$3 / 100)
```
Where:
- `N{row}` = number of futures contracts
- `$F$3` = 50,000 (lbs per contract)
- `$N$3` = 84.987 (theoretical futures price in c/lb)
- `/ 100` = converts cents to dollars

**Plain English:** The notional value of the futures hedge position. This is the total dollar value of the cotton futures contracts entered into at the theoretical futures price.

**Business Purpose:** This is not an actual cash payment — it is the notional exposure of the futures position. It represents the total value that the business has locked in through the futures hedge. The actual margin requirement will be a fraction of this figure (broker-dependent).

**Unit:** USD (notional)

---

#### Formula 9 — Arbitrage Profit (USD)

**For Cotton (Futures Arbitrage):**

**Excel Formula:**
```excel
=N{row} * $F$3 * K{row} / 100
```
Where:
- `N{row}` = futures contracts
- `$F$3` = 50,000 (lbs per contract)
- `K{row}` = mispricing in c/lb (= Market Futures − Theoretical = -0.797 c/lb)
- `/ 100` = cents to dollars

**Plain English:** The expected profit (or loss) at convergence from the futures arbitrage position. If market futures are underpriced versus theoretical (K < 0), going long futures means you buy below fair value and profit when the price converges to theoretical at expiry.

**Business Purpose:** In the current workbook, K = -0.797 c/lb (market is below theoretical), so the arbitrage profit is **negative** for long positions. This means the current strategy is not a pure arbitrage but a **procurement hedge** — the futures are entered to lock in a purchase price, not to generate profit. The "arbitrage profit" column quantifies the cost/benefit of the hedge timing.

**Current result (all cotton orgs):** Total arbitrage = -$48,203 (a cost, not profit, at current market prices).

**For Fiber (Delay-Cost Arbitrage):**

**Excel Formula:**
```excel
=(G{row} / 1000) * $M$3 - O{row}
```
Where:
- `G{row}` = shortfall in Kgs
- `/ 1000` = converts Kgs to metric tons
- `$M$3` = 925 (PSF 30-day forecast in USD/MT)
- `O{row}` = current PO cost at today's price

**Plain English:** The cost saving from buying fiber now versus waiting 30 days. If the 30-day PSF forecast is higher than today's price, acting now avoids the price increase.

**Business Purpose:** Unlike cotton, there is no PSF futures market. The "arbitrage" for fiber is purely temporal — buying today at $908/MT to avoid buying in 30 days at the forecast $925/MT. The saving per metric ton is $925 - $908 = $17/MT.

**Current result (all fiber orgs):** Total delay saving = $23,391 USD.

---

#### Formula 10 — Stock After Procurement

**Excel Formula:**
```excel
=D{row} + M{row}
```
Where `D` = Stock Before and `M` = Physical Qty procured.

**Plain English:** The stock level after receiving the procured quantity.

**Business Purpose:** Verifies that the procurement action restores the org to exactly 45 days of cover. The corresponding Days Cover After (column S) should equal exactly 45 for all active procurement rows.

**Days Cover After Formula:**
```excel
=IF(E{row}=0, "N/A", ROUND(R{row}/E{row}, 1))
```
Displays "N/A" for MSL - Fibres U3 which has zero April consumption data.

---

### 5.5 Section A — Cotton Procurement Results (Rows 8–15)

| Step | Org | Action | Stock (Kgs) | Daily (Kgs) | Days Cover | Shortfall (Kgs) | Contracts | Physical Cost (USD) | Arb Profit (USD) |
|---|---|---|---|---|---|---|---|---|---|
| A1 | MTM - Spinning U5 | BUY | 192,126 | 9,462 | 20.3 days | 233,642 | 11 | $426,238 | -$4,382 |
| A2 | MTM - Spinning U6 | BUY | 976,610 | 35,954 | 27.2 days | 641,334 | 29 | $1,170,000 | -$11,553 |
| A3 | MSM - Spinning U1 | BUY | 1,664,729 | 56,328 | 29.6 days | 870,010 | 39 | $1,587,179 | -$15,536 |
| A4 | MTM - Spinning U2 | BUY | 1,690,207 | 58,448 | 28.9 days | 939,964 | 42 | $1,714,798 | -$16,732 |
| A5 | MTM - Spinning U1 | NO ACTION | 423,797 | 7,865 | 53.9 days | 0 | — | 0 | 0 |
| A6 | MTM - Spinning U3 | NO ACTION | 168,455 | 1,043 | 161.6 days | 0 | — | 0 | 0 |
| A7 | MSL - Fibres U3 | MONITOR | 758,172 | 0 | N/A | N/A | — | 0 | 0 |
| **NET** | **ALL COTTON** | | **4,523,672** | | | **2,684,949** | **121** | **$4,898,215** | **-$48,203** |

### 5.6 Section B — Fiber Procurement Results (Rows 17–24)

| Step | Org | Action | Stock (Kgs) | Daily (Kgs) | Days Cover | Shortfall (Kgs) | PO Cost (USD) | Delay Saving (USD) |
|---|---|---|---|---|---|---|---|---|
| B1 | MTM - Spinning U2 | EMERGENCY | 338 | 798 | 0.4 days | 35,572 | $32,299 | $605 |
| B2 | MTM - Spinning U1 | CRITICAL | 170,867 | 19,108 | 8.9 days | 688,992 | $625,605 | $11,713 |
| B3 | MTM - Spinning U6 | CRITICAL | 161,879 | 13,458 | 12.0 days | 443,709 | $402,888 | $7,543 |
| B4 | MTM - Spinning U3 | LOW | 102,567 | 4,704 | 21.8 days | 109,108 | $99,070 | $1,855 |
| B5 | MSM - Spinning U1 | BELOW POLICY | 326,631 | 8,685 | 37.6 days | 64,190 | $58,284 | $1,091 |
| B6 | MTM - Spinning U5 | BELOW POLICY | 327,549 | 8,042 | 40.7 days | 34,347 | $31,187 | $584 |
| B7 | MSL - Fibres U3 | MONITOR | 287,447 | 0 | N/A | N/A | 0 | 0 |
| **NET** | **ALL FIBER** | | **1,089,831** | | | **1,375,917** | **$1,249,332** | **$23,391** |

### 5.7 Grand Total (Row 26)

| Component | Value |
|---|---|
| Cotton physical procurement cost | $4,898,215 |
| Fiber PO cost | $1,249,332 |
| **Total cash outflow** | **$6,147,547** |
| Cotton futures hedge (notional exposure) | $5,141,698 |
| Net arbitrage / delay saving | -$24,812 |

---

## 6. Formula Catalog

| # | Formula Name | Sheet | Exact Excel Formula | Python Equivalent |
|---|---|---|---|---|
| 1 | Inventory aggregation | Raw Material | `=SUMPRODUCT(($D$6:$D$181=A187)*($F$6:$F$181="Cotton")*$E$6:$E$181)` | `df[(df.org==org) & (df.category=="Cotton")]["qty"].sum()` |
| 2 | Total inventory | Raw Material | `=SUM(B187:E187)` | `row["Cotton"] + row["Fiber"] + row["Stretch Fiber"] + row["Cotton Waste"]` |
| 3 | Daily consumption | Strategy | `=SUMPRODUCT((Consumption!$C$5:$C$27014="ORG")*(Consumption!$I$5:$I$27014="Cat")*ABS(Consumption!$G$5:$G$27014))/$K$3` | `abs(cons_df[(cons_df.org==org)&(cons_df.category==cat)]["qty"].sum()) / period_days` |
| 4 | 45-day requirement | Strategy | `=E{r}*$J$3` | `daily_rate * min_cover_days` |
| 5 | Shortfall | Strategy | `=MAX(0,F{r}-D{r})` | `max(0, need - stock)` |
| 6 | Futures contracts | Strategy | `=CEILING(G{r}*$G$3/$F$3,1)` | `math.ceil(shortfall_kg * 2.20462 / 50000)` |
| 7 | Physical cost USD | Strategy | `=(M{r}*$G$3*$A$3/100)` | `(qty_kg * 2.20462 * spot_cents_lb) / 100` |
| 8 | Futures hedge cost | Strategy | `=(N{r}*$F$3*$N$3/100)` | `(contracts * 50000 * theo_futures_cents) / 100` |
| 9 | Cotton arb profit | Strategy | `=N{r}*$F$3*K{r}/100` | `(contracts * 50000 * mispricing_cents) / 100` |
| 10 | Days cover after | Strategy | `=ROUND(R{r}/E{r},1)` | `round(stock_after / daily_rate, 1)` |
| 11 | Fiber PO cost | Strategy | `=(G{r}/1000*$L$3)` | `(shortfall_kg / 1000) * psf_now_usd_mt` |
| 12 | Fiber delay saving | Strategy | `=(G{r}/1000)*$M$3-O{r}` | `(shortfall_kg / 1000) * psf_30d_forecast - fiber_po_cost` |
| 13 | Implied convenience yield | Market Inputs | `=B25+B26+B27-LN(B7/B6)/(B28/365)` | `r + u + w - math.log(futures/spot) / (days/365)` |
| 14 | Theoretical futures | Market Inputs | `=B6*EXP((B25+B26+B27-B29)*(B28/365))` | `spot * math.exp((r+u+w-y) * days/365)` |
| 15 | Strategy theoretical futures | Strategy N3 | `=A3*EXP((B3+C3+D3)*(E3/365))` | `spot * math.exp((r+u+w) * days/365)` |
| 16 | Mispricing | Strategy | `=O3-N3` | `market_futures - theoretical_futures` |
| 17 | Arb per contract | Strategy | `=($N$3-$O$3)*$F$3/100` | `(theoretical - market) * 50000 / 100` |
| 18 | Predicted call strike | Market Inputs | `=ROUNDUP(B18,0)` | `math.ceil(forecast_60d)` |
| 19 | Predicted put strike | Market Inputs | `=ROUNDDOWN(B17*0.85,0)` | `math.floor(forecast_30d * 0.85)` |
| 20 | Predicted collar cap | Market Inputs | `=ROUNDUP(B18*1.05,0)` | `math.ceil(forecast_60d * 1.05)` |

---

## 7. Organization Mapping

### 7.1 All Organizations in the Workbook

Seven organizations appear across all sheets. Each has a specific role:

| Org Name | Type | Cotton Procurement | Fiber Procurement | MSL/MSM/MTM Group |
|---|---|---|---|---|
| MSL - Fibres U3 | Fibre unit | Monitor (no consumption data) | Monitor (no consumption data) | MSL |
| MSM - Spinning U1 | Spinning unit | BUY — 29.6 days cover | Below policy — buy | MSM |
| MTM - Spinning U1 | Spinning unit | No action — 53.9 days | CRITICAL — 8.9 days fiber | MTM |
| MTM - Spinning U2 | Spinning unit | BUY — 28.9 days cover | EMERGENCY — 0.4 days fiber | MTM |
| MTM - Spinning U3 | Spinning unit | No action — 161.6 days | LOW — 21.8 days fiber | MTM |
| MTM - Spinning U5 | Spinning unit | BUY — 20.3 days cover | Below policy — 40.7 days fiber | MTM |
| MTM - Spinning U6 | Spinning unit | BUY — 27.2 days cover | CRITICAL — 12.0 days fiber | MTM |

### 7.2 Per-Organization Data Sources

| Org | Stock Source (Cotton) | Stock Source (Fiber) | Consumption Source |
|---|---|---|---|
| MSL - Fibres U3 | Raw Material B187 | Raw Material C187 | Consumption (0 rows in April) |
| MSM - Spinning U1 | Raw Material B188 | Raw Material C188 | Consumption C=$org, I="Cotton/Fiber" |
| MTM - Spinning U1 | Raw Material B189 | Raw Material C189 | Consumption C=$org, I="Cotton/Fiber" |
| MTM - Spinning U2 | Raw Material B190 | Raw Material C190 | Consumption C=$org, I="Cotton/Fiber" |
| MTM - Spinning U3 | Raw Material B191 | Raw Material C191 | Consumption C=$org, I="Cotton/Fiber" |
| MTM - Spinning U5 | Raw Material B192 | Raw Material C192 | Consumption C=$org, I="Cotton/Fiber" |
| MTM - Spinning U6 | Raw Material B193 | Raw Material C193 | Consumption C=$org, I="Cotton/Fiber" |

### 7.3 Strategy Row Mapping

| Strategy Row | Section | Org | Category | Action Type |
|---|---|---|---|---|
| 8 | A1 | MTM - Spinning U5 | Cotton | Active buy |
| 9 | A2 | MTM - Spinning U6 | Cotton | Active buy |
| 10 | A3 | MSM - Spinning U1 | Cotton | Active buy |
| 11 | A4 | MTM - Spinning U2 | Cotton | Active buy |
| 12 | A5 | MTM - Spinning U1 | Cotton | No action |
| 13 | A6 | MTM - Spinning U3 | Cotton | No action |
| 14 | A7 | MSL - Fibres U3 | Cotton | Monitor |
| 15 | NET | All orgs | Cotton | Aggregation row |
| 17 | B1 | MTM - Spinning U2 | Fiber | Emergency buy |
| 18 | B2 | MTM - Spinning U1 | Fiber | Active buy |
| 19 | B3 | MTM - Spinning U6 | Fiber | Active buy |
| 20 | B4 | MTM - Spinning U3 | Fiber | Active buy |
| 21 | B5 | MSM - Spinning U1 | Fiber | Active buy |
| 22 | B6 | MTM - Spinning U5 | Fiber | Active buy |
| 23 | B7 | MSL - Fibres U3 | Fiber | Monitor |
| 24 | NET | All orgs | Fiber | Aggregation row |
| 26 | TOTAL | All orgs | Cotton + Fiber | Grand total |

---

## 8. Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES                                        │
├──────────────────┬───────────────────────┬──────────────────────────────────┤
│  Oracle ERP      │  Oracle ERP            │  External Markets                │
│  (Monthly)       │  (Monthly)             │  (Daily / Weekly / After MPC)    │
│                  │                        │                                  │
│ MG_STOCK_TILL_   │ MG_TRANSACTION_        │  ICE Cotton Futures              │
│ DATE_MULTIPLE_   │ REGISTER_INV           │  PSF Asia Benchmark (SunSirs)    │
│ UNIT             │                        │  SBP Policy Rate                 │
│                  │                        │  USD/PKR Rate                    │
└────────┬─────────┴──────────┬────────────┴──────────────────────┬───────────┘
         │                    │                                    │
         ▼                    ▼                                    ▼
┌────────────────┐  ┌──────────────────┐              ┌──────────────────────┐
│ Raw Material   │  │ Consumption      │              │ Market Inputs        │
│ Sheet          │  │ Sheet            │              │ Sheet                │
│                │  │                  │              │                      │
│ Detail:        │  │ 27,010 rows      │              │ Sec A: Manual prices │
│ 176 item rows  │  │ All negative qty │              │ Sec B: ML forecasts  │
│                │  │                  │              │ Sec C: Cost-of-carry │
│ Summary:       │  │ 3 cols used:     │              │ Sec D: Hedge params  │
│ 7 orgs ×       │  │ - Org Name       │              │                      │
│ 4 categories   │  │ - Category       │              │ Parameters pulled    │
│ + Total        │  │ - Primary Qty    │              │ into Strategy row 3  │
└────────┬───────┘  └──────────┬───────┘              └──────────┬───────────┘
         │                     │                                  │
         │    D-column:         │    E-column:                    │
         │    Stock by org      │    SUMPRODUCT daily rate        │  Row 3 params
         └──────────────────────┴──────────────────────────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────┐
                              │     Strategy Sheet        │
                              │                          │
                              │  Section A: Cotton        │
                              │  ├─ Daily rate (E col)   │
                              │  ├─ 45-day need (F col)  │
                              │  ├─ Shortfall (G col)    │
                              │  ├─ Contracts (N col)    │
                              │  ├─ Phys cost (O col)    │
                              │  └─ Arb profit (Q col)   │
                              │                          │
                              │  Section B: Fiber         │
                              │  ├─ Daily rate (E col)   │
                              │  ├─ Shortfall (G col)    │
                              │  ├─ PO cost (O col)      │
                              │  └─ Delay saving (Q col) │
                              │                          │
                              │  NET + GRAND TOTAL        │
                              └─────────────┬────────────┘
                                            │
                                            ▼
                              ┌──────────────────────────┐
                              │     Dashboard Sheet       │
                              │                          │
                              │  KPI row (row 6)          │
                              │  Market snapshot (row 10) │
                              │  Coverage by org (13–21)  │
                              │  Scenario P&L (25–34)*    │
                              │  Fiber summary (37–38)*   │
                              └──────────────────────────┘

* = References to Strategy rows that do not yet exist
```

---

## 9. Implementation Blueprint

### 9.1 What Should Move from Excel to Python

| Calculation | Reason to Move |
|---|---|
| Inventory summary (SUMPRODUCT aggregations) | Fragile hardcoded range `$D$6:$D$181`; breaks when Oracle export grows |
| Daily consumption rate calculation | Same fragile range `$C$5:$C$27014`; breaks when transaction count changes |
| Shortfall calculation (MAX logic) | Business logic should be in code, not spreadsheet cells |
| Futures contracts calculation (CEILING) | Core procurement engine logic |
| Physical cost calculation | Core procurement engine output |
| Hedge cost calculation (notional) | Core procurement engine output |
| Arbitrage profit calculation | Core procurement engine output |
| Days cover calculation | Core procurement engine output |
| Implied convenience yield | Finance formula — should be testable and auditable in code |
| Theoretical futures price | Finance formula — should be testable |
| Mispricing signal | Finance formula — should be testable |
| Fiber delay saving | Core procurement engine output |
| Predicted strike calculations (ROUNDUP/DOWN) | Hedge recommendation — should live in code |

### 9.2 What Should Remain in Excel

| Element | Reason to Keep |
|---|---|
| Column T explanation strings | Presentational text — no business logic; changes frequently to match narratives |
| Market Inputs Section A manual entry cells | Human judgment required for treasury rate, cycle low reference, current strikes |
| Market Inputs Section D strike entry (B38-B41) | Manual broker-confirmed values — not to be auto-populated |
| Dashboard formatting and charts | Visual presentation layer |
| Action urgency labels (column C text) | Currently manual text — appropriate to leave as procurement team judgment |

### 9.3 What Should Appear in Streamlit

| Element | Purpose |
|---|---|
| Market snapshot (ICE spot, PSF, SBP rate, USD/PKR) | Real-time market monitoring |
| Days cover gauge per org per category | Immediate visibility of procurement urgency |
| Shortfall table (org × category) | Procurement action list |
| Total procurement cost summary (cotton + fiber) | Budget impact |
| Futures hedge summary (contracts + notional) | Treasury / risk management view |
| Arbitrage / delay saving summary | Commercial justification for acting now vs later |
| 30/60/90-day price forecasts (cotton + PSF) | Forward planning context |
| Cost-of-carry chart (spot vs theoretical vs market futures) | Hedge strategy visualisation |

### 9.4 What Should Be Stored in Supabase

| Dataset | Table | Frequency |
|---|---|---|
| Inventory snapshot (aggregated per org+category) | `inventory_summary` | Monthly |
| Consumption totals (per org+category+month) | `consumption_monthly` | Monthly |
| Market inputs snapshot (all Section A values) | `market_inputs_daily` | Daily |
| Strategy outputs (shortfall, contracts, costs per org) | `procurement_decisions` | Monthly |
| Model forecasts (cotton + PSF horizons) | `price_forecasts` | On model run |

### 9.5 Recommended Architecture

```
Oracle Extraction Layer
────────────────────────
inventory_data.py        → Downloads MG_STOCK_TILL_DATE_MULTIPLE_UNIT.xlsx
consumption_data.py      → Downloads MG_TRANSACTION_REGISTER_INV.xlsx

Cleaning Layer
────────────────────────
clean_inventory.py       → Filters RM-COTTON/RM-FIBER, classifies categories,
                           builds per-org summary (matching Raw Material rows 186–194)
clean_consumption.py     → ABS(qty), classifies categories,
                           builds per-org monthly totals

Commodity Mapping
────────────────────────
commodity_mapper.py      → Item code prefix → category
commodity_mapping.csv    → Configurable prefix→category rules
                           NOTE: must output "Cotton - Waste" (with hyphen)
                           to match workbook formula strings

Market Inputs Layer
────────────────────────
market_inputs.py         → Fetches ICE cotton, PSF, SBP rate, USD/PKR
                           Outputs standardised MarketRecord dataframe
                           NOTE: SBP rate returned as %, must ÷100 before
                           writing to Market Inputs B12

Procurement Engine (Phase 3)
────────────────────────
procurement_engine.py    → Implements all 20 formulas from the Formula Catalog
                           Accepts: inventory_summary, consumption_monthly,
                                    market_inputs
                           Produces: strategy_output (matching Strategy rows 8–26)

Dashboard / Streamlit
────────────────────────
streamlit_app.py         → Reads procurement_engine output
                           Displays: coverage gauges, shortfall table,
                                     hedge summary, market snapshot

Supabase
────────────────────────
                         → Stores all layer outputs for historical analysis
                            and audit trail
```

---

## 10. Risks, Ambiguities, and Open Questions

### 10.1 Confirmed Bugs in Current Workbook

| # | Sheet | Cell | Bug | Impact |
|---|---|---|---|---|
| B1 | Dashboard | B6 | `=Consumption!E27014` references the Subinventory value of the last data row, not a total. Total Consumed KPI is wrong. | Medium — KPI displays wrong value |
| B2 | Dashboard | A14:A20 | Coverage-by-org table reads Strategy row 6 (header) instead of row 8 (first data row). All org names and values are offset by 2 rows. | High — org coverage table is misaligned |
| B3 | Dashboard | A25:F34 | Scenario P&L references Strategy rows 42–51 which do not exist. All cells will be blank or error. | Medium — section is currently incomplete |
| B4 | Dashboard | A38:D38 | Fiber Procurement Summary references Strategy rows 35+ which do not exist. | Medium — section is currently incomplete |

### 10.2 Design Questions Requiring Business Confirmation

| # | Question | Location | Options |
|---|---|---|---|
| Q1 | Is the Fiber summary column (C) intentionally including Stretch Fiber in its total? This causes Stretch Fiber to be counted twice in Total Inventory. | Raw Material C187:C193 | (a) Intended — Stretch Fiber is a sub-type of Fiber for procurement purposes. (b) Bug — Stretch Fiber should be excluded from the Fiber column. |
| Q2 | The category label in the workbook is `"Cotton - Waste"` (with hyphen). Should external systems use this exact string, or should it be changed to `"Cotton Waste"` (without hyphen)? | Raw Material Column F, SUMPRODUCT formulas | Choose one canonical label and apply everywhere. |
| Q3 | Strategy N3 computes theoretical futures **without** the convenience yield, giving 84.987 c/lb. Market Inputs B30 computes it **with** the convenience yield, giving 84.19 c/lb (equal to market). Which is the authoritative theoretical price for the mispricing signal? | Strategy N3 vs Market Inputs B30 | (a) Strategy N3 is correct — full cost-of-carry without convenience yield offset is the intended conservative benchmark. (b) Market Inputs B30 is correct — convenience yield-adjusted theoretical price. |
| Q4 | The 45-day minimum stock policy is hardcoded in Market Inputs B42. Who has authority to change it? | Market Inputs B42 | Confirm this is board-approved and requires formal approval to change. |
| Q5 | MSL - Fibres U3 has zero consumption in April 2026. Is this expected (seasonal) or an Oracle data quality issue? | Consumption sheet, Strategy A7/B7 | Confirm whether this org should be excluded from the strategy calculation or monitored separately. |
| Q6 | The consumption period is hardcoded to 30 days (K3 = `'Market Inputs'!B43`). For February (28 days) or 31-day months, should this be automatically derived from the reporting period, or always 30? | Market Inputs B43 | Confirm whether the 30-day normalisation is intentional (standardised month) or should vary. |
| Q7 | USD/PKR in Market Inputs B13 is sourced from "Treasury" — not the live market rate. What is the approval process for updating this value? Can it be automated? | Market Inputs B13 | Automation of this field requires explicit treasury sign-off. |
| Q8 | "Days to Jul'26 expiry" (Market Inputs B28 = 66 days) is manually entered. This must be updated every month. Should Phase 3 calculate this automatically from today's date to the ICE expiry date? | Market Inputs B28 | Confirm ICE Jul'26 expiry date and automate calculation. |
| Q9 | The "Cycle Low Reference" (B10 = 60.9 c/lb, ICE Feb-2026) is manually maintained. What is its business purpose — is it used in any formula, or is it purely informational? | Market Inputs B10 | Not referenced by any formula in the current workbook. Confirm if it feeds a planned future calculation. |
| Q10 | The action labels in Strategy column C (EMERGENCY, CRITICAL, LOW, etc.) are manually typed. Should these be auto-generated from the days-cover value in Phase 3? | Strategy Column C | Suggested thresholds: ≤1 day = EMERGENCY, ≤15 days = CRITICAL, ≤30 days = LOW, ≤44 days = BELOW POLICY, ≥45 days = NO ACTION. Confirm. |

### 10.3 Scalability Risks

| # | Risk | Current Impact | Mitigation Needed |
|---|---|---|---|
| R1 | Raw Material SUMPRODUCT range hardcoded to rows 6–181. Oracle exports with >176 items will be silently missed. | Low today (176 items), grows with org count | Phase 3 must use dynamic pandas groupby, not hardcoded ranges. |
| R2 | Consumption SUMPRODUCT range hardcoded to rows 5–27014. Different month = different row count. | High — row count will change every month | Phase 3 must read the full DataFrame regardless of row count. |
| R3 | Strategy sheet has hardcoded org list (7 rows). Adding a new spinning unit requires manual row insertion and formula replication. | Low today | Phase 3 engine should loop over orgs dynamically from config. |
| R4 | Dashboard Coverage table references specific Strategy row numbers (E6, E7... = headers). Adding a new org to Strategy breaks Dashboard. | Already broken (B2 above) | Rebuild Dashboard pull logic using named references or stable table structures. |
| R5 | Forecast section B is manually entered. Strategy decisions flow directly from model output without human review step. | Medium risk if model quality degrades | Consider adding a confidence threshold gate — only use forecast if confidence > X%. |
