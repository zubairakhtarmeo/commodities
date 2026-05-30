# PROCUREMENT ENGINE VALIDATION REPORT

**Phase:** 2D — Business Rule Validation  
**Date:** 2026-05-29  
**Source:** `data/strategy/strategies.xlsx` + Phase 1 and Phase 2 deliverables  
**Status:** Analysis only. No code written or modified.

---

## Table of Contents

1. [Category Mapping Validation](#1-category-mapping-validation)
2. [Fiber Logic Validation](#2-fiber-logic-validation)
3. [Consumption Period Validation](#3-consumption-period-validation)
4. [Monthly Roll-Forward Design](#4-monthly-roll-forward-design)
5. [End-to-End Architecture Validation](#5-end-to-end-architecture-validation)
6. [Risk Register](#6-risk-register)

---

## 1. Category Mapping Validation

### 1.1 Item Code Prefixes Found in the Workbook

A full scan of all 176 item rows in `Raw Material` (rows 6–181) and all 26,997 valid transaction rows in `Consumption` (rows 5–27001) reveals that the current Oracle data contains **only two distinct item code prefixes**.

#### Raw Material Sheet — Prefix Inventory

| Prefix | Row Count | Category Values Assigned | Ambiguous |
|---|---|---|---|
| `COT` | 82 | `Cotton` (81 rows), `Cotton - Waste` (1 row) | **YES** |
| `FIB` | 94 | `Fiber` (72 rows), `Stretch Fiber` (22 rows) | **YES** |

**Total prefixes found: 2**

#### Consumption Sheet — Prefix Inventory (valid rows only)

| Prefix | Row Count | Category Values | Ambiguous |
|---|---|---|---|
| `COT` | 26,535 | `Cotton` only | No |
| `FIB` | 462 | `Fiber` (most), `Stretch Fiber` (some) | **YES** |

> **Note:** Rows 27002–27014 contain an embedded summary section (TOTAL row, aggregation headers, per-org totals). These are not transaction data. The rows for `MTM`, `TOTAL`, `ORG NAME`, `MSL`, `MSM` visible in a raw scan are all from this trailing summary block, not from actual item codes. The valid transaction data ends at row 27001.

### 1.2 Prefixes in commodity_mapping.csv That Do NOT Appear in the Workbook

The current `commodity_mapping.csv` defines 11 prefix rules. **9 of those prefixes are absent from the workbook entirely.**

| Prefix in mapping.csv | Present in Workbook | Action Required |
|---|---|---|
| `ICOT` | No | Retain — may appear in future Oracle item codes |
| `COT` | **Yes** | Retain — primary cotton prefix |
| `PSF` | No | Retain — may appear when PSF items are procured locally |
| `FIB` | **Yes** | Retain — primary fiber prefix |
| `STF` | No | Retain — may appear if Stretch Fiber gets its own prefix |
| `VIS` | No | Retain — viscose items exist in forecasting pipeline |
| `VSF` | No | Retain — VSF items exist in forecasting pipeline |
| `CW` | No | Retain — may appear as Cotton Waste prefix grows |
| `FW` | No | Retain — forward compatibility |
| `CHM` | No | Retain — forward compatibility |
| `PKG` | No | Retain — forward compatibility |

**None of the absent prefixes should be removed** — they are forward-compatibility rules that cost nothing and protect against future Oracle item code additions. However, none of them affect the current workbook's calculations because no items with those prefixes currently exist.

### 1.3 Critical Defect: Prefix-Only Classification Is Insufficient

The `CommodityMapper` classifies items by the leading segment of their item code alone. This approach **fails for the current dataset** because both active prefixes (`COT` and `FIB`) each map to two different categories depending on the specific item, not the prefix.

#### COT Prefix — Two-Category Problem

| Item Code | Description | Category in Workbook |
|---|---|---|
| COT-100020 through COT-100056 (81 items) | Cotton varieties (local, imported, organic, recycled) | `Cotton` |
| **COT-100058** | **PAK RECYCLE WASTE LOCAL** | **`Cotton - Waste`** |

The only Cotton Waste item uses the same `COT` prefix as all other cotton varieties. The prefix alone does not distinguish it. **The distinguishing signal is the item number itself (COT-100058) or the description keyword "WASTE"/"RECYCLE WASTE".**

**Current classifier behaviour:** `CommodityMapper` maps `COT → Cotton` (from commodity_mapping.csv). COT-100058 will be classified as `Cotton`, not `Cotton - Waste`. The Raw Material summary formula `=SUMPRODUCT(... ($F$6:$F$181="Cotton - Waste") ...)` will find zero matches and return 0.

**Impact:** The Cotton Waste stock (currently 52 Kgs at MSM - Spinning U1) is silently dropped from the summary. For the current 52 Kgs this is immaterial, but the classification failure is structural.

#### FIB Prefix — Two-Category Problem

| FIB Sub-Type | Count | Description Pattern | Category |
|---|---|---|---|
| Standard fibers | 72 items | Viscose, polyester, bamboo, flax, cottonised fibers | `Fiber` |
| Elastane / stretch | 22 items | Spandex, Lycra, T400, Elastaine, Tencel/Loycell | `Stretch Fiber` |

Both sub-types share the `FIB` prefix. The classification differentiator is the description keyword. The following keywords reliably identify Stretch Fiber items: `SPANDEX`, `LYCRA`, `ELASTAINE`, `ELASPANE`, `ELASTANE`, `T400`, `LOYCELL`, `TENCEL`.

**Current classifier behaviour:** `CommodityMapper` maps `FIB → Fiber`. All 22 Stretch Fiber items will be classified as `Fiber`, not `Stretch Fiber`. The Raw Material summary formula `=SUMPRODUCT(... ($F$6:$F$181="Stretch Fiber") ...)` will return 0 for all orgs.

**Impact:** The Stretch Fiber column in the Raw Material summary (column D, 144,672 Kgs total) will be zero. The Fiber column (column C) will also be wrong because it uses a formula that reads the Category field rather than summing FIB-prefix items directly.

#### Classification Inconsistency Within the Workbook

Three FIB items with elastane keywords in their descriptions are classified as `Fiber` (not `Stretch Fiber`) in the workbook's own Category column F:

| Row | Item Code | Description | Category (Workbook) | Expected |
|---|---|---|---|---|
| 88 | FIB-100108 | SPANDEX BARE YARN R/W 100 PCT PRE-CONSUMER RECYCLED ELASTANE 70D | Fiber | Stretch Fiber |
| 94 | FIB-100108 | SPANDEX BARE YARN R/W 100 PCT PRE-CONSUMER RECYCLED ELASTANE 70D | Fiber | Stretch Fiber |
| 114 | FIB-100080 | IMPORTED SPANDEX BARE YARN R/W 100 PCT PRE-CONSUMER RECYCLED ELASTANE.40D | Fiber | Stretch Fiber |

These items contain "SPANDEX" and "ELASTANE" in their descriptions but are manually classified as `Fiber` in column F. This is either a manual Oracle data entry error or a deliberate business decision (perhaps pre-consumer recycled spandex is treated as a general fiber for procurement purposes, not as a stretch component).

**This must be confirmed with the procurement team before automation.** If the workbook's column F assignment is the authority, description-based classification will produce different results from the workbook.

### 1.4 Recommended Final Mapping Table

The only reliable source of category truth is the **workbook's own column F assignment**. Prefix-based and description-based classification are both insufficient for the current data. The Phase 3 engine must use item-code-to-category lookup from a master reference table, not prefix matching alone.

**Recommended approach for Phase 3:**

| Priority | Method | When to Apply |
|---|---|---|
| 1 | Item code exact match against a master lookup table | Always first — most reliable |
| 2 | Prefix match (`commodity_mapping.csv`) | When item code is not in the master table (new items) |
| 3 | Description keyword regex | Only for brand-new items with unknown codes |

**Recommended master mapping (derived from current workbook):**

| Item Code | Category | Description |
|---|---|---|
| COT-100058 | `Cotton - Waste` | PAK RECYCLE WASTE LOCAL |
| FIB-100040, FIB-100046, FIB-100012, FIB-100020, FIB-100011, FIB-100073, FIB-100074, FIB-100071, FIB-100114, FIB-100118, FIB-100119, FIB-100046, FIB-100040, FIB-100048 | `Stretch Fiber` | Spandex/Lycra/Elastane items |
| All other COT-* | `Cotton` | All cotton varieties |
| All other FIB-* | `Fiber` | All non-stretch fiber varieties |

> **Open question OQ-1:** Are FIB-100108 and FIB-100080 (Spandex Bare Yarn) intentionally classified as `Fiber`? Confirm with procurement team. If yes, add them to the master lookup as `Fiber`. If no, correct the workbook.

> **Open question OQ-2:** The category label is `"Cotton - Waste"` (hyphen-space-hyphen pattern). Confirm this is the canonical label for all systems including Supabase and the Streamlit dashboard.

---

## 2. Fiber Logic Validation

### 2.1 Formula Verification

The Raw Material Fiber column formula (C187, confirmed from workbook formulas):

```excel
=SUMPRODUCT(($D$6:$D$181=A187)*(($F$6:$F$181="Fiber")+($F$6:$F$181="Stretch Fiber"))*$E$6:$E$181)
```

This formula uses an **OR condition** by adding two boolean arrays. The result includes stock for items where Category = `"Fiber"` **or** Category = `"Stretch Fiber"`. Stretch Fiber stock is included in the Fiber column.

The Stretch Fiber column formula (D187):

```excel
=SUMPRODUCT(($D$6:$D$181=A187)*($F$6:$F$181="Stretch Fiber")*$E$6:$E$181)
```

This formula counts Stretch Fiber stock independently.

The Total Inventory formula (F187):

```excel
=SUM(B187:E187)
```

This sums Cotton + Fiber + Stretch Fiber + Cotton Waste — where Fiber already contains Stretch Fiber.

### 2.2 Double-Counting Confirmed and Quantified

Stretch Fiber appears in both the Fiber column (C) and the Stretch Fiber column (D). The Total Inventory column (F) sums both, counting Stretch Fiber twice.

**Per-organisation overstatement:**

| Org Name | Fiber Incl. Stretch (Kgs) | Stretch Fiber Only (Kgs) | Current Total (Kgs) | Overstatement (Kgs) |
|---|---|---|---|---|
| MSL - Fibres U3 | 287,447 | 19,264 | 1,064,883 | **19,264** |
| MSM - Spinning U1 | 326,631 | 59,937 | 2,051,350 | **59,937** |
| MTM - Spinning U1 | 170,867 | 0 | 594,664 | 0 |
| MTM - Spinning U2 | 338 | 0 | 1,690,546 | 0 |
| MTM - Spinning U3 | 102,567 | 54,979 | 326,001 | **54,979** |
| MTM - Spinning U5 | 327,549 | 0 | 519,675 | 0 |
| MTM - Spinning U6 | 161,879 | 10,492 | 1,148,981 | **10,492** |
| **TOTAL** | **1,377,279** | **144,672** | **7,396,099** | **144,672** |

**Total Inventory is overstated by 144,671.87 Kgs (1.96% of reported total).**

Correct Total Inventory (no double-count): **7,251,427 Kgs**

### 2.3 Business Intent Assessment

The workbook contains an embedded aggregation table in the Consumption sheet (rows 27006–27014) with columns labelled `"Fiber (incl. Stretch)"` and `"Stretch Fiber Only"`. This confirms the workbook designer **intended** to show Fiber as an inclusive total and Stretch Fiber as a separately broken-out sub-total.

The double-count in Total Inventory appears to be a design oversight in the `=SUM(B:E)` formula rather than intentional. The Total Inventory column is supposed to represent total physical stock, not a double-counted figure.

### 2.4 Recommendation

**The business must confirm one of two interpretations:**

**Option A — Fiber = "all non-cotton fiber including stretch" (inclusive Fiber column is correct)**
- Total Inventory formula must change to: `Cotton + Fiber(inclusive) + Cotton Waste` — dropping the separate Stretch Fiber term
- The Stretch Fiber column remains informational only and is not added again to the total
- This is the most likely intended design

**Option B — Fiber = "non-stretch fiber only" (exclusive Fiber column)**
- Fiber formula must change to: `=SUMPRODUCT(... ($F$6:$F$181="Fiber") ...)` — removing the Stretch Fiber OR condition
- Total Inventory formula remains `Cotton + Fiber + Stretch Fiber + Cotton Waste` — no double-count
- This approach separates the two fiber categories cleanly

**Impact on Strategy sheet:** The Fiber column (C) feeds Strategy column D (Stock Before) for all fiber procurement rows. If the inclusive Fiber total (Option A) is used, a unit with 100 Kgs of stretch fiber has those 100 Kgs counted in both its fiber stock total and its stretch fiber breakout. For the 45-day cover calculation, what matters is whether Stretch Fiber is physically substitutable for standard Fiber in production. If they are different production inputs (e.g., a unit uses viscose AND separately uses Lycra), their combined physical stock is correctly the sum — but the Total Inventory double-count is still wrong for inventory reporting.

> **Open question OQ-3:** Are Fiber and Stretch Fiber used in the same production process (substitutable) or separate processes (independent requirements)? This determines whether the combined Fiber stock figure is meaningful or whether they should each have independent 45-day cover targets.

---

## 3. Consumption Period Validation

### 3.1 Hardcoded Period Cell

`Market Inputs B43` (Consumption Period) is a **hardcoded literal integer: `30`**. It is not a formula.

`Strategy K3` pulls this via `='Market Inputs'!B43`.

All seven daily consumption rate formulas in the Strategy sheet divide their SUMPRODUCT total by `$K$3 = 30`.

### 3.2 Current Period Accuracy

For April 2026 (01-Apr to 30-Apr), the value of 30 is correct. April has exactly 30 days.

### 3.3 Error for Non-30-Day Months

The hardcoded 30 produces incorrect daily rates for other months:

| Month | Actual Days | Hardcoded Value | Daily Rate Error |
|---|---|---|---|
| February (non-leap) | 28 | 30 | +7.14% overstatement |
| February (leap year) | 29 | 30 | +3.45% overstatement |
| January, March, May, July, August, October, December | 31 | 30 | -3.23% understatement |
| April, June, September, November | 30 | 30 | 0% — correct |

A 7% error in daily rate propagates directly into the 45-day requirement and shortfall calculations. For an org consuming 50,000 Kgs/day, a 7% daily rate error produces a 45-day requirement error of ~157,500 Kgs, and a corresponding shortfall error of the same magnitude. This can trigger or suppress a procurement action incorrectly.

### 3.4 Date Column Anomaly — All Dates Are 2026-03-31

All 26,997 valid transaction rows in the Consumption sheet have a `Date` column value of `2026-03-31`. The period header (row 2) states `"April 2026 (01-Apr to 30-Apr)"`.

This is Oracle's standard behaviour for the `"Direct Organization Transfer"` transaction type: the accounting (GL) date is set to the **end of the previous month** (March 31) rather than the actual transfer date. The actual consumption occurred in April but the accounting entry is posted March 31.

**Consequence for Phase 3:** Any attempt to filter Consumption rows by date using `date == April 2026` or `date.month == 4` will return zero rows. The date column cannot be used to validate that consumption data matches the reporting period. The only reliable period indicator is the text in row 2 (`April 2026 (01-Apr to 30-Apr)`).

### 3.5 Positive-Quantity Rows in Consumption

The Consumption sheet is expected to contain only negative quantities (Oracle inventory issues). A scan found **10 positive-quantity rows**:

| Row | Item Code | Description | Org | Quantity | Date |
|---|---|---|---|---|---|
| 473 | COT-100052 | AFGHANI SHORT STAPLE K IMPORTED | MTM - Spinning U1 | +5,006.48 | 2026-03-31 |
| 1713 | COT-100004 | USA-MEDIUM STAPLE-IMPORTED | MTM - Spinning U2 | +4,892.97 | 2026-03-31 |
| 5810 | COT-100011 | ARGENTINA IMPORTED | MTM - Spinning U2 | +7,373.00 | 2026-03-31 |
| 5811 | COT-100011 | ARGENTINA IMPORTED | MTM - Spinning U2 | +24,925.42 | 2026-03-31 |
| 10591 | COT-100023 | PAK-LOCAL | MTM - Spinning U3 | +4,747.00 | 2026-03-31 |
| 10643 | COT-100053 | PAK BALOCHI LOCAL K LOCAL | MTM - Spinning U3 | +325.71 | 2026-03-31 |
| 10695 | COT-100044 | BRAZILLIAN LOCAL | MTM - Spinning U3 | +1,143.84 | 2026-03-31 |
| 20919 | FIB-100067 | VISCOSE 1.2 DTEX / 44MM INDO BIRLA | MSM - Spinning U1 | +300.00 | 2026-03-31 |
| 21043 | FIB-100110 | VISCOSE 1.2 DETEX 44MM THAI RAYON | MSM - Spinning U1 | +1,060.50 | 2026-03-31 |
| 26713 | FIB-100114 | SPANDEX BARE YARN R/W 100 PCT PRE-CONSUMER RECYLED 70D | MSM - Spinning U1 | +102.60 | 2026-03-31 |

These positive rows represent **returns or reversal transfers** (material transferred back to raw material stock from production). The Strategy formulas use `ABS()` on all quantities, which means these returns are treated as consumption rather than as reductions to consumption. This overstates consumption by:

- Cotton: +48,413.92 Kgs returned (ABS makes them add to consumption, should subtract)
- Fiber: +1,360.50 Kgs returned
- Stretch Fiber: +102.60 Kgs returned

**Total overstatement of monthly consumption due to positive rows: ~49,877 Kgs**

For reference, total monthly consumption is ~6,631,351 Kgs (from the TOTAL row at row 27002). The overstatement is 0.75% — immaterial at the total level but may be significant for individual orgs. MTM - Spinning U3 has 6,217 Kgs of positive rows against a total monthly consumption of approximately 18,845 Kgs (cotton only), a 33% local overstatement for that org.

> **Open question OQ-4:** Should positive-quantity rows (returns/reversals) be excluded from consumption totals, or are they intentionally included in the `ABS()` sum? Confirm with Oracle team whether `"Direct Organization Transfer"` can include both issues and returns, and whether the strategy formula should net them or sum absolute values.

### 3.6 Recommended Dynamic Period Calculation

Rather than relying on the hardcoded 30, Market Inputs B43 should be populated by Phase 3 based on the actual calendar month of the reporting period. This can be derived from the Consumption sheet row 2 text (`"April 2026 (01-Apr to 30-Apr)"`) or from the start/end dates in that string using the `calendar.monthrange()` function.

**Recommended formula logic for Phase 3:**

```
period_days = calendar.monthrange(year, month)[1]   # returns actual days in month
```

Where `year` and `month` are parsed from the Consumption sheet header row text.

---

## 4. Monthly Roll-Forward Design

### 4.1 Current Structure

The workbook holds one month of data:
- Consumption sheet: 27,010 rows (April 2026 data)
- Raw Material sheet: 181 rows (March 31, 2026 snapshot)
- Strategy sheet: hardcoded references to `Consumption!$C$5:$C$27014`

### 4.2 Option A — Reuse Same Workbook Each Month

**How it works:** Each month, the user (or Phase 3 automation) replaces the Consumption data (rows 5–N) and the Raw Material data (rows 6–181) with the new month's Oracle export. The workbook recalculates automatically.

**Advantages:**
- Simpler file management — one file, always current
- Strategy and Dashboard formulas require no changes
- Business users have a single file to work with

**Disadvantages:**
- Previous month's data is overwritten and lost unless separately archived
- The Consumption row range is hardcoded (`$C$5:$C$27014`). If April has 27,010 rows and May has 31,500 rows, the Strategy SUMPRODUCT formulas will miss 4,490 rows. This is a **silent data error**.
- Row count mismatch is the single highest-risk failure mode for this architecture

**Required safeguard if Option A is chosen:**
- Phase 3 must dynamically write the Consumption SUMPRODUCT upper bound to match the actual row count before the workbook recalculates
- Or the formulas must be rewritten to use a named table / dynamic range reference

### 4.3 Option B — Generate Monthly Workbook Versions

**How it works:** Each month, Phase 3 creates a new copy of the workbook (e.g., `strategies_2026_04.xlsx`, `strategies_2026_05.xlsx`) and populates it with that month's data. The master template workbook is never modified.

**Advantages:**
- Complete audit trail — every month's workbook is preserved
- No risk of overwriting prior data
- Supports month-over-month comparisons by reading prior month files
- Row count mismatch is isolated to each monthly file — no cumulative formula drift

**Disadvantages:**
- More files to manage (one per month)
- Requires a clearly named archive folder and file naming convention
- Business users need to know which file is current

### 4.4 Recommendation

**Option B (monthly versions) is recommended**, with the following naming convention:

```
data/strategy/archive/strategies_YYYY_MM.xlsx    [historical]
data/strategy/strategies.xlsx                    [always the current month, symlink or copy]
```

**Rationale:** The Consumption row count will differ every month (transaction volume varies with production). The hardcoded `$C$5:$C$27014` range is already wrong as soon as transaction volume changes. Option B limits the damage to each month in isolation and makes the row-count problem visible immediately (the new month's file will have correct row totals). Option A requires a row-count fix before it is safe to use — and that fix is easy to forget under time pressure.

> **Decision required DEC-1:** Choose Option A or Option B before Phase 3 begins. This decision determines whether Phase 3 writes to a persistent `strategies.xlsx` or generates dated copies.

---

## 5. End-to-End Architecture Validation

### 5.1 Validated Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ORACLE EXTRACTION LAYER                                                     │
│                                                                             │
│  inventory_data.py (Selenium)                                               │
│  └─ Downloads: MG_STOCK_TILL_DATE_MULTIPLE_UNIT.xlsx                        │
│     Input trigger: End-of-month Oracle stock report                         │
│     Business rule: Run after period close, before strategy session          │
│                                                                             │
│  consumption_data.py (Selenium)                                             │
│  └─ Downloads: MG_TRANSACTION_REGISTER_INV.xlsx                             │
│     Input trigger: End-of-month Oracle transaction register                 │
│     Business rule: Filter = "Direct Organization Transfer" only             │
│     Known anomaly: All transaction dates = last day of PREVIOUS month       │
└─────────────────────────────────────────────────────────────────────────────┘
                    │                           │
                    ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  CLEANING LAYER                                                              │
│                                                                             │
│  clean_inventory.py                                                         │
│  └─ Filters: subinventory IN ['RM-COTTON', 'RM-FIBER']                      │
│     Applies: CommodityMapper classification                                 │
│     Outputs: detail_df, summary_df (org × category)                        │
│     WARNING: Prefix-only classification will misclassify:                  │
│       - COT-100058 as Cotton (should be Cotton - Waste)                     │
│       - All FIB Stretch items as Fiber (should be Stretch Fiber)            │
│                                                                             │
│  clean_consumption.py                                                       │
│  └─ ABS(primary_qty) — absorbs both issues and returns                      │
│     Applies: CommodityMapper classification                                 │
│     Outputs: detail_df, summary_df, diagnostics_df                         │
│     WARNING: 10 positive-qty rows exist (returns). ABS() makes them        │
│       add to, not subtract from, consumption totals.                        │
└─────────────────────────────────────────────────────────────────────────────┘
                    │                           │
                    ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  COMMODITY MAPPING LAYER                                                     │
│                                                                             │
│  commodity_mapper.py + commodity_mapping.csv                                │
│  └─ Current rule: COT → Cotton, FIB → Fiber                                 │
│     Required rule: Item code exact match takes priority over prefix         │
│     Required additions:                                                     │
│       - COT-100058 → Cotton - Waste (exact match rule)                      │
│       - FIB-100040, FIB-100046, FIB-100011, FIB-100012, FIB-100020,        │
│         FIB-100071, FIB-100073, FIB-100074, FIB-100114, FIB-100118,        │
│         FIB-100119 → Stretch Fiber (exact match rules)                      │
│     Category label must be: "Cotton - Waste" (hyphen preserved)            │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  MARKET INPUTS AUTOMATION LAYER                                              │
│                                                                             │
│  market_inputs.py                                                           │
│  └─ fetch_cotton_price() → ICE CT=F spot proxy (c/lb)                       │
│     fetch_fx_rate()     → USD/PKR (note: treasury rate ≠ market rate)      │
│     fetch_sbp_rate()    → SBP Policy Rate (returns %, must ÷100)           │
│     fetch_psf_price()   → SunSirs PSF benchmark (USD/MT)                   │
│                                                                             │
│  VALIDATION GAPS:                                                           │
│  - B43 (consumption period) must be updated to actual month days           │
│  - B28 (days to expiry) must be recalculated from today to ICE expiry      │
│  - B13 (USD/PKR) requires treasury sign-off — cannot be auto-populated     │
│  - B10 (Cycle Low) is manually maintained — not automated                  │
│  - PSF SunSirs commodity_id=839 is UNVERIFIED                              │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STRATEGIES WORKBOOK                                                         │
│                                                                             │
│  data/strategy/strategies.xlsx                                              │
│  └─ Raw Material sheet: receives clean_inventory output                     │
│     Consumption sheet: receives clean_consumption output                    │
│     Market Inputs sheet: receives market_inputs output                      │
│     Strategy sheet: all formula-driven from the above                      │
│     Dashboard: formula-driven from Strategy                                 │
│                                                                             │
│  KNOWN BUGS (must be fixed before Phase 3 reads Strategy output):          │
│  - Dashboard B6: wrong cell reference (Subinventory, not Total Consumed)    │
│  - Dashboard A14:A20: offset by 2 rows (reads header, not first data row)  │
│  - Consumption row range hardcoded to 27014 rows (breaks monthly)          │
│  - Raw Material detail range hardcoded to rows 6–181 (breaks on growth)    │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PROCUREMENT ENGINE (Phase 3 — not yet built)                               │
│                                                                             │
│  procurement_engine.py                                                      │
│  └─ Inputs: inventory_summary, consumption_monthly, market_inputs_df       │
│     Computes: all 20 formulas from the Formula Catalog                      │
│     Outputs: strategy_output (matching Strategy rows 8–26)                 │
│                                                                             │
│  PRE-CONDITIONS BEFORE PHASE 3 CAN BEGIN:                                  │
│  1. Category mapping defect resolved (item-code exact match support)        │
│  2. Fiber double-count business rule confirmed (OQ-3)                       │
│  3. Positive-qty consumption rows handling confirmed (OQ-4)                 │
│  4. Consumption period dynamic calculation agreed (OQ-6)                    │
│  5. Monthly roll-forward design decided (DEC-1)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  DASHBOARD / STREAMLIT                                                       │
│                                                                             │
│  streamlit_app.py (existing)                                                │
│  └─ Must NOT read from strategies.xlsx directly (bugs B1, B2 exist)        │
│     Must read from procurement_engine output                                │
│     Dashboard bugs B1 and B2 must be fixed before linking to workbook      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Layer Boundary Contracts

For Phase 3 to work correctly, each layer must pass a defined contract to the next.

| Layer | Output Contract | Known Violations |
|---|---|---|
| Oracle Extraction | Raw Excel files, unmodified | None — Selenium scripts are read-only |
| Cleaning | `detail_df` with columns: item_code, org_name, category, primary_qty (or consumption_qty), date | Category column will be wrong for COT-100058 and all Stretch Fiber FIB items |
| Commodity Mapping | Category strings matching workbook formula strings exactly | `"Cotton - Waste"` with hyphen required; `"Stretch Fiber"` required |
| Market Inputs | DataFrame with metric_name, metric_value, unit, status | SBP rate unit mismatch (% vs decimal); PSF ID unverified; USD/PKR not automated |
| Procurement Engine | strategy_output with columns matching Strategy rows 8–26 | Not yet built |
| Dashboard | All KPIs sourced from procurement_engine output | Dashboard bugs B1, B2 must be fixed |

---

## 6. Risk Register

The following risks, if unmitigated, will produce **incorrect procurement recommendations**. They are ordered by severity (impact × likelihood).

---

### RISK-01 — Prefix-Only Classification Drops Cotton Waste and Stretch Fiber

**Severity:** HIGH  
**Likelihood:** CERTAIN — will fail on every run until fixed

**Description:** The `CommodityMapper` assigns categories by item code prefix alone. The current Oracle data has only two prefixes (`COT`, `FIB`), each mapping to two categories. COT maps to both `Cotton` and `Cotton - Waste`. FIB maps to both `Fiber` and `Stretch Fiber`. Prefix-only classification cannot distinguish them.

**Failure mode:** All Cotton Waste stock and all Stretch Fiber stock will appear as zero in the Raw Material summary. Strategy formulas that reference `"Cotton - Waste"` and `"Stretch Fiber"` categories will find no matching rows.

**Mitigation required:** Add item-code exact-match lookup as Priority 1 in `CommodityMapper`, before prefix matching. A master lookup CSV with item_code → category overrides is the minimal fix. The current `commodity_mapping.csv` format would need extension to support exact-code rules in addition to prefix rules, OR a separate `item_code_overrides.csv` file is introduced.

---

### RISK-02 — Consumption Row Range Hardcoded (Breaks Every Month)

**Severity:** HIGH  
**Likelihood:** CERTAIN — will fail when monthly row count differs from 27,010

**Description:** Strategy formulas reference `Consumption!$C$5:$C$27014`. This exact row count is specific to April 2026. Any other month with more or fewer transactions will be silently truncated or padded with blank rows.

**Failure mode:** If May 2026 has 31,000 transactions, the last 3,990 are silently excluded from all SUMPRODUCT calculations. Daily consumption rates will be understated. Shortfall will be overstated. Procurement actions may be incorrectly triggered.

**Mitigation required:** Phase 3 must write all consumption calculations in Python (using the full DataFrame regardless of row count) and write only the aggregate results to the workbook — not the raw transaction data. The Strategy SUMPRODUCT formulas must be replaced by static values populated by the procurement engine.

---

### RISK-03 — Hardcoded Consumption Period (30 days) Errors for Non-April Months

**Severity:** MEDIUM  
**Likelihood:** CERTAIN for 8 out of 12 months

**Description:** Market Inputs B43 = 30 (literal integer). Daily consumption rate = total / 30. For February (28 days) this overstates daily rate by 7.1%. For 31-day months this understates daily rate by 3.2%.

**Failure mode:** Wrong daily rate → wrong 45-day requirement → wrong shortfall → wrong procurement quantity and wrong contract count.

**Mitigation required:** Phase 3 must compute `period_days = calendar.monthrange(year, month)[1]` from the Oracle report period and write it to B43 before recalculating.

---

### RISK-04 — Fiber Double-Count Overstates Total Inventory by 144,672 Kgs

**Severity:** MEDIUM  
**Likelihood:** CERTAIN — confirmed from workbook formulas

**Description:** The Fiber column in the Raw Material summary includes Stretch Fiber. The Total Inventory column then adds the Stretch Fiber column again. Stretch Fiber (144,672 Kgs) is counted twice in every Total Inventory figure.

**Failure mode:** Total Inventory is overstated by 144,672 Kgs (1.96%). Strategy rows use the individual Cotton and Fiber columns (not the Total), so shortfall calculations are unaffected by this specific double-count. However, the Dashboard Total Inventory KPI and any capacity analysis derived from it will be incorrect.

**Mitigation required:** Confirm business intent (OQ-3). Fix `=SUM(B:E)` to exclude the double-counted Stretch Fiber term if inclusive Fiber design is confirmed.

---

### RISK-05 — Positive-Quantity Rows in Consumption Overstate Daily Rate

**Severity:** MEDIUM  
**Likelihood:** PRESENT in current data (10 rows confirmed)

**Description:** 10 rows in the Consumption sheet have positive quantities (returns/reversals). The Strategy formula uses `ABS()` on all rows, treating returns as additional consumption rather than as reductions. For MTM - Spinning U3, the positive rows (6,217 Kgs) represent approximately 33% of that org's reported monthly cotton consumption (18,845 Kgs), potentially triggering a false procurement action.

**Failure mode:** Daily consumption rate overstated → 45-day requirement overstated → shortfall appears larger than actual → unnecessary procurement triggered.

**Mitigation required:** Phase 3 should handle sign-convention explicitly: only rows where `primary_qty < 0` count toward consumption. Positive-quantity rows (`primary_qty > 0`) should either be excluded or netted against the negative sum. Confirm with business (OQ-4).

---

### RISK-06 — MSL - Fibres U3 Has Zero Consumption — Procurement Status Unknown

**Severity:** MEDIUM  
**Likelihood:** Present in April 2026; unknown for other months

**Description:** MSL - Fibres U3 has no rows in the April 2026 Consumption sheet. Both its cotton and fiber daily rates compute to zero. The Strategy sheet flags it as `MONITOR` and shows `N/A` for Days Cover. The workbook does not distinguish between "this org is truly inactive" and "the Oracle extract missed this org's transactions".

**Failure mode:** If MSL - Fibres U3 is active but its transactions are missing from the Oracle extract (e.g., wrong business unit filter in `consumption_data.py`), its true shortfall would be invisible. The org's cotton stock is 758,172 Kgs — a procurement decision cannot be made without knowing whether it is being consumed.

**Mitigation required:** Confirm with Oracle team whether MSL - Fibres U3 uses a different transaction type (not `"Direct Organization Transfer"`) or a different subinventory code that is excluded by the current extract filter. Phase 3 should raise an explicit alert when any org's consumption is zero, distinguishing between "zero transactions found" and "confirmed inactive".

---

### RISK-07 — Days-to-Expiry (B28) Must Be Updated Monthly But Is Manual

**Severity:** MEDIUM  
**Likelihood:** Will be wrong if not updated before each monthly run

**Description:** Market Inputs B28 = 66 (hardcoded). This represents the number of days from the run date to the ICE Jul'26 futures expiry. As of the April 2026 run date, 66 days was correct. By May 2026, the correct value is approximately 36 days. By June 2026, it will be approximately 5 days (contract near expiry).

**Failure mode:** Wrong time-to-expiry → wrong convenience yield (B29) → wrong theoretical futures price (N3) → wrong mispricing signal (P3) → wrong arbitrage profit/loss calculation (Q column in Strategy).

**Mitigation required:** Phase 3 must calculate this automatically: `(ICE_expiry_date - today).days`. The ICE Jul'26 contract expires on a specific date (typically the last business day of the delivery month — approximately July 25, 2026). This date should be a configuration parameter, not hardcoded.

---

### RISK-08 — SBP Rate Unit Mismatch (Percentage vs Decimal)

**Severity:** MEDIUM  
**Likelihood:** Will occur on every automated run unless explicitly handled

**Description:** `market_inputs.py` `fetch_sbp_rate()` returns the rate as a percentage (10.5 for 10.5%). Market Inputs B12 stores it as a decimal (0.105). If automation writes 10.5 to B12 directly, all downstream cost-of-carry calculations will use an interest rate of 1050% p.a.

**Failure mode:** Theoretical futures price becomes astronomically large → mispricing signal is wildly wrong → hedge decisions are based on fictional values.

**Mitigation required:** Phase 3 automation must divide the SBP fetched rate by 100 before writing to B12. This is already documented in the specification but must be implemented as an explicit transformation step, not a comment.

---

### RISK-09 — PSF SunSirs Commodity ID Unverified

**Severity:** LOW-MEDIUM  
**Likelihood:** May be wrong — commodity_id=839 is a default assumption

**Description:** `market_inputs.py` uses `psf_commodity_id=839` as the SunSirs commodity ID for PSF (Polyester Staple Fiber). This default was set based on the existing VSF connector (commodity_id=1057) and a placeholder estimate. The actual SunSirs commodity ID for PSF must be verified at sunsirs.com.

**Failure mode:** A wrong commodity ID returns the wrong commodity's price. PSF price is used to calculate fiber PO cost and delay saving. A wrong PSF price leads to incorrect fiber procurement cost estimates and incorrect delay-saving signals.

**Mitigation required:** Verify the correct SunSirs commodity ID for PSF by navigating to the SunSirs commodity catalogue before Phase 3 connects this to the workbook.

---

### RISK-10 — Strategy Org Order vs Raw Material Summary Row Order May Diverge

**Severity:** LOW  
**Likelihood:** Low currently, grows as the org list changes

**Description:** Strategy reads stock levels by hardcoded cell reference: `='Raw Material'!B187`, `='Raw Material'!B188`, etc. The mapping between strategy rows and summary rows is positional. If a new organisation is added to Oracle and appears in the Raw Material summary (e.g., a new row 187 is inserted), all Strategy D-column references shift silently.

**Failure mode:** Strategy computes shortfall for the wrong org's stock level. An org that appears to have surplus may actually have a deficit, and vice versa.

**Mitigation required:** Phase 3 should write summary data to the workbook using a named table with org name as the row key, not by row position. Or Phase 3 should write strategy outputs directly (bypassing the workbook entirely) and read org stock by name from the DataFrame.

---

### RISK-11 — Treasury USD/PKR Rate Cannot Be Automated

**Severity:** LOW  
**Likelihood:** Certain — confirmed from workbook labeling

**Description:** Market Inputs B13 is labeled "Treasury Assumption" — it is not the live market rate but a treasury-approved internal FX rate. The market_inputs.py `fetch_fx_rate()` returns the live open.er-api.com rate, which may differ from the treasury rate by 5–15 PKR.

**Failure mode:** Physical cost and fiber PO cost calculations in Strategy use the spot price (c/lb × USD) without a PKR conversion, so USD/PKR does not directly affect any current Strategy formula. However, if Dashboard or Phase 3 adds PKR-denominated cost outputs, using the live FX rate instead of the treasury rate will produce figures inconsistent with management accounts.

**Mitigation required:** Do not auto-populate B13. Retain it as a manual treasury-approved input. `fetch_fx_rate()` can be used for informational display in Streamlit but must not overwrite B13 without explicit treasury confirmation.

---

### RISK-12 — Dashboard Bugs B1 and B2 Will Persist Until Explicitly Fixed

**Severity:** MEDIUM  
**Likelihood:** Certain — confirmed from formula inspection

**Description:** (Previously documented as B1 and B2 in the Specification.)
- **B1:** Dashboard B6 references `=Consumption!E27014` — the Subinventory field of the last data row, not a total consumption figure. The "Total Consumed" KPI on the Dashboard is displaying a subinventory code string, not a number.
- **B2:** Dashboard Coverage table (A14:A20) starts at `=Strategy!A6` (header row "Step") instead of `=Strategy!A8` (first org row). All org names and coverage values are offset by 2 rows.

**Failure mode:** Management decisions made from the Dashboard are based on incorrect data. The Total Consumed KPI is wrong and the Coverage table does not show the correct orgs.

**Mitigation required:** These bugs must be fixed in the workbook before Phase 3 reads Dashboard output or before the Streamlit app is connected to Strategy data. They are cell-reference corrections, not formula logic changes.

---

## Summary of Required Actions Before Phase 3

| # | Action | Owner | Urgency |
|---|---|---|---|
| A1 | Confirm canonical category label: `"Cotton - Waste"` vs `"Cotton Waste"` | Business / Procurement | BLOCKING |
| A2 | Add item-code exact-match lookup to CommodityMapper for COT-100058 and all Stretch Fiber FIB codes | Engineering | BLOCKING |
| A3 | Confirm Fiber column business rule: inclusive (Fiber + Stretch Fiber) or exclusive (Fiber only) | Business | BLOCKING |
| A4 | Confirm handling of positive-qty consumption rows (returns) | Business / Oracle team | BLOCKING |
| A5 | Choose monthly roll-forward design: Option A (reuse) or Option B (monthly versions) | Business / Engineering | BLOCKING |
| A6 | Verify PSF SunSirs commodity ID (currently 839, unverified) | Engineering | HIGH |
| A7 | Fix Dashboard bugs B1 (B6 reference) and B2 (coverage table offset) | Engineering | HIGH |
| A8 | Confirm whether MSL - Fibres U3 zero consumption is expected or an Oracle extract issue | Business / Oracle team | HIGH |
| A9 | Confirm whether FIB-100108 and FIB-100080 (Spandex Bare Yarn) are intentionally `Fiber` not `Stretch Fiber` | Business | MEDIUM |
| A10 | Confirm whether action urgency labels (EMERGENCY, CRITICAL, etc.) should be auto-generated or remain manual | Business | MEDIUM |
| A11 | Confirm ICE Jul'26 contract expiry date for automated days-to-expiry calculation | Treasury | MEDIUM |
| A12 | Confirm 45-day minimum stock policy authority and change approval process | Management / Board | LOW |
