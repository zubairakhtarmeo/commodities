# PROCUREMENT ENGINE FINAL SPECIFICATION

**Phase:** 2E — Business Decision Integration  
**Date:** 2026-05-29  
**Status:** Final design. No code written or modified.  
**Supersedes:** PROCUREMENT_ENGINE_SPECIFICATION.md, PROCUREMENT_ENGINE_VALIDATION.md  

---

## Table of Contents

1. [Approved Business Decisions](#1-approved-business-decisions)
2. [Final Category Mapping Design](#2-final-category-mapping-design)
3. [Final Consumption Calculation Design](#3-final-consumption-calculation-design)
4. [Final Inventory Calculation Design](#4-final-inventory-calculation-design)
5. [Final Workbook Versioning Design](#5-final-workbook-versioning-design)
6. [Final Procurement Architecture](#6-final-procurement-architecture)
7. [Gap Analysis — Current State vs Required State](#7-gap-analysis--current-state-vs-required-state)
8. [Updated Architecture Diagram](#8-updated-architecture-diagram)
9. [Updated Data Flow](#9-updated-data-flow)
10. [Files to Modify in Phase 3](#10-files-to-modify-in-phase-3)
11. [Remaining Risks](#11-remaining-risks)

---

## 1. Approved Business Decisions

These decisions are final and override all previous assumptions. Every design choice in this document derives from them.

---

### Decision 1 — Fiber and Stretch Fiber Are Independent Commodities

**Rule:** Fiber and Stretch Fiber are completely separate material categories. They must never be combined or aggregated at any point in the pipeline.

**Scope of separation:**
- Inventory snapshot (Raw Material)
- Consumption totals (Consumption)
- Daily consumption rate (Strategy)
- 45-day stock requirement (Strategy)
- Shortfall calculation (Strategy)
- Procurement action (Strategy)
- Forecasting inputs
- All Strategy calculations

**Implementation implication:** The Strategy sheet currently treats all non-cotton fiber as a single "Fiber" procurement category. Phase 3 must produce independent procurement recommendations for Fiber and Stretch Fiber. This adds a third commodity type to the procurement engine and requires one new section in the Strategy output (Section C — Stretch Fiber, alongside existing Section A — Cotton and Section B — Fiber).

---

### Decision 2 — Net Consumption = Issues Minus Returns

**Rule:** Consumption must be calculated as:

```
Net Consumption (Kgs) = Total Issues − Total Returns
```

Where:
- Issues: Oracle rows with `primary_qty < 0` → contribution = `abs(primary_qty)`
- Returns: Oracle rows with `primary_qty > 0` → contribution = `primary_qty` (reduces net)

**Replacing:** The current `ABS(primary_qty)` approach which treats returns as additional consumption.

**Numeric example:**
```
Issues:   10 rows totalling −1,000 Kgs   →  +1,000 Kgs consumed
Returns:   2 rows totalling  +200 Kgs    →    −200 Kgs consumed
Net:                                     =  +800 Kgs consumed (not 1,200)
```

**Implementation implication:** `clean_consumption.py` must change its `consumption_qty` derivation. The Strategy daily rate formula equivalent must also change. Row-level contribution to net consumption is `−primary_qty` (negate the Oracle sign), which makes issues positive and returns negative. Aggregation is then a straightforward sum of `−primary_qty`.

---

### Decision 3 — Three-Level Category Assignment Priority

**Rule:** Every item is classified by attempting three levels in order. The first level that produces a match wins. Lower levels are never consulted once a match is found.

| Level | Method | Trigger |
|---|---|---|
| 1 | Exact item code lookup | Always attempted first |
| 2 | Description keyword rules | When Level 1 yields no match |
| 3 | Prefix rules | When Levels 1 and 2 both yield no match |

**Level 1 — Exact Item Code (confirmed overrides from business review):**

| Item Code | Category | Reason |
|---|---|---|
| `COT-100058` | `Cotton Waste` | Only waste cotton item; description = "PAK RECYCLE WASTE LOCAL" |
| `FIB-100108` | `Fiber` | Spandex Bare Yarn, pre-consumer recycled — business classifies as general fiber |
| `FIB-100080` | `Stretch Fiber` | Imported Spandex Bare Yarn — business classifies as stretch component |

> Level 1 is not exhaustive — it covers only items where prefix + description logic would produce the wrong result. All other items fall through to Level 2 or Level 3.

**Level 2 — Description Keywords:**

| Keyword(s) in Description | Assigned Category | Notes |
|---|---|---|
| `RECYCLE`, `WASTE`, `NOIL` | `Cotton Waste` | Catches future COT-prefixed waste items beyond COT-100058 |
| `SPANDEX`, `LYCRA`, `ELASTANE`, `ELASTAINE`, `ELASPANE`, `40D`, `70D` | `Stretch Fiber` | Denier markers (40D, 70D) reliably identify elastane yarns |

> Level 2 keywords are matched case-insensitively against the full item description. All Level 2 rules are evaluated against the actual description string, not the item code.

> **Important:** `40D` and `70D` as standalone keywords may produce false positives if item descriptions contain numeric codes ending in "D" for reasons unrelated to denier. These keywords should be applied only when accompanying fiber context is present (e.g., description also contains FIB prefix OR description contains elastane-related material names). Document this as a watch item for Phase 3 implementation.

**Level 3 — Prefix Rules (final approved mapping):**

| Prefix | Category | Change from Previous |
|---|---|---|
| `ICOT` | `Cotton` | No change |
| `COT` | `Cotton` | No change |
| `PSF` | `Fiber` | No change |
| `FIB` | `Fiber` | No change |
| `STF` | `Stretch Fiber` | No change |
| `VIS` | `Fiber` | **CHANGED** — was `Viscose`; viscose is a fiber for procurement purposes |
| `VSF` | `Fiber` | **CHANGED** — was `Viscose`; viscose staple fiber classifies as Fiber |
| `CW` | `Cotton Waste` | No change |
| `FW` | `Fiber Waste` | No change |
| `CHM` | `Chemicals` | No change |
| `PKG` | `Packaging` | No change |

> The `Viscose` category is retired as a standalone procurement category. All viscose items (VIS, VSF prefix or viscose keyword in description) are now classified as `Fiber`. This aligns with how the workbook already treats viscose items — they carry `FIB` prefix and are assigned to the `Fiber` category.

---

### Decision 4 — Workbook Versioning

**Rule:**
- `data/strategy/strategies.xlsx` is always the current month's working file.
- At end of each monthly cycle, the current file is copied to `data/strategy/archive/strategies_YYYY_MM.xlsx` before the next month's data is loaded.
- Historical archive files are **never modified** after archiving.

**Naming convention:**
```
data/strategy/archive/strategies_2026_04.xlsx  ← April 2026 (archived)
data/strategy/archive/strategies_2026_05.xlsx  ← May 2026 (archived)
data/strategy/strategies.xlsx                  ← current month (active)
```

---

### Decision 5 — Cotton Waste Auto-Detection

**Rule:** COT-100058 is currently the only known Cotton Waste item, but future waste items are expected as the Oracle item master grows. The system must automatically detect and classify Cotton Waste using description keywords: `RECYCLE`, `WASTE`, `NOIL`.

This is already covered by Decision 3, Level 2 description rules. It is stated separately here to make the future-proofing intent explicit: no manual update to the exact-code lookup is required for new waste items with recognisable descriptions.

---

## 2. Final Category Mapping Design

### 2.1 Canonical Category Names

These are the exact strings that must be used in all Python code, CSV files, database schemas, and Excel formula strings. Case and punctuation are exact.

| Category | Canonical String | Notes |
|---|---|---|
| Cotton | `Cotton` | All cotton varieties |
| Fiber | `Fiber` | Viscose, polyester, bamboo, flax, recycled fiber — all non-stretch, non-cotton fiber |
| Stretch Fiber | `Stretch Fiber` | Spandex, Lycra, elastane, denier yarns used as stretch components |
| Cotton Waste | `Cotton Waste` | **No hyphen.** Previous workbook had `"Cotton - Waste"` with hyphen — this is now standardised to `"Cotton Waste"` without hyphen. |
| Fiber Waste | `Fiber Waste` | Future use |
| Chemicals | `Chemicals` | Future use |
| Packaging | `Packaging` | Future use |
| Unmapped | `Unmapped` | Classification failure — must never appear in a production run |

> **Breaking change on Cotton Waste label:** The current workbook has `"Cotton - Waste"` (with hyphen) in column F and in SUMPRODUCT formula strings. Phase 3 must either (a) write `"Cotton - Waste"` to the workbook to preserve compatibility with existing formulas, or (b) update all workbook formula strings to `"Cotton Waste"`. Since Phase 3 will replace all workbook SUMPRODUCT formulas with Python-computed values anyway, option (b) is the correct path. The workbook's column F and any formula strings referencing this label must be updated to `"Cotton Waste"` during Phase 3 workbook population.

### 2.2 Three-Level Classification Logic (Pseudocode)

```
function classify(item_code, description):
    
    # Level 1: exact item code match
    code = normalise(item_code)  # strip, uppercase
    if code in EXACT_CODE_OVERRIDES:
        return EXACT_CODE_OVERRIDES[code]
    
    # Level 2: description keywords
    if description is not None:
        desc = normalise(description)  # strip, uppercase
        if any keyword in {RECYCLE, WASTE, NOIL} found in desc:
            return "Cotton Waste"
        if any keyword in {SPANDEX, LYCRA, ELASTANE, ELASTAINE, ELASPANE, 40D, 70D} found in desc:
            return "Stretch Fiber"
    
    # Level 3: prefix rules
    prefix = item_code.split("-")[0].uppercase()
    if prefix in PREFIX_MAP:
        return PREFIX_MAP[prefix]
    
    # No match
    return "Unmapped"
```

### 2.3 Exact Code Override Table (Item Code Overrides CSV)

The exact-code overrides should be stored in a separate file from the prefix rules, to keep them independently maintainable.

**Proposed file:** `scripts/item_code_overrides.csv`

```
# item_code_overrides.csv
# Level 1 classification: exact item code → category
# These take priority over ALL description and prefix rules.
# Add new rows here only when description-based detection cannot reliably classify the item.
#
item_code,category
COT-100058,Cotton Waste
FIB-100108,Fiber
FIB-100080,Stretch Fiber
```

### 2.4 Updated Prefix Rules CSV

**File:** `scripts/commodity_mapping.csv` — updated with VIS and VSF reclassified to Fiber.

```
# commodity_mapping.csv  (Level 3 — prefix rules only)
# Last updated: 2026-05-29
# Changes: VIS → Fiber (was Viscose), VSF → Fiber (was Viscose)
item_code_prefix,category
ICOT,Cotton
COT,Cotton
PSF,Fiber
FIB,Fiber
STF,Stretch Fiber
VIS,Fiber
VSF,Fiber
CW,Cotton Waste
FW,Fiber Waste
CHM,Chemicals
PKG,Packaging
```

### 2.5 Description Keyword Rules (Level 2)

These are evaluated in order. First match wins.

| Priority | Keywords (case-insensitive) | Assigned Category |
|---|---|---|
| 1 | `RECYCLE`, `WASTE`, `NOIL` | `Cotton Waste` |
| 2 | `SPANDEX`, `LYCRA`, `ELASTANE`, `ELASTAINE`, `ELASPANE` | `Stretch Fiber` |
| 3 | `40D`, `70D` | `Stretch Fiber` (denier markers — apply only when FIB prefix or no prefix match) |

> The denier keywords (`40D`, `70D`) are weak signals. They must be applied with care. In Phase 3, these should be matched as whole tokens (word boundaries or space-delimited) to avoid false positives from codes like `FIB-100040` (which contains "40" but as an item number, not denier).

---

## 3. Final Consumption Calculation Design

### 3.1 Net Consumption Definition

For any org-category pair over a reporting period:

```
Net_Consumption(org, category, period) =
    SUM(abs(primary_qty) for all rows where primary_qty < 0, org matches, category matches)
  − SUM(primary_qty       for all rows where primary_qty > 0, org matches, category matches)
```

Equivalently, since issues are negative and returns are positive in Oracle:

```
Net_Consumption = −SUM(primary_qty)  for all matching rows
```

This single formula handles both issues (negative → positive consumption) and returns (positive → negative consumption) correctly in one pass. The result is always the net amount consumed.

### 3.2 Daily Consumption Rate

```
Daily_Rate(org, category) = Net_Consumption(org, category, period) / period_days
```

Where `period_days` is derived from the actual calendar days in the reporting month, **not** hardcoded to 30.

```
period_days = calendar.monthrange(year, month)[1]
```

The reporting year and month are parsed from the Consumption sheet row 2 text: `"April 2026 (01-Apr to 30-Apr) | ..."`.

### 3.3 Required Columns in clean_consumption Output

| Column | Name | Formula | Note |
|---|---|---|---|
| 1 | `item_code` | From Oracle | Required |
| 2 | `item_desc` | From Oracle | Required for Level 2 classification |
| 3 | `org_name` | From Oracle | Required — primary filter key |
| 4 | `lot_number` | From Oracle | Audit trail only |
| 5 | `subinventory` | From Oracle | Audit trail only |
| 6 | `txn_type` | From Oracle | Should always = "Direct Organization Transfer" |
| 7 | `primary_qty` | From Oracle | Raw Oracle value (negative = issue, positive = return) |
| 8 | `net_qty` | `= −primary_qty` | Positive = consumption, Negative = return. Replaces `consumption_qty`. |
| 9 | `date` | From Oracle | Note: all dates = last day of PREVIOUS month (Oracle GL posting) |
| 10 | `year_month` | Derived | Period string, e.g. `"2026-04"` |
| 11 | `category` | Classified | Via 3-level CommodityMapper |
| 12 | `txn_sign` | `issue` or `return` | Based on sign of primary_qty — for diagnostics |

> The column previously named `consumption_qty` is renamed to `net_qty` to reflect that it is a signed net contribution (positive for issues, negative for returns), not always-positive.

### 3.4 Summary Aggregation (monthly_summary_df)

For each unique `(org_name, category)` pair:

```
monthly_summary[org, category] = SUM(net_qty) where net_qty comes from −primary_qty
```

This produces the net consumption in Kgs that should feed Strategy daily rate calculation.

The summary pivot shape:

```
org_name | Cotton | Cotton Waste | Fiber | Stretch Fiber | Total Net Consumed
```

Where `Total Net Consumed` = sum of all category columns for that org.

### 3.5 Diagnostics Report (unchanged)

The `txn_diagnostics()` function remains useful for validating sign conventions. It is not affected by the net consumption change — it always reports raw counts of positive/negative/zero rows before any calculation.

### 3.6 Handling of MSL - Fibres U3 Zero Consumption

MSL - Fibres U3 has zero consumption in April 2026. The clean_consumption output will have no rows for this org. The procurement engine must:
1. Detect when an org has zero rows in the consumption DataFrame.
2. Classify this as `MONITOR` status — not an error, not a zero daily rate.
3. Raise a warning log entry stating the org name, period, and category, so the Oracle team can verify the extract is complete.

---

## 4. Final Inventory Calculation Design

### 4.1 Summary Table Structure (Corrected)

The Raw Material summary must have **no double-counting**. Fiber and Stretch Fiber are independent columns. The Total Inventory is the sum of all four material categories only once each.

**Required summary shape:**

```
org_name | Cotton | Fiber | Stretch Fiber | Cotton Waste | Total Inventory
```

**Total Inventory formula (correct):**

```python
summary["Total Inventory"] = (
    summary.get("Cotton", 0)
  + summary.get("Fiber", 0)
  + summary.get("Stretch Fiber", 0)
  + summary.get("Cotton Waste", 0)
)
```

No category is included twice.

### 4.2 Workbook Raw Material Fiber Formula Change

The current workbook formula for the Fiber column is:

```excel
=SUMPRODUCT(($D$6:$D$181=A187)*(($F$6:$F$181="Fiber")+($F$6:$F$181="Stretch Fiber"))*$E$6:$E$181)
```

This must change to:

```excel
=SUMPRODUCT(($D$6:$D$181=A187)*($F$6:$F$181="Fiber")*$E$6:$E$181)
```

The Total Inventory formula currently `=SUM(B187:E187)` remains correct once the Fiber formula no longer double-counts Stretch Fiber.

> **In Phase 3:** The procurement engine will write pre-computed values to the workbook summary cells rather than relying on workbook SUMPRODUCT formulas. These formula changes apply to the workbook template but are superseded by Phase 3 writing aggregate results directly.

### 4.3 Cotton Waste in Inventory

Cotton Waste must appear in the inventory summary as a separate column. Currently COT-100058 (52 Kgs at MSM - Spinning U1) is the only waste item. It is not currently included in any Strategy procurement action — the Strategy has no Section for Cotton Waste procurement.

**Phase 3 treatment of Cotton Waste:** Report as inventory only. Do not generate procurement shortfall calculations for Cotton Waste. Cotton Waste is a by-product, not a procured input.

### 4.4 clean_inventory.py Summary Output Contract

```
detail_df columns: item_code, description, org_name, primary_qty, subinventory_code, category
summary_df columns: org_name, Cotton, Cotton Waste, Fiber, Stretch Fiber, Total Inventory
```

Column order in summary_df must be: `org_name` | categories sorted alphabetically | `Total Inventory`. This matches the existing `sorted(category_cols)` logic in `clean_inventory.py`.

With the fixed mapper, `sorted(category_cols)` will produce: `Cotton, Cotton Waste, Fiber, Stretch Fiber` — alphabetical order. This is the canonical column order for all downstream consumers.

---

## 5. Final Workbook Versioning Design

### 5.1 Directory Structure

```
data/
└── strategy/
    ├── strategies.xlsx                 ← current month (active working file)
    └── archive/
        ├── strategies_2026_04.xlsx     ← April 2026 (archived, read-only)
        ├── strategies_2026_05.xlsx     ← May 2026 (archived, read-only)
        └── strategies_YYYY_MM.xlsx     ← pattern for all future months
```

### 5.2 Monthly Cycle Steps

```
Step 1: Archive
    Copy data/strategy/strategies.xlsx
    To  data/strategy/archive/strategies_YYYY_MM.xlsx
    Where YYYY_MM = current month being closed
    
Step 2: Validate archive
    Confirm archive file exists and is readable before proceeding.
    
Step 3: Populate current workbook
    Write new month's data to data/strategy/strategies.xlsx:
    - Raw Material sheet: inventory detail + summary
    - Consumption sheet: transaction detail + period header
    - Market Inputs sheet: market data (Sections A, C, D auto; Section B from model)
    
Step 4: Recalculate
    Write procurement engine outputs to Strategy sheet.
    
Step 5: Confirm
    Validate Strategy totals match procurement engine output.
    Log completion.
```

### 5.3 Archive File Integrity Rules

1. Archive files are never opened in write mode after archiving.
2. Archive file names include year and month — they are uniquely named and never overwritten.
3. The `data/strategy/archive/` directory must exist before any archive operation. Phase 3 must create it if absent.
4. If `strategies_YYYY_MM.xlsx` already exists in the archive directory (e.g., a re-run), Phase 3 must halt and require explicit confirmation before overwriting.

---

## 6. Final Procurement Architecture

### 6.1 Three Commodity Sections

The Strategy output now covers three independent commodity sections:

| Section | Commodity | Procurement Mechanism | Market Reference |
|---|---|---|---|
| A | Cotton | Physical purchase + ICE futures hedge | ICE CT=F spot; ICE CTN26/CTZ26 futures |
| B | Fiber | Physical PO only | PSF Asia Benchmark (SunSirs) |
| C | Stretch Fiber | Physical PO only | No current market feed — manual price |

> **Open item:** No automated market price source exists for Stretch Fiber (Lycra/Spandex). Phase 3 must accept a manually-entered Stretch Fiber price (analogous to Market Inputs Section A). This is a new Market Inputs field that does not currently exist in the workbook.

### 6.2 Procurement Logic Per Commodity

All three sections share the same shortfall logic. Only the pricing mechanism and hedge instrument differ.

**For any org-commodity pair:**

```
daily_rate = net_consumption(org, category) / period_days
need_45d   = daily_rate × 45
shortfall  = max(0, need_45d − stock_on_hand)
procure    = shortfall
stock_after = stock_on_hand + procure
days_cover  = stock_after / daily_rate   [or N/A if daily_rate = 0]
```

**Cotton-specific (Section A):**
```
futures_contracts  = CEILING(shortfall × 2.20462 / 50000, 1)
physical_cost_usd  = shortfall × 2.20462 × spot_cents_lb / 100
futures_notional   = contracts × 50000 × theoretical_futures_cents / 100
arb_profit_usd     = contracts × 50000 × mispricing_cents / 100
```

**Fiber-specific (Section B):**
```
fiber_po_cost_usd  = (shortfall / 1000) × psf_now_usd_mt
delay_saving_usd   = (shortfall / 1000) × psf_30d_forecast − fiber_po_cost
```

**Stretch Fiber-specific (Section C):**
```
stretch_po_cost_usd = (shortfall / 1000) × stretch_fiber_price_usd_mt
delay_saving_usd    = (shortfall / 1000) × stretch_30d_forecast − stretch_po_cost
```
> Stretch fiber price and 30-day forecast are new Market Inputs fields to be added in Phase 3.

### 6.3 Action Urgency Labels

Auto-generated by Phase 3 procurement engine based on days_cover_before:

| Condition | Label |
|---|---|
| `daily_rate == 0` | `MONITOR` |
| `days_cover < 1` | `EMERGENCY` |
| `1 ≤ days_cover < 15` | `CRITICAL` |
| `15 ≤ days_cover < 30` | `LOW` |
| `30 ≤ days_cover < 45` | `BELOW POLICY` |
| `days_cover ≥ 45` | `NO ACTION` |

These replace the manually-typed action labels in Strategy column C.

### 6.4 Cost-of-Carry Parameters

No change from existing specification. Strategy N3 formula (without convenience yield) remains the authoritative theoretical futures price for the mispricing signal:

```
theoretical_futures = spot × exp((r + u + w) × T)
mispricing          = market_futures − theoretical_futures
```

Where `r` = SBP rate (decimal), `u` = storage cost (0.03), `w` = insurance/logistics (0.0125), `T` = days_to_expiry / 365.

---

## 7. Gap Analysis — Current State vs Required State

### 7.1 commodity_mapper.py

| Dimension | Current State | Required State | Change Type |
|---|---|---|---|
| Level 1 (exact code) | Not supported — no exact-code lookup | Must support `item_code_overrides.csv` lookup before prefix or description matching | **New capability** |
| Level 2 (description) | Fires only when prefix lookup fails; uses a fixed regex list | Must fire before prefix lookup; must include `40D`, `70D` denier keywords; must prioritise Cotton Waste before Stretch Fiber | **Restructure order + add keywords** |
| Level 3 (prefix) | Fires as the first lookup step | Must fire only when Levels 1 and 2 fail | **Restructure order** |
| VIS prefix | Maps to `Viscose` | Must map to `Fiber` | **Category change** |
| VSF prefix | Maps to `Viscose` | Must map to `Fiber` | **Category change** |
| `Viscose` category | Exists as a standalone category | Retired — all viscose classified as `Fiber` | **Remove** |
| `classify()` method signature | `classify(item_code, description=None)` | No change to signature; change to internal logic | **Internal restructure** |
| New constructor | None for overrides file | `from_overrides_csv(path)` or extended `from_csv_or_default()` accepting two file paths | **New constructor** |

### 7.2 commodity_mapping.csv

| Row | Current | Required | Change |
|---|---|---|---|
| `VIS,Viscose` | VIS → Viscose | VIS → Fiber | **Update** |
| `VSF,Viscose` | VSF → Viscose | VSF → Fiber | **Update** |
| All other rows | Unchanged | Unchanged | No change |

### 7.3 item_code_overrides.csv (new file)

| Dimension | Current State | Required State | Change Type |
|---|---|---|---|
| File existence | Does not exist | Must be created at `scripts/item_code_overrides.csv` | **New file** |
| Contents | N/A | `COT-100058 → Cotton Waste`, `FIB-100108 → Fiber`, `FIB-100080 → Stretch Fiber` | **New file** |

### 7.4 clean_consumption.py

| Dimension | Current State | Required State | Change Type |
|---|---|---|---|
| Consumption quantity derivation | `consumption_qty = primary_qty.abs()` | `net_qty = −primary_qty` | **Breaking change** |
| Column name | `consumption_qty` | `net_qty` | **Rename** |
| OUTPUT_COLS | Includes `consumption_qty` | Must include `net_qty`; remove `consumption_qty` | **Update** |
| Aggregation in summary | `agg(total_consumption=("consumption_qty", "sum"))` | `agg(total_net=("net_qty", "sum"))` | **Update** |
| Summary column name | `Total Consumed` | `Total Net Consumed` (or `Total Consumed` if business prefers brevity) | **Rename** |
| Zero-consumption org handling | Returns empty rows; no warning | Must log warning when org has zero rows in consumption DataFrame | **New behaviour** |
| Diagnostics (`txn_diagnostics`) | Runs on raw `primary_qty` before ABS — correct | No change required | No change |
| Period days | Not computed — caller passes `filter_month` for filtering only | Must expose `period_days` derived from the actual month being processed | **New output** |

### 7.5 clean_inventory.py

| Dimension | Current State | Required State | Change Type |
|---|---|---|---|
| Mapper level 1 (exact code) | Not used | Will work correctly once `CommodityMapper` is updated | No code change here |
| Summary pivot | Includes all categories via pivot_table | Unchanged — pivot_table naturally separates all categories if mapper is correct | No change |
| Total Inventory formula | `summary[category_cols].sum(axis=1)` | Unchanged — produces correct total when Fiber no longer double-counts Stretch Fiber | No change |
| Cotton Waste column | Will be zero (mapper misclassifies COT-100058 as Cotton) | Will be correct once mapper is fixed | Mapper fix required; no code change here |
| Stretch Fiber column | Will be zero (mapper misclassifies all FIB Stretch items as Fiber) | Will be correct once mapper is fixed | Mapper fix required; no code change here |

### 7.6 strategies.xlsx (Workbook)

| Sheet | Cell/Range | Current Formula/Value | Required Formula/Value | Change Type |
|---|---|---|---|---|
| Raw Material | C187:C193 | `=SUMPRODUCT(... ("Fiber")+("Stretch Fiber") ...)` | `=SUMPRODUCT(... ("Fiber") ...)` | **Formula change — remove OR clause** |
| Raw Material | F5 header | `"Cotton Waste"` may still show as `"Cotton - Waste"` | Must be `"Cotton Waste"` (no hyphen) | **Label change** |
| Raw Material | F6:F181 | Category values include `"Cotton - Waste"` | Must be updated to `"Cotton Waste"` for consistency | **Data change** |
| Raw Material | Summary SUMPRODUCT | References `"Cotton - Waste"` | Must reference `"Cotton Waste"` | **Formula string change** |
| Strategy | Row 3 parameters | 12 parameters (A3:Q3) | Must add Stretch Fiber price and forecast parameters | **New parameter cells** |
| Strategy | Rows 8–15 (Section A) | Cotton procurement rows | No change to logic; auto-generated by Phase 3 | Phase 3 writes values |
| Strategy | Rows 17–24 (Section B) | Fiber procurement rows | No change to logic; auto-generated by Phase 3 | Phase 3 writes values |
| Strategy | Rows 26–33 (new Section C) | Does not exist | Must be added: Stretch Fiber procurement for 7 orgs | **New rows** |
| Strategy | Rows 35+ (Totals) | Does not exist | Grand total for all three sections | **New rows** |
| Market Inputs | B43 | `30` (hardcoded) | `=DAYS(date_end, date_start)` or populated by Phase 3 with `calendar.monthrange()` result | **Phase 3 writes value** |
| Market Inputs | B28 | `66` (hardcoded) | Populated by Phase 3 with `(ICE_expiry_date − run_date).days` | **Phase 3 writes value** |
| Market Inputs | New rows (Stretch Fiber) | Does not exist | New Section for Stretch Fiber current price + 30/60d forecasts | **New cells** |
| Dashboard | B6 | `=Consumption!E27014` (bug — Subinventory cell) | Must reference net consumption total | **Bug fix** |
| Dashboard | A14:A20 | `=Strategy!A6` (bug — header row offset) | Must start at `=Strategy!A8` (first data row) | **Bug fix** |

### 7.7 market_inputs.py

| Dimension | Current State | Required State | Change Type |
|---|---|---|---|
| Four connectors | Cotton, FX, SBP, PSF | Unchanged | No change |
| PSF commodity_id | 839 (unverified default) | Must be verified against SunSirs before use | **Verification required** |
| Stretch Fiber price | Not supported | New connector or manual input field required | **New capability (Phase 3)** |
| B43 population | Not populated by automation | Phase 3 must write `period_days` to B43 | **New write step** |
| B28 population | Not populated by automation | Phase 3 must write `days_to_expiry` to B28 | **New write step** |
| SBP rate unit | Returns %, must ÷100 before writing B12 | Unchanged constraint | No change — document and enforce |

---

## 8. Updated Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        ORACLE EXTRACTION LAYER                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  inventory_data.py (Selenium — no changes needed)                            ║
║  └─ Downloads MG_STOCK_TILL_DATE_MULTIPLE_UNIT.xlsx                          ║
║     Subinventory filter: RM-COTTON, RM-FIBER                                 ║
║                                                                              ║
║  consumption_data.py (Selenium — no changes needed)                          ║
║  └─ Downloads MG_TRANSACTION_REGISTER_INV.xlsx                               ║
║     Transaction filter: Direct Organization Transfer                         ║
║     Known: all dates = last day of previous month (Oracle GL posting)        ║
╚══════════════════════════════════════════════════════════════════════════════╝
                    │ inventory xlsx                │ consumption xlsx
                    ▼                               ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                         COMMODITY MAPPING LAYER                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  commodity_mapper.py  [REQUIRES CHANGES]                                     ║
║  ├─ Level 1: item_code_overrides.csv (exact code → category)                 ║
║  │    COT-100058 → Cotton Waste                                               ║
║  │    FIB-100108 → Fiber                                                     ║
║  │    FIB-100080 → Stretch Fiber                                             ║
║  │                                                                           ║
║  ├─ Level 2: description keywords                                             ║
║  │    RECYCLE/WASTE/NOIL        → Cotton Waste                               ║
║  │    SPANDEX/LYCRA/ELASTANE/   → Stretch Fiber                             ║
║  │    ELASTAINE/ELASPANE/40D/70D                                             ║
║  │                                                                           ║
║  └─ Level 3: commodity_mapping.csv (prefix → category)                       ║
║       COT→Cotton, FIB→Fiber, VIS→Fiber, VSF→Fiber, STF→Stretch Fiber ...    ║
║                                                                              ║
║  item_code_overrides.csv  [NEW FILE]                                         ║
║  commodity_mapping.csv    [UPDATED: VIS→Fiber, VSF→Fiber]                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
              │ mapper instance                │ mapper instance
              ▼                               ▼
╔═════════════════════════╗     ╔═════════════════════════════════════════════╗
║  INVENTORY CLEANING     ║     ║  CONSUMPTION CLEANING                       ║
╠═════════════════════════╣     ╠═════════════════════════════════════════════╣
║  clean_inventory.py     ║     ║  clean_consumption.py  [REQUIRES CHANGES]  ║
║  [minor changes only]   ║     ║                                             ║
║                         ║     ║  net_qty = −primary_qty                    ║
║  Filters RM-COTTON,     ║     ║  (issues = positive, returns = negative)   ║
║  RM-FIBER subinventory  ║     ║                                             ║
║                         ║     ║  Aggregation: SUM(net_qty) per org+category ║
║  Summary pivot:         ║     ║  period_days = calendar.monthrange(Y,M)[1] ║
║    Cotton               ║     ║                                             ║
║    Cotton Waste         ║     ║  Output:                                    ║
║    Fiber                ║     ║    detail_df (net_qty per transaction)      ║
║    Stretch Fiber        ║     ║    summary_df (net per org × category)      ║
║    Total Inventory      ║     ║    diagnostics_df (sign convention report)  ║
║  (no double-count)      ║     ║    period_days (integer)                   ║
╚═════════════════════════╝     ╚═════════════════════════════════════════════╝
              │ inventory_summary_df          │ consumption_summary_df + period_days
              └─────────────────┬────────────┘
                                │
                                ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                      MARKET INPUTS AUTOMATION LAYER                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  market_inputs.py  [REQUIRES ADDITIONS]                                      ║
║  ├─ fetch_cotton_price()  → ICE spot + futures (c/lb)                        ║
║  ├─ fetch_fx_rate()       → USD/PKR live (display only — not written to B13) ║
║  ├─ fetch_sbp_rate()      → SBP Policy Rate (%, must ÷100 before B12)        ║
║  ├─ fetch_psf_price()     → PSF Asia Benchmark USD/MT [verify ID=839]        ║
║  └─ [NEW] fetch_stretch_price() → Stretch Fiber benchmark USD/MT [TBD]       ║
║                                                                              ║
║  Phase 3 also writes to workbook:                                            ║
║  ├─ B43 ← period_days (from clean_consumption)                               ║
║  └─ B28 ← (ICE_expiry_date − today).days                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
              │ market_inputs_df
              │ inventory_summary_df
              │ consumption_summary_df
              │ period_days
              ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                     PROCUREMENT ENGINE  (Phase 3 — new)                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  procurement_engine.py                                                       ║
║                                                                              ║
║  For each org × commodity in {Cotton, Fiber, Stretch Fiber}:                ║
║    stock_before    ← inventory_summary[org, commodity]                       ║
║    daily_rate      ← consumption_summary[org, commodity] / period_days       ║
║    need_45d        ← daily_rate × 45                                         ║
║    shortfall       ← max(0, need_45d − stock_before)                         ║
║    action_label    ← urgency_label(days_cover_before)                        ║
║                                                                              ║
║  Cotton-specific:                                                            ║
║    contracts       ← CEILING(shortfall × 2.20462 / 50000, 1)                ║
║    physical_cost   ← shortfall × 2.20462 × spot_cents / 100                 ║
║    theoretical_F   ← spot × exp((r+u+w) × T)                               ║
║    mispricing      ← market_futures − theoretical_F                         ║
║    arb_profit      ← contracts × 50000 × mispricing / 100                   ║
║                                                                              ║
║  Fiber-specific:                                                             ║
║    po_cost_usd     ← (shortfall / 1000) × psf_now_usd_mt                    ║
║    delay_saving    ← (shortfall / 1000) × psf_30d − po_cost                 ║
║                                                                              ║
║  Stretch Fiber-specific:                                                     ║
║    po_cost_usd     ← (shortfall / 1000) × stretch_now_usd_mt                ║
║    delay_saving    ← (shortfall / 1000) × stretch_30d − po_cost             ║
║                                                                              ║
║  Output: strategy_df — one row per org × commodity, all calculated fields    ║
╚══════════════════════════════════════════════════════════════════════════════╝
              │ strategy_df
              ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                         WORKBOOK POPULATION LAYER                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  (Part of Phase 3 procurement_engine.py or a separate workbook_writer.py)   ║
║                                                                              ║
║  Step 1: Archive  strategies.xlsx → archive/strategies_YYYY_MM.xlsx          ║
║  Step 2: Write Raw Material sheet  (inventory detail + corrected summary)    ║
║  Step 3: Write Consumption sheet   (transaction detail + period header)      ║
║  Step 4: Write Market Inputs       (Sections A, C, D; Section B from model)  ║
║  Step 5: Write Strategy sheet      (procurement engine output rows)          ║
║  Step 6: Validate totals           (cross-check engine output vs written)    ║
╚══════════════════════════════════════════════════════════════════════════════╝
              │ strategies.xlsx (updated)
              ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                        STREAMLIT DASHBOARD                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Reads from: strategy_df (procurement engine output) + market_inputs_df      ║
║  Does NOT read from strategies.xlsx directly (avoids workbook bug exposure)  ║
║                                                                              ║
║  Displays: coverage gauges, shortfall tables (3 commodities),                ║
║            hedge summary, market snapshot, forecast charts                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 9. Updated Data Flow

### 9.1 Per-Org Processing Flow

```
For each org in {MSL-Fibres-U3, MSM-Spinning-U1, MTM-Spinning-U1,
                 MTM-Spinning-U2, MTM-Spinning-U3, MTM-Spinning-U5, MTM-Spinning-U6}:

    For each commodity in {Cotton, Fiber, Stretch Fiber}:
    
        1. STOCK   = inventory_summary[org][commodity]
        
        2. CONSUMP = consumption_summary[org][commodity]   ← NET (issues − returns)
        
        3. RATE    = CONSUMP / period_days                 ← period_days = actual calendar days
        
        4. NEED    = RATE × 45
        
        5. SHORT   = max(0, NEED − STOCK)
        
        6. ACTION  = urgency_label(STOCK / RATE)
        
        7. COST    = compute_cost(commodity, SHORT, market_inputs)
        
        8. OUTPUT  = {org, commodity, STOCK, RATE, NEED, SHORT, ACTION, COST, ...}
```

### 9.2 Cotton Waste Data Flow (Inventory-Only)

```
Oracle inventory extract
    → clean_inventory.py
        → CommodityMapper (Level 1: COT-100058 → Cotton Waste)
            → summary_df[org]["Cotton Waste"] = qty_on_hand
                → Reported in inventory only
                → No shortfall calculation
                → No procurement action
```

### 9.3 Market Inputs Data Flow

```
ICE CT=F (Yahoo Finance)      → fetch_cotton_price()   → spot_cents_lb, futures_cents_lb
open.er-api.com               → fetch_fx_rate()        → usd_pkr_live [display only]
sbp.org.pk                    → fetch_sbp_rate()       → sbp_pct ÷ 100 → sbp_decimal
SunSirs (ID=839, verify)      → fetch_psf_price()      → psf_usd_mt
[TBD source]                  → fetch_stretch_price()  → stretch_usd_mt [new]
ML forecasting model          → model output            → cotton_30d, cotton_60d, psf_30d, psf_60d

Treasury (manual sign-off)    → B13 (USD/PKR rate)     [NOT automated]
Manual                        → B10 (Cycle Low ref)     [NOT automated]
Manual                        → B38-B41 (strike prices) [NOT automated]
```

---

## 10. Files to Modify in Phase 3

### 10.1 `scripts/commodity_mapper.py`

**Why it must change:** The current classifier has only prefix rules (Level 3) as the first lookup step, then description regex as a fallback. The approved design requires three levels with exact-code lookup at Level 1 and description keywords at Level 2 — before any prefix matching.

**Business rule that caused the change:** Decision 3 (three-level category assignment priority). Decision 5 (Cotton Waste auto-detection). Validation finding RISK-01 (prefix-only classification drops Cotton Waste and Stretch Fiber).

**Specific changes required:**
1. Add `from_overrides_csv(path)` constructor (or extend `from_csv_or_default`) to load `item_code_overrides.csv`.
2. Store exact-code overrides in a separate dict (`_exact_map`) distinct from the prefix map (`_prefix_map`).
3. Restructure `classify()` to: (1) check `_exact_map` first, (2) evaluate description keywords, (3) check `_prefix_map`.
4. Add description keyword constants: Cotton Waste keywords and Stretch Fiber keywords.
5. Update `DEFAULT_PREFIX_RULES`: change VIS → Fiber, VSF → Fiber.
6. Remove `Viscose` from `_DESCRIPTION_FALLBACKS` or redirect to Fiber.
7. Update docstring and CSV format documentation to reflect three levels.

---

### 10.2 `scripts/commodity_mapping.csv`

**Why it must change:** VIS and VSF prefixes currently map to `Viscose`. The approved decision reclassifies viscose as `Fiber`.

**Business rule that caused the change:** Decision 3, Level 3 prefix rules (VIS → Fiber, VSF → Fiber).

**Specific changes required:**
- Line `VIS,Viscose` → `VIS,Fiber`
- Line `VSF,Viscose` → `VSF,Fiber`

---

### 10.3 `scripts/item_code_overrides.csv` *(new file)*

**Why it must be created:** The system needs a Level 1 exact-code lookup for items where prefix or description logic produces the wrong result. This file holds confirmed business exceptions.

**Business rule that caused the change:** Decision 3, Level 1 (exact item code mapping). Validation findings: COT-100058 cannot be distinguished from Cotton by prefix alone; FIB-100108 and FIB-100080 have contradictory description signals that only a business decision can resolve.

**Contents:**
```
item_code,category
COT-100058,Cotton Waste
FIB-100108,Fiber
FIB-100080,Stretch Fiber
```

---

### 10.4 `scripts/clean_consumption.py`

**Why it must change:** The current `consumption_qty = primary_qty.abs()` treats returns identically to issues, overstating net consumption.

**Business rule that caused the change:** Decision 2 (Net Consumption = Issues − Returns).

**Specific changes required:**
1. Replace `detail["consumption_qty"] = detail["primary_qty"].abs()` with `detail["net_qty"] = -detail["primary_qty"]`.
2. Update `OUTPUT_COLS` to replace `consumption_qty` with `net_qty`.
3. Update aggregation: change `("consumption_qty", "sum")` to `("net_qty", "sum")`.
4. Update summary column header from `"Total Consumed"` to `"Total Net Consumed"` (or confirm preferred label with business).
5. Add `period_days` derivation from the Consumption sheet's period header text (row 2 parsing).
6. Add zero-consumption org detection and warning log.
7. Update `run()` to return `period_days` as a fourth return value (or include in a metadata dict).
8. Update `monthly_totals()` to use `net_qty` instead of `consumption_qty`.

---

### 10.5 `scripts/clean_inventory.py`

**Why it must change:** Currently produces incorrect category classifications because it relies on `CommodityMapper` which has the prefix-only defect. Once `commodity_mapper.py` is fixed, `clean_inventory.py` will automatically produce correct output — but minor changes are still needed.

**Business rule that caused the change:** Decision 3 (three-level category mapping). Decision 1 (Fiber and Stretch Fiber are separate). Validation RISK-01.

**Specific changes required:**
1. Update `CommodityMapper.from_csv_or_default()` call signature to pass both `mapping_csv` and `overrides_csv` paths.
2. No logic changes to the summary pivot — it already handles arbitrary categories correctly. Once the mapper is fixed, Stretch Fiber and Cotton Waste will appear as separate columns automatically.
3. Add explicit assertion or validation step: after classification, log a warning if any `Unmapped` rows exist.

---

### 10.6 `scripts/market_inputs.py`

**Why it must change:** Phase 3 introduces a new Stretch Fiber price field. Market Inputs must write two new parameters to the workbook (B43 and B28).

**Business rule that caused the change:** Decision 1 (Stretch Fiber is an independent commodity requiring its own price source). Validation RISK-03 (hardcoded B43), RISK-07 (hardcoded B28).

**Specific changes required:**
1. Add `fetch_stretch_price()` connector once a market data source is confirmed (source TBD — may be a manual input initially).
2. Update `build_market_inputs_dataframe()` to include Stretch Fiber price metric.
3. Add `compute_period_days(year, month)` helper that calculates actual days in month.
4. Add `compute_days_to_expiry(expiry_date)` helper.
5. Document that `fetch_fx_rate()` result must not overwrite B13 (treasury approval required).
6. Verify PSF SunSirs commodity_id (839 is a placeholder) before first live run.

---

### 10.7 `data/strategy/strategies.xlsx` *(workbook template)*

**Why it must change:** Multiple structural issues identified in validation. New Section C (Stretch Fiber) is required. Formulas must be updated.

**Business rule that caused the change:** Decision 1 (Stretch Fiber independent). Decision 3 (Cotton Waste label standardisation). Validation RISK-04 (Fiber double-count). Bugs B1, B2 (Dashboard).

**Specific changes required:**
1. Raw Material C187:C193 Fiber formula: remove `+($F$6:$F$181="Stretch Fiber")` clause.
2. Raw Material column F: update category label from `"Cotton - Waste"` to `"Cotton Waste"` in all data rows.
3. Raw Material summary SUMPRODUCT strings: replace `"Cotton - Waste"` with `"Cotton Waste"`.
4. Strategy sheet: add Section C rows for Stretch Fiber (7 orgs, matching Sections A and B structure).
5. Strategy sheet: add Stretch Fiber price and 30d forecast to parameter row 3.
6. Market Inputs: add Stretch Fiber current price and forecast rows.
7. Dashboard B6: fix cell reference (currently points to Subinventory cell of last data row).
8. Dashboard A14:A20: fix offset (currently reads Strategy header row instead of first data row).

> Note: Items 1–3 are template fixes. Items 4–8 will be applied by the Phase 3 workbook population step on every monthly run — they do not require manual workbook editing before Phase 3 is ready.

---

### 10.8 `data/strategy/archive/` *(new directory)*

**Why it must be created:** The approved workbook versioning design requires an archive directory.

**Business rule that caused the change:** Decision 4 (workbook versioning — monthly archive files).

**Creation:** Phase 3 must create this directory if it does not exist before the first archiving operation.

---

## 11. Remaining Risks

The following risks remain open after the business decisions have been applied. They are not resolved by the approved decisions and require further action.

---

### RISK-A — Stretch Fiber Market Price Source Not Identified

**Severity:** HIGH for Section C procurement accuracy  
**Status:** Open — no decision made

**Description:** Section C (Stretch Fiber) procurement requires a current price and 30-day forecast for Spandex/Lycra/Elastane. No automated market data source has been identified or approved. PSF (Polyester Staple Fiber) has SunSirs. Stretch Fiber has no equivalent confirmed source.

**Consequence if unresolved:** Section C PO cost and delay-saving calculations will use a manually-entered price that may be stale or inconsistent between runs.

**Required action:** Identify whether SunSirs covers Spandex (commodity ID lookup required), or identify an alternative source (ICIS, Bloomberg, Dow Chemical price sheets). Until a source is confirmed, Phase 3 must accept Stretch Fiber price as a manual Market Inputs entry.

---

### RISK-B — PSF SunSirs Commodity ID Unverified

**Severity:** MEDIUM  
**Status:** Open — placeholder ID 839 not confirmed

**Description:** `market_inputs.py` uses `psf_commodity_id=839` as the SunSirs ID for PSF. This is a default assumption. The VSF connector uses ID 1057 (verified from existing production code). PSF ID has not been verified.

**Required action:** Navigate to sunsirs.com, locate the PSF commodity, and confirm the numeric ID before connecting Phase 3 to the live market data feed.

---

### RISK-C — `40D` and `70D` Description Keywords May Produce False Positives

**Severity:** LOW-MEDIUM  
**Status:** Open — keyword matching requires scoping

**Description:** The approved description keywords for Stretch Fiber include `40D` and `70D` (denier markers). Item codes such as `FIB-100040` or descriptions containing numeric sequences ending in "D" for reasons unrelated to denier could trigger a false Stretch Fiber classification.

**Required action:** In Phase 3, `40D` and `70D` must be matched as word-boundary-delimited tokens (e.g., regex `\b40[Dd]\b`) to prevent false positives. Validate against the full item description corpus before enabling these keywords in production.

---

### RISK-D — Oracle GL Posting Date Anomaly Prevents Date-Based Filtering

**Severity:** LOW (for current design) / MEDIUM (if date filtering is added)  
**Status:** Confirmed — all April 2026 consumption dates are `2026-03-31`

**Description:** Oracle posts `"Direct Organization Transfer"` transactions with an accounting date of the last day of the previous period. April 2026 transactions are dated March 31, 2026. Any filtering logic that attempts to validate consumption data by date (e.g., "confirm all rows are within April 2026") will fail.

**Consequence:** Phase 3 must identify the reporting period from the text in Consumption row 2, not from the Date column. The Date column can be used for audit trail and lot tracking but not for period validation.

**Required action:** Document this constraint in `clean_consumption.py` code comments. Derive `year_month` from the header text, not from the date column values.

---

### RISK-E — MSL - Fibres U3 Zero Consumption Origin Unknown

**Severity:** MEDIUM  
**Status:** Open — Oracle team confirmation not received

**Description:** MSL - Fibres U3 has no consumption rows in April 2026. It is unknown whether this org uses a different Oracle transaction type, a different subinventory code, or is genuinely inactive for the period.

**Consequence if wrong Oracle filter:** MSL - Fibres U3's true shortfall is invisible. The org holds 758,172 Kgs of Cotton and 287,447 Kgs of Fiber — these appear as `MONITOR` in Strategy rather than triggering a procurement action.

**Required action before Phase 3 go-live:** Oracle team must confirm whether MSL - Fibres U3 consumption is absent because (a) the org is genuinely inactive, (b) it uses a different transaction type not included in the `"Direct Organization Transfer"` filter, or (c) the extract had a data quality issue for this org.

---

### RISK-F — New Stretch Fiber Items Cannot Be Auto-Detected by Prefix Alone

**Severity:** LOW (existing items covered) / MEDIUM (future items)  
**Status:** Structural — inherent to the prefix-only fallback design

**Description:** If Oracle introduces a new Stretch Fiber item with a description that does not contain any of the approved keywords (`SPANDEX`, `LYCRA`, `ELASTANE`, `ELASTAINE`, `ELASPANE`, `40D`, `70D`), it will fall through to prefix Level 3, which classifies all `FIB-` items as `Fiber`. The new item will be silently misclassified as Fiber instead of Stretch Fiber.

**Consequence:** The Stretch Fiber inventory and consumption figures will be understated; Fiber will be overstated. A procurement action for Stretch Fiber that should be triggered may not appear.

**Mitigation:** Phase 3 must log all items classified as `Fiber` via the Level 3 prefix fallback (rather than Level 1 or Level 2) and flag them for monthly human review. Any newly added FIB-* item that lands in Level 3 should be reviewed and either added to `item_code_overrides.csv` or confirmed as a genuine Fiber item.

---

### RISK-G — Strategy Sheet Row Count Is Fixed; New Orgs Break References

**Severity:** LOW currently / HIGH if org count changes  
**Status:** Structural — inherent to positional cell references in Strategy and Dashboard

**Description:** Strategy reads stock levels from Raw Material by hardcoded row position (`='Raw Material'!B187`). Dashboard reads org names and coverage values from Strategy by hardcoded row position. Adding a new spinning unit to Oracle will cause these references to shift silently.

**Mitigation in Phase 3:** The procurement engine computes all values in Python and writes them to the workbook. If the Strategy sheet is populated by the engine (not by Excel formulas), positional references become irrelevant — the engine writes each org to its designated row by name, not by number. The Dashboard row-reference bugs (B1, B2) must be fixed as part of Phase 3 workbook template preparation.

---

### RISK-H — Workbook Re-Run Overwrites Monthly Archive

**Severity:** LOW (by design) / MEDIUM (if re-runs are common)  
**Status:** Policy decision needed

**Description:** If Phase 3 is re-run in the same month (e.g., due to a data correction), the archive step will attempt to overwrite `archive/strategies_YYYY_MM.xlsx`. The specification requires Phase 3 to halt and require explicit confirmation before overwriting.

**Required action:** Phase 3 must include a guard: if the archive file for the current month already exists, prompt the operator or accept a `--force-overwrite` flag. This prevents accidental silent overwrite of a reviewed and shared archive file.

---

*End of PROCUREMENT_ENGINE_FINAL_SPEC.md*
