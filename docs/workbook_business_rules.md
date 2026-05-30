# Workbook Business Rules
## data/strategy/Strategies.xlsx

**Status:** Aligned with approved business rules as of 2026-05-30.  
**Pre-fix backup:** `data/strategy/archive/Strategies_pre_fix_2026_05.xlsx`

---

## Rule 1 — Fiber and Stretch Fiber Are Independent Categories

### Decision
Fiber and Stretch Fiber must always be reported and calculated as separate,
independent commodity categories. They must never be combined into a single
value at any layer of the workbook.

### Workbook Implementation

**Raw Material sheet — Aggregated Inventory (rows 186–194):**

| Column | Category | Formula pattern |
|--------|----------|-----------------|
| B | Cotton | `=SUMPRODUCT(($D$6:$D$181=A{n})*($F$6:$F$181="Cotton")*$E$6:$E$181)` |
| C | **Fiber only** | `=SUMPRODUCT(($D$6:$D$181=A{n})*($F$6:$F$181="Fiber")*$E$6:$E$181)` |
| D | **Stretch Fiber only** | `=SUMPRODUCT(($D$6:$D$181=A{n})*($F$6:$F$181="Stretch Fiber")*$E$6:$E$181)` |
| E | Cotton Waste | `=SUMPRODUCT(($D$6:$D$181=A{n})*($F$6:$F$181="Cotton Waste")*$E$6:$E$181)` |

**Consumption sheet — Aggregated Consumption (rows 27006–27014):**

| Column | Category | Formula pattern |
|--------|----------|-----------------|
| B | Cotton | `=-SUMPRODUCT(($C$5:$C$27001=A{n})*($I$5:$I$27001="Cotton")*$G$5:$G$27001)` |
| C | **Fiber only** | `=-SUMPRODUCT(($C$5:$C$27001=A{n})*($I$5:$I$27001="Fiber")*$G$5:$G$27001)` |
| D | **Stretch Fiber only** | `=-SUMPRODUCT(($C$5:$C$27001=A{n})*($I$5:$I$27001="Stretch Fiber")*$G$5:$G$27001)` |

### What Was Wrong Before
Column C in both sheets used an OR condition:
`($F$6:$F$181="Fiber")+($F$6:$F$181="Stretch Fiber")` — this caused every
Stretch Fiber row to be counted a second time in the Fiber column, producing
a **144,671.87 Kgs overstatement** in the workbook's Fiber inventory total.

### Cells Fixed
`Raw Material C187:C193` (7 cells) — Fiber column formula  
`Consumption C27007:C27013` (7 cells) — Fiber column formula + header text

### Downstream Impact
Strategy sheet D17:D22 reference `'Raw Material'!C{row}` for Fiber inventory.
These references now correctly read **Fiber-only** stock after the fix.

> **Known gap:** The Strategy sheet has no Stretch Fiber procurement section
> (Section A = Cotton, Section B = Fiber). Stretch Fiber procurement rows are
> out-of-scope for the current workbook version and are tracked only via the
> Python engine (`procurement_engine.py`).

---

## Rule 2 — Net Consumption Formula

### Decision
Net Consumption = Material Issues − Returns

Oracle ERP records material issues as **negative** primary_qty and returns as
**positive** primary_qty. Net consumption is therefore:

```
Net Consumption = ABS(sum of negative quantities) − sum of positive quantities
                = −(sum of ALL primary_qty)
                = −SUMPRODUCT(...)
```

### Why ABS() Was Wrong
`ABS(primary_qty)` converts every transaction to positive before summing.
This means a return of +100 Kgs gets added to consumption as if it were an
issue, **overstating** consumption.

Example:
- Issues: −1,000 Kgs (3 transactions of −333.33)
- Returns: +100 Kgs (1 transaction)
- **ABS approach (wrong):** 1,000 + 100 = 1,100 Kgs consumed
- **Net approach (correct):** −(−1,000 + 100) = **900 Kgs consumed**

### Workbook Implementation

All consumption rate formulas now use negation:

```excel
=-SUMPRODUCT((Consumption!$C$5:$C$27014="<org>")*
             (Consumption!$I$5:$I$27014="<category>")*
              Consumption!$G$5:$G$27014)/$K$3
```

The division by `$K$3` (consumption period in days) converts monthly net
consumption to a daily rate.

### Cells Fixed

**Consumption sheet summary (rows 27007–27013):**
- `B27007:B27013` — Cotton net consumption (7 cells)
- `C27007:C27013` — Fiber net consumption (7 cells)
- `D27007:D27013` — Stretch Fiber net consumption (7 cells)

**Strategy sheet daily rate column (column E):**
- `E8` — Cotton: MTM - Spinning U5 (converted from array formula to regular formula)
- `E9` — Cotton: MTM - Spinning U6
- `E10` — Cotton: MSM - Spinning U1
- `E11` — Cotton: MTM - Spinning U2
- `E12` — Cotton: MTM - Spinning U1
- `E13` — Cotton: MTM - Spinning U3
- `E17` — Fiber: MTM - Spinning U2
- `E18` — Fiber: MTM - Spinning U1
- `E19` — Fiber: MTM - Spinning U6
- `E20` — Fiber: MTM - Spinning U3
- `E21` — Fiber: MSM - Spinning U1
- `E22` — Fiber: MTM - Spinning U5

### Note on E14 (MSL - Fibres U3, Cotton)
`E14 = 0` (literal value, not a formula). MSL had no April 2026 Cotton consumption
transactions and is correctly flagged MONITOR. No change required.

---

## Rule 3 — Cotton Waste Handling

### Decision
Cotton Waste is tracked for **inventory visibility only**. No procurement
recommendation is generated for Cotton Waste. Action is always MONITOR.

### Label Standardisation
The canonical label is `"Cotton Waste"` (no hyphen). The workbook previously
used `"Cotton - Waste"` (with hyphen).

### Cells Fixed
- `Raw Material F181` — data cell: `"Cotton - Waste"` → `"Cotton Waste"`
- `Raw Material E187:E193` — Cotton Waste SUMPRODUCT: `"Cotton - Waste"` → `"Cotton Waste"` (7 cells)

### Why This Matters
When `update_strategy_workbook.py` writes fresh Oracle data to the workbook,
the category column is populated by `clean_inventory.py` which outputs
`"Cotton Waste"` (no hyphen). If the workbook SUMPRODUCT still looked for
`"Cotton - Waste"`, the Cotton Waste inventory column would silently show 0
after every automated refresh.

### Downstream Rules
- `procurement_engine.py`: Cotton Waste action is hardcoded to `MONITOR`
  regardless of inventory or consumption values.
- `commodity_mapper.py`: COT-100058 is classified as `"Cotton Waste"` via
  `item_code_overrides.csv`.

---

## Rule 4 — Monthly Archive Requirement

### Decision
Before any automated data refresh, a timestamped archive copy of the workbook
must be created. Archive copies are never overwritten.

### Implementation
Archive is created by `scripts/update_strategy_workbook.py` function
`create_monthly_archive()` before any sheet modification.

**Naming convention:**
```
data/strategy/archive/strategies_YYYY_MM.xlsx
```

**Examples:**
```
strategies_2026_04.xlsx   ← April 2026 snapshot
strategies_2026_05.xlsx   ← May 2026 snapshot
```

**Safety rules enforced in code:**
1. `create_monthly_archive()` is called as the first step in `run()`, before
   any workbook modification.
2. `FileExistsError` is raised if the archive already exists, preventing
   silent overwrites.
3. `shutil.copy2` preserves file metadata.

**Pre-fix backup:** `Strategies_pre_fix_2026_05.xlsx` was created before the
Phase 3B Fix corrections were applied.

---

## Rule 5 — Missing Consumption Data Handling

### Decision
When consumption data is absent for an org-commodity pair, the action must
be `MONITOR` (not `BUY` or `HOLD`). Confidence is `LOW`.

### Three Distinct MONITOR Triggers

| Trigger | Condition | Example |
|---------|-----------|---------|
| Org absent from consumption | Org does not appear in the consumption extract at all | MSL - Fibres U3 (no April transactions) |
| Commodity absent for org | Org present but this category has no transactions | Org present for Cotton but no Fiber transactions |
| Net consumption ≤ 0 | Transactions exist but returns exceed issues | Rare edge case |

### Workbook Implementation

The Strategy sheet handles missing consumption with literal values:

```excel
E14 = 0            (MSL - Fibres U3 Cotton — zero consumption, manually set)
G14 = "N/A"        (shortfall not applicable)
S14 = "N/A"        (days cover not applicable)
C14 = 'MONITOR — no April consumption data'
```

The Python engine (`procurement_engine.py`) handles this via pair-level
detection:
```python
has_consumption = (org, commodity) in cons_pairs  # pair-level check
```

### Governance
- MSL - Fibres U3 consistently shows zero April consumption (documented).
- If zero consumption persists for 2+ consecutive months, a supply chain
  review should be triggered.
- Cotton Waste is always MONITOR regardless of consumption status (see Rule 3).

---

## Summary — All Cells Changed in Phase 3B Fix

### Total: 49 cells

| Fix | Sheet | Cells | Count | Change |
|-----|-------|-------|-------|--------|
| 1A | Raw Material | C187:C193 | 7 | Fiber formula — remove Stretch Fiber from OR condition |
| 1B | Raw Material | E187:E193 | 7 | Cotton Waste label — `"Cotton - Waste"` → `"Cotton Waste"` |
| 1C | Raw Material | F181 | 1 | Category data label — `"Cotton - Waste"` → `"Cotton Waste"` |
| 2A | Consumption | B27007:B27013 | 7 | Cotton net consumption — add negation |
| 2B | Consumption | C27007:C27013 | 7 | Fiber net consumption — remove Stretch Fiber + add negation |
| 2C | Consumption | D27007:D27013 | 7 | Stretch Fiber net consumption — add negation |
| 2D | Consumption | C27006 | 1 | Column header — `"Fiber (incl. Stretch)"` → `"Fiber"` |
| 3A | Strategy | E8 | 1 | MTM-U5 Cotton — array formula → regular formula, ABS → net |
| 3B | Strategy | E9:E13 | 5 | Cotton rows — `ABS(G)` → `-G` |
| 3C | Strategy | E17:E22 | 6 | Fiber rows — `ABS(G)` → `-G` |

---

## Alignment Checklist

| Business Rule | Python Engine | Workbook | Aligned? |
|---------------|--------------|----------|----------|
| Fiber ≠ Stretch Fiber | `_ALL_KNOWN_COMMODITIES` has 4 separate entries | C and D columns separate after Fix 1A | ✓ Yes |
| Net consumption = issues − returns | `daily_rate = net_cons / period_days` where `net_cons = -primary_qty` | `=-SUMPRODUCT(...)` after Fix 2A/2B/2C/3A/3B/3C | ✓ Yes |
| Cotton Waste = MONITOR always | Hardcoded in `_action()` | E187:E193 track inventory; no procurement rows in Strategy | ✓ Yes |
| Cotton Waste label = "Cotton Waste" | `COMMODITY_COTTON_WASTE = "Cotton Waste"` | F181 and E187:E193 after Fix 1B/1C | ✓ Yes |
| Monthly archive before update | `create_monthly_archive()` runs first in `run()` | N/A (Python-enforced) | ✓ Yes |
| MONITOR when consumption absent | Pair-level `has_consumption` check | E14=0, G14="N/A", C14="MONITOR..." | ✓ Yes |
