# UX-2 — Executive Summary Implementation Specification
## Developer-Ready Technical Plan · MG Apparel Commodity Intelligence

**Source wireframe:** `docs/UX1_EXECUTIVE_SUMMARY_WIREFRAME.md` (approved design)  
**Objective:** Minimum code changes, maximum reuse of existing components  
**Files modified:** `scripts/procurement_dashboard.py` (primary) · `streamlit_app.py` (targeted cuts)  
**Files NOT modified:** All other scripts, pipelines, CSS, data files  
**Date:** May 2026

---

## Document Structure

1. [Current Component Inventory](#1--current-component-inventory)
2. [UX-1 Section Mapping](#2--ux-1-section-mapping)
3. [New Functions Required](#3--new-functions-required)
4. [Functions to Remove / Keep / Move](#4--functions-to-remove--keep--move)
5. [Implementation Order](#5--implementation-order)
6. [Data Requirements](#6--data-requirements)
7. [Risk Review](#7--risk-review)

---

## 1 — Current Component Inventory

### 1.1 `render_executive_summary()` — Top-Level Function
**File:** `streamlit_app.py`  **Lines:** 8479–9127  

| Block / Function Called | Purpose | Lines (approx.) | Reusable? | Decision |
|---|---|---|---|---|
| Data loading: `load_procurement_strategy()` | Load strategy CSV + meta dict | 8486 | YES | KEEP |
| Data loading: `load_commodity_data()` (ICE) | Load cotton price series | 8490–8506 | YES | KEEP + EXTEND (add df/value_col to snapshot) |
| Data loading: `load_commodity_data()` (PSF) | Load polyester price series | 8508–8525 | YES | KEEP + EXTEND (add df/value_col to snapshot) |
| Data loading: `fetch_usd_pkr_rate()` | Live USD/PKR rate | 8527–8535 | YES | KEEP |
| `render_exec_procurement_header(...)` call | Renders Sections 1–5 + forecast divider | 8537 | REPLACE | Change call to `render_exec_procurement_header_v2()` |
| `show_local_in_usd` variable + `local_ccy_label` | Controls local column display | 8482–8483 | NO | REMOVE — no longer needed after forecast blocks removed |
| `_get_usd_pkr_rate_for_summary()` (local def) | FX rate with 3-step fallback chain | 8549–8595 | NO | REMOVE — FX comes from `_market_snapshot["usd_pkr"]["price"]` in new design |
| `usd_pkr_rate, usd_pkr_source` assignment | Assigns FX rate + source label | 8597 | NO | REMOVE |
| `st.caption()` about local column | Explains local USD conversion | 8605–8607 | NO | REMOVE |
| `render_empty_card()` (local def) | Empty data card div | 8610–8617 | NO | REMOVE |
| `build_commodity_payload()` (local def) | Load commodity for forecast blocks | 8619–8648 | NO | REMOVE |
| `render_commodity_chart()` (local def) | Forecast bar chart | 8650–8734 | NO | REMOVE |
| `render_commodity_table()` (local def) | Forecast data table | 8752–8830 | NO | REMOVE |
| `_summary_force_usd_per_kg()` (local def) | Unit override for display | 8842–8869 | NO | REMOVE |
| `commodity_pairs` list definition | Defines 5 commodity pairs | 8541–8547 | NO | REMOVE |
| `for int_name, local_name in commodity_pairs:` | Forecast rendering loop | 8836–9004 | NO | REMOVE |
| Pakistan Market Forecasts block | USD/PKR + Electricity forecast | 9006–9044 | NO | REMOVE |
| Overall Market Summary block | Rising/Falling stats | 9046–9116 | NO | REMOVE |
| `render_executive_signals_table()` call | Market signals table | 9118 | NO | REMOVE (only called here — safe to remove call) |
| Footer `st.markdown()` | "Use individual tabs" label | 9121–9125 | NO | REMOVE |

---

### 1.2 `render_exec_procurement_header()` — Current Header Function
**File:** `scripts/procurement_dashboard.py`  **Lines:** 701–969  

| Block / Helper | Purpose | Lines (approx.) | Decision |
|---|---|---|---|
| Period badge rendering | Blue pill with period + generated date | 723–735 | REUSE — copy block verbatim into v2 |
| `_agg()` (inner def) | Sum column for a given commodity | 740–744 | REUSE — inline in v2 |
| `_snap_card()` (inner def) | Market snapshot metric card HTML | 746–775 | EXTRACT to module-level (see Step 1) |
| Section 1 header bar `exec-section-bar` | "1 — Procurement Status" label | 780–783 | REMOVE FROM PAGE (no numbered bars in new design) |
| Section 1 — 6 KPI cards | Cotton/Fiber Inv, 45d Need, Gap, BUY count | 785–803 | REPLACE — 5 new tiles use `_kpi_card()` with different data |
| Section 2 header bar | "2 — Critical Risks" label | 808–811 | REMOVE FROM PAGE |
| Section 2 — Critical Risks table + row styling | 10-row shortfall table sorted by days_cover | 813–852 | REPLACE with Section C (same data + stockout date + est. cost) |
| `st.caption()` about row colors | Table legend | 852 | REPLACE — move note below Section C table |
| Section 3 header bar | "3 — Market Snapshot" label | 857–860 | REMOVE FROM PAGE |
| Section 3 — 3 market snapshot cards | ICE Cotton, PSF, USD/PKR in 3 columns | 862–894 | REPLACE with Section D (vertical stack + spark lines) |
| Section 4 header bar | "4 — Procurement Recommendation" | 899–902 | REMOVE FROM PAGE |
| Section 4 — BUY/HOLD/MONITOR stacked cards | 3 text cards in col[0] | 919–926 | MOVE to Section F (text only, no donut) |
| Section 4 — donut pie chart | Action distribution pie | 928–944 | REMOVE — eliminated per wireframe |
| Section 5 header bar | "5 — Executive Insights" label | 949–952 | REMOVE FROM PAGE |
| Section 5 — `_insight_bullets()` bullet list | Auto-generated narrative | 954–963 | REPLACE with Section G (modified bullets with cost) |
| Market Forecasts divider bar | Header before commodity forecast blocks | 965–969 | REMOVE — forecast section removed |

---

### 1.3 Module-Level Helper Functions in `procurement_dashboard.py`

| Function | Purpose | Lines | Reusable? | Decision |
|---|---|---|---|---|
| `_kpi_card(label, value, sublabel, border_colour)` | HTML metric card div | 111–117 | YES | REUSE AS-IS for Sections B, E |
| `_action_badge(action)` | Colored BUY/HOLD/MONITOR badge | 84–94 | YES | REUSE for Section C table action column |
| `_confidence_badge(confidence)` | Colored HIGH/MEDIUM/LOW badge | 97–108 | YES | REUSE for Section C table confidence column |
| `_section_header(title, subtitle, colour)` | Blue-bordered section header block | 120–129 | NOT NEEDED | Do not call — replaced by self-evident section structure |
| `_chart_layout(fig, height)` | Standard Plotly layout helper | 132–145 | YES | REUSE if coverage bars use Plotly |
| `_empty_state(message)` | Centered placeholder div | 148–153 | YES | REUSE for no-data states in Sections B/C |
| `load_procurement_strategy()` | Load CSV + meta, cache-busted | 62–77 | YES | REUSE — already called from `render_executive_summary()` |
| `_insight_bullets(df)` | Auto-generated bullet list | 625–698 | MODIFY | Add optional `market_snapshot=None` param for cost dimension |
| `_render_overview(df, meta)` | Procurement Intelligence Overview tab | 160–241 | YES (unchanged) | Not touched |
| `_render_buy(df)` | BUY tab | 248–325 | YES (unchanged) | Not touched |
| `_render_hold(df)` | HOLD tab | 332–369 | YES (unchanged) | Not touched |
| `_render_monitor(df)` | MONITOR tab | 390–? | YES (unchanged) | Not touched |
| `render_procurement_intelligence_page()` | Full Procurement page | 976–? | YES (unchanged) | Not touched |

---

## 2 — UX-1 Section Mapping

| UX-1 Section | Label | Existing Function / Block | Action | File |
|---|---|---|---|---|
| Period badge | (above A) | `render_exec_procurement_header()` lines 723–735 | REUSE — copy block verbatim into v2 | procurement_dashboard.py |
| [A] Situation Brief | Full-width narrative | None | CREATE `_build_situation_brief()` + render via `st.markdown()` | procurement_dashboard.py |
| [B] Status Tiles × 5 | B1–B5 | Section 1, six cards (partial overlap) | REPLACE — new 5-tile layout using existing `_kpi_card()` | procurement_dashboard.py |
| [C] Immediate Actions Table | 2/3 width, left | Section 2 Critical Risks table | REPLACE — same df, new columns (stockout date, est. cost) | procurement_dashboard.py |
| [D] Market Snapshot | 1/3 width, right | Section 3 `_snap_card()` | REPLACE — vertical stack, extract `_snap_card()` + add spark | procurement_dashboard.py |
| [E] Financial Exposure | Bottom-row left (3/8) | None | CREATE — new section, uses `_compute_financial_exposure()` | procurement_dashboard.py |
| [F] Position Summary | Bottom-row centre (2/8) | Section 4 text counts (partial) | REPLACE — remove donut, add coverage bars, keep text counts | procurement_dashboard.py |
| [G] Executive Brief | Bottom-row right (3/8) | Section 5 `_insight_bullets()` | MODIFY — same function with cost dimension added | procurement_dashboard.py |
| Market Forecasts (5 commodities) | (entire bottom half) | Lines 8832–9004 in `render_executive_summary()` | REMOVE FROM PAGE | streamlit_app.py |
| Pakistan Forecasts section | Below commodities | Lines 9006–9044 | REMOVE FROM PAGE | streamlit_app.py |
| Overall Market Summary | Below Pakistan forecasts | Lines 9046–9116 | REMOVE FROM PAGE | streamlit_app.py |
| Executive Signals Table | `render_executive_signals_table()` | Line 9118 | REMOVE FROM PAGE | streamlit_app.py |

---

## 3 — New Functions Required

Only functions that do not exist in any form in the current codebase.

---

### 3.1 `_short_org(org_name: str) -> str`

**File:** `scripts/procurement_dashboard.py` (add before `render_exec_procurement_header_v2`)  
**Inputs:** `org_name` — full org name string from `strategy_df["org_name"]`  
**Returns:** Abbreviated string safe for narrow table columns and tile sub-labels  

**Logic:**
```
"MTM - Spinning U3"  →  "MTM-Spin-U3"
"MSM - Spinning U1"  →  "MSM-Spin-U1"
"MTM - Weaving U1"   →  "MTM-Weav-U1"
```

**Implementation note:** Replace `"- Spinning "` with `"-Spin-"`, `"- Weaving "` with `"-Weav-"`.  
Strip any trailing whitespace. Fall back to `org_name[:15]` if neither pattern matches.

---

### 3.2 `_stockout_date(days_cover: float) -> str`

**File:** `scripts/procurement_dashboard.py`  
**Inputs:** `days_cover` float — may be 0.0 or NaN for MONITOR rows  
**Returns:** Formatted date string (e.g. `"Jun 05"`) or `"N/A"` for invalid input  

**Logic:**
```python
import datetime
if pd.isna(days_cover) or days_cover <= 0:
    return "N/A"
stockout = datetime.date.today() + datetime.timedelta(days=float(days_cover))
return stockout.strftime("%b %d")
```

**Import note:** `datetime` is already available via `import datetime` at the top of the file (verify) — or use `from datetime import date, timedelta`.

---

### 3.3 `_compute_financial_exposure(df: pd.DataFrame, market_snapshot: dict) -> dict`

**File:** `scripts/procurement_dashboard.py`  
**Inputs:**  
- `df` — strategy_df (may be empty)  
- `market_snapshot` — dict with keys `"ice_cotton"`, `"psf"`, `"usd_pkr"` (each a nested dict)  

**Returns:** dict with the following keys:  

| Key | Type | Value | Fallback when data unavailable |
|---|---|---|---|
| `market_data_available` | bool | True if both ICE and PSF prices are non-None non-zero | False |
| `ice_price_usd_per_lb` | float or None | `market_snapshot["ice_cotton"]["price"]` | None |
| `ice_price_usd_per_kg` | float or None | `ice_price_usd_per_lb × 2.20462` | None |
| `psf_price_usd_per_kg` | float or None | `market_snapshot["psf"]["price"]` | None |
| `usd_pkr_rate` | float or None | `market_snapshot["usd_pkr"]["price"]` | 280.0 (hard fallback) |
| `cotton_shortfall_kgs` | float | Sum of shortfall for Cotton BUY rows | 0.0 |
| `fiber_shortfall_kgs` | float | Sum of shortfall for Fiber BUY rows | 0.0 |
| `cotton_cost_usd` | float | `cotton_shortfall_kgs × ice_price_usd_per_kg` | 0.0 |
| `fiber_cost_usd` | float | `fiber_shortfall_kgs × psf_price_usd_per_kg` | 0.0 |
| `total_cost_usd` | float | `cotton_cost_usd + fiber_cost_usd` | 0.0 |
| `total_cost_pkr` | float | `total_cost_usd × usd_pkr_rate` | 0.0 |
| `cotton_cost_display` | str | `"~$7.6M"` formatted | `"N/A"` |
| `fiber_cost_display` | str | `"~$1.4M"` formatted | `"N/A"` |
| `total_cost_display` | str | `"~$8.9M"` formatted | `"N/A"` |

**Display format rule:** If value ≥ 1,000,000 → `"~${v/1_000_000:.1f}M"`. If value ≥ 1,000 → `"~${v:,.0f}"`. Prefix with `~` to signal estimate.

**Guard:** If `df.empty`, all kgs = 0.0, all costs = 0.0, `market_data_available = False`.  
**Guard:** If `ice_price` or `psf_price` is None or 0 → set that commodity cost = 0.0, `market_data_available = False`.  
**Guard:** `usd_pkr_rate` — try `market_snapshot["usd_pkr"]["price"]`; if None, fall back to `280.0`.

---

### 3.4 `_build_situation_brief(df: pd.DataFrame, exposure: dict) -> str`

**File:** `scripts/procurement_dashboard.py`  
**Inputs:**  
- `df` — strategy_df (may be empty)  
- `exposure` — output of `_compute_financial_exposure()`  

**Returns:** Single narrative string for Section A.  

**Logic (exact per wireframe Section 8.5):**

```
If df.empty:
    return "Pipeline data unavailable — run the monthly pipeline to generate this brief."

n_buy = (df["action"] == "BUY").sum()

If n_buy == 0:
    return f"All {len(df)} org-commodity pairs meet the 45-day policy requirement. No procurement action required this period."

buy = df[df["action"] == "BUY"]
worst = buy[buy["days_cover"] > 0].loc[buy["days_cover"].idxmin()]  # guard: check active_buy not empty first
stockout_str = _stockout_date(worst["days_cover"])
short_name = _short_org(worst["org_name"])
cost_str = f" Estimated cost: {exposure['total_cost_display']}." if exposure["market_data_available"] else ""

Build ice_change string from exposure or pass market_snapshot separately:
ice_change = market_snapshot["ice_cotton"].get("change", 0) or 0
market_str = f" ICE Cotton {'up' if ice_change > 0 else 'down'} {abs(ice_change):.1f}% MoM." if ice_change != 0 else ""

Return:
    f"{n_buy} procurement actions required. "
    f"{short_name} {worst['commodity']} stockout in {worst['days_cover']:.0f} days ({stockout_str})."
    + cost_str + market_str
```

**Updated signature:** `_build_situation_brief(df, exposure, market_snapshot)` — three parameters.

---

### 3.5 `_snap_card_module(label, price, change, date, currency) -> None`

**File:** `scripts/procurement_dashboard.py`  
**Purpose:** Module-level extraction of the `_snap_card()` inner function from `render_exec_procurement_header()`.  

**Inputs:** Same as the current inner function (label, price, change, date, currency).  
**Output:** None — renders one `st.markdown()` metric card.  

**Existing code to move:** Lines 746–775 inside `render_exec_procurement_header()`. The logic is already correct. Move verbatim to module level. Rename to `_snap_card_module` to avoid shadowing the inner definition in the existing (unmodified) function.

**After extraction:** The existing `render_exec_procurement_header()` can either keep its own inner `_snap_card()` unchanged (safest) or call `_snap_card_module()` instead. **Recommendation: keep existing inner function untouched** — only the new v2 function calls `_snap_card_module()`.

---

### 3.6 `_render_spark_line(series: pd.Series, up_is_bad: bool, key: str) -> None`

**File:** `scripts/procurement_dashboard.py`  
**Inputs:**  
- `series` — last N price values (pd.Series of floats, already sliced to last 6)  
- `up_is_bad` — True for ICE Cotton and PSF (rising price = cost risk for buyer); False for USD/PKR in PKR terms  
- `key` — unique Plotly chart key string  

**Output:** None — renders one `st.plotly_chart()` at height=55, no axes, no labels, line only.

**Implementation note:**  
- Color: `"#dc2626"` (red) if last value > first value and `up_is_bad` is True; `"#059669"` (green) otherwise.  
- Use `plotly.graph_objects.Scatter` with `mode="lines"`, `line=dict(width=2)`.  
- Set `margin=dict(l=0, r=0, t=0, b=0)`, `height=55`, `paper_bgcolor="rgba(0,0,0,0)"`, `plot_bgcolor="rgba(0,0,0,0)"`.  
- Disable all axes: `xaxis=dict(visible=False)`, `yaxis=dict(visible=False)`.
- If `series` has fewer than 2 non-null values, render nothing (guard with `if len(series.dropna()) < 2: return`).

---

### 3.7 `render_exec_procurement_header_v2(df, meta, market_snapshot) -> None`

**File:** `scripts/procurement_dashboard.py`  
**Purpose:** New 7-section layout replacing `render_exec_procurement_header()` on the Executive Summary page.  
**Inputs:** Same signature as `render_exec_procurement_header()` — drop-in replacement at call site.  
**Output:** None — renders Sections A through G to Streamlit.

This is the main new function. Its internal structure is fully specified below in Section 5 (Implementation Order, Step 6).

---

## 4 — Functions to Remove / Keep / Move

### 4.1 `streamlit_app.py` — Changes Inside `render_executive_summary()`

| Code Block | Lines (approx.) | Action | Reason |
|---|---|---|---|
| `show_local_in_usd = True` | 8482 | REMOVE | Forecast blocks removed — no local column needed |
| `local_ccy_label = ...` | 8483 | REMOVE | Same |
| `_exec_df, _exec_meta = load_procurement_strategy()` | 8486 | KEEP | Still needed |
| ICE Cotton loading block (7 lines) | 8489–8506 | KEEP + EXTEND | Add `"df": _ice_md.get("df"), "value_col": _ice_md.get("value_col")` to `_market_snapshot["ice_cotton"]` |
| PSF loading block (7 lines) | 8508–8525 | KEEP + EXTEND | Add `"df": _psf_md.get("df"), "value_col": _psf_md.get("value_col")` to `_market_snapshot["psf"]` |
| USD/PKR loading block | 8527–8535 | KEEP | No change |
| `render_exec_procurement_header(...)` call | 8537 | REPLACE | Change to `render_exec_procurement_header_v2(_exec_df, _exec_meta, _market_snapshot)` |
| `commodity_pairs` list | 8541–8547 | REMOVE | Forecast blocks removed |
| `_get_usd_pkr_rate_for_summary()` function definition | 8549–8595 | REMOVE | FX rate in new design comes from `_market_snapshot["usd_pkr"]["price"]` |
| `usd_pkr_rate, usd_pkr_source = ...` | 8597 | REMOVE | Same |
| `st.caption(...)` about local column | 8605–8607 | REMOVE | No local column |
| `render_empty_card()` (local def) | 8610–8617 | REMOVE | Forecast blocks removed |
| `build_commodity_payload()` (local def) | 8619–8648 | REMOVE | Forecast blocks removed |
| `render_commodity_chart()` (local def) | 8650–8734 | REMOVE | Forecast blocks removed |
| `render_commodity_table()` (local def) | 8752–8830 | REMOVE | Forecast blocks removed |
| `all_summary = []` setup + commodity loop | 8833–9004 | REMOVE | Entire commodity forecast block |
| Pakistan Market Forecasts header + block | 9006–9044 | REMOVE | Wrong page per wireframe P7 |
| Overall Market Summary block | 9046–9116 | REMOVE | Market analysis section removed |
| `render_executive_signals_table()` call | 9118 | REMOVE | Only called here; safe to remove call (function definition kept) |
| Final `st.markdown("---")` separators | 9117, 9120 | REMOVE | No longer needed |
| Footer caption | 9121–9125 | REMOVE | Replaced by self-evident section structure |

**After all removals, `render_executive_summary()` will consist of:**
1. Data loading block (~15 lines, extended by 4 lines for df/value_col)
2. Single call: `render_exec_procurement_header_v2(_exec_df, _exec_meta, _market_snapshot)`
3. Function end

---

### 4.2 `scripts/procurement_dashboard.py` — Existing Functions

| Function | Action | Reason |
|---|---|---|
| `load_procurement_strategy()` | KEEP — no changes | Called from both streamlit_app.py and new v2 function |
| `_kpi_card()` | KEEP — no changes | Reused in v2 Sections B and E |
| `_action_badge()` | KEEP — no changes | Reused in v2 Section C table |
| `_confidence_badge()` | KEEP — no changes | Reused in v2 Section C table |
| `_section_header()` | KEEP — no changes | Still used in `render_procurement_intelligence_page()` |
| `_chart_layout()` | KEEP — no changes | May be reused for coverage bars in Section F |
| `_empty_state()` | KEEP — no changes | Reused in v2 for no-data states |
| `_insight_bullets()` | MODIFY — add market_snapshot param | Needed for cost dimension in Section G |
| `render_exec_procurement_header()` | KEEP — no changes | Do not modify the existing function; only the call site changes |
| `_render_overview()` | KEEP — no changes | Procurement Intelligence page unaffected |
| `_render_buy()` | KEEP — no changes | Same |
| `_render_hold()` | KEEP — no changes | Same |
| `_render_monitor()` | KEEP — no changes | Same |
| `render_procurement_intelligence_page()` | KEEP — no changes | Entirely separate page |

---

### 4.3 `_insight_bullets()` — Required Modification

**Current signature:** `def _insight_bullets(df: pd.DataFrame) -> list[str]`  
**New signature:** `def _insight_bullets(df: pd.DataFrame, market_snapshot: dict = None) -> list[str]`  

**Change:** When `market_snapshot` is provided and `market_snapshot.get("market_data_available")` is True, append cost string to cotton and fiber bullets.

**Before (cotton bullet):**
```
"Cotton procurement required for 6 units. Total gap: 5,059,191 Kgs."
```

**After (cotton bullet with market_snapshot):**
```
"Cotton procurement required for 6 units. Total gap: 5,059,191 Kgs (~$7.6M)."
```

**Implementation:** Pass `exposure = _compute_financial_exposure(df, market_snapshot)` at start of function if `market_snapshot` is not None. Append `f" (~{exposure['cotton_cost_display']})"` to cotton bullet and `f" (~{exposure['fiber_cost_display']})"` to fiber bullet.

**Backward compatibility:** All existing callers that pass only `df` continue to work unchanged (`market_snapshot=None` → no cost appended).

---

## 5 — Implementation Order

Steps are ordered to minimize risk. Steps 1–4 are pure additions (no deletions). Steps 5–7 change call sites only after the new functions are verified to exist.

---

### Step 1 — Extract `_snap_card()` to module level

**File:** `scripts/procurement_dashboard.py`  
**Action:** Add new module-level function `_snap_card_module()` near the existing `_kpi_card()` helper (around line 111).  
**Source code:** Copy verbatim from lines 746–775 of `render_exec_procurement_header()`.  
**Risk:** None — the existing inner function in `render_exec_procurement_header()` is untouched.  
**Verify:** Call `_snap_card_module("TEST", 1.0, 0.5, "Apr 2026", "USD/lb")` — it must render without error.

---

### Step 2 — Add pure utility helpers

**File:** `scripts/procurement_dashboard.py`  
**Location:** Add immediately after `_snap_card_module()` (new) and before `_insight_bullets()`.  
**Order of addition:**
1. `_short_org(org_name)` — string manipulation, no Streamlit
2. `_stockout_date(days_cover)` — date arithmetic, no Streamlit
3. `_compute_financial_exposure(df, market_snapshot)` — math only, no Streamlit

**Risk:** None — pure functions, no side effects.  
**Verify:** Unit test each with known April 2026 values:
```python
_short_org("MTM - Spinning U3")       == "MTM-Spin-U3"
_stockout_date(6.5)                   == "Jun 05"  (on 2026-05-30)
_stockout_date(0)                     == "N/A"
_compute_financial_exposure(df, snap)["total_cost_display"] == "~$8.9M"
```

---

### Step 3 — Add `_render_spark_line()` helper

**File:** `scripts/procurement_dashboard.py`  
**Location:** After Step 2 helpers, before `_insight_bullets()`.  
**Risk:** Low — isolated Plotly chart; if it fails, wrap in `try/except` to render nothing.  
**Verify:** Call with a 6-element pd.Series — renders a 55px chart with no axes.

---

### Step 4 — Modify `_insight_bullets()` to accept `market_snapshot`

**File:** `scripts/procurement_dashboard.py`  
**Change:** Add `market_snapshot: dict = None` parameter.  
**Risk:** Low — backward-compatible default. Existing callers (`render_procurement_intelligence_page()` → `_render_overview()` does NOT call `_insight_bullets()` directly; only `render_exec_procurement_header()` at line 954 does).  

**After change, verify existing call still works:**
```python
bullets = _insight_bullets(df)   # must produce same output as before
```

---

### Step 5 — Add `_build_situation_brief()`

**File:** `scripts/procurement_dashboard.py`  
**Location:** After `_insight_bullets()`.  
**Risk:** Low — pure string function.  
**Verify:** 
- With 10 BUY rows → returns narrative string with n_buy, org name, days, cost
- With 0 BUY rows → returns "All N pairs meet..." string  
- With empty df → returns "Pipeline data unavailable..." string

---

### Step 6 — Build `render_exec_procurement_header_v2()`

**File:** `scripts/procurement_dashboard.py`  
**Location:** After `render_exec_procurement_header()` (around line 970).  
**Risk:** Medium — new rendering function. Does not affect production until call site changed in Step 7.

**Internal rendering structure (in order):**

#### 6.1 — Period badge + data prep

```
Render period badge (reuse lines 723–735 verbatim)
Compute: no_data = df.empty
Compute: n_buy, n_hold, n_monitor = action counts
Compute: exposure = _compute_financial_exposure(df, market_snapshot)
Compute: usd_pkr_rate = exposure["usd_pkr_rate"]
```

#### 6.2 — Section A: Situation Brief

```
brief_text = _build_situation_brief(df, exposure, market_snapshot)
Render: st.markdown(
    f"<div class='alert-critical'>...<strong>{brief_text}</strong>...</div>",
    unsafe_allow_html=True
)
```

**CSS class:** Reuse `.alert-critical` (already defined in existing CSS). If `.alert-critical` has too much top padding, add inline `padding: 0.6rem 1.1rem` override.  
**Element:** `st.markdown()` — NOT `st.info()`, NOT `st.caption()`.

#### 6.3 — Section B: Status Tiles (5 columns)

```
b1, b2, b3, b4, b5 = st.columns(5)

with b1:  # B1 — Actions Required
    val = str(n_buy) if not no_data else "—"
    border = _C_BUY if n_buy > 0 else "#059669"
    st.markdown(_kpi_card("ACTIONS REQUIRED", val, "BUY recommendations", border), unsafe_allow_html=True)

with b2:  # B2 — Lowest Cover
    if not no_data and n_buy > 0:
        active_buy = df[(df["action"]=="BUY") & (df["days_cover"] > 0)]
        if not active_buy.empty:
            worst = active_buy.loc[active_buy["days_cover"].idxmin()]
            worst_days = worst["days_cover"]
            border_b2 = _C_BUY if worst_days < 15 else (_C_MONITOR if worst_days < 30 else _C_HOLD)
            sub_b2 = f"{_short_org(worst['org_name'])} / {worst['commodity']}  ·  {_stockout_date(worst_days)}"
            st.markdown(_kpi_card("LOWEST COVER", f"{worst_days:.1f} days", sub_b2, border_b2), unsafe_allow_html=True)
        else:
            st.markdown(_kpi_card("LOWEST COVER", "—", "No active BUY rows", "#94a3b8"), unsafe_allow_html=True)
    else:
        st.markdown(_kpi_card("LOWEST COVER", "—", "All positions adequate", "#059669"), unsafe_allow_html=True)

with b3:  # B3 — Estimated Total Cost
    val_b3 = exposure["total_cost_display"] if exposure["market_data_available"] else "—"
    sub_b3 = (f"Cotton {exposure['cotton_cost_display']} + Fiber {exposure['fiber_cost_display']}"
              if exposure["market_data_available"] else "Market data unavailable")
    st.markdown(_kpi_card("EST. PROCUREMENT COST", val_b3, sub_b3, _C_BUY), unsafe_allow_html=True)

with b4:  # B4 — Worst Cotton Cover
    cotton_buy = df[(df["commodity"]=="Cotton") & (df["action"]=="BUY") & (df["days_cover"]>0)]
    if not no_data and not cotton_buy.empty:
        wc = cotton_buy["days_cover"].min()
        wc_org = _short_org(cotton_buy.loc[cotton_buy["days_cover"].idxmin(), "org_name"])
        border_b4 = _C_BUY if wc < 15 else (_C_MONITOR if wc < 30 else _C_HOLD)
        st.markdown(_kpi_card("COTTON COVER", f"{wc:.1f} days", f"Worst: {wc_org}", border_b4), unsafe_allow_html=True)
    else:
        st.markdown(_kpi_card("COTTON COVER", "—", "No cotton BUY rows", "#94a3b8"), unsafe_allow_html=True)

with b5:  # B5 — Worst Fiber Cover
    fiber_buy = df[(df["commodity"]=="Fiber") & (df["action"]=="BUY") & (df["days_cover"]>0)]
    if not no_data and not fiber_buy.empty:
        wf = fiber_buy["days_cover"].min()
        wf_org = _short_org(fiber_buy.loc[fiber_buy["days_cover"].idxmin(), "org_name"])
        border_b5 = _C_BUY if wf < 15 else (_C_MONITOR if wf < 30 else _C_HOLD)
        st.markdown(_kpi_card("FIBER COVER", f"{wf:.1f} days", f"Worst: {wf_org}", border_b5), unsafe_allow_html=True)
    else:
        st.markdown(_kpi_card("FIBER COVER", "—", "No fiber BUY rows", "#94a3b8"), unsafe_allow_html=True)
```

#### 6.4 — Main Row: Sections C + D (2/3 + 1/3 split)

```
col_c, col_d = st.columns([2, 1])
```

**Section C — Immediate Actions Table (in col_c):**

```
st.markdown(header: "IMMEDIATE ACTIONS — {n_buy} pairs require procurement")
st.markdown(link: "View full analysis in Procurement Intelligence →")

if no_data or n_buy == 0:
    _empty_state("No procurement actions required this period.")
else:
    buy_sorted = df[df["action"]=="BUY"].sort_values("days_cover").copy()
    
    # Add computed columns
    buy_sorted["Stockout"] = buy_sorted["days_cover"].apply(_stockout_date)
    buy_sorted["Short Org"] = buy_sorted["org_name"].apply(_short_org)
    
    # Add per-row cost
    commodity_price_map = {
        "Cotton":       exposure["ice_price_usd_per_kg"] or 0.0,
        "Fiber":        exposure["psf_price_usd_per_kg"] or 0.0,
        "Stretch Fiber": exposure["psf_price_usd_per_kg"] or 0.0,
        "Cotton Waste": 0.0,
    }
    def row_cost(row):
        price = commodity_price_map.get(row["commodity"], 0.0)
        cost = float(row["shortfall"]) * price
        if cost == 0:
            return "—"
        return f"~${cost/1_000_000:.2f}M" if cost >= 1_000_000 else f"~${cost:,.0f}"
    buy_sorted["Est. Cost"] = buy_sorted.apply(row_cost, axis=1)
    
    # Build display table
    display_c = buy_sorted[["Short Org", "commodity", "days_cover", "Stockout",
                             "shortfall", "Est. Cost", "confidence"]].copy()
    display_c.columns = ["Org", "Commodity", "Days Cover", "Stockout Date",
                          "Gap (Kgs)", "Est. Cost", "Confidence"]
    display_c["Days Cover"] = display_c["Days Cover"].apply(lambda v: f"{v:.1f}d")
    display_c["Gap (Kgs)"] = display_c["Gap (Kgs)"].map("{:,.0f}".format)
    
    # Row styling (same as _render_buy)
    def _c_row_style(row):
        try:
            dc = float(str(row.get("Days Cover", "99")).replace("d",""))
        except ValueError:
            dc = 99
        if dc < 7:
            return ["background:#fff1f2"] * len(row)
        if dc < 15:
            return ["background:#fffbeb"] * len(row)
        return [""] * len(row)
    
    styled_c = display_c.style.apply(_c_row_style, axis=1)
    st.dataframe(styled_c, use_container_width=True, hide_index=True, height=340)
    st.caption("Red = under 7 days · Amber = under 15 days")
    
    # HOLD/MONITOR footer note
    n_hold_local = (df["action"]=="HOLD").sum()
    n_monitor_local = (df["action"]=="MONITOR").sum()
    st.markdown(
        f"<p style='font-size:0.78rem;color:#64748b;'>"
        f"{n_hold_local} HOLD (adequate stock) · {n_monitor_local} MONITOR (data pending)"
        f"</p>",
        unsafe_allow_html=True
    )
```

**Section D — Market Snapshot (in col_d):**

```
st.markdown("<div class='metric-label'>MARKET SNAPSHOT</div>", unsafe_allow_html=True)

ice   = market_snapshot.get("ice_cotton", {})
psf   = market_snapshot.get("psf", {})
usdpk = market_snapshot.get("usd_pkr", {})

# D1 — ICE Cotton (with spark)
_snap_card_module("ICE COTTON NO. 2",
    ice.get("price"), ice.get("change"), ice.get("date"),
    ice.get("currency", "USD/lb"))
if ice.get("df") is not None and ice.get("value_col"):
    series = ice["df"][ice["value_col"]].tail(6)
    _render_spark_line(series, up_is_bad=True, key="snap_spark_ice")

st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

# D2 — PSF (with spark)
_snap_card_module("PSF (POLYESTER)",
    psf.get("price"), psf.get("change"), psf.get("date"),
    psf.get("currency", "USD/kg"))
if psf.get("df") is not None and psf.get("value_col"):
    series = psf["df"][psf["value_col"]].tail(6)
    _render_spark_line(series, up_is_bad=True, key="snap_spark_psf")

st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

# D3 — USD/PKR (no spark — fetch_usd_pkr_rate() returns scalar only)
_snap_card_module("USD / PKR",
    usdpk.get("price"), usdpk.get("change"), usdpk.get("date"),
    "PKR per USD")
# Note: no spark for USD/PKR — time series not available from the live rate endpoint
```

#### 6.5 — Bottom Row: Sections E + F + G (3/8 + 2/8 + 3/8 split)

```
col_e, col_f, col_g = st.columns([3, 2, 3])
```

**Section E — Financial Exposure (in col_e):**

```
st.markdown("<div class='metric-label'>FINANCIAL EXPOSURE</div>", unsafe_allow_html=True)

if exposure["market_data_available"]:
    # Cotton block
    st.markdown(_kpi_card(
        "EST. COTTON PROCUREMENT",
        exposure["cotton_cost_display"],
        f"{exposure['cotton_shortfall_kgs']:,.0f} Kgs × ICE {exposure['ice_price_usd_per_lb']:.4f} USD/lb",
        _C_BUY
    ), unsafe_allow_html=True)
    
    pkr_cotton = exposure["cotton_cost_usd"] * exposure["usd_pkr_rate"]
    st.markdown(f"<p style='font-size:0.72rem;color:#64748b;margin:0 0 0.6rem 0;'>"
                f"PKR ~{pkr_cotton/1_000_000_000:.2f}B at {exposure['usd_pkr_rate']:.2f}</p>",
                unsafe_allow_html=True)
    
    # Fiber block
    st.markdown(_kpi_card(
        "EST. FIBER PROCUREMENT",
        exposure["fiber_cost_display"],
        f"{exposure['fiber_shortfall_kgs']:,.0f} Kgs × PSF {exposure['psf_price_usd_per_kg']:.4f} USD/kg",
        _C_BUY
    ), unsafe_allow_html=True)
    
    pkr_fiber = exposure["fiber_cost_usd"] * exposure["usd_pkr_rate"]
    st.markdown(f"<p style='font-size:0.72rem;color:#64748b;margin:0 0 0.6rem 0;'>"
                f"PKR ~{pkr_fiber/1_000_000:.0f}M at {exposure['usd_pkr_rate']:.2f}</p>",
                unsafe_allow_html=True)
    
    # Total
    st.markdown(f"""
    <div style='border-top:2px solid #e2e8f0;padding-top:0.6rem;margin-top:0.2rem;'>
        <span class='metric-label'>TOTAL EXPOSURE</span>
        <span style='font-size:1.6rem;font-weight:800;color:{_C_BUY};'>{exposure['total_cost_display']}</span>
        <p style='font-size:0.7rem;color:#94a3b8;margin:0.2rem 0 0 0;'>Estimated at current market rates · BUY rows only</p>
    </div>""", unsafe_allow_html=True)
else:
    _empty_state("Market price data unavailable — cost estimates require ICE Cotton and PSF prices.")
```

**Section F — Position Summary (in col_f):**

```
st.markdown("<div class='metric-label'>POSITION SUMMARY</div>", unsafe_allow_html=True)

# Text counts (no chart)
for label, n, sub, colour in [
    ("BUY",     n_buy,     "Act now",          _C_BUY),
    ("HOLD",    n_hold,    "Adequate",          _C_HOLD),
    ("MONITOR", n_monitor, "Data pending",      _C_MONITOR),
]:
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;padding:0.3rem 0;"
        f"border-bottom:1px solid #f1f5f9;'>"
        f"<span style='font-size:1.15rem;font-weight:800;color:{colour};'>{n}</span>"
        f"<span style='font-size:0.82rem;font-weight:600;color:#64748b;'>{label} — {sub}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

total_pairs = len(df) if not no_data else 0
st.markdown(f"<p style='font-size:0.78rem;color:#94a3b8;margin:0.5rem 0;'>{total_pairs} total pairs tracked</p>",
            unsafe_allow_html=True)

# Coverage bars
if not no_data:
    cotton_all = df[(df["commodity"]=="Cotton") & (df["days_cover"]>0)]
    fiber_all  = df[(df["commodity"]=="Fiber")  & (df["days_cover"]>0)]
    avg_cotton = cotton_all["days_cover"].mean() if not cotton_all.empty else 0.0
    avg_fiber  = fiber_all["days_cover"].mean()  if not fiber_all.empty  else 0.0

    def _bar_color(avg_days):
        if avg_days < 15: return _C_BUY
        if avg_days < 30: return _C_MONITOR
        return _C_HOLD

    for commodity_label, avg_days in [("Cotton", avg_cotton), ("Fiber", avg_fiber)]:
        bar_pct = min(avg_days / 90.0, 1.0) * 100
        target_pct = (45.0 / 90.0) * 100   # 45d target = 50% of 90d scale
        colour = _bar_color(avg_days)
        st.markdown(f"""
        <div style='margin:0.4rem 0;'>
            <div style='display:flex;justify-content:space-between;margin-bottom:0.15rem;'>
                <span style='font-size:0.72rem;font-weight:700;color:#64748b;'>{commodity_label}</span>
                <span style='font-size:0.72rem;font-weight:700;color:{colour};'>{avg_days:.0f}d avg</span>
            </div>
            <div style='background:#f1f5f9;height:8px;border-radius:4px;position:relative;'>
                <div style='width:{bar_pct:.0f}%;background:{colour};height:8px;border-radius:4px;'></div>
                <div style='position:absolute;left:{target_pct:.0f}%;top:-3px;height:14px;
                            width:2px;background:#0f172a;opacity:0.5;'></div>
            </div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("<p style='font-size:0.68rem;color:#94a3b8;'>Bar = avg days cover · line = 45d policy target</p>",
                unsafe_allow_html=True)
```

**Section G — Executive Brief (in col_g):**

```
st.markdown("<div class='metric-label'>EXECUTIVE BRIEF</div>", unsafe_allow_html=True)

bullets = _insight_bullets(df, market_snapshot=market_snapshot)
bullet_html = "".join(
    f"<li style='margin-bottom:0.55rem;font-size:0.9rem;font-weight:600;"
    f"color:#1e293b;line-height:1.5;'>{b}</li>"
    for b in bullets
)
st.markdown(
    f"<ul style='padding-left:1.15rem;margin:0;'>{bullet_html}</ul>",
    unsafe_allow_html=True,
)
```

**Note:** Font size increased from `0.88rem` (current) to `0.9rem` per wireframe.

---

### Step 7 — Update `render_executive_summary()` in `streamlit_app.py`

**Two changes in this step:**

**7a — Extend market snapshot to include history df:**
In the ICE cotton loading block (around line 8499), change:
```python
_market_snapshot["ice_cotton"] = {
    "price":    _ice_md.get("current_price"),
    "change":   _ice_md.get("price_change", 0),
    "date":     _ice_date,
    "currency": INTERNATIONAL_COMMODITIES["Cotton"]["currency"],
}
```
to:
```python
_market_snapshot["ice_cotton"] = {
    "price":     _ice_md.get("current_price"),
    "change":    _ice_md.get("price_change", 0),
    "date":      _ice_date,
    "currency":  INTERNATIONAL_COMMODITIES["Cotton"]["currency"],
    "df":        _ice_md.get("df"),
    "value_col": _ice_md.get("value_col"),
}
```

Apply the same addition to `_market_snapshot["psf"]` using `_psf_md`.

**7b — Replace the header call and remove all forecast content:**

Change line 8537:
```python
render_exec_procurement_header(_exec_df, _exec_meta, _market_snapshot)
```
to:
```python
render_exec_procurement_header_v2(_exec_df, _exec_meta, _market_snapshot)
```

Then delete everything from the line after the call (the comment `# ─────...`) through to the end of `render_executive_summary()`, which ends at line ~9125 (the footer).

The function body after this step is ~25 lines total (data loading + one call).

---

### Step 8 — Verify and Smoke Test

**Case 1 — Pipeline file present, market data present:**
- Load app, go to Executive Summary
- Verify Section A shows narrative with n_buy, org, stockout date, cost string
- Verify Section B shows 5 tiles with correct values
- Verify Section C shows 10 BUY rows with stockout dates and cost column
- Verify Section D shows 3 vertically stacked market cards with spark lines for ICE and PSF
- Verify Section E shows cotton/fiber/total cost in USD with PKR equivalent
- Verify Section F shows 3 count rows + 2 coverage bars, no donut chart
- Verify Section G shows 5 bullets with cost dimensions

**Case 2 — Pipeline file absent (`reports/procurement_strategy.csv` missing):**
- Section A: "Pipeline data unavailable..." string
- Section B: all tiles show "—"
- Section C: `_empty_state()` message
- Section E: `_empty_state()` message
- Section F: all counts = 0, bars not shown

**Case 3 — Market data absent (commodity CSV files missing):**
- Section A: renders without cost string and without market signal
- Section B Tile B3: shows "—" with "Market data unavailable"
- Section D: cards show "N/A" (existing `_snap_card_module()` behavior)
- Section D: spark lines not rendered (guard: `if ice.get("df") is not None`)
- Section E: `_empty_state()` message

**Case 4 — n_buy == 0 (all HOLD or MONITOR):**
- Section A: "All N pairs meet the 45-day policy requirement..."
- Section B Tile B1: "0" with green border
- Section B Tiles B2/B4/B5: "—" 
- Section C: `_empty_state()` message
- Section E: "—" in cost tiles
- Section G: bullet 1 = "All org-commodity pairs meet the 45-day policy stock requirement."

---

## 6 — Data Requirements

| KPI | Available in current code? | Source | Requires new calculation? |
|---|---|---|---|
| `n_buy`, `n_hold`, `n_monitor` | YES | `df["action"].value_counts()` | No |
| `days_cover` per row | YES | `strategy_df["days_cover"]` | No |
| `shortfall` per row | YES | `strategy_df["shortfall"]` | No |
| `procurement_qty` per row | YES | `strategy_df["procurement_qty"]` | No |
| `org_name`, `commodity`, `confidence` | YES | `strategy_df` | No |
| ICE Cotton price (USD/lb) | YES | `_market_snapshot["ice_cotton"]["price"]` | No |
| ICE Cotton price change % | YES | `_market_snapshot["ice_cotton"]["change"]` | No |
| PSF price (USD/kg) | YES | `_market_snapshot["psf"]["price"]` | No |
| USD/PKR rate | YES | `_market_snapshot["usd_pkr"]["price"]` | No |
| ICE Cotton history df (for spark) | PARTIAL — in `_ice_md` local var only | `_market_snapshot["ice_cotton"]["df"]` after Step 7a | YES — add to market_snapshot in Step 7a |
| PSF history df (for spark) | PARTIAL — in `_psf_md` local var only | `_market_snapshot["psf"]["df"]` after Step 7a | YES — add to market_snapshot in Step 7a |
| USD/PKR history df (for spark) | NO | `fetch_usd_pkr_rate()` returns scalar only | NO — omit spark for D3 card |
| Stockout date per BUY row | NO | `today + timedelta(days_cover)` | YES — `_stockout_date()` (Step 2) |
| Short org name | NO | Derived from `org_name` | YES — `_short_org()` (Step 2) |
| Est. cotton cost (USD) | NO | `cotton_shortfall × ice_price_usd_per_kg` | YES — `_compute_financial_exposure()` (Step 2) |
| Est. fiber cost (USD) | NO | `fiber_shortfall × psf_price_usd_per_kg` | YES — same function |
| Est. total cost (USD) | NO | cotton + fiber | YES — same function |
| Est. total cost (PKR) | NO | `total_usd × usd_pkr_rate` | YES — same function |
| Est. cost per BUY row | NO | `row.shortfall × commodity_price_map[commodity]` | YES — computed inline in Section C render (Step 6) |
| Cotton avg days cover (active) | NO | `mean of cotton rows with days_cover > 0` | YES — computed inline in Section F (Step 6) |
| Fiber avg days cover (active) | NO | same for Fiber | YES — same |
| Situation Brief narrative | NO | `_build_situation_brief()` | YES — Step 5 |
| Coverage bar width % | NO | `avg_days / 90 * 100` | YES — computed inline in Section F |

**Summary:** 8 existing KPIs reused as-is. 4 data additions (spark history, stockout date, short org, cost calculations). No new pipeline files or data sources required.

---

## 7 — Risk Review

### R1 — No procurement file

**Scenario:** `reports/procurement_strategy.csv` does not exist.  
**Existing guard:** `load_procurement_strategy()` returns `(pd.DataFrame(), {})` — already handles this.  
**New code must handle:** Every section in `render_exec_procurement_header_v2()` must check `no_data = df.empty` before accessing any column.  
**Fallback behavior:** Section A shows pipeline-unavailable string. Sections B/C/E/F show "—" or `_empty_state()`. Section D still renders (market data is independent of pipeline).  
**Required check in every section:** `if no_data: ...` guard before any `df[...]` access.

---

### R2 — No market snapshot

**Scenario:** Commodity CSV files absent or malformed — `_market_snapshot["ice_cotton"]` and/or `_market_snapshot["psf"]` is `{}`.  
**Existing guard in `render_executive_summary()`:** Both loading blocks are wrapped in `try/except Exception` → snapshot key set to `{}`.  
**New code impact:** `_compute_financial_exposure()` must handle `market_snapshot.get("ice_cotton", {}).get("price")` returning `None`. Set `market_data_available = False`, all cost fields = 0.0.  
**Section D spark lines:** Guard `if ice.get("df") is not None` prevents AttributeError.  
**Section A:** `_build_situation_brief()` must produce a valid string even with no cost data (omit cost clause).

---

### R3 — No USD/PKR rate

**Scenario:** `fetch_usd_pkr_rate()` returns `None` or `{}`.  
**Existing guard:** `_market_snapshot["usd_pkr"]` is `{}` — `usdpk.get("price")` returns `None`.  
**New code impact:** `_compute_financial_exposure()` — set `usd_pkr_rate = market_snapshot.get("usd_pkr", {}).get("price")` or `280.0` fallback.  
**Section E PKR equivalent:** Only show PKR line if `usd_pkr_rate` is not None.  
**Section D Card D3:** `_snap_card_module("USD / PKR", None, ...)` renders "N/A" card (existing behavior).

---

### R4 — Zero BUY rows

**Scenario:** All rows are HOLD or MONITOR. `n_buy == 0`.  
**New code impact:**  
- `active_buy` in Section B Tile B2: empty DataFrame — must guard `if not active_buy.empty` before `idxmin()`.  
- `cotton_buy`, `fiber_buy` in Tiles B4/B5: same guard.  
- `_compute_financial_exposure()`: shortfall sums = 0.0, costs = 0.0.  
- Section C: show `_empty_state()`.  
- Section F coverage bars: `cotton_all` and `fiber_all` may be empty — guard `if not cotton_all.empty`.  
- Section A: `_build_situation_brief()` n_buy == 0 branch returns "All N pairs meet..." string.

---

### R5 — `days_cover == 0` or NaN in BUY rows

**Scenario:** A BUY row has `days_cover = 0.0` (can happen when inventory = 0).  
**New code impact:**  
- `_stockout_date(0)` must return `"N/A"` (not `today + 0 days = today`).  
- `active_buy = df[(df["action"]=="BUY") & (df["days_cover"] > 0)]` filter in Section B2 correctly excludes these rows.  
- `worst = active_buy.loc[active_buy["days_cover"].idxmin()]`: if `active_buy` is empty after this filter, show "—" tile.

---

### R6 — `shortfall` column absent or zero

**Scenario:** Strategy CSV was generated from an older pipeline version that uses different column names.  
**Guard:** In `_compute_financial_exposure()`, check `"shortfall" in df.columns` before summing. If absent, `cotton_shortfall_kgs = 0.0`.  
**Note:** Post Phase 6A pipeline, `shortfall` is guaranteed to be present. This guard is defensive.

---

### R7 — Spark line with fewer than 6 data points

**Scenario:** A commodity CSV has fewer than 6 monthly rows (e.g., a newly added data source).  
**Guard in `_render_spark_line()`:** `series = df[value_col].tail(6)` — `tail(6)` handles fewer rows gracefully.  
**Additional guard:** `if len(series.dropna()) < 2: return` — do not render a 1-point or empty line chart.

---

### R8 — Section D height misalignment with Section C

**Scenario:** Section D (3 stacked cards + 2 spark lines) may be taller than Section C (340px table + caption + footer note).  
**Estimate:** Each D card ≈ 85px, each spark ≈ 55px, 2 spacers × 10px = total ≈ 3×85 + 2×55 + 20 = 385px.  
**Section C:** table 340px + caption ~20px + footer note ~20px = ~380px.  
**Decision:** If D overflows C, reduce spark height to `45` and card padding. Do not hard-code Section D height in outer container — let Streamlit handle natural flow.

---

### R9 — `render_executive_signals_table()` call removal

**Scenario:** The function is defined at line 7306 and called only once (line 9118 in `render_executive_summary()`).  
**Action:** Remove the call only. The function definition is NOT deleted.  
**Verify before removal:** `grep render_executive_signals_table streamlit_app.py` confirms there is exactly one call site (line 9118). Do not delete the function — it may be used by future pages.

---

### R10 — `commodity_pairs` and related locals used outside forecast loop

**Scenario:** `all_summary` list and `commodity_payloads` list are built in the forecast loop — if any subsequent code references these after the loop is removed, it will break.  
**Check:** The Overall Market Summary block (lines 9046–9116) references `all_summary`. Since both the forecast loop and the Overall Summary block are being removed together, this is not a problem.  
**Verify:** After removals, search for any remaining references to `all_summary`, `commodity_payloads`, `usd_pkr_rate` (the local var from `_get_usd_pkr_rate_for_summary()`), and `show_local_in_usd` in the remaining function body. All should be gone.

---

### R11 — `commodity` column casing in strategy_df

**Scenario:** The strategy CSV uses title-case commodity names: `"Cotton"`, `"Fiber"`, `"Stretch Fiber"`, `"Cotton Waste"`.  
**Risk:** Any `df["commodity"] == "cotton"` comparison (lowercase) will fail silently.  
**Rule:** All comparisons in new code must use title-case: `"Cotton"`, `"Fiber"`, `"Stretch Fiber"`.  
**Enforce:** In `_compute_financial_exposure()`, use exact string: `df[df["commodity"] == "Cotton"]`.

---

### R12 — `procurement_dashboard.py` import of `datetime`

**Scenario:** `_stockout_date()` requires `datetime.date.today()` and `datetime.timedelta`.  
**Current state:** Verify whether `datetime` is already imported in `procurement_dashboard.py`.  
**If not:** Add `import datetime` to the imports block at the top.  
**Note:** `pandas` is already imported — `pd.Timestamp.today()` is an alternative, but `datetime.date.today()` is more explicit for this use case.

---

*End of Implementation Specification*  
*Phase: UX-2 · Implementation target: Phase UX-1 approved wireframe · No business logic changes*
