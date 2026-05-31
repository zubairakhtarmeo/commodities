# UX-1 — Executive Summary Redesign
## Wireframe Specification · MG Apparel Commodity Intelligence

**Scope:** Layout · KPI hierarchy · Procurement action visibility · Financial exposure · One-screen design  
**Out of scope:** Styling · Colors · CSS · Component library  
**Data basis:** April 2026 pipeline output — 10 BUY · 2 HOLD · 7 MONITOR  
**Date:** May 2026

---

## Document Structure

1. [Before — Current State Map](#1--before--current-state-map)
2. [Problem Statement](#2--problem-statement)
3. [Design Principles for the Redesign](#3--design-principles-for-the-redesign)
4. [After — Redesigned Layout Wireframe](#4--after--redesigned-layout-wireframe)
5. [Section-by-Section Specification](#5--section-by-section-specification)
6. [KPI Inventory — Before vs After](#6--kpi-inventory--before-vs-after)
7. [Content Moved, Removed, Added](#7--content-moved-removed-added)
8. [Data Derivation — New Metrics](#8--data-derivation--new-metrics)
9. [Scroll Budget Analysis](#9--scroll-budget-analysis)

---

## 1 — Before: Current State Map

### 1.1 Current Scroll Map (top to bottom, actual element order)

```
╔══════════════════════════════════════════════════════════════════════════╗
║  [PAGE HEADER BAR]  "Executive Summary"            MG Apparel · Updated  ║
╠══════════════════════════════════════════════════════════════════════════╣
║  [period badge]  "Procurement period: April 2026 (01-Apr to 30-Apr)"    ║
║                                                                          ║
║  ── 1 — PROCUREMENT STATUS ──────────────────────────────────────────   ║
║  ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐   ║
║  │COTTON    │FIBER     │45-DAY    │45-DAY    │TOTAL     │BUY       │   ║
║  │INVENTORY │INVENTORY │COTTON    │FIBER     │PROCURE-  │RECOMMEN- │   ║
║  │          │          │NEED      │NEED      │MENT GAP  │DATIONS   │   ║
║  │2,976,013 │1,439,149 │7,536,863 │2,484,979 │6,547,679 │10        │   ║
║  │Kgs       │Kgs       │Kgs       │Kgs       │Kgs       │pairs     │   ║
║  └──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘   ║
║                                                                          ║
║  ── 2 — CRITICAL RISKS ──────────────────────────────────────────────   ║
║  ┌─────────────────────────────────────────────────────────────────┐   ║
║  │ Org      │ Commodity │ Inventory │ 45d Need │ Shortfall │ Days  │   ║
║  │ ...10 rows sorted by days cover, 7 columns, full-width table... │   ║
║  └─────────────────────────────────────────────────────────────────┘   ║
║  [caption: "Sorted by lowest days cover. Red = under 7 days..."]        ║
║                                                                          ║
║  ── 3 — MARKET SNAPSHOT ─────────────────────────────────────────────   ║
║  ┌──────────────────┬──────────────────┬──────────────────┐           ║
║  │ICE COTTON NO. 2  │PSF (POLYESTER)   │USD / PKR         │           ║
║  │0.6782            │0.9200            │277.50            │           ║
║  │USD/lb  ↑ 4.2%    │USD/kg  → 0.3%    │↓ 1.1% MoM        │           ║
║  └──────────────────┴──────────────────┴──────────────────┘           ║
║                                                                          ║
║  ── 4 — PROCUREMENT RECOMMENDATION ──────────────────────────────────   ║
║  ┌────────────────────┬────────────────────────────────────────────┐   ║
║  │BUY NOW: 10         │                                            │   ║
║  │53% of tracked pairs│         [DONUT CHART]                      │   ║
║  │HOLD:    2          │   BUY 53% / HOLD 11% / MONITOR 37%         │   ║
║  │11% — stock adequate│                                            │   ║
║  │MONITOR: 7          │                                            │   ║
║  │37% — insuff. data  │                                            │   ║
║  └────────────────────┴────────────────────────────────────────────┘   ║
║                                                                          ║
║  ── 5 — EXECUTIVE INSIGHTS ──────────────────────────────────────────   ║
║  • 10 org-commodity pairs are below the 45-day policy stock threshold.  ║
║  • Cotton procurement required for 6 units. Total gap: 5,059,191 Kgs.  ║
║  • Fiber procurement required for 4 units. Total gap: 1,488,488 Kgs.   ║
║  • Most critical: MTM - Spinning U3 / Fiber has only 6.5 days of cover  ║
║  • 7 pairs flagged MONITOR — consumption data absent                    ║
║                                                                          ║
║  ── MARKET FORECASTS — COMMODITY-BY-COMMODITY ANALYSIS ──────────────   ║
║  ████ COTTON ████████████████████████████████████████████████████████   ║
║  ┌──────────────────────────┬──────────────────────────┐              ║
║  │ International chart      │ Local chart              │              ║
║  └──────────────────────────┴──────────────────────────┘              ║
║  ┌──────────────────────────┬──────────────────────────┐              ║
║  │ International table      │ Local table              │              ║
║  └──────────────────────────┴──────────────────────────┘              ║
║                                                                          ║
║  ████ POLYESTER ████████████████████████████████████████████████████   ║
║  [chart + table] × 2                                                     ║
║                                                                          ║
║  ████ VISCOSE ███████████████████████████████████████████████████████   ║
║  [chart + table] × 2                                                     ║
║                                                                          ║
║  ████ NATURAL GAS ██████████████████████████████████████████████████   ║
║  [chart + table] × 2                                                     ║
║                                                                          ║
║  ████ CRUDE OIL ████████████████████████████████████████████████████   ║
║  [chart + table] × 2                                                     ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 1.2 Current Page Metrics

| Metric | Value |
|---|---|
| Distinct visual sections | 6 (numbered) + market forecast section |
| Scrollable content below first viewport | 5 commodity blocks × ~600px each = ~3,000px additional |
| CEO questions answerable above fold | 0 of 5 |
| Financial data present | None |
| Stockout dates shown | None |
| Time to locate most critical risk | ~30 seconds of scrolling |
| Donut chart information value | Low (3 numbers rendered as a chart) |
| Section that should not be here | "Market Forecasts" (entire bottom half) |

---

## 2 — Problem Statement

**The Executive Summary page currently has two incompatible purposes living in the same view:**

**Purpose A** — Procurement management brief (Sections 1–5): correct intent, execution has structural issues.

**Purpose B** — Commodity market analysis (Market Forecasts section): wrong page entirely. This belongs on the Market pages.

**The result:** A CEO opening the Executive Summary sees a procurement brief that fades into a 3,000px market analysis scroll. The brief itself is well-structured in sequence but fails because:

1. The most critical fact ("MTM Spinning U3 runs out of Fiber in 6 days") is buried in a 10-row table in Section 2 — it is not the first thing visible.
2. There is no financial framing — the procurement recommendation is in kilograms only.
3. Section 4 (donut chart) and Section 3 (market snapshot) are placed after the risk table, meaning a CEO sees the risk, then scrolls down to understand market context. The order is wrong: market context should inform the reading of risk.
4. The Executive Insights bullets — the only narrative in the page — are the last item before the market analysis begins, meaning a CEO who reads in order must work through 4 sections to reach the summary.

---

## 3 — Design Principles for the Redesign

These principles govern every placement decision in the redesign wireframe.

**P1 — Verdict first.** The most important conclusion must be the first thing visible, not the last.

**P2 — Context before detail.** Market prices inform the reading of inventory risk. Show market snapshot before (or beside) the risk table, not after it.

**P3 — Financial framing on every procurement metric.** Every kilogram figure must have a dollar equivalent. A CEO acts on money, not weight.

**P4 — Dates, not days.** "6.5 days of cover" requires mental arithmetic. "Stockout: Jun 05" is a deadline. Show both.

**P5 — One screen for core decisions.** All five CEO questions (risk, inventory, financial, market, action) must be answerable without scrolling on a 1440×900 viewport.

**P6 — Remove Section 4 donut chart.** Three numbers do not need a chart. The space is better used for the action table or financial tiles.

**P7 — Remove the commodity-by-commodity market forecast section.** It belongs on the Market pages. It makes the Executive Summary page infinite in length.

**P8 — Summary counts stay; breakdown detail gets a link.** Show "10 BUY" with a link to Procurement Intelligence. Do not replicate the full 10-row table here.

---

## 4 — After: Redesigned Layout Wireframe

### 4.1 One-Screen Target Layout (1440×900 viewport)

```
╔══════════════════════════════════════════════════════════════════════════╗
║  [PAGE HEADER BAR]  "Executive Summary"     April 2026 · 30 May 2026   ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ╔══════════════════════════════════════════════════════════════════╗   ║
║  ║  SITUATION BRIEF  ─────────────────────────────────────────────  ║   ║
║  ║  "10 procurement actions required. MTM Spinning U3 Fiber         ║   ║
║  ║   stockout in 6 days (Jun 05). Estimated procurement cost:       ║   ║
║  ║   ~$8.9M. ICE Cotton up 4.2% MoM."                               ║   ║
║  ╚══════════════════════════════════════════════════════════════════╝   ║
║  [A]                                                                     ║
║                                                                          ║
║  ┌──────────┬──────────┬──────────┬──────────┬──────────┐             ║
║  │[B1]      │[B2]      │[B3]      │[B4]      │[B5]      │             ║
║  │ACTIONS   │LOWEST    │EST.TOTAL │COTTON    │FIBER     │             ║
║  │REQUIRED  │COVER     │COST (USD)│COVER     │COVER     │             ║
║  │          │          │          │          │          │             ║
║  │   10     │6.5 days  │~$8.9M    │11.5 days │ 6.5 days │             ║
║  │  BUY     │Jun 05    │(BUY rows)│(worst)   │(worst)   │             ║
║  └──────────┴──────────┴──────────┴──────────┴──────────┘             ║
║  [B]                                                                     ║
║                                                                          ║
║  ┌───────────────────────────────────────────┬──────────────────────┐ ║
║  │[C] IMMEDIATE ACTIONS (10 pairs)           │[D] MARKET SNAPSHOT   │ ║
║  │  View all in Procurement Intelligence →   │                      │ ║
║  │                                           │ ICE COTTON NO. 2     │ ║
║  │  ORG       COMMODITY  COVER  STOCKOUT  GAP │ 0.6782 USD/lb        │ ║
║  │  ─────────────────────────────────────── │ ↑ 4.2% MoM  [spark]  │ ║
║  │  MTM-U3    Fiber      6.5d   Jun 05  188K │                      │ ║
║  │  MTM-U5    Cotton    11.5d   Jun 10  317K │ ─────────────────    │ ║
║  │  MTM-U1    Fiber     12.3d   Jun 11  625K │ PSF (POLYESTER)      │ ║
║  │  MTM-U2    Cotton    13.8d   Jun 12 1.8M  │ 0.9200 USD/kg        │ ║
║  │  MTM-U6    Cotton    13.9d   Jun 12 1.1M  │ → 0.3% MoM  [spark]  │ ║
║  │  MSM-U1    Cotton    15.3d   Jun 14 1.7M  │                      │ ║
║  │  MTM-U6    Fiber     16.9d   Jun 15  378K │ ─────────────────    │ ║
║  │  MSM-U1    Fiber     18.1d   Jun 17  240K │ USD / PKR            │ ║
║  │  MTM-U1    Cotton    23.4d   Jun 22  166K │ 277.50               │ ║
║  │  MTM-U5    Fiber     37.8d   Jul 06   58K │ ↓ 1.1% MoM  [spark]  │ ║
║  └───────────────────────────────────────────┴──────────────────────┘ ║
║                                                                          ║
║  ┌─────────────────────────────┬──────────────┬───────────────────────┐║
║  │[E] FINANCIAL EXPOSURE       │[F] POSITION  │[G] EXECUTIVE BRIEF   │║
║  │                             │              │                       │║
║  │ EST. COTTON COST   ~$7.6M   │ 10 BUY       │ • 10 pairs below 45d  │║
║  │ 5,059,191 Kgs shortfall     │ 2 HOLD       │   policy threshold    │║
║  │ @ ICE 0.6782 USD/lb         │ 7 MONITOR    │ • Cotton: 6 units,    │║
║  │                             │              │   5.1M Kgs gap        │║
║  │ EST. FIBER COST    ~$1.4M   │ Total: 19    │ • Fiber: 4 units,     │║
║  │ 1,488,488 Kgs shortfall     │ tracked pairs│   1.5M Kgs gap        │║
║  │ @ PSF 0.9200 USD/kg         │              │ • Most critical:      │║
║  │                             │ [Coverage    │   MTM-U3/Fiber        │║
║  │ TOTAL EXPOSURE     ~$8.9M   │  bar: ██░░░] │   Jun 05 stockout     │║
║  │ (BUY rows, current prices)  │              │ • 2 pairs HOLD —      │║
║  │                             │              │   stock adequate       │║
║  │ Rate: USD/PKR 277.50        │              │ • 7 MONITOR — data    │║
║  │                             │              │   pending              │║
║  └─────────────────────────────┴──────────────┴───────────────────────┘║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 4.2 Section Labels

```
[A] — Situation Brief
[B] — Status Tiles (5 tiles: B1–B5)
[C] — Immediate Actions Table
[D] — Market Snapshot
[E] — Financial Exposure
[F] — Position Summary
[G] — Executive Brief
```

---

## 5 — Section-by-Section Specification

---

### SECTION A — Situation Brief

**Position:** Full width, immediately below page header bar.  
**Height:** ~1 line of large bold text (~40px).  
**Streamlit element:** `st.markdown()` — NOT `st.caption()`.

**Content:** One auto-generated sentence summarizing the top-3 facts.

**Formula:**  
`"{n_buy} procurement actions required. {worst_org_short} {worst_commodity} stockout in {worst_days:.0f} days ({stockout_date}). Estimated cost: ~${total_cost_usd/1_000_000:.1f}M. {top_market_signal}."`

**Real example from April 2026 data:**  
`"10 procurement actions required. MTM Spinning U3 Fiber stockout in 6 days (Jun 05). Estimated procurement cost: ~$8.9M. ICE Cotton up 4.2% MoM."`

**Display rules:**
- When `n_buy == 0`: `"All {total} org-commodity pairs meet 45-day policy requirements. Next review: [next_month]."`
- When pipeline has not run: `"Pipeline data unavailable — run the monthly pipeline to generate this brief."`

**Do NOT use:**
- `st.info()` 
- `st.warning()`
- `st.caption()`
- Any section divider bar (`exec-section-bar`)

---

### SECTION B — Status Tiles (5 tiles across full width)

**Position:** Full width, immediately below Situation Brief.  
**Layout:** `st.columns(5)`  
**Height:** ~100px per card (use existing `.metric-card` class).

#### Tile B1 — Actions Required

```
ACTIONS REQUIRED
10
BUY recommendations
[red border]
```

| Field | Value | Source |
|---|---|---|
| Label | ACTIONS REQUIRED | static |
| Value | `n_buy` | `(df["action"] == "BUY").sum()` |
| Sub-label | "BUY recommendations" | static |
| Border color | `--c-buy` (#dc2626) | conditional: red if > 0, green if == 0 |

#### Tile B2 — Lowest Days Cover

```
LOWEST COVER
6.5 days
MTM-U3 / Fiber  ·  Jun 05
[red border]
```

| Field | Value | Source |
|---|---|---|
| Label | LOWEST COVER | static |
| Value | `{worst_days_cover:.1f} days` | `df[df["action"]=="BUY"]["days_cover"].min()` |
| Sub-label | `{short_org_name} / {commodity}  ·  {stockout_date}` | derived (see Section 8) |
| Border color | `--c-buy` if < 15 days, `--c-monitor` if < 30 days, `--c-hold` if ≥ 30 days | conditional |

#### Tile B3 — Estimated Total Cost

```
EST. PROCUREMENT COST
~$8.9M
Cotton $7.6M + Fiber $1.4M
[dark red border]
```

| Field | Value | Source |
|---|---|---|
| Label | EST. PROCUREMENT COST | static |
| Value | `~${ total_cost_usd / 1_000_000 :.1f}M` | derived (see Section 8) |
| Sub-label | `Cotton ${cotton_cost/1_000_000:.1f}M + Fiber ${fiber_cost/1_000_000:.1f}M` | derived |
| Border color | `--c-buy` | static (procurement cost is always action-oriented) |
| Note | Display only if pipeline has data; show "—" otherwise | conditional |

#### Tile B4 — Cotton Cover (Worst Position)

```
COTTON COVER
11.5 days
Worst: MTM-U5
[red border]
```

| Field | Value | Source |
|---|---|---|
| Label | COTTON COVER | static |
| Value | `{worst_cotton_days:.1f} days` | `df[(df["commodity"]=="Cotton")&(df["action"]=="BUY")]["days_cover"].min()` |
| Sub-label | `Worst: {short_org}` | org with lowest cotton days cover |
| Border color | conditional by days threshold |

#### Tile B5 — Fiber Cover (Worst Position)

```
FIBER COVER
6.5 days
Worst: MTM-U3
[red border]
```

| Field | Value | Source |
|---|---|---|
| Label | FIBER COVER | static |
| Value | `{worst_fiber_days:.1f} days` | `df[(df["commodity"]=="Fiber")&(df["action"]=="BUY")]["days_cover"].min()` |
| Sub-label | `Worst: {short_org}` | org with lowest fiber days cover |
| Border color | conditional by days threshold |

---

### SECTION C — Immediate Actions Table

**Position:** Left 2/3 of the main content row.  
**Layout:** `st.columns([2, 1])` — Section C takes col[0], Section D takes col[1].  
**Header:** "IMMEDIATE ACTIONS — {n_buy} pairs require procurement" + small link "View full analysis →"

**Table columns (7 columns):**

| Column | Width | Content | Source |
|---|---|---|---|
| Org | 20% | Short org name (strip "- Spinning " to save space) | `org_name` |
| Commodity | 12% | Cotton / Fiber / Stretch Fiber | `commodity` |
| Days Cover | 10% | `{days_cover:.1f}d` | `days_cover` |
| Stockout Date | 12% | `{today + timedelta(days_cover):%b %d}` | derived |
| Gap (Kgs) | 15% | `{shortfall:,.0f}` | `shortfall` |
| Est. Cost | 15% | `~${est_cost:,.0f}` | derived (see Section 8) |
| Confidence | 10% | HIGH / MEDIUM / LOW badge | `confidence` |

**Sort order:** ascending `days_cover` (most urgent first).

**Row count displayed:** All 10 BUY rows — the table is the primary decision surface. No truncation.

**Row coloring rules (same as current BUY tab):**
- `days_cover < 7` → light red row background
- `7 ≤ days_cover < 15` → light amber row background  
- `days_cover ≥ 15` → no highlight

**Table height:** Fixed at ~340px to fit within the one-screen layout without internal scroll.

**Below table:**  
`"2 HOLD (adequate stock) · 7 MONITOR (data pending)  [View details →]"`

**"View full analysis →" link** navigates to the Procurement Intelligence page.

**Data populated from April 2026 pipeline:**

```
ORG            COMMODITY  COVER   STOCKOUT   GAP (KGS)  EST. COST    CONF
MTM-Spin-U3    Fiber       6.5d   Jun 05     188,062    ~$173K       HIGH
MTM-Spin-U5    Cotton     11.5d   Jun 10     316,553    ~$473K       HIGH
MTM-Spin-U1    Fiber      12.3d   Jun 11     624,848    ~$575K       HIGH
MTM-Spin-U2    Cotton     13.8d   Jun 12   1,784,825   ~$2.67M       HIGH
MTM-Spin-U6    Cotton     13.9d   Jun 12   1,116,707   ~$1.67M       HIGH
MSM-Spin-U1    Cotton     15.3d   Jun 14   1,675,111   ~$2.51M       HIGH
MTM-Spin-U6    Fiber      16.9d   Jun 15     377,586    ~$347K       HIGH
MSM-Spin-U1    Fiber      18.1d   Jun 17     240,246    ~$221K       HIGH
MTM-Spin-U1    Cotton     23.4d   Jun 22     165,997    ~$248K       HIGH
MTM-Spin-U5    Fiber      37.8d   Jul 06      57,744     ~$53K       HIGH
```

---

### SECTION D — Market Snapshot

**Position:** Right 1/3 of the main content row (beside Section C).  
**Layout:** Continuation of `st.columns([2, 1])` — Section D in col[1].  
**Height:** Matches Section C (~380px total).

**Three stacked price cards (no columns — vertical stack):**

#### Card D1 — ICE Cotton No. 2

```
ICE COTTON NO. 2
0.6782 USD/lb
↑ 4.2% MoM    [Apr 2026]
[spark line: last 6 data points]
```

| Field | Value | Source |
|---|---|---|
| Label | ICE COTTON NO. 2 | static |
| Price | `{ice_price:.4f}` | `_market_snapshot["ice_cotton"]["price"]` |
| Currency | USD/lb | `ice["currency"]` |
| Change | `{arrow} {abs(change):.1f}% MoM` | `ice["change"]` |
| Date | `{ice["date"]}` | displayed as "Apr 2026" |
| Spark line | Last 6 monthly values as a tiny inline line chart | from `load_commodity_data` history |

**Spark line specification:** 60px height, no axis, no labels, line only. Color: red if last change positive (price rising = risk for buyer), green if negative (price falling = opportunity).

#### Card D2 — PSF (Polyester)

```
PSF (POLYESTER)
0.9200 USD/kg
→ 0.3% MoM    [Apr 2026]
[spark line]
```

Same structure as D1. Currency: USD/kg.

#### Card D3 — USD / PKR

```
USD / PKR
277.50
↓ 1.1% MoM    [live rate]
[spark line]
```

Same structure. Currency: PKR per USD. Note: PKR weakening (↑ rate) is a cost risk for USD-denominated purchases.

---

### SECTION E — Financial Exposure

**Position:** Left column of the bottom row.  
**Layout:** `st.columns([3, 2, 3])` — Section E in col[0].

**Content:** Two financial tiles stacked, plus a total line.

```
FINANCIAL EXPOSURE
─────────────────────────────────────
ESTIMATED COTTON PROCUREMENT
~$7.6M
5,059,191 Kgs × ICE 0.6782 USD/lb
(× USD/PKR = PKR ~2.1B)

ESTIMATED FIBER PROCUREMENT
~$1.4M
1,488,488 Kgs × PSF 0.9200 USD/kg
(× USD/PKR = PKR ~388M)

──────────────────────
TOTAL EXPOSURE   ~$8.9M
At current market rates · April 2026
```

**Display rules:**
- Values shown in USD by default.
- PKR equivalent shown in sub-label using live or fallback `usd_pkr_rate`.
- Labeled "ESTIMATED" to communicate that these are directional figures, not locked quotes.
- `"At current market rates"` disclaimer — one line, small text.
- When no pipeline data: show `"—"` in value, explanation in sub-label.

**KPIs in this section (new — not present in current design):**

| KPI | Formula | April 2026 Value |
|---|---|---|
| Est. Cotton Cost (USD) | `cotton_shortfall_kgs × (ice_price_usd_lb × 2.20462)` | ~$7.6M |
| Est. Fiber Cost (USD) | `fiber_shortfall_kgs × psf_price_usd_kg` | ~$1.4M |
| Est. Total Cost (USD) | cotton + fiber | ~$8.9M |
| Est. Total Cost (PKR) | `total_usd × usd_pkr_rate` | ~PKR 2.5B |

---

### SECTION F — Position Summary

**Position:** Middle column of the bottom row.  
**Layout:** `st.columns([3, 2, 3])` — Section F in col[1].

**Content:** Three action counts (text, not a chart) plus a compact coverage bar.

```
POSITION SUMMARY
──────────────────────
10  BUY  — Act now
 2  HOLD — Adequate
 7  MON  — Data pending
──────────────────────
19  Total pairs tracked

COVERAGE (avg: 37.4 days)
Cotton  [████████░░░░] 17.5d avg
Fiber   [██████░░░░░░] 15.1d avg
[45d target line]
```

**Coverage bar specification:**
- Two horizontal bars: Cotton avg days cover, Fiber avg days cover.
- Scale: 0 to 90 days.
- 45-day reference line drawn as a vertical marker.
- Bars colored by threshold: red if avg < 15, amber if 15–30, blue if > 30.
- Bar labels: `{commodity}  {avg_days:.0f}d avg`.

**Do NOT use a donut or pie chart.** The donut is removed. Three numbers plus a bar is clearer.

**Data:**

| KPI | Value | Source |
|---|---|---|
| BUY count | 10 | `(df["action"]=="BUY").sum()` |
| HOLD count | 2 | `(df["action"]=="HOLD").sum()` |
| MONITOR count | 7 | `(df["action"]=="MONITOR").sum()` |
| Total pairs | 19 | `len(df)` |
| Cotton avg days cover (BUY) | 15.3d | mean of cotton BUY rows with days_cover > 0 |
| Fiber avg days cover (BUY) | 15.5d | mean of fiber BUY rows with days_cover > 0 |
| Overall avg days cover (active) | 37.4d | all rows with days_cover > 0 |

---

### SECTION G — Executive Brief

**Position:** Right column of the bottom row.  
**Layout:** `st.columns([3, 2, 3])` — Section G in col[2].

**Content:** Auto-generated bullets — the existing `_insight_bullets()` function output, displayed at a readable size.

**Bullet display rules:**
- Font size: `0.88rem` (current) → **increase to `0.9rem`** minimum.
- Max 5 bullets.
- Bullets must fit within the allocated column height without scroll.
- Title: "EXECUTIVE BRIEF" (small uppercase label, same as metric-card `.metric-label` style).

**Bullet content from April 2026 data:**
1. "10 org-commodity pairs are below the 45-day policy stock threshold."
2. "Cotton procurement required for 6 units. Total gap: 5,059,191 Kgs (~$7.6M)."
3. "Fiber procurement required for 4 units. Total gap: 1,488,488 Kgs (~$1.4M)."
4. "Most critical: MTM - Spinning U3 / Fiber has 6.5 days of cover. Stockout: Jun 05."
5. "2 pairs (HOLD) have adequate stock. 7 pairs in MONITOR (data pending)."

**Change from current:** Bullets in Section G now include the cost dimension in parentheses. The `_insight_bullets()` function needs a `market_snapshot` argument added so it can compute `~$7.6M` alongside the Kgs figure.

---

## 6 — KPI Inventory: Before vs After

### 6.1 KPIs Present in Both (retained, same data)

| KPI | Before Location | After Location | Change |
|---|---|---|---|
| BUY count | Section 1, card 6 | Tile B1 + Section F | Promoted to most prominent position |
| HOLD count | Section 4, stacked card | Section F | Demoted to position summary (less prominent) |
| MONITOR count | Section 4, stacked card | Section F | Same |
| Cotton inventory (Kgs) | Section 1, card 1 | Removed from tiles | Moved to table notes / accessible via Procurement page |
| Fiber inventory (Kgs) | Section 1, card 2 | Removed from tiles | Same |
| 45-day Cotton need (Kgs) | Section 1, card 3 | Removed from tiles | Accessible via Procurement page |
| 45-day Fiber need (Kgs) | Section 1, card 4 | Removed from tiles | Same |
| Total procurement gap (Kgs) | Section 1, card 5 | Section E (sub-label) | Still shown, but subordinate to cost |
| Shortfall per org-commodity (Kgs) | Section 2 table | Section C table | Retained, now with est. cost alongside |
| Days cover per row | Section 2 table | Section C table | Retained, with stockout date added |
| ICE Cotton price | Section 3, card 1 | Section D, card 1 | Same content, spark line added |
| PSF price | Section 3, card 2 | Section D, card 2 | Same content, spark line added |
| USD/PKR | Section 3, card 3 | Section D, card 3 | Same content, spark line added |
| Insight bullets | Section 5, bullet list | Section G | Same content, font size increased, cost dimension added |

### 6.2 KPIs Removed from Executive Summary

| KPI | Current Location | Removed Because | Where to Find It |
|---|---|---|---|
| 45-day Cotton need (Kgs) | Section 1, card 3 | Subordinate metric — useful in Procurement detail, not executive brief | Procurement Intelligence → Overview |
| 45-day Fiber need (Kgs) | Section 1, card 4 | Same | Same |
| Cotton inventory (Kgs) | Section 1, card 1 | Kgs alone without days-cover context is less actionable than the days tile | Procurement Intelligence → Overview |
| Fiber inventory (Kgs) | Section 1, card 2 | Same | Same |
| Section 4 donut chart | Section 4 | 3 numbers rendered as a chart; the numbers in Section F are clearer | Nowhere — eliminated |
| Section 4 BUY/HOLD/MONITOR stacked cards | Section 4 | Duplicates tiles B1 and Section F | Merged into Section F |
| All Market Forecast charts (Cotton, Polyester, Viscose, Natural Gas, Crude Oil) | Bottom of page | Wrong page — this is a market analysis section | International Market / Pakistan Local pages |

### 6.3 KPIs Added (New)

| KPI | After Location | Formula | Why Added |
|---|---|---|---|
| Stockout date per BUY row | Section C table | `today + timedelta(days=days_cover)` | Dates drive action; days require mental arithmetic |
| Lowest days cover (Worst-case) | Tile B2 | `df[df["action"]=="BUY"]["days_cover"].min()` | CEO needs the worst-case number at a glance |
| Worst org + stockout date | Tile B2 sub-label | Same row as above | Gives the worst-case identity without opening a table |
| Worst Cotton days cover | Tile B4 | `df[(df["commodity"]=="Cotton")&(df["action"]=="BUY")]["days_cover"].min()` | Cotton-specific urgency visible on status bar |
| Worst Fiber days cover | Tile B5 | `df[(df["commodity"]=="Fiber")&(df["action"]=="BUY")]["days_cover"].min()` | Fiber-specific urgency visible on status bar |
| Est. Cotton procurement cost (USD) | Section E | `cotton_shortfall_kgs × ice_price_usd_per_kg` | Financial framing — CFO decision metric |
| Est. Fiber procurement cost (USD) | Section E | `fiber_shortfall_kgs × psf_price_usd_per_kg` | Financial framing — CFO decision metric |
| Est. Total procurement cost (USD) | Tile B3 + Section E | Sum of above | Top-level financial exposure |
| Est. Total procurement cost (PKR) | Section E sub-label | `total_usd × usd_pkr_rate` | Budgeting context for local finance team |
| Est. cost per BUY row (USD) | Section C table | `shortfall_kgs × commodity_price_usd_per_kg` | Row-level financial context |
| Situation brief (narrative) | Section A | Auto-generated string | Headline summary — no mental arithmetic required |
| Cotton coverage bar | Section F | Visual bar using avg days cover | Replaces donut with more informative visualization |
| Fiber coverage bar | Section F | Same | Same |

---

## 7 — Content Moved, Removed, Added

### Removed from Executive Summary (not deleted — moved elsewhere)

| Content | Moved to |
|---|---|
| Cotton (International + Local) price chart + forecast table | International Market → Cotton tab |
| Polyester (International + Local) price chart + forecast table | International Market → Polyester tab |
| Viscose (International + Local) price chart + forecast table | International Market → Viscose tab |
| Natural Gas chart + forecast table | International Market → Natural Gas tab |
| Crude Oil chart + forecast table | International Market → Crude Oil tab |
| Section 4 donut chart | Eliminated |
| Section 4 BUY/HOLD/MONITOR stacked cards (duplicate of tiles) | Merged into Section F |
| `st.caption()` for USD/PKR source disclosure | Promoted to `st.markdown()` in Section D sub-label |

### Removed from Executive Summary (eliminated entirely)

| Content | Reason |
|---|---|
| "Market Forecasts — Commodity-by-Commodity Analysis" section header bar | Section does not exist in redesign |
| 10-row Critical Risks table (Section 2) as full-width element | Replaced by Section C which has the same data plus cost + stockout date, in a better layout position |
| Section divider bars numbered "1 — Procurement Status", "2 — Critical Risks", etc. | Visual overhead without structural benefit. Sections are now self-evident from their content. |
| `st.caption()` with pipeline instructions ("Run scripts/...") | Developer text — removed from exec view entirely |

### Added to Executive Summary

| Content | Location | Why |
|---|---|---|
| Situation Brief (one sentence) | Section A | CEO headline — the verdict before any data |
| Est. total cost tile | Tile B3 | Financial framing at top of page |
| Stockout date column | Section C table | "Jun 05" is actionable; "6.5 days" requires arithmetic |
| Est. cost per row column | Section C table | Row-level financial context |
| Financial Exposure section | Section E | Full cost breakdown with Cotton/Fiber split |
| Spark lines (3 mini charts) | Section D | Price direction in 6 data points without a full chart |
| Cotton/Fiber coverage bars | Section F | Visual coverage position replacing donut |

---

## 8 — Data Derivation: New Metrics

All new metrics derive from data already available in `strategy_df`, `market_snapshot`, and `usd_pkr_rate`. No new pipeline work is required.

### 8.1 Stockout Date

```
stockout_date = today + timedelta(days = row["days_cover"])
```

- `today` = `datetime.date.today()`
- `days_cover` is already in `strategy_df`
- Display format: `"%b %d"` → "Jun 05"
- Edge case: if `days_cover == 0` (MONITOR rows), display `"N/A"`

### 8.2 Short Org Name

```
short_org = org_name
    .replace("- Spinning ", "-Spin-")
    .replace("- Weaving ",  "-Weav-")
    → "MTM - Spinning U3"  →  "MTM-Spin-U3"
```

Used in Tile B2, B4, B5, and Section C table Org column to prevent wrapping.

### 8.3 Estimated Procurement Cost per Commodity

**Cotton cost (USD):**
```
ice_price_usd_per_lb = market_snapshot["ice_cotton"]["price"]
ice_price_usd_per_kg = ice_price_usd_per_lb × 2.20462

cotton_shortfall_kgs = df[(df["commodity"]=="Cotton") & (df["action"]=="BUY")]["shortfall"].sum()
cotton_cost_usd      = cotton_shortfall_kgs × ice_price_usd_per_kg
```

**Fiber cost (USD):**
```
psf_price_usd_per_kg = market_snapshot["psf"]["price"]

fiber_shortfall_kgs = df[(df["commodity"]=="Fiber") & (df["action"]=="BUY")]["shortfall"].sum()
fiber_cost_usd      = fiber_shortfall_kgs × psf_price_usd_per_kg
```

**Total:**
```
total_cost_usd = cotton_cost_usd + fiber_cost_usd
```

**PKR equivalent:**
```
total_cost_pkr = total_cost_usd × usd_pkr_rate
```

**April 2026 actuals (for reference):**
- Ice price: 0.6782 USD/lb → 1.4953 USD/kg
- PSF price: 0.9200 USD/kg
- Cotton shortfall: 5,059,191 Kgs → ~$7.57M
- Fiber shortfall: 1,488,488 Kgs → ~$1.37M
- **Total: ~$8.93M**
- PKR @ 277.50: ~PKR 2.48B

**Display as:** `~$8.9M` (one decimal, truncated to M, with tilde to signal "estimate").

**Display fallback:** If `ice_price` or `psf_price` is None → show `"N/A"` in cost tile with note `"Market data unavailable"`.

### 8.4 Estimated Cost per BUY Row

```
commodity_price = {
    "Cotton":       ice_price_usd_per_kg,
    "Fiber":        psf_price_usd_per_kg,
    "Stretch Fiber": psf_price_usd_per_kg,  # use PSF as proxy
    "Cotton Waste": 0.0,                     # no price reference
}
row["est_cost_usd"] = row["shortfall"] × commodity_price[row["commodity"]]
```

**Display format:** `~${ est_cost_usd:,.0f }` when < $1M, `~${ est_cost_usd/1_000_000:.2f }M` when ≥ $1M.

### 8.5 Situation Brief String

```python
def _situation_brief(df, market_snapshot, usd_pkr_rate) -> str:
    if df.empty:
        return "Pipeline data unavailable — run the monthly pipeline to generate this brief."

    n_buy = (df["action"] == "BUY").sum()
    if n_buy == 0:
        return (
            f"All {len(df)} org-commodity pairs meet the 45-day policy requirement. "
            "No procurement action required this period."
        )

    buy      = df[df["action"] == "BUY"]
    worst    = buy.loc[buy["days_cover"].idxmin()]
    stockout = today + timedelta(days=worst["days_cover"])
    cost_str = f"~${total_cost_usd/1_000_000:.1f}M" if market_data_available else ""
    market_str = (
        f"ICE Cotton {'up' if ice_change > 0 else 'down'} {abs(ice_change):.1f}% MoM."
        if ice_price else ""
    )

    return (
        f"{n_buy} procurement actions required. "
        f"{short_org(worst['org_name'])} {worst['commodity']} "
        f"stockout in {worst['days_cover']:.0f} days ({stockout:%b %d}). "
        + (f"Estimated cost: {cost_str}. " if cost_str else "")
        + market_str
    )
```

---

## 9 — Scroll Budget Analysis

### 9.1 Current vs Redesigned Scroll Requirements

| Content | Current Scroll Depth | Redesigned Scroll Depth |
|---|---|---|
| Situation Brief | Does not exist | 0px — above fold |
| Status tiles (B1–B5) | 0px — first section | 50px — just below brief |
| Immediate Actions (all 10 BUY rows) | ~120px (scroll to Section 2) | 160px — same viewport |
| Market Snapshot | ~500px (Section 3, below risk table) | 160px — same viewport as actions |
| Financial Exposure | Does not exist | 480px — below main row |
| Position Summary / Brief | ~650px (Sections 4 + 5) | 480px — same row as exposure |
| Market Forecasts (Cotton) | ~950px | Removed |
| Market Forecasts (Polyester) | ~1,650px | Removed |
| Market Forecasts (Viscose) | ~2,350px | Removed |
| Market Forecasts (Natural Gas) | ~3,050px | Removed |
| Market Forecasts (Crude Oil) | ~3,750px | Removed |

### 9.2 CEO 30-Second Test: Redesigned vs Current

| Question | Current: answered at scroll depth | Redesigned: answered at scroll depth |
|---|---|---|
| How many urgent procurement actions? | 0px (tile) | 0px (Tile B1, same as before) |
| Who is most at risk and when? | ~120px (Section 2 table) | 0px (Tile B2 — worst-case tile) |
| What will this cost? | Never | 0px (Tile B3) |
| What is the market doing? | ~500px (Section 3) | 160px (Section D, beside actions) |
| What should I do? | ~650px (Section 5 bullets) | 480px (Section G) — or 0px via Section A brief |

**Before:** 4 of 5 CEO questions require scrolling. 1 requires interpretation of bare Kgs numbers.  
**After:** All 5 CEO questions answered within 480px (half a standard viewport). The Situation Brief answers 3 of 5 immediately at 0px scroll.

---

*End of Wireframe Specification*  
*Implementation: Phase UX-1 · No styling changes required · All data available from existing pipeline*
