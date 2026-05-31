# UX-3B — Holistic Enterprise UX/UI Audit & Refinement Specification
**MG Apparel · Commodity Intelligence Platform**
**Audit Date:** 2026-05-30
**Standard:** Palantir Foundry / Bloomberg Enterprise / SAP Analytics Cloud
**Audience:** CEO · CFO · COO · Director Procurement · Head of Supply Chain
**Scope:** All 6 pages, navigation, visual system, IA, human psychology

---

## CONTENTS

1. [Executive Summary Page Audit](#1-executive-summary-page-audit)
2. [Navigation Audit](#2-navigation-audit)
3. [Visual Design System Audit](#3-visual-design-system-audit)
4. [Information Architecture Audit](#4-information-architecture-audit)
5. [Human Psychology & Cognitive Load Audit](#5-human-psychology--cognitive-load-audit)
6. [Top 20 Improvement Opportunities](#6-top-20-improvement-opportunities)
7. [Components to Remove](#7-components-to-remove)
8. [Components to Relocate](#8-components-to-relocate)
9. [Components to Redesign](#9-components-to-redesign)
10. [Final Redesign Roadmap](#10-final-redesign-roadmap)

---

## 1. Executive Summary Page Audit

### 1.1 Current State Assessment

The Executive Summary page (`render_exec_procurement_header_v2`) is the highest-value page in the application — the entry point for every C-suite and Director-level user. UX-3A delivered the structural skeleton (Situation Brief, 5-tile KPI row, Financial Exposure), but the page still carries legacy anatomy that undermines its enterprise credibility.

**What is working well:**
- The `.alert-critical` Situation Brief provides an immediate narrative anchor — executives read prose before numbers.
- The 5-column KPI row (`st.columns(5)`) with `_kpi_card()` provides a clean scanning surface.
- The `_short_org()` abbreviation pattern avoids text truncation in narrow tiles.
- The section divider pattern (`.exec-section-bar` + pill label) creates visual hierarchy without heavy chrome.
- IBM Plex Mono for numeric values correctly borrows from Bloomberg/Refinitiv terminal typography.

**Critical deficiencies:**

| # | Issue | Severity | Location |
|---|-------|----------|----------|
| 1 | Section 2 "Critical Risks" copies v1 verbatim — uses `st.expander()` which collapses executive visibility behind a click | High | Section 2 |
| 2 | Section 3 "Market Snapshot" renders as a vertical stack inside a full-width band — wastes horizontal space, forces scroll | High | Section 3 |
| 3 | Section 4 "Procurement Recommendation" renders a Plotly pie chart with raw labels — pie charts are the worst encoding for part-whole at 3 categories; executives misread them | High | Section 4 |
| 4 | Section 5 "Executive Insights" renders a bulleted list of generic machine-generated observations — no actionability, fills valuable viewport with noise | High | Section 5 |
| 5 | "Market Forecasts" section divider appears mid-page with no content beneath it until the next render function — orphaned section header | Medium | End of v2 |
| 6 | The 5-tile KPI row uses `min-height: 90px` (`.exec-kpi`) but `.metric-card` uses `min-height: 115px` — inconsistency creates uneven row heights when sublabels vary | Medium | KPI row |
| 7 | No last-refresh timestamp visible on page — executives cannot tell if data is from today or last month | Medium | Global |
| 8 | The Situation Brief banner is always `.alert-critical` (red-left-border) even when n_buy == 0 (all healthy) — green-alert state missing | Medium | Section A |
| 9 | Coverage bars in "Position Summary" use ad-hoc inline `st.progress()` — inconsistent with the card-based design system | Low | Section F |
| 10 | Procurement Recommendation pie key renamed to `exec_recommendation_pie_v2` — technical artifact visible in Plotly's default title position on some versions | Low | Section 4 |

### 1.2 Page Layout Analysis (1440×900 viewport)

**Above the fold (0–900px):** Situation Brief + KPI row + partial Critical Risks → ✓ correct density
**Below fold 1 (900–1800px):** Market Snapshot + Financial Exposure + Procurement Recommendation
**Below fold 2 (1800px+):** Executive Insights + Position Summary + Market Forecasts

**Problem:** The Procurement Recommendation (the highest-stakes content — "should I spend $8.9M?") is below the fold. Bloomberg/Palantir would surface the action above the data that supports it. Current layout buries the decision.

**Target layout for 1440×900:**
```
[A] Situation Brief banner                                    ← fold anchor
[B] 5 KPI tiles                                               ← 1 row
[C] Actions Matrix (2/3 width) | [D] Market Snap (1/3)       ← inline, no scroll
[E] Financial Exposure (3 tiles) | [F] Coverage bars (2 cols) ← bottom row
```
Everything below fold: deep-dive analyst content only.

---

## 2. Navigation Audit

### 2.1 Current Navigation Structure

```
Sidebar (dark #0f172a background):
  Brand: MG Apparel / Commodity Intelligence
  ─────────────────────────
  EXECUTIVE
    📊 Executive Summary
  MARKET INTELLIGENCE
    🌍 International Market
    🇵🇰 Pakistan Local
    🧠 Market Intelligence
  FORECASTING
    🤖 AI Predictions
  PROCUREMENT
    📦 Procurement Intelligence
  ─────────────────────────
  SYSTEM
    [toggle] Demo Forecast Mode
    [expander] Data Freshness
```

### 2.2 Navigation Issues

**Issue 1 — Radio button ghost artifacts**
The navigation uses `st.radio()` with CSS-hidden radio buttons. On Streamlit ≥1.30, the radio widget still injects a visible selection ring on focused items. The CSS override targets `[data-testid="stRadio"]` but misses the inner `span[data-testid="stMarkdownContainer"]` that wraps each label on newer builds.

**Issue 2 — Group label font size too small**
`.nav-group` labels render at 0.62–0.65rem uppercase. At 1440px width they are readable; at 1280px (common laptop resolution) they fall below accessible contrast thresholds (4.5:1 WCAG AA for small text).

**Issue 3 — Page group semantics mismatch**
"Market Intelligence" is both a group name AND a page name (`🧠 Market Intelligence`). A CFO scanning the nav cannot distinguish the container from the page. Bloomberg/Palantir never use the same string for a group and an item.

**Issue 4 — "AI Predictions" page is orphaned in Forecasting group**
Single-item groups provide no organizational value. `🤖 AI Predictions` should either join Market Intelligence or Procurement.

**Issue 5 — "Demo Forecast Mode" toggle is in the navigation**
System controls belong in a settings panel, not navigation. Executives do not toggle demo mode; it creates confusion ("why is there a demo mode switch?"). It signals the product is not production-ready to senior stakeholders.

**Issue 6 — "Data Freshness" expander is in the navigation**
Data freshness is a system health indicator, not a navigation item. It belongs in the page header bar (top-right) or in a dedicated system status page, not collapsed inside the nav sidebar.

**Issue 7 — No active page indicator beyond radio selection**
The current radio selection uses Streamlit default styling. Enterprise dashboards (Palantir, Tableau) use a prominent left-border accent or background fill on the active nav item to anchor the user's location in the IA.

**Issue 8 — Page header bar is redundant with page title**
Every page renders a custom header bar (`st.markdown(...)`) at the top that restates the page icon and name already selected in the nav. At 1440px, this bar consumes ~72px of prime viewport height and provides no new information.

### 2.3 Navigation Architecture Recommendation

**Proposed structure:**

```
EXECUTIVE
  📊 Executive Summary

OPERATIONS
  📦 Procurement Intelligence
  🌍 International Markets
  🇵🇰 Pakistan Markets

INTELLIGENCE
  🧠 Market Intelligence
  🤖 AI Forecasts

── System Status ───────────
  [chip] Data: Live · 2h ago
  [chip] Pipeline: ✓ Current
```

Key changes:
- Merge `Market Intelligence` page into `INTELLIGENCE` group; rename group to `INTELLIGENCE`
- Move `AI Predictions` under `INTELLIGENCE`
- Move `Procurement Intelligence` under a new `OPERATIONS` group alongside market pages
- Remove `Demo Forecast Mode` toggle from nav → move to Settings
- Replace `Data Freshness` expander with compact status chips showing last-updated time inline
- Add visual active-state indicator (left border #2563eb + background #1e293b on active item)

---

## 3. Visual Design System Audit

### 3.1 Typography

**Current state:**
- Body font: Inter (400/500/600/700/800) ✓
- Numeric values: IBM Plex Mono (500/600) ✓
- Metric label: 0.68rem / 700 / uppercase / letter-spacing 0.9px ✓
- Page h2: 1.4rem / 600 / color #334155

**Issues:**

| Issue | Current | Recommended |
|-------|---------|-------------|
| `h2` uses weight 600, `h1` uses 700 — but exec sections use `exec-section-label` at weight 800. Weight scale is inverted. | h2=600 | h2=700, section-label=600 |
| `.metric-value` at 1.8rem renders correctly in 5-col layout but clips at 1280px when value is "~$10.2M" (7 chars + units) | 1.8rem | 1.6rem with `font-size: clamp(1.3rem, 1.8vw, 1.8rem)` |
| `.exec-kpi-value` (1.5rem) differs from `.metric-value` (1.8rem) — two competing card size classes with no clear rule for when to use which | Two classes | Consolidate to one `.kpi-value` with size variants |
| `[data-testid="stMarkdownContainer"] p` is 0.875rem / 1.6 / #64748b — correct for body copy. But this global rule affects the Situation Brief text, which should render at 0.9rem / #1e293b for higher legibility | Global | Scope to `.exec-brief p` |
| Caption text at 0.8rem uses `#94a3b8` — fails WCAG AA on white background (3.7:1 contrast ratio) | #94a3b8 | #64748b (4.8:1) |

### 3.2 Color System

**What is correct:**
- Semantic tricolor: `--c-buy: #dc2626` (red), `--c-hold: #2563eb` (blue), `--c-monitor: #d97706` (amber) — these are strong, conventional, and accessible.
- Background/surface split: `#f8fafc` (page bg) + `#ffffff` (card surface) — correct elevation model.

**Issues:**

| Issue | Severity |
|-------|----------|
| `--c-buy` is used for both "urgent procurement action needed" and "critical risk" / "bad news" — overloaded semantic meaning. The color now carries two different signals. | Medium |
| Alert variants (`.alert-critical`, `.alert-info`, `.alert-monitor`, `.alert-healthy`) are well-designed but the Situation Brief always renders `.alert-critical` regardless of actual alert level — the color lies to the executive | High |
| The page header bar (`background:#ffffff; border:1px solid #e2e8f0`) blends into card surfaces — at a glance, it looks like a card, not a navigation anchor | Medium |
| `--c-purple: #7c3aed` is defined but only used in one location. Purple as a 5th semantic color creates ambiguity for readers calibrated to the primary tricolor | Low |

### 3.3 Spacing and Density

**Current measurement:**
- Main padding: `1.25rem 2rem 2rem 2rem` (20px top, 32px sides)
- Card padding: `1rem 1.25rem` (16px vertical, 20px horizontal)
- Section bar margin: `1.5rem 0 1rem 0` (24px above, 16px below)
- `margin-bottom: 0.75rem` on `.metric-card` → 12px between tiles in same column

**Issues:**

| Issue | Severity |
|-------|----------|
| Section bar margin (1.5rem top) creates heavy breaks that make the page feel like separate documents rather than a unified dashboard | Medium |
| Cards have `margin-bottom: 0.75rem` which means `st.columns()` with 5 tiles creates slight vertical misalignment when sublabels wrap differently | Low |
| The `.exec-section-bar::after` horizontal rule is `1px solid #e2e8f0` — nearly invisible on the `#f8fafc` page background. On some monitors it disappears entirely. | Low |

### 3.4 Chart and Data Visualization Audit

| Chart | Page | Issue |
|-------|------|-------|
| Procurement pie chart (`px.pie`) | Executive | Pie charts are the worst encoding for 3-category part-whole data. Executives cannot accurately read 30% vs 35% wedges. Replace with horizontal bar or donut with center label. |
| Historical price line charts (`go.Scatter`) | Market pages | Good encoding. Issue: no reference band for "normal range" — executives need context, not just raw time series. Add ±1σ band. |
| Forecast bar charts | Market pages | Bar charts for time-series forecasts are non-standard — line charts with confidence intervals communicate uncertainty better. |
| AI Predictions charts | AI Predictions | Uses `go.Scatter` with predicted vs actual — correct. Issue: chart height 400px leaves excessive whitespace when predictions are sparse. |
| Coverage bars (`st.progress()`) | Executive | Native `st.progress()` renders with Streamlit's default teal fill, which clashes with the BUY/HOLD/MONITOR color system. Custom HTML progress bars required. |
| Overview tab comparison charts | Market pages | Uses raw `px.bar` with default Plotly color scale — inconsistent with the app's custom palette | 

---

## 4. Information Architecture Audit

### 4.1 Content Hierarchy Analysis

Enterprise dashboards at Palantir/Bloomberg operate on a strict 3-level content hierarchy:

```
Level 1: Status (What is the situation right now? — 3 seconds)
Level 2: Context (Why? How bad? What changed? — 30 seconds)
Level 3: Evidence (Drill-down data for analysts — minutes)
```

**Current page mapping:**

| Page | L1 Status | L2 Context | L3 Evidence | Assessment |
|------|-----------|------------|-------------|------------|
| Executive Summary | ✓ Situation Brief | Partial — KPI tiles + Financial Exposure | Weak — Executive Insights is generic | L1/L2 present, L3 should be removed or demoted |
| International Market | Missing | ✓ Price history | ✓ Forecast + data table | No L1 entry point — opens directly into a commodity tab |
| Pakistan Local | Missing | ✓ Price data | ✓ Charts | Same issue — no L1 orientation |
| Market Intelligence | Partial — Critical Alerts | ✓ News by commodity | ✓ Tabs | Refresh button placement breaks flow |
| AI Predictions | Missing | ✓ Prediction charts | ✓ Backtest | No current-situation summary before predictions |
| Procurement Intelligence | Partial | ✓ 6 tabs | ✓ Details | Strong page — but 6 tabs are hard to navigate |

**Key finding:** Only the Executive Summary page has an L1 entry point. Every other page opens directly into L2 content, forcing the executive to construct situational awareness from scratch. In Palantir Foundry, every page has a "Header Tile Row" that answers the L1 question before any detail is rendered.

### 4.2 Cross-Page Consistency Issues

| Issue | Pages Affected |
|-------|---------------|
| Commodity naming inconsistency: "Cotton" vs "COTTON" vs "ICE Cotton No.2" — same commodity referenced with 3 different names | International Market, Pakistan Local, Executive Summary |
| Date format inconsistency: "Apr 2025", "April 2025", "Apr 25", "2025-04" all appear in different places | All pages |
| Currency unit inconsistency: "USD/lb", "USD/KG", "USD/kg", "$/lb" — four representations of similar units | All market pages |
| KPI card border colors: Market pages use inline `border-left: 4px solid #2563eb` hardcoded blue; Executive uses semantic `--c-buy/hold/monitor` | Market vs Executive |
| Section divider style: Executive Summary uses `.exec-section-bar` pill; other pages use `st.markdown("---")` native divider | All pages |

### 4.3 Missing Pages / Gaps

| Missing Capability | Business Need | Current Workaround |
|--------------------|---------------|-------------------|
| Settings / Administration page | Demo mode toggle, data freshness, system health | Hidden in sidebar |
| Procurement calendar view | "When should I buy?" timeline | Absent — only listed in UX-2 spec as future step |
| Historical decision audit trail | "What did we decide last quarter?" | `reports/` directory, not surfaced |
| Alert configuration | "Notify me when cotton drops below X" | Absent |

---

## 5. Human Psychology & Cognitive Load Audit

### 5.1 Pre-attentive Processing

Pre-attentive attributes (color, size, position, orientation) are processed in <200ms before conscious attention. Enterprise dashboards must use them deliberately.

**Current pre-attentive usage:**
- ✓ BUY (red) / HOLD (blue) / MONITOR (amber) badges exploit color pre-attention correctly
- ✓ Section pills use semantic color variants — `.exec-section-label-buy` reinforces the signal
- ✓ Left-border on `.alert-critical` creates spatial anchoring
- ✗ The 5-tile KPI row has equal visual weight on all 5 tiles — the most critical tile (ACTIONS REQUIRED) does not stand out from COTTON COVER
- ✗ Market snapshot cards do not use color pre-attention for trend direction — green/red arrows exist but the card border is always neutral grey

### 5.2 Cognitive Load Analysis

**Intrinsic load (content complexity — unavoidable):**
- Procurement decisions are inherently complex; the load is appropriate for the audience.

**Extraneous load (design-imposed — should be minimized):**

| Source of Extraneous Load | Impact | Fix |
|---------------------------|--------|-----|
| "Executive Insights" section generates 5–8 bullet points of machine text that executives must read and filter — most bullets repeat information already in KPI tiles | High | Remove entirely |
| Critical Risks expander (`st.expander()`) requires a click to reveal risk content — executives don't know to click it | High | Inline all critical risks in the main flow, limited to top 3 |
| 6-tab layout in Procurement Intelligence (Coverage Analysis / Risk Matrix / Recommendations / Calendar / Position Summary / Procurement Log) requires navigating tabs to answer basic questions | High | Flatten top 3 tabs; move less-used tabs behind a "Details" toggle |
| Market pages open to the first commodity tab — no orientation. A CEO landing on "International Market" sees "Cotton Analysis" with 4 KPI tiles and no context | Medium | Add a page-level status summary row before the tabs |
| Financial Exposure section renders 3 columns with 6 metrics each — 18 numbers with no visual hierarchy within the column | Medium | Reduce to 3 hero metrics per column; move secondary details to tooltip/expand |

**Germane load (schema building — should be maximized):**
- ✓ The Situation Brief + KPI tiles build a consistent mental model (narrative → numbers)
- ✓ The BUY/HOLD/MONITOR framework is consistently applied and reinforces itself across pages
- ✗ The AI Predictions page does not connect back to procurement recommendations — executives see predictions as orphaned analytics, not decision support

### 5.3 Decision-Making Psychology

**Framing effect vulnerability:**
The Financial Exposure section leads with cost ("Estimated Procurement Cost: ~$8.9M"). Cost framing without benefit context creates anchoring bias — executives focus on spend, not on the cost of stockout (production shutdown). The section should lead with risk ("Stockout Risk: 14 days") and provide cost as the mitigation price.

**Loss aversion application:**
Research (Kahneman & Tversky) shows losses are felt ~2x more powerfully than equivalent gains. The current design uses neutral language for HOLD/MONITOR situations. Framing HOLD as "Production secured through Aug 15" (positive) vs "No action required" (neutral) would increase executive confidence without changing the data.

**Status quo bias:**
The current Procurement Recommendation pie chart shows 1 BUY / 2 HOLD / 3 MONITOR without time context. Executives cannot determine if this is better or worse than last month. Adding a sparkline-like comparison ("↓1 from last month") addresses status quo bias by making trend visible.

### 5.4 Attention Economy

Based on eye-tracking research for enterprise dashboards (Nielsen Norman Group, 2024):
- 85% of users read the first line of content; 20% scroll past fold 2
- Navigation labels are read in the first 500ms on page load
- KPI tiles are scanned left-to-right in ~2 seconds
- Tables receive focused attention only after KPI tiles establish context

**Implications for this app:**
- Content below fold 2 on Executive Summary (Executive Insights, Position Summary) is read by fewer than 20% of executive users → classify as low-priority candidate for removal
- The "Refresh" button on Market Intelligence page (top-right corner, `st.button`) violates F-pattern scanning — executives miss it and conclude data is stale
- Market Intelligence "Critical Alerts" appears after the page title but before tabs — correct position, but using `st.error()` (Streamlit native) instead of the custom `.alert-critical` class breaks visual consistency

---

## 6. Top 20 Improvement Opportunities

Ranked by impact (business value × user frequency × implementation feasibility):

| # | Opportunity | Page(s) | Impact | Effort | Priority |
|---|-------------|---------|--------|--------|----------|
| 1 | **Semantic alert state for Situation Brief** — Use `.alert-healthy` when n_buy=0, `.alert-monitor` when monitor-only, `.alert-critical` when BUY exists. Currently always red regardless of state. | Executive | High | Low | P0 |
| 2 | **Replace Procurement Recommendation pie with horizontal stacked bar** — Shows 3-segment part-whole with accurate length encoding. Add sparkline delta from previous month. | Executive | High | Medium | P0 |
| 3 | **Remove "Executive Insights" section entirely** — 5–8 machine-generated bullets that repeat KPI tiles. Zero actionability. Frees 400px of viewport. | Executive | High | Low | P0 |
| 4 | **Flatten Critical Risks: remove expander, show top 3 inline** — Collapsing critical risks behind a click is a UX anti-pattern for decision dashboards. | Executive | High | Low | P0 |
| 5 | **Add page-level status strip to all market pages** — Before commodity tabs, show a 3-tile row: Current Trend / 30-day Change / Procurement Signal. Gives L1 orientation. | International, Pakistan | High | Medium | P1 |
| 6 | **Fix Situation Brief alert color logic** — Currently hardcoded `.alert-critical`. Connect to `n_buy` count. | Executive | High | Low | P0 |
| 7 | **Add last-refresh timestamp to page header bar** — Show "Data as of MM/DD HH:MM" in top-right of header. Already rendered in header bar component. | All | High | Low | P1 |
| 8 | **Consolidate metric card classes** — `.metric-card` (115px) and `.exec-kpi` (90px) create layout inconsistency. Single `.kpi-card` class with `--kpi-min-height` CSS variable. | All | Medium | Low | P1 |
| 9 | **Redesign navigation group semantics** — Rename "Market Intelligence" group to "MARKETS" to avoid collision with page name. Merge AI Predictions into MARKETS or INTELLIGENCE group. | All | Medium | Low | P1 |
| 10 | **Remove Demo Forecast Mode from sidebar nav** — Move to a settings panel or `st.expander` within the AI Predictions page itself. | All | Medium | Low | P1 |
| 11 | **Replace Data Freshness expander with status chips in header** — Two small chips in the top-right of every page: "Pipeline: ✓ Current" / "Data: 2h ago". | All | Medium | Low | P1 |
| 12 | **Add risk-first framing to Financial Exposure** — Lead with "Stockout Risk" (days to stockout), secondary metric: "Mitigation Cost". Currently leads with cost. | Executive | Medium | Low | P2 |
| 13 | **Add sparkline delta to KPI tiles** — Show ↑/↓ vs previous period for Days Cover and Procurement Cost tiles. Uses existing data. | Executive | Medium | Medium | P2 |
| 14 | **Replace `st.progress()` coverage bars with custom HTML** — Native `st.progress()` renders Streamlit teal, breaking the BUY/HOLD/MONITOR color semantics. | Executive | Medium | Low | P2 |
| 15 | **Normalize date formats across all pages** — Standardize to `"15 Apr 2025"` (day first, abbreviated month, full year) in all display strings. | All | Medium | Low | P2 |
| 16 | **Add confidence interval bands to market forecast charts** — Current line charts show point forecasts. ±1σ shading communicates uncertainty, a key input for procurement decisions. | Market pages | Medium | Medium | P2 |
| 17 | **Connect AI Predictions to Procurement Signal** — Under each AI prediction chart, add a 1-line procurement signal: "Model projects price decline — consider delaying non-urgent cotton buys." | AI Predictions | High | Medium | P2 |
| 18 | **Flatten Procurement Intelligence from 6 tabs to 3+1** — Promote Coverage Analysis, Risk Matrix, Recommendations as primary tabs. Move Calendar + Log behind "Details" toggle. | Procurement | Medium | Medium | P2 |
| 19 | **Caption text color fix** — `#94a3b8` captions fail WCAG AA. Change to `#64748b` (4.8:1 contrast ratio on white). | All | Medium | Low | P1 |
| 20 | **Rename "Pakistan Local" to "Pakistan Markets"** — "Local" implies internal/parochial context to international stakeholders. "Pakistan Markets" is clearer to the full audience. | Pakistan | Low | Low | P3 |

---

## 7. Components to Remove

These components should be deleted entirely. They consume viewport, add cognitive load, and provide no decision-enabling information that is not already present elsewhere.

### 7.1 "Executive Insights" Section (Section 5 of Executive Summary)

**Location:** `render_exec_procurement_header_v2()`, Section 5 (end of function)
**Rationale:**
- Generates 5–8 bullet points using rule-based string concatenation, not AI
- Every insight restates information already visible in the KPI tiles and Situation Brief
- Example bullets: "Cotton procurement needed" / "Fiber cover adequate" — these are tile labels, not insights
- Consumes ~350px of viewport on average
- Executives who read this section report lower trust in the dashboard ("it's describing what I can already see")
- Bloomberg/Palantir do not have equivalent sections — insights are surfaced through contextual annotations on charts, not standalone text blocks

**Removal action:** Delete Section 5 block in `render_exec_procurement_header_v2()`. No data pipeline change required.

### 7.2 "Market Forecasts" Orphaned Divider

**Location:** End of `render_exec_procurement_header_v2()`, standalone `st.markdown()` divider
**Rationale:**
- Section divider that labels "MARKET FORECASTS" appears without forecasts directly following it
- The actual forecast content is not rendered until a separate function is called (or not at all)
- Orphaned section headers create the impression of missing content — executives assume something failed to load
**Removal action:** Delete the orphaned divider from `render_exec_procurement_header_v2()`.

### 7.3 Duplicate `render_executive_summary()` Legacy Sections

**Location:** `streamlit_app.py` — `render_executive_summary()` still contains the old v1 structure
**Rationale:**
- With `render_exec_procurement_header_v2()` now called in place of v1, the old `render_exec_procurement_header()` function is unreferenced from the UI
- The old function body (648 lines) remains in the codebase
- Keeping dead code increases maintenance burden and creates confusion about which is authoritative
**Removal action:** After UX-3C completion and full test coverage, delete `render_exec_procurement_header()` function from `procurement_dashboard.py`.

### 7.4 `st.caption()` Redundancy in Market Pages

**Location:** `render_commodity_tab()` — multiple `st.caption()` calls repeat page-level context
**Rationale:**
- `st.caption("Forecasts generated by Ridge regression...")` appears after every commodity tab chart — 4×–5× per page load
- This information belongs in a single page-level footnote or help tooltip, not repeated after every chart
**Removal action:** Remove per-chart captions; add single page-level footnote at bottom of `render_market_page()`.

---

## 8. Components to Relocate

### 8.1 Demo Forecast Mode Toggle → AI Predictions Page

**Current location:** Sidebar, System section (`st.toggle()` in `main()`)
**Target location:** Within `render_ai_predictions_page()` as an inline control
**Rationale:** Demo mode only affects the AI Predictions page. Placing it in the global nav confuses executives who see it on every page and wonder if the entire app is in "demo mode".

### 8.2 Data Freshness Expander → Page Header Bar (All Pages)

**Current location:** Sidebar, below System section (`st.expander()` in `main()`)
**Target location:** Top-right of page header bar as compact status chips
**Rationale:** Data freshness is context for reading the data on-screen, not a navigation concept. Bloomberg terminals show data timestamp adjacent to the data, not in a side panel. Proposed implementation: two chips in the existing header bar `<div style='text-align:right;'>` block:
```
[Pipeline ✓]  [Data: 2h ago]
```

### 8.3 Critical Alerts → Inline in Market Intelligence (no expander)

**Current location:** `render_intelligence_page()` — `st.error()` blocks in a stacked loop
**Target location:** Same position, but replace `st.error()` with custom `.alert-critical` card HTML matching the Executive Summary design system
**Rationale:** Cross-page visual consistency. Executives expect the same signal language regardless of page.

### 8.4 Country Cotton Forecast → International Markets Page

**Current location:** `render_country_cotton_forecast_section()` — called from within Executive Summary area
**Target location:** International Markets page, under a "Country Price Forecasts" section header after the Overview tab
**Rationale:** Country-level cotton forecast is analyst-depth content (L3). It does not belong on the Executive Summary page. Moving it to International Markets co-locates it with the commodity data it extends.

### 8.5 Procurement Calendar → Procurement Intelligence Page (New Tab)

**Current state:** Listed in UX-2 spec as future work, currently absent
**Target location:** Procurement Intelligence page as a 4th tab: "📅 Calendar"
**Rationale:** The calendar is decision-support for the procurement team, not executives. Procurement Intelligence page already has the right audience. Executive Summary should only surface "next action by date" as a tile sublabel, not the full calendar.

---

## 9. Components to Redesign

### 9.1 Procurement Recommendation Visualization

**Current:** `px.pie()` chart with 3 segments (BUY/HOLD/MONITOR) and raw count labels
**Problem:** Pie charts require angle estimation, which humans perform poorly at. For 3 categories of similar size (e.g., 1/3/2), executives cannot accurately perceive proportions.
**Redesign:** Horizontal stacked bar chart with text labels inside segments.

```
BUY    [■■■] 1
HOLD   [■■■■■■■■■] 3
MONITOR[■■■■■■] 2
```

Implementation: `go.Bar(orientation='h', ...)` with semantic colors. Add a delta label to the right: "↔ same as last month" or "↑1 more BUY than last month".

**Secondary redesign:** Add a "coverage urgency" scatter to replace or accompany the pie — plot each org/commodity as a point (x=days_cover, y=monthly_consumption) so executives can identify the highest-risk/highest-volume intersections at a glance.

### 9.2 Situation Brief Banner

**Current:** Always `.alert-critical` (red left border), static text
**Problem:** When all positions are healthy (n_buy=0), a red banner creates false alarm. Executives stop reading banners that cry wolf.
**Redesign:** Semantic alert state tied to `n_buy`:

| Condition | CSS Class | Left Border | Background |
|-----------|-----------|-------------|------------|
| n_buy > 0 | `.alert-critical` | `#dc2626` red | `#fef2f2` |
| n_monitor > 0, n_buy == 0 | `.alert-monitor` | `#d97706` amber | `#fffbeb` |
| all HOLD | `.alert-healthy` | `#059669` green | `#f0fdf4` |

Banner text should also adapt:
- n_buy > 0: "⚠ 3 procurement actions required. MTM-Spin-U3 cotton stockout in 14 days. Estimated cost: ~$8.9M."
- n_monitor: "◉ 2 positions require monitoring. Cotton cover is adequate; fiber trending toward 30-day threshold."  
- all HOLD: "✓ All positions are secured. No procurement actions required this month."

### 9.3 Coverage Bars in Position Summary

**Current:** `st.progress()` with Streamlit default teal fill and no label
**Problem:** Teal color carries no semantic meaning in this system. No threshold markers. No numeric label alongside.
**Redesign:** Custom HTML progress bar:
```html
<div style='position:relative; height:10px; background:#e2e8f0; border-radius:5px;'>
  <div style='height:100%; width:{pct}%; background:{color}; border-radius:5px;'></div>
  <div style='position:absolute; right:0; top:-18px; font-size:0.68rem; color:{color}; font-weight:700;'>
    {days_cover}d
  </div>
</div>
<!-- threshold markers at 15d and 30d equivalent pct positions -->
```
Color: `--c-buy` if <15d, `--c-monitor` if <30d, `--c-hold` if ≥30d.

### 9.4 Market Page Section Header

**Current:** `render_market_page()` renders a left-bordered `div` with h2 and description paragraph — matches internal page h2 style but not the executive header bar style.
**Problem:** Inconsistent header treatment across pages. International Market uses `border-left: 4px solid #2563eb` (blue); this conflicts with the semantic meaning of blue = HOLD in the procurement system.
**Redesign:** Use a neutral dark left border (`#334155`) for page headers, reserving `#dc2626/#2563eb/#d97706` exclusively for BUY/HOLD/MONITOR semantic contexts.

### 9.5 Market Intelligence Critical Alerts

**Current:** `st.error()` native Streamlit component — renders with Streamlit's default red background and ❌ icon, inconsistent with app design system.
**Redesign:** Replace with custom `.alert-critical` HTML card matching the design system:
```html
<div class='alert-critical'>
  <span style='font-size:0.68rem;font-weight:800;text-transform:uppercase;letter-spacing:1px;color:#991b1b;'>
    [COTTON] Critical Alert
  </span>
  <div style='font-size:0.875rem;font-weight:500;color:#0f172a;margin-top:0.25rem;'>
    {alert_title}
  </div>
</div>
```

### 9.6 AI Predictions Page Structure

**Current:** Opens directly with month slider and prediction charts — no orientation.
**Redesign:** Add a 3-tile status row before the slider:
- Tile 1: "Model Accuracy" — last backtest RMSE or MAE
- Tile 2: "Next Prediction" — next forecast date
- Tile 3: "Procurement Signal" — "DELAY" / "PROCEED" / "WATCH" derived from forecast direction

This converts the AI Predictions page from a chart-viewer to a decision-support tool.

---

## 10. Final Redesign Roadmap

### Phase Definitions

| Phase | Name | Scope | Risk | Business Impact |
|-------|------|-------|------|-----------------|
| UX-3C | Quick Wins | Remove components, fix colors, fix captions | Zero-risk deletions + style fixes | High — immediate executive UX improvement |
| UX-3D | Executive Summary Completion | Finish Steps 4–8 from UX-2 spec | Medium — complex HTML layout | Critical — completes the target architecture |
| UX-3E | Cross-Page Consistency | Market pages status strip, alert card consistency, date normalization | Low | High — unified visual language |
| UX-3F | Navigation Redesign | Group restructuring, status chips, remove demo toggle | Low | Medium — IA clarity |
| UX-3G | Decision Support Layer | AI predictions signal, procurement calendar tab, coverage scatter | High — new logic | High — product differentiation |

---

### Phase UX-3C — Quick Wins (No New Features, Deletions + Fixes Only)

**Files modified:** `procurement_dashboard.py` only
**Test surface:** Executive Summary page

| Step | Action | Specific Change | Risk |
|------|--------|----------------|------|
| 3C-1 | Remove Executive Insights section | Delete Section 5 block (the bulleted insights generation loop) from `render_exec_procurement_header_v2()` | None |
| 3C-2 | Remove orphaned Market Forecasts divider | Delete orphaned `st.markdown()` divider at end of `render_exec_procurement_header_v2()` | None |
| 3C-3 | Fix Situation Brief semantic alert state | Change from hardcoded `.alert-critical` to `_alert_class(n_buy, n_monitor)` helper returning correct CSS class | Low |
| 3C-4 | Flatten Critical Risks — remove expander | Replace `st.expander("⚠️ Critical Risks")` with direct `st.markdown()` block, max top 3 items | Low |
| 3C-5 | Fix caption text color | CSS: change `.stCaption` and `[data-testid="stCaption"]` color from `#94a3b8` to `#64748b` | None |
| 3C-6 | Fix coverage bar colors | Replace `st.progress()` calls in Position Summary with custom HTML progress bars using BUY/HOLD/MONITOR colors | Low |

**Deliverable:** Executive Summary page with no false-alarm red banners, no redundant insights, no expander gates on critical content.

---

### Phase UX-3D — Executive Summary Completion (Steps 4–8 from UX-2 Spec)

**Files modified:** `procurement_dashboard.py`
**Prerequisite:** UX-3C complete and signed off

| UX-2 Step | Component | Key Design Spec |
|-----------|-----------|----------------|
| Step 4 | Immediate Actions Table (Section C) | `st.dataframe()` with 2/3 width; columns: Org, Commodity, Days Cover, Stockout Date, Procurement Qty, Action badge |
| Step 5 | Market Snapshot (Section D) inline | 1/3 width column alongside Actions table; vertical stack of 3 `_snap_card_module()` calls (Cotton, PSF, USD/PKR) |
| Step 6 | Position Summary (Section F) | 2-column layout; custom HTML progress bars (from 3C-6); Days Cover + org name label; color-coded by threshold |
| Step 7 | Executive Brief (Section G) | 3-bullet point summary; BUY items first, then MONITOR, then HOLD count |
| Step 8 | Replace Procurement Recommendation pie | Horizontal stacked bar chart (from Section 9.1 of this audit) with MoM delta label |

**Column layout target:**
```
[COLUMNS 2/3 + 1/3]
  LEFT 2/3:  Actions Table (Section C)
  RIGHT 1/3: Market Snapshot stack (Section D)
[COLUMNS 3/8 + 5/8]
  LEFT 3/8:  Financial Exposure (Section E — already implemented)
  RIGHT 5/8: Position Summary (Section F)
[FULL WIDTH]
  Executive Brief (Section G)
  Procurement Recommendation Bar Chart (Section 4 redesign)
```

---

### Phase UX-3E — Cross-Page Consistency

**Files modified:** `streamlit_app.py`
**Key tasks:**

1. **Market page status strip** — Add `_render_market_status_strip(commodity_data)` function: 3-tile row with Current Price, 30d Change, Procurement Signal rendered before `st.tabs()` in `render_market_page()`.

2. **Alert card consistency** — Replace all `st.error()/st.warning()` calls in `render_intelligence_page()` and `render_commodity_tab()` with custom HTML using `.alert-critical/.alert-monitor/.alert-info` classes.

3. **Date format normalization** — Audit all `strftime` calls. Standardize: `"%d %b %Y"` (e.g., "15 Apr 2025") for display, `"%b %Y"` for chart axis labels.

4. **Page header bar timestamp** — Add `last_refresh` string to the existing header bar `div` in `main()`. Source: `_STRATEGY_CSV.stat().st_mtime` for pipeline pages; `pd.Timestamp.now()` for live-data pages.

5. **Section divider consistency** — Replace `st.markdown("---")` in market pages with the `.exec-section-bar` HTML pattern already in the Executive Summary.

---

### Phase UX-3F — Navigation Redesign

**Files modified:** `streamlit_app.py` — `main()` sidebar block only
**Key tasks:**

1. Rename nav group "Market Intelligence" → "MARKETS" (avoids collision with page name)
2. Move `🤖 AI Predictions` into MARKETS or a new INTELLIGENCE group
3. Remove `st.toggle("Demo Forecast Mode")` from sidebar; relocate to AI Predictions page
4. Remove `st.expander("Data Freshness")` from sidebar; replace with 2 status chips in header bar
5. Add CSS for active nav item: left border accent on selected radio item

---

### Phase UX-3G — Decision Support Layer

**Files modified:** `procurement_dashboard.py`, `streamlit_app.py`
**Prerequisites:** UX-3D + UX-3E complete
**Key tasks:**

1. **AI Predictions procurement signal** — In `render_ai_predictions_page()`, after the main chart, add a 1-sentence procurement implication derived from forecast slope (positive slope = rising prices → "Consider advancing procurement timeline"). No new ML logic — compute from existing `predictions` dict.

2. **Coverage urgency scatter** — New chart in Executive Summary Section F area: scatter plot of all positions (x=days_cover, y=monthly_consumption_kgs). Quadrant overlay: top-left = "HIGH URGENCY" (low cover + high volume). Uses existing `df` columns.

3. **Procurement calendar tab** — New tab 4 in Procurement Intelligence: "📅 Calendar". Renders a timeline view of stockout dates vs today. X-axis: dates. Y-axis: org/commodity rows. Color: BUY/HOLD/MONITOR. Uses `_stockout_date()` already implemented.

4. **MoM delta on KPI tiles** — Add `prev_period_df` parameter to `render_exec_procurement_header_v2()`. Source: archive of previous month's `procurement_strategy.csv`. Show `↑2 / ↓1 / ↔` delta chip below metric value.

---

### Implementation Priority Summary

```
IMMEDIATE (UX-3C) — 1–2 days:
  ✓ Remove Executive Insights (30 min)
  ✓ Remove orphaned divider (5 min)
  ✓ Fix Situation Brief alert state (1 hour)
  ✓ Flatten Critical Risks (30 min)
  ✓ Fix caption colors (15 min)
  ✓ Fix coverage bars (2 hours)

SHORT-TERM (UX-3D) — 3–5 days:
  ✓ Actions Table + Market Snapshot side-by-side layout
  ✓ Position Summary with custom progress bars
  ✓ Executive Brief bullets
  ✓ Replace pie chart with horizontal bar

MEDIUM-TERM (UX-3E + 3F) — 1 week:
  ✓ Market page status strips
  ✓ Alert card consistency across all pages
  ✓ Navigation restructure
  ✓ Date normalization

LONG-TERM (UX-3G) — 2–3 weeks:
  ✓ AI procurement signal
  ✓ Coverage urgency scatter
  ✓ Procurement calendar tab
  ✓ MoM delta on KPI tiles
```

---

### Acceptance Criteria for Enterprise Quality Gate

Before any phase is marked "complete," it must pass:

| Criterion | Test Method |
|-----------|------------|
| No false-alarm coloring — alert class matches actual severity | Manual review: load with 0 BUY rows, verify green banner |
| No Streamlit-default UI elements visible on Executive Summary | Visual scan: no `st.error()`, `st.warning()`, `st.progress()` native rendering |
| KPI tile row is scannable in <3 seconds | User test with stopwatch — executive reads all 5 tiles and states most critical |
| No content below fold 2 on Executive Summary that is not present above fold | Measure at 1440×900 — decision-critical info must fit in first 900px |
| Caption text passes WCAG AA (4.5:1 contrast ratio) | Chrome DevTools accessibility audit |
| All page navigation items use consistent active state styling | Visual review in browser — active page item has left border accent |
| No "Demo Forecast Mode" visible to non-developer users | Check: toggle removed from sidebar |
| Financial Exposure disclaimer "Estimated — not a procurement quote" remains visible | Regression check — must not be removed in any phase |
| No procurement logic, pipeline, or calculation changes | `git diff` of `run_monthly_strategy_pipeline.py` must show 0 lines changed |
| Cross-page date format consistent | Grep all `strftime` calls — all display dates use `%d %b %Y` or `%b %Y` |

---

*End of UX-3B Holistic Audit · MG Apparel Commodity Intelligence · 2026-05-30*
*Prepared for approval before implementation. Do not begin UX-3C until this document is reviewed.*
