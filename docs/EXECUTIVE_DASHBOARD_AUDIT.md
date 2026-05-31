# Executive Dashboard Audit
## MG Apparel — Commodity Intelligence Platform

**Auditor perspective:** McKinsey Digital · Deloitte Analytics · SAP Analytics Cloud · Enterprise SaaS Product Design  
**Audience evaluated for:** CEO · CFO · COO · Head of Procurement · Directors  
**Date:** May 2026

---

## Table of Contents

1. [Visual Design Audit](#part-1--visual-design-audit)
2. [Information Architecture Audit](#part-2--information-architecture-audit)
3. [Executive Summary Audit](#part-3--executive-summary-audit)
4. [Procurement Intelligence Audit](#part-4--procurement-intelligence-audit)
5. [Market Intelligence Audit](#part-5--market-intelligence-audit)
6. [KPI Audit](#part-6--kpi-audit)
7. [Chart Audit](#part-7--chart-audit)
8. [Redesign Blueprint](#part-8--redesign-blueprint)

---

## PART 1 — Visual Design Audit

### 1.1 Overall Assessment

The dashboard has a **thoughtful design foundation** — Inter typography, a consistent CSS design-token system, dark sidebar, card-based KPI layout, semantic color coding (red/blue/amber), and IBM Plex Mono for data values. Compared to default Streamlit, it is meaningfully above baseline.

However, evaluated against SAP Analytics Cloud, Palantir Foundry, Tableau Executive, or a McKinsey Digital build, it is a **polished prototype, not a production enterprise dashboard**. The gaps are structural, not cosmetic.

---

### 1.2 Typography

| Element | Current State | Issue |
|---|---|---|
| Page heading | `1.75rem`, Inter, `font-weight: 700` | Acceptable. Boardroom-grade dashboards use 1.5rem max at heading level. |
| KPI labels | `0.68rem`, uppercase, `letter-spacing: 0.9px` | **Too small.** At 0.68rem on a 1440p display, labels are unreadable to a 55-year-old CFO in a meeting room. Minimum: 0.75rem. |
| KPI values | IBM Plex Mono, `1.8rem`, `font-weight: 800` | Good choice. The monospace font aids numerical scanning. |
| Body text | `0.875rem`, `color: #64748b` | Mid-gray body text reads as muted/secondary. Fine for captions; wrong for decision-critical text. |
| Section dividers | `0.68rem` uppercase pill labels | Barely visible. Section delineation is weak. Executives need clear visual hierarchy between sections. |
| H3 headings via markdown | `1.15rem`, `color: #475569` | **Gray headings.** On a white background, #475569 for headings signals "optional reading" to the eye. Critical section titles must be `#0f172a` (near-black). |
| Emoji in headings | `### 📈 Historical Trend`, `### 🧠 Market Intelligence` | **Non-enterprise.** Emoji in section headings is appropriate for a startup demo, not for a CEO. Swap for icon fonts or SVG icons with consistent weight. |

**Verdict:** Typography system is structurally sound but needs size corrections and elimination of decorative emoji in business section labels.

---

### 1.3 Color Palette

| Token | Hex | Use | Assessment |
|---|---|---|---|
| `--c-buy` | `#dc2626` | BUY / Critical | Correct. Red signals urgency. |
| `--c-hold` | `#2563eb` | HOLD / Adequate | **Questionable.** Blue is the dominant brand color AND the status color for "no action needed." This creates ambiguity — is blue a warning or a normal state? |
| `--c-monitor` | `#d97706` | MONITOR | Correct. Amber signals attention. |
| `--c-healthy` | `#059669` | Positive trend | Correct. |
| Background | `#f8fafc` | App background | Correct. Near-white is professional. |
| Sidebar | `#0f172a` | Dark nav | Correct. Dark sidebar / light content is the standard enterprise pattern. |

**Problems:**
- **Blue overload.** Blue is used for: brand (`#1e40af`), HOLD status (`#2563eb`), page header bar, section borders, BUY action badges background, and chart lines. A CFO scanning the dashboard cannot instantly distinguish "information" blue from "procurement action" blue.
- **Missing neutral state.** There is no visual treatment for "all clear / no action required." When all items are HOLD, the dashboard looks like a blue alert screen. A "healthy/all-clear" green banner is needed.
- **Dark components inside light layout.** The country-price header cards (`#0b1220` background, light text) and the AI Predictions dark theme appear inside a light-background page. This creates jarring mode-switching within the same viewport. Enterprise dashboards maintain one theme throughout.

---

### 1.4 Card Design

| Observation | Severity |
|---|---|
| `.metric-card` has `min-height: 115px` — cards are taller than necessary for the data they hold | Medium |
| 6-column card rows on Executive Summary and Procurement page cause extreme density. At 1280px viewport, each card is ~190px wide — the KPI label truncates or wraps awkwardly. | High |
| Cards use `border: 1.5px solid var(--c-border)` with `box-shadow: 0 1px 3px rgba(0,0,0,0.06)`. The shadow is too subtle; on `#f8fafc` background, cards visually merge with the page surface. | Low |
| Hover state `transform: translateY(-1px)` on KPI cards signals interactivity that does not exist — cards are not clickable. This trains executives to expect drill-down that is absent. | Medium |
| `.metric-value` at `1.8rem` competes with the page title — no clear typographic hierarchy between primary KPI and secondary context. | Medium |

---

### 1.5 Borders and Dividers

| Location | Issue |
|---|---|
| `exec-section-bar` divider line | The colored pill labels ("1 — Procurement Status") are useful for numbered sections. However, they appear too small (0.68rem) and the line after them is `#e2e8f0` — very light. On `#f8fafc` background, the section divider is nearly invisible. |
| Market page commodity header | Dark gradient `#334155 → #475569` banner for each commodity name is heavy and adds no decision value. It is visual chrome that consumes vertical space without communicating anything. |
| International vs Local column headers | Blue gradient header (`#1e40af → #2563eb`) vs green gradient (`#047857`) for column headers works thematically but creates a loud visual. |
| `st.dataframe` table borders | Default Streamlit table styling bleeds through. The override `border-radius: 6px` on `[data-testid="stDataFrame"]` clips but does not fully restyle the table. Column headers retain the Streamlit default light-gray, inconsistent with the custom design system. |

---

### 1.6 Spacing and Layout

| Issue | Location | Severity |
|---|---|---|
| **Vertical density.** Multiple consecutive 6-card rows with no breathing space. Executives need whitespace between decision zones. | Executive Summary, Procurement Overview | High |
| **No grid system discipline.** Cards, charts, and tables are placed in `st.columns()` with varying ratios `[1,2]`, `[2,1]`, `[1,1]`, `6` across sections, creating no consistent visual grid. | Throughout | Medium |
| **Page header bar** is `font-size: 1.15rem` for page title and `0.68rem` for breadcrumb. The contrast is jarring — too much weight difference. | All pages | Low |
| **Caption abuse.** `st.caption()` is used for critical information (USD/PKR rate source, period label, data freshness). Caption text at `0.8rem`, `#94a3b8` is visually dismissed. | Executive Summary | High |
| `margin: 1.5rem 0 1rem 0` on section bars creates inconsistent rhythm — some sections are 1.5rem apart, others 0.75rem. | Throughout | Low |

---

### 1.7 Sidebar

| Element | Assessment |
|---|---|
| Dark background `#0f172a` | Correct enterprise pattern. |
| Brand area | "MG Apparel / Commodity Intelligence" — appropriate. But font size `1.05rem` for "Commodity Intelligence" is too small for branding. Should be `1.25rem`. |
| Nav group labels at `0.6rem` | **Too small.** Barely readable. These are section dividers for the navigation; they need to be at least `0.7rem`. |
| Radio-as-navigation | Streamlit's `st.radio` used as nav links. Radio buttons have a dot selection indicator that is explicitly hidden via CSS (`display: none`). The result is a nav list that works but lacks the visual affordance of a proper nav item (no chevron, no active-state padding). |
| "Demo Forecast Mode" toggle | **Should not be in the sidebar of an executive dashboard.** This is a developer tool. Executives will toggle it accidentally and be confused. Move to an admin/settings page or remove from production build. |
| "Data Freshness" expander | Correct placement in sidebar, but the pattern of `st.success/warning/error` inside an expander is not readable as a status indicator. A single data health badge (green/amber/red) is sufficient for an executive. |
| System section | Appears at the bottom of the sidebar without a visible separator from the navigation. |

---

### 1.8 Tables

| Issue | Location |
|---|---|
| Streamlit's `st.dataframe` renders with a default header height and font that partially overrides custom table styles | All pages |
| Column names like "45-Day Need (Kgs)", "Procurement Qty (Kgs)", "Monthly Consumption (Kgs)" are too long for column headers and cause wrapping | Procurement tabs |
| No frozen header or sticky first column. With 6–8 columns, an executive scrolling right loses context of which org they are viewing | Risk and BUY tabs |
| Color-coded rows (red for <7 days, amber for <15 days) are not explained above the table — only a small caption below. Executives notice visual treatment first; the legend should precede the data. | BUY tab, Risk tab |
| `st.dataframe` scroll chrome (scrollbar track, resize handle) is visually inconsistent with the design system | Throughout |

---

### 1.9 Charts

| Chart | Issue |
|---|---|
| Historical price line chart | `paper_bgcolor='rgba(0,0,0,0)'`, `plot_bgcolor='rgba(0,0,0,0)'`. Transparent backgrounds cause the chart to visually float. On a `#f8fafc` page background, this creates a faint "hanging" effect. Set `plot_bgcolor='#ffffff'` for clarity. |
| Forecast bar chart | Blue bars for downward forecasts and "rising" framing (`↑` in header) sends mixed signals. If a price is rising, that is a risk for a buyer, not a positive signal. Color semantics should be: rising = red/risk, falling = green/opportunity. |
| Action distribution donut (Procurement) | `textinfo="label+value+percent"` on a small donut (280px height) causes label collision. At typical viewport sizes, the labels overlap. |
| Normalized comparison chart (International Market) | `markers + lines` with `marker size=8` on all traces creates visual clutter at 12-month horizon. Remove markers; use line-only with smooth curves. |
| Days Cover bar chart | Correct use of red/amber/blue color coding by threshold. The `add_hline` reference lines at 15 and 45 days are the best visual element in the entire dashboard — clear, actionable, immediately readable. |
| Country-wise cotton price chart | 4px dot reference line and annotation labels at 9px font are illegible at standard zoom. |

---

### 1.10 What Looks Prototype-Like / Non-Enterprise

1. **Emoji as section labels:** `### 📈`, `### 🔮`, `### 💹`, `### 🧠` throughout
2. **`st.caption()` for critical information:** period labels, data source indicators, rate sources
3. **Developer toggle ("Demo Forecast Mode") in the executive sidebar**
4. **`"Run scripts/event_collector.py"` instruction visible to end users** in Market Intelligence page
5. **`st.info()` with code blocks** — `reports/procurement_strategy.csv` path visible to CEO
6. **`📍 Data Type: Market Data`** caption appearing in commodity tabs — internal jargon
7. **`N=14`** appearing on forecast accuracy mini-cards — raw statistical notation for executives
8. **"Confidence: HIGH/MEDIUM/LOW"** without definition of what these mean
9. **`"Reporting period: April 2026 (01-Apr to 30-Apr) | Transaction Type: Direct Organization Transfer"`** — an Oracle transaction type leaking into the executive view
10. **`"As of {latest_date.strftime('%b %Y')}"`** in tooltip — acceptable formatting, but some instances show raw datetime strings
11. **AI Predictions page shows Supabase configuration error** (`st.warning("Supabase is not configured")`) to end users

---

## PART 2 — Information Architecture Audit

### 2.1 Current Navigation Structure

```
Sidebar Navigation:
│
├── Executive
│   └── 📊 Executive Summary
│
├── Market Intelligence
│   ├── 🌍 International Market
│   ├── 🇵🇰 Pakistan Local
│   └── 🧠 Market Intelligence
│
├── Forecasting
│   └── 🤖 AI Predictions
│
└── Procurement
    └── 📦 Procurement Intelligence
```

### 2.2 Page Structure (What Each Page Contains)

**Executive Summary**
- Section 1: Procurement Status (6 KPI cards: Cotton Inv, Fiber Inv, 45-day Cotton Need, 45-day Fiber Need, Total Gap, BUY count)
- Section 2: Critical Risks table (top-10 shortfalls by days cover)
- Section 3: Market Snapshot (ICE Cotton, PSF, USD/PKR)
- Section 4: Procurement Recommendation (BUY/HOLD/MONITOR counts + donut)
- Section 5: Executive Insights (auto-generated bullet list)
- Section 6: Market Forecasts (Cotton, Polyester, Viscose, Natural Gas, Crude Oil — International vs Local — per commodity: chart + table)

**International Market** (tabs per commodity + Overview tab)
- Cotton, Polyester, Viscose, Natural Gas, Crude Oil individual analysis tabs
- Each tab: 4 KPI cards (Price, 6-month Range, Trend, Data Points) + line chart + forecast bar chart + forecast table + Excel export
- Overview tab: normalized comparison chart + commodity range cards

**Pakistan Local** (same structure as International Market)
- Same pattern applied to local market data

**Market Intelligence**
- Critical alerts banner
- Tabs: News by Commodity | Geopolitical | Weather & Supply
- Each tab shows text-based intelligence cards

**AI Predictions**
- Supabase dependency gate
- Month slider
- Per-commodity predicted vs actual line/bar chart
- Forecast accuracy (backtesting) section with MAPE/RMSE tables

**Procurement Intelligence** (6 tabs)
- Overview: 6 KPI cards + donut + summary table
- BUY — Action Required
- HOLD — Adequate Stock
- MONITOR — Attention
- Inventory Risk: days-cover bar charts + shortfall chart + critical alerts
- Full Report: exportable table + CSV download

### 2.3 Architecture Problems

**Problem 1: Executive Summary is not a summary**

The "Executive Summary" page has 6 sections and, beneath the procurement header, renders a full commodity-by-commodity analysis (5 commodities × 2 markets = 10 chart+table pairs). This is not a summary — it is a second copy of the Market page embedded below the procurement header. A CEO opening the Executive Summary page scrolls through 40+ visual elements before reaching the bottom.

**Problem 2: International Market and Pakistan Local are identical in structure**

Both pages use `render_market_page()` with commodity tabs and an Overview tab. They differ only in the data source. From an executive's perspective, the decision-relevant comparison is **International vs Local for the same commodity** — not "all international" on one page and "all local" on another. The split creates navigational overhead.

**Problem 3: Market Intelligence is orphaned**

The Market Intelligence page (geopolitical events, weather alerts, news) is a data-collection tool, not a decision tool. It has no direct link to procurement decisions or price forecasts. An executive visiting this page sees a list of news events with no "so what" — no connection to the BUY/HOLD/MONITOR recommendations. It exists as a standalone analytical module with no downstream integration.

**Problem 4: AI Predictions belongs inside Market Intelligence or is an analyst tool**

Forecast accuracy (MAPE, RMSE, backtesting) and predicted-vs-actual charts are analyst-grade tools. No CEO or CFO will adjust procurement decisions based on MAPE. This page is a model validation tool, not an executive page.

**Problem 5: Procurement Intelligence is the most valuable page but is buried**

For a procurement-driven business, the BUY/HOLD/MONITOR table is the #1 decision output. It is the last item in the navigation. The CEO must scroll past International Market, Pakistan Local, Market Intelligence, and AI Predictions to reach it.

**Problem 6: Navigation items have inconsistent naming register**

- "📊 Executive Summary" — brand voice
- "🌍 International Market" — geographic descriptor
- "🇵🇰 Pakistan Local" — geographic + locale (different naming convention from above)
- "🧠 Market Intelligence" — function descriptor
- "📦 Procurement Intelligence" — function descriptor
- "🤖 AI Predictions" — capability descriptor

No consistent naming convention. Mix of geographic, functional, and capability naming.

---

### 2.4 Recommended Page Structure

```
Sidebar Navigation:

├── COMMAND CENTER
│   └── Dashboard (renamed from "Executive Summary")
│
├── PROCUREMENT
│   ├── Action Plan (BUY/HOLD/MONITOR + risk — renamed from "Procurement Intelligence")
│   └── Purchasing Tracker (purchase history, forecasts)
│
├── MARKET
│   ├── Cotton (International + Local in one page, side-by-side)
│   ├── Fibers (PSF, Viscose, Stretch Fiber)
│   └── Energy & FX (Natural Gas, Crude Oil, USD/PKR)
│
└── INTELLIGENCE (analyst-only, collapsed by default)
    ├── Market Signals (renamed from "Market Intelligence")
    ├── Forecasting Models (renamed from "AI Predictions")
    └── Data Health (pipeline status, freshness)
```

### 2.5 Merge / Remove / Move Recommendations

| Current Page | Recommendation | Reason |
|---|---|---|
| International Market | **Merge** with Pakistan Local into commodity-specific pages (Cotton, Fibers, Energy) | Decision value is in the comparison, not the separation |
| Pakistan Local | **Merge** (see above) | |
| AI Predictions | **Move** to Intelligence section, collapse by default | Analyst tool, not exec tool |
| Market Intelligence | **Move** to Intelligence section | Raw news feed without decision context |
| "Demo Forecast Mode" toggle | **Remove** from production sidebar | Developer artifact |
| Market Forecasts section (bottom of Executive Summary) | **Remove** from Executive Summary | Creates 10-screen scroll; move to individual commodity pages |
| "Data Points" KPI card | **Remove** from commodity tabs | "1,847 observations" is irrelevant to procurement decisions |
| Excel export button in every commodity tab | **Move** to Intelligence > Data Health | Analyst function, not executive function |

---

## PART 3 — Executive Summary Audit

### 3.1 CEO 30-Second Test

**Can a CEO opening this page for 30 seconds understand:**

| Question | Current Answer | Assessment |
|---|---|---|
| **Procurement Risk** — who needs to buy what now? | **Partial.** Section 2 (Critical Risks) table shows shortfalls. But the table is the 7th–20th visual element on the page. A CEO will not scroll to it in 30 seconds. | ❌ Not immediately visible |
| **Inventory Position** — how much stock do we have? | **Partial.** Section 1 KPIs show Cotton Inventory and Fiber Inventory in Kgs. No context (days, weeks, percentage of need). A bare number "1,096,907 Kgs" tells a CEO nothing without the consumption rate alongside it. | ❌ Requires interpretation |
| **Financial Exposure** — what is the cost implication? | **No.** There is no financial value anywhere on the dashboard. No "estimated procurement cost", no "cost of delay", no "budget vs actual". The dashboard operates entirely in kilograms and days, with zero financial framing. | ❌ Absent |
| **Market Outlook** — are prices going up or down? | **Partial.** Section 3 (Market Snapshot) shows ICE Cotton price and MoM change. But 3 KPI cards buried in Section 3 of 6 sections is not prominent enough. | ❌ Not immediately visible |
| **Recommended Action** — what should we do? | **Partial.** Section 4 (Procurement Recommendation) shows BUY:10, HOLD:2, MONITOR:7 counts. A donut chart of action distribution. But "10 BUY" is not an action — it is a count. The executive needs to know: which commodities, which units, and by when. | ❌ Insufficient specificity |

**Overall: A CEO cannot extract all five answers in 30 seconds from the current Executive Summary.**

The primary failure is structure: the most critical information (procurement actions, risk positions) is separated across 5 numbered sections with significant visual elements between them. The secondary failure is financial absence: there is no monetary framing.

---

### 3.2 Why the Current Design Fails for the CEO

1. **The page does not open with a verdict.** In every McKinsey or Palantir executive dashboard, the first thing an executive sees is the current status in one sentence: "3 URGENT actions required · 2 units below 15-day cover · ICE Cotton up 4.2% MoM." The current page opens with a period badge and a 6-card KPI row that requires the executive to mentally compute the situation.

2. **Financial exposure is completely absent.** A CFO asks: "What is this costing us?" and "What does it cost to delay?" The dashboard has no monetary dimension. If 10 org-commodity pairs require procurement, what is the total estimated purchase value? What is the cost per kg of shortfall by commodity? This is a critical missing layer.

3. **The "Executive Insights" bullets are the only narrative.** 5–6 auto-generated sentences near the bottom of the procurement section are the closest thing to an executive brief. But they are:
   - Below 4 other sections (must scroll to see them)
   - Font size 0.88rem — smallest text used for decision-critical content
   - Presented as a plain bullet list (no visual priority between bullets)
   - Not connected to any action button or drill-down

4. **The page has no information scent.** Information scent refers to the visual cues that tell a reader what they will find if they keep scrolling. After Section 5 (Executive Insights), there is a section bar labeled "Market Forecasts — Commodity-by-Commodity Analysis" followed by 10+ chart/table pairs. An executive does not know this is coming. The page structure is not communicated upfront.

5. **The Market Forecasts section (bottom of Executive Summary) contradicts the "summary" promise.** Scrolling through Cotton International vs Local, Polyester International vs Local, Viscose International vs Local, Natural Gas, and Crude Oil is a 5–8 minute analytical exercise. This belongs on the Market pages, not the Executive Summary.

---

### 3.3 Conceptual Redesign of Executive Summary

The redesigned Executive Summary has one goal: **give the CEO a complete situational brief in 30 seconds, with one-click drill-down for each topic.**

```
┌─────────────────────────────────────────────────────────────────────┐
│  SITUATION BRIEF — April 2026                    Updated: 30 May 2026 │
│                                                                       │
│  "3 units require immediate cotton procurement. ICE Cotton is up      │
│   4.2% MoM. Estimated procurement value: $2.1M. Lowest cover:        │
│   MTM-Spinning U3 / Cotton — 18.4 days."                             │
└─────────────────────────────────────────────────────────────────────┘

ROW 1: STATUS AT A GLANCE (5 large tiles)
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│  10 BUY  │  2 HOLD  │  7 MON   │ 18.4 days│  $2.1M   │
│ URGENT   │ ADEQUATE │ WATCH    │ Lowest   │ Est. Buy │
│ [red]    │ [blue]   │ [amber]  │ Cover    │ Value    │
└──────────┴──────────┴──────────┴──────────┴──────────┘

ROW 2: CRITICAL RISKS (always visible, top 5 only)
┌─────────────────────────────────────────────────────────────────────┐
│  CRITICAL RISKS          [View all 10 →]                             │
│                                                                       │
│  MTM Spinning U3  │ Cotton    │ 18.4 days │ Gap: 146,000 Kgs │ BUY  │
│  MTM Spinning U1  │ Cotton    │ 22.1 days │ Gap: 235,000 Kgs │ BUY  │
│  MSM Spinning U1  │ Fiber     │ 27.8 days │ Gap: 267,000 Kgs │ BUY  │
│  ...              │           │           │                  │      │
└─────────────────────────────────────────────────────────────────────┘

ROW 3: MARKET CONDITIONS (3 tiles, spark lines)
┌──────────────────┬──────────────────┬──────────────────┐
│ ICE COTTON       │ PSF              │ USD/PKR          │
│ 0.6782 USD/lb    │ 0.9200 USD/kg    │ 277.50           │
│ ↑ 4.2% MoM      │ → 0.3% MoM       │ ↓ 1.1% MoM       │
│ [spark line]     │ [spark line]     │ [spark line]     │
└──────────────────┴──────────────────┴──────────────────┘

ROW 4: FINANCIAL EXPOSURE (2 tiles)
┌──────────────────────────────┬──────────────────────────────┐
│ ESTIMATED PROCUREMENT COST   │ INVENTORY AT RISK             │
│ ~$2.1M (10 BUY pairs)        │ 3 units below 30-day cover    │
│ Based on current ICE rates   │ Approx. $1.4M replacement value│
└──────────────────────────────┴──────────────────────────────┘
```

**What is removed from the current Executive Summary:**
- The commodity-by-commodity forecast charts (10 chart+table pairs) — move to Market pages
- The numbered "section bars" with elaborate dividers
- `st.caption()` for anything decision-critical

**What is added:**
- A single one-sentence situation brief at the top
- Estimated financial exposure (procurement cost in USD, using ICE spot × shortfall qty)
- "View all" links instead of showing full tables inline

---

## PART 4 — Procurement Intelligence Audit

### 4.1 Does the Current Page Drive Action?

The Procurement Intelligence page is the most technically complete section of the dashboard. It has:
- 6 tabs (Overview, BUY, HOLD, MONITOR, Inventory Risk, Full Report)
- BUY tab with shortfall bar chart and color-coded risk table
- Days-cover bar charts with threshold reference lines
- Exportable CSV

**However, it does not drive action effectively.** Here is why:

| Gap | Explanation |
|---|---|
| **No financial dimension** | BUY tabs show shortfalls in Kgs only. An executive approving a purchase order needs the estimated value. "Procurement Qty: 267,000 Kgs" means nothing without a price context. |
| **No deadline / urgency signal** | 18.4 days of cover → when does that run out? There is no "stockout date" shown. Executives respond to dates, not abstract numbers. "MTM Spinning U3 runs out of cotton on June 18, 2026" is actionable. "18.4 days cover" requires mental arithmetic. |
| **No supplier / lead time context** | The procurement recommendation provides no context on typical lead times. If lead time is 30 days and cover is 18 days, the shortfall is already overdue. This dimension is absent. |
| **Tab structure buries the action** | An executive looking for what to do must click "BUY — Action Required" tab. The most important information is a secondary tab, not the default view. The default "Overview" tab leads with a donut chart and a summary table. A donut chart does not drive action. |
| **MONITOR explanation is technical** | "Missing Consumption Data" and "Net Consumption <= 0" are data engineering terms. An executive sees MONITOR and needs a clear explanation: "We cannot calculate coverage for this unit because Oracle data is not yet available." |
| **6 tabs is too many for an executive** | Overview, BUY, HOLD, MONITOR, Inventory Risk, Full Report — a CFO opening this page does not know which tab to look at. The action-required items should be the first and default view. |
| **BUY action items are not sorted by financial impact** | Sorted by days cover (ascending). More useful sort: by shortfall value (estimated cost). The most expensive urgent action should lead. |

---

### 4.2 Recommended Procurement Intelligence Redesign

**Concept: Action-First Layout**

```
┌─────────────────────────────────────────────────────────────────────┐
│  PROCUREMENT ACTION PLAN — April 2026                                │
│  Period: 01 Apr to 30 Apr · Pipeline run: 30 May 2026 14:44         │
└─────────────────────────────────────────────────────────────────────┘

SECTION 1: IMMEDIATE ACTIONS REQUIRED  (always first, always visible)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
10 procurement actions required this period

  ORG              COMMODITY  COVER    STOCKOUT    GAP (KGS)  EST. COST
  MTM Spinning U3  Cotton     18.4d   June 18     146,000    ~$99K
  MTM Spinning U1  Cotton     22.1d   June 22     235,000    ~$160K
  MSM Spinning U1  Fiber      27.8d   June 27     267,000    ~$246K
  ...

  [Download Purchase Order Template →]

SECTION 2: INVENTORY POSITION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Days Cover chart — all orgs, all commodities, with 15/45 day lines]

SECTION 3: NO ACTION REQUIRED (HOLD)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2 pairs adequately stocked (collapsed by default)

SECTION 4: MONITORING (DATA PENDING)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7 pairs — consumption data not yet available for this period
```

**Key changes from current:**
- BUY items are the default first view — no tab click required
- Stockout date is calculated and displayed prominently (coverage days + today's date)
- Estimated cost column uses current ICE/market price × shortfall quantity
- HOLD and MONITOR are collapsed sections, not tabs — they are less urgent
- One tab removed (Full Report merged into section 4 with download button)

---

## PART 5 — Market Intelligence Audit

### 5.1 What Executives Care About vs What Is Noise

**What executives care about (signal):**

| Information | Why it matters |
|---|---|
| ICE Cotton No. 2 spot price and MoM direction | Procurement cost benchmark |
| USD/PKR exchange rate direction | All international purchases are USD-denominated; PKR cost impact is direct |
| Whether to accelerate or defer procurement (buy ahead vs wait) | Only actionable question from market data |
| Country-of-origin price benchmarks (Sudan, CCA, USA) | Supplier negotiation reference |
| PSF / Viscose price direction | Substitute fiber procurement cost |

**What is noise for executives (should be analyst-only):**

| Information | Why it is noise |
|---|---|
| Normalized comparison chart (all commodities indexed to 100) | Analytical tool; no procurement decision depends on the normalized index |
| 6-month price range (min–max) | Interesting context but not actionable |
| "Data Points: 1,847 observations" KPI card | Completely irrelevant to procurement decisions |
| Historical volatility | Useful for options pricing; not an executive procurement concern |
| Individual commodity news articles (raw text) | Information overload; executives need curated summaries, not raw feeds |
| "Weather & Supply Chain" tab with empty state | Empty page in current state; should be hidden when no data |
| Geopolitical events tab with card layout | No connection to procurement recommendation |
| Forecast accuracy / MAPE / RMSE (AI Predictions page) | Model validation; analyst tool |
| Excel export in every commodity tab | Analyst workflow, not executive |
| "📅 Data Type: Market Data" / "📊 Futures data · March 2026 contract pricing" | Internal data engineering labels bleeding through |

### 5.2 Market Intelligence Redesign Recommendation

The Market Intelligence section should present one page per commodity class with:

1. **Price position:** current vs 3-month vs 12-month average (is this historically expensive or cheap?)
2. **Directional signal:** up / flat / down over the last 3 months with magnitude
3. **Procurement implication:** at this price level, does the procurement engine recommend front-loading or waiting?
4. **Country comparison:** for Cotton, which origin is currently cheapest per standard quality grade?

Everything else — normalized indices, backtesting, raw news feeds, data points counts — belongs in an analyst workspace, not the executive dashboard.

---

## PART 6 — KPI Audit

### 6.1 KPI Classification

| KPI | Location | Classification | Rationale |
|---|---|---|---|
| Cotton Inventory (Kgs) | Exec Summary, Procurement Overview | **Critical** | Core operational metric |
| Fiber Inventory (Kgs) | Exec Summary, Procurement Overview | **Critical** | Core operational metric |
| 45-Day Cotton Need (Kgs) | Exec Summary, Procurement Overview | **Critical** | Policy stock threshold anchor |
| 45-Day Fiber Need (Kgs) | Exec Summary, Procurement Overview | **Critical** | Policy stock threshold anchor |
| Total Procurement Gap (Kgs) | Exec Summary, Procurement Overview | **Critical** | Top-line action indicator |
| BUY Recommendations (count) | Exec Summary, Procurement Overview | **Critical** | Immediate action count |
| Days Cover (by org/commodity) | Procurement Risk chart | **Critical** | Most actionable single metric |
| ICE Cotton No. 2 price | Market Snapshot | **Critical** | Procurement cost benchmark |
| USD/PKR rate | Market Snapshot | **Critical** | All-in cost impact |
| PSF price | Market Snapshot | **Useful** | Substitute fiber cost reference |
| HOLD count | Procurement Overview | **Useful** | Confirmation of adequacy |
| MONITOR count | Procurement Overview | **Useful** | Data quality indicator |
| Avg Days Cover (fleet) | Procurement Overview | **Useful** | Portfolio health indicator |
| Critical (<15 days) count | Procurement Overview | **Critical** | Urgency signal |
| Total Procurement Kgs (BUY sum) | Procurement Overview | **Useful** | Volume planning |
| 6-Month Price Range (min-max) | Commodity tabs | **Redundant** | Subsumed by full price chart |
| Data Points count | Commodity tabs | **Remove** | Irrelevant to procurement |
| Viscose price | Market Snapshot | **Useful** (if viscose is purchased) | Conditional on product mix |
| Natural Gas price | Local page | **Useful** | Operating cost context |
| Crude Oil price | Local page | **Useful** | Textile input cost context |
| Cotton (Local) PKR/maund | Local page | **Useful** | Local procurement benchmark |
| MAPE / RMSE | AI Predictions | **Remove** from exec view | Analyst metric only |
| "Positive Qty Count / Negative Qty Count" | Diagnostics | **Remove** from exec view | Pipeline diagnostic only |
| Confidence (HIGH/MEDIUM/LOW) | Procurement tables | **Useful** but needs definition | Currently unexplained |
| Estimated Financial Exposure (USD) | **Absent** | **Critical** — must add | Without cost, no budget decision |
| Stockout Date | **Absent** | **Critical** — must add | Days cover without a date is abstract |

---

### 6.2 KPI Sets by Role

**CEO KPI Set (visible in 30 seconds, no navigation)**

| # | KPI | Format | Priority |
|---|---|---|---|
| 1 | BUY actions required | Large number, red | P1 |
| 2 | Lowest days cover (worst position) | Number + org name | P1 |
| 3 | Estimated total procurement cost (USD) | Dollar amount | P1 |
| 4 | ICE Cotton spot + MoM direction | Price + arrow + % | P2 |
| 5 | Cotton inventory (days of consumption) | Days, not Kgs | P2 |
| 6 | Fiber inventory (days of consumption) | Days, not Kgs | P2 |

**CFO KPI Set**

| # | KPI | Format |
|---|---|---|
| 1 | Total estimated procurement cost (USD) | Dollar amount |
| 2 | Budget vs actual procurement spend (if tracked) | Variance % |
| 3 | ICE Cotton vs 12-month average (are we buying high or low?) | % above/below average |
| 4 | USD/PKR rate + trend | Rate + direction |
| 5 | Cost of delay (estimated daily cost of not procuring) | Dollar per day |
| 6 | Total inventory value at cost | Dollar amount |

**Procurement Director KPI Set**

| # | KPI | Format |
|---|---|---|
| 1 | BUY count by commodity (Cotton, Fiber, Stretch Fiber) | Per-commodity counts |
| 2 | Days cover — full matrix view (all orgs × all commodities) | Color-coded grid |
| 3 | Stockout dates for all BUY items | Date column |
| 4 | Shortfall quantity by org and commodity | Sorted by urgency |
| 5 | ICE Cotton + PSF + Viscose current vs forward price | Price table |
| 6 | Country-wise cotton benchmark (Sudan, CCA, Brazil, USA) | Price comparison table |
| 7 | MONITOR items and reason | List with data quality context |

---

## PART 7 — Chart Audit

### 7.1 Chart-by-Chart Assessment

**Historical Price Line Charts (commodity tabs)**
- Verdict: **REDESIGN**
- Current: transparent background, lines+markers, 12-month window
- Issue: Markers at every data point create visual noise on monthly data. No price annotations. No context lines (e.g., 12-month average line).
- Redesign: Clean line, no markers, add 3-month and 12-month average reference lines, area fill below line for visual weight, explicit annotation of current price and YoY change.

**Forecast Bar Charts (commodity tabs)**
- Verdict: **REPLACE**
- Current: bar chart of 1M/3M/6M/12M predictions. Rising price shown in blue bars with upward text. Color does not encode direction — all bars are the same color.
- Issue: A rising commodity price is a risk signal for a buyer, not a neutral data point. The visual framing should encode direction as risk: rising = red, falling = green.
- Replacement: Grouped bar chart showing current price vs forecast with delta coloring. Include confidence interval as error bars.

**Normalized Comparison Chart (Overview tab)**
- Verdict: **KEEP for analysts, REMOVE from executive path**
- Current: Multi-line chart, all commodities indexed to 100. Useful for analysts to see relative performance.
- Issue: Executive cannot extract a procurement decision from this chart.
- Decision: Move to Intelligence section for analyst use.

**Action Distribution Donut (Procurement Overview and Executive Summary)**
- Verdict: **REPLACE**
- Current: Donut showing BUY/HOLD/MONITOR proportions.
- Issue: A donut chart of 3 categories provides almost no information value over three numbers. "10 BUY, 2 HOLD, 7 MONITOR" communicates more in 3 numbers than a donut. If a chart is used, a stacked horizontal bar showing urgency levels is more space-efficient and readable in small sizes.
- Replacement: Stacked horizontal bar by commodity class (Cotton BUY/HOLD, Fiber BUY/HOLD, etc.) — this adds the commodity dimension missing from the current donut.

**Days Cover Bar Charts by Commodity / by Organisation (Risk tab)**
- Verdict: **KEEP — best chart in the dashboard**
- Current: Bar chart with color threshold (red <15, amber <30, blue ≥45), horizontal reference lines at 15 and 45 days.
- This chart is immediately readable, color-encodes urgency correctly, and the reference lines provide instant context. It requires no explanation.
- Improvement: Add org labels to the commodity chart. Current version shows "Cotton avg: 28.3 days" — useful, but the org-level breakdown in the second chart is actually more decision-relevant.

**Shortfall Horizontal Bar Chart (BUY tab)**
- Verdict: **KEEP with modifications**
- Current: Horizontal bar showing shortfall Kgs per org/commodity. Red color, values labeled outside bars.
- Improvement: Add a secondary axis or label showing estimated cost alongside Kgs. Sort by estimated cost, not raw Kgs.

**Country-wise Cotton Price Chart (Executive Summary, Cotton section)**
- Verdict: **REDESIGN**
- Current: Multi-series price chart with 4px dot reference lines and 9px annotation text.
- Issue: Too small. Country names and price annotations are illegible at standard zoom.
- Redesign: Table-based layout with sparklines for trend per country. Much more readable at executive scale.

**Predicted vs Actual Chart (AI Predictions)**
- Verdict: **KEEP in analyst section**
- The predicted (line) vs actual (bars) overlay is correct for model validation. Should not appear in executive path.

**Forecast Accuracy Table (MAPE, RMSE, MAE)**
- Verdict: **KEEP in analyst section**
- Correctly formatted statistical output. Wrong page for executives.

---

## PART 8 — Redesign Blueprint

### Phase 1 — Quick Wins (1–2 weeks, no structural changes)

**Impact: High · Effort: Low**

These are changes that make the dashboard significantly more executive-grade without restructuring pages or adding features.

| Item | Change | Impact |
|---|---|---|
| Q1 | Remove all emoji from section headings — replace with text-only or a consistent icon library | High |
| Q2 | Increase KPI label size from `0.68rem` to `0.75rem` | High |
| Q3 | Move "Demo Forecast Mode" toggle out of the executive sidebar | High |
| Q4 | Replace `st.caption()` with `st.markdown()` for all decision-critical labels (period dates, rate sources, data freshness) | High |
| Q5 | Add estimated stockout date to BUY tab table: `stockout_date = today + days_cover` | High |
| Q6 | Change "Days Cover" display everywhere from a number to a number + date: "18.4 days (Jun 18)" | High |
| Q7 | Remove "Data Points: 1,847" KPI card from commodity tabs — replace with data currency indicator (e.g., "Current as of Apr 2026") | Medium |
| Q8 | Remove "📍 Data Type: Market Data" and Oracle transaction type labels from all user-facing text | Medium |
| Q9 | Remove the `st.info()` / `st.warning()` blocks that show file paths and code snippets to end users | High |
| Q10 | Make BUY tab the default (first/selected) tab in Procurement Intelligence instead of Overview | High |
| Q11 | Change trend colors in price forecast: rising = red/warning, falling = green/opportunity (not neutral blue) | Medium |
| Q12 | Replace the donut chart on Executive Summary with three large horizontal stat blocks (BUY / HOLD / MONITOR) | Medium |
| Q13 | Add a single bold "Situation Brief" sentence at the very top of Executive Summary (auto-generated, same logic as `_insight_bullets` but condensed to one sentence) | High |

---

### Phase 2 — Professional SaaS Upgrade (2–6 weeks, moderate restructuring)

**Impact: Very High · Effort: Medium**

| Item | Change | Impact |
|---|---|---|
| P1 | **Restructure navigation** per the recommended architecture (Command Center, Procurement, Market, Intelligence) | Very High |
| P2 | **Remove Market Forecasts from Executive Summary** — strip the 10-chart commodity section from the bottom of the exec page | Very High |
| P3 | **Add financial layer** — calculate estimated procurement cost (shortfall_qty × commodity spot price / FX rate) and display as USD throughout | Very High |
| P4 | **Merge International and Local Market** into commodity-class pages (Cotton, Fibers, Energy & FX) with side-by-side comparison | High |
| P5 | **Redesign KPI tiles** — convert Kgs-only tiles to dual display: Kgs + days-of-consumption equivalent. "1,096,907 Kgs (≈ 56 days)" is infinitely more readable than bare Kgs. | Very High |
| P6 | **Redesign the Executive Summary layout** per the concept in Part 3 — Situation Brief → Status Tiles → Critical Risks → Market → Financial Exposure | Very High |
| P7 | **Days Cover matrix view** — a heatmap grid (orgs × commodities) color-coded by coverage level. One glance shows the full inventory position. | High |
| P8 | **Replace forecast bar charts** with risk-framed visualization: current price vs 3M forward, colored by direction as risk | High |
| P9 | **Streamline Procurement Intelligence** from 6 tabs to 3 sections (Actions Required, Position, Monitoring) | High |
| P10 | **Add a "Last Pipeline Run" status indicator** prominently — executives need to know if the data is from today or last month | High |

---

### Phase 3 — Boardroom / Executive-Grade Dashboard (6–12 weeks, full build)

**Impact: Transformative · Effort: High**

This phase produces a dashboard that a CEO could project in a board meeting or a CFO could show in a risk committee.

| Item | Change | Impact |
|---|---|---|
| B1 | **Single-screen Command Center** — all critical information visible without any scrolling on a 1440p display. All five CEO questions answered in one viewport. | Transformative |
| B2 | **Financial exposure module** — procurement cost tracker, budget vs. forecast, estimated cost of delay per day. Requires commodity price integration with shortfall quantities. | Transformative |
| B3 | **Procurement calendar view** — show stockout dates on a calendar: "MTM U3 Cotton: Jun 18 · MSM U1 Fiber: Jun 27 · ..." Executive sees timeline, not abstract days. | Transformative |
| B4 | **Replace Streamlit tables with custom HTML/CSS data grids** — full control over typography, sticky headers, custom column widths, and drill-down links | High |
| B5 | **Country origin cost intelligence** — integrated country benchmark table showing current landed cost (CIF Karachi) per kg by origin, updated monthly | High |
| B6 | **Mobile-responsive layout** — executive reads on phone before a meeting. Current 6-column layouts are unusable on mobile. | High |
| B7 | **Print / PDF export** — one-click export of the Executive Summary as a boardroom-ready PDF. Format: A4, clean typography, no interactive elements. | High |
| B8 | **Unified notification system** — when the pipeline detects a BUY item below 15-day cover, an email/SMS alert fires before the executive even opens the dashboard | High |
| B9 | **Role-based views** — CEO view (5 KPIs + situation brief), CFO view (financial framing), Procurement Director view (full operational detail). Single codebase, role parameter in sidebar or URL. | Medium |
| B10 | **Consistent dark mode variant** — the AI Predictions page has a dark theme (`#0b1220`). A coherent dark mode for the entire dashboard would be valuable for board presentations | Medium |

---

### Phase Priority Matrix

| Phase | Impact | Effort | Time | Recommended Next |
|---|---|---|---|---|
| **Phase 1 — Quick Wins** | High | Low | 1–2 weeks | **Start immediately** |
| **Phase 2 — SaaS Upgrade** | Very High | Medium | 2–6 weeks | After Phase 1 complete |
| **Phase 3 — Boardroom Grade** | Transformative | High | 6–12 weeks | After Phase 2 validated |

**Single highest-impact change across all phases:**  
Add estimated financial exposure (procurement cost in USD) to the Executive Summary and Procurement Intelligence pages. This transforms the dashboard from a logistics tool into a financial decision tool. A CEO cannot approve procurement without knowing the cost.

**Single quickest win:**  
Move "Demo Forecast Mode" out of the sidebar and remove all `st.info()` blocks that expose file paths and pipeline commands to executive users. This takes 30 minutes and immediately removes the most obvious "prototype" signals.

---

*End of Audit*
