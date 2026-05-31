# UX-3B.1 — Enterprise Design System Specification
## MG Apparel · Commodity Intelligence Platform
**Version:** 1.0 — Authoritative Standard  
**Date:** 2026-05-30  
**Status:** APPROVED FOR IMPLEMENTATION  
**Reference Level:** Bloomberg Terminal · Palantir Foundry · SAP Analytics Cloud · Microsoft Fabric

---

## EXECUTIVE VERDICT

This product currently reads as a well-intentioned Streamlit application built by a developer with good instincts but without a design system. The foundation is surprisingly strong — the semantic tricolor (BUY/HOLD/MONITOR), IBM Plex Mono for numerics, and the dark sidebar are all correct professional choices. But the surface layer undermines them.

**The ten signals that immediately betray amateur origin:**

1. Emoji in metric labels, tab titles, chart headers, and page headings (`📊 CURRENT PRICE`, `🎯 Procurement Signals Summary`, `📐 Forecast Accuracy`) — enterprise tools do not decorate data with icons
2. `st.success()` / `st.warning()` / `st.error()` / `st.info()` Streamlit native components visible in the commodity tab and market intelligence pages
3. Two pie charts on separate pages — pie charts signal "I visualized this in 5 minutes"
4. `linear-gradient(135deg, ...)` on the `.recommendation` banner — gradient fills are a consumer app pattern
5. Six KPI tiles in a `st.columns(6)` grid on Procurement Overview — developer data dump
6. Nav group labels at `0.6rem` — sub-accessible, sub-professional
7. `st.caption()` repeated after every single chart (5–8 times per page) — tooltip spam
8. `st.subheader()` native rendered in `render_executive_signals_table()` — breaks CSS override
9. Purple `#7c3aed` applied as a 5th semantic color with no defined meaning — color system leak
10. `border-left: 4px solid [color]` on section headers, page headers, and market tiles all using the same pattern — overuse creates visual monotony

The fix is not a visual redesign. It is a design system enforcement pass: every page must use the same token vocabulary, every component must follow the same rules, every native Streamlit widget must be replaced with custom HTML where appearance matters.

---

## SECTION A — TYPOGRAPHY SYSTEM

### A.1 Font Stack

| Role | Family | Fallback |
|------|--------|----------|
| **UI / Body** | `'Inter'` | `-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif` |
| **Numeric / Data** | `'IBM Plex Mono'` | `'Roboto Mono', 'Courier New', monospace` |
| **Code / System** | `'Roboto Mono'` | `'Courier New', monospace` |

**Google Fonts import (already in app — keep):**
```
Inter: 400, 500, 600, 700, 800
IBM Plex Mono: 500, 600
```

---

### A.2 Type Scale — Complete Specification

#### H1 — Page Title (used ONCE per page, in header bar only)
```
font-family:   Inter
font-size:     1.125rem        (18px)
font-weight:   800
letter-spacing: -0.3px
line-height:   1.2
color:         #0f172a
usage:         Page header bar only. NOT used inside page content.
```

#### H2 — Section Title (used for major content sections)
```
font-family:   Inter
font-size:     0.875rem        (14px)
font-weight:   700
letter-spacing: 0px
line-height:   1.3
color:         #1e293b
usage:         Section divider labels when inline prose heading is needed.
               DO NOT use st.markdown("## ...") — use .section-title class.
```

#### H3 — Sub-section / Chart Title
```
font-family:   Inter
font-size:     0.8125rem       (13px)
font-weight:   600
letter-spacing: 0px
line-height:   1.3
color:         #334155
usage:         Chart titles, table titles, subsection labels within a section.
               NO emoji. No icon prefix.
```

#### Section Label (pill dividers, `.exec-section-bar`)
```
font-family:   Inter
font-size:     0.625rem        (10px)
font-weight:   800
letter-spacing: 1.4px
line-height:   1
color:         #64748b (default) | semantic color when status-colored
text-transform: uppercase
usage:         Horizontal rule dividers between page sections.
               Max 3 words. No emoji. No punctuation.
```

#### Card Title / Metric Label
```
font-family:   Inter
font-size:     0.625rem        (10px)
font-weight:   700
letter-spacing: 1.0px
line-height:   1
color:         #64748b (default) | semantic color when card is status-colored
text-transform: uppercase
usage:         Top line of every KPI card. 1–3 words max. NO emoji.
               Example: "ACTIONS REQUIRED" not "📊 ACTIONS REQUIRED"
```

#### Metric Value (hero number in KPI card)
```
font-family:   IBM Plex Mono
font-size:     clamp(1.375rem, 1.6vw, 1.875rem)    (22–30px responsive)
font-weight:   600
letter-spacing: -0.5px
line-height:   1
color:         #0f172a
usage:         The single numeric value in a KPI card.
               Tabular numbers — always IBM Plex Mono.
```

#### Metric Unit / Sublabel
```
font-family:   Inter
font-size:     0.6875rem       (11px)
font-weight:   500
letter-spacing: 0px
line-height:   1.5
color:         #64748b
usage:         Currency, unit, date range, or context below metric value.
               Example: "USD/lb · As of Apr 2025"
```

#### Table Header
```
font-family:   Inter
font-size:     0.6875rem       (11px)
font-weight:   700
letter-spacing: 0.8px
line-height:   1
color:         #e2e8f0
text-transform: uppercase
background:    #0f172a
padding:       8px 12px
```

#### Table Body
```
font-family:   Inter
font-size:     0.8125rem       (13px)
font-weight:   400 (regular rows) | 500 (highlighted rows)
letter-spacing: 0px
line-height:   1.4
color:         #1e293b
numeric cols:  IBM Plex Mono, 0.8125rem, weight 500
```

#### Body Text (prose, descriptions)
```
font-family:   Inter
font-size:     0.875rem        (14px)
font-weight:   500
letter-spacing: 0px
line-height:   1.6
color:         #334155
usage:         Situation Brief, page descriptions, inline prose.
               NOT for labels, NOT for chart captions.
```

#### Caption
```
font-family:   Inter
font-size:     0.75rem         (12px)
font-weight:   400
letter-spacing: 0px
line-height:   1.5
color:         #64748b          ← CRITICAL: was #94a3b8 (fails WCAG AA — 3.7:1). 
                                  Change to #64748b (4.8:1 ✓)
usage:         ONE caption per section, at the bottom. Not repeated per chart.
               No emoji. Plain prose.
```

#### Navigation Item
```
font-family:   Inter
font-size:     0.8125rem       (13px)
font-weight:   600
letter-spacing: 0px
line-height:   1
color:         #94a3b8 (inactive) | #e2e8f0 (active) | #cbd5e1 (hover)
```

#### Navigation Group Label
```
font-family:   Inter
font-size:     0.625rem        (10px)     ← current 0.6rem is acceptable; keep
font-weight:   800
letter-spacing: 1.6px
text-transform: uppercase
color:         #475569
padding-top:   1rem
```

#### Situation Brief (executive narrative)
```
font-family:   Inter
font-size:     0.875rem        (14px)
font-weight:   600
letter-spacing: 0px
line-height:   1.65
color:         contextual — see Color System Section B.3
usage:         Single sentence. Maximum 2 sentences. Plain prose. No emoji.
```

---

### A.3 Type Rules — What Is Prohibited

The following patterns currently exist in the codebase and MUST be eliminated:

| Prohibited Pattern | Found In | Required Fix |
|--------------------|----------|-------------|
| Emoji in metric labels (`📊 CURRENT PRICE`) | `render_commodity_tab()` line ~7686 | Remove emoji, keep uppercase text |
| Emoji in chart/section titles (`🎯 Procurement Signals Summary`) | `render_executive_signals_table()` | Remove emoji, use plain label |
| Emoji in tab labels (`📊 Overview`) | `render_market_page()` tab_labels | Remove emoji from tab label text |
| `st.subheader()` native heading | `render_executive_signals_table()` | Replace with `.section-title` HTML |
| `st.markdown("## ...")` or `st.markdown("### ...")` rendering default Streamlit headings | Multiple pages | Replace with custom `.section-title` div |
| `st.caption()` repeated more than once per page section | Market pages, AI Predictions | Consolidate to one page-level caption per section |
| Inline `font-size: X` overriding type scale | 200+ instances across app | Remove in favor of CSS class references |

---

## SECTION B — COLOR SYSTEM

### B.1 Base Palette

#### Neutrals (Foundation)
```
--n-900:  #0f172a    Text primary — deepest, headers, metric values
--n-800:  #1e293b    Text secondary — dark body text
--n-700:  #334155    Text tertiary — descriptions, prose
--n-600:  #475569    Text muted — nav groups, tertiary labels
--n-500:  #64748b    Text disabled / captions (minimum for body text WCAG AA)
--n-400:  #94a3b8    Icons, decorative text only (NOT body text — fails WCAG AA)
--n-300:  #cbd5e1    Borders (strong)
--n-200:  #e2e8f0    Borders (default)
--n-100:  #f1f5f9    Backgrounds (subtle)
--n-050:  #f8fafc    Page background
--n-000:  #ffffff    Card surface
```

#### Brand (Structural Only)
```
--brand-dark:   #0f172a    Sidebar background
--brand-mid:    #1e293b    Sidebar hover, dividers
--brand-accent: #1e40af    Active nav border, interactive elements
```

### B.2 Semantic Status Colors

These three colors are the ENTIRE status vocabulary of this application. No other colors may carry status meaning.

#### BUY — Urgent Action Required
```
--c-buy:           #dc2626    Primary (text, borders, left-accent)
--c-buy-bg:        #fef2f2    Card/banner background
--c-buy-border:    #fca5a5    Card border
--c-buy-text:      #991b1b    Text on --c-buy-bg
usage:             ONLY for "BUY recommendation" and "critical urgency"
                   NOT for general "danger" or "error" — use --c-critical for system errors
```

#### HOLD — Adequate / No Action
```
--c-hold:          #2563eb    Primary
--c-hold-bg:       #eff6ff    Background
--c-hold-border:   #bfdbfe    Border
--c-hold-text:     #1d4ed8    Text
usage:             ONLY for "HOLD recommendation" and "within policy"
                   NOT for general "information" — use --c-info for informational states
```

#### MONITOR — Attention Needed
```
--c-monitor:       #d97706    Primary
--c-monitor-bg:    #fffbeb    Background
--c-monitor-border:#fde68a    Border
--c-monitor-text:  #92400e    Text
usage:             ONLY for "MONITOR recommendation" and "approaching threshold"
```

### B.3 Supporting Semantic Colors (Non-procurement)

These exist for non-procurement states and must NEVER overlap with BUY/HOLD/MONITOR.

#### Healthy / Success (non-procurement)
```
--c-healthy:        #059669    Primary (green)
--c-healthy-bg:     #f0fdf4    Background
--c-healthy-border: #86efac    Border
--c-healthy-text:   #166534    Text
usage:              "All positions secured", positive trends, data freshness OK
```

#### Critical (system-level errors)
```
--c-critical:       #b91c1c    Primary (darker red, distinct from --c-buy)
--c-critical-bg:    #fff1f2    Background
usage:              System errors, data unavailable, pipeline failures
                    Must be visually distinguishable from --c-buy context
```

#### Info (neutral information)
```
--c-info:           #0891b2    Primary (cyan)
--c-info-bg:        #ecfeff    Background
--c-info-border:    #a5f3fc    Border
usage:              Informational callouts, futures pricing notes, methodology
```

### B.4 Colors to REMOVE or RECLASSIFY

| Color | Current Use | Problem | Resolution |
|-------|-------------|---------|------------|
| `#7c3aed` purple | AI Predictions page header border-left; AVG DAYS COVER KPI card border; `--c-purple` variable | 5th semantic color with no defined meaning. Creates confusion — "what does purple mean?" | **Remove entirely.** AI Predictions header: use `--n-300` neutral border. AVG DAYS COVER: use `--c-info` (cyan) if value is informational, or remove border entirely. |
| `linear-gradient(135deg, #2563eb 0%, #1e40af 100%)` | `.recommendation` banner background | Gradient backgrounds are a consumer app / marketing pattern. Enterprise analytics tools use flat colors. | **Replace with flat `--c-hold-bg` background with `--c-hold` left border**, matching the `.alert-info` pattern. |
| `#059669` inline on Total Procurement Kgs tile | `_render_overview()` 6-tile row | Ad-hoc green outside the color variable system | Reclassify as `--c-healthy` and reference the variable |
| `#fee2e2 / #991b1b` inline signal cell | `_color_signal()` in `render_executive_signals_table()` | Inline style, bypasses CSS variables | Replace with semantic badge HTML using `.badge-buy` / `.badge-hold` / `.badge-monitor` |

### B.5 Color Usage Rules

**Rule 1 — Color communicates status only.**  
Border color on a KPI card = the status of that metric. Color used decoratively (e.g., blue border on every section header) destroys the signal.

**Rule 2 — Neutral borders are the default.**  
All cards, sections, containers default to `--n-200` (`#e2e8f0`) border. Only apply semantic color to the left accent of a card when that card's value carries BUY/HOLD/MONITOR meaning.

**Rule 3 — No gradient fills on data containers.**  
Gradients belong in marketing materials. Every data surface uses a flat background from the palette.

**Rule 4 — BUY/HOLD/MONITOR are procurement terms.**  
A rising price trend is NOT a BUY. A falling price is NOT a "sell" or a HOLD. Market data visualizations use `--c-healthy` (up trend positive for procurement = price falling) and `--c-buy` semantics only when the pipeline engine has issued a BUY recommendation.

**Rule 5 — The Situation Brief alert color matches reality.**  
`n_buy > 0` → `.alert-critical` (BUY red)  
`n_buy == 0, n_monitor > 0` → `.alert-monitor` (MONITOR amber)  
`n_buy == 0, n_monitor == 0` → `.alert-healthy` (green)  
Currently always `.alert-critical` — this is a credibility-destroying bug.

---

## SECTION C — SPACING SYSTEM

### C.1 Base Scale

```
--sp-1:   4px     Micro gaps (icon to label, badge internal padding)
--sp-2:   8px     Tight grouping (within card, between label and value)
--sp-3:   12px    Standard grouping (card padding vertical)
--sp-4:   16px    Card padding (standard)
--sp-5:   20px    Card padding (large)
--sp-6:   24px    Between sections (tight)
--sp-8:   32px    Between sections (standard)
--sp-10:  40px    Major section breaks
--sp-12:  48px    Page-level spacing
```

### C.2 Component Spacing Specification

#### Card Padding
```
Standard KPI card:     padding: 16px 20px    (--sp-4 --sp-5)
Compact KPI card:      padding: 12px 16px    (--sp-3 --sp-4)
Financial exposure:    padding: 16px 20px
Alert/situation brief: padding: 12px 16px    (--sp-3 --sp-4)
```

#### Card Gaps (between cards in a grid)
```
KPI tile row:          gap: 12px             (--sp-3)
Section card grid:     gap: 16px             (--sp-4)
```

#### Section Spacing
```
Between page sections: margin-top: 24px      (--sp-6)
Between subsections:   margin-top: 16px      (--sp-4)
Section bar margin:    margin: 20px 0 12px 0 (--sp-5 top, --sp-3 bottom)
                       CURRENT: 24px top / 16px bottom → REDUCE top to 20px
```

#### Sidebar Spacing
```
Brand block:           padding: 20px 16px 16px 16px
Nav group label:       padding-top: 16px, padding-bottom: 6px
Nav item:              padding: 8px 12px
Nav item gap:          2px (between items)
Bottom controls:       padding-top: 16px, border-top: 1px solid #1e293b
```

#### Table Spacing
```
Header cell:           padding: 8px 12px
Body cell:             padding: 7px 12px
Row height:            36px minimum
```

#### Page Container
```
Main content area:     padding: 0 32px 32px 32px
Max width:             1600px (keep existing)
Block container:       padding-top: 8px (reduce from current 0.5rem)
```

### C.3 Density Target

The goal is **maximum information density without clutter**. For the Executive Summary at 1440×900:

- Above-fold content (0–820px usable): Situation Brief + 5 KPI tiles + Actions Table (with Market Snapshot alongside)
- A 115px card min-height and 24px section margins guarantee content overflows to a second screen — defeating the one-screen goal
- **Target KPI card height:** 88px minimum (compact `exec-kpi` style — reduce from current 90px for tighter row)
- **Target section bar margin:** 20px top, 12px bottom (reduce from 24/16)
- **Target KPI tile gap:** 10px (reduce from 12px to save 8px across 4 gaps = 32px recovered)

---

## SECTION D — KPI CARD SYSTEM

### D.1 Current Problems (Specific)

| Problem | Evidence | Impact |
|---------|----------|--------|
| Two competing card classes: `.metric-card` (115px) vs `.exec-kpi` (90px) with no documented rule for which to use | `procurement_dashboard.py` `_kpi_card()` uses `.metric-card`; `render_exec_procurement_header_v2()` Section B uses `.exec-kpi` | Layout inconsistency — tiles in same row may have unequal heights |
| `.metric-card:hover` uses `transform: translateY(-1px)` — hover lift is a consumer/marketing pattern | `streamlit_app.py` CSS | Cards should not levitate when executives move their mouse over them. Remove. |
| The `metric-label` color is set to the `border_colour` argument — same color for label text AND left border. Fine for strong colors (red, blue), but fails for muted colors like `#94a3b8` (grey N/A card) where label becomes invisible | `_kpi_card()` function | Accessibility issue on grey/muted cards |
| `6 columns` tile row in Procurement Overview — at 1440px, each tile is ~200px wide. Cards become cramped. `font-size: 1.8rem` metric value with `~1,234,567` (7+ chars) clips in this width | `_render_overview()` | Visual degradation for large numeric values |
| The `border-left: 4px solid {colour}` is the ONLY differentiator between a standard card and a status card. There is no visual difference between a green HOLD card and a blue HOLD card beyond the 4px strip | All cards | Weak visual hierarchy |

### D.2 Consolidated Card System

Replace the two-class system with ONE class and modifiers:

#### `.kpi-card` — Base Card (replaces both `.metric-card` and `.exec-kpi`)
```css
background:    #ffffff
border:        1px solid #e2e8f0
border-radius: 8px
padding:       14px 18px
min-height:    86px
display:       flex
flex-direction: column
justify-content: space-between
box-shadow:    0 1px 2px rgba(0,0,0,0.05)
/* NO hover transform */
/* NO transition */
```

#### `.kpi-card--status` — Status Variant (when card carries BUY/HOLD/MONITOR meaning)
```css
border-left:   3px solid [semantic-color]
padding-left:  15px    /* 18px - 3px border compensation */
```
Note: Left border is ONLY 3px (not 4px) — thinner border reads as more refined.

#### `.kpi-card--large` — Hero Metric (used in Financial Exposure totals only)
```css
padding:       18px 22px
min-height:    100px
```

#### `.kpi-card--alert` — Situation Brief Banner (Section A)
```css
/* No separate class needed — use semantic alert classes */
/* .alert-critical / .alert-monitor / .alert-healthy */
padding:       12px 16px
border-left:   4px solid [semantic-color]
border-radius: 6px
/* No card shadow — alert banners are flat */
```

### D.3 Four Card Archetypes

#### Archetype 1 — Executive KPI (5-tile row on Executive Summary)
```
Structure:
  LABEL          ← 10px/700/uppercase/muted color
  VALUE          ← IBM Plex Mono, clamp(1.375rem, 1.6vw, 1.875rem), #0f172a
  SUBLABEL       ← 11px/500/Inter/#64748b (org name, date, or unit)

Status indicator: left border 3px (semantic color) when card has BUY/HOLD/MONITOR meaning
Width: equal in 5-column grid
Height: 86px min-height

CRITICAL RULE: The ACTIONS REQUIRED tile, when n_buy > 0, should use:
  - left border: --c-buy
  - VALUE color: --c-buy (the number itself turns red, not just the border)
  This creates stronger pre-attentive signaling for the most important tile.
```

#### Archetype 2 — Market KPI (4-tile row on commodity pages)
```
Structure:
  LABEL          ← 10px/700/uppercase
  VALUE          ← IBM Plex Mono, same responsive scale
  SUBLABEL       ← Unit + as-of date
  [OPTIONAL] TREND DELTA  ← 11px, --c-healthy or --c-buy color, ↑/↓ + pct

Status indicator: NO left border by default (market KPIs are not procurement-status)
                  Use --c-healthy left border ONLY for "current price" card when data is fresh
Exception: If market signal = BUY/HOLD/MONITOR from pipeline, apply signal color to label
```

#### Archetype 3 — Financial KPI (3-column Financial Exposure section)
```
Structure:
  COMMODITY HEADER    ← Section label (pill)
  HERO METRIC         ← Cost in $M or "—" — IBM Plex Mono, large
  SUPPORTING ROWS     ← 3 rows of supporting data (Kgs, price used, PKR equivalent)
                         Each row: label (10px, muted) + value (13px, IBM Plex Mono)
  DISCLAIMER FOOTER   ← 11px italic "Estimated — not a procurement quote"

This is NOT a standard KPI card — it is a mini data panel. Use `.kpi-card--large`.
```

#### Archetype 4 — Alert/Situation Card (Situation Brief + Critical Risk banners)
```
Structure:
  BANNER TEXT         ← 14px/600/Inter, full-width
  [OPTIONAL] TIMESTAMP ← 11px, right-aligned

Rules:
  - Semantic alert class determines background and border
  - NO icon prefix in text (remove ⚠️, ◉, ✓ icons from Situation Brief)
  - The left-border color communicates the alert level — no icon needed
  - Max 2 sentences
```

---

## SECTION E — SIDEBAR REDESIGN

### E.1 Honest Assessment of Current Sidebar

The current sidebar is 80% of the way to enterprise quality. The dark background (`#0f172a`), the Inter font, and the hidden radio button circles are all correct. What makes it read as developer-made:

1. **Brand block:** "MG Apparel" in tiny uppercase label + "Commodity / Intelligence" in two lines is correct in concept but wrong in proportion. The brand name is too small relative to the product name.
2. **Nav group labels at `0.6rem`** — at this size on a dark background, letter-spacing 1.5px, these are decorative rather than readable. They provide structural hint but no actual label value. This is acceptable IF the groups themselves are obvious from context.
3. **The toggle ("Demo Forecast Mode") lives in nav** — this tells every executive that the data they're looking at might be fake. It must be removed from the default sidebar view.
4. **"Data Freshness" expander** in the nav — see Section H for relocation.
5. **Active state** uses `background: #1e3a8a` (strong blue) with `border-left: 3px solid #3b82f6` — this is a reasonable active state. However the blue `#3b82f6` is slightly off from the app's `--c-hold: #2563eb`. Standardize to `#2563eb`.
6. **The sidebar width** is Streamlit default (~260px). This is fine — do not resize.

### E.2 Sidebar Specification

#### Brand Block
```
Layout:         Stacked, left-aligned
Separator:      1px solid #1e293b, below block

Line 1 (org):   "MG APPAREL"
                0.6rem / 800 / uppercase / letter-spacing 2px / color #475569

Line 2 (product): "Commodity Intelligence"
                1.05rem / 800 / color #e2e8f0 / letter-spacing -0.3px / line-height 1.2
                (keep "Commodity" and "Intelligence" on separate lines — current is correct)

Padding:        20px 16px 16px 16px
```

#### Navigation Groups

**Proposed IA (from UX-3B audit — finalized here):**
```
EXECUTIVE
  📊 Executive Summary         ← keep emoji in nav only (not in content)

OPERATIONS
  📦 Procurement Intelligence
  🌍 International Markets
  🇵🇰 Pakistan Markets

INTELLIGENCE
  🧠 Market Intelligence
  🤖 AI Forecasts

──────────────────────────
  [status bar — see below]
```

**Rationale for IA change:**
- "Market Intelligence" group renamed to INTELLIGENCE (eliminates group/page name collision)
- "AI Predictions" renamed to "AI Forecasts" (clearer deliverable, not a research tool name)
- "Pakistan Local" renamed to "Pakistan Markets" (professional; "Local" is parochial)
- "AI Predictions" moved under INTELLIGENCE (single-item groups are anti-pattern)
- "Procurement Intelligence" moved to OPERATIONS (its audience is procurement/supply chain, not "analytics")
- "International Market" → "International Markets" (plural — multiple commodities)

#### Navigation Item States

```
Default:
  padding:       8px 12px
  border-radius: 6px
  background:    transparent
  color:         #94a3b8
  font:          13px/600/Inter
  border-left:   3px solid transparent  ← reserve space, invisible

Hover:
  background:    #1e293b
  color:         #cbd5e1
  border-left:   3px solid transparent

Active:
  background:    rgba(37, 99, 235, 0.12)   ← subtle blue tint, not block fill
  color:         #dbeafe
  border-left:   3px solid #2563eb
  font-weight:   700
```

#### Status Bar (replaces Demo toggle + Data Freshness expander)
```
Position:       Bottom of sidebar, above the fold on all standard screen heights
Separator:      1px solid #1e293b above

Contents:
  Row 1: Pipeline status chip
    Label:      "PIPELINE"  (0.55rem/800/uppercase/#475569)
    Value:      "Current" (green #059669) or "Stale" (amber #d97706) or "No Data" (muted)
    Source:     _STRATEGY_CSV.stat().st_mtime — "Current" if < 24h ago

  Row 2: Data age chip
    Label:      "MARKET DATA"
    Value:      "2h ago" or "1d ago" or last-updated timestamp
    Source:     Supabase data freshness check (existing logic)

Style:
  padding:      12px 16px
  font-size:    0.7rem (status labels) / 0.75rem (values)
  layout:       flex, space-between per row
```

**Remove entirely from sidebar:**
- `st.toggle("Demo Forecast Mode")` — move to AI Forecasts page header
- `st.expander("Data Freshness")` — replaced by status bar above

---

## SECTION F — TABLE SYSTEM

### F.1 Current Table Audit

All tables in the application use `st.dataframe()` with partial CSS override via `[data-testid="stDataFrame"]`. The current table CSS has several correct elements but multiple gaps.

**What is working:**
- `thead tr th` override: dark header (`#0f172a` background, `#e2e8f0` text, uppercase) — correct enterprise style
- `tbody tr:hover` blue tint hover — acceptable

**What is broken:**

| Issue | Location | Evidence |
|-------|----------|----------|
| No zebra striping — body rows are all white. Dense tables (10+ rows) become hard to scan without alternating rows | All `st.dataframe()` calls | CSS: no `tbody tr:nth-child(even)` rule |
| Row height is Streamlit default (~32px for native dataframe) — too compact for executive scanning | All tables | No explicit row height in CSS |
| Numeric columns not right-aligned — "1,234,567" in a left-aligned column is harder to compare than right-aligned | `_render_buy()`, `_render_overview()` | `st.dataframe()` does not expose column-level alignment |
| Streamlit dataframe renders index/sort arrows/download button that cannot be fully hidden with CSS — visible Streamlit fingerprint | All tables | `hide_index=True` hides index but sort arrows remain |
| `display["Inventory (Kgs)"]` formatted as string ("1,234,567") loses numeric sorting capability | All render functions | Formatted before passing to dataframe |
| Row color coding uses pandas `Styler.apply()` with inline hex strings instead of CSS class names | `_row_style()` in `_render_buy()` | `return ["background:#fff1f2"] * len(row)` |

### F.2 Table Standard Specification

#### Header Row
```
background:     #0f172a
color:          #e5e7eb
font:           11px/700/Inter/uppercase
letter-spacing: 0.8px
padding:        8px 12px
border-bottom:  2px solid #1e293b
```

#### Body Rows
```
Default:        background #ffffff
Alternate:      background #f8fafc    (zebra — nth-child even)
Hover:          background #eff6ff    (current — keep)
Row height:     38px minimum
font:           13px/400/Inter/#1e293b
padding:        7px 12px per cell
border-bottom:  1px solid #f1f5f9     (very subtle row separator)
```

#### Status Row Colors (override alternating)
```
Critical (<7d): background #fff1f2    (current — keep)
Warning (<15d): background #fffbeb    (current — keep)
Healthy row:    no override — use alternating
```

#### Numeric Alignment
```
All Kgs / numeric columns: right-aligned
All date columns:          left-aligned  
Org name:                  left-aligned
Action badges:             center-aligned
Days Cover:                right-aligned, IBM Plex Mono
```

#### Column Width Priorities
```
Org:            200px min, flexible
Commodity:      100px
Days Cover:     90px, fixed
Action:         80px, fixed (badge)
Confidence:     80px, fixed (badge)
Numeric Kgs:    130px, fixed
```

#### What To Do About `st.dataframe()` Limitations
`st.dataframe()` cannot fully match enterprise table standards because sort arrows, the download button, and column resize handles reveal Streamlit origin. For **executive-facing tables** (Actions Matrix on Executive Summary), use **custom HTML tables** rendered with `st.markdown(unsafe_allow_html=True)`. This gives full control. For **analyst tables** (Procurement Intelligence BUY/HOLD detail tabs), `st.dataframe()` with the current CSS override is acceptable — analysts expect interactive tables.

**Executive table (HTML):**
```html
<table class='exec-table'>
  <thead><tr>
    <th>Org</th><th>Commodity</th><th>Cover</th><th>Stockout</th><th>Action</th>
  </tr></thead>
  <tbody>
    <tr class='exec-table-row--critical'>
      <td>MTM-Spin-U3</td><td>Cotton</td>
      <td class='num'>14.2d</td><td>Jun 13</td>
      <td><span class='badge-buy'>BUY</span></td>
    </tr>
  </tbody>
</table>
```

**Analyst table:** `st.dataframe()` with existing CSS — acceptable for Procurement Intelligence page.

---

## SECTION G — CHART SYSTEM

### G.1 Chart Inventory and Decisions

| Chart | Page | Current Type | Decision | Reason |
|-------|------|-------------|----------|--------|
| Procurement Recommendation | Executive Summary | `px.pie` hole=0 | **REPLACE** with horizontal stacked bar | Pie charts require angle estimation; humans perform poorly at it for similar-sized segments |
| Action Distribution | Procurement Overview | `px.pie` hole=0.55 (donut) | **REPLACE** with horizontal bar chart (3 bars: BUY / HOLD / MONITOR) | Same as above; donut with hole is marginally better but still a poor encoding |
| BUY Shortfall | Procurement BUY tab | `go.Bar` vertical | **KEEP + REDESIGN** (horizontal bars, add org labels inline) | Good encoding; redesign to horizontal for readability with long org names |
| Days Cover by Commodity | Procurement Risk tab | existing chart | **KEEP** | Appropriate encoding |
| Historical price line | Market pages | `go.Scatter` | **KEEP + ENHANCE** | Add ±1σ reference band |
| Forecast bar chart | Market pages | `go.Bar` (forecast) | **REPLACE** with `go.Scatter` line + shaded confidence interval | Bar charts imply discrete/categorical data; forecasts are continuous |
| AI Prediction (predicted vs actual) | AI Predictions | `go.Scatter` + `go.Bar` | **KEEP** | Correct encoding: bars for actuals (discrete observations), line for model predictions |
| Country cotton forecast | International Markets | `go.Scatter` | **KEEP** | Good encoding for price comparison |
| Commodity comparison Overview | Market Overview tab | `px.bar` default colors | **REDESIGN** color scheme | Default Plotly colors break app palette |

### G.2 Replacement Designs

#### Horizontal Stacked Bar (replaces both pies)

For Procurement Overview Action Distribution:
```python
go.Bar(
    y=["Portfolio"],
    x=[n_buy],
    name="BUY",
    marker_color=_C_BUY,
    orientation='h',
    text=[f"{n_buy} BUY"],
    textposition='inside',
    textfont=dict(color='white', size=11, family='Inter'),
)
# + separate traces for HOLD (--c-hold) and MONITOR (--c-monitor)

layout:
  height=64
  barmode='stack'
  showlegend=False
  margin=dict(l=0, r=0, t=0, b=0)
  xaxis=dict(showticklabels=False, showgrid=False, showline=False)
  yaxis=dict(showticklabels=False)
  plot_bgcolor='rgba(0,0,0,0)'
  paper_bgcolor='rgba(0,0,0,0)'
```

For the Executive Summary, the stacked bar can be placed at 64px tall in a dedicated row above Position Summary — a compact visual signature rather than a chart-sized element.

#### Forecast Line with Confidence Band (replaces forecast bars)
```python
# Historical line
go.Scatter(x=dates, y=historical, mode='lines',
           line=dict(color='#334155', width=1.5), name='Historical')
# Forecast line
go.Scatter(x=forecast_dates, y=forecast_mean, mode='lines',
           line=dict(color='#2563eb', width=2, dash='dot'), name='Forecast')
# Confidence band (upper + lower filled)
go.Scatter(x=[*forecast_dates, *forecast_dates[::-1]],
           y=[*upper_bound, *lower_bound[::-1]],
           fill='toself',
           fillcolor='rgba(37, 99, 235, 0.08)',
           line=dict(color='rgba(0,0,0,0)'),
           name='Confidence')
```

### G.3 Universal Chart Standards

These rules apply to every Plotly chart in the application. The `_chart_layout()` function in `procurement_dashboard.py` partially implements them — extend it to be the single source of truth for all charts.

#### Colors
```
Historical data line:  #334155  (--n-800, dark slate — neutral)
Forecast/model line:   #2563eb  (--c-hold — consistent with "forward-looking = blue")
BUY signal:            #dc2626  (--c-buy)
HOLD signal:           #2563eb  (--c-hold)
MONITOR signal:        #d97706  (--c-monitor)
Positive trend:        #059669  (--c-healthy)
Negative trend:        #dc2626  (--c-buy — rising commodity price = cost pressure = BUY-adjacent)
Reference bands:       rgba(37, 99, 235, 0.06)  (very light blue fill for ±1σ)
Grid lines:            #f1f5f9  (--n-100 — lighter than current #e2e8f0)
Axis lines:            #e2e8f0  (--n-200)
```

#### Layout
```
plot_bgcolor:     #ffffff      (white plot area — not #fafafa)
paper_bgcolor:    transparent
font.family:      Inter, sans-serif
font.color:       #64748b
margin:           dict(l=48, r=16, t=32, b=48)
                  (reduce from current l=60 r=20 t=40 b=60 — tighter)
height:           320px standard | 200px compact | 400px full-width
```

#### Gridlines
```
X-axis:  showgrid=False, showline=True, linecolor=#e2e8f0
Y-axis:  showgrid=True, gridcolor=#f1f5f9, gridwidth=1, showline=False
         (lighter grid than current #e2e8f0 — less visual noise)
Tick font: 10px / Inter / #94a3b8
```

#### Titles and Labels
```
Chart title:    NOT in fig.update_layout(title=...) 
                Instead use an H3 heading ABOVE the chart via st.markdown()
                Reason: Plotly title rendering is inconsistent cross-platform

Legend:
  bgcolor:      rgba(0,0,0,0)
  borderwidth:  0
  font.size:    10
  orientation:  'h' when more than 2 series
  yanchor:      'bottom', y=1.02  (above chart, not overlapping data)
```

#### Tooltips
```
hovermode:       'x unified'  for time-series charts
                 'closest'    for scatter charts
hoverlabel:
  bgcolor:       #1e293b
  bordercolor:   #334155
  font.color:    #e2e8f0
  font.size:     12
  font.family:   Inter
```

#### What Must Be Removed from Charts
- `textposition="outside"` on bar chart bars with `text=` labels — these overflow containers
- Default Plotly color sequences (`px.bar` etc.) — always pass explicit `color_discrete_map`
- `title=` in `fig.update_layout()` — move all chart titles to `st.markdown()` above chart

---

## SECTION H — INFORMATION ARCHITECTURE

### H.1 Content Audit — Per Page

#### EXECUTIVE SUMMARY — Keep / Remove / Relocate

| Content | Decision | Rationale |
|---------|----------|-----------|
| Situation Brief (Section A) | **KEEP + FIX** | Fix alert semantic state (always red is wrong) |
| 5 KPI Tiles (Section B) | **KEEP** | Correct L1 content |
| Critical Risks (Section 2 v2) | **KEEP + FLATTEN** | Remove expander — show top 3 inline |
| Market Snapshot (Section 3 v2) | **KEEP + RELOCATE** | Move to 1/3 column alongside Actions Table |
| Financial Exposure (Section E) | **KEEP** | Correct L2 content |
| Procurement Recommendation pie | **REPLACE** | See Section G.2 |
| Executive Insights (Section 5) | **REMOVE** | Analyst-level noise; repeats tile data |
| Position Summary | **KEEP + REDESIGN** | Fix coverage bars (Section D) |
| Executive Brief bullets | **KEEP** | Short bullet form appropriate for L2 |
| Procurement Signals Summary table | **RELOCATE** | → Market Intelligence page |
| Country Cotton Section | **RELOCATE** | → International Markets page |
| Market Forecasts divider | **REMOVE** | Orphaned, no content follows |
| `render_executive_signals_table()` | **RELOCATE** | → Market Intelligence page as top section |

**Reason for relocating Procurement Signals Summary:**  
This table shows MoM change, 3M forecast Δ, and BUY NOW/WAIT/MONITOR signals for Cotton, Crude Oil, Natural Gas, Polyester. This is 4 rows × 6 columns of analyst content. An executive reads KPI tiles. The signals summary is the right content for a Market Intelligence analyst page, not the executive dashboard.

---

#### PROCUREMENT INTELLIGENCE — Keep / Remove / Relocate

| Content | Decision | Rationale |
|---------|----------|-----------|
| Tab 1: Procurement Overview (6-tile row + pie + table) | **REDESIGN** | 6 KPI tiles → 4 tiles; pie → stacked bar; table → keep |
| Tab 2: BUY Recommendations (banner + table + bar chart) | **KEEP + MINOR FIXES** | Core content; fix chart to horizontal bars |
| Tab 3: HOLD Recommendations (banner + table) | **KEEP** | Good structure |
| Tab 4: MONITOR Recommendations (grouped by reason) | **KEEP** | Useful for procurement team |
| Tab 5: Inventory Risk (days cover chart + scatter) | **KEEP** | Right page for this content |
| Tab 6: Procurement Log | **KEEP** | Audit trail — analyst-appropriate |
| `_section_header()` card headers | **REDESIGN** | Remove border-left from section headers — overused pattern |

**Procurement Overview 6→4 tile reduction:**  
Remove: "TOTAL PROCUREMENT (Kgs)" and "CRITICAL (<15 days)" from the tile row. These are derivable from the BUY table which follows immediately below. Keep: BUY count, HOLD count, MONITOR count, AVG DAYS COVER.

---

#### INTERNATIONAL MARKETS — Keep / Remove / Relocate

| Content | Decision | Rationale |
|---------|----------|-----------|
| Per-commodity tabs | **KEEP** | Correct page structure |
| 4 KPI tiles per commodity | **REDESIGN** | Remove emoji from labels; fix color system |
| Data freshness indicator (`st.success()/st.info()`) | **REPLACE** | Replace with custom `.data-status-chip` HTML — eliminates Streamlit native |
| Historical line chart | **KEEP + ENHANCE** | Add ±1σ band |
| Forecast bar chart | **REPLACE** | → Line chart with CI |
| Data table + export | **KEEP** | Analyst need |
| Procurement signal (`st.success()/st.warning()/st.error()`) | **REPLACE** | Replace native widgets with custom HTML `.procurement-signal` card |
| Overview comparison tab | **KEEP + REDESIGN** | Fix color scheme |
| Country cotton section | **RELOCATE HERE** | From Executive Summary |
| `st.caption()` per chart | **CONSOLIDATE** | ONE page-level caption per commodity, not per chart |

---

#### PAKISTAN MARKETS — Keep / Remove / Relocate

Same as International Markets. Additionally:
- USD/PKR live data card: **KEEP** — unique to this page, relevant for procurement cost translation
- Electricity cost card: **KEEP** — relevant for Pakistan operations context
- Rename page from "Pakistan Local" → "Pakistan Markets"

---

#### MARKET INTELLIGENCE — Keep / Remove / Relocate

| Content | Decision | Rationale |
|---------|----------|-----------|
| Critical Alerts section | **KEEP + REDESIGN** | Replace `st.error()` with custom HTML |
| News by Commodity tab | **KEEP** | Core content |
| Geopolitical tab | **KEEP** | Core content |
| Weather & Supply tab | **KEEP** | Core content |
| Refresh button (top right) | **KEEP + RELOCATE** | Move to inline with page header, not in a column layout |
| Procurement Signals Summary | **ADD HERE** | Relocated from Executive Summary |
| Executive Signals Table | **ADD HERE** | Relocated from Executive Summary |

---

#### AI FORECASTS — Keep / Remove / Relocate

| Content | Decision | Rationale |
|---------|----------|-----------|
| Purple `border-left: 4px solid #7c3aed` page header | **REPLACE** | Use neutral `--n-300` — purple is not a system color |
| Supabase config warning | **KEEP** | Necessary |
| Months slider | **KEEP** | Useful control |
| Prediction charts | **KEEP** | Core content |
| `st.caption()` below slider | **KEEP + REWRITE** | Remove emoji (`📊`), plain prose |
| Forecast Accuracy backtesting | **KEEP** | Valuable |
| `st.expander()` per commodity | **KEEP** | Appropriate for analyst-level backtesting detail |
| Demo Forecast Mode toggle | **ADD HERE** | Relocated from sidebar |
| Page-level status strip (model accuracy, next forecast, procurement signal) | **ADD** | New L1 orientation (from UX-3B audit) |

---

### H.2 Cross-Page Content Rules

1. **Executive Summary contains ONLY executive content:** Status → context → financial impact → required actions. No analyst tables, no methodology notes, no per-chart captions.

2. **Procurement signal derivation appears on ONE page.** Currently derived and displayed on both Executive Summary (via `render_executive_signals_table()`) and Market Intelligence. Source of truth: Market Intelligence. Executive Summary gets a compressed version in the Situation Brief only.

3. **`st.markdown("---")` native dividers are BANNED.** All section breaks use the `.exec-section-bar` HTML pattern. This eliminates the most visible Streamlit UI artifact.

4. **Native Streamlit status widgets are BANNED on executive-facing content.** `st.success()`, `st.warning()`, `st.info()`, `st.error()` may only appear in system error states (data unavailable, configuration errors). All content-level status uses custom HTML.

---

## SECTION I — HUMAN PSYCHOLOGY REVIEW

### I.1 Cognitive Load

**Three types of cognitive load (Sweller, 1988):**
- **Intrinsic:** complexity inherent to the content — unavoidable for procurement decisions
- **Extraneous:** design-imposed friction — must be minimized
- **Germane:** mental model building — should be maximized

**Extraneous load sources in current application:**

| Source | Severity | Mechanism | Fix |
|--------|----------|-----------|-----|
| "Executive Insights" section generates 5–8 bullets that executives must read and filter | High | Forces serial reading of redundant information | Remove entirely |
| Emoji throughout UI creates micro-interruptions (brain processes image differently from text) | Medium | Dual-coding interference in professional scanning context | Remove from data labels, section headers, chart titles — keep in sidebar nav only |
| Two card size classes create subconscious "why are these different" questions | Low | Unexplained visual inconsistency | Consolidate to one class (Section D) |
| Critical Risks in expander requires "click to see critical information" | High | Forces interaction to access high-importance content | Inline top 3 risks |
| 6-tab Procurement Intelligence with generic tab names ("BUY", "HOLD") | Medium | Tab choice requires mental simulation of content | Rename to "Action Required", "Adequate Stock", "Under Review" |

### I.2 Executive Scanning Behavior

F-pattern (Nielsen, 1997) and Z-pattern scanning research for executive dashboards:
- **First fixation:** Top-left (brand, navigation)
- **Horizontal sweep:** KPI tiles (left to right)
- **Vertical drop:** Left column of page content
- **Final fixation:** Bottom-left or bottom-center summary

**Implications:**
1. ACTIONS REQUIRED tile must be FIRST (leftmost) — currently position B1 ✓
2. Situation Brief banner (full-width, top) receives near-100% attention — currently ✓ but color is wrong
3. Content in right column of 2-column layouts receives ~40% attention — Financial Exposure (right 1/3) is less visible than it should be given its importance
4. The Procurement Recommendation chart — if below fold — receives <20% attention. Executives may never see it.

**Layout recommendation:** In the final layout, the procurement action summary (BUY/HOLD/MONITOR breakdown) should be a compact element IN the KPI row area, not a separate full-width chart section below fold.

### I.3 Decision-Making Support

**Sensemaking model (Dervin, Klein):** Executives do not read dashboards linearly. They form a hypothesis ("is there a procurement problem?") and seek confirming or disconfirming evidence. The design must support this:

1. **Hypothesis confirmation in <5 seconds:** Situation Brief (headline) → ACTIONS REQUIRED tile (count) → LOWEST COVER tile (most urgent) = complete answer to "is there a problem?"

2. **Cost framing problem:** Financial Exposure leads with cost (`~$8.9M`) but not with risk (stockout in 14 days). Executives anchor on cost. The procurement team needs them to act based on risk, not spend. **Lead with risk: "Stockout Risk: Jun 13" → "Mitigation Cost: ~$8.9M"**

3. **Loss framing vs gain framing (Kahneman/Tversky):** Current HOLD message: "No procurement action required." Proposed HOLD message: "Production secured through Aug 15." Same data, framed as gain rather than absence of loss. Executives respond more positively to the gain framing.

4. **Temporal anchoring:** Executives need "compared to what?" context. Without a delta (↑2 from last month), executives cannot determine if the situation is improving or deteriorating. Add delta to every KPI tile and every status section.

### I.4 Visual Trust

Factors that create or destroy executive trust in a dashboard:

| Trust-building | Trust-destroying |
|----------------|-----------------|
| Consistent typographic scale | Multiple font sizes on same level of hierarchy |
| Color that means something specific | Color used for decoration |
| Data-to-ink ratio optimization | Emoji, gradients, decorative borders |
| Professional monospace numerics | Proportional font numbers that shift as values change |
| Specific estimates with clear disclaimers | Either no estimates or estimates without caveats |
| Dates and timestamps on all data | No indication of data freshness |
| Single authoritative layout | Inconsistent card heights, mixed border styles |

**The single most trust-destroying element in this application:** The Situation Brief banner is always red, even when everything is fine. Executives who see a red banner with "All positions meet the 45-day policy requirement" lose trust in the signal system immediately. This must be fixed as the first and highest-priority change.

### I.5 Friction Points

Ranked by frequency × severity:

1. **Incorrect alert color** on all-HOLD or all-MONITOR situations — misleads executives on every page load when no BUY exists
2. **Native Streamlit status widgets** visible on commodity pages — destroys illusion of custom enterprise product
3. **Emoji in data labels** — creates perception of consumer/demo product
4. **Below-fold procurement recommendation** — primary decision support content not visible without scrolling
5. **Orphaned "Market Forecasts" divider** — creates impression of broken/incomplete page
6. **`st.caption()` after every chart** — executive cannot distinguish which captions are important vs boilerplate
7. **Demo Forecast Mode in nav** — executives see this and wonder if current data is real

---

## SECTION J — IMPLEMENTATION ROADMAP

### UX-3C — Design System Foundation

**Objective:** Eliminate every element that makes this product look like a Streamlit app or developer project. No new features. Pure refinement.

**Files affected:**
- `streamlit_app.py` (CSS block only — approx lines 2000–2490)
- `scripts/procurement_dashboard.py` (visual helpers only)

**Risk:** Low — CSS changes only + targeted HTML string edits. No logic changes.

**Business impact:** High — immediately transforms executive perception from "internal tool" to "enterprise product"

**Effort:** 1–2 days

| Step | Change | Specific Location |
|------|--------|------------------|
| 3C-1 | Fix Situation Brief semantic alert state | `_build_situation_brief()` return value used in `render_exec_procurement_header_v2()` Section A — add `_alert_class(n_buy, n_monitor)` helper |
| 3C-2 | Remove Executive Insights section | Delete Section 5 block in `render_exec_procurement_header_v2()` |
| 3C-3 | Remove orphaned Market Forecasts divider | Delete orphaned divider at end of `render_exec_procurement_header_v2()` |
| 3C-4 | Flatten Critical Risks — remove expander | Replace `st.expander()` with direct `st.markdown()`, max 3 items |
| 3C-5 | Remove emoji from all metric labels | All `_kpi_card()` calls — strip emoji from `label` argument |
| 3C-6 | Replace `st.progress()` with custom HTML bars | Position Summary section |
| 3C-7 | Fix caption color `#94a3b8` → `#64748b` | CSS `[data-testid="stCaption"]` |
| 3C-8 | Remove `.metric-card:hover` translateY transform | CSS — delete hover transform, keep shadow |
| 3C-9 | Remove gradient from `.recommendation` class | CSS — replace gradient with flat `--c-hold-bg` |
| 3C-10 | Consolidate duplicate `get_critical_alerts()` | Delete second copy at line ~9129 (duplicate of line ~8147) |
| 3C-11 | Fix purple header on AI Predictions page | `render_ai_predictions_page()` — change `#7c3aed` border to `#e2e8f0` |
| 3C-12 | Move Demo Forecast Mode toggle | From sidebar to AI Forecasts page inline header |

**Acceptance test for 3C:** Load Executive Summary with 0 BUY rows → green banner. Load with BUY rows → red banner. No `st.success()`/`st.error()`/`st.warning()` visible on Executive Summary. No emoji in any KPI label.

---

### UX-3D — Executive Summary Completion

**Objective:** Implement remaining UX-2 Steps 4–8 plus the layout changes specified in this design system.

**Files affected:**
- `scripts/procurement_dashboard.py` (new rendering sections)

**Risk:** Medium — complex HTML layout; `st.columns()` proportion tuning needed

**Business impact:** Critical — completes the approved executive dashboard design

**Effort:** 3–5 days

| Step | Change |
|------|--------|
| 3D-1 | Actions Table (Section C) — custom HTML table, 2/3 width column |
| 3D-2 | Market Snapshot (Section D) — 1/3 width column alongside Actions Table |
| 3D-3 | Position Summary (Section F) — custom HTML progress bars, BUY/HOLD/MONITOR colors |
| 3D-4 | Executive Brief bullets (Section G) — 3 bullets max, BUY first |
| 3D-5 | Replace Procurement Recommendation pie — horizontal stacked bar (64px height) |
| 3D-6 | Add delta chips to KPI tiles (↑/↓ vs last period) — requires `prev_df` parameter |

---

### UX-3E — Cross-Page Consistency

**Objective:** Apply design system to all 5 non-executive pages. Eliminate all native Streamlit status widgets from content-level rendering.

**Files affected:**
- `streamlit_app.py` (market pages, intelligence page, AI predictions page)

**Risk:** Low-Medium — page-by-page substitution of native widgets with custom HTML equivalents

**Business impact:** High — unifies the product into a single visual language

**Effort:** 1 week

| Step | Change |
|------|--------|
| 3E-1 | Replace all `st.success()/st.info()/st.warning()/st.error()` in commodity tab | `render_commodity_tab()` data freshness indicator, procurement signal |
| 3E-2 | Add page-level status strip to market pages | New `_market_status_strip()` helper, called before `st.tabs()` |
| 3E-3 | Replace forecast bar charts with line + CI | `render_commodity_tab()` forecast section |
| 3E-4 | Add ±1σ bands to historical price charts | `render_commodity_tab()` history section |
| 3E-5 | Normalize all date formats to `%d %b %Y` | Audit all `strftime()` calls |
| 3E-6 | Relocate Procurement Signals Summary | From Executive Summary → Market Intelligence page |
| 3E-7 | Relocate Country Cotton section | From Executive Summary → International Markets page |
| 3E-8 | Replace `st.markdown("---")` dividers with `.exec-section-bar` | All pages |
| 3E-9 | Remove emoji from all section/chart titles | All pages (sidebar nav emoji: keep) |
| 3E-10 | Consolidate `st.caption()` — max one per section | All market pages |

---

### UX-3F — Navigation and Global Structure

**Objective:** Implement the IA changes specified in Section E (sidebar), add data freshness status bar, remove Demo Mode from sidebar, add page header timestamps.

**Files affected:**
- `streamlit_app.py` (sidebar block in `main()`, page header bar)

**Risk:** Low — navigation changes + CSS additions; no data pipeline changes

**Business impact:** Medium — professional IA, removes "Demo Mode" credibility issue

**Effort:** 2–3 days

| Step | Change |
|------|--------|
| 3F-1 | Rename nav items per Section E.2 IA specification | `main()` sidebar `_*_pages` list definitions |
| 3F-2 | Remove `st.toggle("Demo Forecast Mode")` from sidebar | Move control to AI Forecasts page |
| 3F-3 | Remove `st.expander("Data Freshness")` from sidebar | Replace with status bar |
| 3F-4 | Add sidebar status bar (pipeline status + data age) | New HTML block at bottom of sidebar section |
| 3F-5 | Add last-refresh timestamp to page header bar | Update header bar `div` in `main()` |
| 3F-6 | Standardize active nav item styling to spec (Section E.2) | CSS `[aria-checked="true"]` rule — change `#1e3a8a` → `rgba(37, 99, 235, 0.12)` |

---

### Phase Summary Table

| Phase | Name | Effort | Risk | Business Impact | Primary Deliverable |
|-------|------|--------|------|-----------------|---------------------|
| UX-3C | Design System Foundation | 1–2 days | Low | High | Credible executive product |
| UX-3D | Executive Summary Completion | 3–5 days | Medium | Critical | Complete approved design |
| UX-3E | Cross-Page Consistency | 1 week | Low-Med | High | Unified visual language |
| UX-3F | Navigation & Structure | 2–3 days | Low | Medium | Professional IA |

**Total estimated effort: 2.5–3.5 weeks**

**Sequencing constraint:** 3C must complete before 3D. 3D can overlap 3E. 3F is independent.

---

## APPENDIX — THE STREAMLIT FINGERPRINT CHECKLIST

Before any phase is marked complete, verify that none of these Streamlit artifacts remain on executive-facing pages:

```
[ ] st.success() / st.warning() / st.error() / st.info() — visible as colored boxes
[ ] st.subheader() / st.header() — native Streamlit heading rendering
[ ] st.markdown("---") — horizontal rule native Streamlit divider
[ ] st.progress() — teal progress bars
[ ] st.caption() repeated more than once per section
[ ] Emoji in metric labels, chart titles, or section headers (nav-only is OK)
[ ] Gradient backgrounds on data containers
[ ] hover: translateY(-1px) on KPI cards
[ ] Purple (#7c3aed) as a design color
[ ] Situation Brief always red regardless of n_buy
[ ] 6-column KPI grids (max 5 columns on executive pages)
[ ] Pie charts anywhere in the application
[ ] "Demo Forecast Mode" visible in sidebar
[ ] Caption text #94a3b8 (must be #64748b minimum)
[ ] Missing timestamp / data-as-of on any page
```

Zero checkmarks = enterprise product.  
Any checkmark = Streamlit application.

---

*End of UX-3B.1 Enterprise Design System Specification*  
*This document is the authoritative standard. All future UX phases must conform to the rules defined here.*  
*Do not implement until this specification is reviewed and approved.*
