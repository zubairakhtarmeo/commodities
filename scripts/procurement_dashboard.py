"""
procurement_dashboard.py
-------------------------
Streamlit rendering module for the PROCUREMENT INTELLIGENCE page.

Data source: reports/procurement_strategy.csv
             reports/procurement_strategy_meta.json

These files are written by run_monthly_strategy_pipeline.py Step 8.
This module contains ONLY visualisation logic — no business rules.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT  = Path(__file__).parent.parent
_STRATEGY_CSV  = _PROJECT_ROOT / "reports" / "procurement_strategy.csv"
_META_JSON     = _PROJECT_ROOT / "reports" / "procurement_strategy_meta.json"

# ---------------------------------------------------------------------------
# Colour palette — matches existing app CSS
# ---------------------------------------------------------------------------

_C_BUY     = "#dc2626"   # red  — urgent, act now
_C_HOLD    = "#2563eb"   # blue — adequate
_C_MONITOR = "#d97706"   # amber — attention needed
_C_GRID    = "#e2e8f0"
_C_TEXT    = "#0f172a"
_C_SUBTEXT = "#64748b"
_C_BG      = "#fafafa"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def _load_strategy_cached(mtime: float) -> pd.DataFrame:
    """Load CSV, keyed by file mtime so cache invalidates on new pipeline run."""
    df = pd.read_csv(_STRATEGY_CSV)
    numeric_cols = [
        "inventory_qty", "monthly_consumption", "daily_consumption",
        "need_45_days", "days_cover", "shortfall", "procurement_qty",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def load_procurement_strategy() -> tuple[pd.DataFrame, dict]:
    """Return (strategy_df, meta_dict). Returns empty df + empty dict if files absent."""
    if not _STRATEGY_CSV.exists():
        return pd.DataFrame(), {}

    mtime = _STRATEGY_CSV.stat().st_mtime
    df = _load_strategy_cached(mtime)

    meta: dict = {}
    if _META_JSON.exists():
        try:
            meta = json.loads(_META_JSON.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    return df, meta


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _action_badge(action: str) -> str:
    colours = {
        "BUY":     ("background:#fef2f2;color:#991b1b;border:1.5px solid #fca5a5;",     "BUY"),
        "HOLD":    ("background:#eff6ff;color:#1d4ed8;border:1.5px solid #bfdbfe;",     "HOLD"),
        "MONITOR": ("background:#fffbeb;color:#92400e;border:1.5px solid #fde68a;",     "MONITOR"),
    }
    style, label = colours.get(action, ("background:#f1f5f9;color:#475569;border:1.5px solid #e2e8f0;", action))
    return (
        f"<span style='display:inline-block;padding:2px 10px;border-radius:12px;"
        f"font-size:0.72rem;font-weight:700;letter-spacing:0.5px;{style}'>{label}</span>"
    )


def _confidence_badge(confidence: str) -> str:
    colours = {
        "HIGH":   "background:#f0fdf4;color:#166534;border:1.5px solid #86efac;",
        "MEDIUM": "background:#eff6ff;color:#1e40af;border:1.5px solid #bfdbfe;",
        "LOW":    "background:#fef3c7;color:#92400e;border:1.5px solid #fcd34d;",
    }
    style = colours.get(str(confidence).upper(),
                        "background:#f1f5f9;color:#475569;border:1.5px solid #e2e8f0;")
    return (
        f"<span style='display:inline-block;padding:2px 8px;border-radius:10px;"
        f"font-size:0.7rem;font-weight:600;{style}'>{confidence}</span>"
    )


def _kpi_card(label: str, value: str, sublabel: str, border_colour: str) -> str:
    return f"""
    <div class='metric-card' style='border-left:3px solid {border_colour};'>
        <div class='metric-label' style='color:{border_colour};'>{label}</div>
        <div class='metric-value' style='color:{_C_TEXT};'>{value}</div>
        <div class='currency-label'>{sublabel}</div>
    </div>"""


# ---------------------------------------------------------------------------
# UX-3A helpers — Steps 1–3  (display-layer only, no business logic)
# ---------------------------------------------------------------------------

def _snap_card_module(
    label: str,
    price,
    change,
    date: str,
    currency: str,
) -> None:
    """Module-level market snapshot card (extracted from render_exec_procurement_header inner fn)."""
    if price is None or price == 0:
        st.markdown(f"""
        <div class='metric-card' style='border-left:4px solid #94a3b8;opacity:0.6;'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value' style='font-size:1.2rem;color:#94a3b8;'>N/A</div>
            <div class='currency-label'>Data unavailable</div>
        </div>""", unsafe_allow_html=True)
        return
    try:
        chg = float(change or 0)
        arrow = "↑" if chg > 0 else ("↓" if chg < 0 else "→")
        trend_col = _C_BUY if chg > 2 else ("#059669" if chg < -2 else "#2563eb")
        st.markdown(f"""
        <div class='metric-card' style='border-left:4px solid {trend_col};'>
            <div class='metric-label' style='color:{trend_col};'>{label}</div>
            <div class='metric-value' style='color:{_C_TEXT};'>{float(price):.4f}</div>
            <div class='currency-label'>
                <strong>{currency}</strong><br>
                <span style='color:{trend_col};font-weight:700;'>{arrow} {abs(chg):.2f}% MoM</span><br>
                <span style='font-size:0.68rem;color:#94a3b8;'>{date or ""}</span>
            </div>
        </div>""", unsafe_allow_html=True)
    except Exception:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{price}</div>
            <div class='currency-label'>{currency}</div>
        </div>""", unsafe_allow_html=True)


def _short_org(org_name: str) -> str:
    """Abbreviate org name for narrow display contexts (tiles, table columns)."""
    s = str(org_name).strip()
    s = s.replace(" - Spinning ", "-Spin-").replace(" - Weaving ", "-Weav-")
    return s if len(s) <= 22 else s[:20] + "…"


def _stockout_date(days_cover: float) -> str:
    """Convert days_cover to a concrete stockout date string, e.g. 'Jun 05'."""
    try:
        dc = float(days_cover)
    except (TypeError, ValueError):
        return "N/A"
    if pd.isna(dc) or dc <= 0:
        return "N/A"
    stockout = datetime.date.today() + datetime.timedelta(days=dc)
    return stockout.strftime("%b %d")


def _compute_financial_exposure(df: pd.DataFrame, market_snapshot: dict) -> dict:
    """Compute display-layer cost estimates for BUY rows.

    All values are ESTIMATES for directional guidance — not procurement quotes.
    No procurement engine logic is executed here.
    """
    ice   = market_snapshot.get("ice_cotton", {}) if market_snapshot else {}
    psf   = market_snapshot.get("psf", {})        if market_snapshot else {}
    usdpk = market_snapshot.get("usd_pkr", {})    if market_snapshot else {}

    try:
        ice_price_lb = float(ice.get("price") or 0)
    except (TypeError, ValueError):
        ice_price_lb = 0.0
    try:
        psf_price_kg = float(psf.get("price") or 0)
    except (TypeError, ValueError):
        psf_price_kg = 0.0
    try:
        usd_pkr = float(usdpk.get("price") or 0)
        if usd_pkr <= 0:
            usd_pkr = 280.0
    except (TypeError, ValueError):
        usd_pkr = 280.0

    ice_price_kg          = ice_price_lb * 2.20462
    market_data_available = (ice_price_lb > 0 and psf_price_kg > 0)

    # Shortfall (BUY rows only) — used for cost estimates
    if df.empty or "shortfall" not in df.columns or "action" not in df.columns:
        cotton_gap = fiber_gap = 0.0
    else:
        buy        = df[df["action"] == "BUY"]
        cotton_gap = float(buy[buy["commodity"] == "Cotton"]["shortfall"].sum())
        fiber_gap  = float(buy[buy["commodity"] == "Fiber"]["shortfall"].sum())

    # Inventory on-hand — shown as context alongside cost, not used in cost calculation
    if df.empty or "inventory_qty" not in df.columns:
        cotton_inv = fiber_inv = 0.0
    else:
        cotton_inv = float(df[df["commodity"] == "Cotton"]["inventory_qty"].sum())
        fiber_inv  = float(df[df["commodity"] == "Fiber"]["inventory_qty"].sum())

    # 45-day requirement — context only
    if df.empty or "need_45_days" not in df.columns:
        cotton_need = fiber_need = 0.0
    else:
        cotton_need = float(df[df["commodity"] == "Cotton"]["need_45_days"].sum())
        fiber_need  = float(df[df["commodity"] == "Fiber"]["need_45_days"].sum())

    cotton_cost = cotton_gap * ice_price_kg if market_data_available else 0.0
    fiber_cost  = fiber_gap  * psf_price_kg  if market_data_available else 0.0
    total_cost  = cotton_cost + fiber_cost
    total_pkr   = total_cost * usd_pkr

    def _fmt(v: float) -> str:
        if v <= 0:
            return "—"
        if v >= 1_000_000:
            return f"~${v / 1_000_000:.1f}M"
        if v >= 1_000:
            return f"~${v:,.0f}"
        return f"~${v:.0f}"

    return {
        "market_data_available": market_data_available,
        "ice_price_usd_per_lb":  ice_price_lb,
        "ice_price_usd_per_kg":  ice_price_kg,
        "psf_price_usd_per_kg":  psf_price_kg,
        "usd_pkr_rate":          usd_pkr,
        "cotton_shortfall_kgs":  cotton_gap,
        "fiber_shortfall_kgs":   fiber_gap,
        "cotton_inv_kgs":        cotton_inv,
        "fiber_inv_kgs":         fiber_inv,
        "cotton_need_kgs":       cotton_need,
        "fiber_need_kgs":        fiber_need,
        "cotton_cost_usd":       cotton_cost,
        "fiber_cost_usd":        fiber_cost,
        "total_cost_usd":        total_cost,
        "total_cost_pkr":        total_pkr,
        "cotton_cost_display":   _fmt(cotton_cost),
        "fiber_cost_display":    _fmt(fiber_cost),
        "total_cost_display":    _fmt(total_cost),
    }


def _build_situation_brief(
    df: pd.DataFrame,
    exposure: dict,
    market_snapshot: dict,
) -> str:
    """Generate the one-sentence executive narrative for Section A (Situation Brief)."""
    if df.empty:
        return (
            "Pipeline data unavailable — run the monthly pipeline to generate this brief."
        )

    n_buy = int((df["action"] == "BUY").sum())

    if n_buy == 0:
        return (
            f"All {len(df)} org-commodity pairs meet the 45-day policy requirement. "
            "No procurement action required this period."
        )

    active_buy = df[(df["action"] == "BUY") & (df["days_cover"] > 0)]
    if active_buy.empty:
        return (
            f"{n_buy} procurement action{'s' if n_buy > 1 else ''} required. "
            "Days cover data unavailable — review Procurement Intelligence for details."
        )

    worst    = active_buy.loc[active_buy["days_cover"].idxmin()]
    days_str = f"{worst['days_cover']:.0f}"
    date_str = _stockout_date(worst["days_cover"])
    org_str  = _short_org(str(worst["org_name"]))

    cost_clause = ""
    if exposure.get("market_data_available") and exposure.get("total_cost_usd", 0) > 0:
        cost_clause = f" Estimated procurement cost: {exposure['total_cost_display']}."

    try:
        ice_change = float(
            (market_snapshot or {}).get("ice_cotton", {}).get("change") or 0
        )
    except (TypeError, ValueError):
        ice_change = 0.0

    market_clause = ""
    if ice_change != 0:
        direction    = "up" if ice_change > 0 else "down"
        market_clause = f" ICE Cotton {direction} {abs(ice_change):.1f}% MoM."

    return (
        f"{n_buy} procurement action{'s' if n_buy > 1 else ''} required. "
        f"{org_str} {worst['commodity']} stockout in {days_str} days ({date_str})."
        + cost_clause
        + market_clause
    )


def _alert_class(n_buy: int, n_monitor: int, no_data: bool = False) -> tuple[str, str]:
    """Return (css_class, text_color) for the Situation Brief banner.

    Rules (from design system spec):
      no_data          → 'alert-info'      blue  — neutral/informational
      n_buy > 0        → 'alert-critical'  red   — BUY action required
      n_monitor > 0    → 'alert-monitor'   amber — attention needed
      all HOLD         → 'alert-healthy'   green — all positions secured
    """
    if no_data:
        return "alert-info", "#1d4ed8"
    if n_buy > 0:
        return "alert-critical", "#991b1b"
    if n_monitor > 0:
        return "alert-monitor", "#92400e"
    return "alert-healthy", "#166534"


def _section_header(title: str, subtitle: str, colour: str = "#1e40af") -> None:
    st.markdown(f"""
    <div style='padding:0.85rem 1.1rem;background:#ffffff;border-radius:8px;
                border:1px solid #e2e8f0;border-left:4px solid {colour};
                margin:0 0 1.1rem 0;box-shadow:0 1px 3px rgba(0,0,0,0.05);'>
        <div style='font-size:1.05rem;font-weight:800;color:#0f172a;
                    letter-spacing:-0.3px;margin:0 0 0.15rem 0;'>{title}</div>
        <div style='font-size:0.78rem;color:#64748b;font-weight:500;
                    margin:0;line-height:1.4;'>{subtitle}</div>
    </div>""", unsafe_allow_html=True)


def _chart_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=60, r=20, t=40, b=60),
        plot_bgcolor=_C_BG,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=_C_SUBTEXT),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )
    fig.update_xaxes(showgrid=False, showline=True,
                     linewidth=1, linecolor=_C_GRID, tickfont=dict(size=10))
    fig.update_yaxes(showgrid=True, gridcolor=_C_GRID, gridwidth=1,
                     showline=True, linewidth=1, linecolor=_C_GRID, tickfont=dict(size=10))
    return fig


def _empty_state(message: str) -> None:
    st.markdown(f"""
    <div style='text-align:center;padding:3rem 1rem;color:#94a3b8;'>
        <div style='font-size:2.5rem;margin-bottom:0.75rem;'>—</div>
        <div style='font-size:0.95rem;font-weight:500;'>{message}</div>
    </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 1 — Procurement Overview
# ---------------------------------------------------------------------------

def _render_overview(df: pd.DataFrame, meta: dict) -> None:
    if df.empty:
        _empty_state("No procurement data available. Run the pipeline to generate recommendations.")
        return

    n_buy     = (df["action"] == "BUY").sum()
    n_hold    = (df["action"] == "HOLD").sum()
    n_monitor = (df["action"] == "MONITOR").sum()
    total_proc_kgs = df.loc[df["action"] == "BUY", "procurement_qty"].sum()
    avg_days_cover = df.loc[df["days_cover"] > 0, "days_cover"].mean()
    critical = (
        ((df["days_cover"] < 15) & (df["days_cover"] > 0)).sum()
        if "days_cover" in df.columns else 0
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    cards = [
        (c1, "BUY RECOMMENDATIONS",  str(n_buy),
         "Immediate procurement required",  _C_BUY),
        (c2, "HOLD RECOMMENDATIONS",  str(n_hold),
         "Stock adequate, no action",       _C_HOLD),
        (c3, "MONITOR",               str(n_monitor),
         "Missing data or Cotton Waste",    _C_MONITOR),
        (c4, "TOTAL PROCUREMENT (Kgs)",
         f"{total_proc_kgs:,.0f}",
         "Sum of all BUY quantities",       "#059669"),
        (c5, "AVG DAYS COVER",
         f"{avg_days_cover:.1f}" if avg_days_cover == avg_days_cover else "N/A",
         "Across active org-commodity pairs", "#0891b2"),
        (c6, "CRITICAL (<15 days)",   str(int(critical)),
         "Pairs below 15-day cover threshold", "#dc2626"),
    ]
    for col, label, val, sub, colour in cards:
        with col:
            st.markdown(_kpi_card(label, val, sub, colour), unsafe_allow_html=True)

    # Action distribution pie
    col_pie, col_tbl = st.columns([1, 2])

    with col_pie:
        st.markdown("#### Action Distribution")
        action_counts = df["action"].value_counts().reset_index()
        action_counts.columns = ["Action", "Count"]
        colour_map = {"BUY": _C_BUY, "HOLD": _C_HOLD, "MONITOR": _C_MONITOR}
        fig = px.pie(
            action_counts, names="Action", values="Count",
            color="Action", color_discrete_map=colour_map,
            hole=0.55,
        )
        fig.update_traces(textposition="outside", textinfo="label+percent",
                          textfont_size=11)
        fig.update_layout(
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key="overview_pie")

    with col_tbl:
        st.markdown("#### All Org–Commodity Pairs")
        display = df[["org_name", "commodity", "inventory_qty",
                      "days_cover", "action", "confidence"]].copy()
        display = display.sort_values(
            ["action", "days_cover"],
            ascending=[True, True],
            key=lambda s: s.map({"BUY": 0, "HOLD": 1, "MONITOR": 2})
                if s.name == "action" else s
        )
        display.columns = ["Org", "Commodity", "Inventory (Kgs)",
                            "Days Cover", "Action", "Confidence"]
        display["Inventory (Kgs)"] = display["Inventory (Kgs)"].map("{:,.0f}".format)
        display["Days Cover"] = display["Days Cover"].apply(
            lambda v: f"{v:.1f}" if v > 0 else "N/A"
        )
        st.dataframe(display, use_container_width=True, hide_index=True)

    # Period metadata footer
    if meta.get("period_label"):
        st.caption(
            f"Reporting period: **{meta['period_label']}** · "
            f"Generated: {meta.get('generated', '—')}"
        )


# ---------------------------------------------------------------------------
# Tab 2 — BUY Recommendations
# ---------------------------------------------------------------------------

def _render_buy(df: pd.DataFrame) -> None:
    buy = df[df["action"] == "BUY"].copy() if not df.empty else pd.DataFrame()

    if buy.empty:
        _empty_state("No BUY recommendations this period. All inventory levels are adequate.")
        return

    buy_sorted = buy.sort_values("days_cover", ascending=True)

    # Summary banner
    total_kgs = buy_sorted["procurement_qty"].sum()
    worst_days = buy_sorted["days_cover"].min()
    st.markdown(f"""
    <div class='alert-critical'>
        <span style='font-size:0.83rem;font-weight:700;color:#991b1b;'>
            {len(buy_sorted)} pair{'s' if len(buy_sorted) != 1 else ''} require immediate procurement
            &nbsp;·&nbsp; Total gap: <strong>{total_kgs:,.0f} Kgs</strong>
            &nbsp;·&nbsp; Lowest cover: <strong>{worst_days:.1f} days</strong>
        </span>
    </div>""", unsafe_allow_html=True)

    cols_show = ["org_name", "commodity", "inventory_qty", "need_45_days",
                 "shortfall", "procurement_qty", "days_cover", "confidence"]
    cols_show = [c for c in cols_show if c in buy_sorted.columns]
    display = buy_sorted[cols_show].copy()
    display.columns = [
        c.replace("org_name", "Org")
         .replace("commodity", "Commodity")
         .replace("inventory_qty", "Inventory (Kgs)")
         .replace("need_45_days", "45-Day Need (Kgs)")
         .replace("shortfall", "Shortfall (Kgs)")
         .replace("procurement_qty", "Procurement Qty (Kgs)")
         .replace("days_cover", "Days Cover")
         .replace("confidence", "Confidence")
        for c in cols_show
    ]
    for num_col in ["Inventory (Kgs)", "45-Day Need (Kgs)",
                    "Shortfall (Kgs)", "Procurement Qty (Kgs)"]:
        if num_col in display.columns:
            display[num_col] = display[num_col].map("{:,.0f}".format)
    if "Days Cover" in display.columns:
        display["Days Cover"] = display["Days Cover"].apply(
            lambda v: f"{float(v):.1f}" if str(v).replace(".", "").isdigit() else v
        )

    def _row_style(row):
        try:
            dc = float(str(row.get("Days Cover", 99)).replace(",", ""))
        except ValueError:
            dc = 99
        if dc < 7:
            return ["background:#fff1f2"] * len(row)
        if dc < 15:
            return ["background:#fffbeb"] * len(row)
        return [""] * len(row)

    styled = display.style.apply(_row_style, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.caption("Rows highlighted red = under 7 days cover · amber = under 15 days cover")

    # Shortfall bar chart
    st.markdown("#### Shortfall by Org–Commodity")
    buy_sorted["label"] = buy_sorted["org_name"] + " / " + buy_sorted["commodity"]
    fig = go.Figure(go.Bar(
        x=buy_sorted["label"],
        y=buy_sorted["shortfall"],
        marker_color=_C_BUY,
        text=buy_sorted["shortfall"].map("{:,.0f}".format),
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="Org / Commodity",
        yaxis_title="Shortfall (Kgs)",
        xaxis_tickangle=-35,
    )
    _chart_layout(fig)
    st.plotly_chart(fig, use_container_width=True, key="buy_shortfall_bar")


# ---------------------------------------------------------------------------
# Tab 3 — HOLD Recommendations
# ---------------------------------------------------------------------------

def _render_hold(df: pd.DataFrame) -> None:
    hold = df[df["action"] == "HOLD"].copy() if not df.empty else pd.DataFrame()

    if hold.empty:
        _empty_state("No HOLD recommendations this period.")
        return

    st.markdown(f"""
    <div class='alert-info'>
        <span style='font-size:0.83rem;font-weight:700;color:#1d4ed8;'>
            {len(hold)} org–commodity pair{'s' if len(hold) != 1 else ''} have adequate stock.
            No procurement action required this period.
        </span>
    </div>""", unsafe_allow_html=True)

    cols_show = ["org_name", "commodity", "inventory_qty", "monthly_consumption",
                 "need_45_days", "days_cover", "confidence"]
    cols_show = [c for c in cols_show if c in hold.columns]
    display = hold.sort_values("days_cover", ascending=False)[cols_show].copy()
    display.columns = [
        c.replace("org_name", "Org")
         .replace("commodity", "Commodity")
         .replace("inventory_qty", "Inventory (Kgs)")
         .replace("monthly_consumption", "Monthly Consumption (Kgs)")
         .replace("need_45_days", "45-Day Need (Kgs)")
         .replace("days_cover", "Days Cover")
         .replace("confidence", "Confidence")
        for c in cols_show
    ]
    for num_col in ["Inventory (Kgs)", "Monthly Consumption (Kgs)", "45-Day Need (Kgs)"]:
        if num_col in display.columns:
            display[num_col] = display[num_col].map("{:,.0f}".format)
    if "Days Cover" in display.columns:
        display["Days Cover"] = display["Days Cover"].apply(
            lambda v: f"{float(v):.1f}" if str(v).replace(".", "").isdigit() else v
        )

    st.dataframe(display, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 4 — MONITOR Recommendations
# ---------------------------------------------------------------------------

def _monitor_reason(row: pd.Series) -> str:
    """Derive human-readable MONITOR reason from row values."""
    commodity = str(row.get("commodity", "")).strip()
    if commodity == "Cotton Waste":
        return "Cotton Waste — MONITOR always"
    inv_qty  = float(row.get("inventory_qty", 0) or 0)
    monthly  = float(row.get("monthly_consumption", 0) or 0)
    if monthly <= 0:
        return "Missing Consumption Data"
    if inv_qty <= 0:
        return "Missing Inventory"
    return "Net Consumption <= 0"


def _render_monitor(df: pd.DataFrame) -> None:
    monitor = df[df["action"] == "MONITOR"].copy() if not df.empty else pd.DataFrame()

    if monitor.empty:
        _empty_state("No MONITOR items this period. All pairs have valid consumption data.")
        return

    st.markdown(f"""
    <div class='alert-monitor'>
        <span style='font-size:0.83rem;font-weight:700;color:#92400e;'>
            {len(monitor)} org–commodity pair{'s' if len(monitor) != 1 else ''} set to MONITOR.
            No procurement recommendation is generated — consumption data absent or Cotton Waste.
        </span>
    </div>""", unsafe_allow_html=True)

    monitor["Reason"] = monitor.apply(_monitor_reason, axis=1)

    # Group by reason for clarity
    for reason in ["Missing Consumption Data", "Cotton Waste — MONITOR always",
                   "Missing Inventory", "Net Consumption <= 0"]:
        sub = monitor[monitor["Reason"] == reason]
        if sub.empty:
            continue
        st.markdown(f"**{reason}** ({len(sub)} pair{'s' if len(sub) > 1 else ''})")
        cols_show = ["org_name", "commodity", "inventory_qty",
                     "monthly_consumption", "confidence", "Reason"]
        cols_show = [c for c in cols_show if c in sub.columns]
        display = sub[cols_show].copy()
        display.columns = [
            c.replace("org_name", "Org")
             .replace("commodity", "Commodity")
             .replace("inventory_qty", "Inventory (Kgs)")
             .replace("monthly_consumption", "Monthly Consumption (Kgs)")
             .replace("confidence", "Confidence")
            for c in cols_show
        ]
        for num_col in ["Inventory (Kgs)", "Monthly Consumption (Kgs)"]:
            if num_col in display.columns:
                display[num_col] = display[num_col].map("{:,.0f}".format)
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 5 — Inventory Risk
# ---------------------------------------------------------------------------

def _render_risk(df: pd.DataFrame) -> None:
    if df.empty:
        _empty_state("No data available for risk analysis.")
        return

    active = df[df["days_cover"] > 0].copy()

    # ── Days Cover by Commodity ────────────────────────────────────────────
    st.markdown("#### Days Cover by Commodity")
    if not active.empty:
        by_commodity = (
            active.groupby("commodity", as_index=False)["days_cover"]
            .mean()
            .sort_values("days_cover")
        )
        colour_list = [
            _C_BUY if v < 15 else (_C_MONITOR if v < 30 else _C_HOLD)
            for v in by_commodity["days_cover"]
        ]
        fig = go.Figure(go.Bar(
            x=by_commodity["commodity"],
            y=by_commodity["days_cover"],
            marker_color=colour_list,
            text=by_commodity["days_cover"].map("{:.1f}".format),
            textposition="outside",
        ))
        fig.add_hline(y=45, line_dash="dot", line_color="#94a3b8",
                      annotation_text="45-day target", annotation_position="right")
        fig.add_hline(y=15, line_dash="dot", line_color=_C_BUY,
                      annotation_text="15-day critical", annotation_position="right")
        fig.update_layout(xaxis_title="Commodity", yaxis_title="Days Cover (avg)")
        _chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True, key="risk_days_commodity")
    else:
        _empty_state("No active pairs with days-cover data.")

    # ── Days Cover by Organisation ─────────────────────────────────────────
    st.markdown("#### Days Cover by Organisation")
    if not active.empty:
        by_org = (
            active.groupby("org_name", as_index=False)["days_cover"]
            .min()
            .sort_values("days_cover")
            .rename(columns={"org_name": "Org", "days_cover": "Min Days Cover"})
        )
        colour_list_org = [
            _C_BUY if v < 15 else (_C_MONITOR if v < 30 else _C_HOLD)
            for v in by_org["Min Days Cover"]
        ]
        fig2 = go.Figure(go.Bar(
            x=by_org["Org"],
            y=by_org["Min Days Cover"],
            marker_color=colour_list_org,
            text=by_org["Min Days Cover"].map("{:.1f}".format),
            textposition="outside",
        ))
        fig2.add_hline(y=45, line_dash="dot", line_color="#94a3b8",
                       annotation_text="45-day target", annotation_position="right")
        fig2.update_layout(
            xaxis_title="Organisation", yaxis_title="Minimum Days Cover",
            xaxis_tickangle=-30,
        )
        _chart_layout(fig2)
        st.plotly_chart(fig2, use_container_width=True, key="risk_days_org")

    col_short, col_alert = st.columns(2)

    # ── Top Shortfalls ─────────────────────────────────────────────────────
    with col_short:
        st.markdown("#### Top Shortfalls")
        shortfall_df = df[df["shortfall"] > 0].copy()
        if shortfall_df.empty:
            _empty_state("No shortfalls recorded.")
        else:
            top = shortfall_df.nlargest(8, "shortfall")
            top["label"] = top["org_name"] + "\n" + top["commodity"]
            fig3 = go.Figure(go.Bar(
                x=top["shortfall"],
                y=top["label"],
                orientation="h",
                marker_color=_C_BUY,
                text=top["shortfall"].map("{:,.0f} Kgs".format),
                textposition="outside",
            ))
            fig3.update_layout(
                xaxis_title="Shortfall (Kgs)",
                yaxis_title="",
                yaxis=dict(autorange="reversed"),
            )
            _chart_layout(fig3, height=320)
            st.plotly_chart(fig3, use_container_width=True, key="risk_shortfall_h")

    # ── Critical Inventory Alerts ──────────────────────────────────────────
    with col_alert:
        st.markdown("#### Critical Inventory Alerts")
        critical = df[df["days_cover"] < 15].copy() if "days_cover" in df.columns else pd.DataFrame()
        critical = critical[critical["days_cover"] > 0]  # exclude MONITOR zeros
        if critical.empty:
            st.markdown("""
            <div style='background:#f0fdf4;border:1.5px solid #86efac;border-radius:8px;
                        padding:1.5rem;text-align:center;margin-top:0.5rem;'>
                <div style='font-size:1.5rem;margin-bottom:0.4rem;'>OK</div>
                <div style='font-size:0.85rem;color:#166534;font-weight:600;'>
                    No pairs below 15-day cover threshold.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            for _, row in critical.iterrows():
                dc = row["days_cover"]
                colour = "#fef2f2" if dc < 7 else "#fffbeb"
                border  = "#fca5a5" if dc < 7 else "#fde68a"
                txt_col = "#991b1b" if dc < 7 else "#92400e"
                st.markdown(f"""
                <div style='background:{colour};border:1.5px solid {border};
                            border-radius:8px;padding:0.6rem 1rem;margin-bottom:0.5rem;'>
                    <span style='font-size:0.8rem;font-weight:700;color:{txt_col};'>
                        {row["org_name"]} / {row["commodity"]}
                    </span>
                    <span style='float:right;font-size:0.8rem;font-weight:800;color:{txt_col};'>
                        {dc:.1f} days
                    </span>
                </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 6 — Procurement Summary (exportable)
# ---------------------------------------------------------------------------

def _render_summary(df: pd.DataFrame, meta: dict) -> None:
    if df.empty:
        _empty_state("No procurement data available.")
        return

    period = meta.get("period_label", "Unknown period")
    st.caption(f"Reporting period: **{period}** · Download CSV for distribution")

    cols_export = [
        "org_name", "commodity", "inventory_qty", "monthly_consumption",
        "need_45_days", "procurement_qty", "action", "confidence",
    ]
    cols_export = [c for c in cols_export if c in df.columns]
    export_df = df[cols_export].copy()
    export_df.columns = [
        c.replace("org_name", "Org")
         .replace("commodity", "Commodity")
         .replace("inventory_qty", "Current Inventory (Kgs)")
         .replace("monthly_consumption", "Monthly Consumption (Kgs)")
         .replace("need_45_days", "45-Day Need (Kgs)")
         .replace("procurement_qty", "Procurement Qty (Kgs)")
         .replace("action", "Action")
         .replace("confidence", "Confidence")
        for c in cols_export
    ]
    for num_col in ["Current Inventory (Kgs)", "Monthly Consumption (Kgs)",
                    "45-Day Need (Kgs)", "Procurement Qty (Kgs)"]:
        if num_col in export_df.columns:
            export_df[num_col] = export_df[num_col].map("{:,.0f}".format)

    def _highlight_action(row):
        action = row.get("Action", "")
        if action == "BUY":
            return ["background:#fef2f2"] * len(row)
        if action == "HOLD":
            return ["background:#eff6ff"] * len(row)
        if action == "MONITOR":
            return ["background:#fffbeb"] * len(row)
        return [""] * len(row)

    styled = export_df.style.apply(_highlight_action, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Download button
    csv_bytes = df[cols_export].to_csv(index=False).encode("utf-8")
    filename = f"procurement_strategy_{meta.get('year', 'YYYY')}_{meta.get('month', 'MM'):02d}.csv" \
        if meta.get("year") else "procurement_strategy.csv"
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        key="download_strategy_csv",
    )


# ---------------------------------------------------------------------------
# Executive Summary — procurement header (5 sections)
# ---------------------------------------------------------------------------

def _insight_bullets(df: pd.DataFrame) -> list[str]:
    """Generate automatic executive insight sentences from strategy_df."""
    if df.empty:
        return ["No procurement data available for this period."]

    bullets: list[str] = []
    buy = df[df["action"] == "BUY"]
    monitor = df[df["action"] == "MONITOR"]

    # Policy stock breach
    n_buy = len(buy)
    if n_buy == 0:
        bullets.append("All org-commodity pairs meet the 45-day policy stock requirement.")
    else:
        bullets.append(
            f"{n_buy} org-commodity pair{'s' if n_buy > 1 else ''} "
            f"{'are' if n_buy > 1 else 'is'} below the 45-day policy stock threshold."
        )

    # Cotton gap
    cotton_buy = buy[buy["commodity"] == "Cotton"]
    cotton_gap = cotton_buy["shortfall"].sum()
    n_cotton_buy = len(cotton_buy)
    if n_cotton_buy > 0:
        bullets.append(
            f"Cotton procurement required for {n_cotton_buy} unit{'s' if n_cotton_buy > 1 else ''}. "
            f"Total gap: {cotton_gap:,.0f} Kgs."
        )

    # Fiber gap
    fiber_buy = buy[buy["commodity"] == "Fiber"]
    fiber_gap = fiber_buy["shortfall"].sum()
    n_fiber_buy = len(fiber_buy)
    if n_fiber_buy > 0:
        bullets.append(
            f"Fiber procurement required for {n_fiber_buy} unit{'s' if n_fiber_buy > 1 else ''}. "
            f"Total gap: {fiber_gap:,.0f} Kgs."
        )

    # Stretch Fiber gap
    sf_buy = buy[buy["commodity"] == "Stretch Fiber"]
    if not sf_buy.empty:
        bullets.append(
            f"Stretch Fiber shortfall identified at {len(sf_buy)} "
            f"unit{'s' if len(sf_buy) > 1 else ''} ({sf_buy['shortfall'].sum():,.0f} Kgs)."
        )

    # Worst case (lowest days cover among BUY rows)
    active_buy = buy[buy["days_cover"] > 0]
    if not active_buy.empty:
        worst = active_buy.loc[active_buy["days_cover"].idxmin()]
        bullets.append(
            f"Most critical: {worst['org_name']} / {worst['commodity']} "
            f"has only {worst['days_cover']:.1f} days of cover remaining."
        )

    # MONITOR items
    n_mon = len(monitor)
    if n_mon > 0:
        cotton_waste_mon = (monitor["commodity"] == "Cotton Waste").sum()
        other_mon = n_mon - cotton_waste_mon
        if other_mon > 0:
            bullets.append(
                f"{other_mon} org-commodity pair{'s' if other_mon > 1 else ''} "
                f"flagged MONITOR — consumption data absent or returns exceed issues."
            )
        if cotton_waste_mon > 0:
            bullets.append(f"Cotton Waste tracked at {int(cotton_waste_mon)} unit{'s' if cotton_waste_mon > 1 else ''} (inventory only, no procurement action).")

    # Positive signal when all HOLD or better
    if n_buy == 0 and n_mon == 0:
        bullets.append("Inventory position is strong across all tracked commodities.")

    return bullets[:6]  # cap at 6 bullets


def render_exec_procurement_header(
    df: pd.DataFrame,
    meta: dict,
    market_snapshot: dict,
) -> None:
    """Render the 5-section procurement management header for the Executive Summary.

    Sections:
        1. Procurement Status   — 6 KPI cards
        2. Critical Risks       — Top-10 shortfall table sorted by days cover
        3. Market Snapshot      — ICE Cotton, PSF, USD/PKR cards
        4. Procurement Recommendation — BUY/HOLD/MONITOR counts + pie
        5. Executive Insights   — auto-generated bullet insights

    Args:
        df:              strategy_df loaded by load_procurement_strategy().
                         Pass pd.DataFrame() when no data is available.
        meta:            metadata dict (period_label, year, month, generated).
        market_snapshot: dict with keys "ice_cotton", "psf", "usd_pkr" each
                         containing {"price", "change", "date", "currency"}.
    """

    # ── period badge ──────────────────────────────────────────────────────────
    period_label = meta.get("period_label", "")
    generated    = meta.get("generated", "")
    if period_label:
        st.markdown(
            f"<span style='background:#eff6ff;color:#1d4ed8;border:1.5px solid #bfdbfe;"
            f"border-radius:12px;padding:3px 14px;font-size:0.78rem;font-weight:700;"
            f"letter-spacing:0.3px;'>Procurement period: {period_label}</span>"
            + (f"&nbsp;&nbsp;<span style='font-size:0.75rem;color:#94a3b8;'>"
               f"Updated {generated}</span>" if generated else ""),
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    no_data = df.empty

    # ── helpers ──────────────────────────────────────────────────────────────
    def _agg(commodity: str, col: str) -> float:
        if no_data:
            return 0.0
        sub = df[df["commodity"] == commodity]
        return float(sub[col].sum()) if not sub.empty else 0.0

    def _snap_card(label: str, price, change, date: str, currency: str, key: str) -> None:
        if price is None or price == 0:
            st.markdown(f"""
            <div class='metric-card' style='border-left:4px solid #94a3b8;opacity:0.6;'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value' style='font-size:1.2rem;color:#94a3b8;'>N/A</div>
                <div class='currency-label'>Data unavailable</div>
            </div>""", unsafe_allow_html=True)
            return
        try:
            chg = float(change or 0)
            arrow = "↑" if chg > 0 else ("↓" if chg < 0 else "→")
            trend_col = _C_BUY if chg > 2 else ("#059669" if chg < -2 else "#2563eb")
            st.markdown(f"""
            <div class='metric-card' style='border-left:4px solid {trend_col};'>
                <div class='metric-label' style='color:{trend_col};'>{label}</div>
                <div class='metric-value' style='color:{_C_TEXT};'>{float(price):.4f}</div>
                <div class='currency-label'>
                    <strong>{currency}</strong><br>
                    <span style='color:{trend_col};font-weight:700;'>{arrow} {abs(chg):.2f}% MoM</span><br>
                    <span style='font-size:0.68rem;color:#94a3b8;'>{date or ""}</span>
                </div>
            </div>""", unsafe_allow_html=True)
        except Exception:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{price}</div>
                <div class='currency-label'>{currency}</div>
            </div>""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — PROCUREMENT STATUS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='exec-section-bar'>
        <span class='exec-section-label exec-section-label-hold'>1 — Procurement Status</span>
    </div>""", unsafe_allow_html=True)

    cotton_inv   = _agg("Cotton",       "inventory_qty")
    fiber_inv    = _agg("Fiber",         "inventory_qty")
    cotton_need  = _agg("Cotton",       "need_45_days")
    fiber_need   = _agg("Fiber",         "need_45_days")
    total_gap    = float(df.loc[df["action"] == "BUY", "shortfall"].sum()) if not no_data else 0.0
    n_buy        = int((df["action"] == "BUY").sum()) if not no_data else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    s1_cards = [
        (c1, "COTTON INVENTORY",    f"{cotton_inv:,.0f}",  "Kgs on hand",              "#2563eb"),
        (c2, "FIBER INVENTORY",     f"{fiber_inv:,.0f}",   "Kgs on hand",              _C_HOLD),
        (c3, "45-DAY COTTON NEED",  f"{cotton_need:,.0f}", "Kgs (policy requirement)", "#0891b2"),
        (c4, "45-DAY FIBER NEED",   f"{fiber_need:,.0f}",  "Kgs (policy requirement)", "#0891b2"),
        (c5, "TOTAL PROCUREMENT GAP",f"{total_gap:,.0f}",  "Kgs across all BUY rows", "#dc2626"),
        (c6, "BUY RECOMMENDATIONS", str(n_buy),             "Pairs requiring action",   "#dc2626"),
    ]
    for col, label, val, sub, colour in s1_cards:
        with col:
            st.markdown(_kpi_card(label, val, sub, colour), unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — CRITICAL RISKS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='exec-section-bar'>
        <span class='exec-section-label exec-section-label-buy'>2 — Critical Risks</span>
    </div>""", unsafe_allow_html=True)

    if no_data or df[df["shortfall"] > 0].empty:
        st.markdown("""
        <div style='background:#f0fdf4;border:1.5px solid #86efac;border-radius:8px;
                    padding:1rem 1.25rem;'>
            <span style='font-size:0.85rem;font-weight:700;color:#166534;'>
                No shortfalls recorded. All inventory positions meet policy requirements.
            </span>
        </div>""", unsafe_allow_html=True)
    else:
        risk_df = df[df["shortfall"] > 0].nlargest(10, "shortfall").sort_values("days_cover")
        risk_show = risk_df[
            ["org_name", "commodity", "inventory_qty",
             "need_45_days", "shortfall", "days_cover", "action"]
        ].copy()
        risk_show.columns = ["Org", "Commodity", "Current Inventory (Kgs)",
                             "45-Day Need (Kgs)", "Shortfall (Kgs)", "Days Cover", "Action"]
        for nc in ["Current Inventory (Kgs)", "45-Day Need (Kgs)", "Shortfall (Kgs)"]:
            risk_show[nc] = risk_show[nc].map("{:,.0f}".format)
        risk_show["Days Cover"] = risk_show["Days Cover"].apply(
            lambda v: f"{float(v):.1f}" if str(v).replace(".", "").isdigit() and float(v) > 0 else "N/A"
        )

        def _risk_row_style(row):
            try:
                dc_str = str(row.get("Days Cover", "N/A"))
                dc = float(dc_str) if dc_str not in ("N/A", "") else 99.0
            except ValueError:
                dc = 99.0
            if dc < 7:
                return ["background:#fff1f2"] * len(row)
            if dc < 15:
                return ["background:#fffbeb"] * len(row)
            return [""] * len(row)

        st.dataframe(
            risk_show.style.apply(_risk_row_style, axis=1),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Sorted by lowest days cover. Red = under 7 days · Amber = under 15 days. Top 10 highest shortfalls shown.")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — MARKET SNAPSHOT
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='exec-section-bar'>
        <span class='exec-section-label'>3 — Market Snapshot</span>
    </div>""", unsafe_allow_html=True)

    mc1, mc2, mc3 = st.columns(3)
    ice   = market_snapshot.get("ice_cotton", {})
    psf   = market_snapshot.get("psf", {})
    usdpk = market_snapshot.get("usd_pkr", {})

    with mc1:
        _snap_card("ICE COTTON NO. 2",
                   ice.get("price"), ice.get("change"), ice.get("date"),
                   ice.get("currency", "USD/lb"), "snap_ice")
    with mc2:
        _snap_card("PSF (POLYESTER)",
                   psf.get("price"), psf.get("change"), psf.get("date"),
                   psf.get("currency", "USD/kg"), "snap_psf")
    with mc3:
        usd_price = usdpk.get("price")
        usd_label = f"{usd_price:,.2f}" if usd_price else "N/A"
        chg_usdpk = usdpk.get("change", 0) or 0
        arrow_usd = "↑" if chg_usdpk > 0 else ("↓" if chg_usdpk < 0 else "→")
        trend_usd = _C_BUY if chg_usdpk > 0 else ("#059669" if chg_usdpk < 0 else "#2563eb")
        st.markdown(f"""
        <div class='metric-card' style='border-left:4px solid {trend_usd};'>
            <div class='metric-label' style='color:{trend_usd};'>USD / PKR</div>
            <div class='metric-value' style='color:{_C_TEXT};'>{usd_label}</div>
            <div class='currency-label'>
                <strong>PKR per USD</strong><br>
                <span style='color:{trend_usd};font-weight:700;'>
                    {arrow_usd} {abs(float(chg_usdpk)):.2f}% MoM
                </span><br>
                <span style='font-size:0.68rem;color:#94a3b8;'>
                    {usdpk.get("date", "") or "Live rate"}
                </span>
            </div>
        </div>""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4 — PROCUREMENT RECOMMENDATION
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='exec-section-bar'>
        <span class='exec-section-label'>4 — Procurement Recommendation</span>
    </div>""", unsafe_allow_html=True)

    rc1, rc2 = st.columns([1, 2])

    if no_data:
        with rc1:
            _empty_state("No data")
        with rc2:
            _empty_state("Run the pipeline first")
    else:
        n_hold    = int((df["action"] == "HOLD").sum())
        n_monitor = int((df["action"] == "MONITOR").sum())
        total_pairs = len(df)

        def _pct(n: int) -> str:
            return f"{n / total_pairs * 100:.0f}%" if total_pairs > 0 else "—"

        with rc1:
            for label, n, sub_txt, colour in [
                ("BUY NOW",  n_buy,     f"{_pct(n_buy)} of tracked pairs",    _C_BUY),
                ("HOLD",     n_hold,    f"{_pct(n_hold)} — stock adequate",    _C_HOLD),
                ("MONITOR",  n_monitor, f"{_pct(n_monitor)} — insufficient data", _C_MONITOR),
            ]:
                st.markdown(_kpi_card(label, str(n), sub_txt, colour),
                            unsafe_allow_html=True)

        with rc2:
            action_counts = df["action"].value_counts().reset_index()
            action_counts.columns = ["Action", "Count"]
            colour_map = {"BUY": _C_BUY, "HOLD": _C_HOLD, "MONITOR": _C_MONITOR}
            import plotly.express as _px
            fig = _px.pie(
                action_counts, names="Action", values="Count",
                color="Action", color_discrete_map=colour_map, hole=0.6,
            )
            fig.update_traces(textposition="outside", textinfo="label+value+percent",
                              textfont_size=11)
            fig.update_layout(
                height=280, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True,
                            key="exec_recommendation_pie")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5 — EXECUTIVE INSIGHTS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='exec-section-bar'>
        <span class='exec-section-label exec-section-label-monitor'>5 — Executive Insights</span>
    </div>""", unsafe_allow_html=True)

    bullets = _insight_bullets(df)
    bullet_html = "".join(
        f"<li style='margin-bottom:0.6rem;font-size:0.88rem;font-weight:600;"
        f"color:#1e293b;line-height:1.5;'>{b}</li>"
        for b in bullets
    )
    st.markdown(
        f"<ul style='padding-left:1.25rem;margin:0;'>{bullet_html}</ul>",
        unsafe_allow_html=True,
    )

    # ── divider before existing forecast content ──────────────────────────────
    st.markdown("""
    <div class='exec-section-bar' style='margin-top:2rem;'>
        <span class='exec-section-label'>Market Forecasts — Commodity-by-Commodity Analysis</span>
    </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UX-3A — Executive Summary header v2  (Steps 1–3 implemented)
# ---------------------------------------------------------------------------

def render_exec_procurement_header_v2(
    df: pd.DataFrame,
    meta: dict,
    market_snapshot: dict,
) -> None:
    """Executive Summary procurement header — UX-3A partial implementation.

    Sections implemented in this version:
        A  — Situation Brief            (UX-3A Step 1)
        B  — Status Tiles × 5           (UX-3A Step 2)
        E  — Financial Exposure         (UX-3A Step 3)

    Sections carried forward unchanged from v1:
        2  — Critical Risks
        3  — Market Snapshot
        4  — Procurement Recommendation
        5  — Executive Insights

    No business logic is changed. BUY/HOLD/MONITOR decisions are read
    from strategy_df exactly as loaded by load_procurement_strategy().
    """

    # ── period badge ──────────────────────────────────────────────────────────
    period_label = meta.get("period_label", "")
    generated    = meta.get("generated", "")
    if period_label:
        st.markdown(
            f"<span style='background:#eff6ff;color:#1d4ed8;border:1.5px solid #bfdbfe;"
            f"border-radius:12px;padding:3px 14px;font-size:0.78rem;font-weight:700;"
            f"letter-spacing:0.3px;'>Procurement period: {period_label}</span>"
            + (f"&nbsp;&nbsp;<span style='font-size:0.75rem;color:#94a3b8;'>"
               f"Updated {generated}</span>" if generated else ""),
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    no_data     = df.empty
    n_buy       = int((df["action"] == "BUY").sum())     if not no_data else 0
    n_hold      = int((df["action"] == "HOLD").sum())    if not no_data else 0
    n_monitor   = int((df["action"] == "MONITOR").sum()) if not no_data else 0
    total_pairs = len(df) if not no_data else 0

    # Financial exposure computed once; used by Section A and Section E
    exposure = _compute_financial_exposure(df, market_snapshot)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION A — SITUATION BRIEF  (Step 1)
    # Alert class is derived from actual portfolio state — not hardcoded red.
    # ─────────────────────────────────────────────────────────────────────────
    brief_text = _build_situation_brief(df, exposure, market_snapshot)
    _brief_class, _brief_color = _alert_class(n_buy, n_monitor, no_data)
    st.markdown(
        f"<div class='{_brief_class}' style='padding:0.75rem 1.1rem;margin-bottom:0.8rem;'>"
        f"<span style='font-size:0.875rem;font-weight:600;color:{_brief_color};line-height:1.6;'>"
        f"{brief_text}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION B — STATUS TILES × 5  (Step 2)
    # Replaces the previous 6-tile "1 — Procurement Status" row.
    # Cotton Inventory / Fiber Inventory standalone tiles removed;
    # on-hand Kgs are shown in Section E (Financial Exposure) only.
    # ─────────────────────────────────────────────────────────────────────────
    b1, b2, b3, b4, b5 = st.columns(5)

    with b1:
        val_b1    = str(n_buy) if not no_data else "—"
        border_b1 = _C_BUY if n_buy > 0 else "#059669"
        sub_b1    = "BUY recommendations" if n_buy > 0 else "No action required"
        st.markdown(_kpi_card("ACTIONS REQUIRED", val_b1, sub_b1, border_b1),
                    unsafe_allow_html=True)

    with b2:
        if not no_data and n_buy > 0:
            active_buy = df[(df["action"] == "BUY") & (df["days_cover"] > 0)]
            if not active_buy.empty:
                worst      = active_buy.loc[active_buy["days_cover"].idxmin()]
                worst_days = float(worst["days_cover"])
                border_b2  = (_C_BUY if worst_days < 15
                               else (_C_MONITOR if worst_days < 30 else _C_HOLD))
                sub_b2 = (f"{_short_org(str(worst['org_name']))} / {worst['commodity']}"
                          f"  ·  {_stockout_date(worst_days)}")
                st.markdown(_kpi_card("LOWEST COVER",
                                      f"{worst_days:.1f} days", sub_b2, border_b2),
                            unsafe_allow_html=True)
            else:
                st.markdown(_kpi_card("LOWEST COVER", "—",
                                      "No active BUY rows", "#94a3b8"),
                            unsafe_allow_html=True)
        else:
            st.markdown(_kpi_card("LOWEST COVER", "—",
                                  "All positions adequate", "#059669"),
                        unsafe_allow_html=True)

    with b3:
        val_b3 = exposure["total_cost_display"] if exposure["market_data_available"] else "—"
        sub_b3 = (
            f"Cotton {exposure['cotton_cost_display']} + "
            f"Fiber {exposure['fiber_cost_display']}"
            if exposure["market_data_available"] else "Market data unavailable"
        )
        # Color: red only when there are BUY rows AND a cost estimate exists.
        # Muted grey when no market data; neutral blue when all HOLD (no cost).
        _cost_has_value = exposure["market_data_available"] and exposure.get("total_cost_usd", 0) > 0
        border_b3 = _C_BUY if (n_buy > 0 and _cost_has_value) else (
            _C_HOLD if _cost_has_value else "#94a3b8"
        )
        st.markdown(_kpi_card("EST. PROCUREMENT COST", val_b3, sub_b3, border_b3),
                    unsafe_allow_html=True)

    with b4:
        cotton_buy = (
            df[(df["commodity"] == "Cotton") & (df["action"] == "BUY") & (df["days_cover"] > 0)]
            if not no_data else pd.DataFrame()
        )
        if not cotton_buy.empty:
            wc     = float(cotton_buy["days_cover"].min())
            wc_org = _short_org(
                str(cotton_buy.loc[cotton_buy["days_cover"].idxmin(), "org_name"])
            )
            border_b4 = _C_BUY if wc < 15 else (_C_MONITOR if wc < 30 else _C_HOLD)
            st.markdown(_kpi_card("COTTON COVER", f"{wc:.1f} days",
                                  f"Worst: {wc_org}", border_b4),
                        unsafe_allow_html=True)
        else:
            st.markdown(_kpi_card("COTTON COVER", "—",
                                  "No cotton BUY rows", "#94a3b8"),
                        unsafe_allow_html=True)

    with b5:
        fiber_buy = (
            df[(df["commodity"] == "Fiber") & (df["action"] == "BUY") & (df["days_cover"] > 0)]
            if not no_data else pd.DataFrame()
        )
        if not fiber_buy.empty:
            wf     = float(fiber_buy["days_cover"].min())
            wf_org = _short_org(
                str(fiber_buy.loc[fiber_buy["days_cover"].idxmin(), "org_name"])
            )
            border_b5 = _C_BUY if wf < 15 else (_C_MONITOR if wf < 30 else _C_HOLD)
            st.markdown(_kpi_card("FIBER COVER", f"{wf:.1f} days",
                                  f"Worst: {wf_org}", border_b5),
                        unsafe_allow_html=True)
        else:
            st.markdown(_kpi_card("FIBER COVER", "—",
                                  "No fiber BUY rows", "#94a3b8"),
                        unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — CRITICAL RISKS  (unchanged from v1)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='exec-section-bar'>
        <span class='exec-section-label exec-section-label-buy'>2 — Critical Risks</span>
    </div>""", unsafe_allow_html=True)

    _has_shortfall = (
        not no_data
        and "shortfall" in df.columns
        and not df[df["shortfall"] > 0].empty
    )
    if not _has_shortfall:
        st.markdown("""
        <div class='alert-healthy' style='padding:0.7rem 1rem;'>
            <span style='font-size:0.8125rem;font-weight:600;color:#166534;'>
                No shortfalls recorded. All inventory positions meet the 45-day policy requirement.
            </span>
        </div>""", unsafe_allow_html=True)
    else:
        # Top 5 by shortfall magnitude, sorted by urgency (lowest days cover first)
        risk_df   = df[df["shortfall"] > 0].nlargest(5, "shortfall").sort_values("days_cover")
        risk_show = risk_df[
            ["org_name", "commodity", "shortfall", "days_cover", "action"]
        ].copy()
        risk_show.columns = ["Org", "Commodity", "Shortfall (Kgs)", "Days Cover", "Action"]
        risk_show["Shortfall (Kgs)"] = risk_show["Shortfall (Kgs)"].map("{:,.0f}".format)
        risk_show["Days Cover"] = risk_show["Days Cover"].apply(
            lambda v: f"{float(v):.1f}"
            if str(v).replace(".", "").isdigit() and float(v) > 0 else "N/A"
        )

        def _risk_row_style(row):
            try:
                dc_str = str(row.get("Days Cover", "N/A"))
                dc = float(dc_str) if dc_str not in ("N/A", "") else 99.0
            except ValueError:
                dc = 99.0
            if dc < 7:
                return ["background:#fff1f2"] * len(row)
            if dc < 15:
                return ["background:#fffbeb"] * len(row)
            return [""] * len(row)

        st.dataframe(
            risk_show.style.apply(_risk_row_style, axis=1),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Top 5 shortfalls · sorted by urgency (lowest days cover first) · "
            "red < 7 days · amber < 15 days"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — MARKET SNAPSHOT  (unchanged from v1)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='exec-section-bar'>
        <span class='exec-section-label'>3 — Market Snapshot</span>
    </div>""", unsafe_allow_html=True)

    mc1, mc2, mc3 = st.columns(3)
    ice   = market_snapshot.get("ice_cotton", {}) if market_snapshot else {}
    psf   = market_snapshot.get("psf", {})        if market_snapshot else {}
    usdpk = market_snapshot.get("usd_pkr", {})    if market_snapshot else {}

    with mc1:
        _snap_card_module(
            "ICE COTTON NO. 2",
            ice.get("price"), ice.get("change"), ice.get("date"),
            ice.get("currency", "USD/lb"),
        )
    with mc2:
        _snap_card_module(
            "PSF (POLYESTER)",
            psf.get("price"), psf.get("change"), psf.get("date"),
            psf.get("currency", "USD/kg"),
        )
    with mc3:
        usd_price = usdpk.get("price")
        usd_label = f"{usd_price:,.2f}" if usd_price else "N/A"
        try:
            chg_usdpk = float(usdpk.get("change") or 0)
        except (TypeError, ValueError):
            chg_usdpk = 0.0
        arrow_usd = "↑" if chg_usdpk > 0 else ("↓" if chg_usdpk < 0 else "→")
        trend_usd = _C_BUY if chg_usdpk > 0 else ("#059669" if chg_usdpk < 0 else "#2563eb")
        st.markdown(f"""
        <div class='metric-card' style='border-left:4px solid {trend_usd};'>
            <div class='metric-label' style='color:{trend_usd};'>USD / PKR</div>
            <div class='metric-value' style='color:{_C_TEXT};'>{usd_label}</div>
            <div class='currency-label'>
                <strong>PKR per USD</strong><br>
                <span style='color:{trend_usd};font-weight:700;'>
                    {arrow_usd} {abs(chg_usdpk):.2f}% MoM
                </span><br>
                <span style='font-size:0.68rem;color:#94a3b8;'>
                    {usdpk.get("date", "") or "Live rate"}
                </span>
            </div>
        </div>""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION E — FINANCIAL EXPOSURE  (Step 3)
    # Shows estimated procurement cost alongside on-hand, 45d need, and gap.
    # All cost figures are labeled ESTIMATED — not procurement quotes.
    # Inventory Kgs appears here only (not in standalone tiles per design decision).
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='exec-section-bar'>
        <span class='exec-section-label' style='color:#b45309;'>
            Financial Exposure (Estimated)
        </span>
    </div>""", unsafe_allow_html=True)

    if no_data:
        _empty_state("No pipeline data — run the monthly pipeline to populate this section.")
    elif not exposure["market_data_available"]:
        st.markdown("""
        <div style='background:#fffbeb;border:1.5px solid #fde68a;border-radius:8px;
                    padding:0.9rem 1.1rem;margin-bottom:0.75rem;'>
            <span style='font-size:0.83rem;font-weight:600;color:#92400e;'>
                Market price data unavailable — cost estimates require ICE Cotton and PSF prices.
                Inventory and gap figures are shown below without cost.
            </span>
        </div>""", unsafe_allow_html=True)
        fe1, fe2 = st.columns(2)
        with fe1:
            st.markdown(_kpi_card(
                "COTTON",
                f"{exposure['cotton_shortfall_kgs']:,.0f} Kgs gap",
                (f"On hand: {exposure['cotton_inv_kgs']:,.0f} Kgs  ·  "
                 f"45d need: {exposure['cotton_need_kgs']:,.0f} Kgs"),
                "#2563eb",
            ), unsafe_allow_html=True)
        with fe2:
            st.markdown(_kpi_card(
                "FIBER",
                f"{exposure['fiber_shortfall_kgs']:,.0f} Kgs gap",
                (f"On hand: {exposure['fiber_inv_kgs']:,.0f} Kgs  ·  "
                 f"45d need: {exposure['fiber_need_kgs']:,.0f} Kgs"),
                _C_MONITOR,
            ), unsafe_allow_html=True)
    else:
        fe1, fe2, fe3 = st.columns(3)

        with fe1:
            _cotton_buy_exists = (
                not no_data and not df[
                    (df["commodity"] == "Cotton") & (df["action"] == "BUY")
                ].empty
            )
            st.markdown(_kpi_card(
                "EST. COTTON PROCUREMENT",
                exposure["cotton_cost_display"],
                (f"{exposure['cotton_shortfall_kgs']:,.0f} Kgs gap  ·  "
                 f"ICE {exposure['ice_price_usd_per_lb']:.4f} USD/lb"),
                _C_BUY if _cotton_buy_exists else _C_HOLD,
            ), unsafe_allow_html=True)
            pkr_cotton = exposure["cotton_cost_usd"] * exposure["usd_pkr_rate"]
            st.markdown(
                f"<p style='font-size:0.72rem;color:#64748b;margin:0.2rem 0 0.6rem 0.25rem;'>"
                f"On hand: {exposure['cotton_inv_kgs']:,.0f} Kgs  ·  "
                f"45d need: {exposure['cotton_need_kgs']:,.0f} Kgs<br>"
                f"PKR ~{pkr_cotton / 1_000_000:.0f}M @ {exposure['usd_pkr_rate']:.2f}</p>",
                unsafe_allow_html=True,
            )

        with fe2:
            _fiber_buy_exists = (
                not no_data and not df[
                    (df["commodity"] == "Fiber") & (df["action"] == "BUY")
                ].empty
            )
            st.markdown(_kpi_card(
                "EST. FIBER PROCUREMENT",
                exposure["fiber_cost_display"],
                (f"{exposure['fiber_shortfall_kgs']:,.0f} Kgs gap  ·  "
                 f"PSF {exposure['psf_price_usd_per_kg']:.4f} USD/kg"),
                _C_BUY if _fiber_buy_exists else _C_HOLD,
            ), unsafe_allow_html=True)
            pkr_fiber = exposure["fiber_cost_usd"] * exposure["usd_pkr_rate"]
            st.markdown(
                f"<p style='font-size:0.72rem;color:#64748b;margin:0.2rem 0 0.6rem 0.25rem;'>"
                f"On hand: {exposure['fiber_inv_kgs']:,.0f} Kgs  ·  "
                f"45d need: {exposure['fiber_need_kgs']:,.0f} Kgs<br>"
                f"PKR ~{pkr_fiber / 1_000_000:.0f}M @ {exposure['usd_pkr_rate']:.2f}</p>",
                unsafe_allow_html=True,
            )

        with fe3:
            pkr_total_b = exposure["total_cost_pkr"] / 1_000_000_000
            _fe3_border = _C_BUY if n_buy > 0 else _C_HOLD
            st.markdown(f"""
            <div class='metric-card' style='border-left:3px solid {_fe3_border};'>
                <div class='metric-label' style='color:{_fe3_border};'>TOTAL ESTIMATED EXPOSURE</div>
                <div class='metric-value' style='color:{_C_TEXT};'>{exposure['total_cost_display']}</div>
                <div class='currency-label'>
                    <strong>BUY rows · current market rates</strong><br>
                    PKR ~{pkr_total_b:.2f}B @ {exposure['usd_pkr_rate']:.2f}<br>
                    <span style='font-size:0.68rem;color:#94a3b8;font-style:italic;'>
                        Estimated — not a procurement quote
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4 — PROCUREMENT RECOMMENDATION  (unchanged from v1)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='exec-section-bar'>
        <span class='exec-section-label'>4 — Procurement Recommendation</span>
    </div>""", unsafe_allow_html=True)

    rc1, rc2 = st.columns([1, 2])

    if no_data:
        with rc1:
            _empty_state("No data")
        with rc2:
            _empty_state("Run the pipeline first")
    else:
        def _pct(n: int) -> str:
            return f"{n / total_pairs * 100:.0f}%" if total_pairs > 0 else "—"

        with rc1:
            for label, n, sub_txt, colour in [
                ("BUY NOW",  n_buy,     f"{_pct(n_buy)} of tracked pairs",        _C_BUY),
                ("HOLD",     n_hold,    f"{_pct(n_hold)} — stock adequate",         _C_HOLD),
                ("MONITOR",  n_monitor, f"{_pct(n_monitor)} — insufficient data",   _C_MONITOR),
            ]:
                st.markdown(_kpi_card(label, str(n), sub_txt, colour),
                            unsafe_allow_html=True)

        with rc2:
            action_counts = df["action"].value_counts().reset_index()
            action_counts.columns = ["Action", "Count"]
            colour_map = {"BUY": _C_BUY, "HOLD": _C_HOLD, "MONITOR": _C_MONITOR}
            import plotly.express as _px
            fig = _px.pie(
                action_counts, names="Action", values="Count",
                color="Action", color_discrete_map=colour_map, hole=0.6,
            )
            fig.update_traces(
                textposition="outside", textinfo="label+value+percent", textfont_size=11
            )
            fig.update_layout(
                height=280, margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True,
                            key="exec_recommendation_pie_v2")

    # Executive Insights section removed (UX-3C) — redundant with KPI tiles and Situation Brief.
    # Orphaned Market Forecasts divider removed (UX-3C) — no forecast content follows here.


# ---------------------------------------------------------------------------
# Phase 2 business-oriented page composition
# ---------------------------------------------------------------------------

def _load_execution_plan_for_display() -> tuple[dict, dict, str]:
    """Return (execution_plan, decision_impact, caption) from the shared cache.

    Returns empty dicts when the workbook is absent or the pipeline fails so
    callers never need to guard against None.
    """
    today_str = datetime.date.today().isoformat()
    result = _run_pse5b_cached(today_str)
    if not result.get("ok"):
        return {}, {}, result.get("error", "Execution plan unavailable")
    return (
        result.get("execution_plan", {}),
        result.get("decision_impact", {}),
        f"As of {today_str}",
    )


def _render_plan_unavailable(message: str) -> None:
    st.info(
        "The current procurement plan is unavailable. "
        "Please verify the latest inventory workbook and refresh."
    )
    if message and message != "Inventory workbook unavailable":
        with st.expander("Technical details"):
            st.code(message)


# ---------------------------------------------------------------------------
# PSE-4.1 — Decision Impact rendering helpers (presentation only, no logic)
# ---------------------------------------------------------------------------

_IMPACT_RISK_STYLE = {
    "HIGH":    ("#dc2626", "#fee2e2", "#fca5a5"),
    "MEDIUM":  ("#d97706", "#fef3c7", "#fcd34d"),
    "LOW":     ("#059669", "#d1fae5", "#6ee7b7"),
    "UNKNOWN": ("#64748b", "#f1f5f9", "#cbd5e1"),
}

_IMPACT_UNKNOWN_LABEL = {
    "inventory_outlook":          "Insufficient data to assess inventory outcome.",
    "procurement_progress_impact": "No annual procurement target is configured — progress cannot be assessed.",
    "mix_outlook":                "Portfolio mix impact cannot be determined.",
    "market_exposure":            "Not enough market information is currently available.",
}


def _impact_card(label: str, text: str, detail: str = "", accent: str = "#1e40af") -> str:
    """Styled card matching the existing dashboard metric-card aesthetic."""
    # Friendly fallback if value is flagged UNKNOWN by the engine
    if text.startswith("UNKNOWN"):
        text = _IMPACT_UNKNOWN_LABEL.get(label.lower().replace(" ", "_"), "Not enough data is available.")
        accent = "#94a3b8"
    detail_html = (
        f"<div style='margin-top:8px;font-size:0.72rem;color:#64748b;"
        f"line-height:1.45;border-top:1px solid #f1f5f9;padding-top:6px;'>{detail}</div>"
        if detail and not detail.startswith("UNKNOWN") else ""
    )
    return (
        f"<div style='background:#ffffff;border-radius:8px;padding:16px 18px;"
        f"border:1px solid #e2e8f0;border-left:3px solid {accent};"
        f"height:100%;box-sizing:border-box;'>"
        f"<div style='font-size:0.63rem;font-weight:700;letter-spacing:0.9px;"
        f"color:{accent};text-transform:uppercase;margin-bottom:8px;'>{label}</div>"
        f"<div style='font-size:0.86rem;font-weight:500;color:#1e293b;line-height:1.55;'>{text}</div>"
        f"{detail_html}"
        f"</div>"
    )


def _impact_risk_card(level: str, reason: str) -> str:
    """Operational Risk card with a coloured badge for the risk level."""
    text_col, bg_col, border_col = _IMPACT_RISK_STYLE.get(level, _IMPACT_RISK_STYLE["UNKNOWN"])
    badge = (
        f"<span style='display:inline-block;padding:3px 12px;border-radius:20px;"
        f"font-size:0.72rem;font-weight:700;letter-spacing:0.6px;"
        f"color:{text_col};background:{bg_col};border:1.5px solid {border_col};"
        f"margin-bottom:8px;'>{level}</span>"
    )
    reason_html = (
        f"<div style='font-size:0.80rem;color:#475569;line-height:1.5;margin-top:4px;'>{reason}</div>"
        if reason and not reason.startswith("UNKNOWN") else ""
    )
    return (
        f"<div style='background:#ffffff;border-radius:8px;padding:16px 18px;"
        f"border:1px solid #e2e8f0;border-left:3px solid {text_col};"
        f"height:100%;box-sizing:border-box;'>"
        f"<div style='font-size:0.63rem;font-weight:700;letter-spacing:0.9px;"
        f"color:{text_col};text-transform:uppercase;margin-bottom:8px;'>Operational Risk</div>"
        f"{badge}"
        f"{reason_html}"
        f"</div>"
    )


def _render_decision_impact(impact: dict) -> None:
    """Render the Expected Business Impact section from a DecisionImpact dict.

    Purely presentational — reads fields from the already-computed impact dict.
    No business logic, no recalculation, no engine calls.
    """
    if not impact:
        return

    st.markdown("---")
    st.markdown(
        "<h3 style='color:#1e40af;font-size:1.05rem;font-weight:700;"
        "margin-bottom:4px;'>Expected Business Impact</h3>"
        "<p style='color:#64748b;font-size:0.78rem;margin-top:0;margin-bottom:16px;'>"
        "If this recommendation is followed, the organisation can expect the outcomes below.</p>",
        unsafe_allow_html=True,
    )

    # --- Row 1: Inventory · Procurement Progress · Mix ---
    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        st.markdown(
            _impact_card(
                "Inventory Outlook",
                impact.get("inventory_outlook", "UNKNOWN"),
                impact.get("inventory_outlook_detail", ""),
                "#1e40af",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        prog = impact.get("procurement_progress_impact", "UNKNOWN")
        prog_accent = "#94a3b8" if prog.startswith("UNKNOWN") else "#0891b2"
        st.markdown(
            _impact_card("Procurement Progress", prog, "", prog_accent),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _impact_card(
                "Portfolio Mix Outlook",
                impact.get("mix_outlook", "UNKNOWN"),
                "",
                "#7c3aed",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)

    # --- Row 2: Market Exposure · Operational Risk · Review Guidance ---
    c4, c5, c6 = st.columns(3, gap="small")
    with c4:
        mkt = impact.get("market_exposure", "UNKNOWN")
        mkt_accent = "#94a3b8" if mkt.startswith("UNKNOWN") else "#d97706"
        st.markdown(
            _impact_card("Market Exposure", mkt, "", mkt_accent),
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            _impact_risk_card(
                impact.get("operational_risk_level", "UNKNOWN"),
                impact.get("operational_risk_reason", ""),
            ),
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            _impact_card(
                "Review Guidance",
                impact.get("review_guidance", "Review timing not available."),
                "",
                "#059669",
            ),
            unsafe_allow_html=True,
        )


def render_procurement_decision_page() -> None:
    """Executive landing page, composed only from ProcurementExecutionPlan."""
    _section_header(
        "Procurement Decision",
        "What to buy · how much · when · and why",
        colour="#1e40af",
    )
    plan, impact, caption = _load_execution_plan_for_display()
    if not plan:
        _render_plan_unavailable(caption)
        return

    events = plan.get("procurement_events", [])
    first = events[0] if events else {}
    action = plan.get("recommended_strategy", "BUY" if events else "WAIT")
    window = first.get("preferred_execution_window", "No action required")
    benefit = plan.get("expected_benefits", {})
    benefit_text = benefit.get("expected_inventory_stability", "NOT ASSESSED")

    cols = st.columns(5)
    cards = (
        ("RECOMMENDED STRATEGY", str(action).replace("_", " "), f"{len(events)} planned event(s)", _C_BUY if events else _C_HOLD),
        ("IMMEDIATE QUANTITY", f"{plan.get('immediate_quantity_tons', plan.get('total_planned_quantity_tons', 0)):,.0f} t", "Commit now", "#7c3aed"),
        ("DEFERRED QUANTITY", f"{plan.get('deferred_quantity_tons', 0):,.0f} t", str(plan.get('opportunity_window', window)).replace("_", " "), _C_MONITOR),
        ("CONFIDENCE", plan.get("planning_confidence_level", "N/A"), f"Score {plan.get('planning_confidence_score', 0):.0f}", "#059669"),
        ("EXPECTED BENEFIT", str(benefit_text).replace("_", " "), "Inventory stability", "#0891b2"),
    )
    for col, card in zip(cols, cards):
        with col:
            st.markdown(_kpi_card(*card), unsafe_allow_html=True)

    st.markdown("### Current procurement status")
    st.write(plan.get("plan_summary", "No plan summary available."))
    st.caption(plan.get("strategy_reason", ""))

    # PSE-4.1 — Expected Business Impact (read from decision_impact, never recomputed)
    _render_decision_impact(impact)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("### Recommended actions")
        if not events:
            st.success("No procurement events are required in the current planning cycle.")
        for index, event in enumerate(events, 1):
            st.markdown(
                f"**{index}. {event.get('source', 'Unknown').title()} — "
                f"{event.get('planned_quantity_tons', 0):,.0f} tons now Â· "
                f"{event.get('deferred_quantity_tons', 0):,.0f} tons deferred**  \n"
                f"{event.get('preferred_execution_window', 'N/A').replace('_', ' ')}"
            )
            st.caption(event.get("reason", ""))

    with right:
        st.markdown("### Constraint validation")
        constraints = plan.get("constraint_validation", {})
        if constraints:
            constraint_df = pd.DataFrame(
                [(key.replace("_", " ").title(), value) for key, value in constraints.items()],
                columns=["Constraint", "Status"],
            )
            st.dataframe(constraint_df, hide_index=True, use_container_width=True)
        else:
            st.caption("No constraint results available.")

    st.markdown("### Top reasons")
    for reason in plan.get("reasoning", [])[:4]:
        st.markdown(f"- {reason}")
    st.markdown("### Decision safeguards")
    financial = plan.get("expected_benefits", {})
    financial_usd = financial.get("expected_cost_avoidance_usd")
    financial_basis = financial.get("expected_cost_avoidance_pct_basis")
    if financial_usd is not None:
        st.markdown(f"**Expected financial benefit:** ${financial_usd:,.0f}")
    elif financial_basis is not None:
        st.markdown(
            f"**Expected financial benefit:** {financial_basis:+.2f}% price basis; "
            "USD benefit unavailable pending a verified price and FX basis."
        )
    else:
        st.markdown("**Expected financial benefit:** Not reliably quantifiable from current market inputs.")
    st.markdown(f"**Risk if delayed:** {plan.get('risk_if_delayed', 'N/A')}")
    st.markdown(
        f"**Review:** {plan.get('next_review_date', 'N/A')} â€” "
        f"{plan.get('review_trigger', 'No trigger available.')}"
    )
    st.markdown(
        f"**Alternative considered:** {str(plan.get('alternative_strategy_considered', 'N/A')).replace('_', ' ')}  \n"
        f"{plan.get('alternative_rejection_reason', '')}"
    )
    st.caption(f"Procurement Execution Planning Engine · {caption}")


def render_procurement_plan_page() -> None:
    """Ordered ProcurementExecutionPlan events and their traceability."""
    _section_header(
        "Procurement Plan",
        "Execution sequence · planning windows · dependencies · reasons",
        colour="#7c3aed",
    )
    plan, _, caption = _load_execution_plan_for_display()
    if not plan:
        _render_plan_unavailable(caption)
        return

    events = plan.get("procurement_events", [])
    if not events:
        st.success(plan.get("plan_summary", "No procurement events are required."))
        return

    rows = []
    for index, event in enumerate(events, 1):
        rows.append({
            "Sequence": index,
            "Event": event.get("event_id"),
            "Source": event.get("source"),
            "Quantity (t)": event.get("planned_quantity_tons"),
            "Deferred (t)": event.get("deferred_quantity_tons", 0),
            "Planning window": str(event.get("preferred_execution_window", "")).replace("_", " "),
            "Priority": event.get("planning_priority_level"),
            "Confidence": event.get("confidence_level"),
            "Depends on": ", ".join(event.get("dependencies", [])) or "None",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("### Event detail")
    for event in events:
        label = (
            f"{event.get('event_id')} · {event.get('source', '').title()} · "
            f"{event.get('planned_quantity_tons', 0):,.0f} tons"
        )
        with st.expander(label):
            st.write(event.get("reason", ""))
            st.markdown("**Dependencies:** " + (", ".join(event.get("dependencies", [])) or "None"))
            st.markdown("**Reasoning chain**")
            for reason in event.get("reasoning_chain", []):
                st.markdown(f"- {reason}")
    st.caption(f"Procurement Execution Planning Engine · {caption}")


def render_procurement_analytics_page() -> None:
    """Analyst workspace composed from the established rendering helpers."""
    _section_header(
        "Analytics",
        "Inventory position · mix · gap analysis · risk · supporting metrics",
        colour="#475569",
    )
    df, meta = load_procurement_strategy()
    if df.empty:
        st.info("Procurement analytics are unavailable for the current period.")
        return

    _render_overview(df, meta)
    with st.expander("BUY — Action required", expanded=True):
        _render_buy(df)
    with st.expander("HOLD — Adequate stock"):
        _render_hold(df)
    with st.expander("MONITOR — Attention"):
        _render_monitor(df)
    with st.expander("Inventory risk", expanded=True):
        _render_risk(df)
    with st.expander("Full supporting report"):
        _render_summary(df, meta)
    with st.expander("Monitoring and alerts"):
        render_pse7a_monitoring_panel()


def render_procurement_settings_page() -> None:
    """Operational settings and the preserved scenario-analysis workspace."""
    _section_header(
        "Settings",
        "Scenario analysis and platform controls",
        colour="#64748b",
    )
    st.markdown("### What-if scenario lab")
    render_pse6b_whatif_panel()


# ---------------------------------------------------------------------------
# Legacy page entry point (kept as a compatibility wrapper)
# ---------------------------------------------------------------------------

def render_procurement_intelligence_page() -> None:
    """Render the full PROCUREMENT INTELLIGENCE page.

    Called from main() in streamlit_app.py when the user selects
    the Procurement Intelligence navigation item.
    """
    _section_header(
        "Procurement Intelligence",
        "Inventory coverage · BUY/HOLD/MONITOR recommendations · Shortfall analysis",
        colour="#1e40af",
    )

    df, meta = load_procurement_strategy()

    # Data availability guard
    if df.empty:
        st.warning(
            "No procurement strategy data found. "
            "Run the monthly pipeline to generate recommendations:\n\n"
            "```\npython scripts/run_monthly_strategy_pipeline.py\n```"
        )
        st.info(
            f"Expected file: `reports/procurement_strategy.csv`  \n"
            f"This file is created by Step 8 of the pipeline after the "
            f"procurement engine runs successfully."
        )
        return

    # Period badge
    if meta.get("period_label"):
        st.markdown(
            f"<span style='background:#eff6ff;color:#1d4ed8;border:1.5px solid #bfdbfe;"
            f"border-radius:12px;padding:3px 12px;font-size:0.78rem;font-weight:700;"
            f"letter-spacing:0.3px;'>Period: {meta['period_label']}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Executive Decision",
        "Overview",
        "BUY — Action Required",
        "HOLD — Adequate Stock",
        "MONITOR — Attention",
        "Inventory Risk",
        "Full Report",
        "What-If Lab",              # PSE-6B
        "Monitoring",               # PSE-7A
    ])

    with tab0:
        render_pse6a_executive_panel()

    with tab1:
        _render_overview(df, meta)

    with tab2:
        _render_buy(df)

    with tab3:
        _render_hold(df)

    with tab4:
        _render_monitor(df)

    with tab5:
        _render_risk(df)

    with tab6:
        _render_summary(df, meta)

    with tab7:
        render_pse6b_whatif_panel()

    with tab8:
        render_pse7a_monitoring_panel()


# ===========================================================================
# PSE-6A  EXECUTIVE PROCUREMENT DECISION PANEL
# ===========================================================================
# Consumes procurement_decision_engine.py (PSE-5B) output only.
# No quantities are recomputed here — display layer only.
# ===========================================================================

def _build_default_repository():
    """Repository factory — selects the correct backend for this deployment.

    Priority:
        1. Supabase  — when st.secrets contains SUPABASE_URL and
                       SUPABASE_SERVICE_ROLE_KEY (Streamlit Cloud / production).
        2. Workbook  — when data/strategy/Strategies.xlsx exists on disk
                       (local development / pipeline runner).
        3. Neither   — returns a stub repository whose get_snapshot() raises
                       RepositoryUnavailableError so ProcurementRuntimeService
                       surfaces is_available=False without crashing the app.

    The fallback NEVER raises here; the error propagates inside RuntimeService.run()
    so to_legacy_dict() returns {"ok": False, "error": "..."} and every dashboard
    section renders its unavailable state rather than showing a Streamlit traceback.
    """
    from procurement_data_repository import (
        ProcurementInventoryRepository,
        WorkbookInventoryRepository,
        SupabaseInventoryRepository,
        RepositoryHealth,
        RepositoryUnavailableError,
    )

    # ── 1. Supabase (production / Streamlit Cloud) ───────────────────────────
    try:
        url = st.secrets.get("SUPABASE_URL", "") or ""
        key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "") or ""
        if url and key:
            return SupabaseInventoryRepository(url=url, key=key)
    except Exception:
        # st.secrets unavailable (e.g. local dev without secrets.toml); fall through
        pass

    # ── 2. Workbook (local development) ─────────────────────────────────────
    _workbook_path = _PROJECT_ROOT / "data" / "strategy" / "Strategies.xlsx"
    if _workbook_path.exists():
        return WorkbookInventoryRepository(_workbook_path)

    # ── 3. No backend available — return a stub so the error flows through
    #        RuntimeService.run() as is_available=False, not a Streamlit crash.
    _workbook_path_str = str(_workbook_path)

    class _NoBackendRepository(ProcurementInventoryRepository):
        _msg = (
            "No inventory backend is configured for this deployment. "
            "Production: set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in Streamlit secrets. "
            f"Development: place Strategies.xlsx at {_workbook_path_str}"
        )

        def get_snapshot(self, as_of=None):
            raise RepositoryUnavailableError(self._msg)

        def health(self):
            return RepositoryHealth(
                backend="unavailable",
                status="unavailable",
                is_available=False,
                last_refresh=None,
                latency_ms=None,
                record_count=None,
                freshness_hours=None,
                message=self._msg,
            )

    return _NoBackendRepository()


def _build_runtime_service():
    """Return a configured ProcurementRuntimeService."""
    from procurement_runtime_service import ProcurementRuntimeService
    return ProcurementRuntimeService(_build_default_repository())


_U_COLOR = {
    "CRITICAL": ("#dc2626", "#fef2f2", "#fca5a5"),  # (text, bg, border)
    "HIGH":     ("#b45309", "#fffbeb", "#fde68a"),
    "MEDIUM":   ("#1d4ed8", "#eff6ff", "#bfdbfe"),
    "LOW":      ("#059669", "#f0fdf4", "#86efac"),
}
_A_COLOR = {
    "BUY_NOW":     "#dc2626",
    "BUY_FORWARD": "#b45309",
    "BUY_SPLIT":   "#7c3aed",
    "DEFER":       "#1d4ed8",
    "HOLD":        "#64748b",
}
_FI_COLOR = {
    "COST_AVOIDANCE": "#059669",
    "SAVING":         "#1d4ed8",
    "NONE":           "#94a3b8",
}


@st.cache_data(ttl=1800, show_spinner=False)
def _run_pse5b_cached(today_str: str) -> dict:
    """Single shared cache for the full PSE pipeline.  Cached 30 min.

    Delegates to ProcurementRuntimeService.run() and converts the result to
    the legacy dict format so all dashboard rendering helpers work unchanged.
    The dashboard never imports a repository or calls an engine directly.
    """
    from datetime import date as _date
    svc = _build_runtime_service()
    result = svc.run(as_of=_date.fromisoformat(today_str))
    return result.to_legacy_dict()


def _badge(text: str, text_color: str, bg: str, border: str, size: str = "0.72rem") -> str:
    return (
        f"<span style='display:inline-block;padding:2px 10px;border-radius:12px;"
        f"font-size:{size};font-weight:700;letter-spacing:0.4px;"
        f"color:{text_color};background:{bg};border:1.5px solid {border};'>"
        f"{text}</span>"
    )


def _urgency_pill(urgency: str) -> str:
    tc, bg, bd = _U_COLOR.get(urgency, ("#475569", "#f1f5f9", "#e2e8f0"))
    return _badge(urgency, tc, bg, bd)


def _action_pill(action_label: str, action_code: str) -> str:
    col = _A_COLOR.get(action_code, "#475569")
    return _badge(action_label.upper(), col, "#ffffff", col)


def _fi_pill(fi_type: str) -> str:
    col = _FI_COLOR.get(fi_type, "#94a3b8")
    labels = {"COST_AVOIDANCE": "COST AVOIDANCE", "SAVING": "EXPECTED SAVING", "NONE": "NO PRICE BENEFIT"}
    return _badge(labels.get(fi_type, fi_type), col, "#ffffff", col)


def _conf_bar(score: int, label: str) -> str:
    col = "#059669" if score >= 75 else ("#1d4ed8" if score >= 50 else "#b45309")
    bar_w = max(4, score)
    return (
        f"<div style='display:flex;align-items:center;gap:8px;'>"
        f"<div style='flex:1;background:#e2e8f0;border-radius:4px;height:6px;'>"
        f"<div style='width:{bar_w}%;background:{col};height:6px;border-radius:4px;'></div>"
        f"</div>"
        f"<span style='font-size:0.72rem;font-weight:700;color:{col};white-space:nowrap;'>"
        f"{score}/100 · {label}</span>"
        f"</div>"
    )


def _qty_row(label: str, qty: float, note: str = "", color: str = "#0f172a") -> str:
    if qty == 0 and not note:
        return ""
    note_html = f"<span style='color:#94a3b8;font-size:0.72rem;margin-left:6px;'>{note}</span>" if note else ""
    return (
        f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
        f"padding:4px 0;border-bottom:1px solid #f1f5f9;'>"
        f"<span style='font-size:0.78rem;color:#64748b;'>{label}</span>"
        f"<span style='font-size:0.88rem;font-weight:700;color:{color};'>"
        f"{qty:,.0f} t{note_html}</span>"
        f"</div>"
    )


def _pse6a_source_card(dec: dict, source_label: str) -> None:
    """Render one source decision card (LOCAL or IMPORTED)."""
    urgency     = dec.get("urgency", "LOW")
    action_code = dec.get("action_code", "HOLD")
    action_lbl  = dec.get("action", "Hold")
    conf        = dec.get("confidence", {})
    risk        = dec.get("risk", {})
    fi          = dec.get("expected_cost_impact", {})

    tc, bg, bd = _U_COLOR.get(urgency, ("#475569", "#f1f5f9", "#e2e8f0"))

    qty_now      = dec.get("qty_now_tons", 0.0)
    qty_later    = dec.get("qty_later_tons", 0.0)
    mandatory    = dec.get("mandatory_tons", 0.0)
    structural   = dec.get("base_structural_tons", 0.0)
    opportunistic = dec.get("opportunistic_tons", 0.0)
    deferred     = dec.get("deferred_tons", 0.0)
    order_date   = dec.get("order_date") or "—"
    arrival      = dec.get("expected_arrival") or "—"
    defer_until  = dec.get("defer_until") or "—"

    fi_type     = fi.get("impact_type", "NONE")
    fi_narrative = fi.get("narrative", "")
    avoidance   = fi.get("cost_avoidance_usd")
    savings     = fi.get("expected_savings_usd")

    qty_rows_html = "".join(filter(None, [
        _qty_row("Mandatory replenishment", mandatory, "stock-protection", "#dc2626") if mandatory > 0 else "",
        _qty_row("Structural (base plan)", structural, "calendar/mix", "#1d4ed8") if structural > 0 else "",
        _qty_row("Opportunistic (forward buy)", opportunistic, "price-timed", "#7c3aed") if opportunistic > 0 else "",
        _qty_row("Deferred to later window", deferred, f"by {defer_until}", "#059669") if deferred > 0 else "",
    ]))

    total_label = "Total — order today" if qty_now > 0 else "Total — deferred"
    total_qty   = qty_now if qty_now > 0 else qty_later
    total_color = "#dc2626" if urgency == "CRITICAL" else ("#b45309" if urgency == "HIGH" else "#0f172a")
    qty_rows_html += _qty_row(total_label, total_qty, "", total_color)

    if avoidance:
        fi_amount = f"USD {avoidance:,.0f}"
    elif savings:
        fi_amount = f"USD {savings:,.0f}"
    else:
        fi_amount = "—"

    st.markdown(f"""
    <div style='background:#ffffff;border-radius:10px;border:1px solid {bd};
                border-left:4px solid {tc};padding:1rem 1.2rem;
                box-shadow:0 2px 6px rgba(0,0,0,0.06);margin-bottom:0.5rem;'>

      <div style='display:flex;justify-content:space-between;align-items:flex-start;
                  margin-bottom:0.65rem;flex-wrap:wrap;gap:6px;'>
        <div>
          <span style='font-size:1rem;font-weight:800;color:#0f172a;
                       letter-spacing:-0.3px;'>{source_label}</span>
          &nbsp;&nbsp;{_action_pill(action_lbl, action_code)}&nbsp;{_urgency_pill(urgency)}
        </div>
        <div style='text-align:right;'>
          {_conf_bar(conf.get("score", 0), conf.get("label", ""))}
        </div>
      </div>

      <div style='margin-bottom:0.75rem;'>{qty_rows_html}</div>

      <div style='display:grid;grid-template-columns:1fr 1fr;gap:6px;
                  margin-bottom:0.75rem;'>
        <div style='background:#f8fafc;border-radius:6px;padding:6px 10px;'>
          <div style='font-size:0.68rem;color:#94a3b8;font-weight:600;
                      letter-spacing:0.3px;margin-bottom:2px;'>ORDER DATE</div>
          <div style='font-size:0.85rem;font-weight:700;color:#0f172a;'>{order_date}</div>
        </div>
        <div style='background:#f8fafc;border-radius:6px;padding:6px 10px;'>
          <div style='font-size:0.68rem;color:#94a3b8;font-weight:600;
                      letter-spacing:0.3px;margin-bottom:2px;'>
            {'EXPECTED ARRIVAL' if qty_now > 0 else 'DEFER UNTIL'}</div>
          <div style='font-size:0.85rem;font-weight:700;color:#0f172a;'>
            {arrival if qty_now > 0 else defer_until}</div>
        </div>
      </div>

      <div style='display:flex;align-items:center;justify-content:space-between;
                  flex-wrap:wrap;gap:6px;margin-bottom:0.65rem;'>
        {_fi_pill(fi_type)}
        <span style='font-size:0.82rem;font-weight:700;
                     color:{_FI_COLOR.get(fi_type,"#94a3b8")};'>{fi_amount}</span>
      </div>

      <div style='font-size:0.78rem;color:#374151;line-height:1.55;
                  padding-top:0.5rem;border-top:1px solid #f1f5f9;'>
        {dec.get("executive_summary","").replace(". ", ".<br>")}
      </div>
    </div>
    """, unsafe_allow_html=True)


def _pse6a_portfolio_card(er: dict) -> None:
    """Render portfolio-level summary strip."""
    local    = er.get("local", {})
    imported = er.get("imported", {})
    urgency  = er.get("portfolio_urgency", "LOW")
    tc, bg, bd = _U_COLOR.get(urgency, ("#475569", "#f1f5f9", "#e2e8f0"))

    total_now   = local.get("qty_now_tons", 0) + imported.get("qty_now_tons", 0)
    total_defer = local.get("qty_later_tons", 0) + imported.get("qty_later_tons", 0)

    # combined financial
    total_avoid = (
        (local.get("expected_cost_impact", {}).get("cost_avoidance_usd") or 0) +
        (imported.get("expected_cost_impact", {}).get("cost_avoidance_usd") or 0)
    )
    total_save  = (
        (local.get("expected_cost_impact", {}).get("expected_savings_usd") or 0) +
        (imported.get("expected_cost_impact", {}).get("expected_savings_usd") or 0)
    )
    fi_str = (
        f"USD {total_avoid:,.0f} avoided" if total_avoid > 0 else
        f"USD {total_save:,.0f} saving"   if total_save  > 0 else "—"
    )

    risks   = er.get("key_risks", [])
    actions = er.get("recommended_actions", [])

    risk_html = "".join(
        f"<div style='font-size:0.75rem;color:#dc2626;padding:2px 0;'>&#9655; {r}</div>"
        for r in risks
    )
    action_html = "".join(
        f"<div style='font-size:0.75rem;color:#0f172a;padding:2px 0;'>&#10003; {a}</div>"
        for a in actions
    )

    st.markdown(f"""
    <div style='background:{bg};border-radius:10px;border:1px solid {bd};
                border-left:5px solid {tc};padding:1rem 1.4rem;
                margin-bottom:1rem;box-shadow:0 2px 8px rgba(0,0,0,0.07);'>
      <div style='display:flex;justify-content:space-between;align-items:center;
                  margin-bottom:0.75rem;flex-wrap:wrap;gap:8px;'>
        <div>
          <span style='font-size:0.68rem;font-weight:700;color:{tc};
                       letter-spacing:1px;text-transform:uppercase;
                       margin-bottom:2px;display:block;'>Portfolio Risk</span>
          <span style='font-size:1.1rem;font-weight:800;color:#0f172a;'>
            Executive Procurement Decision</span>
        </div>
        {_urgency_pill(urgency)}
      </div>

      <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:10px;
                  margin-bottom:0.85rem;'>
        <div style='background:#ffffff;border-radius:7px;padding:8px 12px;
                    border:1px solid {bd};'>
          <div style='font-size:0.65rem;color:#94a3b8;font-weight:700;
                      letter-spacing:0.5px;margin-bottom:3px;'>BUY TODAY</div>
          <div style='font-size:1.05rem;font-weight:800;color:{tc};'>
            {total_now:,.0f} t</div>
        </div>
        <div style='background:#ffffff;border-radius:7px;padding:8px 12px;
                    border:1px solid {bd};'>
          <div style='font-size:0.65rem;color:#94a3b8;font-weight:700;
                      letter-spacing:0.5px;margin-bottom:3px;'>DEFERRED</div>
          <div style='font-size:1.05rem;font-weight:800;color:#1d4ed8;'>
            {total_defer:,.0f} t</div>
        </div>
        <div style='background:#ffffff;border-radius:7px;padding:8px 12px;
                    border:1px solid {bd};'>
          <div style='font-size:0.65rem;color:#94a3b8;font-weight:700;
                      letter-spacing:0.5px;margin-bottom:3px;'>FINANCIAL IMPACT</div>
          <div style='font-size:0.88rem;font-weight:800;color:#059669;'>{fi_str}</div>
        </div>
        <div style='background:#ffffff;border-radius:7px;padding:8px 12px;
                    border:1px solid {bd};'>
          <div style='font-size:0.65rem;color:#94a3b8;font-weight:700;
                      letter-spacing:0.5px;margin-bottom:3px;'>LOCAL · IMPORTED</div>
          <div style='font-size:0.8rem;font-weight:700;color:#0f172a;'>
            {local.get("action","—")} · {imported.get("action","—")}</div>
        </div>
      </div>

      <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>
        <div>
          <div style='font-size:0.68rem;font-weight:700;color:#94a3b8;
                      letter-spacing:0.5px;margin-bottom:4px;'>KEY RISKS</div>
          {risk_html if risk_html else
           "<div style='font-size:0.75rem;color:#059669;'>No elevated risks.</div>"}
        </div>
        <div>
          <div style='font-size:0.68rem;font-weight:700;color:#94a3b8;
                      letter-spacing:0.5px;margin-bottom:4px;'>RECOMMENDED ACTIONS</div>
          {action_html if action_html else
           "<div style='font-size:0.75rem;color:#64748b;'>No actions required.</div>"}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def _pse6a_timeline(er: dict) -> None:
    """Render procurement action timeline as a Plotly Gantt chart."""
    import datetime as _dt

    local    = er.get("local",    {})
    imported = er.get("imported", {})
    today    = _dt.date.today()

    bars = []
    for dec, label in [(local, "LOCAL"), (imported, "IMPORTED")]:
        ac   = dec.get("action_code", "HOLD")
        qty  = dec.get("qty_now_tons", 0)
        qltr = dec.get("qty_later_tons", 0)
        urg  = dec.get("urgency", "LOW")
        col  = _U_COLOR.get(urg, ("#475569", "#f1f5f9", "#e2e8f0"))[0]

        if ac in ("BUY_NOW", "BUY_FORWARD", "BUY_SPLIT") and qty > 0:
            arr_str  = dec.get("expected_arrival")
            arr_date = (_dt.date.fromisoformat(arr_str)
                        if arr_str else today + _dt.timedelta(days=30))
            bars.append(dict(
                Task=label, Start=str(today), Finish=str(arr_date),
                Action=f"Order: {qty:,.0f} t  →  Arrival: {arr_date}",
                Color=col, LT=arr_date,
            ))

        if qltr > 0:
            def_end_str = dec.get("defer_until")
            def_end = (_dt.date.fromisoformat(def_end_str)
                       if def_end_str else today + _dt.timedelta(days=14))
            bars.append(dict(
                Task=label + " (deferred)", Start=str(today),
                Finish=str(def_end),
                Action=f"Defer {qltr:,.0f} t  →  latest safe date: {def_end}",
                Color="#1d4ed8", LT=def_end,
            ))

    if not bars:
        st.markdown(
            "<div style='color:#94a3b8;font-size:0.82rem;padding:0.5rem 0;'>"
            "No active procurement actions to display on the timeline.</div>",
            unsafe_allow_html=True,
        )
        return

    fig = go.Figure()
    y_pos = {t: i for i, t in enumerate(dict.fromkeys(b["Task"] for b in bars))}

    for b in bars:
        t0  = _dt.date.fromisoformat(b["Start"])
        t1  = _dt.date.fromisoformat(b["Finish"])
        days = max((t1 - t0).days, 1)
        fig.add_trace(go.Bar(
            x=[days], y=[b["Task"]], orientation="h",
            base=[(t0 - today).days],
            marker_color=b["Color"], marker_line_width=0,
            text=b["Action"], textposition="inside",
            textfont=dict(size=10, color="#ffffff"),
            hovertemplate=b["Action"] + "<extra></extra>",
            name=b["Task"],
            showlegend=False,
        ))
        fig.add_annotation(
            x=(t1 - today).days + 1, y=b["Task"],
            text=str(b["Finish"]), showarrow=False,
            font=dict(size=9, color="#64748b"), xanchor="left",
        )

    max_day = max(((_dt.date.fromisoformat(b["Finish"]) - today).days + 14) for b in bars)
    fig.update_layout(
        barmode="overlay",
        height=max(160, 70 * len(y_pos) + 60),
        margin=dict(l=10, r=60, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fafafa",
        xaxis=dict(
            title="Days from today",
            tickfont=dict(size=9),
            range=[-2, max_day],
            showgrid=True, gridcolor="#e2e8f0",
            zeroline=True, zerolinecolor="#dc2626", zerolinewidth=2,
        ),
        yaxis=dict(showgrid=False, tickfont=dict(size=10)),
        font=dict(family="Inter, sans-serif"),
    )
    fig.add_vline(x=0, line_width=2, line_color="#dc2626",
                  annotation_text="TODAY", annotation_position="top right",
                  annotation_font_size=9)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _pse6a_insights(er: dict) -> None:
    """Expandable Executive Insights section — structured reasoning per source."""
    for source_key, label in [("local", "LOCAL COTTON"), ("imported", "IMPORTED COTTON")]:
        dec = er.get(source_key, {})
        if not dec:
            continue
        with st.expander(
            f"Why is the engine recommending this for {label}?",
            expanded=False,
        ):
            fields = [
                ("Inventory Position",  dec.get("inventory_reason", "")),
                ("Price Intelligence",  dec.get("price_reason", "")),
                ("Timing Rationale",    dec.get("timing_reason", "")),
                ("Quantity Breakdown",  dec.get("quantity_reason", "")),
                ("Business Rationale",  dec.get("business_reason", "")),
                ("Risk if Not Followed", dec.get("expected_risk_if_ignored", "")),
                ("Financial Impact",    dec.get("expected_cost_impact", {}).get("narrative", "")),
                ("Confidence",          dec.get("confidence", {}).get("explanation", "")),
            ]
            for title, body in fields:
                if not body:
                    continue
                st.markdown(
                    f"<div style='margin-bottom:0.6rem;'>"
                    f"<div style='font-size:0.72rem;font-weight:700;color:#1d4ed8;"
                    f"letter-spacing:0.5px;text-transform:uppercase;"
                    f"margin-bottom:3px;'>{title}</div>"
                    f"<div style='font-size:0.82rem;color:#374151;line-height:1.6;'>"
                    f"{body}</div></div>",
                    unsafe_allow_html=True,
                )


def render_pse6a_executive_panel() -> None:
    """
    PSE-6A — Executive Procurement Decision Panel.

    Runs PSE-5B (cached 30 min) and renders:
      1. Portfolio summary card
      2. LOCAL + IMPORTED source decision cards
      3. Action timeline
      4. Expandable executive insights

    Local:  reads live data from Strategies.xlsx via PSE-5B; writes
            reports/executive_decision.json as a deployment cache.
    Cloud:  if workbook is absent, falls back to reports/executive_decision.json.
    """
    import datetime as _dt
    import json as _json

    _EXEC_JSON = _PROJECT_ROOT / "reports" / "executive_decision.json"

    # ── Resolve data source ──────────────────────────────────────────────────
    today_str = _dt.date.today().isoformat()
    er = None
    caption_suffix = ""

    with st.spinner("Loading procurement decision engine…"):
        result = _run_pse5b_cached(today_str)

    if result.get("ok"):
        # ── Live path: Runtime Service produced a result ─────────────────────
        er = result["report"]
        caption_suffix = (
            f"Refreshed: {today_str}  ·  Cache TTL: 30 min"
        )

        # Write deployment cache so Streamlit Cloud can render without workbook.
        # Non-fatal: a filesystem error must never break local execution.
        try:
            _EXEC_JSON.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(er)
            payload["_generated"] = today_str
            _EXEC_JSON.write_text(
                _json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
    else:
        # ── Cloud path: fall back to cached JSON snapshot ────────────────────
        if _EXEC_JSON.exists():
            try:
                payload = _json.loads(_EXEC_JSON.read_text(encoding="utf-8"))
                # Validate required top-level keys before accepting the cache
                if (
                    isinstance(payload, dict)
                    and "local" in payload
                    and "imported" in payload
                    and "portfolio_urgency" in payload
                ):
                    generated = payload.pop("_generated", "unknown")
                    er = payload
                    caption_suffix = (
                        f"Data: cached snapshot  ·  "
                        f"Generated: {generated}  ·  "
                        f"Source: reports/executive_decision.json"
                    )
            except Exception:
                pass  # malformed cache → fall through to info message

        if er is None:
            st.info(
                "Inventory workbook not found. "
                "Expected: `data/strategy/Strategies.xlsx`  \n"
                "Place the workbook file there and refresh the page."
            )
            return

    # ── Render (identical path regardless of data source) ────────────────────

    # 1. Portfolio summary
    _pse6a_portfolio_card(er)

    # 2. Source decision cards
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        _pse6a_source_card(er.get("local",    {}), "LOCAL COTTON")
    with c2:
        _pse6a_source_card(er.get("imported", {}), "IMPORTED COTTON")

    # 3. Action timeline
    st.markdown(
        "<div style='font-size:0.72rem;font-weight:700;color:#94a3b8;"
        "letter-spacing:0.8px;text-transform:uppercase;"
        "margin:1.1rem 0 0.4rem 0;'>Procurement Timeline</div>",
        unsafe_allow_html=True,
    )
    _pse6a_timeline(er)

    # 4. Executive insights (expandable)
    st.markdown(
        "<div style='font-size:0.72rem;font-weight:700;color:#94a3b8;"
        "letter-spacing:0.8px;text-transform:uppercase;"
        "margin:0.8rem 0 0.4rem 0;'>Executive Insights</div>",
        unsafe_allow_html=True,
    )
    _pse6a_insights(er)

    st.caption(f"Decision engine: PSE-5B  ·  {caption_suffix}")


# ===========================================================================
# PSE-6B  EXECUTIVE WHAT-IF SIMULATION LABORATORY
# ===========================================================================
# Allows management to test procurement scenarios without changing production
# calculations.  Production recommendation is read-only.
# Simulation calls the existing engine (_build_source_decision, run_pse3d,
# compute_procurement_calendar, compute_order_consolidation) with
# user-supplied parameter overrides — no calculations are duplicated.
# ===========================================================================

def _pse6b_load_defaults() -> dict:
    """Build What-If Lab defaults by reading production constants at import time.

    Avoids duplicating the approved PSE-2.7 constants (DAILY_CONSUMPTION_TOTAL,
    LOCAL_MIX_TARGET, LOCAL_LEAD_TIME_DAYS, IMPORTED_LEAD_TIME_DAYS) so that a
    constant change in procurement_strategy_engine.py is automatically reflected
    in the simulator without a separate edit here.
    """
    from procurement_strategy_engine import (
        DAILY_CONSUMPTION_TOTAL,
        LOCAL_MIX_TARGET,
        LOCAL_LEAD_TIME_DAYS,
        IMPORTED_LEAD_TIME_DAYS,
    )
    return {
        "total_consumption":  float(DAILY_CONSUMPTION_TOTAL),
        "local_mix_pct":      float(LOCAL_MIX_TARGET * 100),
        "local_lead_time":    int(LOCAL_LEAD_TIME_DAYS),
        "imported_lead_time": int(IMPORTED_LEAD_TIME_DAYS),
        "pkr_rate":           281.0,   # market rate — not a static constant
        "current_price":        0.78,
        "forecast_h1":          0.78,
        "forecast_h3":          0.78,
    }

_PSE6B_PROD_DEFAULTS = _pse6b_load_defaults()


def _pse6b_get_prod_so(today_str: str) -> dict:
    """Return inventory/status data from the shared PSE-5B cache.

    No I/O here — data comes from _run_pse5b_cached() which already holds it.
    """
    result = _run_pse5b_cached(today_str)
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error", "Pipeline error")}
    return {"ok": True, **result["inventory"]}


def _pse6b_run_simulation(so_dict: dict, sim_params: dict, today_str: str) -> dict:
    """
    Run the procurement engine with user-overridden parameters.

    Calls existing engine functions with simulation inputs — no procurement
    calculations are duplicated.  Returns serialisable dict.
    """
    try:
        from datetime import date as _date
        from procurement_strategy_engine import MIN_STOCK_DAYS, _classify_supply_status
        from procurement_planning_engine import run_pse3d
        from procurement_calendar_engine import compute_procurement_calendar
        from procurement_consolidation_engine import compute_order_consolidation
        from procurement_scenario_engine import (
            _build_source_decision, ScenarioReport,
            ACTION_BUY_NOW, ACTION_BUY_FORWARD, ACTION_BUY_SPLIT,
            ACTION_DEFER, ACTION_HOLD,
        )
        from procurement_decision_engine import generate_executive_report

        today = _date.fromisoformat(today_str)

        local_mix  = sim_params["local_mix_pct"] / 100.0
        imp_mix    = 1.0 - local_mix
        total_cons = sim_params["total_consumption"]
        sim_rate_l = total_cons * local_mix
        sim_rate_i = total_cons * imp_mix
        sim_ss_l   = MIN_STOCK_DAYS * sim_rate_l
        sim_ss_i   = MIN_STOCK_DAYS * sim_rate_i
        sim_lt_l   = int(sim_params["local_lead_time"])
        sim_lt_i   = int(sim_params["imported_lead_time"])
        sim_rop_l  = (MIN_STOCK_DAYS + sim_lt_l) * sim_rate_l
        sim_rop_i  = (MIN_STOCK_DAYS + sim_lt_i) * sim_rate_i

        sim_local_inv = float(sim_params["local_inventory"])
        sim_imp_inv   = float(sim_params["imported_inventory"])
        sim_total_inv = sim_local_inv + sim_imp_inv

        sim_local_status = _classify_supply_status(sim_local_inv, sim_ss_l, sim_rop_l)
        sim_imp_status   = _classify_supply_status(sim_imp_inv,   sim_ss_i, sim_rop_i)

        current_price = float(sim_params["current_price"])
        forecast_h1   = float(sim_params["forecast_h1"])
        forecast_h3   = float(sim_params["forecast_h3"])
        pkr_rate      = float(sim_params["pkr_rate"])

        plan = run_pse3d(
            local_inventory_tons=sim_local_inv,
            imported_inventory_tons=sim_imp_inv,
            local_status=sim_local_status,
            imported_status=sim_imp_status,
            total_inventory_tons=sim_total_inv,
            current_price_usd_per_lb=current_price,
            forecast_h1_usd_per_lb=forecast_h1,
            forecast_h3_usd_per_lb=forecast_h3,
            pkr_rate=pkr_rate,
            today=today,
        )
        cal  = compute_procurement_calendar(
            local_inventory_tons=sim_local_inv,
            imported_inventory_tons=sim_imp_inv,
            local_status=sim_local_status,
            imported_status=sim_imp_status,
            total_inventory_tons=sim_total_inv,
            today=today,
        )
        cons = compute_order_consolidation(calendar_result=cal)
        req  = plan.requirement

        local_sd = _build_source_decision(
            source="LOCAL",
            status=sim_local_status,
            inventory_tons=sim_local_inv,
            total_inventory_tons=sim_total_inv,
            days_cover=round(sim_local_inv / max(sim_rate_l, 0.01), 1),
            rop_tons=sim_rop_l,
            safety_stock_tons=sim_ss_l,
            lead_time_days=sim_lt_l,
            daily_rate=sim_rate_l,
            deficit_from_plan=req["deficit_local_tons"],
            latest_safe_order_date=plan.local_recommendation["latest_safe_order_date"],
            consolidation_result=cons,
            plan=plan,
            today=today,
            current_price=current_price,
            forecast_price=forecast_h1,
            forecast_h_bounds=None,
            pkr_rate=pkr_rate,
        )
        imp_sd = _build_source_decision(
            source="IMPORTED",
            status=sim_imp_status,
            inventory_tons=sim_imp_inv,
            total_inventory_tons=sim_total_inv,
            days_cover=round(sim_imp_inv / max(sim_rate_i, 0.01), 1),
            rop_tons=sim_rop_i,
            safety_stock_tons=sim_ss_i,
            lead_time_days=sim_lt_i,
            daily_rate=sim_rate_i,
            deficit_from_plan=req["deficit_imported_tons"],
            latest_safe_order_date=plan.imported_recommendation["latest_safe_order_date"],
            consolidation_result=cons,
            plan=plan,
            today=today,
            current_price=current_price,
            forecast_price=forecast_h3,
            forecast_h_bounds=None,
            pkr_rate=pkr_rate,
        )

        _prio = {ACTION_BUY_NOW: 1, ACTION_BUY_FORWARD: 2, ACTION_BUY_SPLIT: 3,
                 ACTION_DEFER: 4, ACTION_HOLD: 5}
        lp = _prio.get(local_sd.final_action, 9)
        ip = _prio.get(imp_sd.final_action, 9)
        portfolio_action = local_sd.final_action if lp <= ip else imp_sd.final_action

        sim_sr = ScenarioReport(
            run_date=today.isoformat(),
            local=local_sd,
            imported=imp_sd,
            portfolio_action=portfolio_action,
            portfolio_risk_level="SIM",
            portfolio_reasoning="Simulation run — parameters overridden by user.",
            price_inputs_used={
                "current_price_usd_per_lb": current_price,
                "forecast_h1_usd_per_lb":   forecast_h1,
                "forecast_h3_usd_per_lb":   forecast_h3,
                "pkr_rate":                  pkr_rate,
            },
            assumptions=["SIMULATION — parameters overridden by user."],
        )

        exec_report = generate_executive_report(
            scenario_report=sim_sr,
            local_status=sim_local_status,
            imported_status=sim_imp_status,
            pkr_rate=pkr_rate,
        )

        return {
            "ok":          True,
            "exec_report": exec_report.to_dict(),
            "sim_rates": {
                "local":           sim_rate_l,
                "imported":        sim_rate_i,
                "rop_local":       sim_rop_l,
                "rop_imported":    sim_rop_i,
                "local_status":    sim_local_status,
                "imported_status": sim_imp_status,
            },
        }
    except Exception as exc:
        import traceback
        return {"ok": False, "error": str(exc) + "\n" + traceback.format_exc()}


# ---------------------------------------------------------------------------
# Score, diff, adopt — pure functions (no Streamlit)
# ---------------------------------------------------------------------------

def _pse6b_score(dec: dict, risk_appetite: str) -> dict:
    """Return 5-dimension scores (0-100, higher is better)."""
    urgency     = dec.get("urgency", "LOW")
    risk_level  = dec.get("risk", {}).get("level", "LOW")
    conf_score  = dec.get("confidence", {}).get("score", 50)
    action_code = dec.get("action_code", "HOLD")
    fi          = dec.get("expected_cost_impact", {})
    fi_type     = fi.get("impact_type", "NONE")
    avoidance   = fi.get("cost_avoidance_usd") or 0
    savings     = fi.get("expected_savings_usd") or 0

    op_map  = {"CRITICAL": 15, "HIGH": 45, "MEDIUM": 72, "LOW": 95}
    op_safe = op_map.get(urgency, 50)

    if fi_type == "COST_AVOIDANCE" and avoidance > 0:
        fin = min(95, 60 + int(avoidance / 60_000))
    elif fi_type == "SAVING" and savings > 0:
        fin = min(90, 55 + int(savings / 60_000))
    elif action_code == "DEFER":
        fin = 60
    elif action_code in ("BUY_NOW", "BUY_FORWARD"):
        fin = 52
    else:
        fin = 50

    inv_stab = min(95, conf_score + 8)

    risk_map  = {"CRITICAL": 10, "HIGH": 35, "MEDIUM": 65, "LOW": 92}
    sup_risk  = risk_map.get(risk_level, 50)

    if risk_appetite == "Conservative":
        w = (0.40, 0.15, 0.25, 0.20)
    elif risk_appetite == "Aggressive":
        w = (0.15, 0.40, 0.20, 0.25)
    else:
        w = (0.30, 0.25, 0.20, 0.25)

    overall = int(w[0] * op_safe + w[1] * fin + w[2] * inv_stab + w[3] * sup_risk)

    return {
        "operational_safety":  op_safe,
        "financial_benefit":   fin,
        "inventory_stability": inv_stab,
        "supply_risk":         sup_risk,
        "overall":             overall,
    }


def _pse6b_diff_lines(sim_params: dict, prod_so: dict) -> list:
    """Return executive-language bullets describing what changed."""
    lines = []
    D = _PSE6B_PROD_DEFAULTS

    delta_cons = sim_params["total_consumption"] - D["total_consumption"]
    if abs(delta_cons) > 0.5:
        verb   = "Increasing" if delta_cons > 0 else "Reducing"
        effect = "increases" if delta_cons > 0 else "decreases"
        lines.append(
            f"{verb} daily consumption by {abs(delta_cons):.1f} t/day {effect} "
            f"all recommended purchase quantities proportionally."
        )

    delta_lt_l = sim_params["local_lead_time"] - D["local_lead_time"]
    if delta_lt_l:
        verb   = "Extending" if delta_lt_l > 0 else "Reducing"
        effect = ("advances the procurement trigger and raises the local reorder point"
                  if delta_lt_l > 0 else
                  "relaxes the local procurement window and lowers the reorder point")
        lines.append(
            f"{verb} local lead time by {abs(delta_lt_l)} days — {effect}."
        )

    delta_lt_i = sim_params["imported_lead_time"] - D["imported_lead_time"]
    if delta_lt_i:
        verb   = "Adding" if delta_lt_i > 0 else "Removing"
        effect = ("advances imported procurement by approximately one order cycle "
                  "and increases shortage risk"
                  if delta_lt_i > 0 else
                  "reduces the imported reorder threshold, relaxing imported urgency")
        lines.append(
            f"{verb} {abs(delta_lt_i)} days to imported lead time — {effect}."
        )

    delta_price_pct = ((sim_params["current_price"] - D["current_price"])
                       / max(D["current_price"], 0.01) * 100)
    if abs(delta_price_pct) > 1.5:
        verb   = "Raising" if delta_price_pct > 0 else "Lowering"
        effect = ("weakens the case for forward purchasing"
                  if delta_price_pct > 0 else
                  "strengthens the buy-now signal")
        lines.append(
            f"{verb} the market price by {delta_price_pct:+.1f}% to "
            f"USD {sim_params['current_price']:.3f}/lb — {effect}."
        )

    h1_chg = ((sim_params["forecast_h1"] - sim_params["current_price"])
              / max(sim_params["current_price"], 0.01) * 100)
    if abs(h1_chg) > 2:
        outlook = "rising" if h1_chg > 0 else "falling"
        effect  = ("engine will favour forward purchases to lock in today's price"
                   if h1_chg > 0 else
                   "engine will favour deferral to capture the projected lower price")
        lines.append(
            f"Short-term (H1) price outlook {outlook} ({h1_chg:+.1f}%) — {effect}."
        )

    h3_chg = ((sim_params["forecast_h3"] - sim_params["current_price"])
              / max(sim_params["current_price"], 0.01) * 100)
    if abs(h3_chg) > 2 and abs(h3_chg - h1_chg) > 3:
        outlook = "rising" if h3_chg > 0 else "falling"
        effect  = ("supports early commitment of imported volume"
                   if h3_chg > 0 else
                   "creates opportunity to defer imported orders and reduce near-term cash outlay")
        lines.append(
            f"3-month (H3) imported outlook {outlook} ({h3_chg:+.1f}%) — {effect}."
        )

    delta_mix = sim_params["local_mix_pct"] - D["local_mix_pct"]
    if abs(delta_mix) > 1.5:
        verb   = "Increasing" if delta_mix > 0 else "Reducing"
        effect = ("shifts procurement toward local cotton, reducing imported volume requirements"
                  if delta_mix > 0 else
                  "shifts volume toward imported cotton, reducing local procurement requirements")
        lines.append(
            f"{verb} local cotton mix to {sim_params['local_mix_pct']:.0f}% — {effect}."
        )

    max_storage = sim_params.get("max_storage")
    if max_storage:
        effect = ("creates additional headroom for forward buying"
                  if max_storage > 30_000 else
                  "constrains opportunistic forward buying capacity")
        lines.append(
            f"Storage capacity set to {max_storage:,.0f} t — {effect}."
        )

    prod_local_inv = prod_so.get("local_inventory_tons", 0)
    prod_imp_inv   = prod_so.get("imported_inventory_tons", 0)
    if abs(sim_params["local_inventory"] - prod_local_inv) > 50:
        delta = sim_params["local_inventory"] - prod_local_inv
        verb  = "Adding" if delta > 0 else "Reducing"
        lines.append(
            f"{verb} {abs(delta):,.0f} t to local inventory position — "
            f"{'improves days cover and reduces urgency' if delta > 0 else 'reduces days cover and may increase urgency'}."
        )
    if abs(sim_params["imported_inventory"] - prod_imp_inv) > 50:
        delta = sim_params["imported_inventory"] - prod_imp_inv
        verb  = "Adding" if delta > 0 else "Reducing"
        lines.append(
            f"{verb} {abs(delta):,.0f} t to imported inventory — "
            f"{'improves imported cover and may defer the next order' if delta > 0 else 'tightens imported cover and may advance urgency'}."
        )

    if not lines:
        lines.append(
            "Parameters unchanged from production defaults — simulation matches "
            "the live production recommendation exactly."
        )

    return lines


def _pse6b_adopt(sc_l: dict, sc_i: dict, pc_l: dict, pc_i: dict,
                 risk_appetite: str) -> dict:
    """Return YES / NO / PARTIALLY verdict with explanation."""
    sim_avg  = (sc_l["overall"]  + sc_i["overall"])  / 2
    prod_avg = (pc_l["overall"]  + pc_i["overall"])  / 2
    delta    = sim_avg - prod_avg

    dims = ("operational_safety", "financial_benefit", "inventory_stability", "supply_risk")
    _labels = {
        "operational_safety":  "Operational Safety",
        "financial_benefit":   "Financial Benefit",
        "inventory_stability": "Inventory Stability",
        "supply_risk":         "Supply Risk",
    }
    better = [d for d in dims if (sc_l[d] + sc_i[d]) > (pc_l[d] + pc_i[d]) + 5]
    worse  = [d for d in dims if (sc_l[d] + sc_i[d]) < (pc_l[d] + pc_i[d]) - 5]

    if delta >= 8:
        verdict = "YES"
        color   = "#059669"
        explanation = (
            f"This simulation improves the overall procurement strategy score by "
            f"{delta:.0f} points ({prod_avg:.0f} → {sim_avg:.0f}). "
        )
        if better:
            explanation += (
                f"It outperforms production on: "
                f"{', '.join(_labels[d] for d in better)}. "
            )
        if worse:
            explanation += (
                f"Trade-off accepted — lower scores on: "
                f"{', '.join(_labels[d] for d in worse)}. "
            )
        explanation += "Management should consider adopting these parameters in the next cycle."
    elif delta <= -8:
        verdict = "NO"
        color   = "#dc2626"
        explanation = (
            f"This simulation reduces the overall strategy score by "
            f"{abs(delta):.0f} points ({prod_avg:.0f} → {sim_avg:.0f}). "
        )
        if worse:
            explanation += (
                f"Underperforms production on: "
                f"{', '.join(_labels[d] for d in worse)}. "
            )
        explanation += "The production recommendation remains the stronger strategy."
    else:
        verdict = "PARTIALLY"
        color   = "#b45309"
        explanation = (
            f"The simulation produces a comparable outcome (score delta: {delta:+.0f} pts). "
        )
        if better:
            explanation += (
                f"Elements worth adopting: improvements in "
                f"{', '.join(_labels[d] for d in better)}. "
            )
        if worse:
            explanation += (
                f"Elements to avoid: weaker performance on "
                f"{', '.join(_labels[d] for d in worse)}. "
            )
        explanation += "Selective adoption of specific parameters may capture the upside without the downside."

    if risk_appetite == "Conservative" and verdict == "YES":
        explanation += (
            "  Note (Conservative appetite): verify Operational Safety score "
            "before committing."
        )
    elif risk_appetite == "Aggressive" and verdict == "NO":
        explanation += (
            "  Note (Aggressive appetite): financial upside in this simulation "
            "may still merit partial consideration."
        )

    return {
        "verdict":  verdict,
        "color":    color,
        "explanation": explanation,
        "sim_avg":  sim_avg,
        "prod_avg": prod_avg,
        "delta":    delta,
    }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _pse6b_render_comparison_table(prod_dec: dict, sim_dec: dict,
                                   label: str) -> None:
    """Side-by-side production vs simulation comparison for one source."""
    p_fi  = prod_dec.get("expected_cost_impact", {})
    s_fi  = sim_dec.get("expected_cost_impact", {})
    p_amt = p_fi.get("cost_avoidance_usd") or p_fi.get("expected_savings_usd") or 0
    s_amt = s_fi.get("cost_avoidance_usd") or s_fi.get("expected_savings_usd") or 0

    rows = [
        ("Action",
         prod_dec.get("action", "—"),
         sim_dec.get("action", "—")),
        ("Urgency",
         prod_dec.get("urgency", "—"),
         sim_dec.get("urgency", "—")),
        ("Confidence",
         f"{prod_dec.get('confidence',{}).get('score',0)}/100"
         f" {prod_dec.get('confidence',{}).get('label','')}",
         f"{sim_dec.get('confidence',{}).get('score',0)}/100"
         f" {sim_dec.get('confidence',{}).get('label','')}"),
        ("Qty Now (t)",
         f"{prod_dec.get('qty_now_tons',0):,.0f}",
         f"{sim_dec.get('qty_now_tons',0):,.0f}"),
        ("Deferred (t)",
         f"{prod_dec.get('qty_later_tons',0):,.0f}",
         f"{sim_dec.get('qty_later_tons',0):,.0f}"),
        ("Order Date",
         prod_dec.get("order_date") or "—",
         sim_dec.get("order_date") or "—"),
        ("Expected Arrival",
         prod_dec.get("expected_arrival") or "—",
         sim_dec.get("expected_arrival") or "—"),
        ("Financial Impact",
         p_fi.get("impact_type", "NONE"),
         s_fi.get("impact_type", "NONE")),
        ("Impact (USD)",
         f"{p_amt:,.0f}" if p_amt else "—",
         f"{s_amt:,.0f}" if s_amt else "—"),
        ("Risk Level",
         prod_dec.get("risk", {}).get("level", "—"),
         sim_dec.get("risk", {}).get("level", "—")),
    ]

    tbl = (
        f"<div style='font-size:0.72rem;font-weight:700;color:#1d4ed8;"
        f"letter-spacing:0.6px;text-transform:uppercase;margin-bottom:6px;'>"
        f"{label}</div>"
        f"<table style='width:100%;border-collapse:collapse;font-size:0.75rem;'>"
        f"<thead><tr>"
        f"<th style='text-align:left;padding:5px 7px;background:#f8fafc;"
        f"color:#64748b;font-weight:700;border-bottom:2px solid #e2e8f0;'>Metric</th>"
        f"<th style='text-align:right;padding:5px 7px;background:#eff6ff;"
        f"color:#1d4ed8;font-weight:700;border-bottom:2px solid #bfdbfe;'>Production</th>"
        f"<th style='text-align:right;padding:5px 7px;background:#f0fdf4;"
        f"color:#059669;font-weight:700;border-bottom:2px solid #86efac;'>Simulation</th>"
        f"</tr></thead><tbody>"
    )
    for i, (metric, pval, sval) in enumerate(rows):
        bg      = "#ffffff" if i % 2 == 0 else "#f8fafc"
        changed = (pval != sval)
        sstyle  = "font-weight:700;color:#059669;" if changed else "color:#374151;"
        tbl += (
            f"<tr style='background:{bg};'>"
            f"<td style='padding:5px 7px;color:#374151;"
            f"border-bottom:1px solid #f1f5f9;'>{metric}</td>"
            f"<td style='padding:5px 7px;text-align:right;color:#374151;"
            f"border-bottom:1px solid #f1f5f9;'>{pval}</td>"
            f"<td style='padding:5px 7px;text-align:right;{sstyle}"
            f"border-bottom:1px solid #f1f5f9;'>{sval}</td>"
            f"</tr>"
        )
    tbl += "</tbody></table>"
    st.markdown(tbl, unsafe_allow_html=True)


def _pse6b_render_score_card(sc_l: dict, sc_i: dict,
                              pc_l: dict, pc_i: dict,
                              risk_appetite: str) -> None:
    """5-dimension score card with overlaid production vs simulation bars."""
    sim_avg  = int((sc_l["overall"]  + sc_i["overall"])  / 2)
    prod_avg = int((pc_l["overall"]  + pc_i["overall"])  / 2)
    delta    = sim_avg - prod_avg
    dc       = "#059669" if delta >= 0 else "#dc2626"
    ds       = "+" if delta >= 0 else ""

    st.markdown(
        f"<div style='display:flex;justify-content:space-between;"
        f"align-items:center;margin-bottom:0.6rem;'>"
        f"<div style='font-size:0.72rem;font-weight:700;color:#94a3b8;"
        f"letter-spacing:0.8px;text-transform:uppercase;'>Scenario Score</div>"
        f"<div style='display:flex;gap:10px;align-items:center;'>"
        f"<span style='font-size:0.72rem;color:#64748b;'>Prod <b>{prod_avg}</b></span>"
        f"<span style='font-size:0.72rem;color:{dc};font-weight:700;'>"
        f"Sim <b>{sim_avg}</b> ({ds}{delta})</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    dim_defs = [
        ("Operational Safety",   "operational_safety",   "Safety from procurement shortages"),
        ("Financial Benefit",    "financial_benefit",    "Expected cost avoidance or savings"),
        ("Inventory Stability",  "inventory_stability",  "Confidence in inventory position"),
        ("Supply Risk",          "supply_risk",          "Overall supply chain risk level"),
    ]
    for lbl, key, desc in dim_defs:
        sim_v  = int((sc_l[key] + sc_i[key]) / 2)
        prod_v = int((pc_l[key] + pc_i[key]) / 2)
        d_dim  = sim_v - prod_v
        dc_dim = "#059669" if d_dim >= 0 else "#dc2626"
        ds_dim = "+" if d_dim >= 0 else ""
        bar_c  = "#059669" if sim_v >= prod_v else "#dc2626"
        st.markdown(
            f"<div style='margin-bottom:0.5rem;'>"
            f"<div style='display:flex;justify-content:space-between;"
            f"align-items:baseline;margin-bottom:3px;'>"
            f"<span style='font-size:0.74rem;color:#374151;font-weight:600;'>{lbl}</span>"
            f"<span style='font-size:0.68rem;color:{dc_dim};font-weight:700;'>"
            f"Prod {prod_v} → Sim {sim_v} ({ds_dim}{d_dim})</span>"
            f"</div>"
            f"<div style='position:relative;height:8px;background:#e2e8f0;"
            f"border-radius:4px;margin-bottom:2px;'>"
            f"<div style='position:absolute;left:0;top:0;height:8px;"
            f"width:{prod_v}%;background:#1d4ed8;border-radius:4px;opacity:0.35;'></div>"
            f"<div style='position:absolute;left:0;top:0;height:8px;"
            f"width:{sim_v}%;background:{bar_c};border-radius:4px;opacity:0.80;'></div>"
            f"</div>"
            f"<div style='font-size:0.63rem;color:#94a3b8;'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<div style='font-size:0.65rem;color:#94a3b8;margin-top:4px;'>"
        f"Risk Appetite: <b>{risk_appetite}</b> — dimension weights adjusted accordingly.</div>",
        unsafe_allow_html=True,
    )


def _pse6b_render_diff_analysis(diff_lines: list) -> None:
    bullets = "".join(
        f"<div style='display:flex;gap:8px;padding:6px 0;"
        f"border-bottom:1px solid #f1f5f9;align-items:flex-start;'>"
        f"<span style='color:#1d4ed8;font-size:0.8rem;flex-shrink:0;"
        f"margin-top:1px;'>&#9654;</span>"
        f"<span style='font-size:0.78rem;color:#374151;line-height:1.55;'>{l}</span>"
        f"</div>"
        for l in diff_lines
    )
    st.markdown(
        f"<div style='background:#f8fafc;border-radius:8px;"
        f"border:1px solid #e2e8f0;padding:0.85rem 1rem;'>{bullets}</div>",
        unsafe_allow_html=True,
    )


def _pse6b_render_adopt_card(adopt: dict) -> None:
    vc   = adopt["color"]
    vd   = adopt["verdict"]
    v_bg = {"YES": "#f0fdf4", "NO": "#fef2f2", "PARTIALLY": "#fffbeb"}.get(vd, "#f8fafc")
    v_bd = {"YES": "#86efac", "NO": "#fca5a5", "PARTIALLY": "#fde68a"}.get(vd, "#e2e8f0")
    st.markdown(
        f"<div style='background:{v_bg};border:1px solid {v_bd};"
        f"border-left:5px solid {vc};border-radius:8px;padding:0.85rem 1.1rem;'>"
        f"<div style='font-size:0.68rem;font-weight:700;color:#94a3b8;"
        f"letter-spacing:0.8px;text-transform:uppercase;margin-bottom:4px;'>"
        f"Would I adopt this strategy?</div>"
        f"<div style='font-size:1.05rem;font-weight:900;color:{vc};"
        f"letter-spacing:-0.3px;margin-bottom:0.5rem;'>{vd}</div>"
        f"<div style='font-size:0.78rem;color:#374151;line-height:1.6;'>"
        f"{adopt['explanation']}</div>"
        f"<div style='display:flex;gap:16px;margin-top:0.55rem;'>"
        f"<span style='font-size:0.68rem;color:#64748b;'>"
        f"Production score: <b>{adopt['prod_avg']:.0f}</b></span>"
        f"<span style='font-size:0.68rem;color:{vc};font-weight:700;'>"
        f"Simulation score: <b>{adopt['sim_avg']:.0f}</b></span>"
        f"<span style='font-size:0.68rem;color:{vc};font-weight:700;'>"
        f"Delta: {adopt['delta']:+.0f} pts</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )


def _pse6b_render_saved_scenarios() -> None:
    saved = st.session_state.get("pse6b_saved", [])
    if not saved:
        st.caption(
            "No saved simulations yet.  "
            "Run a simulation and click **Save** to store it here."
        )
        return

    for i, s in enumerate(reversed(saved)):
        real_i = len(saved) - 1 - i
        with st.expander(
            f"{s['name']}  ·  {s['created']}  ·  Score {s['score']}/100  ·  {s['verdict']}",
            expanded=False,
        ):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Parameters**")
                for k, v in s.get("params", {}).items():
                    st.markdown(f"- {k.replace('_',' ').title()}: **{v}**")
            with col2:
                st.markdown("**Outcome**")
                o = s.get("outcome", {})
                st.markdown(
                    f"- LOCAL: **{o.get('local_action','—')}** "
                    f"{o.get('local_qty_now',0):,.0f} t now"
                )
                st.markdown(
                    f"- IMPORTED: **{o.get('imported_action','—')}** "
                    f"{o.get('imported_qty_now',0):,.0f} t now"
                )
                st.markdown(f"- Portfolio Urgency: **{o.get('portfolio_urgency','—')}**")
            st.markdown(
                f"*{s['verdict']}* — {s['explanation'][:180]}…"
            )
            if st.button("Delete this scenario", key=f"pse6b_del_{real_i}"):
                st.session_state["pse6b_saved"].pop(real_i)
                st.rerun()


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def render_pse6b_whatif_panel() -> None:
    """
    PSE-6B — Executive What-If Simulation Laboratory.

    Allows management to test procurement scenarios without changing production
    calculations.  Production recommendation is never overwritten.

    Flow:
      1. Load production baseline from cached workbook run (PSE-5B)
      2. Render parameter controls in a form
      3. On submit: run simulation via existing engine functions with overrides
      4. Display: comparison table, scenario score, diff analysis, adopt verdict
      5. Optionally save simulation to session state
    """
    import datetime as _dt

    today_str = _dt.date.today().isoformat()

    with st.spinner("Loading production baseline…"):
        prod_so = _pse6b_get_prod_so(today_str)

    if not prod_so.get("ok"):
        st.info(
            "Inventory data unavailable. "
            "Verify the inventory workbook and refresh."
        )
        return

    with st.spinner("Loading production decisions…"):
        prod_pse6a = _run_pse5b_cached(today_str)

    if not prod_pse6a.get("ok"):
        st.error(
            f"**Could not load production decisions.**\n\n"
            f"```\n{prod_pse6a.get('error','Unknown error')}\n```"
        )
        return

    prod_er        = prod_pse6a["report"]
    prod_local_dec = prod_er.get("local",    {})
    prod_imp_dec   = prod_er.get("imported", {})

    # ── Parameter Controls ────────────────────────────────────────────────────
    st.markdown(
        "<div style='background:#1e293b;border-radius:10px;"
        "padding:0.75rem 1.2rem;margin-bottom:1rem;'>"
        "<div style='color:#f1f5f9;font-size:0.72rem;font-weight:700;"
        "letter-spacing:1px;text-transform:uppercase;'>"
        "What-If Simulation Parameters</div>"
        "<div style='color:#94a3b8;font-size:0.7rem;margin-top:2px;'>"
        "Adjust parameters and click <b style='color:#e2e8f0;'>Run Simulation</b>. "
        "Production recommendation is not affected.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    prod_local_inv = float(prod_so["local_inventory_tons"])
    prod_imp_inv   = float(prod_so["imported_inventory_tons"])

    with st.form("pse6b_sim_form"):

        st.markdown("**Market Prices**")
        pc1, pc2, pc3, pc4 = st.columns(4)
        with pc1:
            current_price = st.number_input(
                "Current Price (USD/lb)",
                min_value=0.30, max_value=2.00, value=0.78, step=0.01, format="%.3f",
            )
        with pc2:
            forecast_h1 = st.number_input(
                "Forecast H1 (USD/lb)",
                min_value=0.30, max_value=2.00, value=0.78, step=0.01, format="%.3f",
                help="1-month forecast used for local cotton signal",
            )
        with pc3:
            forecast_h3 = st.number_input(
                "Forecast H3 (USD/lb)",
                min_value=0.30, max_value=2.00, value=0.78, step=0.01, format="%.3f",
                help="3-month forecast used for imported cotton signal",
            )
        with pc4:
            pkr_rate = st.number_input(
                "PKR/USD Rate",
                min_value=100.0, max_value=500.0, value=281.0, step=1.0, format="%.1f",
            )

        st.markdown("**Inventory Position**")
        ic1, ic2 = st.columns(2)
        with ic1:
            local_inv = st.number_input(
                "Local Inventory (t)",
                min_value=0.0, max_value=50_000.0,
                value=prod_local_inv, step=100.0,
                help=f"Live production value: {prod_local_inv:,.0f} t",
            )
        with ic2:
            imp_inv = st.number_input(
                "Imported Inventory (t)",
                min_value=0.0, max_value=50_000.0,
                value=prod_imp_inv, step=100.0,
                help=f"Live production value: {prod_imp_inv:,.0f} t",
            )

        st.markdown("**Operational Parameters**")
        oc1, oc2, oc3, oc4, oc5 = st.columns(5)
        with oc1:
            total_consumption = st.number_input(
                "Daily Consumption (t/day)",
                min_value=50.0, max_value=250.0, value=110.0, step=1.0,
            )
        with oc2:
            local_mix_pct = st.slider(
                "Local Mix (%)", min_value=20, max_value=80, value=45, step=1,
            )
        with oc3:
            local_lead_time = st.number_input(
                "Local Lead Time (days)",
                min_value=1, max_value=90, value=10, step=1,
            )
        with oc4:
            imported_lead_time = st.number_input(
                "Imported Lead Time (days)",
                min_value=14, max_value=270, value=90, step=7,
            )
        with oc5:
            max_storage = st.number_input(
                "Max Storage (t)",
                min_value=0.0, max_value=100_000.0, value=0.0, step=500.0,
                help="0 = no storage constraint",
            )

        st.markdown("**Risk Appetite**")
        risk_appetite = st.radio(
            "Risk Appetite",
            ["Conservative", "Balanced", "Aggressive"],
            index=1,
            horizontal=True,
            label_visibility="collapsed",
        )

        scenario_name = st.text_input(
            "Scenario Name (for saving)", value="Scenario A", max_chars=60,
        )

        btn1, btn2, _ = st.columns([2, 1, 3])
        with btn1:
            submitted = st.form_submit_button(
                "Run Simulation", type="primary", use_container_width=True,
            )
        with btn2:
            save_requested = st.form_submit_button(
                "Save", use_container_width=True,
            )

    # ── Execute simulation ────────────────────────────────────────────────────
    if submitted or save_requested:
        sim_params = {
            "current_price":      current_price,
            "forecast_h1":        forecast_h1,
            "forecast_h3":        forecast_h3,
            "pkr_rate":           pkr_rate,
            "local_inventory":    local_inv,
            "imported_inventory": imp_inv,
            "total_consumption":  total_consumption,
            "local_mix_pct":      float(local_mix_pct),
            "local_lead_time":    int(local_lead_time),
            "imported_lead_time": int(imported_lead_time),
            "max_storage":        float(max_storage) if max_storage > 0 else None,
            "risk_appetite":      risk_appetite,
        }
        with st.spinner("Running simulation…"):
            sim_result = _pse6b_run_simulation(prod_so, sim_params, today_str)

        if not sim_result.get("ok"):
            st.error(
                f"**Simulation error.**\n\n"
                f"```\n{sim_result.get('error','Unknown error')}\n```"
            )
        else:
            st.session_state["pse6b_sim_result"] = sim_result
            st.session_state["pse6b_sim_params"]  = sim_params

            if save_requested:
                if "pse6b_saved" not in st.session_state:
                    st.session_state["pse6b_saved"] = []
                sim_er  = sim_result["exec_report"]
                sim_l   = sim_er.get("local",    {})
                sim_i   = sim_er.get("imported",  {})
                sc_l    = _pse6b_score(sim_l, risk_appetite)
                sc_i    = _pse6b_score(sim_i, risk_appetite)
                pc_l    = _pse6b_score(prod_local_dec, risk_appetite)
                pc_i    = _pse6b_score(prod_imp_dec,   risk_appetite)
                adopt   = _pse6b_adopt(sc_l, sc_i, pc_l, pc_i, risk_appetite)
                avg_sc  = int((sc_l["overall"] + sc_i["overall"]) / 2)
                st.session_state["pse6b_saved"].append({
                    "name":        scenario_name,
                    "created":     today_str,
                    "score":       avg_sc,
                    "verdict":     adopt["verdict"],
                    "explanation": adopt["explanation"],
                    "params": {
                        k: v for k, v in sim_params.items()
                        if k not in ("local_inventory", "imported_inventory")
                    },
                    "outcome": {
                        "local_action":      sim_l.get("action", "—"),
                        "local_qty_now":     sim_l.get("qty_now_tons", 0),
                        "imported_action":   sim_i.get("action", "—"),
                        "imported_qty_now":  sim_i.get("qty_now_tons", 0),
                        "portfolio_urgency": sim_er.get("portfolio_urgency", "—"),
                    },
                })
                st.success(f"Simulation '{scenario_name}' saved.")

    # ── Display results ───────────────────────────────────────────────────────
    sim_result_state = st.session_state.get("pse6b_sim_result")
    sim_params_state = st.session_state.get("pse6b_sim_params", {})

    if sim_result_state and sim_result_state.get("ok"):
        sim_er  = sim_result_state["exec_report"]
        sim_l   = sim_er.get("local",    {})
        sim_i   = sim_er.get("imported", {})
        rap     = sim_params_state.get("risk_appetite", "Balanced")

        sc_l  = _pse6b_score(sim_l,           rap)
        sc_i  = _pse6b_score(sim_i,           rap)
        pc_l  = _pse6b_score(prod_local_dec,  rap)
        pc_i  = _pse6b_score(prod_imp_dec,    rap)
        adopt = _pse6b_adopt(sc_l, sc_i, pc_l, pc_i, rap)

        # Comparison tables
        st.markdown(
            "<div style='font-size:0.72rem;font-weight:700;color:#94a3b8;"
            "letter-spacing:0.8px;text-transform:uppercase;"
            "margin:1rem 0 0.4rem 0;'>Production vs Simulation</div>",
            unsafe_allow_html=True,
        )
        cmp1, cmp2 = st.columns(2, gap="medium")
        with cmp1:
            _pse6b_render_comparison_table(prod_local_dec, sim_l, "LOCAL COTTON")
        with cmp2:
            _pse6b_render_comparison_table(prod_imp_dec, sim_i, "IMPORTED COTTON")

        st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

        # Score + Adopt
        score_col, adopt_col = st.columns([1, 1], gap="medium")
        with score_col:
            _pse6b_render_score_card(sc_l, sc_i, pc_l, pc_i, rap)
        with adopt_col:
            _pse6b_render_adopt_card(adopt)

        # Diff analysis
        st.markdown(
            "<div style='font-size:0.72rem;font-weight:700;color:#94a3b8;"
            "letter-spacing:0.8px;text-transform:uppercase;"
            "margin:0.8rem 0 0.4rem 0;'>Executive Difference Summary</div>",
            unsafe_allow_html=True,
        )
        _pse6b_render_diff_analysis(_pse6b_diff_lines(sim_params_state, prod_so))

    else:
        st.info(
            "Configure parameters above and click **Run Simulation** "
            "to generate a what-if analysis."
        )

    # Saved scenarios
    st.markdown(
        "<div style='font-size:0.72rem;font-weight:700;color:#94a3b8;"
        "letter-spacing:0.8px;text-transform:uppercase;"
        "margin:1.2rem 0 0.4rem 0;'>Saved Simulations</div>",
        unsafe_allow_html=True,
    )
    _pse6b_render_saved_scenarios()

    st.caption(
        f"PSE-6B What-If Lab  ·  Production data: live inventory  ·  {today_str}  ·  "
        f"Engine: PSE-5A/5B (simulation mode — production values unchanged)"
    )


# ===========================================================================
# PSE-7A  EXECUTIVE MONITORING & EARLY WARNING SYSTEM
# ===========================================================================
# Consumes PSE-5A (ScenarioReport) and PSE-5B (ExecutiveReport) outputs.
# Generates proactive alerts and an executive watchlist.
# No procurement calculations are performed here.
# ===========================================================================

_SEV_COLORS = {
    "CRITICAL": ("#dc2626", "#fef2f2", "#fca5a5"),
    "HIGH":     ("#b45309", "#fffbeb", "#fde68a"),
    "WARNING":  ("#7c3aed", "#f5f3ff", "#ddd6fe"),
    "NOTICE":   ("#1d4ed8", "#eff6ff", "#bfdbfe"),
    "INFO":     ("#059669", "#f0fdf4", "#86efac"),
}

_CAT_ICON = {
    "INVENTORY":   "&#9679;",
    "PRICE":       "&#9650;",
    "TIMING":      "&#9719;",
    "OPPORTUNITY": "&#9733;",
    "RISK":        "&#9888;",
    "SYSTEM":      "&#9881;",
}


@st.cache_data(ttl=900, show_spinner=False)
def _run_pse7a_cached(
    today_str: str,
    best_sim_score: Optional[int] = None,
    prod_score: Optional[int] = None,
) -> dict:
    """Run PSE-7A monitoring pipeline.  Cached 15 min (fresher than PSE-6A).

    All upstream data (executive report, scenario report, inventory) is read
    from _run_pse5b_cached() — no additional I/O occurs here.
    """
    try:
        from datetime import date as _date
        from procurement_monitoring_engine import generate_monitoring_report

        # All data comes from the shared 30-min cache — zero additional I/O
        prod_result = _run_pse5b_cached(today_str)

        if not prod_result.get("ok"):
            return {"ok": False, "error": prod_result.get("error")}

        today = _date.fromisoformat(today_str)
        inv   = prod_result["inventory"]

        report = generate_monitoring_report(
            exec_report_dict=prod_result["report"],
            scenario_report_dict=prod_result["scenario"],
            local_inventory_tons=inv["local_inventory_tons"],
            imported_inventory_tons=inv["imported_inventory_tons"],
            total_inventory_tons=inv["total_inventory_tons"],
            today=today,
            best_sim_score=best_sim_score,
            prod_score=prod_score,
        )
        return {"ok": True, "report": report.to_dict()}
    except Exception as exc:
        import traceback
        return {"ok": False, "error": str(exc) + "\n" + traceback.format_exc()}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _sev_badge(severity: str) -> str:
    tc, bg, bd = _SEV_COLORS.get(severity, ("#475569", "#f1f5f9", "#e2e8f0"))
    return (
        f"<span style='display:inline-block;padding:2px 10px;"
        f"border-radius:12px;font-size:0.7rem;font-weight:700;"
        f"letter-spacing:0.4px;color:{tc};background:{bg};"
        f"border:1.5px solid {bd};'>{severity}</span>"
    )


def _pse7a_render_severity_strip(count_by_sev: dict, highest: str) -> None:
    """Horizontal severity count strip at top of panel."""
    cells = ""
    for sev in ("CRITICAL", "HIGH", "WARNING", "NOTICE", "INFO"):
        cnt = count_by_sev.get(sev, 0)
        tc, bg, bd = _SEV_COLORS.get(sev, ("#475569", "#f1f5f9", "#e2e8f0"))
        opacity = "1.0" if cnt > 0 else "0.35"
        cells += (
            f"<div style='background:{bg};border:1.5px solid {bd};"
            f"border-radius:8px;padding:6px 12px;text-align:center;"
            f"opacity:{opacity};min-width:70px;'>"
            f"<div style='font-size:1.1rem;font-weight:900;color:{tc};'>{cnt}</div>"
            f"<div style='font-size:0.62rem;font-weight:700;color:{tc};"
            f"letter-spacing:0.5px;'>{sev}</div>"
            f"</div>"
        )
    st.markdown(
        f"<div style='display:flex;gap:8px;align-items:stretch;"
        f"margin-bottom:1rem;flex-wrap:wrap;'>{cells}</div>",
        unsafe_allow_html=True,
    )


def _pse7a_render_alerts(alerts: list) -> None:
    if not alerts:
        st.success("No active alerts.  All monitoring checks are within normal thresholds.")
        return

    for a in alerts:
        tc, bg, bd = _SEV_COLORS.get(a["severity"], ("#475569", "#f1f5f9", "#e2e8f0"))
        with st.expander(
            f"{a['severity']}  ·  {a['title']}  [{a['source']}]",
            expanded=(a["severity"] in ("CRITICAL", "HIGH")),
        ):
            st.markdown(
                f"<div style='background:{bg};border:1px solid {bd};"
                f"border-left:4px solid {tc};border-radius:8px;"
                f"padding:0.75rem 1rem;'>"

                f"<div style='display:flex;justify-content:space-between;"
                f"align-items:center;margin-bottom:0.6rem;'>"
                f"{_sev_badge(a['severity'])}"
                f"<span style='font-size:0.68rem;color:#94a3b8;'>"
                f"Alert ID: {a['alert_id']}  ·  {a['triggered_at']}</span>"
                f"</div>"

                f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;"
                f"gap:10px;margin-bottom:0.6rem;'>"

                f"<div style='background:#ffffff;border-radius:6px;padding:8px 10px;"
                f"border:1px solid {bd};'>"
                f"<div style='font-size:0.63rem;color:#94a3b8;font-weight:700;"
                f"letter-spacing:0.5px;margin-bottom:3px;'>REASON</div>"
                f"<div style='font-size:0.74rem;color:#374151;line-height:1.5;'>"
                f"{a['reason']}</div></div>"

                f"<div style='background:#ffffff;border-radius:6px;padding:8px 10px;"
                f"border:1px solid {bd};'>"
                f"<div style='font-size:0.63rem;color:#94a3b8;font-weight:700;"
                f"letter-spacing:0.5px;margin-bottom:3px;'>BUSINESS IMPACT</div>"
                f"<div style='font-size:0.74rem;color:#374151;line-height:1.5;'>"
                f"{a['business_impact']}</div></div>"

                f"<div style='background:#ffffff;border-radius:6px;padding:8px 10px;"
                f"border:1px solid {bd};'>"
                f"<div style='font-size:0.63rem;color:#94a3b8;font-weight:700;"
                f"letter-spacing:0.5px;margin-bottom:3px;'>RECOMMENDED ACTION</div>"
                f"<div style='font-size:0.74rem;color:{tc};font-weight:600;line-height:1.5;'>"
                f"{a['recommended_action']}</div></div>"
                f"</div>"

                f"<div style='display:flex;gap:16px;'>"
                f"<span style='font-size:0.68rem;color:#64748b;'>"
                f"Expected urgency: <b>{a['expected_urgency']}</b></span>"
                + (f"<span style='font-size:0.68rem;color:#64748b;'>"
                   f"Metric: <b>{a['metric_value']:,.1f}</b></span>"
                   if a.get("metric_value") is not None else "")
                + (f"<span style='font-size:0.68rem;color:#64748b;'>"
                   f"Threshold: <b>{a['threshold_value']:,.1f}</b></span>"
                   if a.get("threshold_value") is not None else "")
                + f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


def _pse7a_render_watchlist(watchlist: list) -> None:
    if not watchlist:
        st.caption("Watchlist is clear — nothing requires immediate attention.")
        return

    for w in watchlist:
        tc, bg, bd = _SEV_COLORS.get(w["severity"], ("#475569", "#f1f5f9", "#e2e8f0"))
        icon = _CAT_ICON.get(w["category"], "&#9679;")
        days_s = (
            f"<span style='font-size:0.68rem;font-weight:700;color:{tc};"
            f"background:{bg};border:1px solid {bd};border-radius:8px;"
            f"padding:1px 7px;margin-left:6px;'>"
            f"in {w['days_until_event']}d</span>"
            if w.get("days_until_event") is not None else ""
        )
        st.markdown(
            f"<div style='display:flex;gap:8px;padding:7px 0;"
            f"border-bottom:1px solid #f1f5f9;align-items:flex-start;'>"
            f"<span style='color:{tc};font-size:0.85rem;flex-shrink:0;"
            f"margin-top:1px;'>{icon}</span>"
            f"<div>"
            f"<div style='font-size:0.78rem;color:#0f172a;font-weight:600;'>"
            f"{w['headline']}{days_s}</div>"
            f"<div style='font-size:0.7rem;color:#64748b;margin-top:1px;'>"
            f"{w['detail']}</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )


def _pse7a_render_upcoming_events(events: list) -> None:
    if not events:
        st.caption("No upcoming procurement events.")
        return

    for ev in events:
        urg  = ev.get("urgency", "LOW")
        tc   = _U_COLOR.get(urg, ("#475569", "#f1f5f9", "#e2e8f0"))[0]
        days = ev.get("days_from_now", 0)
        days_s = "today" if days <= 0 else f"in {days} days"
        arr_s  = (f"  ·  Arrival: {ev['arrival_date']}"
                  if ev.get("arrival_date") else "")
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"align-items:center;padding:7px 10px;"
            f"background:#f8fafc;border-radius:7px;margin-bottom:6px;"
            f"border-left:3px solid {tc};'>"
            f"<div>"
            f"<div style='font-size:0.78rem;font-weight:700;color:#0f172a;'>"
            f"{ev.get('description','')}</div>"
            f"<div style='font-size:0.68rem;color:#64748b;margin-top:1px;'>"
            f"Source: {ev.get('source','')}  ·  {ev.get('quantity_tons',0):,.0f} t"
            f"{arr_s}</div>"
            f"</div>"
            f"<div style='text-align:right;'>"
            f"<div style='font-size:0.78rem;font-weight:700;color:{tc};'>{days_s}</div>"
            f"<div style='font-size:0.65rem;color:#94a3b8;'>{urg}</div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _pse7a_render_risk_events(risk_events: list) -> None:
    if not risk_events:
        st.caption("No high or critical risk events.")
        return

    for re in risk_events:
        rl = re.get("risk_level", "INFO")
        tc, bg, bd = _SEV_COLORS.get(rl, ("#475569", "#f1f5f9", "#e2e8f0"))
        st.markdown(
            f"<div style='background:{bg};border:1px solid {bd};"
            f"border-left:4px solid {tc};border-radius:7px;"
            f"padding:8px 12px;margin-bottom:6px;'>"
            f"<div style='display:flex;justify-content:space-between;"
            f"align-items:center;margin-bottom:3px;'>"
            f"<span style='font-size:0.78rem;font-weight:700;color:{tc};'>"
            f"{re.get('headline','')}</span>"
            f"{_sev_badge(rl)}"
            f"</div>"
            f"<div style='font-size:0.72rem;color:#374151;'>{re.get('detail','')}</div>"
            f"<div style='font-size:0.68rem;color:#94a3b8;margin-top:3px;'>"
            f"If ignored: {re.get('if_ignored','')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _pse7a_render_savings(savings_opps: list) -> None:
    if not savings_opps:
        st.caption("No significant savings opportunities currently identified.")
        return

    for op in savings_opps:
        amt = op.get("amount_usd") or 0
        amt_s = f"USD {amt:,.0f}" if amt else "—"
        tc, bg, bd = _SEV_COLORS.get("NOTICE", ("#1d4ed8", "#eff6ff", "#bfdbfe"))
        opp_type_color = (
            "#059669" if op.get("type") == "PRICE_DEFERRAL" else "#1d4ed8"
        )
        st.markdown(
            f"<div style='background:{bg};border:1px solid {bd};"
            f"border-left:4px solid {opp_type_color};border-radius:7px;"
            f"padding:8px 12px;margin-bottom:6px;'>"
            f"<div style='display:flex;justify-content:space-between;"
            f"align-items:center;margin-bottom:3px;'>"
            f"<span style='font-size:0.78rem;font-weight:700;"
            f"color:#0f172a;'>{op.get('headline','')}</span>"
            f"<span style='font-size:0.88rem;font-weight:900;"
            f"color:{opp_type_color};'>{amt_s}</span>"
            f"</div>"
            f"<div style='font-size:0.72rem;color:#374151;'>{op.get('detail','')}</div>"
            f"<div style='font-size:0.68rem;color:{opp_type_color};"
            f"font-weight:600;margin-top:4px;'>&#10003; {op.get('action','')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def render_pse7a_monitoring_panel() -> None:
    """
    PSE-7A — Executive Monitoring & Early Warning System.

    Consumes PSE-5A/5B outputs (cached) and generates:
      1. Alert severity summary strip
      2. Current alerts (sorted by severity, expandable)
      3. Executive watchlist
      4. Upcoming procurement events
      5. Upcoming risk events
      6. Savings opportunities
    """
    import datetime as _dt

    today_str = _dt.date.today().isoformat()

    # Best saved simulation score from session state (optional)
    best_sim  = None
    prod_scr  = None
    saved_sims = st.session_state.get("pse6b_saved", [])
    if saved_sims:
        best_sim = max(s["score"] for s in saved_sims)
        # Compute a rough production score from PSE-6A report
        prod_er = (_run_pse5b_cached(today_str) or {}).get("report", {})
        if prod_er:
            pc_l = _pse6b_score(prod_er.get("local", {}),    "Balanced")
            pc_i = _pse6b_score(prod_er.get("imported", {}), "Balanced")
            prod_scr = int((pc_l["overall"] + pc_i["overall"]) / 2)

    with st.spinner("Running monitoring checks…"):
        result = _run_pse7a_cached(
            today_str,
            best_sim_score=best_sim,
            prod_score=prod_scr,
        )

    if not result.get("ok"):
        st.error(
            f"**Monitoring engine error.**\n\n"
            f"```\n{result.get('error','Unknown error')}\n```"
        )
        return

    rpt      = result["report"]
    alerts   = rpt.get("alerts", [])
    watchlist = rpt.get("watchlist", [])
    events   = rpt.get("upcoming_procurement_events", [])
    risks    = rpt.get("upcoming_risk_events", [])
    savings  = rpt.get("upcoming_savings_opportunities", [])
    summary  = rpt.get("summary", "")
    highest  = rpt.get("highest_severity", "INFO")
    counts   = rpt.get("alert_count_by_severity", {})

    # ── Portfolio status banner ───────────────────────────────────────────────
    tc, bg, bd = _SEV_COLORS.get(highest, ("#475569", "#f1f5f9", "#e2e8f0"))
    st.markdown(
        f"<div style='background:{bg};border:1px solid {bd};"
        f"border-left:5px solid {tc};border-radius:10px;"
        f"padding:0.75rem 1.2rem;margin-bottom:0.75rem;'>"
        f"<div style='font-size:0.68rem;font-weight:700;color:{tc};"
        f"letter-spacing:0.8px;text-transform:uppercase;margin-bottom:4px;'>"
        f"Monitoring Status: {highest}</div>"
        f"<div style='font-size:0.82rem;color:#374151;line-height:1.55;'>"
        f"{summary}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Severity summary strip ────────────────────────────────────────────────
    _pse7a_render_severity_strip(counts, highest)

    # ── Active alerts ─────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.72rem;font-weight:700;color:#94a3b8;"
        "letter-spacing:0.8px;text-transform:uppercase;"
        "margin:0 0 0.4rem 0;'>Current Alerts</div>",
        unsafe_allow_html=True,
    )
    _pse7a_render_alerts(alerts)

    # ── Watchlist + Events (two columns) ─────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.72rem;font-weight:700;color:#94a3b8;"
        "letter-spacing:0.8px;text-transform:uppercase;"
        "margin:1rem 0 0.4rem 0;'>Executive Watchlist</div>",
        unsafe_allow_html=True,
    )
    _pse7a_render_watchlist(watchlist)

    col_ev, col_ri = st.columns(2, gap="medium")
    with col_ev:
        st.markdown(
            "<div style='font-size:0.72rem;font-weight:700;color:#94a3b8;"
            "letter-spacing:0.8px;text-transform:uppercase;"
            "margin:1rem 0 0.4rem 0;'>Upcoming Procurement Events</div>",
            unsafe_allow_html=True,
        )
        _pse7a_render_upcoming_events(events)

    with col_ri:
        st.markdown(
            "<div style='font-size:0.72rem;font-weight:700;color:#94a3b8;"
            "letter-spacing:0.8px;text-transform:uppercase;"
            "margin:1rem 0 0.4rem 0;'>Upcoming Risk Events</div>",
            unsafe_allow_html=True,
        )
        _pse7a_render_risk_events(risks)

    # ── Savings opportunities ─────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.72rem;font-weight:700;color:#94a3b8;"
        "letter-spacing:0.8px;text-transform:uppercase;"
        "margin:1rem 0 0.4rem 0;'>Savings Opportunities</div>",
        unsafe_allow_html=True,
    )
    _pse7a_render_savings(savings)

    st.caption(
        f"PSE-7A Monitoring  ·  {rpt.get('run_date',today_str)}  ·  "
        f"{len(alerts)} alert(s) active  ·  Cache TTL: 15 min"
    )
