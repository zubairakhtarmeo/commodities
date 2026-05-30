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
    <div class='metric-card' style='border-left:4px solid {border_colour};'>
        <div class='metric-label' style='color:{border_colour};'>{label}</div>
        <div class='metric-value' style='color:{_C_TEXT};'>{value}</div>
        <div class='currency-label'>{sublabel}</div>
    </div>"""


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
         "Across active org-commodity pairs", "#7c3aed"),
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
        (c2, "FIBER INVENTORY",     f"{fiber_inv:,.0f}",   "Kgs on hand",              "#7c3aed"),
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
# Main page entry point
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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview",
        "BUY — Action Required",
        "HOLD — Adequate Stock",
        "MONITOR — Attention",
        "Inventory Risk",
        "Full Report",
    ])

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
