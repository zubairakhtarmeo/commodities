from __future__ import annotations

from pathlib import Path
from datetime import datetime
from contextlib import nullcontext
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import requests
from typing import Optional
import json
import sys
import importlib.util

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent / "scripts"))

ARTIFACTS_DIR = Path("artifacts")
RAW_DATA_DIR = Path("data/raw")
EVENTS_DIR = Path("data/events")

# API Configuration
USD_PKR_API = "https://open.er-api.com/v6/latest/USD"  # Free, no API key needed
BACKUP_USD_PKR_API = "https://api.exchangerate-api.com/v4/latest/USD"


def _get_streamlit_secret(key: str) -> Optional[str]:
    """Read a secret from Streamlit secrets or environment variables."""
    import os
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            val = st.secrets.get(key)
            return str(val) if val is not None else None
    except Exception:
        pass
    env_val = os.environ.get(key)
    return str(env_val) if env_val else None


@st.cache_resource
def get_supabase_config() -> tuple[str, str] | None:
    """Return (url, key) for Supabase PostgREST calls, or None if not configured."""
    url = _get_streamlit_secret("SUPABASE_URL")
    key = _get_streamlit_secret("SUPABASE_SERVICE_ROLE_KEY") or _get_streamlit_secret("SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    return (str(url).rstrip("/"), str(key).strip())


def _supabase_headers(key: str) -> dict:
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def _supabase_rest_url(base_url: str, table: str) -> str:
    return f"{base_url}/rest/v1/{table}"


def supabase_rest_select(
    *,
    table: str,
    select: str,
    eq_filters: dict[str, str] | None = None,
    order: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    cfg = get_supabase_config()
    if cfg is None:
        return []
    base_url, key = cfg
    params: dict[str, str] = {"select": select}
    if eq_filters:
        for k, v in eq_filters.items():
            params[k] = f"eq.{v}"
    if order:
        params["order"] = order
    if limit is not None:
        params["limit"] = str(int(limit))

    try:
        resp = requests.get(
            _supabase_rest_url(base_url, table),
            headers=_supabase_headers(key),
            params=params,
            timeout=30,
        )
        if not resp.ok:
            return []
        data = resp.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


def supabase_rest_upsert(*, table: str, rows: list[dict], on_conflict: str | None = None) -> bool:
    cfg = get_supabase_config()
    if cfg is None:
        return False
    base_url, key = cfg
    try:
        params = {}
        if on_conflict:
            params["on_conflict"] = on_conflict
        resp = requests.post(
            _supabase_rest_url(base_url, table),
            headers={
                **_supabase_headers(key),
                "Prefer": "resolution=merge-duplicates,return=minimal",
            },
            params=params,
            data=json.dumps(rows),
            timeout=60,
        )
        return bool(resp.ok)
    except Exception:
        return False


def supabase_is_configured() -> bool:
    return get_supabase_config() is not None


def supabase_upsert_prediction_record(
    *,
    asset: str,
    as_of_date: str,
    target_date: str,
    predicted_value: float,
    actual_value: float | None,
    unit: str,
    model_name: str = "default",
    frequency: str = "daily",
    horizon: str = "1d",
) -> bool:
    """Upsert a prediction record into Supabase.

    Expects a table named `prediction_records` with (at minimum) these columns:
    - asset (text)
    - as_of_date (date or text)
    - target_date (date or text)
    - predicted_value (numeric)
    - actual_value (numeric, nullable)
    - unit (text)
    - model_name (text)
    - frequency (text)
    - horizon (text)

    Recommended unique constraint: (asset, as_of_date, target_date, model_name, horizon)
    """
    row = {
        "asset": asset,
        "as_of_date": as_of_date,
        "target_date": target_date,
        "predicted_value": float(predicted_value),
        "actual_value": float(actual_value) if actual_value is not None else None,
        "unit": unit,
        "model_name": model_name,
        "frequency": frequency,
        "horizon": horizon,
    }

    return supabase_rest_upsert(
        table="prediction_records",
        rows=[row],
        on_conflict="asset,as_of_date,target_date,model_name,horizon",
    )


def supabase_fetch_prediction_history(
    *,
    asset: str,
    days: int = 30,
    model_name: str = "default",
    horizon: str = "1d",
) -> pd.DataFrame:
    """Fetch prediction history for charting validation (predicted vs actual)."""
    try:
        rows = supabase_rest_select(
            table="prediction_records",
            select="asset,as_of_date,target_date,predicted_value, actual_value,unit,model_name,frequency,horizon",
            eq_filters={"asset": asset, "model_name": model_name, "horizon": horizon},
            order="target_date.desc",
            limit=int(days) * 3,
        )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        # Keep last N unique target dates
        df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce")
        df = df.dropna(subset=["target_date"]).sort_values("target_date")
        df = df.drop_duplicates(subset=["target_date"], keep="last")
        return df.tail(int(days))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def supabase_fetch_commodity_series(asset_path: str, limit: int = 600) -> pd.DataFrame:
    """Fetch commodity time series from Supabase if available.

    Expects a table named `commodity_prices` with columns:
    - asset_path (text)
    - timestamp (date or timestamptz)
    - value (numeric)
    Optional: currency (text), source (text)
    """
    try:
        rows = supabase_rest_select(
            table="commodity_prices",
            select="timestamp,value",
            eq_filters={"asset_path": asset_path},
            order="timestamp.asc",
            limit=int(limit),
        )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
        return df
    except Exception:
        return pd.DataFrame()


def render_ai_predictions_page():
    st.markdown("""
    <div style='border-left: 4px solid #7c3aed; padding-left: 1rem; margin: 1rem 0 1.25rem 0;'>
        <h2 style='font-size: 1.5rem; font-weight: 800; color: #0f172a; letter-spacing: -0.4px; margin: 0 0 0.35rem 0;'>
            🤖 AI Predictions
        </h2>
        <p style='font-size: 0.95rem; color: #475569; font-weight: 600; margin: 0; line-height: 1.5;'>
            Predicted (line) vs Actual (bars) — rolling validation
        </p>
    </div>
    """, unsafe_allow_html=True)

    supa_ok = supabase_is_configured()
    if not supa_ok:
        st.warning(
            "Supabase is not configured. Add `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` "
            "(or `SUPABASE_ANON_KEY`) in Streamlit Secrets to enable long-term prediction storage."
        )
        st.info("Prediction validation charts are disabled until Supabase is configured.")

    if supa_ok:
        assets: list[str] = []
        try:
            rows = supabase_rest_select(table="prediction_records", select="asset", limit=5000)
            assets = sorted({r.get("asset") for r in (rows or []) if r.get("asset")})
        except Exception:
            assets = []

        if not assets:
            st.info("No prediction records found in Supabase yet.")
            st.caption("Tip: push daily predictions from your pipeline into `prediction_records`.")
        else:
            colA, colB, colC = st.columns([2, 1, 1])
            with colA:
                asset = st.selectbox("Asset", assets, index=0)
            with colB:
                days = st.number_input("Days", min_value=7, max_value=365, value=30, step=1)
            with colC:
                horizon = st.selectbox("Horizon", ["1d", "24h", "7d", "30d"], index=0)

            df = supabase_fetch_prediction_history(asset=asset, days=int(days), horizon=str(horizon))
            if df.empty:
                st.info("No data for this asset/horizon yet.")
            else:
                unit = df["unit"].dropna().iloc[-1] if "unit" in df.columns and df["unit"].notna().any() else ""

                # Accuracy (simple): 100 * (1 - MAPE)
                valid = df.dropna(subset=["predicted_value", "actual_value"]).copy()
                acc_txt = ""
                if not valid.empty:
                    actual = pd.to_numeric(valid["actual_value"], errors="coerce")
                    pred = pd.to_numeric(valid["predicted_value"], errors="coerce")
                    mask = (actual.notna()) & (pred.notna()) & (actual != 0)
                    if mask.any():
                        mape = float((abs(pred[mask] - actual[mask]) / abs(actual[mask])).mean())
                        accuracy = max(0.0, 100.0 * (1.0 - mape))
                        acc_txt = f" · Accuracy: {accuracy:.1f}%"

                title = f"Rolling Validation: Predicted vs Actual{acc_txt}"
                st.markdown(f"### {title}")
                st.caption("Bars = Actual · Line = Predicted")

                fig = go.Figure()

                fig.add_trace(
                    go.Bar(
                        x=df["target_date"],
                        y=pd.to_numeric(df["actual_value"], errors="coerce"),
                        name="Actual",
                        marker_color="#10b981",
                        text=pd.to_numeric(df["actual_value"], errors="coerce"),
                        texttemplate="%{text:,.0f}",
                        textposition="inside",
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=df["target_date"],
                        y=pd.to_numeric(df["predicted_value"], errors="coerce"),
                        name=f"Predicted ({horizon})",
                        mode="lines+markers+text",
                        line=dict(color="#818cf8", width=3),
                        marker=dict(size=7),
                        text=pd.to_numeric(df["predicted_value"], errors="coerce"),
                        texttemplate="%{text:,.0f}",
                        textposition="top center",
                    )
                )

                fig.update_layout(
                    height=420,
                    margin=dict(l=40, r=20, t=30, b=40),
                    plot_bgcolor="#0b1220",
                    paper_bgcolor="#0b1220",
                    font=dict(color="#e5e7eb"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(title="Date", gridcolor="rgba(148,163,184,0.15)", tickfont=dict(color="#cbd5e1")),
                    yaxis=dict(
                        title=f"Price ({unit})" if unit else "Price",
                        gridcolor="rgba(148,163,184,0.15)",
                        tickfont=dict(color="#cbd5e1"),
                    ),
                )
                fig.update_traces(cliponaxis=False)

                st.plotly_chart(fig, use_container_width=True, key=f"ai_pred_{asset}_{horizon}")

                out = df[["target_date", "actual_value", "predicted_value"]].copy()
                out.columns = ["Date", "Actual", "Predicted"]
                out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
                st.dataframe(out, use_container_width=True, height=260)


@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_latest_events() -> dict:
    """Load latest market events and news."""
    events_file = EVENTS_DIR / "events_latest.json"
    if events_file.exists():
        try:
            with open(events_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {"news": {}, "weather": [], "geopolitical": []}

# Commodity configuration - SCALABLE & CONFIG-DRIVEN
# International: ALL prices in standard USD units for consistent comparison
INTERNATIONAL_COMMODITIES = {
    "Cotton": {"path": "cotton/cotton_usd_monthly", "currency": "USD/lb", "icon": "🌱", "type": "Futures"},
    "Polyester": {"path": "polyester/polyester_usd_monthly", "currency": "USD/kg", "icon": "🧵", "type": "Futures"},
    "Viscose": {"path": "viscose/viscose_usd_monthly", "currency": "USD/kg", "icon": "🧬", "type": "Spot"},
    "Natural Gas": {"path": "energy/natural_gas_usd_monthly_clean", "currency": "USD/MMBTU", "icon": "🔥", "type": "Spot"},
    "Crude Oil": {"path": "energy/crude_oil_brent_usd_monthly_clean", "currency": "USD/barrel", "icon": "🛢️", "type": "Spot"}
}

LOCAL_COMMODITIES = {
    "Cotton (Local)": {"path": "cotton/cotton_pkr_monthly", "currency": "PKR/maund", "icon": "🌱", "type": "Local Market"},
    "Polyester (Local)": {"path": "polyester/polyester_pkr_monthly", "currency": "PKR/kg", "icon": "🧵", "type": "Import Cost"},
    "Viscose (Local)": {"path": "viscose/viscose_pkr_monthly", "currency": "PKR/kg", "icon": "🧬", "type": "Local Market"},
    "Natural Gas": {"path": "energy/natural_gas_pkr_monthly_clean", "currency": "PKR/MMBTU", "icon": "🔥", "type": "Import Cost"},
    "Crude Oil": {"path": "energy/crude_oil_brent_pkr_monthly_clean", "currency": "PKR/barrel", "icon": "🛢️", "type": "Import Cost"}
}

# Live fetched data for local page
LIVE_LOCAL_DATA = {
    "USD/PKR Rate": {"fetch_func": "fetch_usd_pkr_rate", "icon": "💱", "type": "Live Exchange Rate"},
    "Electricity": {"fetch_func": "fetch_wapda_electricity_rate", "icon": "⚡", "type": "Industrial Tariff"}
}

# Placeholder for future data sources (only items we don't have yet)
PENDING_DATA = {
    # All major commodities now integrated
}

# Live data sources
LIVE_DATA_SOURCES = {
    "USD/PKR": {"api": "exchangerate-api.com", "type": "forex"},
    "Electricity": {"api": "WAPDA/NEPRA", "type": "energy"}
}

# Page config
st.set_page_config(
    page_title="Commodity Intelligence · Procurement Analytics",
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Commodity Dashboard Styling
st.markdown("""
<style>
    /* ============================================
       PROFESSIONAL COMMODITY DASHBOARD DESIGN
       Clean, Data-Driven, Responsive Layout
    ============================================ */
    
    /* Base Container Setup */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
        padding-top: 0 !important;
    }
    
    .main {
        padding: 1rem 2.5rem 2rem 2.5rem;
        max-width: 1600px;
        margin: 0 auto;
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    /* ============================================
       TYPOGRAPHY - Clean & Professional
    ============================================ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@500;600&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        color: #0f172a;
        line-height: 1.2;
    }
    
    [data-testid="stMarkdownContainer"] h1 {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1e293b;
        letter-spacing: -0.5px;
        margin-bottom: 0.25rem;
    }
    
    [data-testid="stMarkdownContainer"] h2 {
        font-size: 1.4rem;
        font-weight: 600;
        color: #334155;
        letter-spacing: -0.3px;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    [data-testid="stMarkdownContainer"] h3 {
        font-size: 1.15rem;
        font-weight: 600;
        color: #475569;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stMarkdownContainer"] p {
        font-size: 0.875rem;
        line-height: 1.6;
        color: #64748b;
        font-weight: 500;
    }
    
    [data-testid="stCaption"] {
        font-size: 0.8rem;
        color: #94a3b8;
        font-weight: 500;
    }
    
    /* ============================================
       METRIC CARDS - Data-Focused Design
    ============================================ */
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        border: 1.5px solid #e2e8f0;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
        transition: all 0.25s ease;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transform: translateY(-1px);
        border-color: #cbd5e1;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0.4rem 0;
        font-family: 'IBM Plex Mono', 'Courier New', monospace;
        letter-spacing: -0.8px;
        line-height: 1;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    .metric-label {
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    
    .currency-label {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 500;
        margin-top: 0.4rem;
        line-height: 1.4;
    }
    
    .trend-positive {
        color: #059669;
        font-weight: 700;
        font-size: 1rem;
    }
    
    .trend-negative {
        color: #dc2626;
        font-weight: 700;
        font-size: 1rem;
    }
    
    .recommendation {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 0.75rem 0;
        font-weight: 600;
        font-size: 0.9rem;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25);
        border: none;
    }
    
    [data-testid="stDataFrame"] {
        border-radius: 6px;
        overflow: hidden;
        font-size: 0.85rem;
    }
    
    /* ============================================
       TABS - Clean Commodity Trading Style
    ============================================ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding: 8px 0 0 0;
        background: transparent;
        border-bottom: none;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 0;
        padding: 0 20px;
        font-weight: 600;
        font-size: 0.85rem;
        background: transparent;
        border: none;
        border-bottom: 3px solid #e2e8f0;
        color: #64748b;
        transition: all 0.2s ease;
        margin-bottom: -1px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #f8fafc;
        color: #334155;
        border-bottom-color: #94a3b8;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #ffffff;
        color: #1e40af;
        border-bottom-color: #2563eb;
        font-weight: 700;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem;
        border-top: 2px solid #e2e8f0;
    }
    
    /* ============================================
       NAVIGATION - Professional Button Style
    ============================================ */
    [data-testid="stHorizontalBlock"] [data-testid="stRadio"] > div {
        gap: 10px;
        padding: 6px 0;
    }
    
    [data-testid="stRadio"] label {
        background: #ffffff;
        border: 1.5px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 0.85rem;
        color: #475569;
        transition: all 0.2s ease;
        cursor: pointer;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        min-width: 180px;
        text-align: center;
    }
    
    [data-testid="stRadio"] label:hover {
        background: #f8fafc;
        border-color: #3b82f6;
        transform: translateY(-1px);
        box-shadow: 0 3px 8px rgba(59, 130, 246, 0.15);
    }
    
    [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
        display: none;
    }
    
    [data-testid="stRadio"] input:checked + label {
        background: #2563eb;
        color: white;
        border-color: #2563eb;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        transform: translateY(-2px);
    }
    
    /* ============================================
       RESPONSIVE DESIGN
    ============================================ */
    @media (max-width: 768px) {
        .main { padding: 0.75rem 1rem; }
        .metric-card { padding: 1rem; min-height: 120px; }
        .metric-value { font-size: 1.5rem; }
        [data-testid="stRadio"] label { min-width: 120px; padding: 10px 16px; font-size: 0.75rem; }
    }
    
    /* Streamlit Component Overrides */
    .stButton > button {
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.5rem 1.25rem;
        border: 1.5px solid #e2e8f0;
        background: white;
        color: #334155;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: #f8fafc;
        border-color: #2563eb;
        color: #2563eb;
    }
    
    /* Info/Success/Warning Boxes */
    .stAlert {
        border-radius: 8px;
        border-left-width: 4px;
        padding: 0.875rem 1rem;
        font-size: 0.85rem;
    }

    /* ============================================
       CALL/PUT ADVISOR - Clean Signal Card
    ============================================ */
    .cp-card {
        background: #ffffff;
        border: 1.5px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
        margin: 0.5rem 0 1rem 0;
    }

    .cp-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.3px;
        margin: 0;
    }

    .cp-subtitle {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        margin-top: 0.2rem;
    }

    .cp-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.7rem;
        border-radius: 999px;
        font-weight: 800;
        font-size: 0.85rem;
        letter-spacing: 0.2px;
        border: 1px solid #e2e8f0;
        background: #f8fafc;
        color: #0f172a;
        white-space: nowrap;
    }

    .cp-pill-call {
        background: #dcfce7;
        border-color: #86efac;
        color: #166534;
    }

    .cp-pill-put {
        background: #fee2e2;
        border-color: #fecaca;
        color: #991b1b;
    }

    .cp-pill-hold {
        background: #e2e8f0;
        border-color: #cbd5e1;
        color: #334155;
    }

    .cp-kv {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        gap: 0.25rem;
        background: #ffffff;
        border: 1.5px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        min-height: 92px;
    }

    .cp-kv-label {
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 700;
    }

    .cp-kv-value {
        font-size: 1.25rem;
        font-weight: 900;
        color: #0f172a;
        font-family: 'IBM Plex Mono', 'Courier New', monospace;
        letter-spacing: -0.5px;
        line-height: 1.15;
    }

    .cp-kv-sub {
        font-size: 0.82rem;
        color: #475569;
        font-weight: 700;
    }

    .cp-note {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)  # Cache for 5 minutes for live data
def fetch_usd_pkr_rate() -> Optional[dict]:
    """Fetch live USD/PKR exchange rate from public API."""
    try:
        response = requests.get(USD_PKR_API, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'rates' in data and 'PKR' in data['rates']:
                rate = data['rates']['PKR']
                timestamp = data.get('time_last_update_utc', datetime.now().isoformat())
                return {
                    'current_price': rate,
                    'price_change': 0,
                    'data_points': 1,
                    'df': None,
                    'value_col': 'rate',
                    'time_col': 'timestamp',
                    'currency': 'PKR per USD',
                    'last_update': timestamp,
                    'source': 'exchangerate-api.com'
                }
    except Exception as e:
        try:
            response = requests.get(BACKUP_USD_PKR_API, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'rates' in data and 'PKR' in data['rates']:
                    rate = data['rates']['PKR']
                    return {
                        'current_price': rate,
                        'price_change': 0,
                        'data_points': 1,
                        'df': None,
                        'value_col': 'rate',
                        'time_col': 'timestamp',
                        'currency': 'PKR per USD',
                        'last_update': data.get('date', 'N/A'),
                        'source': 'exchangerate-api.com'
                    }
        except:
            pass
    return None


@st.cache_data(ttl=3600)  # Cache WAPDA data for 1 hour
def fetch_wapda_electricity_rate() -> Optional[dict]:
    """Fetch WAPDA/NEPRA electricity tariff data.
    Note: This is a placeholder - actual WAPDA API endpoint needed.
    Using approximate current industrial tariff for now.
    """
    try:
        # TODO: Replace with actual WAPDA/NEPRA API when endpoint is available
        # Updated to reflect 2025-2026 NEPRA B-3 Industrial Tariff for textile industry
        # Peak: 42.50 PKR/Unit, Off-Peak: 32.50 PKR/Unit, Blended Average: 37.50 PKR/Unit
        industrial_tariff = 37.50  # PKR per Unit (1 Unit = 1 kWh, textile industry avg Jan 2026)
        return {
            'current_price': industrial_tariff,
            'price_change': 0,
            'data_points': 1,
            'df': None,
            'value_col': 'tariff',
            'time_col': 'timestamp',
            'currency': 'PKR/Unit',
            'last_update': datetime.now().strftime('%Y-%m-%d'),
            'source': 'NEPRA B-3 (Industrial Tariff)'
        }
    except Exception as e:
        return None


@st.cache_data
def load_commodity_data(asset_path: str, currency: str):
    """Load commodity price data with full dataframe."""
    ton_assets_to_kg = (
        "polyester/polyester_usd_monthly",
        "viscose/viscose_usd_monthly",
        "polyester/polyester_pkr_monthly",
        "viscose/viscose_pkr_monthly",
    )
    want_per_kg = "/kg" in str(currency).lower()

    csv_files = list(RAW_DATA_DIR.glob(f"{asset_path}*.csv"))
    if csv_files:
        try:
            df = pd.read_csv(csv_files[0])
            
            # Look for value/price column (multiple naming conventions)
            value_col = None
            for possible_name in ['value', 'price_usd', 'price_rmb', 'price', 'close']:
                if possible_name in df.columns:
                    value_col = possible_name
                    break
            
            # If not found by name, find first numeric column that's not timestamp/date
            if value_col is None:
                for col in df.columns:
                    if str(col).lower() in ['timestamp', 'date', 'time', 'datetime', 'index']:
                        continue
                    if pd.api.types.is_numeric_dtype(df[col]):
                        value_col = col
                        break
            
            if value_col and len(df) > 0:
                # If the configured unit is per-kg but the underlying series is per-ton, convert values.
                if want_per_kg and any(k in str(asset_path) for k in ton_assets_to_kg):
                    df[value_col] = pd.to_numeric(df[value_col], errors="coerce") / 1000.0

                latest_price = float(df[value_col].iloc[-1])
                prev_price = float(df[value_col].iloc[-2]) if len(df) > 1 else latest_price
                price_change = ((latest_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0.0
                
                # Get timestamp column
                time_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
                
                return {
                    'current_price': latest_price,
                    'price_change': price_change,
                    'data_points': len(df),
                    'df': df,
                    'value_col': value_col,
                    'time_col': time_col,
                    'currency': currency
                }
        except Exception as e:
            st.error(f"Error loading {asset_path}: {str(e)}")
            pass

    # Cloud fallback: load from Supabase (keeps repo code-only, data stays private)
    sb_df = supabase_fetch_commodity_series(asset_path)
    if sb_df is not None and not sb_df.empty:
        if want_per_kg and any(k in str(asset_path) for k in ton_assets_to_kg):
            sb_df = sb_df.copy()
            sb_df["value"] = pd.to_numeric(sb_df["value"], errors="coerce") / 1000.0
        latest_price = float(sb_df["value"].iloc[-1])
        prev_price = float(sb_df["value"].iloc[-2]) if len(sb_df) > 1 else latest_price
        price_change = ((latest_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0.0
        return {
            'current_price': latest_price,
            'price_change': price_change,
            'data_points': len(sb_df),
            'df': sb_df,
            'value_col': 'value',
            'time_col': 'timestamp',
            'currency': currency,
            'source': 'supabase'
        }
    
    # Fallback
    return {
        'current_price': np.random.uniform(100, 200),
        'price_change': np.random.uniform(-2, 2),
        'data_points': 48,
        'df': None,
        'value_col': None,
        'time_col': None,
        'currency': currency
    }


def get_month_horizons(count: int = 12):
    """Get next N month labels (e.g., January 2026)."""
    start = pd.Timestamp.today().replace(day=1)
    months = pd.date_range(start=start, periods=count, freq='MS')
    return [m.strftime('%B %Y') for m in months]


def get_prediction_horizons(predictions: dict, count: int = 12):
    """Get ordered horizons from predictions, preferring next N month labels."""
    horizons = get_month_horizons(count)
    if predictions:
        filtered = [h for h in horizons if h in predictions]
        if filtered:
            return filtered
        return list(predictions.keys())
    return horizons


def get_prediction_by_index(predictions: dict, month_index: int):
    """Get prediction by 1-based month index from ordered horizons."""
    horizons = get_prediction_horizons(predictions)
    if 1 <= month_index <= len(horizons):
        return predictions.get(horizons[month_index - 1], {})
    return {}


def annual_pct_to_monthly_rate(annual_pct: float) -> float:
    """Convert annual nominal rate (%) to an effective monthly rate."""
    annual = annual_pct / 100.0
    if annual <= -1:
        return -1.0
    return (1.0 + annual) ** (1.0 / 12.0) - 1.0


def build_monthly_rate_schedule(months: int, annual_start_pct: float, annual_end_pct: float, mode: str) -> list[float]:
    """Build a monthly rate schedule; supports declining rates."""
    months = max(1, int(months))
    if mode == "Constant" or months == 1:
        annuals = [annual_start_pct] * months
    else:
        annuals = list(np.linspace(annual_start_pct, annual_end_pct, months))
    return [annual_pct_to_monthly_rate(a) for a in annuals]


def compound_factor(monthly_rates: list[float]) -> float:
    """Compounding factor for a sequence of monthly rates."""
    if not monthly_rates:
        return 1.0
    return float(np.prod([1.0 + r for r in monthly_rates]))


def render_hedging_compounding_simulator(*, expander_title: str, expanded: bool = False, key_prefix: str = "hedge") -> None:
    """Planning calculator: bank compounding vs fixed-price agreement vs call-style ceiling."""
    with st.expander(expander_title, expanded=expanded):
        st.caption(
            "Compare procurement strategies using bank compounding and a fixed-price agreement (forward) or a call-style ceiling. "
            "This is a planning calculator (not full options pricing / Greeks)."
        )

        all_assets: dict[str, dict] = {}
        all_assets.update(INTERNATIONAL_COMMODITIES)
        all_assets.update(LOCAL_COMMODITIES)

        asset_names = list(all_assets.keys())
        if not asset_names:
            st.info("No commodities configured.")
            return

        c1, c2, c3, c4 = st.columns([2.2, 1.2, 1.2, 1.4])
        with c1:
            selected_name = st.selectbox("Commodity", asset_names, index=0, key=f"{key_prefix}_asset")
        with c2:
            months = st.number_input("Months (n)", min_value=1, max_value=24, value=6, step=1, key=f"{key_prefix}_months")
        with c3:
            qty = st.number_input("Quantity (units)", min_value=1.0, value=1.0, step=1.0, key=f"{key_prefix}_qty")
        with c4:
            strategy_view = st.selectbox(
                "View",
                ["Leftover Cash at Maturity", "Effective Purchase Cost"],
                index=0,
                key=f"{key_prefix}_view",
            )

        info = all_assets[selected_name]
        md = load_commodity_data(info["path"], info["currency"])
        current_price = float(md.get("current_price", 0.0)) if md else 0.0
        predictions = load_predictions(info["path"]) if info else {}
        pred = get_prediction_by_index(predictions, int(months))
        pred_base = float(pred.get("price", current_price)) if pred else current_price
        pred_low = float(pred.get("lower", pred_base)) if pred else pred_base
        pred_high = float(pred.get("upper", pred_base)) if pred else pred_base

        st.markdown("#### Inputs")
        i1, i2, i3 = st.columns([1.3, 1.3, 1.4])
        with i1:
            s0 = st.number_input(
                f"Current price (S) [{info['currency']}]",
                min_value=0.0,
                value=float(current_price) if current_price else 10.0,
                key=f"{key_prefix}_s",
            )
            cash_today = st.number_input(
                f"Cash today to invest [{info['currency']}]",
                min_value=0.0,
                value=float(s0) * float(qty),
                help="Typical use: set cash = S × quantity (your purchase budget parked in bank).",
                key=f"{key_prefix}_cash",
            )
        with i2:
            k = st.number_input(
                f"Agreement / Strike price (K) [{info['currency']}]",
                min_value=0.0,
                value=float(s0) * 1.3 if s0 else 13.0,
                key=f"{key_prefix}_k",
            )
            premium = st.number_input(
                f"Call premium (paid today, per unit) [{info['currency']}]",
                min_value=0.0,
                value=0.0,
                help="Set to 0 if you only want to compare bank vs fixed-price agreement.",
                key=f"{key_prefix}_premium",
            )
        with i3:
            rate_mode = st.selectbox(
                "Interest-rate scenario",
                ["Constant", "Declining (linear)"],
                key=f"{key_prefix}_rate_mode",
            )
            annual_start = st.number_input(
                "Annual interest rate start (%)",
                min_value=0.0,
                max_value=100.0,
                value=18.0,
                step=0.5,
                key=f"{key_prefix}_r_start",
            )
            annual_end = annual_start
            if rate_mode != "Constant":
                annual_end = st.number_input(
                    "Annual interest rate end (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=14.0,
                    step=0.5,
                    key=f"{key_prefix}_r_end",
                )

        monthly_rates = build_monthly_rate_schedule(int(months), float(annual_start), float(annual_end), rate_mode)
        cf = compound_factor(monthly_rates)
        fv_cash = float(cash_today) * cf

        dec = 3 if "/lb" in str(info.get("currency", "")).lower() else 2

        st.markdown("#### Price scenarios at month n")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Low (forecast lower)", f"{pred_low:,.{dec}f}")
        with s2:
            st.metric("Base (forecast)", f"{pred_base:,.{dec}f}")
        with s3:
            st.metric("High (forecast upper)", f"{pred_high:,.{dec}f}")

        scenarios = [("Low", pred_low), ("Base", pred_base), ("High", pred_high)]

        def _leftover_wait(spot_t: float) -> float:
            return fv_cash - float(qty) * float(spot_t)

        def _leftover_forward() -> float:
            return fv_cash - float(qty) * float(k)

        def _leftover_call_ceiling(spot_t: float) -> float:
            cash_after_premium = float(cash_today) - float(qty) * float(premium)
            cash_after_premium = max(0.0, cash_after_premium)
            fv_after = cash_after_premium * cf
            return fv_after - float(qty) * min(float(spot_t), float(k))

        rows = []
        for label, spot_t in scenarios:
            rows.append(
                {
                    "Scenario": label,
                    "Spot @ n": float(spot_t),
                    "Wait+Bank": _leftover_wait(spot_t),
                    "Fixed agreement": _leftover_forward(),
                    "Call ceiling": _leftover_call_ceiling(spot_t),
                }
            )

        df_out = pd.DataFrame(rows)
        if strategy_view == "Effective Purchase Cost":
            df_out["Wait+Bank"] = float(cash_today) - (df_out["Wait+Bank"] / cf)
            df_out["Fixed agreement"] = float(cash_today) - (df_out["Fixed agreement"] / cf)
            df_out["Call ceiling"] = float(cash_today) - (df_out["Call ceiling"] / cf)
            df_out = df_out.rename(
                columns={
                    "Wait+Bank": "Wait+Bank effective cost",
                    "Fixed agreement": "Fixed agreement effective cost",
                    "Call ceiling": "Call ceiling effective cost",
                }
            )

        st.markdown("#### Results")
        st.dataframe(df_out, use_container_width=True, height=180)


def _annualized_volatility_from_history(series: pd.Series, periods_per_year: int = 12) -> float:
    """Estimate annualized volatility from a price series."""
    try:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) < 6:
            return 0.25
        returns = np.log(s).diff().dropna()
        if returns.empty:
            return 0.25
        vol = float(returns.std()) * float(np.sqrt(periods_per_year))
        if not np.isfinite(vol) or vol <= 0:
            return 0.25
        return min(max(vol, 0.05), 1.50)
    except Exception:
        return 0.25


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using erf (no scipy dependency)."""
    import math

    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _bs_price(*, s: float, k: float, t: float, sigma: float, opt_type: str) -> float:
    """Black-Scholes price with r=0, q=0 (good enough for guidance UI)."""
    import math

    s = float(max(s, 1e-12))
    k = float(max(k, 1e-12))
    t = float(max(t, 1e-9))
    sigma = float(max(sigma, 1e-9))
    d1 = (math.log(s / k) + 0.5 * sigma * sigma * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)

    opt_type = str(opt_type).upper()
    if opt_type == "CALL":
        return s * _norm_cdf(d1) - k * _norm_cdf(d2)
    if opt_type == "PUT":
        return k * _norm_cdf(-d2) - s * _norm_cdf(-d1)
    return 0.0


def _lognormal_prob_gt(*, s0: float, s_mean: float, sigma_ann: float, t_years: float, k: float) -> float:
    """P(S_T > K) for lognormal with mean matched to s_mean."""
    import math

    s0 = float(max(s0, 1e-12))
    s_mean = float(max(s_mean, 1e-12))
    k = float(max(k, 1e-12))
    t_years = float(max(t_years, 1e-9))
    sigma_ann = float(max(sigma_ann, 1e-9))

    sigma_t = sigma_ann * math.sqrt(t_years)
    sigma_t = float(max(sigma_t, 1e-9))

    # Choose drift so that E[S_T] = s_mean
    mu = math.log(s_mean / s0) / t_years + 0.5 * sigma_ann * sigma_ann
    mean_log = math.log(s0) + (mu - 0.5 * sigma_ann * sigma_ann) * t_years
    z = (math.log(k) - mean_log) / sigma_t
    return float(1.0 - _norm_cdf(z))


def _momentum_pct(*, hist_df: pd.DataFrame, value_col: str, months: int) -> float:
    try:
        s = pd.to_numeric(hist_df[value_col], errors="coerce").dropna()
        if len(s) < months + 1:
            return 0.0
        last = float(s.iloc[-1])
        prior = float(s.iloc[-(months + 1)])
        if prior == 0:
            return 0.0
        return (last / prior - 1.0) * 100.0
    except Exception:
        return 0.0


def _blend_sigma_from_forecast_interval(*, sigma_ann: float, lower: float | None, upper: float | None, t_years: float) -> float:
    """If forecast provides bounds, infer a minimum sigma consistent with that interval."""
    import math

    sigma_ann = float(sigma_ann)
    if lower is None or upper is None:
        return sigma_ann
    try:
        lower = float(lower)
        upper = float(upper)
        if not (lower > 0 and upper > 0 and upper > lower and t_years > 0):
            return sigma_ann
        # Assume bounds are roughly ~90% interval => z ≈ 1.645
        z = 1.645
        sigma_t = (math.log(upper) - math.log(lower)) / (2.0 * z)
        sigma_from_interval = sigma_t / math.sqrt(float(max(t_years, 1e-9)))
        if np.isfinite(sigma_from_interval) and sigma_from_interval > 0:
            return float(min(max(max(sigma_ann, sigma_from_interval), 0.05), 1.50))
    except Exception:
        return sigma_ann
    return sigma_ann


def _suggest_strikes(*, s0: float, sigma_ann: float, t_years: float, risk: str) -> tuple[float, float, float, float]:
    """Return (cap_k1, cap_k2, floor_k1, floor_k2) based on risk + volatility."""
    import math

    s0 = float(max(s0, 1e-12))
    sigma_ann = float(max(sigma_ann, 1e-9))
    t_years = float(max(t_years, 1e-9))
    sigma_t = sigma_ann * math.sqrt(t_years)

    risk = str(risk).lower()
    if "conserv" in risk:
        a1, a2 = 0.25, 0.75
    elif "aggress" in risk:
        a1, a2 = 0.60, 1.25
    else:  # balanced
        a1, a2 = 0.40, 1.00

    cap_k1 = s0 * math.exp(a1 * sigma_t)
    cap_k2 = s0 * math.exp(a2 * sigma_t)
    floor_k1 = s0 * math.exp(-a1 * sigma_t)
    floor_k2 = s0 * math.exp(-a2 * sigma_t)

    # Avoid strikes too close when sigma is tiny
    cap_k1 = max(cap_k1, s0 * 1.02)
    cap_k2 = max(cap_k2, cap_k1 * 1.05)
    floor_k1 = min(floor_k1, s0 * 0.98)
    floor_k2 = min(floor_k2, floor_k1 * 0.95)
    return float(cap_k1), float(cap_k2), float(floor_k1), float(floor_k2)


def _recommend_hedge_strategy(
    *,
    exposure: str,
    s0: float,
    s_mean: float,
    sigma_ann: float,
    t_years: float,
    risk_profile: str,
    budget_priority: str,
    allow_selling: bool,
    qty: float,
    unit: str,
) -> dict:
    """Return a strategy dict with legs + metrics.

    exposure: Procurement / Sales / Inventory
    budget_priority: Low / Medium / High
    """
    exposure = str(exposure)
    budget_priority = str(budget_priority).lower()
    risk_profile = str(risk_profile)

    cap_k1, cap_k2, floor_k1, floor_k2 = _suggest_strikes(s0=s0, sigma_ann=sigma_ann, t_years=t_years, risk=risk_profile)

    exp_ret = ((float(s_mean) - float(s0)) / float(s0) * 100.0) if s0 else 0.0
    p_up_5 = _lognormal_prob_gt(s0=s0, s_mean=s_mean, sigma_ann=sigma_ann, t_years=t_years, k=s0 * 1.05)
    p_dn_5 = 1.0 - _lognormal_prob_gt(s0=s0, s_mean=s_mean, sigma_ann=sigma_ann, t_years=t_years, k=s0 * 0.95)

    legs: list[dict] = []
    title = "HOLD / WAIT"
    rationale_bits: list[str] = []

    # Decide hedge intensity
    need_up_protection = (exp_ret >= 2.0) or (p_up_5 >= 0.55)
    need_down_protection = (exp_ret <= -2.0) or (p_dn_5 >= 0.55)
    high_vol = sigma_ann >= 0.30

    # Procurement: protect against price rising (cap)
    if exposure.startswith("Procurement"):
        if need_up_protection or high_vol:
            if "high" in budget_priority:
                title = "CALL SPREAD (CAP COST, LOWER PREMIUM)"
                legs = [
                    {"side": "BUY", "type": "CALL", "strike": cap_k1, "note": "Caps your purchase price"},
                    {"side": "SELL", "type": "CALL", "strike": cap_k2, "note": "Funds premium; caps benefit above K2"},
                ]
                if not allow_selling:
                    # Selling requires explicit permission
                    title = "BUY CALL (CAP COST)"
                    legs = [{"side": "BUY", "type": "CALL", "strike": cap_k1, "note": "Caps your purchase price"}]
                    rationale_bits.append("Selling legs disabled → using a simple call instead of a spread.")
            else:
                title = "BUY CALL (CAP COST)"
                legs = [{"side": "BUY", "type": "CALL", "strike": cap_k1, "note": "Caps your purchase price"}]

            # Optional zero-cost collar style for importers (buy call, sell put)
            if allow_selling and ("high" in budget_priority) and ("conserv" in risk_profile.lower() or high_vol):
                title = "RANGE FORWARD (BUY CALL + SELL PUT)"
                legs = [
                    {"side": "BUY", "type": "CALL", "strike": cap_k1, "note": "Cap"},
                    {"side": "SELL", "type": "PUT", "strike": floor_k1, "note": "Reduces premium; commits you to buy if price falls"},
                ]
                rationale_bits.append("This is often structured near zero-cost, but adds obligation risk.")

            rationale_bits.append("Goal: protect budget if prices rise.")
        else:
            title = "HOLD / WAIT"
            rationale_bits.append("Forecast not strongly up; keep flexibility and re-check monthly.")

    # Sales: protect against price falling (floor)
    elif exposure.startswith("Sales"):
        if need_down_protection or high_vol:
            if "high" in budget_priority:
                title = "PUT SPREAD (FLOOR, LOWER PREMIUM)"
                legs = [
                    {"side": "BUY", "type": "PUT", "strike": floor_k1, "note": "Protects your selling price"},
                    {"side": "SELL", "type": "PUT", "strike": floor_k2, "note": "Funds premium; limits protection below K2"},
                ]
                if not allow_selling:
                    title = "BUY PUT (PROTECT FLOOR)"
                    legs = [{"side": "BUY", "type": "PUT", "strike": floor_k1, "note": "Protects your selling price"}]
                    rationale_bits.append("Selling legs disabled → using a simple put instead of a spread.")
            else:
                title = "BUY PUT (PROTECT FLOOR)"
                legs = [{"side": "BUY", "type": "PUT", "strike": floor_k1, "note": "Protects your selling price"}]

            if allow_selling and ("high" in budget_priority):
                # Classic collar for sellers: buy put financed by selling call
                title = "COLLAR (BUY PUT + SELL CALL)"
                legs = [
                    {"side": "BUY", "type": "PUT", "strike": floor_k1, "note": "Floor"},
                    {"side": "SELL", "type": "CALL", "strike": cap_k1, "note": "Funds premium; caps upside"},
                ]
                rationale_bits.append("Collar reduces premium but caps upside.")

            rationale_bits.append("Goal: protect margin if prices fall.")
        else:
            title = "HOLD / WAIT"
            if allow_selling and exp_ret > 2.0 and ("aggress" in risk_profile.lower()):
                title = "COVERED CALL (MONETIZE UPSIDE)"
                legs = [{"side": "SELL", "type": "CALL", "strike": cap_k1, "note": "Collect premium; caps upside"}]
                rationale_bits.append("Only appropriate if you are naturally short/covered (sales exposure).")
            else:
                rationale_bits.append("No strong downside risk signal; avoid paying premium.")

    # Inventory: protect downside but keep some upside
    else:
        if high_vol or need_down_protection:
            if allow_selling and ("high" in budget_priority):
                title = "COLLAR (PROTECT + LOWER PREMIUM)"
                legs = [
                    {"side": "BUY", "type": "PUT", "strike": floor_k1, "note": "Downside protection"},
                    {"side": "SELL", "type": "CALL", "strike": cap_k1, "note": "Funds premium; caps upside"},
                ]
            else:
                title = "PROTECTIVE PUT"
                legs = [{"side": "BUY", "type": "PUT", "strike": floor_k1, "note": "Downside protection"}]
            rationale_bits.append("Goal: protect inventory value during high volatility.")
        else:
            title = "HOLD / WAIT"
            rationale_bits.append("Volatility is manageable; re-check if moves accelerate.")

    # Premium estimates
    est_premium_per_unit = 0.0
    for leg in legs:
        px = _bs_price(s=s0, k=float(leg["strike"]), t=t_years, sigma=sigma_ann, opt_type=str(leg["type"]))
        est = float(px)
        leg["est_premium"] = est
        if str(leg["side"]).upper() == "BUY":
            est_premium_per_unit += est
        else:
            est_premium_per_unit -= est

        leg["prob_itm"] = (
            _lognormal_prob_gt(s0=s0, s_mean=s_mean, sigma_ann=sigma_ann, t_years=t_years, k=float(leg["strike"]))
            if str(leg["type"]).upper() == "CALL"
            else 1.0
            - _lognormal_prob_gt(s0=s0, s_mean=s_mean, sigma_ann=sigma_ann, t_years=t_years, k=float(leg["strike"]))
        )

    metrics = {
        "exp_ret_pct": float(exp_ret),
        "p_up_5": float(p_up_5),
        "p_dn_5": float(p_dn_5),
        "est_premium_per_unit": float(est_premium_per_unit),
        "est_premium_total": float(est_premium_per_unit * float(max(qty, 0.0))),
        "unit": unit,
    }

    rationale = " ".join([b for b in rationale_bits if b])
    return {"title": title, "legs": legs, "rationale": rationale, "metrics": metrics}


def _parse_horizon_months(horizon: str) -> int:
    """Infer months from horizon label.

    Supports "6M" style as well as month labels like "December 2026".
    """
    try:
        h = str(horizon).strip()
        if h.upper().endswith("M"):
            m = int(h[:-1])
            return int(max(1, min(24, m)))

        dt = pd.to_datetime(h, errors="coerce")
        if pd.isna(dt):
            return 6
        now = pd.Timestamp.now()
        months = (int(dt.year) * 12 + int(dt.month)) - (int(now.year) * 12 + int(now.month))
        return int(max(1, min(24, months)))
    except Exception:
        return 6


def _pick_best_horizon_for_payload(*, payload: dict, exposure: str) -> tuple[str, dict | None, int]:
    """Pick horizon that best matches exposure risk from the available forecasts."""
    exposure = str(exposure)
    preds = payload.get("predictions") or {}
    horizons = get_prediction_horizons(preds) if preds else []
    if not horizons:
        horizons = get_month_horizons(12)

    # Default to ~6M if possible
    default_h = None
    for cand in ("6M", "6", "Jun", "June"):
        for h in horizons:
            if cand.lower() in str(h).lower():
                default_h = h
                break
        if default_h:
            break
    if default_h is None:
        default_h = horizons[min(len(horizons) - 1, 5)] if horizons else "6M"

    best_h = default_h
    best_pred = preds.get(best_h)
    best_m = _parse_horizon_months(best_h)

    s0 = float(payload.get("current_price") or 0.0)
    scale = float(payload.get("display_scale", 1.0) or 1.0)
    s0 *= scale
    if s0 <= 0:
        return best_h, best_pred, best_m

    def _pred_change(p: dict | None) -> float:
        if not p:
            return 0.0
        try:
            ch = p.get("change")
            if ch is not None and np.isfinite(float(ch)):
                return float(ch)
        except Exception:
            pass
        try:
            px = float(p.get("price") or s0)
            return (px / s0 - 1.0) * 100.0
        except Exception:
            return 0.0

    best_score = -1e9
    for h in horizons:
        p = preds.get(h)
        ch = _pred_change(p)
        if exposure.startswith("Procurement"):
            score = ch
        elif exposure.startswith("Sales"):
            score = -ch
        else:
            score = abs(ch)
        # Prefer nearer horizons slightly (avoid always picking far future)
        m = _parse_horizon_months(h)
        score = float(score) - 0.08 * float(m)
        if score > best_score:
            best_score = score
            best_h = h
            best_pred = p
            best_m = m

    return best_h, best_pred, best_m


def _timing_label(*, score: float, sigma_ann: float) -> str:
    """Translate risk score into an action timing label."""
    try:
        score = float(score)
        sigma_ann = float(sigma_ann)
    except Exception:
        return "MONITOR"

    if sigma_ann >= 0.35 or score >= 6.0:
        return "DO NOW"
    if sigma_ann >= 0.25 or score >= 3.0:
        return "PLAN (2–4 WEEKS)"
    return "MONITOR"


def _mc_expected_payoffs_lognormal(
    *,
    s0: float,
    s_mean: float,
    sigma_ann: float,
    t_years: float,
    k: float,
    n: int = 20000,
    seed: int = 7,
) -> tuple[float, float, float, float]:
    """Return (p_itm_call, p_itm_put, e_call_payoff, e_put_payoff)."""
    s0 = float(s0)
    s_mean = float(s_mean)
    sigma_ann = float(sigma_ann)
    t_years = float(max(t_years, 1e-6))
    k = float(k)

    sigma_t = sigma_ann * np.sqrt(t_years)
    sigma_t = float(max(sigma_t, 1e-6))
    mu = np.log(max(s_mean, 1e-9)) - 0.5 * sigma_t * sigma_t
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(int(n))
    st_sim = np.exp(mu + sigma_t * z)

    call_payoff = np.maximum(st_sim - k, 0.0)
    put_payoff = np.maximum(k - st_sim, 0.0)

    p_itm_call = float(np.mean(st_sim > k))
    p_itm_put = float(np.mean(st_sim < k))
    return p_itm_call, p_itm_put, float(np.mean(call_payoff)), float(np.mean(put_payoff))


def render_call_put_hedge_advisor(
    *,
    expander_title: str,
    expanded: bool = False,
    key_prefix: str = "cp_advisor",
    commodity_payloads: list[dict] | None = None,
    variant: str = "full",
    use_expander: bool = True,
    show_portfolio_view: bool = True,
) -> None:
    """Strategy advisor: recommends hedges (buy/sell call/put structures) using history + forecast."""
    variant = str(variant or "full").strip().lower()
    container = st.expander(expander_title, expanded=expanded) if use_expander else nullcontext()
    with container:
        exposure = "Procurement (we will BUY later)"
        risk_profile = "Balanced"
        budget_priority = "Medium"
        allow_selling = False
        qty = 1.0

        if variant != "portfolio":
            st.markdown(
                """
<div class="cp-card">
    <div style="display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap;">
        <div>
            <div class="cp-title">Hedge Advisor</div>
            <div class="cp-subtitle">Strategy suggestions using history (volatility + momentum) and forecast distribution.</div>
        </div>
        <div class="cp-pill cp-pill-hold">Mode: Strategist</div>
    </div>
</div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div class='cp-kv-label' style='margin: 0.25rem 0 0.35rem 0;'>Exposure</div>", unsafe_allow_html=True)
            exposure = st.radio(
                "Exposure",
                ["Procurement (we will BUY later)", "Inventory (we HOLD stock)", "Sales (we will SELL later)"],
                horizontal=True,
                key=f"{key_prefix}_exposure",
                label_visibility="collapsed",
            )

            # Minimal UI (no month/qty selectors). Advanced is optional.
            with st.expander("Advanced settings (optional)", expanded=False):
                copt1, copt2, copt3, copt4 = st.columns([1.2, 1.0, 1.0, 1.0])
                with copt1:
                    risk_profile = st.selectbox(
                        "Risk profile",
                        ["Conservative", "Balanced", "Aggressive"],
                        index=1,
                        key=f"{key_prefix}_risk",
                    )
                with copt2:
                    budget_priority = st.selectbox(
                        "Premium budget",
                        ["Low", "Medium", "High"],
                        index=1,
                        key=f"{key_prefix}_budget",
                    )
                with copt3:
                    allow_selling = st.checkbox(
                        "Allow selling legs",
                        value=False,
                        key=f"{key_prefix}_sell",
                        help="Enables spreads/collars that SELL an option leg (adds obligation risk).",
                    )
                with copt4:
                    qty = st.number_input(
                        "Qty (units)",
                        min_value=0.0,
                        value=1.0,
                        step=1.0,
                        key=f"{key_prefix}_qty",
                        help="Used only for premium totals; base recommendation is per-unit.",
                    )

        if not commodity_payloads:
            st.info("Advisor needs commodity data (run from Executive Summary).")
            return

        # Build candidate list (no dropdowns)
        candidates: list[dict] = []
        for item in commodity_payloads:
            for side in ("int_payload", "local_payload"):
                p = item.get(side)
                if p and p.get("name"):
                    candidates.append(
                        {
                            "label": f"{p['name']} ({'International' if side=='int_payload' else 'Local'})",
                            "payload": p,
                        }
                    )

        if not candidates:
            st.info("No commodities available for recommendation.")
            return

        recs: list[dict] = []
        for c in candidates:
            payload = c["payload"]

            scale = float(payload.get("display_scale", 1.0) or 1.0)
            unit = str(payload.get("display_currency") or payload.get("info", {}).get("currency", ""))
            nm = str(payload.get("name", "")).lower()
            three_dec_assets = ("cotton", "polyester", "viscose", "crude", "natural gas")
            dec = 3 if (any(k in nm for k in three_dec_assets) or "/lb" in unit.lower()) else 2

            s0_raw = float(payload.get("current_price") or 0.0)
            s0 = s0_raw * scale

            best_h, best_pred, months = _pick_best_horizon_for_payload(payload=payload, exposure=exposure)
            t_years = float(months) / 12.0

            s_mean_raw = s0_raw
            f_lower = None
            f_upper = None
            if isinstance(best_pred, dict):
                try:
                    s_mean_raw = float(best_pred.get("price") or s0_raw)
                except Exception:
                    s_mean_raw = s0_raw
                try:
                    f_lower = float(best_pred.get("lower")) * scale if best_pred.get("lower") is not None else None
                    f_upper = float(best_pred.get("upper")) * scale if best_pred.get("upper") is not None else None
                except Exception:
                    f_lower, f_upper = None, None

            s_mean = float(s_mean_raw) * scale

            hist_df = payload.get("history_df")
            if hist_df is None:
                hist_df = payload.get("info", {}).get("df")
            if hist_df is None:
                hist_df = payload.get("df")

            sigma_ann = 0.25
            mom3 = 0.0
            mom6 = 0.0
            if isinstance(hist_df, pd.DataFrame):
                vcol = payload.get("info", {}).get("value_col") or payload.get("value_col") or "value"
                if vcol in hist_df.columns:
                    sigma_ann = _annualized_volatility_from_history(hist_df[vcol])
                    mom3 = _momentum_pct(hist_df=hist_df, value_col=vcol, months=3)
                    mom6 = _momentum_pct(hist_df=hist_df, value_col=vcol, months=6)
                elif "value" in hist_df.columns:
                    sigma_ann = _annualized_volatility_from_history(hist_df["value"])

            sigma_ann = _blend_sigma_from_forecast_interval(sigma_ann=sigma_ann, lower=f_lower, upper=f_upper, t_years=t_years)

            exp_ret = ((s_mean - s0) / s0 * 100.0) if s0 else 0.0
            score = float(exp_ret) + 0.35 * float(mom3) + 0.15 * float(mom6)
            if exposure.startswith("Procurement"):
                score = max(0.0, score)
            elif exposure.startswith("Sales"):
                score = max(0.0, -score)
            else:
                score = abs(score)

            when = _timing_label(score=score, sigma_ann=sigma_ann)

            strat = _recommend_hedge_strategy(
                exposure=exposure,
                s0=s0,
                s_mean=s_mean,
                sigma_ann=sigma_ann,
                t_years=t_years,
                risk_profile=risk_profile,
                budget_priority=budget_priority,
                allow_selling=allow_selling,
                qty=float(qty),
                unit=unit,
            )

            recs.append(
                {
                    "when": when,
                    "label": c["label"],
                    "horizon": best_h,
                    "months": months,
                    "unit": unit,
                    "dec": dec,
                    "s0": s0,
                    "s_mean": s_mean,
                    "sigma_ann": sigma_ann,
                    "mom3": mom3,
                    "mom6": mom6,
                    "score": float(score),
                    "exp_ret": float(exp_ret),
                    "strategy": strat,
                }
            )

        # Rank: DO NOW first, then by score
        when_rank = {"DO NOW": 0, "PLAN (2–4 WEEKS)": 1, "MONITOR": 2}
        recs = sorted(recs, key=lambda r: (when_rank.get(str(r.get("when")), 9), -float(r.get("score", 0.0))))

        if variant == "portfolio":
            st.markdown(
                """
<div class="cp-card" style="padding: 1.0rem 1.0rem;">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px; flex-wrap:wrap;">
    <div>
      <div class="cp-title">Options Hedge — Portfolio Dashboard</div>
      <div class="cp-subtitle">Ranked hedges across commodities. Includes both <b>CALL</b> (cap cost) and <b>PUT</b> (protect floor) suggestions.</div>
    </div>
    <div class="cp-pill cp-pill-hold">Auto</div>
  </div>
</div>
                """,
                unsafe_allow_html=True,
            )

            def _legs_one_liner(legs: list[dict], dec: int) -> str:
                if not legs:
                    return "—"
                parts: list[str] = []
                for l in legs[:3]:
                    try:
                        side = str(l.get("side", "")).upper()
                        typ = str(l.get("type", "")).upper()
                        k = float(l.get("strike"))
                        prem = float(l.get("est_premium", 0.0))
                        parts.append(f"{side} {typ} @ {k:,.{dec}f} (≈{prem:.{dec}f})")
                    except Exception:
                        continue
                return " · ".join(parts) if parts else "—"

            def _timing_style(val: str) -> str:
                v = str(val)
                if "DO NOW" in v:
                    return "background-color: #7f1d1d; color: #ffffff; font-weight: 800;"
                if "PLAN" in v:
                    return "background-color: #92400e; color: #ffffff; font-weight: 800;"
                return "background-color: #0f172a; color: #e5e7eb; font-weight: 800;"

            rows = []
            for r in recs[:8]:
                unit = str(r.get("unit", ""))
                dec = int(r.get("dec", 2))
                s0 = float(r.get("s0", 0.0))
                s_mean = float(r.get("s_mean", 0.0))
                exp_ret = float(r.get("exp_ret", 0.0))
                sigma_ann = float(r.get("sigma_ann", 0.25))
                t_years = float(int(r.get("months", 6))) / 12.0

                # Primary (exposure-based) recommendation for the current table is procurement-driven.
                call_strat = _recommend_hedge_strategy(
                    exposure="Procurement (we will BUY later)",
                    s0=s0,
                    s_mean=s_mean,
                    sigma_ann=sigma_ann,
                    t_years=t_years,
                    risk_profile="Balanced",
                    budget_priority="Medium",
                    allow_selling=False,
                    qty=1.0,
                    unit=unit,
                )
                put_strat = _recommend_hedge_strategy(
                    exposure="Sales (we will SELL later)",
                    s0=s0,
                    s_mean=s_mean,
                    sigma_ann=sigma_ann,
                    t_years=t_years,
                    risk_profile="Balanced",
                    budget_priority="Medium",
                    allow_selling=False,
                    qty=1.0,
                    unit=unit,
                )

                rows.append(
                    {
                        "Timing": r.get("when"),
                        "Commodity": r.get("label"),
                        "Target Month": r.get("horizon"),
                        "Spot": f"{s0:,.{dec}f}",
                        "Forecast": f"{s_mean:,.{dec}f}",
                        "Move %": round(exp_ret, 1),
                        "Vol %": round(sigma_ann * 100.0, 0),
                        "Priority": round(float(r.get("score", 0.0)), 1),
                        "CALL Hedge (Cap Cost)": f"{call_strat.get('title', '—')} · {_legs_one_liner(call_strat.get('legs') or [], dec)}",
                        "PUT Hedge (Protect Floor)": f"{put_strat.get('title', '—')} · {_legs_one_liner(put_strat.get('legs') or [], dec)}",
                        "Unit": unit,
                    }
                )

            dfp = pd.DataFrame(rows)

            def _chg_style(val) -> str:
                try:
                    v = float(val)
                except Exception:
                    return ""
                if v >= 0:
                    return "background-color: #052e16; color: #dcfce7; font-weight: 800;"
                return "background-color: #3f1d1d; color: #fee2e2; font-weight: 800;"

            styled = (
                dfp.style
                .applymap(_timing_style, subset=["Timing"])
                .applymap(_chg_style, subset=["Move %"])
                .set_properties(
                    **{
                        "text-align": "left",
                        "font-size": "0.85rem",
                        "font-weight": "700",
                        "padding": "10px 12px",
                        "border": "1px solid rgba(148,163,184,0.20)",
                    }
                )
                .set_table_styles(
                    [
                        {
                            "selector": "thead th",
                            "props": [
                                ("background-color", "#0b1220"),
                                ("color", "#e5e7eb"),
                                ("font-weight", "900"),
                                ("padding", "12px 12px"),
                                ("font-size", "0.75rem"),
                                ("text-transform", "uppercase"),
                                ("border", "1px solid rgba(148,163,184,0.25)"),
                            ],
                        },
                        {
                            "selector": "tbody tr:hover",
                            "props": [("background-color", "rgba(59,130,246,0.10)")],
                        },
                    ]
                )
            )

            st.caption("CALL hedge caps future purchase cost. PUT hedge protects selling price / downside. Premiums shown are rough estimates per unit.")
            st.dataframe(styled, use_container_width=True, height=360)
            return

        top = recs[0] if recs else None
        if not top:
            st.info("No recommendations available.")
            return

        # Top recommendation (actionable)
        top_strat = top.get("strategy") or {}
        top_title = str(top_strat.get("title", "HOLD / WAIT"))
        top_unit = str(top.get("unit", ""))
        top_dec = int(top.get("dec", 2))
        months = int(top.get("months", 6))
        horizon = str(top.get("horizon", ""))
        s0 = float(top.get("s0", 0.0))
        s_mean = float(top.get("s_mean", 0.0))
        sigma_ann = float(top.get("sigma_ann", 0.25))
        mom3 = float(top.get("mom3", 0.0))
        exp_ret = float(top.get("exp_ret", 0.0))
        score = float(top.get("score", 0.0))

        pill_class = "cp-pill-hold"
        if "CALL" in top_title:
            pill_class = "cp-pill-call"
        if "PUT" in top_title:
            pill_class = "cp-pill-put"

        st.markdown(
            f"""
<div class="cp-card" style="display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap;">
  <div>
    <div class="cp-title" style="margin-bottom: 0.2rem;">What to do</div>
    <div class="cp-subtitle"><b>{top.get('when')}</b> · {top.get('label')} · Horizon: {horizon} · Score {score:.1f}</div>
  </div>
  <div class="cp-pill {pill_class}">{top_title}</div>
</div>
            """,
            unsafe_allow_html=True,
        )

        kv1, kv2, kv3 = st.columns(3)
        with kv1:
            st.markdown(
                f"""
<div class="cp-kv">
  <div class="cp-kv-label">Spot</div>
  <div class="cp-kv-value">{s0:,.{top_dec}f}</div>
  <div class="cp-kv-sub">{top_unit}</div>
</div>
                """,
                unsafe_allow_html=True,
            )
        with kv2:
            delta_color = "#166534" if exp_ret > 0 else ("#991b1b" if exp_ret < 0 else "#334155")
            delta_prefix = "+" if exp_ret > 0 else ""
            st.markdown(
                f"""
<div class="cp-kv">
  <div class="cp-kv-label">Forecast (auto)</div>
  <div class="cp-kv-value">{s_mean:,.{top_dec}f}</div>
  <div class="cp-kv-sub" style="color:{delta_color};">{delta_prefix}{exp_ret:.1f}% · Mom3 {mom3:+.1f}%</div>
</div>
                """,
                unsafe_allow_html=True,
            )
        with kv3:
            vol_band = "HIGH" if sigma_ann >= 0.30 else ("MED" if sigma_ann >= 0.20 else "LOW")
            st.markdown(
                f"""
<div class="cp-kv">
  <div class="cp-kv-label">Volatility</div>
  <div class="cp-kv-value">{sigma_ann*100:.0f}%</div>
  <div class="cp-kv-sub">{vol_band} · {months}M</div>
</div>
                """,
                unsafe_allow_html=True,
            )

        legs = top_strat.get("legs") or []
        if legs:
            legs_df = pd.DataFrame(
                [
                    {
                        "Action": l.get("side"),
                        "Type": l.get("type"),
                        "Strike": f"{float(l.get('strike')):,.{top_dec}f}",
                        "Prob ITM": f"{float(l.get('prob_itm', 0.0))*100:.0f}%",
                        "Est Premium": f"{float(l.get('est_premium', 0.0)):.{top_dec}f}",
                        "Note": l.get("note"),
                    }
                    for l in legs
                ]
            )
            st.markdown(
                "<div class='cp-note'><b>Ask the bank for these legs</b> (per unit). Multiply by your hedge size.</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(legs_df, use_container_width=True, height=200)

        if top_strat.get("rationale"):
            st.caption(str(top_strat.get("rationale")))

        if show_portfolio_view:
            # Small table for other assets (so it still feels like a strategist across the portfolio)
            table_rows = []
            for r in recs[:8]:
                stt = r.get("strategy") or {}
                table_rows.append(
                    {
                        "When": r.get("when"),
                        "Asset": r.get("label"),
                        "Strategy": stt.get("title"),
                        "Horizon": r.get("horizon"),
                        "Score": round(float(r.get("score", 0.0)), 1),
                    }
                )
            st.markdown("#### Portfolio View")
            st.caption("Auto-ranked recommendations across all commodities (top 8).")
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, height=260)

        st.caption("Guidance tool: confirm with your bank and internal policy before executing options trades.")

        return


def _theoretical_futures_price_commodity(*, s: float, r: float, storage_cost: float, convenience_yield: float, t_years: float) -> float:
    """Cost-of-carry theoretical futures/forward price for commodities."""
    import math

    s = float(s)
    r = float(r)
    storage_cost = float(storage_cost)
    convenience_yield = float(convenience_yield)
    t_years = float(max(t_years, 1e-9))
    return float(s * math.exp((r + storage_cost - convenience_yield) * t_years))


def _pv_strike(*, k: float, r: float, t_years: float) -> float:
    import math

    k = float(k)
    r = float(r)
    t_years = float(max(t_years, 1e-9))
    return float(k * math.exp(-r * t_years))


def _put_call_parity_gap(*, c: float, p: float, s: float, k: float, r: float, t_years: float) -> float:
    """Parity gap: (C - P) - (S - PV(K)). 0 means no-arbitrage parity holds."""
    return float(c - p) - float(s - _pv_strike(k=k, r=r, t_years=t_years))


def _implied_volatility_bs(*, price: float, s: float, k: float, t_years: float, opt_type: str) -> float | None:
    """Solve for implied vol using bisection. Returns sigma_ann or None if not solvable."""
    try:
        price = float(price)
        s = float(s)
        k = float(k)
        t_years = float(t_years)
        if not (np.isfinite(price) and np.isfinite(s) and np.isfinite(k) and np.isfinite(t_years)):
            return None
        if price <= 0 or s <= 0 or k <= 0 or t_years <= 0:
            return None
    except Exception:
        return None

    opt_type = str(opt_type).upper().strip()
    lo, hi = 0.01, 2.50
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        try:
            px = float(_bs_price(s=s, k=k, t=t_years, sigma=mid, opt_type=opt_type))
        except Exception:
            return None
        if not np.isfinite(px):
            return None
        if px > price:
            hi = mid
        else:
            lo = mid
    sigma = 0.5 * (lo + hi)
    return float(sigma) if np.isfinite(sigma) else None


def render_no_arbitrage_strategist(
    *,
    expander_title: str,
    key_prefix: str,
    commodity_payloads: list[dict] | None,
    expanded: bool = False,
) -> None:
    """No-arbitrage strategist (cost-of-carry + put-call parity) for Summary page.

    Note: This requires market quotes (futures price and/or option quotes) to detect true arbitrage.
    We provide a clean editor to paste bank/broker quotes; the engine then flags mispricing.
    """
    with st.expander(expander_title, expanded=expanded):
        st.markdown(
            """
<div class="cp-card" style="padding: 1.0rem 1.0rem;">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px; flex-wrap:wrap;">
    <div>
      <div class="cp-title">No‑Arbitrage Strategist</div>
      <div class="cp-subtitle">Detects futures & options mispricing using <b>cost‑of‑carry</b> and <b>put‑call parity</b>. Paste market quotes to activate.</div>
    </div>
    <div class="cp-pill cp-pill-hold">Institutional</div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

        if not commodity_payloads:
            st.info("No commodities available.")
            return

        # Global assumptions (kept light; editable)
        c1, c2, c3, c4 = st.columns([1.0, 1.0, 1.0, 1.2])
        with c1:
            r = st.number_input(
                "Risk‑free rate (annual)",
                min_value=0.0,
                max_value=0.25,
                value=0.05,
                step=0.005,
                format="%.3f",
                key=f"{key_prefix}_r",
                help="Use USD risk-free rate for USD series. Example: 0.050 = 5%.",
            )
        with c2:
            storage_cost = st.number_input(
                "Storage cost (annual)",
                min_value=0.0,
                max_value=0.50,
                value=0.00,
                step=0.005,
                format="%.3f",
                key=f"{key_prefix}_storage",
            )
        with c3:
            convenience_yield = st.number_input(
                "Convenience yield (annual)",
                min_value=0.0,
                max_value=0.50,
                value=0.00,
                step=0.005,
                format="%.3f",
                key=f"{key_prefix}_cy",
            )
        with c4:
            threshold_pct = st.number_input(
                "Mispricing threshold (%)",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1,
                format="%.1f",
                key=f"{key_prefix}_thr",
                help="Below this, treat as efficient after costs/slippage.",
            )

        # Build candidate list
        candidates: list[dict] = []
        for item in commodity_payloads:
            for side in ("int_payload", "local_payload"):
                p = item.get(side)
                if p and p.get("name"):
                    candidates.append(
                        {
                            "key": f"{p['name']}::{side}",
                            "label": f"{p['name']} ({'International' if side=='int_payload' else 'Local'})",
                            "payload": p,
                        }
                    )

        if not candidates:
            st.info("No commodities available.")
            return

        # Market quotes editor (bank/broker) + model fair values to avoid "empty" look.
        st.markdown(
            "<div class='cp-kv-label' style='margin-top: 0.25rem;'>Market Quotes (Optional)</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Paste/enter broker quotes in the editable columns (same unit as Spot). "
            "We also compute fair values (carry + Black‑Scholes using historical vol) to benchmark mispricing."
        )

        base_rows: list[dict] = []
        for c in candidates[:10]:
            p = c["payload"]
            scale = float(p.get("display_scale", 1.0) or 1.0)
            unit = str(p.get("display_currency") or p.get("info", {}).get("currency", ""))
            s0 = float(p.get("current_price") or 0.0) * scale

            best_h, _best_pred, months = _pick_best_horizon_for_payload(payload=p, exposure="Procurement (we will BUY later)")
            t_years = float(months) / 12.0
            t_years = float(max(t_years, 1e-6))

            k_atm = float(s0) if np.isfinite(s0) and s0 > 0 else np.nan
            f_fair = (
                _theoretical_futures_price_commodity(
                    s=s0,
                    r=float(r),
                    storage_cost=float(storage_cost),
                    convenience_yield=float(convenience_yield),
                    t_years=t_years,
                )
                if np.isfinite(s0) and s0 > 0
                else np.nan
            )

            sigma_model = np.nan
            try:
                hist_df = p.get("history_df")
                vcol = p.get("info", {}).get("value_col") or p.get("value_col") or "value"
                if isinstance(hist_df, pd.DataFrame) and vcol in hist_df.columns:
                    sigma_model = float(_annualized_volatility_from_history(hist_df[vcol]))
            except Exception:
                sigma_model = np.nan

            c_fair = np.nan
            p_fair = np.nan
            try:
                if np.isfinite(s0) and s0 > 0 and np.isfinite(k_atm) and k_atm > 0 and np.isfinite(t_years) and t_years > 0 and np.isfinite(sigma_model) and sigma_model > 0:
                    c_fair = float(_bs_price(s=s0, k=k_atm, t=t_years, sigma=float(sigma_model), opt_type="CALL"))
                    p_fair = float(_bs_price(s=s0, k=k_atm, t=t_years, sigma=float(sigma_model), opt_type="PUT"))
            except Exception:
                c_fair, p_fair = np.nan, np.nan

            base_rows.append(
                {
                    "Commodity": c["label"],
                    "Target": str(best_h),
                    "T (yrs)": round(t_years, 3),
                    "Spot": s0,
                    "Futures Fair": f_fair,
                    "Futures Quote": np.nan,
                    "Strike (K)": k_atm,
                    "Call Fair": c_fair,
                    "Call Quote": np.nan,
                    "Put Fair": p_fair,
                    "Put Quote": np.nan,
                    "Unit": unit,
                }
            )

        default_quotes = pd.DataFrame(base_rows)
        ss_key = f"{key_prefix}_quotes_df"
        if ss_key not in st.session_state or not isinstance(st.session_state.get(ss_key), pd.DataFrame):
            st.session_state[ss_key] = default_quotes

        # Keep rows in sync if assets list changes
        try:
            cur_df: pd.DataFrame = st.session_state[ss_key]
            if set(cur_df.get("Commodity", [])) != set(default_quotes["Commodity"]):
                st.session_state[ss_key] = default_quotes
        except Exception:
            st.session_state[ss_key] = default_quotes

        with st.expander("Enter / paste market quotes", expanded=False):
            edited = st.data_editor(
                st.session_state[ss_key],
                use_container_width=True,
                hide_index=True,
                key=f"{key_prefix}_editor",
                column_config={
                    "Target": st.column_config.TextColumn(help="Model-selected horizon"),
                    "T (yrs)": st.column_config.NumberColumn(format="%.3f"),
                    "Spot": st.column_config.NumberColumn(format="%.4f"),
                    "Futures Fair": st.column_config.NumberColumn(format="%.4f", help="Carry fair value (S·e^{(r+storage−cy)T})"),
                    "Futures Quote": st.column_config.NumberColumn(format="%.4f", help="Broker/market futures quote"),
                    "Strike (K)": st.column_config.NumberColumn(format="%.4f", help="Default = ATM (Spot). Adjust if quoting a different strike."),
                    "Call Fair": st.column_config.NumberColumn(format="%.4f", help="Black‑Scholes fair premium using historical vol"),
                    "Call Quote": st.column_config.NumberColumn(format="%.4f", help="Broker/market call premium"),
                    "Put Fair": st.column_config.NumberColumn(format="%.4f", help="Black‑Scholes fair premium using historical vol"),
                    "Put Quote": st.column_config.NumberColumn(format="%.4f", help="Broker/market put premium"),
                },
                disabled=["Commodity", "Target", "T (yrs)", "Spot", "Futures Fair", "Call Fair", "Put Fair", "Unit"],
            )
        st.session_state[ss_key] = edited

        # Build strategy outputs (only when at least one quote exists)
        any_quotes = False
        try:
            any_quotes = bool(
                (
                    edited[["Futures Quote", "Call Quote", "Put Quote"]]
                    .apply(pd.to_numeric, errors="coerce")
                    .notna()
                    .any()
                    .any()
                )
            )
        except Exception:
            any_quotes = False

        if not any_quotes:
            st.info("Enter at least one Futures/Call/Put quote above to generate arbitrage & parity strategies.")
            st.caption("Tip: Strike (K) is prefilled ATM to avoid blanks; adjust if you have a quoted strike.")
            return

        outputs: list[dict] = []
        for _, row in edited.iterrows():
            try:
                commodity = str(row["Commodity"])
                unit = str(row.get("Unit", ""))
                t_years = float(row.get("T (yrs)", 0.5))
                s0 = float(row.get("Spot", 0.0))
            except Exception:
                continue

            f_mkt = row.get("Futures Quote")
            c_mkt = row.get("Call Quote")
            p_mkt = row.get("Put Quote")
            k = row.get("Strike (K)")

            # Skip rows with no quotes to keep output clean
            if not any(pd.notna(x) for x in (f_mkt, c_mkt, p_mkt)):
                continue

            opportunity = "—"
            strategy = "—"
            trade = "—"
            signal = "—"
            logic = "—"
            risk = "Costs, funding/margin, execution, liquidity"
            confidence = "Medium"

            # Step 1–3: Cost-of-carry futures mispricing
            if pd.notna(f_mkt) and np.isfinite(float(f_mkt)) and s0 > 0:
                f_mkt_f = float(f_mkt)
                f_th = _theoretical_futures_price_commodity(
                    s=s0,
                    r=float(r),
                    storage_cost=float(storage_cost),
                    convenience_yield=float(convenience_yield),
                    t_years=t_years,
                )
                mis = f_mkt_f - f_th
                mis_pct = (mis / f_th) * 100.0 if f_th else 0.0
                if abs(mis_pct) < float(threshold_pct):
                    opportunity = "Efficient"
                    strategy = "No action"
                    trade = "—"
                    signal = f"F_quote vs F_fair: {mis_pct:+.2f}%"
                    logic = "Futures aligns with carry fair value within threshold."
                    confidence = "Medium"
                elif mis_pct < 0:
                    opportunity = "Carry Mispricing"
                    strategy = "Cash‑and‑Carry Arbitrage"
                    trade = "Short spot, invest cash, long futures"
                    signal = f"F_quote vs F_fair: {mis_pct:+.2f}%"
                    logic = "Forward underpriced vs carry fair value."
                    confidence = "High" if abs(mis_pct) >= 2 * float(threshold_pct) else "Medium"
                else:
                    opportunity = "Carry Mispricing"
                    strategy = "Reverse Cash‑and‑Carry"
                    trade = "Long spot (financed), short futures"
                    signal = f"F_quote vs F_fair: {mis_pct:+.2f}%"
                    logic = "Forward overpriced vs carry fair value."
                    confidence = "High" if abs(mis_pct) >= 2 * float(threshold_pct) else "Medium"

            # Step 4: Put-call parity / synthetic forward
            if all(pd.notna(x) for x in (c_mkt, p_mkt, k)) and s0 > 0:
                try:
                    c_f = float(c_mkt)
                    p_f = float(p_mkt)
                    k_f = float(k)
                    gap = _put_call_parity_gap(c=c_f, p=p_f, s=s0, k=k_f, r=float(r), t_years=t_years)
                    gap_pct = (gap / s0) * 100.0 if s0 else 0.0
                    parity_thr = float(threshold_pct) / 2.0
                    if abs(gap_pct) >= parity_thr:
                        # Classic parity conversion / reversal framing
                        if gap > 0:
                            parity_cond = "Parity distorted (C−P too high)"
                            parity_strat = "Reversal (Sell Synthetic Forward)"
                            parity_trade = "Sell Call, Buy Put (same K,T) + Buy PV(K) + Short Spot"
                        else:
                            parity_cond = "Parity distorted (C−P too low)"
                            parity_strat = "Conversion (Buy Synthetic Forward)"
                            parity_trade = "Buy Call, Sell Put (same K,T) + Short PV(K) + Buy Spot"

                        if opportunity == "—" or opportunity == "Efficient":
                            opportunity = "Parity Distortion"
                        strategy = parity_strat if strategy in ("—", "No action") else f"{strategy} + {parity_strat}"
                        trade = parity_trade if trade == "—" else f"{trade} | Options: {parity_trade}"
                        signal = f"Parity gap: {gap_pct:+.2f}% of spot"
                        logic = "Put‑call parity violation (should be ~0 after carry + funding)."
                        confidence = "High" if abs(gap_pct) >= 2 * parity_thr else "Medium"

                    # Volatility mispricing (market IV vs model sigma)
                    # Compare implied vol from call with our historical sigma (if available)
                    sigma_model = None
                    try:
                        # Find matching payload to pull sigma
                        pld = next((c["payload"] for c in candidates if c["label"] == commodity), None)
                        if isinstance(pld, dict):
                            hist_df = pld.get("history_df")
                            vcol = pld.get("info", {}).get("value_col") or pld.get("value_col") or "value"
                            if isinstance(hist_df, pd.DataFrame) and vcol in hist_df.columns:
                                sigma_model = float(_annualized_volatility_from_history(hist_df[vcol]))
                    except Exception:
                        sigma_model = None

                    iv_call = _implied_volatility_bs(price=c_f, s=s0, k=k_f, t_years=t_years, opt_type="CALL")
                    if iv_call is not None and sigma_model is not None and np.isfinite(iv_call) and np.isfinite(sigma_model):
                        iv_gap = (iv_call - sigma_model) * 100.0
                        if abs(iv_gap) >= 10.0:
                            if opportunity == "—" or opportunity == "Efficient":
                                opportunity = "Vol Relative‑Value"
                            strategy = "Vol Relative‑Value" if strategy in ("—", "No action") else f"{strategy} + Vol RV"
                            signal = f"IV(call) − σ_model: {iv_gap:+.0f} vol pts"
                            logic = f"Implied vol {iv_call*100:.0f}% vs model vol {sigma_model*100:.0f}%."
                            risk = "Model risk, vega exposure, liquidity"
                except Exception:
                    pass

            outputs.append(
                {
                    "Commodity": commodity,
                    "Opportunity": opportunity,
                    "Strategy": strategy,
                    "Trade Steps": trade,
                    "Key Signal": signal,
                    "Rationale": logic,
                    "Risk Notes": risk,
                    "Confidence": confidence,
                }
            )

        st.markdown("<div class='cp-kv-label' style='margin-top: 0.75rem;'>Strategy Output</div>", unsafe_allow_html=True)
        out_df = pd.DataFrame(outputs)
        if out_df.empty:
            st.info("No actionable rows yet. Add a Futures/Call/Put quote for at least one commodity.")
            return

        try:
            out_df["__conf_rank"] = out_df["Confidence"].map({"High": 2, "Medium": 1, "Low": 0}).fillna(1)
            out_df = out_df.sort_values(["__conf_rank", "Commodity"], ascending=[False, True]).drop(columns=["__conf_rank"])
        except Exception:
            pass

        def _conf_style(v: str) -> str:
            vv = str(v)
            if "High" in vv:
                return "background-color:#052e16; color:#dcfce7; font-weight:900;"
            if "Medium" in vv:
                return "background-color:#1e3a8a; color:#e0e7ff; font-weight:900;"
            return "background-color:#0f172a; color:#e5e7eb; font-weight:900;"

        styled = (
            out_df.style
            .applymap(_conf_style, subset=["Confidence"])
            .set_properties(**{"font-size": "0.85rem", "font-weight": "700", "padding": "10px 12px"})
            .set_table_styles(
                [
                    {
                        "selector": "thead th",
                        "props": [
                            ("background-color", "#0b1220"),
                            ("color", "#e5e7eb"),
                            ("font-weight", "900"),
                            ("padding", "12px 12px"),
                            ("font-size", "0.75rem"),
                            ("text-transform", "uppercase"),
                        ],
                    }
                ]
            )
        )
        st.dataframe(styled, use_container_width=True, height=360)

        st.caption(
            "For real arbitrage, include transaction costs, funding/margin, and whether physical storage is feasible. This is a decision-support tool, not a guarantee."
        )
        return


def _render_pakistan_forecast_chart_table(*, title: str, caption: str, predictions: dict, currency: str, key_prefix: str) -> None:
    st.markdown(f"### {title}")
    st.caption(caption)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(create_forecast_bar_chart(predictions, currency), use_container_width=True, key=f"{key_prefix}_chart")
    with col2:
        forecast_df = create_forecast_table(predictions, currency)

        def color_change(val):
            if isinstance(val, str) and '%' in val:
                try:
                    num = float(val.replace('%', '').replace('+', '').replace(' ', ''))
                    if num > 5:
                        return 'background-color: #dcfce7; color: #166534; font-weight: bold'
                    if num > 0:
                        return 'background-color: #e0f2fe; color: #075985; font-weight: bold'
                    if num < -5:
                        return 'background-color: #fee2e2; color: #991b1b; font-weight: bold'
                    if num < 0:
                        return 'background-color: #fed7aa; color: #9a3412; font-weight: bold'
                except Exception:
                    return ''
            return ''

        styled_df = forecast_df.style.applymap(color_change, subset=['Change']).set_properties(**{
            'text-align': 'center',
            'font-size': '12px',
            'font-family': 'Arial, sans-serif',
            'padding': '10px',
            'border': '1px solid #e2e8f0'
        }).set_table_styles([
            {'selector': 'thead th', 'props': [
                ('background-color', '#1e40af'),
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('font-size', '13px'),
                ('padding', '12px 8px'),
                ('border', '1px solid #1e3a8a')
            ]},
            {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#f8fafc')]},
            {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', '#ffffff')]},
        ])

        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=440)


@st.cache_data
def load_predictions(asset: str):
    """Generate predictions for commodity (12 monthly horizons)."""
    pred_files = list(ARTIFACTS_DIR.glob(f"{asset}/predictions_*.csv"))
    base_price = 1500
    
    if pred_files:
        try:
            df = pd.read_csv(pred_files[0])
            # Find numeric price column (skip date/text columns)
            for col in df.columns:
                col_lower = str(col).lower()
                if col_lower not in ['date', 'time', 'datetime', 'horizon', 'index']:
                    try:
                        # Try to convert first value to float
                        test_val = pd.to_numeric(df[col].iloc[0], errors='coerce')
                        if pd.notna(test_val):
                            base_price = float(test_val)
                            break
                    except:
                        continue
        except Exception as e:
            # If anything fails, use default
            pass

    # If we don't have artifact predictions, derive base price from raw history
    if base_price == 1500:
        raw_files = list(RAW_DATA_DIR.glob(f"{asset}*.csv"))
        if raw_files:
            try:
                raw_df = pd.read_csv(raw_files[0])
                value_col = None
                for possible_name in ['value', 'price_usd', 'price_rmb', 'price_pkr', 'price', 'close']:
                    if possible_name in raw_df.columns:
                        value_col = possible_name
                        break
                if value_col is None:
                    for col in raw_df.columns:
                        if str(col).lower() in ['timestamp', 'date', 'time', 'datetime', 'index']:
                            continue
                        if pd.api.types.is_numeric_dtype(raw_df[col]):
                            value_col = col
                            break
                if value_col and len(raw_df) > 0:
                    derived = pd.to_numeric(raw_df[value_col].iloc[-1], errors='coerce')
                    if pd.notna(derived):
                        base_price = float(derived)
            except Exception:
                pass

    # Cloud fallback: derive base price from Supabase commodity series
    if base_price == 1500:
        sb_df = supabase_fetch_commodity_series(asset)
        try:
            if sb_df is not None and not sb_df.empty:
                base_price = float(sb_df["value"].iloc[-1])
        except Exception:
            pass
    
    # Commodity forecasting: Use MONTHS not hours (standard industry practice)
    predictions = {}
    horizons = get_month_horizons(12)
    for i, horizon in enumerate(horizons, start=1):
        # Longer horizon = more uncertainty
        prediction = base_price * (1 + np.random.normal(0, i * 1.0/100))
        change = np.random.uniform(-5, 5) * (1 + i * 0.15)  # More volatility for longer periods
        
        # Calculate confidence interval (wider for longer horizons)
        uncertainty = i * 2.0  # percentage uncertainty
        lower_bound = prediction * (1 - uncertainty/100)
        upper_bound = prediction * (1 + uncertainty/100)
        confidence = max(60, 95 - i * 2)  # Decrease confidence for longer periods
        
        predictions[horizon] = {
            'price': prediction,
            'change': change,
            'action': 'BUY NOW' if change > 5 else 'HOLD' if abs(change) < 3 else 'WAIT',
            'lower': lower_bound,
            'upper': upper_bound,
            'confidence': confidence
        }

    # Convert ton-based series to per-kg when the market UOM is kg
    ton_assets_to_kg = (
        "polyester/polyester_usd_monthly",
        "viscose/viscose_usd_monthly",
        "polyester/polyester_pkr_monthly",
        "viscose/viscose_pkr_monthly",
    )
    if any(k in str(asset) for k in ton_assets_to_kg):
        for h in list(predictions.keys()):
            p = predictions[h]
            p['price'] = float(p['price']) / 1000.0
            p['lower'] = float(p['lower']) / 1000.0
            p['upper'] = float(p['upper']) / 1000.0

    return predictions


def generate_usd_pkr_forecast(current_rate: float):
    """Generate USD/PKR exchange rate forecast."""
    predictions = {}
    horizons = get_month_horizons(12)
    
    # Pakistan rupee typically depreciates 3-8% annually
    annual_depreciation_rate = 0.05  # 5% base annual depreciation
    
    for i, horizon in enumerate(horizons, start=1):
        months = i
        
        # Calculate expected depreciation
        expected_rate = current_rate * (1 + (annual_depreciation_rate * months / 12))
        
        # Add some volatility
        volatility = current_rate * 0.02 * (months / 12)  # 2% volatility per year
        prediction = expected_rate + np.random.normal(0, volatility)
        
        change = ((prediction - current_rate) / current_rate) * 100
        
        # Calculate confidence interval
        uncertainty = 1.5 * (i + 1)  # percentage uncertainty
        lower_bound = prediction * (1 - uncertainty/100)
        upper_bound = prediction * (1 + uncertainty/100)
        confidence = max(65, 92 - i * 4)
        
        predictions[horizon] = {
            'price': prediction,
            'change': change,
            'action': 'HEDGE' if change > 4 else 'MONITOR' if change > 2 else 'STABLE',
            'lower': lower_bound,
            'upper': upper_bound,
            'confidence': confidence
        }
    
    return predictions


def generate_energy_forecast(current_price: float, commodity_type: str):
    """Generate energy price forecast (electricity, gas, oil)."""
    predictions = {}
    horizons = get_month_horizons(12)
    
    # Energy typically has seasonal variations and general upward trend
    if commodity_type == 'electricity':
        annual_increase = 0.08  # 8% annual increase typical for Pakistan
    elif commodity_type == 'natural_gas':
        annual_increase = 0.06  # 6% annual increase
    else:  # crude oil
        annual_increase = 0.04  # 4% annual volatility
    
    for i, horizon in enumerate(horizons, start=1):
        months = i
        
        # Base forecast with seasonal adjustment
        seasonal_factor = 1 + 0.05 * np.sin(months * np.pi / 6)  # Seasonal variation
        expected_price = current_price * (1 + (annual_increase * months / 12)) * seasonal_factor
        
        # Add volatility
        volatility = current_price * 0.03 * (months / 12)
        prediction = expected_price + np.random.normal(0, volatility)
        
        change = ((prediction - current_price) / current_price) * 100
        
        # Calculate confidence interval
        uncertainty = 2.0 * (i + 1)
        lower_bound = prediction * (1 - uncertainty/100)
        upper_bound = prediction * (1 + uncertainty/100)
        confidence = max(60, 90 - i * 4)
        
        predictions[horizon] = {
            'price': prediction,
            'change': change,
            'action': 'LOCK RATES' if change > 6 else 'PLAN AHEAD' if change > 3 else 'MONITOR',
            'lower': lower_bound,
            'upper': upper_bound,
            'confidence': confidence
        }
    
    return predictions


def get_recommendation(predictions):
    """Generate procurement recommendation based on commodity forecast."""
    # Look at medium-term (3rd and 6th month) trend
    horizons = get_prediction_horizons(predictions)
    if len(horizons) >= 6:
        medium_term = [horizons[2], horizons[5]]
    elif len(horizons) >= 3:
        medium_term = [horizons[2]]
    else:
        medium_term = horizons

    changes = [predictions[h]['change'] for h in medium_term if h in predictions]
    avg_change = sum(changes) / len(changes) if changes else 0
    
    if avg_change > 5:
        return "🚨 **STRONG UPWARD TREND**: Prices expected to rise significantly. Lock in contracts NOW at current prices."
    elif avg_change > 2:
        return "📈 **MODERATE INCREASE**: Gradual price rise expected. Consider increasing inventory levels."
    elif avg_change < -5:
        return "📉 **DECLINING MARKET**: Prices falling. Delay non-urgent procurement for better rates."
    elif avg_change < -2:
        return "⬇️ **SOFTENING**: Slight price decline expected. Good opportunity for spot market purchases."
    else:
        return "📊 **STABLE MARKET**: No major price movement forecast. Continue normal procurement cycles."


def create_price_chart(metadata, name):
    """Create clean price trend chart."""
    if metadata['df'] is not None:
        df = metadata['df'].tail(12)  # Last 12 months
        
        fig = go.Figure()
        # Format values for text labels
        values = df[metadata['value_col']].values
        dec = 3 if "/lb" in str(metadata.get("currency", "")).lower() else 2
        text_labels = [f"{val:,.{dec}f}" for val in values]
        
        fig.add_trace(go.Scatter(
            x=df[metadata['time_col']],
            y=df[metadata['value_col']],
            mode='lines+markers+text',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=8, color='#3b82f6'),
            text=text_labels,
            textposition='top center',
            textfont=dict(size=9, color='#1e40af'),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)',
            name=name
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=60, r=20, t=40, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', size=12),
            showlegend=False,
            xaxis=dict(
                gridcolor='rgba(148, 163, 184, 0.2)',
                showgrid=True,
                title='Month',
                tickangle=-45,
                tickfont=dict(size=10)
            ),
            yaxis=dict(
                gridcolor='rgba(148, 163, 184, 0.2)',
                showgrid=True,
                title=f'{metadata["currency"]}',
                tickformat=(f',.{dec}f' if metadata['df'][metadata['value_col']].mean() < 100 else ',.0f'),
                tickfont=dict(size=11)
            ),
            hovermode='x unified'
        )
        
        return fig
    return None


def create_forecast_table(predictions, currency):
    """Create forecast data table for commodity (monthly horizons)."""
    dec = 3 if "/lb" in str(currency).lower() else 2
    data = []
    horizons = get_prediction_horizons(predictions)
    for horizon in horizons:
        pred = predictions[horizon]
        price_range = f"{pred['lower']:,.{dec}f} - {pred['upper']:,.{dec}f}"
        data.append({
            'Period': horizon,
            f'Price ({currency})': f"{pred['price']:,.{dec}f}",
            'Range': price_range,
            'Confidence': f"{pred['confidence']}%",
            'Change': f"{pred['change']:+.1f}%"
        })
    return pd.DataFrame(data)


def create_forecast_bar_chart(predictions, currency):
    """Create bar chart visualization for price forecasts with confidence intervals."""
    dec = 3 if "/lb" in str(currency).lower() else 2
    horizons = get_prediction_horizons(predictions)
    prices = [predictions[h]['price'] for h in horizons]
    changes = [predictions[h]['change'] for h in horizons]
    lower_bounds = [predictions[h]['lower'] for h in horizons]
    upper_bounds = [predictions[h]['upper'] for h in horizons]
    confidences = [predictions[h]['confidence'] for h in horizons]
    
    # Professional financial color scheme - inspired by Bloomberg/Reuters terminals
    # Strong decline: deep red, moderate decline: coral, stable: navy blue, 
    # moderate rise: teal, strong rise: forest green
    def get_professional_color(change):
        if change > 7:
            return '#16a34a'  # Strong rise - forest green
        elif change > 3:
            return '#0891b2'  # Moderate rise - teal
        elif change > -3:
            return '#1e40af'  # Stable - navy blue
        elif change > -7:
            return '#ea580c'  # Moderate decline - deep orange
        else:
            return '#dc2626'  # Strong decline - strong red
    
    colors = [get_professional_color(c) for c in changes]
    
    # Calculate error bar sizes
    error_y = [u - p for u, p in zip(upper_bounds, prices)]
    error_y_minus = [p - l for p, l in zip(prices, lower_bounds)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=horizons,
            y=prices,
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.2)', width=1.5),
                # Add subtle gradient for premium look
                pattern=dict(shape="")
            ),
            error_y=dict(
                type='data',
                symmetric=False,
                array=error_y,
                arrayminus=error_y_minus,
                color='rgba(100,116,139,0.25)',
                thickness=1,
                width=0  # Remove the "I" caps for cleaner look
            ),
            text=[f"<b>{p:,.{dec}f}</b><br><span style='font-size:11px'>{c:+.1f}%</span>" for p, c in zip(prices, changes)],
            textposition='outside',
            textfont=dict(size=12, color='#1e293b', family='Arial, sans-serif'),
            hovertemplate='<b>%{x}</b><br>' +
                         f'<b>Price:</b> %{{y:,.{dec}f}} ' + currency + '<br>' +
                         '<b>Change:</b> %{customdata[0]:+.1f}%<br>' +
                         '<b>Confidence:</b> %{customdata[1]}%<br>' +
                         f'<b>Range:</b> %{{customdata[2]:,.{dec}f}} - %{{customdata[3]:,.{dec}f}}<br>' +
                         '<extra></extra>',
            customdata=[[c, conf, l, u] for c, conf, l, u in zip(changes, confidences, lower_bounds, upper_bounds)]
        )
    ])
    
    fig.update_layout(
        title=dict(
            text=f"<b>Price Forecast with Confidence Intervals</b>",
            font=dict(size=14, color='#1e293b', family='Inter, sans-serif'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=dict(text='<b>Month</b>', font=dict(size=12, color='#475569')),
            tickangle=-35,
            tickfont=dict(size=9, color='#64748b')
        ),
        yaxis=dict(
            title=dict(text=f'<b>Price ({currency})</b>', font=dict(size=12, color='#475569')),
            tickformat=f',.{dec}f',
            tickfont=dict(size=10, color='#64748b')
        ),
        height=420,
        margin=dict(l=60, r=20, t=50, b=110),
        plot_bgcolor='#fafafa',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        font=dict(family='Inter, sans-serif')
    )
    
    fig.update_xaxes(showgrid=False, showline=True, linewidth=1, linecolor='#cbd5e1')
    fig.update_yaxes(showgrid=True, gridcolor='#e2e8f0', gridwidth=1, showline=True, linewidth=1, linecolor='#cbd5e1')
    fig.update_traces(cliponaxis=False)
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    
    return fig


def render_pending_data_card(name: str, info: dict):
    """Render placeholder for pending data sources."""
    st.markdown(f"""
    <div class='metric-card' style='opacity: 0.6; border: 2px dashed rgba(59, 130, 246, 0.3);'>
        <div class='metric-label'>{name}</div>
        <div class='metric-value' style='font-size: 1.2rem; color: #94a3b8;'>Data Pending</div>
        <div class='currency-label'>📡 Source: {info['source']}</div>
        <div style='margin-top: 0.5rem; font-size: 0.75rem; color: #64748b;'>
            Integration planned
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_market_page(commodities_config: dict, page_title: str, page_description: str):
    """Generic market page renderer - works for both international and local."""
    st.markdown(f"""
    <div style='border-left: 4px solid #2563eb; padding-left: 1rem; margin: 1rem 0 1.25rem 0;'>
        <h2 style='font-size: 1.3rem; font-weight: 700; color: #1e293b; letter-spacing: -0.3px; margin: 0 0 0.25rem 0;'>
            {page_title}
        </h2>
        <p style='font-size: 0.825rem; color: #64748b; font-weight: 500; margin: 0; line-height: 1.4;'>
            {page_description}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load all commodity data
    commodities_data = {}
    for name, info in commodities_config.items():
        metadata = load_commodity_data(info["path"], info["currency"])
        predictions = load_predictions(info["path"])
        commodities_data[name] = (metadata, predictions, info["icon"], info.get("type", "Market Data"))
    
    # Create tabs for each commodity
    tab_labels = [f"{info['icon']} {name}" for name, info in commodities_config.items()] + ["📊 Overview"]
    tabs = st.tabs(tab_labels)
    
    # Individual commodity tabs
    for idx, (name, (metadata, predictions, icon, data_type)) in enumerate(commodities_data.items()):
        with tabs[idx]:
            render_commodity_tab(name, metadata, predictions, icon, data_type)
    
    # Overview/Comparison tab
    with tabs[-1]:
        render_overview_tab(commodities_data, page_title)


def render_commodity_tab(name: str, metadata: dict, predictions: dict, icon: str, data_type: str):
    """Render individual commodity analysis tab."""
    st.markdown(f"### {icon} {name} Analysis")
    st.caption(f"📍 Data Type: {data_type}")
    
    # Data freshness indicator - calmer approach
    if metadata['df'] is not None:
        latest_date = pd.to_datetime(metadata['df'][metadata['time_col']].iloc[-1])
        data_age_days = (pd.Timestamp.now() - latest_date).days
        
        if data_age_days < 0:
            st.info(f"📊 Futures data · {latest_date.strftime('%B %Y')} contract pricing")
        elif data_age_days > 180:
            st.info(f"📅 Historical data · Last update: {latest_date.strftime('%B %Y')}")
        else:
            st.success(f"✓ Current data · As of {latest_date.strftime('%B %Y')}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Current/Latest Price
        if metadata['df'] is not None:
            current_price = metadata['df'][metadata['value_col']].iloc[-1]
            is_cotton_lb = "/lb" in str(metadata.get("currency", "")).lower()
            if is_cotton_lb:
                price_display = f"{float(current_price):.3f}"
            else:
                price_display = f"{current_price:,.0f}" if current_price > 100 else f"{current_price:.2f}"
            latest_date = pd.to_datetime(metadata['df'][metadata['time_col']].iloc[-1])
            
            st.markdown(f"""
            <div class='metric-card' style='border-left: 4px solid #2563eb;'>
                <div class='metric-label' style='color: #2563eb;'>📍 CURRENT PRICE</div>
                <div class='metric-value' style='color: #1e40af;'>{price_display}</div>
                <div class='currency-label' style='color: #64748b;'><strong>{metadata['currency']}</strong><br><span style='font-size: 0.7rem;'>As of {latest_date.strftime('%b %Y')}</span></div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Historical volatility - Last 6 months
        if metadata['df'] is not None and len(metadata['df']) >= 6:
            recent_df = metadata['df'].tail(6)
            price_min = recent_df[metadata['value_col']].min()
            price_max = recent_df[metadata['value_col']].max()

            is_cotton_lb = "/lb" in str(metadata.get("currency", "")).lower()
            if is_cotton_lb:
                range_display = f"{float(price_min):.3f} — {float(price_max):.3f}"
            else:
                range_display = f"{price_min:,.0f} — {price_max:,.0f}" if price_min > 100 else f"{price_min:.2f} — {price_max:.2f}"
            
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label' style='color: #059669;'>📊 6-MONTH RANGE</div>
                <div class='metric-value' style='font-size: 1.4rem; color: #059669;'>{range_display}</div>
                <div class='currency-label' style='color: #64748b;'><strong>{metadata['currency']}</strong><br><span style='font-size: 0.7rem;'>Price volatility</span></div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        change_class = 'trend-positive' if metadata['price_change'] >= 0 else 'trend-negative'
        arrow = '↑' if metadata['price_change'] >= 0 else '↓'
        trend_text = "Rising" if metadata['price_change'] > 2 else "Falling" if metadata['price_change'] < -2 else "Stable"
        trend_color = '#059669' if metadata['price_change'] > 0 else '#dc2626' if metadata['price_change'] < 0 else '#0891b2'
        
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label' style='color: {trend_color};'>📈 TREND</div>
            <div class='metric-value'><span class='{change_class}' style='color: {trend_color};'>{arrow} {trend_text}</span></div>
            <div class='currency-label'><strong style='color: {trend_color};'>{abs(metadata['price_change']):.1f}%</strong> change<br><span style='font-size: 0.7rem; color: #64748b;'>vs prior period</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card' style='background: linear-gradient(135deg, #fefce8 0%, #f8fafc 100%);'>
            <div class='metric-label' style='color: #a16207;'>💾 Data Points</div>
            <div class='metric-value' style='color: #a16207;'>{metadata['data_points']}</div>
            <div class='currency-label'><strong>Observations</strong><br><span style='font-size: 0.7rem; color: #64748b;'>Historical records</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"<div class='recommendation'>{get_recommendation(predictions)}</div>", unsafe_allow_html=True)
    
    # Historical Price Chart
    st.markdown("#### 📈 Historical Trend")
    chart = create_price_chart(metadata, name)
    if chart:
        st.plotly_chart(chart, use_container_width=True, key=f"chart_{name}")
    
    st.markdown("---")
    
    # Forecast Visualization - Bar Chart and Table
    st.markdown("#### 🔮 Price Forecast & Procurement Guidance")
    st.caption("📊 Directional guidance based on historical patterns · Confidence intervals shown")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Bar chart for visual forecast
        forecast_chart = create_forecast_bar_chart(predictions, metadata['currency'])
        st.plotly_chart(forecast_chart, use_container_width=True, key=f"forecast_{name}")
    
    with col2:
        # Data table for details with professional styling
        forecast_df = create_forecast_table(predictions, metadata['currency'])
        
        # Define function to color code change values
        def color_change(val):
            if isinstance(val, str) and '%' in val:
                num = float(val.replace('%', '').replace('+', '').replace(' ', ''))
                if num > 5:
                    return 'background-color: #dcfce7; color: #166534; font-weight: bold'
                elif num > 0:
                    return 'background-color: #e0f2fe; color: #075985; font-weight: bold'
                elif num < -5:
                    return 'background-color: #fee2e2; color: #991b1b; font-weight: bold'
                elif num < 0:
                    return 'background-color: #fed7aa; color: #9a3412; font-weight: bold'
            return ''
        
        # Style the dataframe professionally
        styled_df = forecast_df.style.applymap(color_change, subset=['Change']).set_properties(**{
            'text-align': 'center',
            'font-size': '12px',
            'font-family': 'Arial, sans-serif',
            'padding': '10px',
            'border': '1px solid #e2e8f0'
        }).set_table_styles([
            {'selector': 'thead th', 'props': [
                ('background-color', '#1e40af'),
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('font-size', '13px'),
                ('padding', '12px 8px'),
                ('border', '1px solid #1e3a8a')
            ]},
            {'selector': 'tbody tr:nth-child(even)', 'props': [
                ('background-color', '#f8fafc')
            ]},
            {'selector': 'tbody tr:nth-child(odd)', 'props': [
                ('background-color', '#ffffff')
            ]},
            {'selector': 'tbody tr:hover', 'props': [
                ('background-color', '#e0e7ff'),
                ('transition', 'background-color 0.2s ease')
            ]},
            {'selector': 'tbody td', 'props': [
                ('color', '#1e293b')
            ]}
        ])
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=440)
    
    # Export to Excel button
    st.markdown("---")
    st.markdown("#### 📥 Export Predictions for Verification")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("Download detailed predictions with historical data, statistics, and confidence intervals for Excel analysis")
    with col2:
        # Create export data
        export_data = {
            'Forecast Period': [],
            'Predicted Price': [],
            'Change %': [],
            'Lower Bound': [],
            'Upper Bound': [],
            'Confidence %': [],
            'Recommendation': []
        }
        
        horizons = get_prediction_horizons(predictions)
        for horizon in horizons:
            pred = predictions[horizon]
            export_data['Forecast Period'].append(horizon)
            export_data['Predicted Price'].append(pred['price'])
            export_data['Change %'].append(pred['change'])
            export_data['Lower Bound'].append(pred['lower'])
            export_data['Upper Bound'].append(pred['upper'])
            export_data['Confidence %'].append(pred['confidence'])
            export_data['Recommendation'].append(pred['action'])
        
        forecast_export_df = pd.DataFrame(export_data)
        
        # Download button
        csv_data = forecast_export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📊 Download Excel",
            data=csv_data,
            file_name=f"{name.replace(' ', '_')}_predictions_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )


def render_overview_tab(commodities_data: dict, title: str):
    """Render overview/comparison tab."""
    st.markdown(f"### 📊 {title} - Overview")
    
    # Summary cards
    cols = st.columns(len(commodities_data))
    for idx, (name, (metadata, predictions, icon, data_type)) in enumerate(commodities_data.items()):
        with cols[idx]:
            st.markdown(f"#### {icon} {name}")
            
            if metadata['df'] is not None and len(metadata['df']) >= 6:
                recent_df = metadata['df'].tail(6)
                price_min = recent_df[metadata['value_col']].min()
                price_max = recent_df[metadata['value_col']].max()

                is_cotton_lb = "/lb" in str(metadata.get("currency", "")).lower()
                if is_cotton_lb:
                    range_display = f"{float(price_min):.3f}-{float(price_max):.3f}"
                else:
                    range_display = f"{price_min:,.0f}-{price_max:,.0f}" if price_min > 100 else f"{price_min:.2f}-{price_max:.2f}"
                change_class = 'trend-positive' if metadata['price_change'] >= 0 else 'trend-negative'
                arrow = '↑' if metadata['price_change'] >= 0 else '↓'
                
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Range</div>
                    <div class='metric-value' style='font-size: 1.3rem;'>{range_display}</div>
                    <div class='currency-label'>{metadata['currency']}</div>
                    <div class='{change_class}' style='margin-top: 0.5rem;'>{arrow} {abs(metadata['price_change']):.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("#### 📈 Normalized Comparison")
    st.caption("All series indexed to 100 at start period")
    
    fig = go.Figure()
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
    
    for idx, (name, (metadata, _, icon, _)) in enumerate(commodities_data.items()):
        if metadata['df'] is not None:
            df = metadata['df'].tail(12).copy()
            first_val = df[metadata['value_col']].iloc[0]
            df['normalized'] = ((df[metadata['value_col']] / first_val) - 1) * 100
            
            latest_date = pd.to_datetime(df[metadata['time_col']].iloc[-1])
            data_age_days = (pd.Timestamp.now() - latest_date).days
            label_suffix = " (futures)" if data_age_days < 0 else " (outdated)" if data_age_days > 180 else ""
            
            fig.add_trace(go.Scatter(
                x=df[metadata['time_col']],
                y=df['normalized'],
                mode='lines+markers',
                name=name + label_suffix,
                line=dict(width=3, color=colors[idx % len(colors)]),
                marker=dict(size=8)
            ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=30, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', showgrid=True, title='% Change (Indexed)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Hedge Advisor (Strategist) — moved here from Summary
    st.markdown("---")
    st.markdown("### 🧠 Hedge Advisor (Options Strategist)")
    st.caption("Actionable option structures using history (volatility + momentum) and forecast distribution.")

    usd_pkr_rate = None
    try:
        live = fetch_usd_pkr_rate()
        if live and live.get("current_price"):
            usd_pkr_rate = float(live["current_price"])
    except Exception:
        usd_pkr_rate = None

    commodity_payloads, _all_summary = build_summary_commodity_payloads(
        show_local_in_usd=True,
        usd_pkr_rate=usd_pkr_rate,
    )
    render_call_put_hedge_advisor(
        expander_title="Hedge Advisor",
        expanded=True,
        key_prefix="ai_cp",
        commodity_payloads=commodity_payloads,
        variant="full",
        use_expander=False,
        show_portfolio_view=False,
    )

    return


SUMMARY_COMMODITY_PAIRS: list[tuple[str, str | None]] = [
    ("Cotton", "Cotton (Local)"),
    ("Polyester", "Polyester (Local)"),
    ("Viscose", "Viscose (Local)"),
    ("Natural Gas", "Natural Gas"),
    ("Crude Oil", "Crude Oil"),
]


def build_commodity_payload(commodity_name: str | None, commodities_dict: dict) -> dict | None:
    if not commodity_name or commodity_name not in commodities_dict:
        return None

    info = commodities_dict[commodity_name]
    metadata = load_commodity_data(info["path"], info["currency"])
    predictions = load_predictions(info["path"])

    if metadata["df"] is None:
        return None

    current_price = metadata["df"][metadata["value_col"]].iloc[-1]
    price_change = metadata["price_change"]
    pred_1m = get_prediction_by_index(predictions, 1)
    pred_3m = get_prediction_by_index(predictions, 3)

    return {
        "name": commodity_name,
        "info": info,
        "current_price": current_price,
        "price_change": price_change,
        "pred_1m": pred_1m,
        "pred_3m": pred_3m,
        "predictions": predictions,
        "history_df": metadata.get("df"),
        "value_col": metadata.get("value_col"),
        # Display overrides (used by Summary page only)
        "display_scale": 1.0,
        "display_currency": info.get("currency"),
    }


def _summary_force_usd_per_kg(p: dict | None) -> None:
    if not p:
        return
    nm = str(p.get("name", "")).lower()
    if not ("polyester" in nm or "viscose" in nm):
        return

    cur = str(p.get("display_currency") or p.get("info", {}).get("currency", ""))
    cur_lower = cur.lower()
    scale0 = float(p.get("display_scale", 1.0) or 1.0)
    uom_scale = 1.0

    # If current unit is /lb, convert to /kg
    if "/lb" in cur_lower:
        uom_scale *= 2.20462262185  # USD/lb -> USD/kg
        cur = cur.replace("/lb", "/kg").replace("/LB", "/kg").replace("/Lb", "/kg")

    # If current unit is /ton, convert to /kg
    if "/ton" in cur_lower:
        uom_scale *= 1.0 / 1000.0
        cur = cur.replace("/ton", "/kg").replace("/TON", "/kg").replace("/Ton", "/kg")

    # Ensure USD prefix (for Summary consistency)
    if "USD" not in cur:
        cur = cur.replace("PKR", "USD", 1) if "PKR" in cur else f"USD ({cur})"

    p["display_scale"] = scale0 * uom_scale
    p["display_currency"] = cur


def _summary_apply_local_usd_conversion(*, local_payload: dict | None, usd_pkr_rate: float | None) -> None:
    if not local_payload:
        return
    if not usd_pkr_rate or usd_pkr_rate <= 0:
        return

    base_scale = 1.0 / float(usd_pkr_rate)
    cur = str(local_payload["info"].get("currency", ""))

    # Keep UOM consistent and market-practice.
    # - Cotton: Local is PKR/maund, convert to USD/lb (1 maund = 40kg = 88.1849 lb).
    # - If any legacy local series still uses /ton while CEO wants /kg, convert /ton -> /kg (÷1000).
    uom_scale = 1.0
    cur_uom = cur
    cur_lower = cur.lower()

    if "cotton" in str(local_payload.get("name", "")).lower() and "/maund" in cur_lower:
        maund_lb = 40.0 * 2.20462262185
        uom_scale = 1.0 / maund_lb
        cur_uom = cur_uom.replace("/maund", "/lb").replace("/MAUND", "/lb").replace("/Maund", "/lb")
    elif (
        "polyester" in str(local_payload.get("name", "")).lower()
        or "viscose" in str(local_payload.get("name", "")).lower()
    ) and "/ton" in cur_lower:
        uom_scale = 1.0 / 1000.0
        cur_uom = cur_uom.replace("/ton", "/kg").replace("/TON", "/kg").replace("/Ton", "/kg")

    cur_usd = cur_uom.replace("PKR", "USD", 1) if "PKR" in cur_uom else f"USD ({cur_uom})"
    local_payload["display_scale"] = base_scale * uom_scale
    local_payload["display_currency"] = cur_usd

    # Apply Summary-only Polyester/Viscose USD/kg override after FX conversion too
    _summary_force_usd_per_kg(local_payload)


def build_summary_commodity_payloads(*, show_local_in_usd: bool, usd_pkr_rate: float | None) -> tuple[list[dict], list[dict]]:
    all_summary: list[dict] = []
    commodity_payloads: list[dict] = []

    for int_name, local_name in SUMMARY_COMMODITY_PAIRS:
        int_payload = build_commodity_payload(int_name, INTERNATIONAL_COMMODITIES)
        local_payload = build_commodity_payload(local_name, LOCAL_COMMODITIES) if local_name else None

        _summary_force_usd_per_kg(int_payload)
        _summary_force_usd_per_kg(local_payload)

        if show_local_in_usd:
            _summary_apply_local_usd_conversion(local_payload=local_payload, usd_pkr_rate=usd_pkr_rate)

        if int_payload:
            all_summary.append(
                {
                    "name": int_payload["name"],
                    "change_1m": int_payload["pred_1m"].get("change", 0) if int_payload["pred_1m"] else 0,
                    "trend": int_payload["price_change"],
                    "market": "International",
                }
            )

        if local_payload:
            all_summary.append(
                {
                    "name": local_payload["name"],
                    "change_1m": local_payload["pred_1m"].get("change", 0) if local_payload["pred_1m"] else 0,
                    "trend": local_payload["price_change"],
                    "market": "Local",
                }
            )

        commodity_payloads.append(
            {
                "int_name": int_name,
                "local_name": local_name,
                "int_payload": int_payload,
                "local_payload": local_payload,
            }
        )

    return commodity_payloads, all_summary


def get_critical_alerts(events: dict) -> list:
    """Extract critical alerts from events."""
    alerts = []
    high_impact = ["surge", "crisis", "shortage", "ban", "war", "disruption", "spike"]
    
    for commodity, news_items in events.get("news", {}).items():
        for item in news_items:
            text = (item.get("title", "") + " " + item.get("description", "")).lower()
            if any(keyword in text for keyword in high_impact):
                alerts.append({
                    "commodity": commodity,
                    "title": item.get("title"),
                    "category": item.get("category", "general"),
                    "timestamp": item.get("timestamp")
                })
    return alerts


def main():
    """Main application with 3-page structure."""
    
    # Professional header with clean data-driven design
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%); padding: 1.5rem 2rem; border-radius: 10px; margin-bottom: 1.25rem; box-shadow: 0 4px 12px rgba(30, 64, 175, 0.2);'>
        <h1 style='font-size: 1.6rem; font-weight: 700; color: #ffffff; letter-spacing: -0.5px; margin: 0; margin-bottom: 0.35rem; line-height: 1.2;'>
            📊 Commodity Procurement Intelligence
        </h1>
        <p style='font-size: 0.85rem; color: #dbeafe; font-weight: 500; letter-spacing: 0.3px; margin: 0; line-height: 1.3;'>
            Real-time market analytics · Strategic procurement planning · Risk management
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load events for alerts
    events = load_latest_events()
    critical_alerts = get_critical_alerts(events)
    
    # Show critical alerts banner if any
    if critical_alerts:
        st.markdown(f"""
        <div style='background: #fef2f2; padding: 0.75rem 1rem; border-radius: 8px; border-left: 4px solid #dc2626; 
                    margin-bottom: 1rem; border: 1px solid #fee2e2;'>
            <p style='margin: 0; font-size: 0.825rem; font-weight: 600; color: #991b1b;'>
                ⚠️ {len(critical_alerts)} Critical Market Alerts — Review Intelligence section
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    
    # Main navigation - 4 pages with Executive Summary first
    page = st.radio(
        "Select View:",
        ["📊 Executive Summary", "🌍 International Market", "🇵🇰 Pakistan Local", "🧠 Market Intelligence", "🤖 AI Predictions"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if page == "📊 Executive Summary":
        render_executive_summary()
    
    elif page == "🌍 International Market":
        render_market_page(
            INTERNATIONAL_COMMODITIES,
            "International Market Indicators",
            "Global benchmarks for trend analysis and strategic planning"
        )
    
    elif page == "🇵🇰 Pakistan Local":
        st.markdown("""
        <div style='border-left: 4px solid #2563eb; padding-left: 1rem; margin: 1rem 0 1.25rem 0;'>
            <h2 style='font-size: 1.3rem; font-weight: 700; color: #1e293b; letter-spacing: -0.3px; margin: 0 0 0.25rem 0;'>
                🇵🇰 Pakistan Local Market Analysis
            </h2>
            <p style='font-size: 0.825rem; color: #64748b; font-weight: 500; margin: 0; line-height: 1.4;'>
                Real-time local factors and import costs affecting procurement
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show live data section with predictions
        st.markdown("### 💹 Live Market Data & Forecasts")
        
        # Create columns for live data cards
        live_cols = st.columns(len(LIVE_LOCAL_DATA))
        live_data_values = {}
        
        for idx, (name, config) in enumerate(LIVE_LOCAL_DATA.items()):
            with live_cols[idx]:
                if config['fetch_func'] == 'fetch_usd_pkr_rate':
                    data = fetch_usd_pkr_rate()
                elif config['fetch_func'] == 'fetch_wapda_electricity_rate':
                    data = fetch_wapda_electricity_rate()
                else:
                    data = None
                
                if data:
                    live_data_values[name] = data['current_price']
                    st.markdown(f"""
                    <div class='metric-card' style='border-left: 4px solid #10b981;'>
                        <div class='metric-label' style='color: #059669;'>{config['icon']} {name.upper()}</div>
                        <div class='metric-value' style='color: #059669;'>{data['current_price']:,.2f}</div>
                        <div class='currency-label' style='color: #64748b;'>{data['currency']}</div>
                        <div style='margin-top: 0.5rem; font-size: 0.7rem; color: #10b981; font-weight: 600;'>
                            🔴 LIVE · {data.get('last_update', 'Now')}
                        </div>
                        <div style='font-size: 0.65rem; color: #94a3b8; margin-top: 0.25rem;'>
                            Source: {data.get('source', 'API')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='metric-card' style='opacity: 0.6; border: 1.5px dashed #fca5a5;'>
                        <div class='metric-label'>{config['icon']} {name.upper()}</div>
                        <div class='metric-value' style='font-size: 1.2rem; color: #ef4444;'>⚠️ Unavailable</div>
                        <div class='currency-label' style='color: #94a3b8;'>API connection error</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # USD/PKR Forecast Section
        if 'USD/PKR Rate' in live_data_values:
            st.markdown("---")
            st.markdown("### 💱 USD/PKR Exchange Rate Forecast")
            st.caption("📊 Currency forecast for procurement planning · Hedge against depreciation")
            
            usd_pkr_predictions = generate_usd_pkr_forecast(live_data_values['USD/PKR Rate'])
            
            col1, col2 = st.columns([2, 1])
            with col1:
                forecast_chart = create_forecast_bar_chart(usd_pkr_predictions, 'PKR/USD')
                st.plotly_chart(forecast_chart, use_container_width=True, key="usd_pkr_forecast")
            
            with col2:
                forecast_df = create_forecast_table(usd_pkr_predictions, 'PKR/USD')
                
                # Define function to color code change values
                def color_change(val):
                    if isinstance(val, str) and '%' in val:
                        num = float(val.replace('%', '').replace('+', '').replace(' ', ''))
                        if num > 5:
                            return 'background-color: #dcfce7; color: #166534; font-weight: bold'
                        elif num > 0:
                            return 'background-color: #e0f2fe; color: #075985; font-weight: bold'
                        elif num < -5:
                            return 'background-color: #fee2e2; color: #991b1b; font-weight: bold'
                        elif num < 0:
                            return 'background-color: #fed7aa; color: #9a3412; font-weight: bold'
                    return ''
                
                # Style the dataframe professionally
                styled_df = forecast_df.style.applymap(color_change, subset=['Change']).set_properties(**{
                    'text-align': 'center',
                    'font-size': '12px',
                    'font-family': 'Arial, sans-serif',
                    'padding': '10px',
                    'border': '1px solid #e2e8f0'
                }).set_table_styles([
                    {'selector': 'thead th', 'props': [
                        ('background-color', '#1e40af'),
                        ('color', 'white'),
                        ('font-weight', 'bold'),
                        ('text-align', 'center'),
                        ('font-size', '13px'),
                        ('padding', '12px 8px'),
                        ('border', '1px solid #1e3a8a')
                    ]},
                    {'selector': 'tbody tr:nth-child(even)', 'props': [
                        ('background-color', '#f8fafc')
                    ]},
                    {'selector': 'tbody tr:nth-child(odd)', 'props': [
                        ('background-color', '#ffffff')
                    ]},
                    {'selector': 'tbody tr:hover', 'props': [
                        ('background-color', '#e0e7ff'),
                        ('transition', 'background-color 0.2s ease')
                    ]},
                    {'selector': 'tbody td', 'props': [
                        ('color', '#1e293b')
                    ]}
                ])
                
                st.dataframe(styled_df, use_container_width=True, hide_index=True, height=440)
        
        # Electricity Forecast Section
        if 'Electricity' in live_data_values:
            st.markdown("---")
            st.markdown("### ⚡ Electricity Tariff Forecast")
            st.caption("📊 Industrial tariff projections · Budget for energy costs")
            
            elec_predictions = generate_energy_forecast(live_data_values['Electricity'], 'electricity')
            
            col1, col2 = st.columns([2, 1])
            with col1:
                forecast_chart = create_forecast_bar_chart(elec_predictions, 'PKR/Unit')
                st.plotly_chart(forecast_chart, use_container_width=True, key="electricity_forecast")
            
            with col2:
                forecast_df = create_forecast_table(elec_predictions, 'PKR/Unit')
                
                # Define function to color code change values
                def color_change(val):
                    if isinstance(val, str) and '%' in val:
                        num = float(val.replace('%', '').replace('+', '').replace(' ', ''))
                        if num > 5:
                            return 'background-color: #dcfce7; color: #166534; font-weight: bold'
                        elif num > 0:
                            return 'background-color: #e0f2fe; color: #075985; font-weight: bold'
                        elif num < -5:
                            return 'background-color: #fee2e2; color: #991b1b; font-weight: bold'
                        elif num < 0:
                            return 'background-color: #fed7aa; color: #9a3412; font-weight: bold'
                    return ''
                
                # Style the dataframe professionally
                styled_df = forecast_df.style.applymap(color_change, subset=['Change']).set_properties(**{
                    'text-align': 'center',
                    'font-size': '12px',
                    'font-family': 'Arial, sans-serif',
                    'padding': '10px',
                    'border': '1px solid #e2e8f0'
                }).set_table_styles([
                    {'selector': 'thead th', 'props': [
                        ('background-color', '#1e40af'),
                        ('color', 'white'),
                        ('font-weight', 'bold'),
                        ('text-align', 'center'),
                        ('font-size', '13px'),
                        ('padding', '12px 8px'),
                        ('border', '1px solid #1e3a8a')
                    ]},
                    {'selector': 'tbody tr:nth-child(even)', 'props': [
                        ('background-color', '#f8fafc')
                    ]},
                    {'selector': 'tbody tr:nth-child(odd)', 'props': [
                        ('background-color', '#ffffff')
                    ]},
                    {'selector': 'tbody tr:hover', 'props': [
                        ('background-color', '#e0e7ff'),
                        ('transition', 'background-color 0.2s ease')
                    ]},
                    {'selector': 'tbody td', 'props': [
                        ('color', '#1e293b')
                    ]}
                ])
                
                st.dataframe(styled_df, use_container_width=True, hide_index=True, height=440)
        
        st.markdown("---")
        
        # Render available local commodity data
        if LOCAL_COMMODITIES:
            render_market_page(
                LOCAL_COMMODITIES,
                "Pakistan Local Commodities",
                "Local market prices and import costs"
            )
        
        # Show remaining pending integrations
        if PENDING_DATA:
            st.markdown("### 📡 Upcoming Data Integrations")
            cols = st.columns(len(PENDING_DATA))
            for idx, (name, info) in enumerate(PENDING_DATA.items()):
                with cols[idx]:
                    render_pending_data_card(name, info)
    
    elif page == "🧠 Market Intelligence":
        render_intelligence_page(events)

    elif page == "🤖 AI Predictions":
        render_ai_predictions_page()


def render_executive_summary():
    """Executive Summary - Commodity-by-commodity comparison (International vs Local)."""
    # Team request: show Local column in USD (USDT-equivalent) on Summary page.
    show_local_in_usd = True
    local_ccy_label = "USD" if show_local_in_usd else "PKR"

    st.markdown(f"""
    <div style='border-left: 4px solid #2563eb; padding-left: 1rem; margin: 1rem 0 1.25rem 0;'>
        <h2 style='font-size: 1.5rem; font-weight: 800; color: #0f172a; letter-spacing: -0.4px; margin: 0 0 0.35rem 0;'>
            📊 Executive Summary
        </h2>
        <p style='font-size: 0.95rem; color: #475569; font-weight: 600; margin: 0; line-height: 1.5;'>
            Commodity-by-commodity comparison: International (USD) vs Pakistan Local ({local_ccy_label})
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Define commodity pairs (International and Local versions)
    commodity_pairs = [
        ("Cotton", "Cotton (Local)"),
        ("Polyester", "Polyester (Local)"),
        ("Viscose", "Viscose (Local)"),
        ("Natural Gas", "Natural Gas"),
        ("Crude Oil", "Crude Oil")
    ]

    def _get_usd_pkr_rate_for_summary() -> tuple[float | None, str]:
        if not show_local_in_usd:
            return None, "disabled"

        # 1) Prefer live FX
        live = fetch_usd_pkr_rate()
        try:
            if live and live.get("current_price"):
                rate = float(live["current_price"])
                if rate > 0:
                    return rate, "live"
        except Exception:
            pass

        # 2) Fallback: infer FX from commodities that have both USD and PKR series
        def implied_fx(usd_asset: str, pkr_asset: str, usd_ccy: str, pkr_ccy: str) -> float | None:
            usd_md = load_commodity_data(usd_asset, usd_ccy)
            pkr_md = load_commodity_data(pkr_asset, pkr_ccy)
            try:
                usd_val = float(usd_md.get("current_price", 0.0)) if usd_md else 0.0
                pkr_val = float(pkr_md.get("current_price", 0.0)) if pkr_md else 0.0
                if usd_val > 0 and pkr_val > 0:
                    return pkr_val / usd_val
            except Exception:
                return None
            return None

        fx = implied_fx(
            INTERNATIONAL_COMMODITIES["Crude Oil"]["path"],
            LOCAL_COMMODITIES["Crude Oil"]["path"],
            INTERNATIONAL_COMMODITIES["Crude Oil"]["currency"],
            LOCAL_COMMODITIES["Crude Oil"]["currency"],
        )
        if fx and fx > 0:
            return float(fx), "implied_from_crude"

        fx = implied_fx(
            INTERNATIONAL_COMMODITIES["Natural Gas"]["path"],
            LOCAL_COMMODITIES["Natural Gas"]["path"],
            INTERNATIONAL_COMMODITIES["Natural Gas"]["currency"],
            LOCAL_COMMODITIES["Natural Gas"]["currency"],
        )
        if fx and fx > 0:
            return float(fx), "implied_from_gas"

        # 3) Last resort default (keeps Summary usable offline)
        return 280.0, "default"

    usd_pkr_rate, usd_pkr_source = _get_usd_pkr_rate_for_summary()
    if show_local_in_usd and usd_pkr_rate and usd_pkr_rate > 0:
        label = {
            "live": "live",
            "implied_from_crude": "derived from Crude Oil USD vs PKR",
            "implied_from_gas": "derived from Natural Gas USD vs PKR",
            "default": "default",
        }.get(usd_pkr_source, usd_pkr_source)
        st.caption(f"Local column shown in USD using USD/PKR = {usd_pkr_rate:,.2f} ({label}).")
    
    # Helper function to render commodity chart and table
    def render_empty_card(message: str):
        st.markdown(f"""
        <div style='background: #f8fafc; padding: 2rem; border-radius: 8px; text-align: center; border: 2px dashed #cbd5e1;'>
            <p style='font-size: 0.9rem; font-weight: 700; color: #64748b; margin: 0;'>
                {message}
            </p>
        </div>
        """, unsafe_allow_html=True)

    def build_commodity_payload(commodity_name, commodities_dict):
        if not commodity_name or commodity_name not in commodities_dict:
            return None

        info = commodities_dict[commodity_name]
        metadata = load_commodity_data(info["path"], info["currency"])
        predictions = load_predictions(info["path"])

        if metadata['df'] is None:
            return None

        current_price = metadata['df'][metadata['value_col']].iloc[-1]
        price_change = metadata['price_change']
        pred_1m = get_prediction_by_index(predictions, 1)
        pred_3m = get_prediction_by_index(predictions, 3)

        return {
            "name": commodity_name,
            "info": info,
            "current_price": current_price,
            "price_change": price_change,
            "pred_1m": pred_1m,
            "pred_3m": pred_3m,
            "predictions": predictions,
            "history_df": metadata.get("df"),
            "value_col": metadata.get("value_col"),
            # Display overrides (used by Summary page only)
            "display_scale": 1.0,
            "display_currency": info.get("currency")
        }

    def render_commodity_chart(payload, market_type: str):
        if payload is None:
            render_empty_card("No data available")
            return

        if market_type == "International":
            header_color = "#1e40af"
            chart_colors = ['#64748b', '#3b82f6', '#2563eb']
        else:
            header_color = "#047857"
            chart_colors = ['#64748b', '#10b981', '#059669']

        st.markdown(f"""
        <h4 style='font-size: 1rem; font-weight: 800; color: {header_color}; margin: 1rem 0 0.5rem 0;'>
            📊 Price Forecast
        </h4>
        """, unsafe_allow_html=True)

        scale = float(payload.get("display_scale", 1.0) or 1.0)
        display_currency = payload.get("display_currency") or payload["info"]["currency"]
        # Summary-only formatting rules:
        # - Cotton, Polyester, Viscose, Crude Oil: 3 decimals
        # - Others: keep existing behavior
        name_lower = str(payload.get("name", "")).lower()
        three_dec_assets = ("cotton", "polyester", "viscose", "crude", "natural gas")
        is_three_dec = any(k in name_lower for k in three_dec_assets)
        dec = 3 if is_three_dec else (3 if "/lb" in str(display_currency).lower() else 1)

        horizons = get_prediction_horizons(payload.get("predictions", {}))
        prices = []
        periods = []
        if payload.get("predictions"):
            for h in horizons:
                pred = payload["predictions"].get(h)
                if pred:
                    periods.append(h)
                    prices.append(float(pred["price"]) * scale)
        else:
            periods = horizons
            prices = [float(payload["current_price"]) * scale for _ in horizons]

        chart_data = pd.DataFrame({
            'Period': periods,
            'Price': prices
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=chart_data['Period'],
            y=chart_data['Price'],
            marker_color=chart_colors,
            text=chart_data['Price'],
            texttemplate=f'%{{text:,.{dec}f}}',
            textposition='outside',
            textfont=dict(size=9, family='IBM Plex Mono', weight=700, color='#1e293b'),
            showlegend=False
        ))

        fig.update_layout(
            height=320,
            margin=dict(l=50, r=20, t=10, b=90),
            plot_bgcolor='#fafafa',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                tickfont=dict(size=9, family='Inter', weight=700, color='#1e293b'),
                showgrid=False,
                showline=True,
                linewidth=2,
                linecolor='#cbd5e1',
                tickangle=-35
            ),
            yaxis=dict(
                title=dict(text=f'<b>{display_currency}</b>', font=dict(size=11, family='Inter', weight=800, color=header_color)),
                tickfont=dict(size=10, family='IBM Plex Mono', weight=600, color='#334155'),
                showgrid=True,
                gridcolor='#e2e8f0',
                showline=True,
                linewidth=2,
                linecolor='#cbd5e1'
            )
        )
        fig.update_traces(cliponaxis=False)
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

        st.plotly_chart(fig, use_container_width=True, key=f"chart_{market_type}_{payload['name'].replace(' ', '_')}")

    def render_commodity_table(payload, market_type: str):
        if payload is None:
            render_empty_card("No data available")
            return

        if market_type == "International":
            header_color = "#1e40af"
        else:
            header_color = "#047857"

        st.markdown(f"""
        <h4 style='font-size: 1rem; font-weight: 800; color: {header_color}; margin: 1rem 0 0.5rem 0;'>
            📋 Forecast Data
        </h4>
        """, unsafe_allow_html=True)

        scale = float(payload.get("display_scale", 1.0) or 1.0)
        display_currency = payload.get("display_currency") or payload["info"]["currency"]
        # Summary-only formatting rules:
        # - Cotton, Polyester, Viscose, Crude Oil: 3 decimals
        # - Others: keep existing behavior
        name_lower = str(payload.get("name", "")).lower()
        three_dec_assets = ("cotton", "polyester", "viscose", "crude", "natural gas")
        is_three_dec = any(k in name_lower for k in three_dec_assets)
        dec = 3 if is_three_dec else (3 if "/lb" in str(display_currency).lower() else 2)

        table_rows = []
        if payload.get("predictions"):
            for h in get_prediction_horizons(payload["predictions"]):
                pred = payload["predictions"].get(h)
                if pred:
                    table_rows.append({
                        "Period": h,
                        "Value": f"{(float(pred.get('price', 0)) * scale):,.{dec}f}",
                        "Change": f"{pred.get('change', 0):+.1f}%"
                    })
        else:
            for h in get_prediction_horizons({}):
                table_rows.append({
                    "Period": h,
                    "Value": f"{(float(payload['current_price']) * scale):,.{dec}f}",
                    "Change": f"{'↑' if payload['price_change'] > 0 else '↓'} {abs(payload['price_change']):.1f}%"
                })

        table_data = pd.DataFrame(table_rows)

        def color_change(val):
            if isinstance(val, str) and '%' in val:
                try:
                    if '↑' in val or '+' in val:
                        return 'background-color: #fee2e2; color: #991b1b; font-weight: 700'
                    elif '↓' in val or '-' in val:
                        return 'background-color: #dcfce7; color: #166534; font-weight: 700'
                except:
                    pass
            return ''

        styled_table = table_data.style.map(color_change, subset=['Change'])\
                                       .set_properties(**{
                                           'text-align': 'left',
                                           'font-size': '0.85rem',
                                           'font-weight': '700',
                                           'padding': '8px 12px',
                                           'border': '1px solid #cbd5e1'
                                       }).set_table_styles([
                                           {'selector': 'thead th', 'props': [
                                               ('background-color', header_color),
                                               ('color', 'white'),
                                               ('font-weight', '800'),
                                               ('padding', '10px 12px'),
                                               ('font-size', '0.75rem'),
                                               ('text-transform', 'uppercase')
                                           ]},
                                           {'selector': 'tbody tr:hover', 'props': [
                                               ('background-color', '#f0fdf4' if market_type == "Local" else '#eff6ff')
                                           ]}
                                       ])

        st.dataframe(styled_table, use_container_width=True, height=360)
    
    # Render each commodity pair
    all_summary = []
    commodity_payloads = []
    
    for int_name, local_name in commodity_pairs:
        int_payload = build_commodity_payload(int_name, INTERNATIONAL_COMMODITIES)
        local_payload = build_commodity_payload(local_name, LOCAL_COMMODITIES) if local_name else None

        # Summary-only unit override: Polyester & Viscose should display as USD/kg (not USD/lb).
        # We do this via display_currency/display_scale so other pages are unaffected.
        def _summary_force_usd_per_kg(p: dict | None) -> None:
            if not p:
                return
            nm = str(p.get("name", "")).lower()
            if not ("polyester" in nm or "viscose" in nm):
                return

            cur = str(p.get("display_currency") or p.get("info", {}).get("currency", ""))
            cur_lower = cur.lower()
            scale0 = float(p.get("display_scale", 1.0) or 1.0)
            uom_scale = 1.0

            # If current unit is /lb, convert to /kg
            if "/lb" in cur_lower:
                uom_scale *= 2.20462262185  # USD/lb -> USD/kg
                cur = cur.replace("/lb", "/kg").replace("/LB", "/kg").replace("/Lb", "/kg")

            # If current unit is /ton, convert to /kg
            if "/ton" in cur_lower:
                uom_scale *= 1.0 / 1000.0
                cur = cur.replace("/ton", "/kg").replace("/TON", "/kg").replace("/Ton", "/kg")

            # Ensure USD prefix (for Summary consistency)
            if "USD" not in cur:
                cur = cur.replace("PKR", "USD", 1) if "PKR" in cur else f"USD ({cur})"

            p["display_scale"] = scale0 * uom_scale
            p["display_currency"] = cur

        _summary_force_usd_per_kg(int_payload)
        _summary_force_usd_per_kg(local_payload)

        # Convert Local column display to USD (USDT-equivalent) for Summary page
        if show_local_in_usd and local_payload and usd_pkr_rate and usd_pkr_rate > 0:
            base_scale = 1.0 / float(usd_pkr_rate)
            cur = str(local_payload["info"].get("currency", ""))

            # Keep UOM consistent and market-practice.
            # - Cotton: Local is PKR/maund, convert to USD/lb (1 maund = 40kg = 88.1849 lb).
            # - If any legacy local series still uses /ton while the CEO wants /kg, convert /ton -> /kg (÷1000).
            uom_scale = 1.0
            cur_uom = cur
            cur_lower = cur.lower()

            # Cotton special-case: maund -> lb
            if "cotton" in str(local_payload.get("name", "")).lower() and "/maund" in cur_lower:
                maund_lb = 40.0 * 2.20462262185
                uom_scale = 1.0 / maund_lb
                cur_uom = cur_uom.replace("/maund", "/lb").replace("/MAUND", "/lb").replace("/Maund", "/lb")
            elif ("polyester" in str(local_payload.get("name", "")).lower() or "viscose" in str(local_payload.get("name", "")).lower()) and "/ton" in cur_lower:
                uom_scale = 1.0 / 1000.0
                cur_uom = cur_uom.replace("/ton", "/kg").replace("/TON", "/kg").replace("/Ton", "/kg")

            # Preserve unit after '/', only replace the currency prefix.
            cur_usd = cur_uom.replace("PKR", "USD", 1) if "PKR" in cur_uom else f"USD ({cur_uom})"

            local_payload["display_scale"] = base_scale * uom_scale
            local_payload["display_currency"] = cur_usd

            # Apply Summary-only Polyester/Viscose USD/kg override after FX conversion too
            _summary_force_usd_per_kg(local_payload)

        if int_payload:
            all_summary.append({
                "name": int_payload["name"],
                "change_1m": int_payload["pred_1m"].get('change', 0) if int_payload["pred_1m"] else 0,
                "trend": int_payload["price_change"],
                "market": "International"
            })

        if local_payload:
            all_summary.append({
                "name": local_payload["name"],
                "change_1m": local_payload["pred_1m"].get('change', 0) if local_payload["pred_1m"] else 0,
                "trend": local_payload["price_change"],
                "market": "Local"
            })

        commodity_payloads.append({
            "int_name": int_name,
            "local_name": local_name,
            "int_payload": int_payload,
            "local_payload": local_payload
        })

    # === CHARTS FIRST (for all commodities) ===
    for item in commodity_payloads:
        display_name = item["int_name"]
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #334155 0%, #475569 100%); 
                    padding: 0.75rem 1.25rem; 
                    border-radius: 8px; 
                    margin: 1.5rem 0 1rem 0;'>
            <h3 style='color: white; font-size: 1.2rem; font-weight: 800; margin: 0; text-align: center; letter-spacing: 0.5px;'>
                {INTERNATIONAL_COMMODITIES.get(item["int_name"], {}).get('icon', '📦')} {display_name.upper()}
            </h3>
        </div>
        """, unsafe_allow_html=True)

        col_int, col_local = st.columns(2)

        with col_int:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%); 
                        padding: 0.75rem; 
                        border-radius: 6px; 
                        margin-bottom: 1rem;
                        text-align: center;'>
                <p style='color: white; font-size: 0.95rem; font-weight: 800; margin: 0;'>
                    🌍 INTERNATIONAL (USD)
                </p>
            </div>
            """, unsafe_allow_html=True)

            render_commodity_chart(item["int_payload"], "International")

        with col_local:
            if item["local_name"]:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #047857 0%, #059669 100%); 
                            padding: 0.75rem; 
                            border-radius: 6px; 
                            margin-bottom: 1rem;
                            text-align: center;'>
                    <p style='color: white; font-size: 0.95rem; font-weight: 800; margin: 0;'>
                        🇵🇰 PAKISTAN LOCAL (USD)
                    </p>
                </div>
                """, unsafe_allow_html=True)

                render_commodity_chart(item["local_payload"], "Local")
            else:
                render_empty_card("No Local Market Data")

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # Replace forecast tables with Pakistan-market forecasts (requested by team lead)
    st.markdown("---")
    st.markdown(
        """
        <div style='background: #f1f5f9; padding: 0.75rem 1rem; border-radius: 8px; margin: 1.25rem 0 0.75rem 0;'>
            <h3 style='font-size: 1.1rem; font-weight: 800; color: #0f172a; margin: 0; text-align: center;'>
                🇵🇰 Pakistan Market Forecasts
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    usd = fetch_usd_pkr_rate()
    if usd and usd.get("current_price"):
        usd_preds = generate_usd_pkr_forecast(float(usd["current_price"]))
        _render_pakistan_forecast_chart_table(
            title="💱 USD/PKR Exchange Rate Forecast",
            caption="Pakistani market indicator for procurement planning",
            predictions=usd_preds,
            currency="PKR/USD",
            key_prefix="summary_usd_pkr",
        )
    else:
        st.info("USD/PKR live rate unavailable right now.")

    st.markdown("---")
    elec = fetch_wapda_electricity_rate()
    if elec and elec.get("current_price"):
        elec_preds = generate_energy_forecast(float(elec["current_price"]), "electricity")
        _render_pakistan_forecast_chart_table(
            title="⚡ Electricity Tariff Forecast",
            caption="Pakistani market indicator (industrial tariff) for budgeting energy cost",
            predictions=elec_preds,
            currency="PKR/Unit",
            key_prefix="summary_electricity",
        )
    else:
        st.info("Electricity tariff data unavailable right now.")

    # No‑arbitrage strategist (futures mispricing + put‑call parity)
    st.markdown("---")
    render_no_arbitrage_strategist(
        expander_title="🧮 No‑Arbitrage Strategist (Futures + Call/Put Parity)",
        expanded=False,
        key_prefix="summary_arb",
        commodity_payloads=commodity_payloads,
    )
    
    # === OVERALL SUMMARY - Key insights across all commodities ===
    if len(all_summary) > 0:
        st.markdown("---")
        st.markdown("""
        <h3 style='font-size: 1.25rem; font-weight: 800; color: #0f172a; margin: 1.5rem 0 1rem 0; text-align: center;'>
            📈 Overall Market Summary
        </h3>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <h4 style='font-size: 1.05rem; font-weight: 800; color: #dc2626; margin: 0.5rem 0;'>
                🔴 Price Alerts
            </h4>
            """, unsafe_allow_html=True)
            
            risks = [item for item in all_summary if item['change_1m'] > 5]
            opportunities = [item for item in all_summary if item['change_1m'] < -5]
            
            if risks:
                st.markdown("""
                <div style='background: #fef2f2; padding: 1rem; border-radius: 8px; border-left: 4px solid #dc2626; margin: 1rem 0;'>
                    <p style='margin: 0 0 0.5rem 0; font-size: 0.85rem; font-weight: 800; color: #991b1b;'>RISING >5%</p>
                </div>
                """, unsafe_allow_html=True)
                for r in risks[:5]:
                    st.markdown(f"<p style='font-size: 0.9rem; font-weight: 700; color: #1e293b; margin: 0.4rem 0;'>• {r['name']} ({r['market']}): +{r['change_1m']:.1f}%</p>", unsafe_allow_html=True)
            else:
                st.success("✓ No major price increases")
            
            if opportunities:
                st.markdown("""
                <div style='background: #f0fdf4; padding: 1rem; border-radius: 8px; border-left: 4px solid #059669; margin: 1rem 0;'>
                    <p style='margin: 0 0 0.5rem 0; font-size: 0.85rem; font-weight: 800; color: #166534;'>FALLING >5%</p>
                </div>
                """, unsafe_allow_html=True)
                for o in opportunities[:5]:
                    st.markdown(f"<p style='font-size: 0.9rem; font-weight: 700; color: #1e293b; margin: 0.4rem 0;'>• {o['name']} ({o['market']}): {o['change_1m']:.1f}%</p>", unsafe_allow_html=True)
            else:
                st.info("→ Prices stable")
        
        with col2:
            st.markdown("""
            <h4 style='font-size: 1.05rem; font-weight: 800; color: #2563eb; margin: 0.5rem 0;'>
                📊 Quick Stats
            </h4>
            """, unsafe_allow_html=True)
            
            total = len(all_summary)
            rising = len([item for item in all_summary if item['trend'] > 0])
            falling = len([item for item in all_summary if item['trend'] < 0])
            
            st.markdown(f"""
            <div style='background: white; padding: 1.25rem; border-radius: 8px; border: 2px solid #cbd5e1;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 0.75rem; padding-bottom: 0.75rem; border-bottom: 1px solid #e2e8f0;'>
                    <span style='font-size: 0.95rem; font-weight: 800; color: #334155;'>Total Tracked:</span>
                    <span style='font-size: 1.1rem; font-weight: 800; color: #1e293b;'>{total}</span>
                </div>
                <div style='display: flex; justify-content: space-between; margin-bottom: 0.75rem;'>
                    <span style='font-size: 0.95rem; font-weight: 800; color: #059669;'>↑ Rising:</span>
                    <span style='font-size: 1.1rem; font-weight: 800; color: #059669;'>{rising}</span>
                </div>
                <div style='display: flex; justify-content: space-between;'>
                    <span style='font-size: 0.95rem; font-weight: 800; color: #dc2626;'>↓ Falling:</span>
                    <span style='font-size: 1.1rem; font-weight: 800; color: #dc2626;'>{falling}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(f"""
    <p style='font-size: 0.85rem; color: #64748b; font-weight: 600; text-align: center; margin: 1rem 0;'>
        💾 Use individual commodity tabs for detailed analysis • Last updated: {datetime.now().strftime("%B %d, %Y %H:%M")}
    </p>
    """, unsafe_allow_html=True)


def get_critical_alerts(events: dict) -> list:
    """Extract critical alerts from events."""
    alerts = []
    high_impact = ["surge", "crisis", "shortage", "ban", "war", "disruption", "spike"]
    
    for commodity, news_items in events.get("news", {}).items():
        for item in news_items:
            text = (item.get("title", "") + " " + item.get("description", "")).lower()
            if any(keyword in text for keyword in high_impact):
                alerts.append(
                    {
                        "commodity": commodity,
                        "title": item.get("title"),
                        "category": item.get("category", "general"),
                        "timestamp": item.get("timestamp"),
                    }
                )

    return alerts


def render_intelligence_page(events: dict):
    """Render market intelligence and events page."""
    st.markdown("""
    <h2 style='font-size: 1.875rem; font-weight: 700; color: #1e293b; letter-spacing: -0.5px; margin-bottom: 0.75rem;'>
        🧠 Market Intelligence & Events
    </h2>
    <p style='font-size: 0.95rem; color: #64748b; font-weight: 500; letter-spacing: 0.2px; margin-bottom: 1.5rem;'>
        Latest news, weather impacts, and geopolitical factors
    </p>
    """, unsafe_allow_html=True)
    st.caption("Real-time news, geopolitics, and events affecting commodity prices")
    
    # Update button
    _, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            try:
                scripts_path = Path(__file__).parent / "scripts" / "event_collector.py"
                spec = importlib.util.spec_from_file_location("event_collector", scripts_path)
                if spec and spec.loader:
                    event_collector = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(event_collector)
                    event_collector.collect_all_events()
                else:
                    raise ImportError("Unable to load event_collector")
                st.success("✅ Updated!")
                st.rerun()
            except Exception as e:
                st.warning(f"Run scripts/event_collector.py to collect events ({e})")
    
    # Show collection time
    collection_time = events.get("collection_time", "Not collected yet")
    st.caption(f"📅 Last updated: {collection_time}")
    
    st.markdown("---")
    
    # Critical Alerts Section
    alerts = get_critical_alerts(events)
    if alerts:
        st.markdown("### 🚨 Critical Alerts")
        for alert in alerts[:5]:
            st.error(f"**[{alert['commodity'].upper()}]** {alert['title']}")
    
    # Tabs for different intelligence types
    tabs = st.tabs(["📰 News by Commodity", "🌍 Geopolitical", "🌦️ Weather & Supply"])
    
    with tabs[0]:
        st.markdown("#### Commodity-Specific News")
        
        news_data = events.get("news", {})
        if not news_data or all(len(items) == 0 for items in news_data.values()):
            st.info("No events collected. Run `python scripts/event_collector.py` to collect market intelligence.")
        else:
            for commodity, items in news_data.items():
                if items:
                    with st.expander(f"📌 {commodity.upper()} News ({len(items)} items)", expanded=False):
                        for item in items[:5]:
                            category_emoji = {
                                "geopolitics": "🌍",
                                "weather": "🌦️",
                                "policy": "📜",
                                "supply": "📦",
                                "demand": "📈"
                            }.get(item.get("category"), "📰")
                            
                            st.markdown(f"""
                            **{category_emoji} {item.get('title', 'N/A')}**
                            
                            {item.get('description', '')}
                            
                            *Source: {item.get('source', 'Unknown')} • {item.get('timestamp', '')[:10]}*
                            
                            [Read more]({item.get('url', '#')})
                            
                            ---
                            """)
    
    with tabs[1]:
        st.markdown("#### Geopolitical Events")
        geo_events = events.get("geopolitical", [])
        
        if not geo_events:
            st.info("No geopolitical events tracked yet.")
        else:
            for event in geo_events:
                affected = ", ".join(event.get("affected_commodities", []))
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>🌍 {event.get('region', 'Unknown')}</div>
                    <div style='margin-top: 0.5rem;'>
                        <strong>Affected:</strong> {affected}
                    </div>
                    <div style='margin-top: 0.5rem; font-size: 0.85rem; color: #64748b;'>
                        Status: {event.get('status', 'monitoring')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("#### Weather & Supply Chain")
        weather_alerts = events.get("weather", [])
        
        if not weather_alerts:
            st.info("No weather alerts.")
        else:
            for alert in weather_alerts:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>🌦️ {alert.get('region', 'Unknown')}</div>
                    <div style='margin-top: 0.5rem;'>
                        <strong>Crop:</strong> {alert.get('crop', 'N/A')}
                    </div>
                    <div style='margin-top: 0.5rem;'>
                        {alert.get('alert', 'No details')}
                    </div>
                    <div style='margin-top: 0.5rem; font-size: 0.85rem;'>
                        Severity: {alert.get('severity', 'info')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("Options Hedge Advisor: Portfolio View is on 📊 Executive Summary; full Strategist view is on 🤖 AI Predictions.")


if __name__ == "__main__":
    main()
