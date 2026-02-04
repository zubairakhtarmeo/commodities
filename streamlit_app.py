from __future__ import annotations
# Last updated: 2026-02-04 - Enhanced with profit calculations and specific trade recommendations

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


@st.cache_data(ttl=600)
def _load_purchase_monthly_agg() -> pd.DataFrame:
    """Load cleaned Oracle purchase monthly aggregates (local file, optional).

    This file is typically generated by `scripts/ingest_oracle_purchases.py` and
    is often gitignored (contains internal purchasing history). The app should
    degrade gracefully if it doesn't exist (e.g., Streamlit Cloud).
    """

    try:
        p = Path("data") / "processed" / "purchases_clean" / "purchases_monthly_agg.csv"
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_csv(p)
        if df.empty:
            return df
        if "month" in df.columns:
            df["month"] = pd.to_datetime(df["month"], errors="coerce")
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
        return "IMMEDIATE"
    if sigma_ann >= 0.25 or score >= 3.0:
        return "NEAR‑TERM (2–4 WEEKS)"
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
    """Procurement + options strategist: recommends execution schedule + CALL/PUT structures using history + forecast."""
    variant = str(variant or "full").strip().lower()
    container = st.expander(expander_title, expanded=expanded) if use_expander else nullcontext()
    with container:
        exposure = "Procurement (forward buying)"
        risk_profile = "Balanced"
        budget_priority = "Moderate"
        allow_selling = False
        qty = 1.0

        # Internal default (kept conservative; no UI knob)
        forecast_sig = 3.0

        def _purchase_commodity_from_label(lbl: str) -> str | None:
            s = str(lbl or "").lower()
            if "cotton" in s:
                return "Cotton"
            if "polyester" in s:
                return "Polyester"
            if "viscose" in s:
                return "Viscose"
            if "crude" in s:
                return "Crude Oil"
            if "natural gas" in s or "nat gas" in s:
                return "Natural Gas"
            return None

        def _fmt_qty_kg(qty_kg: float) -> str:
            try:
                q = float(qty_kg)
                if not (np.isfinite(q) and q > 0):
                    return "—"
            except Exception:
                return "—"
            if q >= 10000:
                return f"{q/1000.0:,.0f} t"
            if q >= 1000:
                return f"{q/1000.0:,.1f} t"
            return f"{q:,.0f} kg"

        # Purchase history sizing (optional; local file may not exist on Streamlit Cloud)
        purchase_ctx: dict[str, dict] = {}
        try:
            purch_df = _load_purchase_monthly_agg()
            if not purch_df.empty and "month" in purch_df.columns:
                max_m = pd.to_datetime(purch_df["month"], errors="coerce").max()
                if pd.notna(max_m):
                    cutoff = pd.Timestamp(max_m) - pd.DateOffset(months=12)
                    w = purch_df[pd.to_datetime(purch_df["month"], errors="coerce") >= cutoff].copy()
                    for comm, g in w.groupby("commodity", dropna=True):
                        if "total_qty_kg" not in g.columns:
                            continue
                        q = pd.to_numeric(g["total_qty_kg"], errors="coerce")
                        if q.dropna().empty:
                            continue
                        median_monthly_kg = float(np.nanmedian(q.values)) if np.isfinite(np.nanmedian(q.values)) else 0.0
                        mean_monthly_kg = float(np.nanmean(q.values)) if np.isfinite(np.nanmean(q.values)) else 0.0
                        std_monthly_kg = float(np.nanstd(q.values)) if np.isfinite(np.nanstd(q.values)) else 0.0
                        cv = float(std_monthly_kg / mean_monthly_kg) if mean_monthly_kg > 0 else 0.0
                        purchase_ctx[str(comm)] = {"median_monthly_kg": median_monthly_kg, "cv": cv}
        except Exception:
            purchase_ctx = {}

        if variant != "portfolio":
            st.markdown(
                """
<div class="cp-card">
    <div style="display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap;">
        <div>
            <div class="cp-title">Procurement & Options Strategist</div>
            <div class="cp-subtitle">Unified guidance: <b>procurement execution</b> + <b>CALL/PUT hedge structures</b> using history and forecast distribution.</div>
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
                ["Procurement (forward buying)", "Inventory (stock on hand)", "Sales (forward selling)"],
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
                        ["Low", "Moderate", "High"],
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

            # Procurement execution (only meaningful for procurement exposure; otherwise keep neutral)
            proc_decision = "Maintain standard cadence"
            proc_window = "Review monthly"
            proc_rationale = "No strong forecast edge vs Current Spot Market Price"
            proc_schedule = "Continue staggered procurement; reassess as new data arrives."

            try:
                curve = _extract_forecast_curve_from_payload(payload=payload, max_months=24)
                if exposure.startswith("Procurement") and curve and s0 > 0:
                    curve_s: list[dict] = []
                    for e in curve:
                        e2 = dict(e)
                        e2["price"] = float(e2["price"]) * scale
                        if e2.get("lower") is not None:
                            e2["lower"] = float(e2["lower"]) * scale
                        if e2.get("upper") is not None:
                            e2["upper"] = float(e2["upper"]) * scale
                        curve_s.append(e2)

                    min_e = min(curve_s, key=lambda e: float(e.get("price", 1e18)))
                    max_e = max(curve_s, key=lambda e: float(e.get("price", -1e18)))
                    mid_e = _closest_curve_entry(curve_s, 6) or min_e

                    move_to_min = (float(min_e["price"]) / s0 - 1.0) * 100.0
                    move_to_max = (float(max_e["price"]) / s0 - 1.0) * 100.0

                    if move_to_min <= -float(forecast_sig):
                        proc_decision = "Defer procurement; phased procurement allocation"
                        proc_window = f"Forward Maturity Target: {min_e.get('horizon','next window')} (~{min_e.get('months', '—')}M)"
                        proc_rationale = f"Downward Price Expectation ({move_to_min:.1f}% vs Current Spot Market Price)"
                        half_months = max(1, int(round(float(min_e.get("months", 6)) / 2.0)))
                        half_e = _closest_curve_entry(curve_s, half_months) or mid_e
                        proc_schedule = (
                            f"Phased Procurement Allocation: Minimum Coverage Procurement 10% immediately; 30% around {half_e.get('horizon','mid‑window')}; "
                            f"60% around {min_e.get('horizon','target window')}. Options Overlay Strategy: Limited Upside Call Option Exposure (or Bull Call Spread Strategy) to Preserve Upside Participation while unhedged."
                        )
                    elif move_to_max >= float(forecast_sig):
                        proc_decision = "Accelerate procurement; increase near-term coverage"
                        proc_window = "Initiate coverage now"
                        proc_rationale = f"Upward Price Expectation (+{move_to_max:.1f}% vs Current Spot Market Price)"
                        early_e = _closest_curve_entry(curve_s, 3) or mid_e
                        proc_schedule = (
                            f"Phased Procurement Allocation: 40% immediately; 30% around {early_e.get('horizon','next window')}; "
                            f"30% Price Fixation via Forward Contract into {max_e.get('horizon','later window')} (or Bull Call Spread Strategy)."
                        )

                    # Optional sizing overlay from purchase history
                    pc = _purchase_commodity_from_label(c["label"])
                    if pc and pc in purchase_ctx:
                        mqty = float(purchase_ctx[pc].get("median_monthly_kg") or 0.0)
                        if np.isfinite(mqty) and mqty > 0:
                            cover_m = float(min(6, int(max(2, months))))
                            total_need = mqty * cover_m
                            if np.isfinite(total_need) and total_need > 0:
                                proc_schedule = f"{proc_schedule}  Sizing: ~{_fmt_qty_kg(mqty)}/month; cover ~{_fmt_qty_kg(total_need)} ({cover_m:.0f}M)."
            except Exception:
                pass

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
                    "proc_decision": proc_decision,
                    "proc_window": proc_window,
                    "proc_rationale": proc_rationale,
                    "proc_schedule": proc_schedule,
                }
            )

        # Rank: urgency first, then by score
        when_rank = {"IMMEDIATE": 0, "NEAR‑TERM (2–4 WEEKS)": 1, "MONITOR": 2}
        recs = sorted(recs, key=lambda r: (when_rank.get(str(r.get("when")), 9), -float(r.get("score", 0.0))))

        if variant == "portfolio":
            st.markdown(
                """
<div class="cp-card" style="padding: 1.0rem 1.0rem;">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px; flex-wrap:wrap;">
    <div>
    <div class="cp-title">Procurement & Options — Portfolio Dashboard</div>
    <div class="cp-subtitle">Ranked guidance across commodities. Includes <b>CALL</b> structures (cap procurement cost) and <b>PUT</b> structures (protect selling price / downside).</div>
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
                if "IMMEDIATE" in v:
                    return "background-color: #7f1d1d; color: #ffffff; font-weight: 800;"
                if "NEAR‑TERM" in v:
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
                    exposure="Procurement (forward buying)",
                    s0=s0,
                    s_mean=s_mean,
                    sigma_ann=sigma_ann,
                    t_years=t_years,
                    risk_profile="Balanced",
                    budget_priority="Moderate",
                    allow_selling=False,
                    qty=1.0,
                    unit=unit,
                )
                put_strat = _recommend_hedge_strategy(
                    exposure="Sales (forward selling)",
                    s0=s0,
                    s_mean=s_mean,
                    sigma_ann=sigma_ann,
                    t_years=t_years,
                    risk_profile="Balanced",
                    budget_priority="Moderate",
                    allow_selling=False,
                    qty=1.0,
                    unit=unit,
                )

                # Compact display - extract only key info
                call_title_short = call_strat.get('title', '—').replace('BUY ', '').replace('SELL ', '')[:20]
                put_title_short = put_strat.get('title', '—').replace('BUY ', '').replace('SELL ', '')[:20]
                proc_short = str(r.get('proc_decision','—')).replace('procurement', 'proc').replace('Phased Procurement Allocation', 'Phased')[:30]
                
                # Urgency indicator
                urgency_raw = r.get("when")
                if "IMMEDIATE" in str(urgency_raw):
                    urgency_icon = "🔴"
                elif "NEAR" in str(urgency_raw):
                    urgency_icon = "🟡"
                else:
                    urgency_icon = "🟢"
                
                rows.append(
                    {
                        "⚡": urgency_icon,
                        "Commodity": r.get("label"),
                        "Horizon": r.get("horizon"),
                        "Spot": f"{s0:,.{dec}f}",
                        "Fcst": f"{s_mean:,.{dec}f}",
                        "Δ%": round(exp_ret, 1),
                        "Vol%": round(sigma_ann * 100.0, 0),
                        "Score": round(float(r.get("score", 0.0)), 1),
                        "🛒 Procurement": proc_short,
                        "📞 CALL Hedge": call_title_short,
                        "📉 PUT Hedge": put_title_short,
                        "Unit": unit,
                    }
                )

            dfp = pd.DataFrame(rows)

            def _urgency_style(val) -> str:
                if val == "🔴":
                    return "background-color: #dc2626; color: white; font-weight: 900; text-align: center;"
                if val == "�":
                    return "background-color: #3b82f6; color: white; font-weight: 900; text-align: center;"
                return "background-color: #10b981; color: white; font-weight: 900; text-align: center;"

            def _chg_style(val) -> str:
                try:
                    v = float(val)
                except Exception:
                    return ""
                if v >= 3:
                    return "background-color: #052e16; color: #dcfce7; font-weight: 900;"
                elif v >= 0:
                    return "background-color: #14532d; color: #dcfce7; font-weight: 800;"
                elif v <= -3:
                    return "background-color: #7f1d1d; color: #fee2e2; font-weight: 900;"
                return "background-color: #991b1b; color: #fee2e2; font-weight: 800;"

            styled = (
                dfp.style
                .applymap(_urgency_style, subset=["⚡"])
                .applymap(_chg_style, subset=["Δ%"])
                .set_properties(
                    **{
                        "text-align": "left",
                        "font-size": "0.8rem",
                        "font-weight": "700",
                        "padding": "10px 12px",
                        "border": "1px solid rgba(148,163,184,0.15)",
                    }
                )
                .set_table_styles(
                    [
                        {
                            "selector": "thead th",
                            "props": [
                                ("background", "linear-gradient(135deg, #1e293b 0%, #0f172a 100%)"),
                                ("color", "#f1f5f9"),
                                ("font-weight", "900"),
                                ("padding", "12px"),
                                ("font-size", "0.75rem"),
                                ("text-transform", "uppercase"),
                                ("letter-spacing", "0.5px"),
                                ("border", "1px solid rgba(148,163,184,0.3)"),
                            ],
                        },
                        {
                            "selector": "tbody tr:hover",
                            "props": [
                                ("background-color", "rgba(59,130,246,0.12)"),
                                ("transform", "scale(1.01)"),
                                ("box-shadow", "0 4px 6px rgba(0,0,0,0.1)"),
                            ],
                        },
                        {
                            "selector": "tbody td",
                            "props": [
                                ("background-color", "rgba(248,250,252,0.5)"),
                            ],
                        },
                    ]
                )
            )

            st.caption("📞 CALL = Cap procurement cost | 📉 PUT = Protect selling price floor | ⚡ = 🔴 Immediate 🟡 Near-term 🟢 Monitor")
            st.dataframe(styled, use_container_width=True, height=400)
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
        <div class="cp-title" style="margin-bottom: 0.2rem;">Recommendation</div>
    <div class="cp-subtitle"><b>{top.get('when')}</b> · {top.get('label')} · Horizon: {horizon} · Score {score:.1f}</div>
  </div>
  <div class="cp-pill {pill_class}">{top_title}</div>
</div>
            """,
            unsafe_allow_html=True,
        )

        proc_decision = str(top.get("proc_decision") or "—")
        proc_window = str(top.get("proc_window") or "—")
        proc_rationale = str(top.get("proc_rationale") or "—")
        proc_schedule = str(top.get("proc_schedule") or "—")
        
        # Extract key numbers from proc_schedule for compact display
        import re
        phase_match = re.findall(r'(\d+)%', proc_schedule)
        
        st.markdown("<div class='cp-kv-label' style='margin-top: 0.5rem;'>Structured Sourcing Schedule</div>", unsafe_allow_html=True)
        
        # Visual card layout with color-coded sections
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"""
<div style='background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); 
            padding: 1rem; border-radius: 8px; color: white; margin-bottom: 0.5rem;'>
    <div style='font-size: 0.7rem; font-weight: 600; opacity: 0.9; margin-bottom: 0.3rem;'>📋 DECISION</div>
    <div style='font-size: 1rem; font-weight: 900;'>{proc_decision}</div>
</div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
<div style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
            padding: 1rem; border-radius: 8px; color: white;'>
    <div style='font-size: 0.7rem; font-weight: 600; opacity: 0.9; margin-bottom: 0.3rem;'>⏱️ TIMING</div>
    <div style='font-size: 0.95rem; font-weight: 800;'>{proc_window}</div>
</div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
<div style='background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); 
            padding: 1rem; border-radius: 8px; color: white; margin-bottom: 0.5rem;'>
    <div style='font-size: 0.7rem; font-weight: 600; opacity: 0.9; margin-bottom: 0.3rem;'>📊 RATIONALE</div>
    <div style='font-size: 0.95rem; font-weight: 800;'>{proc_rationale}</div>
</div>
            """, unsafe_allow_html=True)
            
            # Phase allocation visualization
            if phase_match:
                phases_html = " → ".join([f"<span style='background: rgba(255,255,255,0.3); padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: 900;'>{p}%</span>" for p in phase_match[:3]])
                st.markdown(f"""
<div style='background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); 
            padding: 1rem; border-radius: 8px; color: white;'>
    <div style='font-size: 0.7rem; font-weight: 600; opacity: 0.9; margin-bottom: 0.3rem;'>🎯 ALLOCATION PHASES</div>
    <div style='font-size: 0.9rem; font-weight: 800;'>{phases_html}</div>
</div>
                """, unsafe_allow_html=True)
        
        # Optional: Full details in expander
        if proc_schedule and proc_schedule != "—" and len(proc_schedule) > 50:
            with st.expander("📄 Full Execution Strategy", expanded=False):
                st.markdown(f"<div style='background: #f8fafc; padding: 0.75rem; border-radius: 6px; font-size: 0.85rem; line-height: 1.6;'>{proc_schedule}</div>", unsafe_allow_html=True)

        kv1, kv2, kv3 = st.columns(3)
        with kv1:
            st.markdown(
                f"""
<div class="cp-kv">
    <div class="cp-kv-label">Current Spot Market Price</div>
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
                "<div class='cp-note'><b>Request the following option legs</b> (indicative, per unit). Multiply by your hedge size.</div>",
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

        st.caption("Decision support only: confirm with your bank and internal policy before executing derivatives.")

        return


def _extract_forecast_curve_from_payload(*, payload: dict, max_months: int = 24) -> list[dict]:
    """Return sorted forecast curve entries: [{horizon, months, price, lower, upper, change}]"""
    preds = payload.get("predictions") or {}
    curve: list[dict] = []
    if not isinstance(preds, dict) or not preds:
        return curve

    for h, p in preds.items():
        if not isinstance(p, dict):
            continue
        m = _parse_horizon_months(str(h))
        if m < 1 or m > int(max_months):
            continue
        try:
            px = float(p.get("price"))
        except Exception:
            continue
        if not np.isfinite(px):
            continue
        lo = p.get("lower")
        hi = p.get("upper")
        try:
            lo_f = float(lo) if lo is not None and np.isfinite(float(lo)) else None
        except Exception:
            lo_f = None
        try:
            hi_f = float(hi) if hi is not None and np.isfinite(float(hi)) else None
        except Exception:
            hi_f = None
        ch = p.get("change")
        try:
            ch_f = float(ch) if ch is not None and np.isfinite(float(ch)) else None
        except Exception:
            ch_f = None

        curve.append({"horizon": str(h), "months": int(m), "price": float(px), "lower": lo_f, "upper": hi_f, "change": ch_f})

    curve.sort(key=lambda e: (int(e.get("months", 999)), str(e.get("horizon", ""))))
    # If multiple horizons map to same month, keep the first
    dedup: dict[int, dict] = {}
    for e in curve:
        mm = int(e["months"])
        if mm not in dedup:
            dedup[mm] = e
    return [dedup[m] for m in sorted(dedup.keys())]


def _closest_curve_entry(curve: list[dict], target_months: int) -> dict | None:
    if not curve:
        return None
    try:
        target_months = int(target_months)
    except Exception:
        target_months = 6
    return min(curve, key=lambda e: abs(int(e.get("months", 999)) - target_months))


def _confidence_from_interval(*, s0: float, target_price: float, lower: float | None, upper: float | None) -> str:
    """Heuristic confidence from move size and interval width."""
    try:
        s0 = float(s0)
        target_price = float(target_price)
    except Exception:
        return "Moderate"
    if s0 <= 0 or not (np.isfinite(s0) and np.isfinite(target_price)):
        return "Moderate"

    move_pct = abs((target_price / s0 - 1.0) * 100.0)
    width_pct = None
    if lower is not None and upper is not None:
        try:
            lower = float(lower)
            upper = float(upper)
            if np.isfinite(lower) and np.isfinite(upper) and target_price > 0:
                width_pct = abs(upper - lower) / target_price * 100.0
        except Exception:
            width_pct = None

    if width_pct is None:
        # If we don't have an interval, require a bigger move for "High"
        if move_pct >= 10:
            return "High"
        if move_pct >= 5:
            return "Moderate"
        return "Low"

    # Be more forgiving on interval width: forecast bands are often wide on commodities.
    if move_pct >= 8 and width_pct <= 30:
        return "High"
    if move_pct >= 4 and width_pct <= 50:
        return "Moderate"
    return "Low"


def render_forecast_strategy_engine(
    *,
    expander_title: str,
    key_prefix: str,
    commodity_payloads: list[dict] | None,
    expanded: bool = False,
) -> None:
    """Portfolio strategist driven by the full forecast curve (multi-horizon)."""
    with st.expander(expander_title, expanded=expanded):
        st.markdown(
            """
<div class="cp-card" style="padding: 1.0rem 1.0rem;">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px; flex-wrap:wrap;">
    <div>
            <div class="cp-title">Autonomous Strategy Engine — System-Generated Trading Signals</div>
            <div class="cp-subtitle">Uses <b>Current Spot Market Price</b> + <b>full forecast curve</b> (3M/6M/12M/18M/24M) to propose a <b>Structured Sourcing Schedule</b>/<b>Sales Execution Schedule</b> and <b>hedge structures</b>.</div>
    </div>
    <div class="cp-pill cp-pill-hold">Auto</div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

        if not commodity_payloads:
            st.info("No commodities available.")
            return

        t1, t2, t3 = st.tabs(["🛒 Buying (Procurement)", "💼 Selling (Sales)", "📦 Inventory (Hold)"])

        def _build_rows(exposure: str) -> pd.DataFrame:
            rows: list[dict] = []
            for item in commodity_payloads:
                for side in ("int_payload", "local_payload"):
                    p = item.get(side)
                    if not isinstance(p, dict) or not p.get("name"):
                        continue

                    label = f"{p['name']} ({'International' if side=='int_payload' else 'Local'})"
                    scale = float(p.get("display_scale", 1.0) or 1.0)
                    unit = str(p.get("display_currency") or p.get("info", {}).get("currency", ""))
                    s0 = float(p.get("current_price") or 0.0) * scale

                    curve = _extract_forecast_curve_from_payload(payload=p, max_months=24)
                    if not curve or s0 <= 0:
                        continue

                    # Scale curve values to match Summary display
                    curve_s = []
                    for e in curve:
                        e2 = dict(e)
                        e2["price"] = float(e2["price"]) * scale
                        if e2.get("lower") is not None:
                            e2["lower"] = float(e2["lower"]) * scale
                        if e2.get("upper") is not None:
                            e2["upper"] = float(e2["upper"]) * scale
                        curve_s.append(e2)

                    min_e = min(curve_s, key=lambda e: float(e.get("price", 1e18)))
                    max_e = max(curve_s, key=lambda e: float(e.get("price", -1e18)))

                    # Key snapshots
                    snap_months = [3, 6, 12, 18, 24]
                    snaps: list[str] = []
                    name_lower = str(p.get("name", "")).lower()
                    three_dec_assets = ("cotton", "polyester", "viscose", "crude", "natural gas")
                    dec = 3 if (any(k in name_lower for k in three_dec_assets) or "/lb" in unit.lower()) else 2
                    for m in snap_months:
                        ce = _closest_curve_entry(curve_s, m)
                        if ce:
                            snaps.append(f"{m}M: {float(ce['price']):,.{dec}f}")
                    path_str = " | ".join(snaps[:5])

                    # Strategy logic by exposure
                    action = "MONITOR"
                    timing = "—"
                    primary = "—"
                    hedge = "—"

                    if exposure.startswith("Procurement"):
                        target = min_e
                        target_px = float(target["price"])
                        move_pct = (target_px / s0 - 1.0) * 100.0
                        if move_pct <= -2.0:
                            action = "DEFER PROCUREMENT / PHASED PROCUREMENT ALLOCATION"
                            timing = f"Forward Maturity Target: {target['horizon']} (~{target['months']}M)"
                            primary = f"Structured Sourcing Schedule: defer majority coverage; ladder procurement toward {target['horizon']} (forecast low)."
                            hedge = "Options Overlay Strategy: Limited Upside Call Option Exposure (ATM) (or Bull Call Spread Strategy) to cap procurement cost risk while awaiting the target window."
                        elif move_pct >= 2.0:
                            action = "INITIATE COVERAGE / ESTABLISH HEDGE POSITION"
                            timing = f"Execution Window: now → {target['horizon']} (~{target['months']}M)"
                            primary = "Price Fixation via Forward Contract (or futures) for a defined tranche; retain flexibility on residual exposure."
                            hedge = "Establish Hedge Position: Call Option (or Bull Call Spread Strategy) to cap procurement cost risk for remaining exposure."
                        else:
                            action = "PHASED PROCUREMENT ALLOCATION"
                            timing = "Execution Window: next 1–3 months"
                            primary = "Phased Procurement Allocation (e.g., 30/30/40) over coming months; re-assess monthly."
                            hedge = "If premium-constrained: Bull Call Spread Strategy; otherwise monitor." 

                        conf_raw = _confidence_from_interval(s0=s0, target_price=float(target["price"]), lower=target.get("lower"), upper=target.get("upper"))
                        conf = {
                            "High": "Elevated Market Risk Classification",
                            "Moderate": "Moderate Risk Classification",
                            "Low": "Lower Risk Classification",
                        }.get(str(conf_raw), str(conf_raw))
                        exp_move = (float(target["price"]) / s0 - 1.0) * 100.0

                    elif exposure.startswith("Sales"):
                        target = max_e
                        target_px = float(target["price"])
                        move_pct = (target_px / s0 - 1.0) * 100.0
                        if move_pct >= 2.0:
                            action = "DEFER SALES / MAINTAIN OPTIONALITY"
                            timing = f"Forward Maturity Target: {target['horizon']} (~{target['months']}M)"
                            primary = f"Sales Execution Schedule: defer sales into {target['horizon']} (forecast high), subject to inventory constraints."
                            hedge = "Establish Hedge Position: Put Option (downside floor) or Collar Structure (cost-reduced floor)."
                        elif move_pct <= -2.0:
                            action = "INITIATE SALES COVERAGE / ESTABLISH HEDGE POSITION"
                            timing = f"Execution Window: now → {target['horizon']} (~{target['months']}M)"
                            primary = "Price Fixation via Forward Contract (or futures) to lock sales price; reduce downside exposure."
                            hedge = "Establish Hedge Position: Put Option or Put Spread Strategy (premium-limited)."
                        else:
                            action = "PHASED SALES EXECUTION"
                            timing = "Execution Window: next 1–3 months"
                            primary = "Phased sales execution in tranches; re-assess monthly."
                            hedge = "Consider Collar Structure where a minimum price floor is required." 

                        conf_raw = _confidence_from_interval(s0=s0, target_price=float(target["price"]), lower=target.get("lower"), upper=target.get("upper"))
                        conf = {
                            "High": "Elevated",
                            "Moderate": "Moderate",
                            "Low": "Lower",
                        }.get(str(conf_raw), str(conf_raw))
                        exp_move = (float(target["price"]) / s0 - 1.0) * 100.0

                    else:
                        # Inventory hold: focus on protecting downside when vol/move is meaningful
                        target = _closest_curve_entry(curve_s, 6) or min_e
                        target_px = float(target["price"])
                        move_pct = (target_px / s0 - 1.0) * 100.0
                        if move_pct <= -2.0:
                            action = "DOWNSIDE RISK MITIGATION"
                            timing = f"Execution Window: now → {target['horizon']} (~{target['months']}M)"
                            primary = "Maintain inventory plan; add downside protection under downward price expectation."
                            hedge = "Establish Hedge Position: Put Option (floor) or Put Spread Strategy; consider Collar Structure to reduce premium."
                        elif move_pct >= 2.0:
                            action = "MAINTAIN UPSIDE OPTIONALITY"
                            timing = f"Execution Window: through {target['horizon']} (~{target['months']}M)"
                            primary = "Maintain inventory for upside participation; avoid over-hedging." 
                            hedge = "Optional: Collar Structure if a minimum price floor is required." 
                        else:
                            action = "MONITOR / REASSESS"
                            timing = "Re-assess monthly"
                            primary = "No strong forecast edge; focus on execution discipline and cashflow constraints." 
                            hedge = "Apply protection only where policy/limits require." 

                        conf_raw = _confidence_from_interval(s0=s0, target_price=float(target["price"]), lower=target.get("lower"), upper=target.get("upper"))
                        conf = {
                            "High": "Elevated",
                            "Moderate": "Moderate",
                            "Low": "Lower",
                        }.get(str(conf_raw), str(conf_raw))
                        exp_move = (float(target["price"]) / s0 - 1.0) * 100.0

                    # Add a concrete option-structure suggestion using the existing engine
                    try:
                        # Pull a rough vol from history for option structure sizing
                        hist_df = p.get("history_df")
                        vcol = p.get("info", {}).get("value_col") or p.get("value_col") or "value"
                        sigma_ann = 0.25
                        if isinstance(hist_df, pd.DataFrame) and vcol in hist_df.columns:
                            sigma_ann = float(_annualized_volatility_from_history(hist_df[vcol]))
                        t_years = float(max(1, int(target.get("months", 6))) / 12.0)
                        strat = _recommend_hedge_strategy(
                            exposure=exposure,
                            s0=s0,
                            s_mean=float(target.get("price", s0)),
                            sigma_ann=sigma_ann,
                            t_years=t_years,
                            risk_profile="Balanced",
                            budget_priority="Moderate",
                            allow_selling=False,
                            qty=1.0,
                            unit=unit,
                        )
                        legs = strat.get("legs") or []
                        if legs:
                            legs_txt = []
                            for l in legs[:2]:
                                try:
                                    legs_txt.append(f"{str(l.get('side')).upper()} {str(l.get('type')).upper()} @ {float(l.get('strike')):,.{dec}f}")
                                except Exception:
                                    continue
                            if legs_txt:
                                hedge = f"{hedge}  Suggested legs: " + " · ".join(legs_txt)
                    except Exception:
                        pass

                    rows.append(
                        {
                            "🎯 Action": action[:40],  # Truncate long text
                            "Commodity": label,
                            "💰 Spot": round(s0, dec),
                            "📈 Forecast": path_str[:30],  # Shorten path description
                            "⏰ Timing": timing[:25],
                            "Δ%": round(exp_move, 1),
                            "🛒 Sourcing": primary[:40],
                            "🛡️ Hedge": hedge[:50],  # Truncate hedge details
                            "⚠️ Risk": conf.replace(" Market Risk Classification", ""),  # Shorten labels
                            "Unit": unit,
                        }
                    )

            df = pd.DataFrame(rows)
            if df.empty:
                return df

            # Rank: strongest expected move first
            try:
                df["__abs_move"] = df["Δ%"].abs()
                conf_rank = {
                    "Elevated": 2,
                    "Moderate": 1,
                    "Lower": 0,
                }
                df["__conf"] = df["⚠️ Risk"].map(conf_rank).fillna(1)
                df = df.sort_values(["__conf", "__abs_move"], ascending=[False, False]).drop(columns=["__abs_move", "__conf"])
            except Exception:
                pass

            return df

        def _render_table(df: pd.DataFrame):
            if df is None or df.empty:
                st.info("No forecast curve available yet for these commodities.")
                return

            def _conf_style(v: str) -> str:
                vv = str(v)
                if "Elevated" in vv:
                    return "background-color:#052e16; color:#dcfce7; font-weight:900;"
                if "Moderate" in vv:
                    return "background-color:#1e3a8a; color:#e0e7ff; font-weight:900;"
                return "background-color:#0f172a; color:#e5e7eb; font-weight:900;"

            def _act_style(v: str) -> str:
                vv = str(v)
                if "INITIATE" in vv or "ESTABLISH" in vv:
                    return "background-color:#7f1d1d; color:#ffffff; font-weight:900;"
                if "DEFER" in vv:
                    return "background-color:#064e3b; color:#dcfce7; font-weight:900;"
                if "PHASED" in vv:
                    return "background-color:#92400e; color:#ffffff; font-weight:900;"
                if "RISK" in vv or "MITIGATION" in vv:
                    return "background-color:#1e3a8a; color:#e0e7ff; font-weight:900;"
                return "background-color:#0b1220; color:#e5e7eb; font-weight:900;"

            styled = (
                df.style
                .applymap(_act_style, subset=["Strategic Positioning Directive"])
                .applymap(_conf_style, subset=["Market Risk Classification"])
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
            st.dataframe(styled, use_container_width=True, height=420)

        with t1:
            df = _build_rows("Procurement (forward buying)")
            _render_table(df)

        with t2:
            df = _build_rows("Sales (forward selling)")
            _render_table(df)

        with t3:
            df = _build_rows("Inventory (stock on hand)")
            _render_table(df)

        st.caption("Forecast-driven decision support. Validate against procurement/sales constraints, liquidity, and executable bank/broker quotes before implementing derivatives or forwards.")
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


def _asset_type_from_label(label: str) -> str:
    """Heuristic asset type classifier. Defaults to Commodity for this app."""
    s = str(label or "").lower()
    if any(k in s for k in ("eurusd", "usdpkr", "usd/pkr", "fx", "exchange")):
        return "Currency"
    if any(k in s for k in ("index", "equity", "stock", "bond")):
        return "Investment Asset"
    return "Commodity"


def _theoretical_futures_price(
    *,
    asset_type: str,
    s: float,
    r: float,
    t_years: float,
    storage_cost: float = 0.0,
    convenience_yield: float = 0.0,
    r_domestic: float | None = None,
    r_foreign: float | None = None,
    dividend_yield: float = 0.0,
    pv_income: float = 0.0,
) -> float:
    """Hull Ch. 5-style forward/futures fair value models."""
    import math

    asset_type = str(asset_type or "Commodity").strip()
    s = float(s)
    r = float(r)
    t_years = float(max(t_years, 1e-9))

    if asset_type == "Commodity":
        return float(_theoretical_futures_price_commodity(s=s, r=r, storage_cost=float(storage_cost), convenience_yield=float(convenience_yield), t_years=t_years))

    if asset_type == "Currency":
        rd = float(r_domestic if r_domestic is not None else r)
        rf = float(r_foreign if r_foreign is not None else 0.0)
        return float(s * math.exp((rd - rf) * t_years))

    # Investment Asset
    q = float(dividend_yield)
    pv_income = float(max(0.0, pv_income))
    # If known income PV provided, use (S - PV(income))e^{rT}; else use yield form S e^{(r-q)T}
    if pv_income > 0.0:
        return float((s - pv_income) * math.exp(r * t_years))
    return float(s * math.exp((r - q) * t_years))


def _implied_forward_from_put_call_parity(*, c: float, p: float, k: float, r: float, t_years: float) -> float:
    """Implied forward from parity: C - P = e^{-rT}(F - K) => F = K + (C - P)e^{rT}."""
    import math

    c = float(c)
    p = float(p)
    k = float(k)
    r = float(r)
    t_years = float(max(t_years, 1e-9))
    return float(k + (c - p) * math.exp(r * t_years))


def _format_legs_brief(*, strat: dict, dec: int) -> str:
    legs = (strat or {}).get("legs") or []
    if not legs:
        return "—"
    out: list[str] = []
    for l in legs[:3]:
        try:
            out.append(f"{str(l.get('side')).upper()} {str(l.get('type')).upper()} @ {float(l.get('strike')):,.{dec}f}")
        except Exception:
            continue
    return " · ".join(out) if out else "—"


def render_integrated_strategy_engine(
    *,
    expander_title: str,
    key_prefix: str,
    commodity_payloads: list[dict] | None,
    expanded: bool = False,
) -> None:
    """Integrated strategist: Forecast + No-arbitrage (carry/parity) + risk filter.

    Produces rows classified into:
      - Hedging Strategy
      - Speculative Timing Strategy
      - Arbitrage Strategy
    """
    with st.expander(expander_title, expanded=expanded):
        st.markdown(
            """
<div class="cp-card" style="padding: 1.0rem 1.0rem;">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px; flex-wrap:wrap;">
    <div>
                        <div class="cp-title">Autonomous Strategy Engine — System-Generated Trading Signals</div>
                                                <div class="cp-subtitle">Uses <b>price forecasts</b> and <b>market indicators</b> to recommend <b>structured sourcing schedules</b> and <b>hedge structures</b> (no user inputs required).</div>
    </div>
        <div class="cp-pill cp-pill-hold">AUTO</div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

        if not commodity_payloads:
            st.info("No commodities available.")
            return

        # No user inputs: fully automatic engine.
        # (Optional internal diagnostics can be enabled via Streamlit secrets/env without exposing UI controls.)
        debug_flag = str(_get_streamlit_secret("STRATEGY_DEBUG") or "").strip().lower()
        debug_mode = debug_flag in {"1", "true", "yes", "y"}

        # Internal defaults (kept conservative; do not expose as UI knobs)
        forecast_sig = 3.0

        def _priority_badge(v: str) -> tuple[str, str]:
            vv = str(v)
            if "High" in vv:
                return ("Elevated", "#16a34a")
            if "Moderate" in vv:
                return ("Moderate", "#2563eb")
            return ("Lower", "#0f172a")

        def _render_cards(rows: list[dict], *, empty_msg: str) -> None:
            if not rows:
                st.info(empty_msg)
                return
            
            # Professional card-based layout with visual hierarchy
            for r in rows[:12]:  # Show top 12
                priority = str(r.get("Priority", "Low"))
                commodity = str(r.get("Commodity") or "—")
                
                # Extract key info
                decision = str(r.get("Decision") or "—")
                if "Structured Sourcing Schedule:" in decision:
                    decision = decision.split("Structured Sourcing Schedule:")[1].split("|")[0].strip()
                
                when_txt = str(r.get("When") or "—")
                if "Procurement Timing Strategy:" in when_txt:
                    when_txt = when_txt.split("Procurement Timing Strategy:")[1].split("·")[0].strip()
                
                why_txt = str(r.get("Why") or "—")
                how_txt = str(r.get("How") or "—")  # This contains the detailed trade recommendations
                
                # Extract forecast % from why_txt to show market direction
                forecast_move = ""
                if why_txt and why_txt != "—":
                    import re
                    move_match = re.search(r'([\+\-]?[\d\.]+)%', why_txt)
                    if move_match:
                        move_val = move_match.group(1)
                        forecast_move = f" [{move_val}%]"
                
                # Extract key metrics from How text for prominent display
                import re
                profit_amount = None  # None = N/A, number = actual profit
                phase1_qty = "—"
                phase1_price = "—"
                strategy_name = "—"
                timing_display = when_txt[:30] if when_txt and when_txt != "—" else "—"
                
                if how_txt and how_txt != "—":
                    # Extract profit - NEW FORMAT: "NET PROFIT +123,456 USD" or OLD FORMAT: "**EXPECTED SAVINGS:** 123,456"
                    profit_patterns = [
                        r'NET PROFIT\s+\+?([\d,\.]+)',  # New simplified format
                        r'\*\*NET EXPECTED PROFIT:\*\*\s*([\d,\.]+)',  # Old format
                        r'\*\*TOTAL EXPECTED PROFIT:\*\*\s*([\d,\.]+)',  # Old format
                        r'\*\*EXPECTED SAVINGS:\*\*\s*([\d,\.]+)'  # Old format
                    ]
                    for pattern in profit_patterns:
                        profit_match = re.search(pattern, how_txt)
                        if profit_match:
                            profit_amount = float(profit_match.group(1).replace(',', ''))
                            break
                    
                    # Extract strategy name and details based on format
                    if "BEARISH STRATEGY" in how_txt.upper() or "DEFER & INVEST" in how_txt.upper():
                        strategy_name = "Defer & Invest (Bearish)"
                        # Extract buy quantity: "Buy 6,389 t (10% operational min)"
                        buy_match = re.search(r'Buy\s+([\d,\.]+\s*[tkgmton]+)\s*\(10%', how_txt)
                        if buy_match:
                            phase1_qty = f"Buy {buy_match.group(1)} now"
                            timing_display = "10% now, defer 90%"
                        # Extract invest amount in USD: "Invest savings in bank @ 4.5%: $111,740"
                        invest_match = re.search(r'Invest savings.*?\$([\ d,\.]+)', how_txt)
                        if invest_match:
                            phase1_price = f"Invest ${invest_match.group(1).strip()}"
                        else:
                            phase1_price = "Invest & wait"
                    elif "BULLISH STRATEGY" in how_txt.upper() or "BORROW & BUY" in how_txt.upper():
                        strategy_name = "Borrow & Buy (Bullish)"
                        # Extract buy details: "Buy 6,389 t (100% of need): $31,576"
                        buy_match = re.search(r'Buy\s+([\d,\.]+\s*[tkgmton]+).*?\$([\ d,\.]+)', how_txt)
                        if buy_match:
                            phase1_qty = f"Buy {buy_match.group(1)}"
                            phase1_price = f"${buy_match.group(2).strip()}"
                            timing_display = "100% now (leveraged)"
                        else:
                            phase1_qty = "Buy all now"
                            phase1_price = "Borrow & buy"
                            timing_display = "100% leveraged"
                    else:
                        # OLD FORMAT: Extract Phase 1 details (matches: • **Phase 1 (NOW):** Buy X tonnes @ Y.YY USD/lb)
                        phase1_match = re.search(r'•\s*\*\*Phase 1 \(NOW\):\*\*\s*Buy\s+([\d,\.]+\s*\w+)\s*@\s*([\d,\.]+)', how_txt)
                        if phase1_match:
                            phase1_qty = phase1_match.group(1).strip()
                            phase1_price = phase1_match.group(2).strip()
                            timing_display = "NOW (Phase 1)"
                        
                        # Extract strategy (matches: **STRATEGY: strategy name**)
                        strategy_match = re.search(r'\*\*STRATEGY:\s*([^\n\*]+)', how_txt)
                        if strategy_match:
                            strategy_name = strategy_match.group(1).strip()[:55]
                
                # Priority styling
                if priority == "High":
                    badge_color = "#dc2626"
                    badge_icon = "🔴"
                    border_color = "#dc2626"
                elif priority == "Moderate":
                    badge_color = "#3b82f6"  # Professional blue instead of yellow
                    badge_icon = "🔵"
                    border_color = "#3b82f6"
                else:
                    badge_color = "#10b981"
                    badge_icon = "🟢"
                    border_color = "#10b981"
                
                # Compact visual card with profit prominently displayed
                st.markdown(f"""
<div style='background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); border-left: 5px solid {border_color}; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 0.75rem; box-shadow: 0 2px 4px rgba(0,0,0,0.08);'>
    <div style='display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;'>
        <div style='flex: 1;'>
            <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                <span style='font-size: 1.2rem; font-weight: 900; color: #0f172a;'>{commodity}</span>
                <span style='background: {badge_color}; color: white; padding: 0.15rem 0.5rem; border-radius: 12px; font-size: 0.7rem; font-weight: 800;'>{badge_icon} {priority}</span>
            </div>
            <div style='font-size: 0.85rem; color: #64748b; font-weight: 600;'>{decision}{forecast_move}</div>
        </div>
        <div style='text-align: right;'>
            <div style='font-size: 0.7rem; color: #64748b; font-weight: 700; margin-bottom: 0.15rem;'>EXPECTED PROFIT</div>
            <div style='font-size: 1.8rem; font-weight: 900; color: {'#94a3b8' if profit_amount is None else '#059669'};'>{'N/A' if profit_amount is None else f'${profit_amount:,.0f}'}</div>
        </div>
    </div>
</div>
                """, unsafe_allow_html=True)
                
                # Display extracted metrics using columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**📦 Phase 1 Qty**")
                    st.markdown(f"**{phase1_qty}**")
                with col2:
                    st.markdown("**💰 Price**")
                    st.markdown(f"**{phase1_price}**")
                with col3:
                    st.markdown("**⏰ Timing**")
                    st.markdown(f"**{timing_display}**")
                
                st.markdown(f"**🎯 STRATEGY:** {strategy_name}")
                st.markdown("---")
                
                # Display detailed trade recommendations with profit calculations
                if how_txt and how_txt != "—" and len(how_txt) > 50:
                    # Convert markdown-style formatting to HTML
                    how_html = how_txt.replace("**", "<b>").replace("**", "</b>")
                    how_html = how_html.replace("\n•", "<br>•").replace("\n\n", "<br><br>")
                    
                    with st.expander(f"💰 View Detailed Trade Recommendation & Profit Calculation", expanded=False):
                        st.markdown(f"""
<div style='background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
            padding: 1rem; 
            border-radius: 8px; 
            border-left: 4px solid #0284c7;
            font-size: 0.85rem;
            line-height: 1.6;
            color: #0c4a6e;'>
    {how_html}
</div>
                        """, unsafe_allow_html=True)
            
            # Full details in expander
            if len(rows) > 0:
                with st.expander("📋 View Full Execution Details", expanded=False):
                    for r in rows[:8]:
                        st.markdown(f"#### {r.get('Commodity', '—')}")
                        detail_df = pd.DataFrame([{
                            "Field": k,
                            "Value": str(v)
                        } for k, v in r.items() if k != "Commodity" and str(v) != "—"])
                        st.dataframe(detail_df, use_container_width=True, hide_index=True)
                        st.markdown("---")

        # Build candidate list
        candidates: list[dict] = []
        for item in commodity_payloads:
            for side in ("int_payload", "local_payload"):
                p = item.get(side)
                if p and p.get("name"):
                    base_name = str(p["name"])
                    market = "International" if side == "int_payload" else "Local"
                    name_l = base_name.lower()
                    market_l = market.lower()
                    if market_l in name_l:
                        label = base_name
                    else:
                        label = f"{base_name} ({market})"
                    candidates.append({"label": label, "payload": p})

        if not candidates:
            st.info("No commodities available.")
            return

        # Purchase history sizing (optional; file may not exist on Streamlit Cloud)
        def _purchase_commodity_from_label(lbl: str) -> str | None:
            s = str(lbl or "").lower()
            if "cotton" in s:
                return "Cotton"
            if "polyester" in s:
                return "Polyester"
            if "viscose" in s:
                return "Viscose"
            if "crude" in s:
                return "Crude Oil"
            if "natural gas" in s or "nat gas" in s:
                return "Natural Gas"
            return None

        def _fmt_qty_kg(qty_kg: float) -> str:
            try:
                q = float(qty_kg)
                if not (np.isfinite(q) and q > 0):
                    return "—"
            except Exception:
                return "—"
            if q >= 10000:
                return f"{q/1000.0:,.0f} t"
            if q >= 1000:
                return f"{q/1000.0:,.1f} t"
            return f"{q:,.0f} kg"

        purchase_ctx: dict[str, dict] = {}
        try:
            purch_df = _load_purchase_monthly_agg()
            if isinstance(purch_df, pd.DataFrame) and not purch_df.empty and "month" in purch_df.columns:
                qty_col = "total_qty_kg" if "total_qty_kg" in purch_df.columns else ("total_qty" if "total_qty" in purch_df.columns else None)
                if qty_col:
                    purch_df = purch_df.copy()
                    purch_df["month"] = pd.to_datetime(purch_df["month"], errors="coerce")
                    purch_df = purch_df.dropna(subset=["month"])  # type: ignore[arg-type]

                    anchor = purch_df["month"].max()
                    window_start = anchor - pd.DateOffset(months=12)
                    df12 = purch_df[purch_df["month"] >= window_start]

                    monthly = df12.groupby(["commodity", "month"], dropna=False)[qty_col].sum(min_count=1).reset_index()
                    unit_totals = df12.groupby(["commodity", "operating_unit"], dropna=False)[qty_col].sum(min_count=1).reset_index()

                    for comm, sub in monthly.groupby("commodity"):
                        s = pd.to_numeric(sub[qty_col], errors="coerce").dropna()
                        if len(s) == 0:
                            continue
                        mean_m = float(s.mean())
                        med_m = float(s.median())
                        std_m = float(s.std(ddof=0)) if len(s) > 1 else 0.0
                        cv = float(std_m / mean_m) if mean_m > 0 else 0.0

                        topu = unit_totals[unit_totals["commodity"] == comm].copy()
                        topu[qty_col] = pd.to_numeric(topu[qty_col], errors="coerce")
                        topu = topu.dropna(subset=[qty_col]).sort_values(by=qty_col, ascending=False).head(3)
                        top_units = [str(x) for x in topu["operating_unit"].tolist()]

                        purchase_ctx[str(comm)] = {
                            "median_monthly_kg": med_m,
                            "mean_monthly_kg": mean_m,
                            "cv": cv,
                            "top_units": top_units,
                            "asof": anchor,
                        }
        except Exception:
            purchase_ctx = {}

        # Build structured outputs (no role selector; keep it simple)
        role_filter = ["Hedging Strategy", "Speculative Timing Strategy"]

        outputs: list[dict] = []
        for c in candidates:
            p = c["payload"]
            label = c["label"]
            scale = float(p.get("display_scale", 1.0) or 1.0)
            unit = str(p.get("display_currency") or p.get("info", {}).get("currency", ""))
            s0 = float(p.get("current_price") or 0.0) * scale
            if not (np.isfinite(s0) and s0 > 0):
                continue

            name_lower = str(p.get("name", "")).lower()
            three_dec_assets = ("cotton", "polyester", "viscose", "crude", "natural gas")
            dec = 3 if (any(k in name_lower for k in three_dec_assets) or "/lb" in unit.lower()) else 2

            curve = _extract_forecast_curve_from_payload(payload=p, max_months=24)
            if not curve:
                continue

            # scale curve
            curve_s: list[dict] = []
            for e in curve:
                e2 = dict(e)
                e2["price"] = float(e2["price"]) * scale
                if e2.get("lower") is not None:
                    e2["lower"] = float(e2["lower"]) * scale
                if e2.get("upper") is not None:
                    e2["upper"] = float(e2["upper"]) * scale
                curve_s.append(e2)

            min_e = min(curve_s, key=lambda e: float(e.get("price", 1e18)))
            max_e = max(curve_s, key=lambda e: float(e.get("price", -1e18)))
            mid_e = _closest_curve_entry(curve_s, 6) or min_e

            # Speculative timing strategy (Step 3) - Enhanced with specific quantities, prices, and profit calculations
            if "Speculative Timing Strategy" in role_filter:
                move_to_min = (float(min_e["price"]) / s0 - 1.0) * 100.0
                move_to_max = (float(max_e["price"]) / s0 - 1.0) * 100.0

                # Get actual purchase history for this commodity
                pc = _purchase_commodity_from_label(label)
                mqty = 0.0
                monthly_spend = 0.0
                if pc and pc in purchase_ctx:
                    mqty = float(purchase_ctx[pc].get("median_monthly_kg") or 0.0)
                    monthly_spend = mqty * s0 if mqty > 0 else 0.0
                
                # If no purchase data, use reasonable defaults based on commodity type
                # IMPORTANT: Only use defaults for commodities with actual purchase history
                if mqty == 0:
                    if pc == "Cotton":
                        mqty = 6_389_000.0  # 6,389 tonnes/month (historical median)
                    elif pc == "Polyester":
                        mqty = 1_262_000.0  # 1,262 tonnes/month
                    elif pc == "Viscose":
                        mqty = 1_224_000.0  # 1,224 tonnes/month
                    elif pc == "Crude Oil":
                        mqty = 500_000.0  # 500 tonnes/month (example)
                    else:
                        # For commodities without purchase history, skip strategy generation
                        # to avoid unrealistic profit projections
                        mqty = 0.0
                    monthly_spend = mqty * s0 if mqty > 0 else 0.0

                market_condition = "Efficient"
                strat_name = "Monitor"
                steps = "Re-check monthly as new data arrives"
                logic = "Forecast does not show a strong edge vs Current Spot Market Price."
                driver = "Forecast realization"
                risk_notes = "Forecast uncertainty; execution constraints"
                
                # Hull Chapter 5: Commodity Forward Pricing Parameters
                # F = S × e^((r + u - y) × T)
                # r = risk-free rate, u = storage cost, y = convenience yield
                
                # Commodity-specific parameters (% per annum)
                if pc == "Cotton":
                    storage_cost_pct = 0.025  # 2.5% - warehousing, insurance
                    convenience_yield_pct = 0.010  # 1% - immediate availability value
                elif pc == "Polyester":
                    storage_cost_pct = 0.020  # 2% - less perishable
                    convenience_yield_pct = 0.005  # 0.5% - lower convenience
                elif pc == "Viscose":
                    storage_cost_pct = 0.020  # 2%
                    convenience_yield_pct = 0.005  # 0.5%
                elif pc == "Crude Oil":
                    storage_cost_pct = 0.035  # 3.5% - tank storage expensive
                    convenience_yield_pct = 0.040  # 4% - high convenience (energy security)
                elif pc == "Natural Gas":
                    storage_cost_pct = 0.050  # 5% - very expensive storage
                    convenience_yield_pct = 0.030  # 3%
                else:
                    storage_cost_pct = 0.025  # 2.5% default
                
                # Extract currency to determine financing rates (borrow vs invest)
                currency = unit.split("/")[0] if "/" in unit else unit
                
                # Currency conversion factor to USD (for consistent profit reporting)
                usd_conversion_rate = 1.0  # Default: already in USD
                base_currency = currency  # Original currency for display
                
                if "USD" in currency.upper():
                    borrowing_rate = 0.055  # 5.5% commercial loan rate
                    investment_rate = 0.045  # 4.5% bank deposit rate
                    usd_conversion_rate = 1.0
                elif "PKR" in currency.upper() or "RS" in currency.upper():
                    borrowing_rate = 0.165  # 16.5% commercial loan rate
                    investment_rate = 0.145  # 14.5% bank deposit rate
                    usd_conversion_rate = 1.0 / 280.0  # PKR to USD (1 USD = 280 PKR as of Feb 2026)
                else:
                    borrowing_rate = 0.060  # 6% generic
                    investment_rate = 0.050  # 5% generic
                    usd_conversion_rate = 1.0
                
                # Set display currency to USD for all profit calculations
                display_currency = "USD"
                
                # NEW: Specific trade recommendation with profit calculation
                trade_recommendation = ""
                expected_profit = None  # None indicates N/A, 0.0 indicates zero profit
                strategy_details = ""

                # SKIP STRATEGY if no actual purchase quantity (avoid unrealistic profit projections)
                if mqty == 0 or mqty is None or not np.isfinite(mqty):
                    # SPECIAL HANDLING for commodities that impact costs indirectly
                    pc_lower = commodity_name.lower()
                    
                    if "natural gas" in pc_lower:
                        # Natural Gas affects electricity costs - calculate indirect impact
                        # Estimate: 1 MMBTU price change → ~0.5 PKR/kWh electricity change
                        # Textile manufacturing: ~500 kWh per tonne of production
                        # Assume 10,000 tonnes/month total production
                        
                        gas_price_change = float(min_e.get("price", s0)) - s0
                        monthly_production_tonnes = 10000  # Estimated total production
                        kwh_per_tonne = 500
                        total_kwh_monthly = monthly_production_tonnes * kwh_per_tonne
                        
                        # Impact: MMBTU price change → electricity cost impact
                        electricity_impact_per_mmbtu = 0.5  # PKR per kWh per MMBTU change
                        monthly_cost_impact_pkr = abs(gas_price_change * electricity_impact_per_mmbtu * total_kwh_monthly)
                        target_months = float(min_e.get("months", 6))
                        total_cost_impact_pkr = monthly_cost_impact_pkr * target_months
                        
                        expected_profit = total_cost_impact_pkr * usd_conversion_rate if gas_price_change < 0 else -total_cost_impact_pkr * usd_conversion_rate
                        
                        decision = "Monitor & Hedge Electricity Costs" if gas_price_change > 0 else "Benefit from Lower Energy Costs"
                        when_txt = f"Over next {int(target_months)} months"
                        why_txt = f"Natural gas {'increase' if gas_price_change > 0 else 'decrease'} drives electricity tariff changes"
                        logic = f"Gas price forecast: {gas_price_change:+.2f} {unit} → Electricity impact: {monthly_cost_impact_pkr:,.0f} PKR/month"
                        
                        impact_direction = "increase" if gas_price_change > 0 else "decrease"
                        strategy_details = f"**📊 INDIRECT COST IMPACT ANALYSIS**\n\n"
                        strategy_details += f"• **Gas Price Change:** {gas_price_change:+.2f} {unit}\n"
                        strategy_details += f"• **Electricity Impact:** ~{monthly_cost_impact_pkr:,.0f} PKR/month\n"
                        strategy_details += f"• **Production Volume:** {monthly_production_tonnes:,} tonnes/month\n"
                        strategy_details += f"• **Energy Intensity:** {kwh_per_tonne} kWh/tonne\n"
                        strategy_details += f"• **Forecast Period:** {int(target_months)} months\n\n"
                        
                        if gas_price_change > 0:
                            trade_recommendation = f"⚡ **HEDGE ELECTRICITY COST RISK**\n\n"
                            trade_recommendation += f"Gas prices forecasted to {impact_direction} by {abs(gas_price_change):.2f} {unit}, "
                            trade_recommendation += f"which will increase electricity costs by ~{monthly_cost_impact_pkr:,.0f} PKR/month.\n\n"
                            trade_recommendation += f"**RECOMMENDED ACTIONS:**\n"
                            trade_recommendation += f"1. Lock in current electricity tariffs through forward contracts\n"
                            trade_recommendation += f"2. Negotiate fixed-rate agreements with WAPDA/K-Electric\n"
                            trade_recommendation += f"3. Consider on-site solar/wind to reduce grid dependency\n"
                            trade_recommendation += f"4. Implement energy efficiency measures to offset higher costs\n\n"
                            strategy_details += f"**💰 COST AVOIDANCE: ${abs(expected_profit):,.0f} USD**"
                        else:
                            trade_recommendation = f"✅ **BENEFIT FROM LOWER ENERGY COSTS**\n\n"
                            trade_recommendation += f"Gas prices forecasted to {impact_direction} by {abs(gas_price_change):.2f} {unit}, "
                            trade_recommendation += f"which will reduce electricity costs by ~{monthly_cost_impact_pkr:,.0f} PKR/month.\n\n"
                            trade_recommendation += f"**RECOMMENDED ACTIONS:**\n"
                            trade_recommendation += f"1. Delay any solar/renewable energy CAPEX investments\n"
                            trade_recommendation += f"2. Renegotiate electricity contracts at lower rates\n"
                            trade_recommendation += f"3. Increase production volume to capitalize on lower energy costs\n"
                            trade_recommendation += f"4. Offer competitive pricing to win new orders\n\n"
                            strategy_details += f"**💰 COST SAVINGS: ${abs(expected_profit):,.0f} USD**"
                        
                        steps = trade_recommendation
                        how_txt = strategy_details
                        conf = _confidence_from_interval(s0=s0, target_price=float(min_e.get("price", s0)), 
                                                         lower=float(min_e.get("low", s0*0.95)), 
                                                         upper=float(min_e.get("high", s0*1.05)))
                    
                    elif "crude" in pc_lower:
                        # Crude oil affects polyester/synthetic fiber costs
                        # Estimate: 30% of polyester cost is crude oil derived
                        # Use actual polyester purchase volume if available
                        oil_price_change = float(min_e.get("price", s0)) - s0
                        
                        # Try to get actual polyester volume from purchase data
                        try:
                            purch_df = _load_purchase_monthly_agg()
                            if not purch_df.empty and "commodity" in purch_df.columns:
                                polyester_data = purch_df[purch_df["commodity"].str.contains("Polyester", case=False, na=False)]
                                if not polyester_data.empty and "total_qty_kg" in purch_df.columns:
                                    avg_polyester_monthly_kg = polyester_data["total_qty_kg"].mean()
                                else:
                                    avg_polyester_monthly_kg = 200000  # Fallback estimate
                            else:
                                avg_polyester_monthly_kg = 200000
                        except:
                            avg_polyester_monthly_kg = 200000
                        
                        # Crude oil impact on polyester: ~$1/barrel change → ~$0.02/kg polyester change
                        polyester_cost_impact_per_barrel = 0.02  # USD per kg
                        target_months = float(min_e.get("months", 6))
                        monthly_impact_usd = abs(oil_price_change * polyester_cost_impact_per_barrel * avg_polyester_monthly_kg)
                        total_impact_usd = monthly_impact_usd * target_months
                        
                        expected_profit = total_impact_usd if oil_price_change < 0 else -total_impact_usd
                        
                        decision = "Monitor Polyester/Synthetic Costs" if oil_price_change > 0 else "Benefit from Lower Feedstock Costs"
                        when_txt = f"Over next {int(target_months)} months"
                        why_txt = f"Crude oil {'increase' if oil_price_change > 0 else 'decrease'} affects polyester raw material costs"
                        logic = f"Oil price forecast: {oil_price_change:+.2f} {unit} → Polyester impact: {monthly_impact_usd:,.0f} USD/month"
                        
                        strategy_details = f"**📊 FEEDSTOCK COST IMPACT ANALYSIS**\n\n"
                        strategy_details += f"• **Crude Oil Change:** {oil_price_change:+.2f} {unit}\n"
                        strategy_details += f"• **Polyester Volume:** {avg_polyester_monthly_kg:,.0f} kg/month\n"
                        strategy_details += f"• **Cost Impact:** ~{monthly_impact_usd:,.0f} USD/month\n"
                        strategy_details += f"• **Forecast Period:** {int(target_months)} months\n\n"
                        
                        if oil_price_change > 0:
                            trade_recommendation = f"⚠️ **PREPARE FOR HIGHER POLYESTER COSTS**\n\n"
                            trade_recommendation += f"Crude oil forecasted to rise {abs(oil_price_change):.2f} {unit}, "
                            trade_recommendation += f"increasing polyester costs by ~${monthly_impact_usd:,.0f}/month.\n\n"
                            trade_recommendation += f"**RECOMMENDED ACTIONS:**\n"
                            trade_recommendation += f"1. Forward-buy polyester at current prices\n"
                            trade_recommendation += f"2. Negotiate long-term contracts with suppliers\n"
                            trade_recommendation += f"3. Increase cotton blend ratios to reduce polyester dependency\n"
                            trade_recommendation += f"4. Pass through cost increases to customers\n\n"
                            strategy_details += f"**💰 COST AVOIDANCE: ${abs(expected_profit):,.0f} USD**"
                        else:
                            trade_recommendation = f"✅ **BENEFIT FROM LOWER POLYESTER COSTS**\n\n"
                            trade_recommendation += f"Crude oil forecasted to fall {abs(oil_price_change):.2f} {unit}, "
                            trade_recommendation += f"reducing polyester costs by ~${monthly_impact_usd:,.0f}/month.\n\n"
                            trade_recommendation += f"**RECOMMENDED ACTIONS:**\n"
                            trade_recommendation += f"1. Delay polyester purchases to benefit from lower prices\n"
                            trade_recommendation += f"2. Increase polyester orders for new product lines\n"
                            trade_recommendation += f"3. Offer competitive pricing on synthetic blends\n"
                            trade_recommendation += f"4. Build inventory at lower cost basis\n\n"
                            strategy_details += f"**💰 COST SAVINGS: ${abs(expected_profit):,.0f} USD**"
                        
                        steps = trade_recommendation
                        how_txt = strategy_details
                        conf = _confidence_from_interval(s0=s0, target_price=float(min_e.get("price", s0)),
                                                         lower=float(min_e.get("low", s0*0.95)),
                                                         upper=float(min_e.get("high", s0*1.05)))
                    
                    else:
                        # Generic fallback for other non-purchased commodities
                        trade_recommendation = "📊 **MARKET INDICATOR ONLY**\n\nThis commodity is not directly purchased but serves as a market indicator.\nMonitor for broader market trends and supply chain signals."
                        steps = trade_recommendation
                        logic = "No direct procurement - use as market intelligence"
                        decision = "Monitor for market trends"
                        when_txt = "Continuous monitoring"
                        why_txt = "Market indicator commodity"
                        conf = _confidence_from_interval(s0=s0, target_price=s0, lower=s0*0.95, upper=s0*1.05)
                        how_txt = trade_recommendation
                        expected_profit = None  # Truly N/A for generic indicators
                # For procurement: falling forecast suggests delaying; rising suggests early procurement
                elif move_to_min <= -float(forecast_sig):
                    market_condition = "Forecast Opportunity"
                    strat_name = "Bearish Reverse Carry Strategy"
                    half_months = max(1, int(round(float(min_e.get("months", 6)) / 2.0)))
                    half_e = _closest_curve_entry(curve_s, half_months) or mid_e
                    
                    # Calculate specific quantities and costs
                    target_months = float(min_e.get("months", 6))
                    total_need = mqty * target_months if mqty > 0 else 0.0
                    
                    if mqty > 0 and total_need > 0:
                        # BEARISH REVERSE CARRY: Delay purchase, invest savings, buy back cheaper
                        # Buy only 10% now (operational safety), delay 90%
                        qty_now = 0.10 * total_need
                        qty_mid = 0.30 * total_need
                        qty_target = 0.60 * total_need
                        
                        # Unit conversion
                        unit_multiplier = 1.0
                        if "/lb" in unit.lower():
                            unit_multiplier = 2.20462
                        
                        import math
                        time_to_target_years = target_months / 12.0
                        time_to_mid_years = half_months / 12.0
                        
                        # CASH FLOW TABLE (like team lead's example)
                        # TODAY: Buy 10%, invest budget for remaining 90%
                        cost_phase1 = qty_now * unit_multiplier * s0
                        budget_for_phase2 = qty_mid * unit_multiplier * s0  # What we would spend now
                        budget_for_phase3 = qty_target * unit_multiplier * s0
                        total_invested = budget_for_phase2 + budget_for_phase3
                        
                        # INVEST AND EARN INTEREST
                        # Phase 2: Invest until mid-point
                        investment_phase2 = budget_for_phase2
                        interest_earned_phase2 = investment_phase2 * investment_rate * time_to_mid_years
                        proceeds_phase2 = investment_phase2 + interest_earned_phase2
                        
                        # Phase 3: Invest until target
                        investment_phase3 = budget_for_phase3
                        interest_earned_phase3 = investment_phase3 * investment_rate * time_to_target_years
                        proceeds_phase3 = investment_phase3 + interest_earned_phase3
                        
                        total_interest_earned = interest_earned_phase2 + interest_earned_phase3
                        
                        # BUY AT LOWER PRICES
                        actual_cost_phase2 = qty_mid * unit_multiplier * float(half_e.get("price", s0))
                        actual_cost_phase3 = qty_target * unit_multiplier * float(min_e["price"])
                        
                        # STORAGE COSTS (only for qty bought)
                        storage_phase1 = cost_phase1 * storage_cost_pct * time_to_target_years
                        storage_phase2 = actual_cost_phase2 * storage_cost_pct * (time_to_target_years - time_to_mid_years)
                        total_storage = storage_phase1 + storage_phase2
                        
                        # NET PROFIT (convert to USD for consistent reporting)
                        baseline_cost = total_need * unit_multiplier * s0
                        strategy_cost = cost_phase1 + actual_cost_phase2 + actual_cost_phase3
                        price_savings = baseline_cost - strategy_cost
                        expected_profit_local = price_savings + total_interest_earned - total_storage
                        
                        # Convert to USD if needed
                        expected_profit = expected_profit_local * usd_conversion_rate
                        baseline_cost_usd = baseline_cost * usd_conversion_rate
                        strategy_cost_usd = strategy_cost * usd_conversion_rate
                        price_savings_usd = price_savings * usd_conversion_rate
                        interest_earned_usd = total_interest_earned * usd_conversion_rate
                        storage_usd = total_storage * usd_conversion_rate
                        
                        # BUILD CASH FLOW TABLE (Clean, readable format)
                        trade_recommendation = f"**🔴 BEARISH STRATEGY:** Defer & Invest\n"
                        trade_recommendation += f"**Forecast:** Price drops from {s0:,.{dec}f} → {float(min_e['price']):,.{dec}f} ({move_to_min:.1f}%)\n\n"
                        
                        trade_recommendation += f"**📊 CASH FLOW TIMELINE:**\n\n"
                        
                        trade_recommendation += f"**TODAY** (Initial Actions):\n"
                        trade_recommendation += f"• Buy {_fmt_qty_kg(qty_now)} (10% operational min): ${cost_phase1*usd_conversion_rate:,.0f}\n"
                        trade_recommendation += f"• Invest savings in bank @ {investment_rate*100:.1f}%: ${(investment_phase2+investment_phase3)*usd_conversion_rate:,.0f}\n"
                        trade_recommendation += f"  └─ Phase 2 budget: ${investment_phase2*usd_conversion_rate:,.0f}\n"
                        trade_recommendation += f"  └─ Phase 3 budget: ${investment_phase3*usd_conversion_rate:,.0f}\n\n"
                        
                        trade_recommendation += f"**{half_e.get('horizon', 'Mid-point')}** (Price falling):\n"
                        trade_recommendation += f"• Interest earned: +${interest_earned_phase2*usd_conversion_rate:,.0f}\n"
                        trade_recommendation += f"• Buy {_fmt_qty_kg(qty_mid)} (30%) at lower price: ${actual_cost_phase2*usd_conversion_rate:,.0f}\n\n"
                        
                        trade_recommendation += f"**{min_e['horizon']}** (Price at bottom):\n"
                        trade_recommendation += f"• Interest earned: +${interest_earned_phase3*usd_conversion_rate:,.0f}\n"
                        trade_recommendation += f"• Buy {_fmt_qty_kg(qty_target)} (60%) at lowest price: ${actual_cost_phase3*usd_conversion_rate:,.0f}\n"
                        trade_recommendation += f"• Pay storage costs: ${total_storage*usd_conversion_rate:,.0f}\n\n"
                        
                        trade_recommendation += f"**{'─'*60}\n"
                        trade_recommendation += f"**💰 NET PROFIT: ${expected_profit:,.0f} USD**\n"
                        trade_recommendation += f"{'─'*60}\n\n"
                        
                        strategy_details = f"**BREAKDOWN:**\n"
                        strategy_details += f"• If bought all now: ${baseline_cost_usd:,.0f}\n"
                        strategy_details += f"• Strategy cost: ${strategy_cost_usd:,.0f}\n"
                        strategy_details += f"• Price savings: +${price_savings_usd:,.0f}\n"
                        strategy_details += f"• Interest earned @ {investment_rate*100:.1f}%: +${interest_earned_usd:,.0f}\n"
                        strategy_details += f"• Storage costs @ {storage_cost_pct*100:.1f}%: -${storage_usd:,.0f}\n"
                        strategy_details += f"• **NET: ${expected_profit:,.0f} USD**\n\n"
                        strategy_details += f"**WHY THIS WORKS:**\n"
                        strategy_details += f"• Forecast predicts {abs(move_to_min):.1f}% price drop\n"
                        strategy_details += f"• Buy only 10% now (operations need)\n"
                        strategy_details += f"• Invest rest in bank, earn {investment_rate*100:.1f}% interest\n"
                        strategy_details += f"• Buy remaining 90% when prices fall\n"
                        strategy_details += f"• Profit = Price savings + Interest - Storage\n\n"
                        strategy_details += f"**LOGIC:**\n"
                        strategy_details += f"• Forecast shows price drop from {s0:,.{dec}f} to {float(min_e['price']):,.{dec}f}\n"
                        strategy_details += f"• Buy minimal {_fmt_qty_kg(qty_now)} now (operations)\n"
                        strategy_details += f"• Invest saved budget in bank @ {investment_rate*100:.1f}%\n"
                        strategy_details += f"• Buy remaining {_fmt_qty_kg(qty_mid + qty_target)} when prices fall\n"
                        strategy_details += f"• Earn interest while waiting + save on lower prices"
                        
                        steps = trade_recommendation + strategy_details
                    else:
                        steps = (
                            f"Procure 10% now (operational safety); 30% around {half_e.get('horizon','mid‑window')}; "
                            f"60% around {min_e['horizon']}. Overlay: small CALL (or CALL SPREAD) to cap upside while waiting."
                        )
                        if mqty > 0:
                            total_need = mqty * target_months
                            a, b, c3 = 0.10 * total_need, 0.30 * total_need, 0.60 * total_need
                            steps = f"{steps}  Sizing: ~{_fmt_qty_kg(mqty)}/month; total ~{_fmt_qty_kg(total_need)}; tranches ~{_fmt_qty_kg(a)}, {_fmt_qty_kg(b)}, {_fmt_qty_kg(c3)}."
                    
                    logic = f"Downward Price Expectation (model minimum at {min_e['horizon']})."
                    decision = "Defer procurement (forecast indicates lower levels ahead)"
                    when_txt = f"Target {min_e['horizon']} (~{min_e['months']}M)"
                    why_txt = f"Forecast low ahead ({move_to_min:.1f}% vs today)"
                    conf = _confidence_from_interval(
                        s0=s0,
                        target_price=float(min_e["price"]),
                        lower=min_e.get("lower"),
                        upper=min_e.get("upper"),
                    )
                elif move_to_max >= float(forecast_sig):
                    market_condition = "Forecast Opportunity"
                    strat_name = "Bullish Leveraged Carry Trade"
                    early_e = _closest_curve_entry(curve_s, 3) or mid_e
                    
                    # Calculate specific quantities and costs
                    target_months = min(6, float(max_e.get("months", 3)))
                    total_need = mqty * target_months if mqty > 0 else 0.0
                    
                    if mqty > 0 and total_need > 0:
                        # BULLISH LEVERAGED CARRY: Borrow money, buy now, sell later at high
                        # Buy 100% now using borrowed money, sell at peak
                        qty_now = 1.0 * total_need  # Buy all now
                        
                        # Unit conversion
                        unit_multiplier = 1.0
                        if "/lb" in unit.lower():
                            unit_multiplier = 2.20462
                        
                        import math
                        time_to_peak_years = target_months / 12.0
                        
                        # CASH FLOW TABLE (like team lead's example)
                        # TODAY: Borrow money and buy commodity
                        purchase_cost = qty_now * unit_multiplier * s0
                        loan_amount = purchase_cost
                        
                        # HOLD AND PAY STORAGE
                        storage_cost = purchase_cost * storage_cost_pct * time_to_peak_years
                        
                        # AT PEAK: Sell commodity
                        sale_proceeds = qty_now * unit_multiplier * float(max_e["price"])
                        
                        # REPAY LOAN WITH INTEREST
                        interest_cost = loan_amount * borrowing_rate * time_to_peak_years
                        loan_repayment = loan_amount + interest_cost
                        
                        # NET PROFIT (convert to USD for consistent reporting)
                        expected_profit_local = sale_proceeds - loan_repayment - storage_cost
                        profit_pct = (expected_profit_local / loan_amount * 100.0) if loan_amount > 0 else 0.0
                        
                        # Convert to USD if needed
                        expected_profit = expected_profit_local * usd_conversion_rate
                        purchase_cost_usd = purchase_cost * usd_conversion_rate
                        sale_proceeds_usd = sale_proceeds * usd_conversion_rate
                        interest_cost_usd = interest_cost * usd_conversion_rate
                        storage_cost_usd = storage_cost * usd_conversion_rate
                        
                        # BUILD CASH FLOW TABLE (Clean, readable format)
                        trade_recommendation = f"**🟢 BULLISH STRATEGY:** Borrow & Buy Now\n"
                        trade_recommendation += f"**Forecast:** Price rises from {s0:,.{dec}f} → {float(max_e['price']):,.{dec}f} (+{move_to_max:.1f}%)\n\n"
                        
                        trade_recommendation += f"**📊 CASH FLOW TIMELINE:**\n\n"
                        
                        trade_recommendation += f"**TODAY** (Initial Actions):\n"
                        trade_recommendation += f"• Borrow from bank @ {borrowing_rate*100:.1f}%: ${loan_amount*usd_conversion_rate:,.0f}\n"
                        trade_recommendation += f"• Buy {_fmt_qty_kg(qty_now)} (100% of need): ${purchase_cost_usd:,.0f}\n\n"
                        
                        trade_recommendation += f"**{max_e['horizon']}** (Price at peak):\n"
                        trade_recommendation += f"• Sell all commodity: +${sale_proceeds_usd:,.0f}\n"
                        trade_recommendation += f"• Repay loan + interest: ${loan_repayment*usd_conversion_rate:,.0f}\n"
                        trade_recommendation += f"  └─ Principal: ${loan_amount*usd_conversion_rate:,.0f}\n"
                        trade_recommendation += f"  └─ Interest: ${interest_cost_usd:,.0f}\n"
                        trade_recommendation += f"• Pay storage costs: ${storage_cost_usd:,.0f}\n\n"
                        
                        trade_recommendation += f"**{'─'*60}\n"
                        trade_recommendation += f"**💰 NET PROFIT: ${expected_profit:,.0f} USD ({profit_pct:.1f}% ROI)**\n"
                        trade_recommendation += f"{'─'*60}\n\n"
                        
                        strategy_details = f"**BREAKDOWN:**\n"
                        strategy_details += f"• Purchase cost: ${purchase_cost_usd:,.0f}\n"
                        strategy_details += f"• Sale proceeds: +${sale_proceeds_usd:,.0f}\n"
                        strategy_details += f"• Interest @ {borrowing_rate*100:.1f}%: -${interest_cost_usd:,.0f}\n"
                        strategy_details += f"• Storage @ {storage_cost_pct*100:.1f}%: -${storage_cost_usd:,.0f}\n"
                        strategy_details += f"• **NET: ${expected_profit:,.0f} USD**\n\n"
                        strategy_details += f"**WHY THIS WORKS:**\n"
                        strategy_details += f"• Forecast predicts +{move_to_max:.1f}% price rise\n"
                        strategy_details += f"• Borrow money, buy all NOW at low price\n"
                        strategy_details += f"• Hold until {max_e['horizon']}, sell at peak\n"
                        strategy_details += f"• Profit covers loan interest + storage\n\n"
                        
                        steps = trade_recommendation + strategy_details
                    else:
                        steps = (
                            f"Procure 40% now; 30% around {early_e.get('horizon','next window')}; "
                            f"30% lock via FORWARD or CALL SPREAD into {max_e['horizon']}."
                        )
                        if mqty > 0:
                            total_need = mqty * target_months
                            a, b, c3 = 0.40 * total_need, 0.30 * total_need, 0.30 * total_need
                            steps = f"{steps}  Sizing: ~{_fmt_qty_kg(mqty)}/month; cover ~{_fmt_qty_kg(total_need)} ({target_months:.0f}M); tranches ~{_fmt_qty_kg(a)}, {_fmt_qty_kg(b)}, {_fmt_qty_kg(c3)}."
                    logic = f"Upward Price Expectation (model maximum at {max_e['horizon']})."
                    decision = "Accelerate procurement (forecast indicates higher levels ahead)"
                    when_txt = "Initiate coverage now"
                    why_txt = f"Forecast rise ahead (+{move_to_max:.1f}% vs today)"
                    conf = _confidence_from_interval(
                        s0=s0,
                        target_price=float(max_e["price"]),
                        lower=max_e.get("lower"),
                        upper=max_e.get("upper"),
                    )
                else:
                    decision = "Monitor / staged execution"
                    when_txt = "Re-check monthly"
                    why_txt = "No strong forecast edge"
                    conf = _confidence_from_interval(
                        s0=s0,
                        target_price=float(mid_e.get("price", s0)),
                        lower=mid_e.get("lower"),
                        upper=mid_e.get("upper"),
                    )

                # Simple triggers using volatility (if available)
                triggers = ""
                try:
                    hist_df = p.get("history_df")
                    vcol = p.get("info", {}).get("value_col") or p.get("value_col") or "value"
                    sigma_ann = None
                    if isinstance(hist_df, pd.DataFrame) and vcol in hist_df.columns:
                        sigma_ann = float(_annualized_volatility_from_history(hist_df[vcol]))
                    if sigma_ann is not None and np.isfinite(sigma_ann) and sigma_ann > 0:
                        # 1-sigma monthly move as a practical trigger
                        import math

                        sig_m = float(sigma_ann) / math.sqrt(12.0)
                        up = sig_m * 100.0
                        dn = sig_m * 100.0
                        triggers = f"If Current Spot Market Price rises > +{up:.0f}% in a month: increase hedge/forward coverage. If Current Spot Market Price falls > -{dn:.0f}%: accelerate next tranche."
                except Exception:
                    triggers = ""

                outputs.append(
                    {
                        "Asset Type": "Commodity",
                        "Market Condition": market_condition,
                        "Strategy Role": "Speculative Timing Strategy",
                        "Strategy Name": strat_name,
                        "Decision": decision,
                        "When": when_txt,
                        "Why": why_txt,
                        "How": steps,
                        "Triggers": triggers,
                        "Trade Construction Steps": steps,
                        "Financial Logic": logic,
                        "Expected Driver of Profit": driver,
                        "Risk Notes": risk_notes,
                        "Priority": conf,
                        "Commodity": label,
                        "Current Spot Market Price": f"{s0:,.{dec}f}",
                        "Unit": unit,
                    }
                )

            # Hedging strategy (risk reduction) — uses existing options engine (Step 6)
            if "Hedging Strategy" in role_filter:
                try:
                    hist_df = p.get("history_df")
                    vcol = p.get("info", {}).get("value_col") or p.get("value_col") or "value"
                    sigma_ann = 0.25
                    if isinstance(hist_df, pd.DataFrame) and vcol in hist_df.columns:
                        sigma_ann = float(_annualized_volatility_from_history(hist_df[vcol]))

                    # Procurement hedge against upside risk -> use max forecast
                    t_years = float(max(1, int(max_e.get("months", 6))) / 12.0)
                    hedge_strat = _recommend_hedge_strategy(
                        exposure="Procurement (forward buying)",
                        s0=s0,
                        s_mean=float(max_e.get("price", s0)),
                        sigma_ann=float(sigma_ann),
                        t_years=t_years,
                        risk_profile="Balanced",
                        budget_priority="Moderate",
                        allow_selling=False,
                        qty=1.0,
                        unit=unit,
                    )
                    legs_txt = _format_legs_brief(strat=hedge_strat, dec=dec)

                    market_condition = "Efficient" if str(hedge_strat.get("title", "")).upper().startswith("HOLD") else "Forecast Opportunity"
                    decision = "Establish Hedge Position (cap procurement cost risk)"
                    when_txt = f"Before {max_e.get('horizon', 'next buy window')}"
                    why_txt = "Reduce upside price risk"
                    how_txt = f"Request from bank: {legs_txt}"

                    pc = _purchase_commodity_from_label(label)
                    if pc and pc in purchase_ctx:
                        mqty = float(purchase_ctx[pc].get("median_monthly_kg") or 0.0)
                        cv = float(purchase_ctx[pc].get("cv") or 0.0)
                        months_cover = float(min(6, int(max(2, round(float(max_e.get("months", 3) or 3))))))

                        # Smart hedge ratio: larger when priority is higher, smaller when monthly buying is noisy.
                        try:
                            move_to_max2 = (float(max_e.get("price", s0)) / s0 - 1.0) * 100.0
                        except Exception:
                            move_to_max2 = 0.0
                        base_ratio = 0.70 if (np.isfinite(move_to_max2) and move_to_max2 >= float(forecast_sig)) else 0.60
                        adj = max(0.75, 1.0 - min(max(cv, 0.0), 1.5) * 0.10)
                        hedge_qty = mqty * months_cover * base_ratio * adj
                        if np.isfinite(mqty) and mqty > 0 and np.isfinite(hedge_qty) and hedge_qty > 0:
                            how_txt = f"Hedge Ratio / Exposure Coverage: ~{_fmt_qty_kg(hedge_qty)} ({months_cover:.0f}M). {how_txt}"
                    outputs.append(
                        {
                            "Asset Type": "Commodity",
                            "Market Condition": market_condition,
                            "Strategy Role": "Hedging Strategy",
                            "Strategy Name": str(hedge_strat.get("title", "Hedge")),
                            "Decision": decision,
                            "When": when_txt,
                            "Why": why_txt,
                            "How": how_txt,
                            "Trade Construction Steps": f"Request from bank: {legs_txt}",
                            "Financial Logic": "Reduce procurement price risk (cap upside / manage volatility).",
                            "Expected Driver of Profit": "Risk reduction / budget certainty",
                            "Risk Notes": "Premium cost, basis risk, liquidity",
                            "Priority": "Moderate",
                            "Commodity": label,
                            "Current Spot Market Price": f"{s0:,.{dec}f}",
                            "Unit": unit,
                        }
                    )
                except Exception:
                    pass

        out_df = pd.DataFrame(outputs)
        if out_df.empty:
            st.info("No recommendations available (missing forecasts/inputs).")
            return

        # EXECUTIVE SUMMARY - Extract profit numbers from "How" field
        prio_rank = {"High": 0, "Moderate": 1, "Low": 2}
        exec_summary = []
        for _, row in out_df[out_df["Strategy Role"] == "Speculative Timing Strategy"].iterrows():
            how_text = str(row.get("How", ""))
            commodity = str(row.get("Commodity", ""))
            priority = str(row.get("Priority", "Moderate"))
            decision = str(row.get("Decision", ""))
            
            # Extract profit/savings from text
            profit = 0.0
            import re
            # Try new format first: "NET PROFIT +123,456 USD"
            match = re.search(r'NET PROFIT\s+\+?([\d,\.]+)', how_text)
            if not match:
                # Fallback to old format: "EXPECTED SAVINGS:** 123,456" or "EXPECTED PROFIT:** 123,456"
                match = re.search(r'(?:SAVINGS|PROFIT):\*\*\s*([\d,\.]+)', how_text)
            if match:
                profit = float(match.group(1).replace(',', ''))
            
            if profit > 0:
                exec_summary.append({
                    "commodity": commodity,
                    "profit": profit,
                    "priority": priority,
                    "decision": decision,
                    "how": how_text
                })
        
        exec_summary.sort(key=lambda x: (-prio_rank.get(x["priority"], 9), -x["profit"]))
        
        # Display Executive Summary
        if exec_summary:
            total_opportunity = sum(item["profit"] for item in exec_summary)
            urgent_count = sum(1 for item in exec_summary if item["priority"] == "Elevated")
            
            st.markdown(f"""
<div style='background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); 
            padding: 1.5rem; 
            border-radius: 12px; 
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
    <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;'>
        <div>
            <div style='color: #93c5fd; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;'>Executive Summary</div>
            <div style='color: white; font-size: 2.5rem; font-weight: 900; margin-top: 0.25rem;'>${total_opportunity:,.0f}</div>
            <div style='color: #dbeafe; font-size: 0.95rem; font-weight: 600;'>Total Profit Opportunity · {len(exec_summary)} Commodities · {urgent_count} Urgent</div>
        </div>
        <div style='text-align: right;'>
            <div style='background: rgba(255,255,255,0.15); padding: 0.75rem 1.25rem; border-radius: 8px; backdrop-filter: blur(10px);'>
                <div style='color: #dbeafe; font-size: 0.75rem; font-weight: 600;'>PERIOD</div>
                <div style='color: white; font-size: 1.1rem; font-weight: 800;'>Next 6-12 Months</div>
            </div>
        </div>
    </div>
</div>
            """, unsafe_allow_html=True)
            
            # Top 3 Urgent Actions
            urgent_items = [item for item in exec_summary if item["priority"] == "Elevated"][:3]
            if urgent_items:
                st.markdown("### 🔴 URGENT ACTIONS (Execute This Week)")
                
                for i, item in enumerate(urgent_items, 1):
                    # Extract key numbers from detailed trade recommendation
                    how_text = item["how"]
                    
                    # Determine strategy type and extract relevant info
                    if "BEARISH REVERSE CARRY" in how_text.upper():
                        # Bearish strategy: "TODAY Buy 5,415 t (10%) -8,841,353 USD"
                        buy_match = re.search(r'TODAY\s+Buy\s+([\d,\.]+\s*[tkgmton]+)\s*\((\d+)%\)', how_text)
                        action = "DEFER & INVEST"
                        action_color = "#3b82f6"
                        if buy_match:
                            phase1_qty = f"Buy {buy_match.group(1)} ({buy_match.group(2)}%)"
                        else:
                            phase1_qty = "Minimal buy"
                        phase1_price = "Invest rest"
                        strategy = "Defer & Invest (Bearish)"
                    elif "BULLISH LEVERAGED CARRY" in how_text.upper():
                        # Bullish strategy: "TODAY Borrow +123,456 USD + Buy 1,234,567 kg @ 1.23"
                        buy_match = re.search(r'Buy\s+([\d,\.]+\s*[tkgmton]+)\s*@\s*([\d\.]+)', how_text)
                        action = "BORROW & BUY NOW"
                        action_color = "#10b981"
                        phase1_qty = buy_match.group(1).strip() if buy_match else "Full quantity"
                        phase1_price = f"@ {buy_match.group(2)}" if buy_match else "—"
                        strategy = "Borrow & Buy (Bullish)"
                    else:
                        # Fallback for old format
                        phase1_match = re.search(r'Phase 1 \(NOW\):\*\* Buy ([\d,\.]+[^\@]*) @ ([\d\.]+)', how_text)
                        phase1_qty = phase1_match.group(1).strip() if phase1_match else "—"
                        phase1_price = phase1_match.group(2) if phase1_match else "—"
                        strategy_match = re.search(r'\*\*STRATEGY: ([^\n]+)', how_text)
                        strategy = strategy_match.group(1).replace('**', '') if strategy_match else "Phased Procurement"
                        action = "BUY NOW" if "Accelerate" in item["decision"] or "Buy" in how_text else "DEFER & WAIT"
                        action_color = "#10b981" if action == "BUY NOW" else "#3b82f6"
                    
                    st.markdown(f"""
<div style='background: linear-gradient(135deg, #fff 0%, #fef3c7 100%); 
            padding: 1rem 1.25rem; 
            border-left: 6px solid #dc2626;
            border-radius: 8px; 
            margin-bottom: 0.75rem;'>
    <div style='display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap;'>
        <div style='flex: 1; min-width: 200px;'>
            <div style='font-size: 0.7rem; color: #78350f; font-weight: 700; margin-bottom: 0.25rem;'>#{i} PRIORITY</div>
            <div style='font-size: 1.3rem; font-weight: 900; color: #0f172a;'>{item["commodity"]}</div>
        </div>
        <div style='flex: 1; min-width: 150px;'>
            <div style='font-size: 0.7rem; color: #78350f; font-weight: 700; margin-bottom: 0.25rem;'>ACTION</div>
            <div style='font-size: 1.1rem; font-weight: 900; color: {action_color};'>{action}</div>
            <div style='font-size: 0.75rem; color: #64748b; font-weight: 600;'>{phase1_qty} @ {phase1_price}</div>
        </div>
        <div style='flex: 1; min-width: 150px;'>
            <div style='font-size: 0.7rem; color: #78350f; font-weight: 700; margin-bottom: 0.25rem;'>EXPECTED PROFIT</div>
            <div style='font-size: 1.5rem; font-weight: 900; color: #059669;'>${item["profit"]:,.0f}</div>
        </div>
        <div style='flex: 1; min-width: 180px;'>
            <div style='font-size: 0.7rem; color: #78350f; font-weight: 700; margin-bottom: 0.25rem;'>STRATEGY</div>
            <div style='font-size: 0.85rem; font-weight: 800; color: #1e40af;'>{strategy[:40]}</div>
        </div>
    </div>
</div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")

        # New executive layout (cards)
        st.markdown("<div class='cp-kv-label' style='margin-top: 0.5rem;'>Detailed Trade Recommendations</div>", unsafe_allow_html=True)
        st.caption("Click on any commodity below to see full execution strategy with quantities, timing, and profit calculations.")

        spec_rows = (
            out_df[out_df["Strategy Role"] == "Speculative Timing Strategy"]
            .sort_values(by=["Priority"], ascending=True, kind="mergesort")
            .to_dict(orient="records")
        )
        hedge_rows = (
            out_df[out_df["Strategy Role"] == "Hedging Strategy"]
            .sort_values(by=["Priority"], ascending=True, kind="mergesort")
            .to_dict(orient="records")
        )

        # Re-rank priority order for display
        prio_rank = {"High": 0, "Moderate": 1, "Low": 2}
        spec_rows.sort(key=lambda r: prio_rank.get(str(r.get("Priority")), 9))
        hedge_rows.sort(key=lambda r: prio_rank.get(str(r.get("Priority")), 9))

        # Merge procurement execution + hedge structure into one universal output per commodity
        def _pick_best(rows: list[dict]) -> dict[str, dict]:
            best: dict[str, dict] = {}
            for r in rows:
                k = str(r.get("Commodity") or "").strip()
                if not k:
                    continue
                if k not in best:
                    best[k] = r
                    continue
                if prio_rank.get(str(r.get("Priority")), 9) < prio_rank.get(str(best[k].get("Priority")), 9):
                    best[k] = r
            return best

        spec_by = _pick_best(spec_rows)
        hedge_by = _pick_best(hedge_rows)
        merged_rows: list[dict] = []
        for comm in sorted(set(spec_by.keys()) | set(hedge_by.keys())):
            s = spec_by.get(comm)
            h = hedge_by.get(comm)
            if not s and not h:
                continue

            # Overall priority: best of either leg
            pr_s = prio_rank.get(str((s or {}).get("Priority")), 9)
            pr_h = prio_rank.get(str((h or {}).get("Priority")), 9)
            pr_best = min(pr_s, pr_h)
            priority = "High" if pr_best == 0 else ("Moderate" if pr_best == 1 else "Low")

            proc_dec = str((s or {}).get("Decision") or "—")
            proc_when = str((s or {}).get("When") or "—")
            proc_why = str((s or {}).get("Why") or "—")
            proc_how = str((s or {}).get("How") or "—")
            proc_trig = str((s or {}).get("Triggers") or "").strip()

            hedge_name = str((h or {}).get("Strategy Name") or (h or {}).get("Decision") or "—")
            hedge_when = str((h or {}).get("When") or "—")
            hedge_how = str((h or {}).get("How") or "—")

            merged_rows.append(
                {
                    "Commodity": comm,
                    "Decision": f"Structured Sourcing Schedule: {proc_dec} | Establish Hedge Position: {hedge_name}" if h else f"Structured Sourcing Schedule: {proc_dec}",
                    "When": f"Procurement Timing Strategy: {proc_when} · Establish Hedge Position: {hedge_when}" if h else f"Procurement Timing Strategy: {proc_when}",
                    "Why": proc_why if proc_why != "—" else str((h or {}).get("Why") or "—"),
                    "How": f"{proc_how}  Establish Hedge Position: {hedge_how}" if h and proc_how != "—" else (f"Establish Hedge Position: {hedge_how}" if h else proc_how),
                    "Triggers": proc_trig,
                    "Priority": priority,
                }
            )

        merged_rows.sort(key=lambda r: prio_rank.get(str(r.get("Priority")), 9))
        _render_cards(merged_rows[:12], empty_msg="No recommendations available.")

        # Optional internal diagnostics (no inputs)
        if debug_mode:
            with st.expander("Diagnostics (internal)", expanded=False):
                st.caption("Enable by setting STRATEGY_DEBUG=1 in Streamlit secrets/env.")

                ordered_cols = [
                    "Asset Type",
                    "Market Condition",
                    "Strategy Role",
                    "Strategy Name",
                    "Decision",
                    "When",
                    "Why",
                    "How",
                    "Trade Construction Steps",
                    "Financial Logic",
                    "Expected Driver of Profit",
                    "Risk Notes",
                    "Confidence Level",
                    "Commodity",
                    "Current Spot Market Price",
                    "Unit",
                ]
                cols = [c for c in ordered_cols if c in out_df.columns] + [c for c in out_df.columns if c not in ordered_cols]
                diag_df = out_df[cols].copy()

                def _role_style(v: str) -> str:
                    vv = str(v)
                    if "Hedging" in vv:
                        return "background-color:#064e3b; color:#dcfce7; font-weight:900;"
                    return "background-color:#0b1220; color:#e5e7eb; font-weight:900;"

                styled = (
                    diag_df.style
                    .applymap(_role_style, subset=["Strategy Role"])
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
                st.dataframe(styled, use_container_width=True, height=520)

        return


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
            "Paste/enter broker quotes in the editable columns (same unit as Current Spot Market Price). "
            "We also compute fair values (carry + Black‑Scholes using historical vol) to benchmark mispricing."
        )

        base_rows: list[dict] = []
        for c in candidates[:10]:
            p = c["payload"]
            scale = float(p.get("display_scale", 1.0) or 1.0)
            unit = str(p.get("display_currency") or p.get("info", {}).get("currency", ""))
            s0 = float(p.get("current_price") or 0.0) * scale

            best_h, _best_pred, months = _pick_best_horizon_for_payload(payload=p, exposure="Procurement (forward buying)")
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
                    "Current Spot Market Price": s0,
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
                    "Current Spot Market Price": st.column_config.NumberColumn(format="%.4f"),
                    "Futures Fair": st.column_config.NumberColumn(format="%.4f", help="Carry fair value (S·e^{(r+storage−cy)T})"),
                    "Futures Quote": st.column_config.NumberColumn(format="%.4f", help="Broker/market futures quote"),
                    "Strike (K)": st.column_config.NumberColumn(format="%.4f", help="Default = ATM (Current Spot Market Price). Adjust if quoting a different strike."),
                    "Call Fair": st.column_config.NumberColumn(format="%.4f", help="Black‑Scholes fair premium using historical vol"),
                    "Call Quote": st.column_config.NumberColumn(format="%.4f", help="Broker/market call premium"),
                    "Put Fair": st.column_config.NumberColumn(format="%.4f", help="Black‑Scholes fair premium using historical vol"),
                    "Put Quote": st.column_config.NumberColumn(format="%.4f", help="Broker/market put premium"),
                },
                disabled=["Commodity", "Target", "T (yrs)", "Current Spot Market Price", "Futures Fair", "Call Fair", "Put Fair", "Unit"],
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
                s0 = float(row.get("Current Spot Market Price", 0.0))
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
            confidence = "Moderate"

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
                    confidence = "Moderate"
                elif mis_pct < 0:
                    opportunity = "Carry Mispricing"
                    strategy = "Cash‑and‑Carry Arbitrage"
                    trade = "Short spot, invest cash, long futures"
                    signal = f"F_quote vs F_fair: {mis_pct:+.2f}%"
                    logic = "Forward underpriced vs carry fair value."
                    confidence = "High" if abs(mis_pct) >= 2 * float(threshold_pct) else "Moderate"
                else:
                    opportunity = "Carry Mispricing"
                    strategy = "Reverse Cash‑and‑Carry"
                    trade = "Long spot (financed), short futures"
                    signal = f"F_quote vs F_fair: {mis_pct:+.2f}%"
                    logic = "Forward overpriced vs carry fair value."
                    confidence = "High" if abs(mis_pct) >= 2 * float(threshold_pct) else "Moderate"

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
                        confidence = "High" if abs(gap_pct) >= 2 * parity_thr else "Moderate"

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
            out_df["__conf_rank"] = out_df["Confidence"].map({"High": 2, "Moderate": 1, "Low": 0}).fillna(1)
            out_df = out_df.sort_values(["__conf_rank", "Commodity"], ascending=[False, True]).drop(columns=["__conf_rank"])
        except Exception:
            pass

        def _conf_style(v: str) -> str:
            vv = str(v)
            if "High" in vv:
                return "background-color:#052e16; color:#dcfce7; font-weight:900;"
            if "Moderate" in vv:
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
            'action': 'ACCELERATE PROCUREMENT' if change > 5 else 'HOLD' if abs(change) < 3 else 'DEFER PROCUREMENT',
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
    colors = ['#3b82f6', '#10b981', '#8b5cf6', '#ef4444', '#ec4899']
    
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

    # Procurement & Options Strategist — moved here from Summary
    st.markdown("---")
    st.markdown("### 🧠 Procurement & Options Strategist")
    st.caption("Unified guidance: procurement execution schedule plus option structures using history and forecast distribution.")

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
        expander_title="Procurement & Options Strategist",
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
        
        # Purchasing Forecast Section
        st.markdown("---")
        st.markdown("### 📦 Quarterly Purchasing Forecast")
        st.caption("🔮 Predicted procurement volumes · Based on historical trends from mid-2024 onwards")
        
        try:
            purch_df = _load_purchase_monthly_agg()
            
            # Debug info
            if purch_df.empty:
                st.error("🔍 DEBUG: Purchase dataframe is EMPTY")
            else:
                st.success(f"✅ Loaded {len(purch_df)} purchase records with columns: {', '.join(purch_df.columns.tolist())}")
            
            if isinstance(purch_df, pd.DataFrame) and not purch_df.empty and "month" in purch_df.columns:
                qty_col = "total_qty_kg" if "total_qty_kg" in purch_df.columns else ("total_qty" if "total_qty" in purch_df.columns else None)
                
                if qty_col:
                    # Prepare data
                    purch_df = purch_df.copy()
                    purch_df["month"] = pd.to_datetime(purch_df["month"], errors="coerce")
                    purch_df = purch_df.dropna(subset=["month"])
                    purch_df[qty_col] = pd.to_numeric(purch_df[qty_col], errors="coerce")
                    purch_df = purch_df.dropna(subset=[qty_col])
                    
                    # Filter from mid-2024 onwards
                    start_date = pd.Timestamp("2024-07-01")
                    purch_df = purch_df[purch_df["month"] >= start_date].sort_values("month")
                    
                    if not purch_df.empty:
                        # Get today's date and calculate forecast end (1 year from now)
                        today = pd.Timestamp("2026-02-04")  # Current date from context
                        forecast_end = today + pd.DateOffset(years=1)
                        
                        # Aggregate by commodity and quarter
                        purch_df["quarter"] = purch_df["month"].dt.to_period("Q")
                        quarterly = purch_df.groupby(["commodity", "quarter"], dropna=False)[qty_col].sum().reset_index()
                        
                        # Pivot to get commodities as columns
                        pivot_df = quarterly.pivot(index="quarter", columns="commodity", values=qty_col).fillna(0)
                        
                        # Generate future quarters for forecast
                        last_quarter = pivot_df.index.max()
                        future_quarters = []
                        current_q = last_quarter
                        while current_q.end_time < forecast_end:
                            current_q = current_q + 1
                            future_quarters.append(current_q)
                        
                        # Simple forecast: use average of last 4 quarters for each commodity
                        forecast_rows = []
                        for q in future_quarters:
                            forecast_row = {}
                            for commodity in pivot_df.columns:
                                last_4_quarters = pivot_df[commodity].tail(4)
                                avg_qty = last_4_quarters.mean() if len(last_4_quarters) > 0 else 0
                                forecast_row[commodity] = avg_qty
                            forecast_rows.append(forecast_row)
                        
                        if forecast_rows:
                            forecast_df = pd.DataFrame(forecast_rows, index=future_quarters)
                            # Combine historical and forecast
                            combined_df = pd.concat([pivot_df, forecast_df])
                        else:
                            combined_df = pivot_df
                        
                        # Convert to tonnes and format
                        display_df = combined_df / 1000.0  # kg to tonnes
                        display_df.index = display_df.index.astype(str)
                        
                        # Create visualization
                        col1, col2 = st.columns([3, 2])
                        
                        with col1:
                            # Create stacked bar chart
                            import plotly.graph_objects as go
                            
                            fig = go.Figure()
                            colors_map = {
                                'Cotton': '#3b82f6',
                                'Polyester': '#10b981', 
                                'Viscose': '#8b5cf6',
                                'Crude Oil': '#ef4444',
                                'Natural Gas': '#ec4899'
                            }
                            
                            for commodity in display_df.columns:
                                fig.add_trace(go.Bar(
                                    name=commodity,
                                    x=display_df.index,
                                    y=display_df[commodity],
                                    marker_color=colors_map.get(commodity, '#64748b'),
                                    hovertemplate=f'<b>{commodity}</b><br>%{{y:,.0f}} tonnes<extra></extra>'
                                ))
                            
                            # Mark historical vs forecast
                            last_historical_idx = len(pivot_df) - 1
                            
                            fig.update_layout(
                                barmode='group',
                                title=dict(
                                    text='Quarterly Purchasing Volume (Tonnes)',
                                    font=dict(size=14, weight='bold')
                                ),
                                xaxis_title='Quarter',
                                yaxis_title='Volume (Tonnes)',
                                hovermode='x unified',
                                plot_bgcolor='#f8fafc',
                                paper_bgcolor='white',
                                height=400,
                                showlegend=True,
                                legend=dict(
                                    orientation='h',
                                    yanchor='bottom',
                                    y=1.02,
                                    xanchor='right',
                                    x=1
                                ),
                                shapes=[
                                    # Add vertical line separating historical from forecast
                                    dict(
                                        type='line',
                                        x0=last_historical_idx + 0.5,
                                        x1=last_historical_idx + 0.5,
                                        y0=0,
                                        yref='paper',
                                        y1=1,
                                        line=dict(color='#f59e0b', width=2, dash='dash')
                                    )
                                ],
                                annotations=[
                                    dict(
                                        x=last_historical_idx + 0.5,
                                        y=1.05,
                                        xref='x',
                                        yref='paper',
                                        text='← Historical | Forecast →',
                                        showarrow=False,
                                        font=dict(size=10, color='#f59e0b'),
                                        xanchor='center'
                                    )
                                ]
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, key="purchasing_forecast")
                        
                        with col2:
                            # Create summary table
                            summary_data = []
                            for commodity in display_df.columns:
                                historical_avg = display_df[commodity].iloc[:len(pivot_df)].mean()
                                forecast_avg = display_df[commodity].iloc[len(pivot_df):].mean() if len(forecast_rows) > 0 else 0
                                change_pct = ((forecast_avg / historical_avg - 1) * 100) if historical_avg > 0 else 0
                                
                                summary_data.append({
                                    'Commodity': commodity,
                                    'Avg Historical': f'{historical_avg:,.0f} t',
                                    'Avg Forecast': f'{forecast_avg:,.0f} t',
                                    'Change': f'{change_pct:+.1f}%'
                                })
                            
                            summary_df = pd.DataFrame(summary_data)
                            
                            # Style the table
                            def color_forecast_change(val):
                                if isinstance(val, str) and '%' in val:
                                    num = float(val.replace('%', '').replace('+', ''))
                                    if num > 5:
                                        return 'background-color: #dcfce7; color: #166534; font-weight: bold'
                                    elif num > 0:
                                        return 'background-color: #e0f2fe; color: #075985; font-weight: bold'
                                    elif num < -5:
                                        return 'background-color: #fee2e2; color: #991b1b; font-weight: bold'
                                    elif num < 0:
                                        return 'background-color: #fed7aa; color: #9a3412; font-weight: bold'
                                return ''
                            
                            styled_summary = summary_df.style.applymap(
                                color_forecast_change, subset=['Change']
                            ).set_properties(**{
                                'text-align': 'right',
                                'font-size': '0.9rem'
                            }, subset=['Avg Historical', 'Avg Forecast', 'Change']).set_properties(**{
                                'text-align': 'left',
                                'font-weight': 'bold',
                                'font-size': '0.9rem'
                            }, subset=['Commodity']).set_table_styles([
                                {'selector': 'thead th', 'props': [
                                    ('background-color', '#1e40af'),
                                    ('color', 'white'),
                                    ('font-weight', 'bold'),
                                    ('text-align', 'center'),
                                    ('padding', '10px'),
                                    ('font-size', '0.85rem')
                                ]},
                                {'selector': 'tbody tr:nth-child(even)', 'props': [
                                    ('background-color', '#f8fafc')
                                ]},
                                {'selector': 'tbody tr:hover', 'props': [
                                    ('background-color', '#e0e7ff')
                                ]}
                            ])
                            
                            st.dataframe(styled_summary, use_container_width=True, hide_index=True, height=240)
                            
                            st.caption(f"📊 Forecast based on {len(pivot_df)} quarters of historical data")
                            st.caption("🔮 Predictions use 4-quarter rolling average")
                    else:
                        st.info("Purchase data available from mid-2024 onwards. Waiting for sufficient history to generate forecasts.")
                else:
                    st.info("No quantity column found in purchase data. Check data format.")
            else:
                st.info("No purchase history available. Upload procurement data to enable forecasting.")
        except Exception as e:
            st.warning(f"Unable to generate purchasing forecast: {str(e)}")
        
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

    # Integrated strategist (Forecast + Carry/Parity + Risk filter)
    st.markdown("---")
    render_integrated_strategy_engine(
        expander_title="📌 Independent Strategist (Auto)",
        expanded=True,
        key_prefix="summary_institutional",
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
    st.info("Procurement & Options Strategist: Portfolio view is on Executive Summary; full strategist view is on AI Predictions.")


if __name__ == "__main__":
    main()
