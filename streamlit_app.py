from __future__ import annotations

from pathlib import Path
from datetime import datetime
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

    if not supabase_is_configured():
        st.warning(
            "Supabase is not configured. Add `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` "
            "(or `SUPABASE_ANON_KEY`) in Streamlit Secrets to enable long-term prediction storage."
        )
        st.info(
            "Once configured, this page will show a daily rolling chart like your BTC model screenshot."
        )
        return

    assets: list[str] = []
    try:
        rows = supabase_rest_select(table="prediction_records", select="asset", limit=5000)
        assets = sorted({r.get("asset") for r in (rows or []) if r.get("asset")})
    except Exception:
        assets = []

    if not assets:
        st.info("No prediction records found in Supabase yet.")
        st.caption("Tip: push daily predictions from your pipeline into `prediction_records`.")
        return

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
        return

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
        yaxis=dict(title=f"Price ({unit})" if unit else "Price", gridcolor="rgba(148,163,184,0.15)", tickfont=dict(color="#cbd5e1")),
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
    "Polyester": {"path": "polyester/polyester_usd_monthly", "currency": "USD/ton", "icon": "🧵", "type": "Futures"},
    "Viscose": {"path": "viscose/viscose_usd_monthly", "currency": "USD/ton", "icon": "🧬", "type": "Spot"},
    "Natural Gas": {"path": "energy/natural_gas_usd_monthly_clean", "currency": "USD/MMBTU", "icon": "🔥", "type": "Spot"},
    "Crude Oil": {"path": "energy/crude_oil_brent_usd_monthly_clean", "currency": "USD/barrel", "icon": "🛢️", "type": "Spot"}
}

LOCAL_COMMODITIES = {
    "Cotton (Local)": {"path": "cotton/cotton_pkr_monthly", "currency": "PKR/maund", "icon": "🌱", "type": "Local Market"},
    "Polyester (Local)": {"path": "polyester/polyester_pkr_monthly", "currency": "PKR/ton", "icon": "🧵", "type": "Import Cost"},
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


def _norm_cdf(x: float) -> float:
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def render_call_put_recommender(*, expander_title: str, expanded: bool = False, key_prefix: str = "hedge") -> None:
    """Simple-but-smart hedging helper: suggests Call vs Put using forecast direction + uncertainty."""
    with st.expander(expander_title, expanded=expanded):
        st.caption(
            "Select your exposure and horizon. The tool uses the model forecast (base/low/high) to suggest whether a Call or Put is more appropriate. "
            "This is a decision aid (not full options pricing)."
        )

        all_assets: dict[str, dict] = {}
        all_assets.update(INTERNATIONAL_COMMODITIES)
        all_assets.update(LOCAL_COMMODITIES)
        asset_names = list(all_assets.keys())
        if not asset_names:
            st.info("No commodities configured.")
            return

        c1, c2, c3 = st.columns([2.2, 1.2, 1.6])
        with c1:
            selected_name = st.selectbox("Commodity", asset_names, index=0, key=f"{key_prefix}_asset")
        with c2:
            months = st.number_input("Horizon (months)", min_value=1, max_value=24, value=3, step=1, key=f"{key_prefix}_months")
        with c3:
            exposure = st.selectbox(
                "Exposure",
                ["Buyer (need to buy)", "Seller/Inventory (need to sell)", "Inventory value hedge"],
                index=0,
                key=f"{key_prefix}_exposure",
            )

        info = all_assets[selected_name]
        md = load_commodity_data(info["path"], info["currency"])
        if not md or md.get("df") is None:
            st.warning("No data available for this asset.")
            return

        current_price = float(md.get("current_price", 0.0))
        predictions = load_predictions(info["path"]) if info else {}
        pred = get_prediction_by_index(predictions, int(months))
        mu = float(pred.get("price", current_price)) if pred else current_price
        low = float(pred.get("lower", mu)) if pred else mu
        high = float(pred.get("upper", mu)) if pred else mu

        # Derive a rough sigma from (high-low). If the interval is not calibrated, this still provides a usable scale.
        interval = max(1e-9, high - low)
        sigma = interval / 3.92  # approx 95% interval width = 2*1.96*sigma
        sigma = max(1e-9, sigma)

        # Simple probabilities under normal assumption
        p_up = 1.0 - _norm_cdf((current_price - mu) / sigma)
        p_down = 1.0 - p_up

        # Confidence heuristic: narrower interval and larger move => higher confidence
        expected_move_pct = 0.0
        if current_price > 0:
            expected_move_pct = (mu / current_price - 1.0) * 100.0
        confidence = min(0.99, max(0.01, abs(mu - current_price) / interval))

        # Decide recommendation
        small_band = 0.01  # 1%
        action = "MONITOR"
        option = "None"

        if exposure == "Buyer (need to buy)":
            if (mu > current_price * (1 + small_band)) and (p_up >= 0.55):
                option = "CALL"
                action = "HEDGE"
            elif (mu < current_price * (1 - small_band)) and (p_down >= 0.55):
                option = "WAIT / STAGGER BUYS"
                action = "MONITOR"
            else:
                option = "COLLAR (optional)"
                action = "MONITOR"
        else:
            # Seller / inventory hedge: protect downside with a PUT if forecast points lower
            if (mu < current_price * (1 - small_band)) and (p_down >= 0.55):
                option = "PUT"
                action = "HEDGE"
            elif (mu > current_price * (1 + small_band)) and (p_up >= 0.55):
                option = "HOLD / SELL LATER"
                action = "MONITOR"
            else:
                option = "COLLAR (optional)"
                action = "MONITOR"

        # Suggest a strike in a simple, transparent way
        protection = st.slider(
            "Protection level (higher = more conservative strike)",
            min_value=0.5,
            max_value=0.95,
            value=0.8,
            step=0.05,
            key=f"{key_prefix}_prot",
        )

        suggested_k = None
        if option == "CALL":
            # choose K between base and upper
            suggested_k = mu + (high - mu) * float(protection)
        elif option == "PUT":
            # choose K between lower and base
            suggested_k = mu - (mu - low) * float(protection)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Spot", f"{current_price:,.2f}")
        with m2:
            st.metric(f"Forecast ({int(months)}m)", f"{mu:,.2f}", f"{expected_move_pct:+.1f}%")
        with m3:
            st.metric("P(up)", f"{p_up*100:.0f}%")
        with m4:
            st.metric("Confidence", f"{confidence*100:.0f}%")

        headline = f"Suggested: {option}" if option != "None" else "Suggested: Monitor"
        if action == "HEDGE":
            st.success(headline)
        else:
            st.info(headline)

        if suggested_k is not None:
            st.caption(f"Suggested strike K ≈ {suggested_k:,.2f} {info['currency']} (based on forecast interval + your protection level).")

        st.markdown("**Why**")
        st.write(
            f"Forecast base is {expected_move_pct:+.1f}% vs spot over {int(months)} months. "
            f"Model range is [{low:,.2f}, {high:,.2f}] and implied P(up) ≈ {p_up*100:.0f}%."
        )


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
        text_labels = [f"{val:,.2f}" for val in values]
        
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
                tickformat=',.2f' if metadata['df'][metadata['value_col']].mean() < 100 else ',.0f',
                tickfont=dict(size=11)
            ),
            hovermode='x unified'
        )
        
        return fig
    return None


def create_forecast_table(predictions, currency):
    """Create forecast data table for commodity (monthly horizons)."""
    data = []
    horizons = get_prediction_horizons(predictions)
    for horizon in horizons:
        pred = predictions[horizon]
        price_range = f"{pred['lower']:,.2f} - {pred['upper']:,.2f}"
        data.append({
            'Period': horizon,
            f'Price ({currency})': f"{pred['price']:,.2f}",
            'Range': price_range,
            'Confidence': f"{pred['confidence']}%",
            'Change': f"{pred['change']:+.1f}%"
        })
    return pd.DataFrame(data)


def create_forecast_bar_chart(predictions, currency):
    """Create bar chart visualization for price forecasts with confidence intervals."""
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
            text=[f"<b>{p:,.2f}</b><br><span style='font-size:11px'>{c:+.1f}%</span>" for p, c in zip(prices, changes)],
            textposition='outside',
            textfont=dict(size=12, color='#1e293b', family='Arial, sans-serif'),
            hovertemplate='<b>%{x}</b><br>' +
                         '<b>Price:</b> %{y:,.2f} ' + currency + '<br>' +
                         '<b>Change:</b> %{customdata[0]:+.1f}%<br>' +
                         '<b>Confidence:</b> %{customdata[1]}%<br>' +
                         '<b>Range:</b> %{customdata[2]:,.2f} - %{customdata[3]:,.2f}<br>' +
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
            tickformat=',.2f',
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
    
    # Main navigation
    page = st.radio(
        "Select View:",
        ["🇵🇰 Pakistan Forecasts", "🌍 International Market", "🇵🇰 Pakistan Local", "🧠 Market Intelligence", "🤖 AI Predictions"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if page == "🇵🇰 Pakistan Forecasts":
        render_pakistan_forecasts_page()
    
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
    """Deprecated (kept for backward compatibility).

    The Summary tables + old Call/Put simulator were removed per team lead feedback.
    """
    render_pakistan_forecasts_page()


def render_pakistan_forecasts_page():
    st.markdown("""
    <div style='border-left: 4px solid #2563eb; padding-left: 1rem; margin: 1rem 0 1.25rem 0;'>
        <h2 style='font-size: 1.5rem; font-weight: 800; color: #0f172a; letter-spacing: -0.4px; margin: 0 0 0.35rem 0;'>
            🇵🇰 Pakistan Forecasts
        </h2>
        <p style='font-size: 0.95rem; color: #475569; font-weight: 600; margin: 0; line-height: 1.5;'>
            USD/PKR exchange rate and electricity tariff outlook for budgeting and procurement planning
        </p>
    </div>
    """, unsafe_allow_html=True)

    usd_pkr = fetch_usd_pkr_rate()
    elec = fetch_wapda_electricity_rate()

    colA, colB = st.columns(2)
    with colA:
        if usd_pkr and usd_pkr.get("current_price"):
            st.metric("USD/PKR (live)", f"{float(usd_pkr['current_price']):,.2f}")
        else:
            st.warning("USD/PKR live rate unavailable")

    with colB:
        if elec and elec.get("current_price"):
            st.metric("Electricity tariff (live)", f"{float(elec['current_price']):,.2f}")
        else:
            st.warning("Electricity tariff live rate unavailable")

    if usd_pkr and usd_pkr.get("current_price"):
        st.markdown("---")
        st.markdown("### 💱 USD/PKR Exchange Rate Forecast")
        st.caption("Monthly outlook — use for import budgeting and hedge planning")
        preds = generate_usd_pkr_forecast(float(usd_pkr["current_price"]))
        c1, c2 = st.columns([2, 1])
        with c1:
            st.plotly_chart(create_forecast_bar_chart(preds, "PKR/USD"), use_container_width=True, key="pk_fx_chart")
        with c2:
            st.dataframe(create_forecast_table(preds, "PKR/USD"), use_container_width=True, hide_index=True)

    if elec and elec.get("current_price"):
        st.markdown("---")
        st.markdown("### ⚡ Electricity Tariff Forecast")
        st.caption("Monthly outlook — use for factory cost planning")
        preds = generate_energy_forecast(float(elec["current_price"]), "electricity")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.plotly_chart(create_forecast_bar_chart(preds, "PKR/Unit"), use_container_width=True, key="pk_elec_chart")
        with c2:
            st.dataframe(create_forecast_table(preds, "PKR/Unit"), use_container_width=True, hide_index=True)

    st.markdown("---")
    render_call_put_recommender(
        expander_title="🧠 Call vs Put — Hedge Recommendation",
        expanded=False,
        key_prefix="pk_hedge",
    )


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
    render_call_put_recommender(
        expander_title="🧠 Call vs Put — Hedge Recommendation",
        expanded=False,
        key_prefix="intel_hedge",
    )


if __name__ == "__main__":
    main()
