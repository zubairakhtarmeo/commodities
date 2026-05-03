"""
Phase 6: Improved ML Forecasting Pipeline

Changes from previous version:
- Rich feature engineering: lags, rolling stats, momentum, trend, seasonality
- Recursive multi-step forecasting (non-flat outputs)
- Three models: baseline, Ridge, RandomForest
- PKR predictions generated from USD × live FX rate
- No complex abstractions — pure pandas + sklearn
"""
import os
import sys
import warnings
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, date, timezone
from pathlib import Path
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            LOG_DIR / f"forecasts_{datetime.today().strftime('%Y%m%d')}.log",
            encoding="utf-8",
        ),
    ],
)

sys.path.insert(0, str(BASE_DIR / "src"))

print("ENV CHECK:")
print("SUPABASE_URL:", bool(os.getenv("SUPABASE_URL")))
print("SUPABASE_KEY:", bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY")))
print("FRED_API_KEY:", bool(os.getenv("FRED_API_KEY")))

try:
    import toml
except ImportError:
    toml = None

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


# ── Config ────────────────────────────────────────────────────────────────────

COMMODITIES_USD = [
    "cotton_usd",
    "crude_oil_usd",
    "natural_gas_usd",
    "polyester_usd",
]

HORIZONS = [1, 3, 6]
MIN_ROWS = 24  # months of history required to train

FEATURE_COLS = [
    "lag_1", "lag_3", "lag_6",
    "rolling_mean_3", "rolling_mean_6", "rolling_std_3",
    "pct_change_1", "pct_change_3",
    "trend_3",
    "month", "quarter",
]


# ── Supabase ──────────────────────────────────────────────────────────────────

def get_supabase_credentials():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        try:
            if toml:
                secrets = toml.load(BASE_DIR / ".streamlit/secrets.toml")
                url = url or secrets.get("SUPABASE_URL")
                key = key or secrets.get("SUPABASE_SERVICE_ROLE_KEY")
        except Exception:
            pass
    return url, key


def _get_supabase_client():
    url, key = get_supabase_credentials()
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception as e:
        logging.warning(f"⚠️ Could not create Supabase client: {e}")
        return None


# ── FX rate ───────────────────────────────────────────────────────────────────

def get_fx_rate(currency: str = "PKR") -> float:
    """Fetch live USD → currency rate. Falls back to last-known value."""
    try:
        r = requests.get("https://open.er-api.com/v6/latest/USD", timeout=10)
        rate = float(r.json()["rates"][currency])
        logging.info(f"  FX: 1 USD = {rate:.2f} {currency}")
        return rate
    except Exception:
        fallback = 278.5 if currency == "PKR" else 7.25
        logging.warning(f"  ⚠️ FX fetch failed — using fallback {fallback} {currency}/USD")
        return fallback


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_series(supabase, commodity: str) -> pd.Series | None:
    """Fetch monthly price series from commodity_prices table."""
    try:
        res = (
            supabase.table("commodity_prices")
            .select("date,value")
            .eq("commodity", commodity)
            .order("date")
            .execute()
        )
        if not res.data:
            return None
        df = pd.DataFrame(res.data)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna().set_index("date").sort_index()
        series = df["value"].resample("MS").mean().dropna()
        return series
    except Exception as e:
        logging.warning(f"  ⚠️ Fetch error for {commodity}: {e}")
        return None


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(series: pd.Series) -> pd.DataFrame:
    """
    Build a feature matrix from a monthly price series.

    All lag/rolling features are shifted by 1 so they use only
    information available *before* the target month (no leakage).
    """
    df = pd.DataFrame({"value": series})

    # Lags (of the raw price, shifted so row t uses t-1, t-3, t-6)
    df["lag_1"] = df["value"].shift(1)
    df["lag_3"] = df["value"].shift(3)
    df["lag_6"] = df["value"].shift(6)

    # Rolling statistics on already-shifted lag_1 window
    df["rolling_mean_3"] = df["lag_1"].rolling(3).mean()
    df["rolling_mean_6"] = df["lag_1"].rolling(6).mean()
    df["rolling_std_3"]  = df["lag_1"].rolling(3).std()

    # Momentum: % change over 1 and 3 months (lagged by 1)
    shifted = df["value"].shift(1)
    df["pct_change_1"] = shifted.pct_change(1)
    df["pct_change_3"] = shifted.pct_change(3)

    # Trend: absolute change over 3 months (lagged)
    df["trend_3"] = shifted - shifted.shift(3)

    # Seasonality
    df["month"]   = df.index.month
    df["quarter"] = df.index.quarter

    df = df.dropna()
    return df


# ── Recursive multi-step forecasting ─────────────────────────────────────────

def recursive_forecast(
    model,
    scaler: StandardScaler | None,
    series: pd.Series,
    horizons: list[int],
    as_of: pd.Timestamp,
) -> dict[int, float]:
    """
    Step-by-step forecasting: each predicted value feeds back into the next step.
    Avoids the flat-output problem caused by predicting all horizons from the
    same static feature vector.

    Returns {horizon_months: predicted_value}.
    """
    # Seed the rolling window with the last 12 known observations
    window = list(series.values[-12:])
    predictions: dict[int, float] = {}
    current = as_of

    for step in range(1, max(horizons) + 1):
        next_dt = current + relativedelta(months=1)

        w = np.array(window)
        n = len(w)

        # Compute features from the rolling window
        lag_1 = w[-1]
        lag_3 = w[-3] if n >= 3 else w[0]
        lag_6 = w[-6] if n >= 6 else w[0]

        roll3 = float(np.mean(w[-3:])) if n >= 3 else lag_1
        roll6 = float(np.mean(w[-6:])) if n >= 6 else lag_1
        std3  = float(np.std(w[-3:], ddof=1)) if n >= 3 else 0.0

        prev1 = w[-2] if n >= 2 else lag_1
        prev3 = w[-4] if n >= 4 else lag_1
        pct1  = (lag_1 - prev1) / prev1 if prev1 != 0 else 0.0
        pct3  = (lag_1 - prev3) / prev3 if prev3 != 0 else 0.0
        tr3   = lag_1 - prev3

        month   = next_dt.month
        quarter = (next_dt.month - 1) // 3 + 1

        x_new = np.array([[lag_1, lag_3, lag_6,
                           roll3, roll6, std3,
                           pct1, pct3, tr3,
                           month, quarter]])

        if scaler is not None:
            x_new = scaler.transform(x_new)

        pred = float(model.predict(x_new)[0])
        pred = max(pred, 0.0)  # prices cannot be negative

        if step in horizons:
            predictions[step] = pred

        window.append(pred)
        current = next_dt

    return predictions


# ── Model validation ─────────────────────────────────────────────────────────

def _build_feature_vector(window: list[float], target_dt: pd.Timestamp) -> np.ndarray:
    """Build a single feature row from a rolling window + target date."""
    w = np.array(window)
    n = len(w)
    lag_1 = w[-1]
    lag_3 = w[-3] if n >= 3 else w[0]
    lag_6 = w[-6] if n >= 6 else w[0]
    roll3 = float(np.mean(w[-3:])) if n >= 3 else lag_1
    roll6 = float(np.mean(w[-6:])) if n >= 6 else lag_1
    std3  = float(np.std(w[-3:], ddof=1)) if n >= 3 else 0.0
    prev1 = w[-2] if n >= 2 else lag_1
    prev3 = w[-4] if n >= 4 else lag_1
    pct1  = (lag_1 - prev1) / prev1 if prev1 != 0 else 0.0
    pct3  = (lag_1 - prev3) / prev3 if prev3 != 0 else 0.0
    tr3   = lag_1 - prev3
    return np.array([[lag_1, lag_3, lag_6,
                      roll3, roll6, std3,
                      pct1, pct3, tr3,
                      target_dt.month, (target_dt.month - 1) // 3 + 1]])


def walk_forward_mae(
    model,
    scaler: StandardScaler | None,
    train_series: pd.Series,
    val_series: pd.Series,
    is_baseline: bool = False,
) -> float:
    """
    Walk-forward 1-step MAE on val_series.

    At each validation step we use the actual observed value (not the
    prediction) as the next window entry — this is standard held-out
    evaluation, not recursive extrapolation.
    """
    window = list(train_series.values)
    if len(window) < 6:
        return float("inf")

    errors = []
    for dt, actual in val_series.items():
        if is_baseline:
            pred = window[-1]  # last known value
        else:
            x = _build_feature_vector(window, dt)
            if scaler is not None:
                x = scaler.transform(x)
            pred = max(0.0, float(model.predict(x)[0]))

        errors.append(abs(pred - actual))
        window.append(actual)  # advance with real observation

    return float(np.mean(errors)) if errors else float("inf")


# ── Push to Supabase ──────────────────────────────────────────────────────────

def push_predictions(supabase, records: list[dict]) -> bool:
    try:
        supabase.table("prediction_records").upsert(records).execute()
        return True
    except Exception as e:
        logging.warning(f"  ✗ Supabase push failed: {e}")
        return False


# ── Per-commodity forecasting ─────────────────────────────────────────────────

VAL_MONTHS = 6  # hold-out window for model selection


def forecast_commodity(
    supabase,
    commodity: str,
    as_of: pd.Timestamp,
    pkr_rate: float,
) -> str:
    """
    Select the best model via walk-forward MAE on the last VAL_MONTHS months,
    retrain on the full series, then push only the winner's predictions.
    """
    logging.info(f"\n📊 {commodity}")

    series = fetch_series(supabase, commodity)
    if series is None or len(series) < MIN_ROWS:
        n = len(series) if series is not None else 0
        msg = f"⚠️ skipped ({n} rows, need {MIN_ROWS})"
        logging.info(f"  {msg}")
        return msg

    # ── Split train / validation ──────────────────────────────────────────────
    train_series = series.iloc[:-VAL_MONTHS]
    val_series   = series.iloc[-VAL_MONTHS:]

    df_train = build_features(train_series)
    if len(df_train) < 8:
        msg = f"⚠️ skipped (only {len(df_train)} train samples after feature build)"
        logging.info(f"  {msg}")
        return msg

    X_train = df_train[FEATURE_COLS].values
    y_train = df_train["value"].values

    model_specs = [
        ("baseline_last_value", None,                                                False),
        ("linear_ridge",        Ridge(alpha=1.0),                                    True),
        ("random_forest",       RandomForestRegressor(n_estimators=100, max_depth=5,
                                                      random_state=42),              False),
    ]

    # ── Evaluate each model on validation set ─────────────────────────────────
    maes: dict[str, float] = {}
    for model_name, model, use_scaler in model_specs:
        try:
            if model_name == "baseline_last_value":
                mae = walk_forward_mae(None, None, train_series, val_series, is_baseline=True)
            else:
                scaler = StandardScaler() if use_scaler else None
                X_fit  = scaler.fit_transform(X_train) if scaler else X_train
                model.fit(X_fit, y_train)
                mae = walk_forward_mae(model, scaler, train_series, val_series)
            maes[model_name] = mae
            logging.info(f"  MAE  {model_name}: {mae:.4f}")
        except Exception as e:
            logging.warning(f"  ✗ {model_name} eval failed: {e}")
            maes[model_name] = float("inf")

    if not maes or all(v == float("inf") for v in maes.values()):
        return "⚠️ all models failed evaluation"

    best_name = min(maes, key=lambda k: maes[k])
    logging.info(f"  ★  best model: {best_name} (MAE={maes[best_name]:.4f})")

    # ── Retrain winner on full series ─────────────────────────────────────────
    df_full = build_features(series)
    X_full  = df_full[FEATURE_COLS].values
    y_full  = df_full["value"].values

    if best_name == "baseline_last_value":
        preds_usd = {h: float(series.iloc[-1]) for h in HORIZONS}
    else:
        # Reconstruct a fresh model instance (avoids contamination from val run)
        best_spec = {s[0]: (s[1], s[2]) for s in model_specs}
        fresh_model, use_scaler = best_spec[best_name]
        scaler = StandardScaler() if use_scaler else None
        X_fit  = scaler.fit_transform(X_full) if scaler else X_full
        fresh_model.fit(X_fit, y_full)
        preds_usd = recursive_forecast(fresh_model, scaler, series, HORIZONS, as_of)

    logging.info(
        f"  ✓ {best_name} (full retrain): "
        + "  ".join(f"h{h}={preds_usd[h]:.4f}" for h in HORIZONS)
    )

    # ── Build records ─────────────────────────────────────────────────────────
    hist_std     = float(series.std())
    created_at   = datetime.now(timezone.utc).isoformat()
    pkr_commodity = commodity.replace("_usd", "_pkr")
    records: list[dict] = []

    for h in HORIZONS:
        val      = preds_usd[h]
        variance = hist_std * 0.5 * (1 + h / 12)
        target   = (as_of + relativedelta(months=h)).strftime("%Y-%m-%d")

        records.append({
            "commodity":       commodity,
            "model_name":      best_name,
            "as_of_date":      as_of.strftime("%Y-%m-%d"),
            "horizon_months":  h,
            "target_date":     target,
            "predicted_value": round(val, 6),
            "lower_bound":     round(max(0.0, val - variance), 6),
            "upper_bound":     round(val + variance, 6),
            "unit":            "USD",
            "is_demo":         False,
            "created_at":      created_at,
        })

        val_pkr = val * pkr_rate
        var_pkr = variance * pkr_rate
        records.append({
            "commodity":       pkr_commodity,
            "model_name":      best_name,
            "as_of_date":      as_of.strftime("%Y-%m-%d"),
            "horizon_months":  h,
            "target_date":     target,
            "predicted_value": round(val_pkr, 2),
            "lower_bound":     round(max(0.0, val_pkr - var_pkr), 2),
            "upper_bound":     round(val_pkr + var_pkr, 2),
            "unit":            "PKR",
            "is_demo":         False,
            "created_at":      created_at,
        })

    ok = push_predictions(supabase, records)
    status = f"✅ {len(records)} records pushed ({best_name})" if ok else "❌ push failed"
    logging.info(f"  {status}")
    return status


# ── Entry point ───────────────────────────────────────────────────────────────

def run_ml_forecasts():
    logging.info("=" * 50)
    logging.info("🚀 STARTING ML FORECASTING (PHASE 6 — IMPROVED)")
    logging.info("=" * 50)

    supabase = _get_supabase_client()
    if supabase is None:
        logging.error("❌ No Supabase connection — cannot run forecasts.")
        print("\n=== PIPELINE STATUS ===")
        print("Ingestion completed")
        print("Forecast completed (skipped — no Supabase connection)")
        return

    as_of    = pd.Timestamp(date.today().replace(day=1))
    pkr_rate = get_fx_rate("PKR")
    logging.info(f"As-of: {as_of.date()}  |  FX: {pkr_rate:.2f} PKR/USD")

    results: dict[str, str] = {}
    for commodity in COMMODITIES_USD:
        try:
            results[commodity] = forecast_commodity(supabase, commodity, as_of, pkr_rate)
        except Exception as e:
            logging.error(f"  ❌ Unexpected error for {commodity}: {e}")
            results[commodity] = f"❌ error: {e}"

    logging.info("\n📊 viscose_usd: skipping (no data — manual upload required)")
    results["viscose_usd"] = "⚠️ skipped (manual)"

    logging.info("\n=== FORECAST SUMMARY ===")
    for k, v in results.items():
        logging.info(f"  {k}: {v}")

    logging.info("\n✅ ML Forecasting complete.")
    print("\n=== PIPELINE STATUS ===")
    print("Ingestion completed")
    print("Forecast completed")


if __name__ == "__main__":
    run_ml_forecasts()
