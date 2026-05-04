"""
Phase 6 (enhanced): Multi-signal, economically meaningful forecasting

Improvements over previous version:
- Extended features: rolling_std_6, rolling_mean_12, momentum_strength,
  lag×pct and lag×std interactions, cyclic month (sin/cos)
- Cross-commodity external signals: crude_oil affects natural_gas & polyester
- Baseline anti-domination: 5% handicap makes baseline harder to win
- Flat-output rejection: rejects any model whose h1/h3/h6 vary < 1%
- Economic sanity: clips predictions to ±50% of last known price
- Richer logging per commodity
- Lightweight viscose validation (safe skip if unavailable)
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
LOG_DIR  = BASE_DIR / "logs"
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

HORIZONS     = [1, 3, 6]
MIN_ROWS     = 24
VAL_MONTHS   = 6
BASELINE_PENALTY  = 0.05   # baseline must beat best ML by >5% to be selected
FLAT_THRESHOLD    = 1.0    # % variation across horizons below which output is "flat"
MAX_JUMP_PCT      = 50.0   # clamp predictions to ±50% of last known price

# External signals injected as features per commodity
COMMODITY_EXTERNALS: dict[str, list[str]] = {
    "cotton_usd":      [],
    "crude_oil_usd":   [],
    "natural_gas_usd": ["crude_oil_usd"],
    "polyester_usd":   ["crude_oil_usd", "natural_gas_usd"],
}

# Base feature columns (same for all commodities)
BASE_FEATURE_COLS = [
    "lag_1", "lag_3", "lag_6",
    "rolling_mean_3", "rolling_mean_6", "rolling_mean_12",
    "rolling_std_3", "rolling_std_6",
    "pct_change_1", "pct_change_3",
    "price_momentum_strength",
    "trend_3",
    "lag1_x_pct1", "lag1_x_std3",
    "month_sin", "month_cos",
    "month", "quarter",
]


def get_feature_cols(commodity: str) -> list[str]:
    ext_cols = [f"ext_{e}_lag1" for e in COMMODITY_EXTERNALS.get(commodity, [])]
    return BASE_FEATURE_COLS + ext_cols


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
        df["date"]  = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna().set_index("date").sort_index()
        return df["value"].resample("MS").mean().dropna()
    except Exception as e:
        logging.warning(f"  ⚠️ Fetch error for {commodity}: {e}")
        return None


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(
    series: pd.Series,
    externals: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """
    Build a feature matrix.  All features shifted by 1 to prevent leakage.
    externals: dict of {commodity_key: aligned monthly series} for cross-signals.
    """
    df = pd.DataFrame({"value": series})

    # ── Lags ──────────────────────────────────────────────────────────────────
    df["lag_1"] = df["value"].shift(1)
    df["lag_3"] = df["value"].shift(3)
    df["lag_6"] = df["value"].shift(6)

    # ── Rolling stats (on lag_1 window — no leakage) ──────────────────────────
    df["rolling_mean_3"]  = df["lag_1"].rolling(3).mean()
    df["rolling_mean_6"]  = df["lag_1"].rolling(6).mean()
    df["rolling_mean_12"] = df["lag_1"].rolling(12).mean()
    df["rolling_std_3"]   = df["lag_1"].rolling(3).std()
    df["rolling_std_6"]   = df["lag_1"].rolling(6).std()

    # ── Momentum ──────────────────────────────────────────────────────────────
    shifted = df["value"].shift(1)
    df["pct_change_1"]           = shifted.pct_change(1)
    df["pct_change_3"]           = shifted.pct_change(3)
    df["price_momentum_strength"] = df["pct_change_3"].abs()

    # ── Trend ─────────────────────────────────────────────────────────────────
    df["trend_3"] = shifted - shifted.shift(3)

    # ── Interactions ──────────────────────────────────────────────────────────
    df["lag1_x_pct1"] = df["lag_1"] * df["pct_change_1"]
    df["lag1_x_std3"] = df["lag_1"] * df["rolling_std_3"]

    # ── Cyclic month encoding ─────────────────────────────────────────────────
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    df["month"]     = df.index.month
    df["quarter"]   = df.index.quarter

    # ── External cross-commodity signals ─────────────────────────────────────
    if externals:
        for ext_key, ext_series in externals.items():
            col = f"ext_{ext_key}_lag1"
            aligned = ext_series.reindex(df.index).ffill().bfill()
            df[col] = aligned.shift(1)

    df = df.dropna()
    return df


# ── Feature vector builder (for walk-forward + recursive steps) ───────────────

def _build_feature_vector(
    window: list[float],
    target_dt: pd.Timestamp,
    feature_cols: list[str],
    external_vals: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Compute one feature row from a rolling price window + date.
    external_vals: {col_name: value} for cross-commodity signals.
    Must produce columns in the same order as feature_cols.
    """
    w = np.array(window)
    n = len(w)

    lag_1 = w[-1]
    lag_3 = w[-3] if n >= 3 else w[0]
    lag_6 = w[-6] if n >= 6 else w[0]

    roll3  = float(np.mean(w[-3:]))  if n >= 3  else lag_1
    roll6  = float(np.mean(w[-6:]))  if n >= 6  else lag_1
    roll12 = float(np.mean(w[-12:])) if n >= 12 else float(np.mean(w))
    std3   = float(np.std(w[-3:],  ddof=1)) if n >= 3 else 0.0
    std6   = float(np.std(w[-6:],  ddof=1)) if n >= 6 else 0.0

    prev1 = w[-2] if n >= 2 else lag_1
    prev3 = w[-4] if n >= 4 else lag_1
    pct1  = (lag_1 - prev1) / prev1 if prev1 != 0 else 0.0
    pct3  = (lag_1 - prev3) / prev3 if prev3 != 0 else 0.0
    tr3   = lag_1 - prev3

    month   = target_dt.month
    quarter = (target_dt.month - 1) // 3 + 1

    computed: dict[str, float] = {
        "lag_1": lag_1, "lag_3": lag_3, "lag_6": lag_6,
        "rolling_mean_3": roll3, "rolling_mean_6": roll6, "rolling_mean_12": roll12,
        "rolling_std_3": std3, "rolling_std_6": std6,
        "pct_change_1": pct1, "pct_change_3": pct3,
        "price_momentum_strength": abs(pct3),
        "trend_3": tr3,
        "lag1_x_pct1": lag_1 * pct1,
        "lag1_x_std3": lag_1 * std3,
        "month_sin": float(np.sin(2 * np.pi * month / 12)),
        "month_cos": float(np.cos(2 * np.pi * month / 12)),
        "month": float(month),
        "quarter": float(quarter),
    }
    if external_vals:
        computed.update(external_vals)

    return np.array([[computed[col] for col in feature_cols]])


# ── Quality checks ────────────────────────────────────────────────────────────

def is_flat(preds: dict[int, float], threshold_pct: float = FLAT_THRESHOLD) -> bool:
    """Return True if all predicted values are identical or vary < threshold_pct %."""
    vals = list(preds.values())
    if len(vals) < 2:
        return True
    if len(set(round(v, 8) for v in vals)) == 1:
        return True
    ref = vals[0]
    if ref == 0:
        return True
    return (max(vals) - min(vals)) / ref * 100 < threshold_pct


def sanity_clip(
    preds: dict[int, float],
    ref_value: float,
    max_jump_pct: float = MAX_JUMP_PCT,
) -> dict[int, float]:
    """Clip predictions to ±max_jump_pct% of the last known price."""
    lo = max(0.0, ref_value * (1 - max_jump_pct / 100))
    hi = ref_value * (1 + max_jump_pct / 100)
    return {h: float(np.clip(v, lo, hi)) for h, v in preds.items()}


# ── Recursive multi-step forecasting ─────────────────────────────────────────

def recursive_forecast(
    model,
    scaler: StandardScaler | None,
    series: pd.Series,
    horizons: list[int],
    as_of: pd.Timestamp,
    feature_cols: list[str],
    external_vals: dict[str, float] | None = None,
) -> dict[int, float]:
    """
    Step-by-step forecasting: each predicted value feeds back as lag_1 for the
    next step.  External signal values are held constant at their last known
    level (we cannot forecast them here).
    """
    window = list(series.values[-12:])
    predictions: dict[int, float] = {}
    current = as_of

    for step in range(1, max(horizons) + 1):
        next_dt = current + relativedelta(months=1)
        x_new   = _build_feature_vector(window, next_dt, feature_cols, external_vals)

        if scaler is not None:
            x_new = scaler.transform(x_new)

        pred = max(0.0, float(model.predict(x_new)[0]))

        if step in horizons:
            predictions[step] = pred

        window.append(pred)
        current = next_dt

    return predictions


# ── Model validation (walk-forward MAE) ──────────────────────────────────────

def walk_forward_mae(
    model,
    scaler: StandardScaler | None,
    train_series: pd.Series,
    val_series: pd.Series,
    feature_cols: list[str],
    is_baseline: bool = False,
    external_vals: dict[str, float] | None = None,
) -> float:
    """
    Walk-forward 1-step MAE on val_series.
    Feeds actual observed values between steps (proper held-out evaluation).
    External signal values held constant at last training value.
    """
    window = list(train_series.values)
    if len(window) < 6:
        return float("inf")

    errors = []
    for dt, actual in val_series.items():
        if is_baseline:
            pred = window[-1]
        else:
            x = _build_feature_vector(window, dt, feature_cols, external_vals)
            if scaler is not None:
                x = scaler.transform(x)
            pred = max(0.0, float(model.predict(x)[0]))

        errors.append(abs(pred - actual))
        window.append(actual)

    return float(np.mean(errors)) if errors else float("inf")


# ── Push to Supabase ──────────────────────────────────────────────────────────

def push_predictions(supabase, records: list[dict]) -> bool:
    try:
        supabase.table("prediction_records").upsert(
            records,
            on_conflict="commodity,horizon_months,as_of_date",
        ).execute()
        return True
    except Exception as e:
        logging.warning(f"  ✗ Supabase push failed: {e}")
        return False


# ── Per-commodity forecasting ─────────────────────────────────────────────────

def forecast_commodity(
    supabase,
    commodity: str,
    as_of: pd.Timestamp,
    pkr_rate: float,
    external_cache: dict[str, pd.Series],
) -> str:
    """
    1. Fetch series + align external signals
    2. Train/eval each model on train split; apply baseline penalty
    3. Reject flat models; pick best non-flat
    4. Retrain winner on full data; apply sanity clip
    5. Push USD + PKR records
    """
    logging.info(f"\n📊 {commodity}")

    series = fetch_series(supabase, commodity)
    if series is None or len(series) < MIN_ROWS:
        n = len(series) if series is not None else 0
        msg = f"⚠️ skipped ({n} rows, need {MIN_ROWS})"
        logging.info(f"  {msg}")
        return msg

    # ── External signals ──────────────────────────────────────────────────────
    ext_keys     = COMMODITY_EXTERNALS.get(commodity, [])
    externals    = {k: external_cache[k] for k in ext_keys if k in external_cache}
    feature_cols = get_feature_cols(commodity)

    # Last known external values (constant during recursive forecast)
    ext_vals: dict[str, float] = {}
    for ext_key, ext_series in externals.items():
        col = f"ext_{ext_key}_lag1"
        shared_idx = ext_series.index.intersection(series.index)
        if len(shared_idx):
            ext_vals[col] = float(ext_series.loc[shared_idx[-1]])
        else:
            ext_vals[col] = 0.0

    # ── Split ─────────────────────────────────────────────────────────────────
    train_series = series.iloc[:-VAL_MONTHS]
    val_series   = series.iloc[-VAL_MONTHS:]

    df_train = build_features(train_series, externals)
    if len(df_train) < 8:
        msg = f"⚠️ skipped (only {len(df_train)} train samples after feature build)"
        logging.info(f"  {msg}")
        return msg

    # Only keep feature_cols that actually exist in df_train
    available_cols = [c for c in feature_cols if c in df_train.columns]
    if len(available_cols) < len(feature_cols):
        missing = set(feature_cols) - set(available_cols)
        logging.info(f"  ⚠️ missing features (will skip): {missing}")
    feature_cols = available_cols

    logging.info(f"  Features used: {len(feature_cols)} — {feature_cols}")

    X_train = df_train[feature_cols].values
    y_train = df_train["value"].values

    model_specs = [
        ("baseline_last_value", None,                                               False),
        ("linear_ridge",        Ridge(alpha=1.0),                                   True),
        ("random_forest",       RandomForestRegressor(n_estimators=100, max_depth=5,
                                                      random_state=42),             False),
    ]

    # ── Evaluate models ───────────────────────────────────────────────────────
    raw_maes: dict[str, float] = {}
    for model_name, model, use_scaler in model_specs:
        try:
            if model_name == "baseline_last_value":
                mae = walk_forward_mae(
                    None, None, train_series, val_series,
                    feature_cols, is_baseline=True,
                )
            else:
                scaler = StandardScaler() if use_scaler else None
                X_fit  = scaler.fit_transform(X_train) if scaler else X_train
                model.fit(X_fit, y_train)
                mae = walk_forward_mae(
                    model, scaler, train_series, val_series,
                    feature_cols, external_vals=ext_vals or None,
                )
            raw_maes[model_name] = mae
            logging.info(f"  MAE  {model_name}: {mae:.4f}")
        except Exception as e:
            logging.warning(f"  ✗ {model_name} eval failed: {e}")
            raw_maes[model_name] = float("inf")

    if all(v == float("inf") for v in raw_maes.values()):
        return "⚠️ all models failed evaluation"

    # ── Apply baseline penalty ────────────────────────────────────────────────
    effective_maes = dict(raw_maes)
    if "baseline_last_value" in effective_maes and effective_maes["baseline_last_value"] < float("inf"):
        effective_maes["baseline_last_value"] *= (1 + BASELINE_PENALTY)
        logging.info(
            f"  Baseline effective MAE after {BASELINE_PENALTY*100:.0f}% penalty: "
            f"{effective_maes['baseline_last_value']:.4f}"
        )

    # ── Retrain on full series + flat check ───────────────────────────────────
    df_full = build_features(series, externals)
    X_full  = df_full[feature_cols].values
    y_full  = df_full["value"].values
    last_known = float(series.iloc[-1])

    # Try models in MAE order; skip flat ones
    ranked = sorted(effective_maes.items(), key=lambda x: x[1])
    best_name   = None
    preds_usd   = None
    flat_skipped = []

    for candidate_name, _ in ranked:
        if candidate_name == "baseline_last_value":
            candidate_preds = {h: last_known for h in HORIZONS}
        else:
            spec_map = {s[0]: (s[1], s[2]) for s in model_specs}
            fresh_model, use_scaler = spec_map[candidate_name]
            scaler = StandardScaler() if use_scaler else None
            X_fit  = scaler.fit_transform(X_full) if scaler else X_full
            fresh_model.fit(X_fit, y_full)
            candidate_preds = recursive_forecast(
                fresh_model, scaler, series, HORIZONS, as_of,
                feature_cols, ext_vals or None,
            )

        if is_flat(candidate_preds):
            flat_skipped.append(candidate_name)
            logging.info(f"  ⚠️ {candidate_name} output is flat — skipping")
            continue

        best_name = candidate_name
        preds_usd = candidate_preds
        break

    # If every model is flat, fall back to baseline (accept flat rather than nothing)
    if best_name is None:
        logging.info(
            f"  ⚠️ All models produced flat output — using baseline as fallback"
        )
        best_name = "baseline_last_value"
        preds_usd = {h: last_known for h in HORIZONS}

    if flat_skipped:
        logging.info(f"  Flat-output prevention triggered for: {flat_skipped}")

    baseline_overridden = "baseline_last_value" in flat_skipped or (
        best_name != "baseline_last_value"
        and effective_maes.get("baseline_last_value", float("inf"))
        < effective_maes.get(best_name, float("inf"))
    )
    if baseline_overridden:
        logging.info("  Baseline was overridden by penalty or flat-rejection")

    # ── Sanity clip ───────────────────────────────────────────────────────────
    preds_usd = sanity_clip(preds_usd, last_known)

    logging.info(
        f"  ★  selected: {best_name}  "
        + "  ".join(f"h{h}={preds_usd[h]:.4f}" for h in HORIZONS)
    )

    # ── Build records ─────────────────────────────────────────────────────────
    hist_std      = float(series.std())
    created_at    = datetime.now(timezone.utc).isoformat()
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

    ok     = push_predictions(supabase, records)
    status = f"✅ {len(records)} records pushed ({best_name})" if ok else "❌ push failed"
    logging.info(f"  {status}")
    return status


# ── Viscose lightweight validation ────────────────────────────────────────────

def try_viscose(supabase) -> pd.Series | None:
    """
    Attempt to fetch viscose_usd data.  Returns the series if it meets minimum
    quality standards; otherwise logs the issue and returns None.
    """
    try:
        series = fetch_series(supabase, "viscose_usd")
        if series is None or len(series) == 0:
            logging.info("  viscose_usd: no data in commodity_prices — skipping")
            return None

        # Continuity check: flag gaps > 2 months
        gaps = series.index.to_series().diff().dt.days.dropna()
        large_gaps = (gaps > 62).sum()
        nan_count  = series.isna().sum()

        logging.info(
            f"  viscose_usd: {len(series)} rows, {nan_count} NaN, {large_gaps} gap(s) > 2 mo"
        )

        if len(series) < MIN_ROWS:
            logging.info(f"  viscose_usd: insufficient rows ({len(series)} < {MIN_ROWS}) — skipping")
            return None
        if nan_count > 0:
            logging.info("  viscose_usd: NaN values detected — skipping")
            return None

        return series
    except Exception as e:
        logging.warning(f"  viscose_usd: fetch/validation error — {e}")
        return None


# ── Entry point ───────────────────────────────────────────────────────────────

def run_ml_forecasts():
    logging.info("=" * 50)
    logging.info("🚀 STARTING ML FORECASTING (PHASE 6 — ENHANCED)")
    logging.info("=" * 50)

    supabase = _get_supabase_client()
    if supabase is None:
        logging.error("❌ No Supabase connection — cannot run forecasts.")
        print("\n=== PIPELINE STATUS ===")
        print("Ingestion completed")
        print("Forecast completed (skipped — no Supabase connection)")
        return

    today = pd.Timestamp.today()
    as_of = pd.Timestamp(year=today.year, month=today.month, day=1)
    if as_of > pd.Timestamp.today():
        raise ValueError(f"Invalid as_of_date computed: {as_of} is in the future")

    pkr_rate = get_fx_rate("PKR")
    logging.info(f"As-of: {as_of.date()}  |  FX: {pkr_rate:.2f} PKR/USD")

    # ── Pre-fetch external signal series (used by multiple commodities) ────────
    logging.info("\n⬇️  Pre-fetching external signal series...")
    all_ext_keys = {e for exts in COMMODITY_EXTERNALS.values() for e in exts}
    external_cache: dict[str, pd.Series] = {}
    for ext_key in all_ext_keys:
        s = fetch_series(supabase, ext_key)
        if s is not None:
            external_cache[ext_key] = s
            logging.info(f"  {ext_key}: {len(s)} rows loaded")
        else:
            logging.warning(f"  {ext_key}: not available — dependent commodities will run without it")

    # ── Main forecast loop ────────────────────────────────────────────────────
    results: dict[str, str] = {}
    for commodity in COMMODITIES_USD:
        try:
            results[commodity] = forecast_commodity(
                supabase, commodity, as_of, pkr_rate, external_cache
            )
        except Exception as e:
            logging.error(f"  ❌ Unexpected error for {commodity}: {e}")
            results[commodity] = f"❌ error: {e}"

    # ── Viscose ───────────────────────────────────────────────────────────────
    logging.info(f"\n📊 viscose_usd")
    visc_series = try_viscose(supabase)
    if visc_series is not None:
        try:
            results["viscose_usd"] = forecast_commodity(
                supabase, "viscose_usd", as_of, pkr_rate, external_cache
            )
        except Exception as e:
            logging.error(f"  ❌ viscose_usd forecast failed: {e}")
            results["viscose_usd"] = f"❌ error: {e}"
    else:
        results["viscose_usd"] = "⚠️ skipped (no valid data)"

    # ── Summary ───────────────────────────────────────────────────────────────
    logging.info("\n=== FORECAST SUMMARY ===")
    for k, v in results.items():
        logging.info(f"  {k}: {v}")

    logging.info("\n✅ ML Forecasting complete.")
    print("\n=== PIPELINE STATUS ===")
    print("Ingestion completed")
    print("Forecast completed")


if __name__ == "__main__":
    run_ml_forecasts()
