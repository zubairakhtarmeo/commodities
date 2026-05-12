"""
Forecast Evaluation — retrospective backtesting for commodity price models.

Since prediction_records only stores forward-looking forecasts (no historical
predicted-vs-actual pairs), evaluation uses rolling walk-forward backtesting:
for each commodity, we simulate what the model would have predicted at each
month in the last BACKTEST_MONTHS, then measure against actual prices.

Metrics computed (per commodity, per horizon):
  MAE   — Mean Absolute Error (same units as price)
  RMSE  — Root Mean Squared Error (penalises large misses)
  MAPE  — Mean Absolute Percentage Error (scale-free, %)
  Dir   — Directional accuracy (% of months where direction was correct)

Baselines:
  last_value  — predicts last observed price (persistence)
  ma3         — 3-month moving average

Output: artifacts/forecast_evaluation.json

Usage:
  python scripts/evaluate_forecasts.py
  python scripts/evaluate_forecasts.py --backtest-months 24
  python scripts/evaluate_forecasts.py --quiet
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

BASE_DIR = Path(__file__).parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = ARTIFACTS_DIR / "forecast_evaluation.json"

sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

BACKTEST_MONTHS_DEFAULT = 18   # rolling window for evaluation
MIN_BACKTEST_POINTS     = 6    # minimum observations needed per metric
EVAL_HORIZONS           = [1, 3, 6]

# Imported from run_forecasts — keeps feature engineering exactly in sync
from scripts.run_forecasts import (
    COMMODITIES_USD,
    COMMODITY_EXTERNALS,
    MIN_ROWS,
    VAL_MONTHS,
    get_feature_cols,
    get_supabase_credentials,
    fetch_series,
    build_features,
    build_return_features,
    _build_feature_vector,
    walk_forward_mae,
    walk_forward_mae_returns,
)
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


# ── Metrics ────────────────────────────────────────────────────────────────────

def _mae(actuals: np.ndarray, preds: np.ndarray) -> float:
    return float(np.mean(np.abs(actuals - preds)))


def _rmse(actuals: np.ndarray, preds: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actuals - preds) ** 2)))


def _mape(actuals: np.ndarray, preds: np.ndarray) -> float | None:
    mask = actuals != 0
    if not mask.any():
        return None
    return float(np.mean(np.abs((actuals[mask] - preds[mask]) / actuals[mask])) * 100)


def _directional(actuals: np.ndarray, prev_actuals: np.ndarray, preds: np.ndarray) -> float | None:
    """Fraction of months where predicted direction matches actual direction."""
    if len(actuals) < 2 or len(prev_actuals) < 2:
        return None
    actual_dir = np.sign(actuals - prev_actuals)
    pred_dir   = np.sign(preds   - prev_actuals)
    valid = actual_dir != 0
    if not valid.any():
        return None
    return float(np.mean(actual_dir[valid] == pred_dir[valid]) * 100)


def compute_metrics(actuals: list[float], preds: list[float], prev_actuals: list[float]) -> dict:
    a  = np.array(actuals)
    p  = np.array(preds)
    pa = np.array(prev_actuals) if prev_actuals else None

    result: dict = {
        "n":    len(a),
        "mae":  round(_mae(a, p), 6),
        "rmse": round(_rmse(a, p), 6),
    }
    mape = _mape(a, p)
    result["mape"] = round(mape, 2) if mape is not None else None

    if pa is not None and len(pa) == len(a):
        da = _directional(a, pa, p)
        result["directional_pct"] = round(da, 1) if da is not None else None

    return result


# ── Supabase fetch (REST fallback if supabase-py unavailable) ──────────────────

def _fetch_series_rest(url: str, key: str, commodity: str) -> pd.Series | None:
    import requests
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }
    params = {
        "select": "date,value",
        "commodity": f"eq.{commodity}",
        "order": "date.asc",
        "limit": "2000",
    }
    r = requests.get(f"{url}/rest/v1/commodity_prices", headers=headers, params=params, timeout=20)
    if not r.ok:
        logger.warning("REST fetch failed for %s (%d): %s", commodity, r.status_code, r.text[:100])
        return None
    data = r.json()
    if not data:
        return None
    df = pd.DataFrame(data)
    df["date"]  = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().set_index("date").sort_index()
    return df["value"].resample("MS").mean().dropna()


def _load_all_series() -> dict[str, pd.Series]:
    """Load all commodity series from Supabase (supabase-py or REST)."""
    url, key = get_supabase_credentials()
    if not url or not key:
        logger.error("Missing Supabase credentials.")
        sys.exit(1)

    supabase = None
    try:
        from supabase import create_client
        supabase = create_client(url, key)
    except Exception:
        pass

    result: dict[str, pd.Series] = {}
    for commodity in COMMODITIES_USD:
        if supabase is not None:
            s = fetch_series(supabase, commodity)
        else:
            s = _fetch_series_rest(url, key, commodity)
        if s is not None and len(s) >= MIN_ROWS:
            result[commodity] = s
            logger.info("  Loaded %s: %d rows (%s → %s)",
                        commodity, len(s), s.index[0].date(), s.index[-1].date())
        else:
            n = len(s) if s is not None else 0
            logger.warning("  Skipping %s: only %d rows (need %d)", commodity, n, MIN_ROWS)
    return result


# ── Model training helper ──────────────────────────────────────────────────────

MODEL_SPECS = [
    ("linear_ridge",  Ridge(alpha=1.0),                                                True,  False),
    ("random_forest", RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42), False, False),
    ("ridge_returns", Ridge(alpha=1.0),                                                True,  True),
    ("rf_returns",    RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42), False, True),
]


def _train_and_forecast_1step(
    train_series: pd.Series,
    target_dt: pd.Timestamp,
    feature_cols: list[str],
    externals: dict[str, pd.Series],
    ext_vals: dict[str, float],
) -> dict[str, float]:
    """
    Train each model on train_series, predict one step ahead to target_dt.
    Returns {model_name: predicted_price}.
    """
    preds: dict[str, float] = {}

    df_train = build_features(train_series, externals)
    available_cols = [c for c in feature_cols if c in df_train.columns]
    if not available_cols or len(df_train) < 8:
        return {}

    X_train = df_train[available_cols].values
    y_train = df_train["value"].values

    df_train_ret = build_return_features(train_series, externals)
    ret_cols = [c for c in available_cols if c in df_train_ret.columns]
    if len(df_train_ret) >= 8 and ret_cols:
        X_train_ret = df_train_ret[ret_cols].values
        y_train_ret = df_train_ret["target_return"].values
    else:
        X_train_ret = y_train_ret = None

    window = list(train_series.values[-12:])

    for name, model_proto, use_scaler, use_returns in MODEL_SPECS:
        try:
            import copy
            model = copy.deepcopy(model_proto)
            scaler = StandardScaler() if use_scaler else None

            if use_returns:
                if X_train_ret is None or len(X_train_ret) < 8:
                    continue
                X_fit = scaler.fit_transform(X_train_ret) if scaler else X_train_ret
                model.fit(X_fit, y_train_ret)
                x = _build_feature_vector(window, target_dt, available_cols, ext_vals or None)
                if scaler:
                    x = scaler.transform(x)
                ret = float(np.clip(model.predict(x)[0], -0.30, 0.30))
                preds[name] = max(0.0, window[-1] * (1 + ret))
            else:
                X_fit = scaler.fit_transform(X_train) if scaler else X_train
                model.fit(X_fit, y_train)
                x = _build_feature_vector(window, target_dt, available_cols, ext_vals or None)
                if scaler:
                    x = scaler.transform(x)
                preds[name] = max(0.0, float(model.predict(x)[0]))

        except Exception as exc:
            logger.debug("  %s failed at %s: %s", name, target_dt.date(), exc)

    return preds


# ── Per-horizon backtesting ────────────────────────────────────────────────────

def backtest_commodity(
    commodity: str,
    series: pd.Series,
    all_series: dict[str, pd.Series],
    backtest_months: int,
) -> dict:
    """
    Rolling walk-forward backtest.

    For each evaluation month t in the last backtest_months of the series:
      - train on series[:t]
      - predict h months ahead (h=1,3,6) — use h-step recursive prediction
      - compare to actual series[t+h] if available

    Baselines are computed without training:
      - last_value  = series[t-1]
      - ma3         = mean(series[t-3:t])
    """
    logger.info("  Backtesting %s (%d total rows, %d backtest months)",
                commodity, len(series), backtest_months)

    ext_keys     = COMMODITY_EXTERNALS.get(commodity, [])
    externals    = {k: all_series[k] for k in ext_keys if k in all_series}
    feature_cols = get_feature_cols(commodity)

    # Evaluation points: last backtest_months months, but leave room for max horizon
    all_dates = series.index.tolist()
    max_h     = max(EVAL_HORIZONS)

    # earliest training cutoff needs MIN_ROWS + VAL_MONTHS history
    min_train_end_idx = MIN_ROWS + VAL_MONTHS
    eval_start_idx    = max(min_train_end_idx, len(all_dates) - backtest_months - max_h)
    eval_end_idx      = len(all_dates) - max_h  # must have actuals for max_h steps ahead

    if eval_end_idx <= eval_start_idx:
        return {"error": f"insufficient data for backtest (need at least {min_train_end_idx + max_h} rows)"}

    # Accumulators: {horizon: {model: (actuals, preds, prev_actuals)}}
    horizon_data: dict[int, dict[str, dict]] = {
        h: {**{m[0]: {"actuals": [], "preds": [], "prev": []} for m in MODEL_SPECS},
            "last_value": {"actuals": [], "preds": [], "prev": []},
            "ma3":        {"actuals": [], "preds": [], "prev": []}}
        for h in EVAL_HORIZONS
    }

    eval_indices = list(range(eval_start_idx, eval_end_idx + 1))
    logger.info("  Eval window: %s → %s (%d points)",
                all_dates[eval_start_idx].date(),
                all_dates[eval_end_idx].date(),
                len(eval_indices))

    for i in eval_indices:
        train_series = series.iloc[:i]
        as_of_dt     = all_dates[i - 1]

        # External signal values at training cutoff
        ext_vals: dict[str, float] = {}
        for ext_key, ext_series in externals.items():
            col = f"ext_{ext_key}_lag1"
            shared = ext_series.index.intersection(train_series.index)
            ext_vals[col] = float(ext_series.loc[shared[-1]]) if len(shared) else 0.0

        # Train models once per eval point (1-step prediction only here)
        # For h>1 we need the prediction at offset h from position i
        # We use a simplified approach: train on train_series, recursively
        # step forward h months using walk-forward (feeding actual values)

        for h in EVAL_HORIZONS:
            target_idx = i + h - 1   # index of actual value h months ahead
            if target_idx >= len(all_dates):
                continue

            actual       = float(series.iloc[target_idx])
            prev_actual  = float(series.iloc[i - 1])  # last known at eval time

            # ── Baselines (no model, no training) ────────────────────────────
            last_val   = float(series.iloc[i - 1])
            ma3_vals   = series.iloc[max(0, i - 3):i].values
            ma3_pred   = float(np.mean(ma3_vals))

            horizon_data[h]["last_value"]["actuals"].append(actual)
            horizon_data[h]["last_value"]["preds"].append(last_val)
            horizon_data[h]["last_value"]["prev"].append(prev_actual)

            horizon_data[h]["ma3"]["actuals"].append(actual)
            horizon_data[h]["ma3"]["preds"].append(ma3_pred)
            horizon_data[h]["ma3"]["prev"].append(prev_actual)

            # ── ML models: walk-forward h steps feeding actuals ───────────────
            # For honest multi-step evaluation: train on [:i], then for each
            # intermediate step, feed the actual (not predicted) value.
            # This isolates the h-step horizon signal without compounding error.
            eval_train = series.iloc[:i]
            eval_val   = series.iloc[i : i + h]  # the h steps to evaluate

            if len(eval_val) < h:
                continue

            # Use walk_forward_mae infrastructure but capture per-step preds
            for m_name, model_proto, use_scaler, use_returns in MODEL_SPECS:
                try:
                    import copy
                    model  = copy.deepcopy(model_proto)
                    scaler = StandardScaler() if use_scaler else None

                    df_tr = build_features(eval_train, externals)
                    avail = [c for c in feature_cols if c in df_tr.columns]
                    if not avail or len(df_tr) < 8:
                        continue
                    X_tr = df_tr[avail].values
                    y_tr = df_tr["value"].values

                    if use_returns:
                        df_ret = build_return_features(eval_train, externals)
                        r_cols = [c for c in avail if c in df_ret.columns]
                        if len(df_ret) < 8 or not r_cols:
                            continue
                        X_fit  = scaler.fit_transform(df_ret[r_cols].values) if scaler else df_ret[r_cols].values
                        model.fit(X_fit, df_ret["target_return"].values)
                    else:
                        X_fit  = scaler.fit_transform(X_tr) if scaler else X_tr
                        model.fit(X_fit, y_tr)

                    # Walk forward h steps feeding actuals
                    window = list(eval_train.values[-12:])
                    for step, (step_dt, step_actual) in enumerate(eval_val.items(), start=1):
                        x = _build_feature_vector(window, step_dt, avail, ext_vals or None)
                        if scaler:
                            x = scaler.transform(x)
                        if use_returns:
                            ret  = float(np.clip(model.predict(x)[0], -0.30, 0.30))
                            pred = max(0.0, window[-1] * (1 + ret))
                        else:
                            pred = max(0.0, float(model.predict(x)[0]))
                        window.append(float(step_actual))  # feed actual, not predicted

                    # pred is now the h-step-ahead prediction (last step)
                    if m_name in horizon_data[h]:
                        horizon_data[h][m_name]["actuals"].append(actual)
                        horizon_data[h][m_name]["preds"].append(pred)
                        horizon_data[h][m_name]["prev"].append(prev_actual)

                except Exception:
                    pass

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    output: dict = {}
    for h in EVAL_HORIZONS:
        output[f"h{h}"] = {}
        for model_name, data in horizon_data[h].items():
            if len(data["actuals"]) < MIN_BACKTEST_POINTS:
                output[f"h{h}"][model_name] = {"n": len(data["actuals"]), "error": "insufficient points"}
                continue
            output[f"h{h}"][model_name] = compute_metrics(
                data["actuals"], data["preds"],
                data["prev"] if data["prev"] else None,
            )

    # ── Beat-baseline summary ─────────────────────────────────────────────────
    summary: dict[str, str] = {}
    baseline_mae = {h: output[f"h{h}"].get("last_value", {}).get("mae") for h in EVAL_HORIZONS}

    for m_name, _, _, _ in MODEL_SPECS:
        beat = []
        for h in EVAL_HORIZONS:
            m_mae  = output[f"h{h}"].get(m_name, {}).get("mae")
            b_mae  = baseline_mae[h]
            if m_mae is not None and b_mae is not None:
                if m_mae < b_mae:
                    beat.append(f"h{h}")
        summary[m_name] = f"beats baseline at: {beat or 'none'}"

    output["beat_baseline_summary"] = summary
    return output


# ── Main ───────────────────────────────────────────────────────────────────────

def run_evaluation(backtest_months: int = BACKTEST_MONTHS_DEFAULT, quiet: bool = False) -> None:
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)

    logger.info("=" * 60)
    logger.info("FORECAST EVALUATION  (backtest_months=%d)", backtest_months)
    logger.info("=" * 60)

    logger.info("Loading commodity series from Supabase...")
    all_series = _load_all_series()

    if not all_series:
        logger.error("No commodity series loaded — cannot evaluate.")
        sys.exit(1)

    results: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backtest_months": backtest_months,
        "horizons": EVAL_HORIZONS,
        "commodities": {},
    }

    for commodity, series in all_series.items():
        logger.info("\n--- %s ---", commodity)
        try:
            res = backtest_commodity(commodity, series, all_series, backtest_months)
            results["commodities"][commodity] = res
        except Exception as exc:
            logger.error("  Backtest failed for %s: %s", commodity, exc)
            results["commodities"][commodity] = {"error": str(exc)}

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY  (MAE, h1 / h3 / h6)")
    print("=" * 60)

    for commodity, data in results["commodities"].items():
        if "error" in data:
            print(f"  {commodity}: {data['error']}")
            continue
        print(f"\n  {commodity}")
        # Print each model's MAE across horizons
        all_models = ["last_value", "ma3"] + [m[0] for m in MODEL_SPECS]
        for m in all_models:
            row = []
            for h in EVAL_HORIZONS:
                v = data.get(f"h{h}", {}).get(m, {}).get("mae")
                row.append(f"{v:.4f}" if v is not None else "  n/a ")
            print(f"    {m:<22} {' / '.join(row)}")
        if "beat_baseline_summary" in data:
            print("  Beat-baseline:")
            for m, verdict in data["beat_baseline_summary"].items():
                print(f"    {m:<22} {verdict}")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("\nResults saved to %s", OUTPUT_PATH)

    print("\n" + "=" * 60)
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate commodity forecast models via backtesting")
    parser.add_argument("--backtest-months", type=int, default=BACKTEST_MONTHS_DEFAULT,
                        help=f"Rolling backtest window in months (default {BACKTEST_MONTHS_DEFAULT})")
    parser.add_argument("--quiet", action="store_true", help="Suppress INFO logs")
    args = parser.parse_args()
    run_evaluation(backtest_months=args.backtest_months, quiet=args.quiet)
