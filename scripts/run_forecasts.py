import os
import sys
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import warnings
import logging

BASE_DIR = Path(__file__).parent.parent

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / f"forecasts_{datetime.today().strftime('%Y%m%d')}.log", encoding="utf-8")
    ]
)

warnings.filterwarnings('ignore')

sys.path.insert(0, str(BASE_DIR / "src"))

print("ENV CHECK:")
print("SUPABASE_URL:", bool(os.getenv("SUPABASE_URL")))
print("SUPABASE_KEY:", bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY")))
print("FRED_API_KEY:", bool(os.getenv("FRED_API_KEY")))

try:
    import toml
except ImportError:
    toml = None

from forecasting.features.builder import FeatureBuilder
from forecasting.config import FeaturePackConfig, DatasetConfig, ModelSpec
from forecasting.dataset.builder import build_supervised_dataset
from forecasting.models.factory import build_model


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


COMMODITIES_TO_FORECAST = [
    "cotton_usd",
    "crude_oil_usd",
    "natural_gas_usd",
    "polyester_usd"
]


def get_commodity_data(supabase, commodity: str) -> pd.DataFrame:
    res = supabase.table("commodity_prices").select("date,value").eq("commodity", commodity).order("date").execute()
    return pd.DataFrame(res.data)


def run_ml_forecasts():
    logging.info("=" * 50)
    logging.info("🚀 STARTING ML FORECASTING (PHASE 3)")
    logging.info("=" * 50)

    supabase = _get_supabase_client()
    if supabase is None:
        logging.error("❌ Supabase credentials missing or client failed to initialize. Cannot run forecasts.")
        print("\n=== PIPELINE STATUS ===")
        print("Ingestion completed")
        print("Forecast completed (skipped — no Supabase connection)")
        return

    forecast_results = {}

    for commodity in COMMODITIES_TO_FORECAST:
        logging.info(f"\n📊 Processing {commodity}...")
        try:
            df = get_commodity_data(supabase, commodity)
        except Exception as e:
            logging.warning(f"  ⚠️ Could not fetch data for {commodity}: {e}")
            forecast_results[commodity] = f"❌ fetch failed: {e}"
            continue

        if len(df) < 24:
            logging.info(f"  ⚠️ Not enough data (needs >= 24, has {len(df)}). Skipping.")
            forecast_results[commodity] = f"⚠️ skipped (only {len(df)} rows)"
            continue

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values("date").set_index('date').resample('MS').mean().ffill().dropna()
        df.rename(columns={'value': 'target'}, inplace=True)

        packs = [
            FeaturePackConfig(name="lags", params={"lags": [1, 2, 3, 6, 12]}),
            FeaturePackConfig(name="rolling_stats", params={"windows": [3, 6, 12], "stats": ["mean", "std"], "min_periods": 3}),
            FeaturePackConfig(name="volatility", params={"windows": [3, 6, 12], "min_periods": 3})
        ]

        try:
            fb = FeatureBuilder.from_configs(packs)
            X = fb.build(df)

            horizons = [1, 3, 6]
            d_cfg = DatasetConfig(horizon_steps=horizons, drop_na_target=True)
            ds = build_supervised_dataset(features=X, target=df["target"], cfg=d_cfg)
        except Exception as e:
            logging.warning(f"  ✗ Feature building failed: {e}")
            forecast_results[commodity] = f"❌ feature build failed: {e}"
            continue

        if len(ds.X) == 0:
            logging.info(f"  ⚠️ 0 training samples after dataset construction. Need more historical data. Skipping.")
            forecast_results[commodity] = "⚠️ skipped (0 training samples)"
            continue

        specs = [
            ModelSpec(name="baseline_last_value", type="naive"),
            ModelSpec(name="linear_ridge", type="ridge", params={"alpha": 1.0})
        ]

        predictions = []
        for spec in specs:
            try:
                model = build_model(spec, horizon_count=len(horizons))
                model.estimator.fit(ds.X, ds.y)

                asof = df.index[-1]
                x_pred = X.loc[[asof]]
                y_pred = model.estimator.predict(x_pred)[0]

                for i, h in enumerate(horizons):
                    val = float(y_pred[i])
                    val = max(val, 0.0)
                    variance = val * 0.1 * (1 + h / 12)

                    predictions.append({
                        "commodity": commodity,
                        "model_name": spec.name,
                        "as_of_date": asof.strftime("%Y-%m-%d"),
                        "horizon_months": h,
                        "target_date": (asof + pd.DateOffset(months=h)).strftime("%Y-%m-%d"),
                        "predicted_value": val,
                        "lower_bound": max(0.0, val - variance),
                        "upper_bound": val + variance,
                        "unit": "USD",
                        "is_demo": False
                    })
            except Exception as e:
                logging.warning(f"  ✗ Model {spec.name} failed: {e}")

        if predictions:
            logging.info(f"  ✓ Models finished. Pushing {len(predictions)} forecasts to Supabase...")
            try:
                supabase.table("prediction_records").upsert(predictions).execute()
                logging.info(f"  ✓ Pushed to Supabase ({commodity})")
                forecast_results[commodity] = f"✅ {len(predictions)} forecasts pushed"
            except Exception as e:
                logging.warning(f"  ✗ Supabase push failed: {e}")
                forecast_results[commodity] = f"❌ push failed: {e}"
        else:
            forecast_results[commodity] = "⚠️ no predictions generated"

    logging.info(f"\n📊 Processing viscose_usd...")
    logging.info("  ⚠️ Skipping Viscose as requested.")
    forecast_results["viscose_usd"] = "⚠️ skipped (manual)"

    logging.info("\n=== FORECAST SUMMARY ===")
    for k, v in forecast_results.items():
        logging.info(f"  {k}: {v}")

    logging.info("\n✅ ML Forecasting complete.")
    print("\n=== PIPELINE STATUS ===")
    print("Ingestion completed")
    print("Forecast completed")


if __name__ == "__main__":
    run_ml_forecasts()
