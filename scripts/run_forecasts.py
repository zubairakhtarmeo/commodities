import os
import sys
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
import warnings
import logging
import sys
from datetime import datetime

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

# Add src to path
sys.path.insert(0, str(BASE_DIR / "src"))

try:
    import toml
except ImportError:
    pass

from forecasting.features.builder import FeatureBuilder
from forecasting.config import FeaturePackConfig, DatasetConfig, ModelSpec
from forecasting.dataset.builder import build_supervised_dataset
from forecasting.models.factory import build_model
from supabase import create_client

def get_supabase_credentials():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        try:
            secrets = toml.load(BASE_DIR / ".streamlit/secrets.toml")
            url = secrets.get("SUPABASE_URL")
            key = secrets.get("SUPABASE_SERVICE_ROLE_KEY")
        except Exception:
            pass
    if not url or not key:
        raise ValueError("Supabase credentials not found. Make sure .streamlit/secrets.toml is configured.")
    return url, key

supabase = create_client(*get_supabase_credentials())

COMMODITIES_TO_FORECAST = [
    "cotton_usd",
    "crude_oil_usd",
    "natural_gas_usd",
    "polyester_usd"
]

def get_commodity_data(commodity: str) -> pd.DataFrame:
    res = supabase.table("commodity_prices").select("date,value").eq("commodity", commodity).order("date").execute()
    df = pd.DataFrame(res.data)
    return df

def run_ml_forecasts():
    logging.info("=" * 50)
    logging.info("🚀 STARTING ML FORECASTING (PHASE 3)")
    logging.info("=" * 50)
    
    for commodity in COMMODITIES_TO_FORECAST:
        logging.info(f"\n📊 Processing {commodity}...")
        df = get_commodity_data(commodity)
        
        if len(df) < 24:
            logging.info(f"  ⚠️ Not enough data (needs >= 24, has {len(df)}). Skipping.")
            continue
            
        # Clean and format
        df['date'] = pd.to_datetime(df['date'])
        # Upsample gracefully
        df = df.sort_values("date").set_index('date').resample('MS').mean().ffill().dropna()
        df.rename(columns={'value': 'target'}, inplace=True)
        
        # Prepare features configured identically as `real_asset_monthly.yml`
        packs = [
            FeaturePackConfig(name="lags", params={"lags": [1, 2, 3, 6, 12]}),
            FeaturePackConfig(name="rolling_stats", params={"windows": [3, 6, 12], "stats": ["mean", "std"], "min_periods": 3}),
            FeaturePackConfig(name="volatility", params={"windows": [3, 6, 12], "min_periods": 3})
        ]
        
        try:
            fb = FeatureBuilder.from_configs(packs)
            X = fb.build(df)
            
            # Predict only exactly what's requested: 1, 3, 6
            horizons = [1, 3, 6]
            d_cfg = DatasetConfig(horizon_steps=horizons, drop_na_target=True)
            ds = build_supervised_dataset(features=X, target=df["target"], cfg=d_cfg)
        except Exception as e:
            logging.info(f"  ✗ Feature building failed: {e}")
            continue
            
        if len(ds.X) == 0:
            logging.info(f"  ⚠️ After dataset construction (due to NAs/lookback), 0 training samples remain. Need more historical data. Skipping.")
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
                y_pred = model.estimator.predict(x_pred)[0] # Shape (len(horizons),) corresponding to the MultiOutputRegressor
                
                for i, h in enumerate(horizons):
                    val = float(y_pred[i])
                    # Ensure positive predicted values and provide bounds
                    val = max(val, 0.0) 
                    variance = val * 0.1 * (1 + h/12) # Bounds widen over time
                    
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
                logging.info(f"  ✗ Model {spec.name} failed: {e}")
                
        if predictions:
            logging.info(f"  ✓ Models finished. Pushing {len(predictions)} forecasts to Supabase...")
            try:
                supabase.table("prediction_records").upsert(
                    predictions
                ).execute()
                logging.info(f"  ✓ Validated in Supabase ({commodity})")
            except Exception as e:
                logging.info(f"  ✗ Supabase exception: {e}")
                
    # Skip viscose explicitly
    logging.info(f"\n📊 Processing viscose_usd...")
    logging.info("  ⚠️ Skipping Viscose as requested.")
    
    logging.info("\n✅ ML Forecasting complete.")

if __name__ == "__main__":
    run_ml_forecasts()
