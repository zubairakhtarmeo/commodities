"""
Generate ML forecasts for each country's cotton price.
Reads from cotton_country_prices, writes to cotton_country_predictions.
Requires at least 6 months of data per country to forecast.

Usage:
  python scripts/run_cotton_country_forecasts.py

Credentials: set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY as env vars,
or they will be read from .streamlit/secrets.toml automatically.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from datetime import date

import requests
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

MIN_ROWS = 6
HORIZONS = [1, 3, 6]


def get_supabase_config() -> tuple[str, str]:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

    if not url or not key:
        secrets_path = BASE_DIR / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            try:
                import tomllib
                with open(secrets_path, "rb") as f:
                    secrets = tomllib.load(f)
            except ImportError:
                import tomli
                with open(secrets_path, "rb") as f:
                    secrets = tomli.load(f)
            url = url or secrets.get("SUPABASE_URL")
            key = key or secrets.get("SUPABASE_SERVICE_ROLE_KEY") or secrets.get("SUPABASE_ANON_KEY")

    if not url or not key:
        raise SystemExit("Missing SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY. "
                         "Set as env vars or add to .streamlit/secrets.toml.")
    return str(url).rstrip("/"), str(key).strip()


def _headers(key: str) -> dict:
    return {"apikey": key, "Authorization": f"Bearer {key}"}


def fetch_all_countries(url: str, key: str) -> list[str]:
    r = requests.get(
        f"{url}/rest/v1/cotton_country_prices",
        headers=_headers(key),
        params={"select": "country", "limit": "1000"},
        timeout=30,
    )
    if not r.ok or not r.json():
        return []
    return sorted(set(row["country"] for row in r.json()))


def fetch_country_history(country: str, url: str, key: str) -> pd.DataFrame | None:
    r = requests.get(
        f"{url}/rest/v1/cotton_country_prices",
        headers=_headers(key),
        params={
            "country": f"eq.{country}",
            "select": "date,price_usd_per_lb",
            "order": "date.asc",
            "limit": "200",
        },
        timeout=30,
    )
    if not r.ok or not r.json():
        return None
    df = pd.DataFrame(r.json())
    df["date"] = pd.to_datetime(df["date"])
    df["price_usd_per_lb"] = pd.to_numeric(df["price_usd_per_lb"], errors="coerce")
    df = df.dropna().set_index("date").sort_index()
    return df


def forecast_country(country: str, df: pd.DataFrame, as_of: date) -> list[dict] | None:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    series = df["price_usd_per_lb"].dropna()
    if len(series) < MIN_ROWS:
        return None

    feat_df = pd.DataFrame({"y": series})
    feat_df["lag1"] = feat_df["y"].shift(1)
    feat_df["lag2"] = feat_df["y"].shift(2)
    feat_df["lag3"] = feat_df["y"].shift(3)
    feat_df["roll3"] = feat_df["y"].rolling(3).mean()
    feat_df = feat_df.dropna()

    if len(feat_df) < 4:
        return None

    X = feat_df[["lag1", "lag2", "lag3", "roll3"]].values
    y = feat_df["y"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    last_vals = list(series.values[-3:])
    std = float(series.std())
    predictions = []

    for h in HORIZONS:
        vals = last_vals.copy()
        pred = None
        for _ in range(h):
            l1, l2, l3 = vals[-1], vals[-2], vals[-3]
            r3 = float(np.mean(vals[-3:]))
            x_new = scaler.transform([[l1, l2, l3, r3]])
            pred = float(model.predict(x_new)[0])
            vals.append(pred)

        target_date = as_of + relativedelta(months=h)
        predictions.append({
            "country": country,
            "as_of_date": str(as_of),
            "horizon_months": h,
            "target_date": str(target_date),
            "predicted_usd_per_lb": round(float(pred), 6),
            "lower_bound": round(float(pred) - std * 0.5, 6),
            "upper_bound": round(float(pred) + std * 0.5, 6),
            "model_name": "ridge",
        })

    return predictions


def push_predictions(predictions: list[dict], url: str, key: str) -> bool:
    headers = {
        **_headers(key),
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }
    r = requests.post(
        f"{url}/rest/v1/cotton_country_predictions",
        headers=headers,
        json=predictions,
        timeout=30,
    )
    return r.status_code in (200, 201)


def run_all() -> None:
    url, key = get_supabase_config()
    as_of = date.today().replace(day=1)

    countries = fetch_all_countries(url, key)
    if not countries:
        print("❌ No countries found in cotton_country_prices. Run ingest first.")
        sys.exit(1)

    print(f"Countries found: {countries}")
    results: dict[str, str] = {}

    for country in countries:
        df = fetch_country_history(country, url, key)
        n = len(df) if df is not None else 0
        if df is None or n < MIN_ROWS:
            print(f"⚠️  {country}: insufficient data ({n} rows, need {MIN_ROWS})")
            results[country] = "skipped"
            continue

        preds = forecast_country(country, df, as_of)
        if preds and push_predictions(preds, url, key):
            print(f"✅ {country}: {len(preds)} forecasts pushed")
            results[country] = "success"
        else:
            print(f"❌ {country}: forecast or push failed")
            results[country] = "failed"

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    run_all()
