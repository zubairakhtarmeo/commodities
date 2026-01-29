from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests


def _fetch_fred_monthly(series_id: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    if df.shape[1] < 2:
        raise ValueError(f"Unexpected CSV format from FRED for {series_id}")

    df.columns = ["timestamp", "value"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["timestamp", "value"]).copy()

    # Convert to month-start and keep last observation in each month
    df["timestamp"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()
    df = df.groupby("timestamp", as_index=False).agg(usd_pkr=("value", "last"))
    return df


def _fetch_latest_usd_pkr() -> float:
    # Same primary endpoint used by the Streamlit app.
    url = "https://open.er-api.com/v6/latest/USD"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    rate = float(payload["rates"]["PKR"])
    if rate <= 0:
        raise ValueError("Invalid USD/PKR rate")
    return rate


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    poly_usd_path = root / "data" / "raw" / "polyester" / "polyester_usd_monthly.csv"
    out_path = root / "data" / "raw" / "polyester" / "polyester_pkr_monthly.csv"

    if not poly_usd_path.exists():
        raise FileNotFoundError(f"Missing input file: {poly_usd_path}")

    poly = pd.read_csv(poly_usd_path)
    poly["timestamp"] = pd.to_datetime(poly["timestamp"], errors="coerce")
    poly = poly.dropna(subset=["timestamp"]).copy()

    # Bucket to month-start for stable joins
    poly["timestamp"] = poly["timestamp"].dt.to_period("M").dt.to_timestamp()
    if "price_usd" not in poly.columns:
        raise ValueError("Expected column 'price_usd' in polyester_usd_monthly.csv")

    poly = poly.groupby("timestamp", as_index=False).agg(price_usd=("price_usd", "mean"))

    # Try common USD/PKR series IDs on FRED
    candidate_series = [
        "DEXPKUS",            # FRED daily: Pakistan Rupees per U.S. Dollar
        "PAKEXCHUSDM",        # alt naming (may not exist)
        "CCUSMA02PKM618N",    # possible OECD/MEI series (may not exist)
    ]

    fx = None
    last_error: Exception | None = None
    for sid in candidate_series:
        try:
            fx = _fetch_fred_monthly(sid)
            # crude plausibility checks
            if len(fx) >= 12 and fx["usd_pkr"].mean() > 10:
                print(f"Using USD/PKR series from FRED: {sid} ({len(fx)} rows)")
                break
            fx = None
        except Exception as e:
            last_error = e
            fx = None

    if fx is None:
        rate = _fetch_latest_usd_pkr()
        print(
            "FRED USD/PKR series not available; falling back to latest live USD/PKR rate "
            f"(applied to all months): {rate:.4f}"
        )
        fx = poly[["timestamp"]].drop_duplicates().copy()
        fx["usd_pkr"] = rate

    merged = poly.merge(fx, on="timestamp", how="left")
    merged["usd_pkr"] = merged["usd_pkr"].ffill()
    merged = merged.dropna(subset=["usd_pkr"]).copy()

    merged["price_pkr"] = merged["price_usd"] * merged["usd_pkr"]

    out = merged[["timestamp", "price_pkr"]].copy()
    out = out.sort_values("timestamp")

    # Basic validation
    if out.empty:
        raise RuntimeError("Output is empty; check input series overlap")
    if not out["price_pkr"].notna().all():
        raise RuntimeError("Output contains NaNs")
    if not (out["price_pkr"] > 0).all():
        raise RuntimeError("Output contains non-positive values")

    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Wrote: {out_path} (rows={len(out)})")
    print(out.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
