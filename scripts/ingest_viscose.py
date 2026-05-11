"""
Production viscose (VSF) ingestion pipeline.

Instrument:  China Viscose Staple Fiber (VSF) spot price
Source:      SunSirs (commodity_id=1057), RMB/ton daily → monthly
Persistence: CSV (data/raw/viscose/) + Supabase commodity_prices

Unit convention (matches dashboard ton_to_kg display logic):
  viscose_usd  → USD/ton  (dashboard ÷1000 → USD/kg for display)
  viscose_pkr  → PKR/ton  (dashboard ÷1000 → PKR/kg for display)

Design rules:
  - Never fabricate or back-fill data.
  - SunSirs failures are non-fatal: use CSV history as fallback.
  - All historical CSV data is preserved; new rows are merged.
  - Failures are reported clearly; no silent swallowing.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.processing.units import ton_to_kg, usd_to_pkr

logger = logging.getLogger(__name__)

VISCOSE_DIR = BASE_DIR / "data" / "raw" / "viscose"
USD_CSV = VISCOSE_DIR / "viscose_usd_monthly.csv"
PKR_CSV = VISCOSE_DIR / "viscose_pkr_monthly.csv"


# ── Supabase helpers ──────────────────────────────────────────────────────────

def _get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        try:
            import toml
            secrets = toml.load(BASE_DIR / ".streamlit" / "secrets.toml")
            url = url or secrets.get("SUPABASE_URL")
            key = key or secrets.get("SUPABASE_SERVICE_ROLE_KEY")
        except Exception:
            pass
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception as exc:
        logger.warning(f"Supabase client creation failed: {exc}")
        return None


def _push_to_supabase(
    supabase,
    df: pd.DataFrame,
    commodity: str,
    unit: str,
    source: str,
    derived_flag: bool = False,
    derived_from: str | None = None,
) -> bool:
    """Push a (timestamp, value) DataFrame to commodity_prices via upsert."""
    if supabase is None or df is None or df.empty:
        return False

    created_at = datetime.now(timezone.utc).isoformat()
    records = []
    for _, row in df.iterrows():
        val = float(row["value"]) if pd.notna(row["value"]) else None
        if val is None:
            continue
        records.append({
            "commodity": commodity,
            "date": pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d"),
            "value": val,
            "source": source,
            "unit": unit,
            "derived_flag": derived_flag,
            "derived_from": derived_from,
            "created_at": created_at,
        })

    if not records:
        logger.warning(f"  No valid records to push for {commodity}")
        return False

    try:
        supabase.table("commodity_prices").upsert(
            records, on_conflict="commodity,date"
        ).execute()
        logger.info(f"  ✓ {len(records)} rows pushed ({commodity})")
        return True
    except Exception as exc:
        logger.error(f"  ✗ Supabase push failed for {commodity}: {exc}")
        return False


# ── FX helpers ────────────────────────────────────────────────────────────────

def _fetch_fx_rates() -> dict[str, float]:
    """Return {PKR: float, CNY: float} from open.er-api or fallbacks."""
    fallbacks = {"PKR": 278.5, "CNY": 7.25}
    try:
        resp = requests.get("https://open.er-api.com/v6/latest/USD", timeout=8)
        rates = resp.json().get("rates", {})
        pkr = float(rates.get("PKR", fallbacks["PKR"]))
        cny = float(rates.get("CNY", fallbacks["CNY"]))
        logger.info(f"  FX: 1 USD = {pkr:.2f} PKR, {cny:.4f} CNY")
        return {"PKR": pkr, "CNY": cny}
    except Exception as exc:
        logger.warning(f"  FX fetch failed ({exc}); using fallbacks {fallbacks}")
        return fallbacks


# ── SunSirs fetch ─────────────────────────────────────────────────────────────

def _fetch_sunsirs_recent(cny_rate: float) -> pd.DataFrame | None:
    """
    Fetch recent VSF daily data from SunSirs and aggregate to monthly.

    Returns DataFrame(timestamp, value) with USD/ton, or None on failure.
    """
    try:
        from src.forecasting.ingestion.sunsirs_connector import SunSirsConnector
        from src.forecasting.ingestion.aggregation import aggregate_daily_to_monthly
    except ImportError as exc:
        logger.warning(f"  SunSirs connector import failed: {exc}")
        return None

    try:
        connector = SunSirsConnector()
        df_daily = connector.fetch_daily_prices()
        logger.info(
            f"  SunSirs: {len(df_daily)} daily rows "
            f"({df_daily['timestamp'].min().date()} → {df_daily['timestamp'].max().date()})"
        )
    except Exception as exc:
        logger.warning(f"  SunSirs fetch failed: {exc}")
        return None

    try:
        # aggregate_daily_to_monthly expects columns: timestamp, price_rmb
        df_monthly_rmb = aggregate_daily_to_monthly(df_daily, value_col="price_rmb")

        # Convert RMB/ton → USD/ton
        df_monthly_rmb["value"] = df_monthly_rmb["value"] / cny_rate
        df_monthly_rmb = df_monthly_rmb[["timestamp", "value"]].copy()
        df_monthly_rmb["timestamp"] = pd.to_datetime(df_monthly_rmb["timestamp"])

        logger.info(
            f"  SunSirs aggregated: {len(df_monthly_rmb)} monthly rows "
            f"({df_monthly_rmb['timestamp'].min().date()} → "
            f"{df_monthly_rmb['timestamp'].max().date()})"
        )
        return df_monthly_rmb
    except Exception as exc:
        logger.warning(f"  SunSirs aggregation failed: {exc}")
        return None


# ── CSV history helpers ───────────────────────────────────────────────────────

def _load_usd_csv() -> pd.DataFrame:
    """Load the historical USD/ton baseline CSV. Returns empty DataFrame if missing."""
    if not USD_CSV.exists():
        logger.warning(f"  USD baseline CSV not found: {USD_CSV}")
        return pd.DataFrame(columns=["timestamp", "value"])
    df = pd.read_csv(USD_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["timestamp", "value"])
    logger.info(f"  CSV history: {len(df)} rows ({df['timestamp'].min().date()} → {df['timestamp'].max().date()})")
    return df


def _merge_and_deduplicate(
    historical: pd.DataFrame,
    recent: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Merge historical CSV with freshly-scraped monthly data.
    Recent data wins on any overlapping months.
    """
    if recent is None or recent.empty:
        return historical.sort_values("timestamp").reset_index(drop=True)

    combined = pd.concat([historical, recent], ignore_index=True)
    combined = combined.sort_values("timestamp")
    # Keep last (most recent fetch wins) for any duplicate month
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    combined = combined.reset_index(drop=True)
    logger.info(
        f"  Merged: {len(combined)} rows after dedup "
        f"({combined['timestamp'].min().date()} → {combined['timestamp'].max().date()})"
    )
    return combined


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_usd(df: pd.DataFrame) -> list[str]:
    """Run basic sanity checks on the USD/ton series. Returns list of warnings."""
    warnings: list[str] = []
    if df.empty:
        warnings.append("Series is empty.")
        return warnings

    if df["value"].isna().any():
        warnings.append(f"{df['value'].isna().sum()} NaN values in series.")

    min_v, max_v = float(df["value"].min()), float(df["value"].max())
    # Reasonable VSF spot range: 500–10000 USD/ton
    if min_v < 500:
        warnings.append(f"Suspicious min value: {min_v:.1f} USD/ton (expected ≥500)")
    if max_v > 10_000:
        warnings.append(f"Suspicious max value: {max_v:.1f} USD/ton (expected ≤10000)")

    # Check for large gaps (> 2 months)
    diffs = df.sort_values("timestamp")["timestamp"].diff().dt.days.dropna()
    large_gaps = (diffs > 62).sum()
    if large_gaps > 0:
        warnings.append(f"{large_gaps} gap(s) > 2 months in series.")

    return warnings


# ── Main entry point ──────────────────────────────────────────────────────────

def ingest_viscose(rates: dict) -> str:
    """
    Full viscose ingestion: fetch → merge → validate → push to Supabase + CSVs.

    Args:
        rates: dict with keys 'PKR' and 'CNY' (from fetch_exchange_rates).

    Returns:
        Status string for the ingestion summary.
    """
    logger.info("\n📊 Ingesting Viscose (VSF)...")
    VISCOSE_DIR.mkdir(parents=True, exist_ok=True)

    cny_rate: float = rates.get("CNY", 7.25)
    pkr_rate: float = rates.get("PKR", 278.5)

    # 1. Load historical baseline from CSV
    hist_usd = _load_usd_csv()

    # 2. Try SunSirs for recent months (non-fatal)
    recent_usd = _fetch_sunsirs_recent(cny_rate)

    # 3. Merge historical + recent
    df_usd = _merge_and_deduplicate(hist_usd, recent_usd)

    if df_usd.empty:
        msg = "⚠️ No viscose data available (CSV missing and SunSirs failed)"
        logger.warning(f"  {msg}")
        return msg

    # 4. Validate
    validation_warnings = _validate_usd(df_usd)
    for w in validation_warnings:
        logger.warning(f"  ⚠️ Validation: {w}")

    if df_usd.empty or len(df_usd) < 12:
        msg = f"⚠️ Insufficient data ({len(df_usd)} rows — need ≥12)"
        logger.warning(f"  {msg}")
        return msg

    # 5. Build PKR/ton series (USD/ton × PKR/USD = PKR/ton)
    df_pkr = df_usd.copy()
    df_pkr["value"] = usd_to_pkr(df_pkr["value"], pkr_rate)

    # 6. Save updated CSVs
    df_usd_out = df_usd[["timestamp", "value"]].copy()
    df_usd_out["timestamp"] = df_usd_out["timestamp"].dt.strftime("%Y-%m-%d")
    df_usd_out.to_csv(USD_CSV, index=False)
    logger.info(f"  ✓ CSV saved: {USD_CSV} ({len(df_usd_out)} rows, USD/ton)")

    df_pkr_out = df_pkr[["timestamp", "value"]].copy()
    df_pkr_out["timestamp"] = df_pkr_out["timestamp"].dt.strftime("%Y-%m-%d")
    df_pkr_out.to_csv(PKR_CSV, index=False)
    logger.info(f"  ✓ CSV saved: {PKR_CSV} ({len(df_pkr_out)} rows, PKR/ton)")

    # 7. Push to Supabase
    supabase = _get_supabase_client()
    if supabase is None:
        logger.warning("  ⚠️ Supabase not configured — CSV-only mode")
        return f"✅ CSV updated ({len(df_usd)} rows) | ⚠️ Supabase skipped (no credentials)"

    ok_usd = _push_to_supabase(
        supabase, df_usd_out,
        commodity="viscose_usd",
        unit="USD/ton",
        source="SunSirs+CSV",
        derived_flag=False,
    )
    ok_pkr = _push_to_supabase(
        supabase, df_pkr_out,
        commodity="viscose_pkr",
        unit="PKR/ton",
        source="SunSirs+CSV",
        derived_flag=True,
        derived_from="viscose_usd + FX",
    )

    status_parts = [f"{len(df_usd)} rows"]
    if ok_usd:
        status_parts.append("viscose_usd ✅")
    else:
        status_parts.append("viscose_usd ❌")
    if ok_pkr:
        status_parts.append("viscose_pkr ✅")
    else:
        status_parts.append("viscose_pkr ❌")

    status = "✅ " + " | ".join(status_parts) if (ok_usd and ok_pkr) else "⚠️ " + " | ".join(status_parts)
    logger.info(f"  {status}")
    return status


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from scripts.run_ingestion import fetch_exchange_rates
    rates = fetch_exchange_rates()
    result = ingest_viscose(rates)
    print(f"\nResult: {result}")
