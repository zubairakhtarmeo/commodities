"""
Live Commodity Data Ingestion (Phase 2)
Explicitly fetches and updates commodity prices to Supabase.
"""
import os
import sys
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path
import logging
import smtplib
from email.mime.text import MIMEText

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Set up file logging (not just print)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            LOG_DIR / f"ingestion_{datetime.today().strftime('%Y%m%d')}.log",
            encoding="utf-8"
        )
    ]
)

sys.path.insert(0, str(BASE_DIR))

from src.processing.units import (
    usd_to_pkr, lb_to_maund, ton_to_kg, rmb_to_usd, get_unit, COMMODITY_UNITS
)
from src.processing.validation import validate_dataframe

def get_supabase_credentials():
    # Try environment variables first
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    # Fallback to .streamlit/secrets.toml
    if not url or not key:
        try:
            import toml
            secrets = toml.load(BASE_DIR / ".streamlit/secrets.toml")
            url = secrets.get("SUPABASE_URL")
            key = secrets.get("SUPABASE_SERVICE_ROLE_KEY")
        except Exception as e:
            print("⚠️ Failed to load secrets.toml:", e)

    return url, key

SUPABASE_URL, SUPABASE_KEY = get_supabase_credentials()

print("DEBUG SUPABASE URL:", SUPABASE_URL)
print("DEBUG SUPABASE KEY:", "FOUND" if SUPABASE_KEY else "MISSING")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials not found. Check environment variables or secrets.toml")

from supabase import create_client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def push_to_supabase(df: pd.DataFrame, commodity: str, unit: str, source: str, derived_flag: bool = False, derived_from: str = None) -> bool:
    if df is None or df.empty:
        return False
    records = []
    from datetime import timezone
    created_at = datetime.now(timezone.utc).isoformat()
    for _, row in df.iterrows():
        val = float(row['value']) if pd.notna(row['value']) else None
        if val is None:
            continue
        record = {
            "commodity": commodity,
            "date": pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d'),
            "value": val,
            "source": source,
            "unit": unit,
            "derived_flag": derived_flag,
            "derived_from": derived_from,
            "created_at": created_at
        }
        records.append(record)
    if not records:
        print(f"  ✗ No valid records to push for {commodity}")
        return False
    print(f"  Pushing {len(records)} records to Supabase...")
    try:
        supabase.table("commodity_prices").upsert(records, on_conflict="commodity,date").execute()
        success = True
    except Exception as e:
        success = False
        print(f"  ✗ Supabase exception: {e}")

    if success:
        print(f"  ✓ Up-to-date in Supabase ({commodity})")
    else:
        print(f"  ✗ Supabase upsert failed ({commodity})")
    return success

def save_to_csv(df: pd.DataFrame, file_path: Path):
    if df is None or df.empty:
        return
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)

def fetch_exchange_rates():
    print("\n💱 Fetching exchange rates...")
    rates = {'PKR': 278.5, 'CNY': 7.25}
    try:
        url = "https://open.er-api.com/v6/latest/USD"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'PKR' in data.get('rates', {}): rates['PKR'] = data['rates']['PKR']
            if 'CNY' in data.get('rates', {}): rates['CNY'] = data['rates']['CNY']
            print(f"✓ Rates: {rates['PKR']:.2f} PKR/USD, {rates['CNY']:.2f} CNY/USD")
    except Exception as e:
        print(f"✗ Rate fetch error: {e}. Using fallbacks.")
    return rates

def ingest_cotton(rates: dict):
    print("\n📊 Ingesting Cotton...")
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=PCOTTINDUSDM"
        response = requests.get(url, timeout=10)
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        df.columns = ['timestamp', 'value']
        df['value'] = pd.to_numeric(df['value'], errors='coerce') / 100
        df = df.dropna()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[df['timestamp'] >= '2020-01-01']
        push_to_supabase(df, 'cotton_usd', COMMODITY_UNITS.get('cotton_usd', 'USD/lb'), 'FRED', False)
        save_to_csv(df, DATA_DIR / "cotton" / "cotton_usd_monthly.csv")
        df_pkr = df.copy()
        df_pkr['value'] = lb_to_maund(usd_to_pkr(df_pkr['value'], rates['PKR']))
        push_to_supabase(df_pkr, 'cotton_pkr', COMMODITY_UNITS.get('cotton_pkr', 'PKR/maund'), 'FRED', True, 'cotton_usd + FX')
        save_to_csv(df_pkr, DATA_DIR / "cotton" / "cotton_pkr_monthly.csv")
    except Exception as e:
        raise RuntimeError(f"Cotton ingestion failed: {e}")

def ingest_crude_oil(rates: dict):
    print("\n📊 Ingesting Crude Oil...")
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"
        response = requests.get(url, timeout=10)
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        df.columns = ['timestamp', 'value']
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[df['timestamp'] >= '2020-01-01']
        df = df.set_index('timestamp').resample('MS').last().reset_index().dropna()
        push_to_supabase(df, 'crude_oil_usd', COMMODITY_UNITS.get('crude_oil_usd', 'USD/barrel'), 'FRED', False)
        save_to_csv(df, DATA_DIR / "energy" / "crude_oil_brent_usd_monthly_clean.csv")
        df_pkr = df.copy()
        df_pkr['value'] = usd_to_pkr(df_pkr['value'], rates['PKR'])
        push_to_supabase(df_pkr, 'crude_oil_pkr', COMMODITY_UNITS.get('crude_oil_pkr', 'PKR/barrel'), 'FRED', True, 'crude_oil_usd + FX')
        save_to_csv(df_pkr, DATA_DIR / "energy" / "crude_oil_brent_pkr_monthly_clean.csv")
    except Exception as e:
        raise RuntimeError(f"Crude Oil ingestion failed: {e}")

def ingest_natural_gas(rates: dict):
    print("\n📊 Ingesting Natural Gas...")
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DHHNGSP"
        response = requests.get(url, timeout=10)
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        df.columns = ['timestamp', 'value']
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[df['timestamp'] >= '2020-01-01']
        df = df.set_index('timestamp').resample('MS').mean().reset_index().dropna()
        push_to_supabase(df, 'natural_gas_usd', COMMODITY_UNITS.get('natural_gas_usd', 'USD/MMBTU'), 'FRED', False)
        save_to_csv(df, DATA_DIR / "energy" / "natural_gas_usd_monthly_clean.csv")
        df_pkr = df.copy()
        df_pkr['value'] = usd_to_pkr(df_pkr['value'], rates['PKR'])
        push_to_supabase(df_pkr, 'natural_gas_pkr', COMMODITY_UNITS.get('natural_gas_pkr', 'PKR/MMBTU'), 'FRED', True, 'natural_gas_usd + FX')
        save_to_csv(df_pkr, DATA_DIR / "energy" / "natural_gas_pkr_monthly_clean.csv")
    except Exception as e:
        raise RuntimeError(f"Natural Gas ingestion failed: {e}")

def ingest_polyester(rates: dict):
    print("\n📊 Ingesting Polyester...")
    try:
        csv_path = DATA_DIR / "polyester" / "polyester_futures_monthly_clean.csv"
        if not csv_path.exists():
            print(f"✗ Missing source file: {csv_path}")
            return
        df_rmb = pd.read_csv(csv_path)
        df_rmb.columns = ['timestamp', 'value']
        df_rmb['timestamp'] = pd.to_datetime(df_rmb['timestamp'])
        df_rmb = df_rmb.dropna()
        df_usd = df_rmb.copy()
        df_usd['value'] = ton_to_kg(rmb_to_usd(df_usd['value'], rates['CNY']))
        push_to_supabase(df_usd, 'polyester_usd', COMMODITY_UNITS.get('polyester_usd', 'USD/kg'), 'CSV_futures', True, 'RMB_futures + FX')
        save_to_csv(df_usd, DATA_DIR / "polyester" / "polyester_usd_monthly.csv")
        df_pkr = df_usd.copy()
        df_pkr['value'] = usd_to_pkr(df_pkr['value'], rates['PKR'])
        push_to_supabase(df_pkr, 'polyester_pkr', COMMODITY_UNITS.get('polyester_pkr', 'PKR/kg'), 'CSV_futures', True, 'polyester_usd + FX')
        save_to_csv(df_pkr, DATA_DIR / "polyester" / "polyester_pkr_monthly.csv")
    except Exception as e:
        raise RuntimeError(f"Polyester ingestion failed: {e}")

def ingest_viscose(rates: dict):
    logging.info("\n📊 Ingesting Viscose...")
    logging.warning("⚠️ Skipping Viscose as requested. Data must be updated manually.")
    return "⚠️ skipped"

def send_failure_alert(failures: dict):
    """Send email if any ingestion fails."""
    if not os.getenv("ALERT_EMAIL"):
        return    # skip if not configured

    body = "Commodity ingestion failures:\n\n"
    for commodity, error in failures.items():
        body += f"  {commodity}: {error}\n"

    msg = MIMEText(body)
    msg["Subject"] = "⚠️ Commodity System: Ingestion Failed"
    msg["From"]    = os.getenv("ALERT_FROM_EMAIL") or "system@yourdomain.com"
    msg["To"]      = os.getenv("ALERT_EMAIL")

    try:
        with smtplib.SMTP(os.getenv("SMTP_HOST", "smtp.gmail.com"), 587) as s:
            s.starttls()
            s.login(
                os.getenv("ALERT_FROM_EMAIL", ""),
                os.getenv("ALERT_EMAIL_PASSWORD", "")
            )
            s.send_message(msg)
    except Exception as e:
        logging.warning(f"Alert email failed to send: {e}")

def run_all():
    logging.info("=" * 50)
    logging.info("🚀 STARTING COMMODITY INGESTION (PHASE 5 AUTOMATION)")
    logging.info("=" * 50)
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        logging.warning("⚠️  Warning: Supabase is NOT configured. Data will only be saved to CSVs.")
    
    rates = fetch_exchange_rates()
    
    results = {}
    tasks = [
        ("Cotton", lambda: ingest_cotton(rates)),
        ("Crude Oil", lambda: ingest_crude_oil(rates)),
        ("Natural Gas", lambda: ingest_natural_gas(rates)),
        ("Polyester", lambda: ingest_polyester(rates)),
        ("Viscose", lambda: ingest_viscose(rates))
    ]
    
    for name, task in tasks:
        try:
            task()
            results[name] = "✅ success"
            logging.info(f"{name}: success")
        except Exception as e:
            results[name] = f"❌ failed: {e}"
            logging.error(f"{name}: {e}")

    # Print clean summary at end
    logging.info("\n=== INGESTION SUMMARY ===")
    for k, v in results.items():
        logging.info(f"  {k}: {v}")

    # Exit with error code if any commodity failed
    failures = {k: v for k, v in results.items() if "❌" in v}
    if failures:
        send_failure_alert(failures)
        sys.exit(1)

if __name__ == "__main__":
    run_all()

