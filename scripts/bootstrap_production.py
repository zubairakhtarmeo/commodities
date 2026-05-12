"""
bootstrap_production.py -Production health check for the commodity intelligence platform.

Validates environment, Supabase tables, data freshness, forecast completeness,
and deployment readiness. Safe to run at any time -read-only, no side effects.

Usage:
    python scripts/bootstrap_production.py
    python scripts/bootstrap_production.py --no-color
    python scripts/bootstrap_production.py --fail-fast
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

BASE_DIR = Path(__file__).parent.parent

# ── Credentials ───────────────────────────────────────────────────────────────

def _load_credentials() -> tuple[str | None, str | None]:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

    if not url or not key:
        secrets_path = BASE_DIR / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            try:
                try:
                    import tomllib
                    with open(secrets_path, "rb") as f:
                        s = tomllib.load(f)
                except ImportError:
                    import tomli
                    with open(secrets_path, "rb") as f:
                        s = tomli.load(f)
                url = url or s.get("SUPABASE_URL")
                key = key or s.get("SUPABASE_SERVICE_ROLE_KEY") or s.get("SUPABASE_ANON_KEY")
            except Exception:
                pass

    return (str(url).rstrip("/") if url else None,
            str(key).strip() if key else None)


# ── Output helpers ─────────────────────────────────────────────────────────────

_USE_COLOR = True

_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

_COUNTERS: dict[str, int] = {"OK": 0, "WARN": 0, "FAIL": 0}


def _c(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}" if _USE_COLOR else text


def _log(level: str, message: str) -> None:
    _COUNTERS[level] = _COUNTERS.get(level, 0) + 1
    color = {
        "OK":   _GREEN,
        "WARN": _YELLOW,
        "FAIL": _RED,
    }.get(level, "")
    tag = _c(f"[{level:4s}]", color + (_BOLD if level == "FAIL" else ""))
    print(f"  {tag} {message}")


def _section(title: str) -> None:
    print()
    print(_c(f"-- {title} {'-' * (54 - len(title))}", _BOLD))


# ── Supabase REST helpers ──────────────────────────────────────────────────────

def _headers(key: str) -> dict[str, str]:
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def _select(url: str, key: str, table: str, params: dict,
            timeout: int = 15) -> tuple[list[dict] | None, str | None]:
    try:
        r = requests.get(
            f"{url}/rest/v1/{table}",
            headers=_headers(key),
            params=params,
            timeout=timeout,
        )
        if not r.ok:
            return None, f"HTTP {r.status_code}: {r.text[:200]}"
        data = r.json()
        return (data if isinstance(data, list) else []), None
    except Exception as e:
        return None, str(e)


def _count(url: str, key: str, table: str,
           extra_params: dict | None = None) -> tuple[int | None, str | None]:
    """Return exact row count via PostgREST Content-Range header."""
    params = {"select": "id", **(extra_params or {}), "limit": "1"}
    try:
        r = requests.get(
            f"{url}/rest/v1/{table}",
            headers={**_headers(key), "Prefer": "count=exact"},
            params=params,
            timeout=15,
        )
        if not r.ok:
            return None, f"HTTP {r.status_code}: {r.text[:200]}"
        cr = r.headers.get("Content-Range", "")
        # Format: "0-0/12482" or "*/0" when empty
        if "/" in cr:
            total = cr.split("/")[-1]
            if total == "*":
                return 0, None
            return int(total), None
        return len(r.json()), None
    except Exception as e:
        return None, str(e)


def _latest_date(url: str, key: str, table: str,
                 date_col: str = "date",
                 extra_params: dict | None = None) -> tuple[str | None, str | None]:
    params = {"select": date_col, "order": f"{date_col}.desc", "limit": "1",
              **(extra_params or {})}
    rows, err = _select(url, key, table, params)
    if err:
        return None, err
    if not rows:
        return None, None
    return rows[0].get(date_col), None


# ── Individual check groups ────────────────────────────────────────────────────

EXPECTED_COMMODITIES = [
    "cotton_usd", "cotton_pkr",
    "crude_oil_usd", "crude_oil_pkr",
    "natural_gas_usd", "natural_gas_pkr",
    "polyester_usd", "polyester_pkr",
    "viscose_usd", "viscose_pkr",
]

FORECAST_COMMODITIES = [
    "cotton_usd", "crude_oil_usd", "natural_gas_usd",
    "polyester_usd", "viscose_usd",
]

REQUIRED_HORIZONS = [1, 3, 6]

STALE_WARN_DAYS  = 35   # warn if latest data is older than this
STALE_FAIL_DAYS  = 60   # fail if older than this
FORECAST_MAX_AGE = 45   # days since forecast was generated

# Source classification — determines staleness expectations and warnings.
# "fred"     : FRED API, typically 4-6 week publication lag; max delay ~70 days acceptable
# "futures"  : Futures contract CSV with forward dates beyond today; don't flag as stale
# "static"   : No live feed; CSV-only; warn if older than 120 days
# "live"     : Live API with daily/weekly updates; standard thresholds apply
COMMODITY_SOURCES: dict[str, str] = {
    "cotton_usd":      "fred",     # FRED PCOTTINDUSDM, ~4-6 week lag
    "cotton_pkr":      "fred",     # derived from cotton_usd
    "crude_oil_usd":   "live",     # FRED DCOILBRENTEU, daily data
    "crude_oil_pkr":   "live",
    "natural_gas_usd": "live",     # FRED DHHNGSP, daily data
    "natural_gas_pkr": "live",
    "polyester_usd":   "futures",  # Futures CSV, includes forward contract months
    "polyester_pkr":   "futures",
    "viscose_usd":     "static",   # SunSirs scraper dead; CSV baseline only (Jan 2026)
    "viscose_pkr":     "static",   # derived from viscose_usd
}


def check_environment() -> None:
    _section("Environment")

    url, key = _load_credentials()

    if url:
        _log("OK", f"SUPABASE_URL found ({url[:40]}{'…' if len(url) > 40 else ''})")
    else:
        _log("FAIL", "SUPABASE_URL missing -set env var or .streamlit/secrets.toml")

    if key:
        _log("OK", f"SUPABASE_SERVICE_ROLE_KEY found ({'*' * 8}{key[-4:]})")
    else:
        _log("FAIL", "SUPABASE_SERVICE_ROLE_KEY missing")

    # Required Python dependencies
    missing_deps = []
    for pkg in ("requests", "pandas", "streamlit"):
        try:
            __import__(pkg)
        except ImportError:
            missing_deps.append(pkg)
    if missing_deps:
        _log("FAIL", f"Missing Python packages: {', '.join(missing_deps)}")
    else:
        _log("OK", "Core Python dependencies present")

    return url, key


def check_connectivity(url: str, key: str) -> bool:
    _section("Supabase Connectivity")
    try:
        r = requests.get(
            f"{url}/rest/v1/",
            headers=_headers(key),
            timeout=10,
        )
        if r.ok:
            _log("OK", f"Supabase REST API reachable ({r.status_code})")
            return True
        else:
            _log("FAIL", f"Supabase REST returned {r.status_code}: {r.text[:100]}")
            return False
    except Exception as e:
        _log("FAIL", f"Cannot reach Supabase: {e}")
        return False


def check_tables(url: str, key: str) -> None:
    _section("Database Tables")

    tables = [
        ("commodity_prices",          "commodity,date,value"),
        ("prediction_records",        "commodity,target_date,predicted_value"),
        ("cotton_country_prices",     "date,country,price_usd_per_lb"),
        ("cotton_country_predictions","country,target_date,predicted_usd_per_lb"),
    ]

    for table, cols in tables:
        count, err = _count(url, key, table)
        if err:
            _log("FAIL", f"{table}: {err}")
        elif count == 0:
            _log("WARN", f"{table}: table exists but is EMPTY")
        else:
            _log("OK", f"{table}: {count:,} rows")


def check_commodity_data(url: str, key: str) -> None:
    _section("Commodity Data (commodity_prices)")

    today = date.today()

    for commodity in EXPECTED_COMMODITIES:
        source_type = COMMODITY_SOURCES.get(commodity, "live")

        # Check for presence + latest date directly per commodity (avoids sampling gaps)
        latest, err = _latest_date(url, key, "commodity_prices",
                                   extra_params={"commodity": f"eq.{commodity}"})
        if err:
            _log("FAIL", f"{commodity}: query error - {err}")
            continue
        if latest is None:
            _log("WARN" if source_type == "static" else "FAIL",
                 f"{commodity}: no rows in commodity_prices")
            continue

        try:
            latest_dt = datetime.strptime(latest[:10], "%Y-%m-%d").date()
        except ValueError:
            _log("WARN", f"{commodity}: unparseable date '{latest}'")
            continue

        age_days = (today - latest_dt).days

        if source_type == "futures":
            # Futures data legitimately contains forward contract months
            if age_days < -90:
                _log("WARN", f"{commodity}: futures data extends {-age_days}d ahead "
                              f"(latest {latest_dt}) - verify source CSV is current")
            else:
                _log("OK", f"{commodity}: latest {latest_dt} [futures data, fwd contract]")

        elif source_type == "fred":
            # FRED primary commodity prices publish with ~4-6 week delay
            fred_warn = 75   # warn if older than 75 days (2.5× normal lag)
            fred_fail = 120  # fail if older than 4 months (clearly stale)
            if age_days > fred_fail:
                _log("FAIL", f"{commodity}: {age_days}d old ({latest_dt}) "
                              "[FRED source - check if ingestion is running]")
            elif age_days > fred_warn:
                _log("WARN", f"{commodity}: {age_days}d old ({latest_dt}) "
                              "[FRED lag expected up to ~70d; this is borderline]")
            else:
                _log("OK", f"{commodity}: latest {latest_dt} ({age_days}d ago) [FRED]")

        elif source_type == "static":
            # Static/CSV-only sources; warn once they're significantly stale
            static_warn = 120
            static_fail = 365
            if age_days > static_fail:
                _log("FAIL", f"{commodity}: {age_days}d old ({latest_dt}) "
                              "[static source - manual update required]")
            elif age_days > static_warn:
                _log("WARN", f"{commodity}: {age_days}d old ({latest_dt}) "
                              "[static/CSV source - no live feed available]")
            else:
                _log("OK", f"{commodity}: latest {latest_dt} ({age_days}d ago) [static]")

        else:  # live
            if age_days > STALE_FAIL_DAYS:
                _log("FAIL", f"{commodity}: {age_days}d old ({latest_dt}) "
                              "- check ingestion pipeline")
            elif age_days > STALE_WARN_DAYS:
                _log("WARN", f"{commodity}: {age_days}d old ({latest_dt})")
            else:
                _log("OK", f"{commodity}: latest {latest_dt} ({age_days}d ago)")

    # Duplicate detection (sample up to 5000 rows)
    dupes, err3 = _select(url, key, "commodity_prices",
                          {"select": "commodity,date", "limit": "5000"})
    if dupes and not err3:
        seen: dict[tuple, int] = {}
        for r in dupes:
            k = (r.get("commodity"), r.get("date"))
            seen[k] = seen.get(k, 0) + 1
        dup_count = sum(1 for v in seen.values() if v > 1)
        if dup_count:
            _log("WARN", f"commodity_prices: {dup_count} duplicate (commodity, date) pairs")
        else:
            _log("OK", "commodity_prices: no duplicate (commodity, date) pairs")

def check_forecasts(url: str, key: str) -> None:
    _section("Forecast Records (prediction_records)")

    today = date.today()

    # Check column name: some deployments use 'asset', some use 'commodity'
    rows, err = _select(url, key, "prediction_records", {"select": "commodity", "limit": "1"})
    commodity_col = "commodity"
    if err and "column" in err.lower():
        rows, err = _select(url, key, "prediction_records", {"select": "asset", "limit": "1"})
        commodity_col = "asset"

    if err:
        _log("FAIL", f"Cannot read prediction_records: {err}")
        return

    for commodity in FORECAST_COMMODITIES:
        rows, err2 = _select(url, key, "prediction_records", {
            "select": f"{commodity_col},target_date,predicted_value,horizon_months",
            commodity_col: f"eq.{commodity}",
            "order": "target_date.desc",
            "limit": "20",
        })
        if err2:
            _log("WARN", f"{commodity} forecasts: {err2}")
            continue

        if not rows:
            _log("FAIL", f"{commodity}: no forecast rows in prediction_records")
            continue

        # Check horizons present
        horizons_found = {r.get("horizon_months") for r in rows
                          if r.get("horizon_months") is not None}
        missing_h = [h for h in REQUIRED_HORIZONS if h not in horizons_found]
        if missing_h:
            _log("WARN", f"{commodity}: missing horizons {missing_h}")

        # Check latest forecast target date
        latest_target = rows[0].get("target_date", "")
        try:
            latest_dt = datetime.strptime(latest_target[:10], "%Y-%m-%d").date()
            if latest_dt < today:
                _log("WARN", f"{commodity}: latest forecast target {latest_dt} is in the past")
            else:
                _log("OK", f"{commodity}: forecasts OK, latest target {latest_dt}, "
                            f"horizons {sorted(horizons_found)}")
        except (ValueError, TypeError):
            _log("WARN", f"{commodity}: unparseable target_date '{latest_target}'")

        # Check for NaN/null predicted values
        null_count = sum(1 for r in rows if r.get("predicted_value") is None)
        if null_count:
            _log("WARN", f"{commodity}: {null_count} null predicted_value rows in sample")

        # Sanity check: no value should be negative or astronomically large
        values = [r["predicted_value"] for r in rows
                  if r.get("predicted_value") is not None]
        if values:
            min_v, max_v = min(values), max(values)
            if min_v <= 0:
                _log("WARN", f"{commodity}: predicted_value <= 0 found (min={min_v:.4f})")
            elif max_v / (min_v + 1e-9) > 100:
                _log("WARN", f"{commodity}: suspicious value spread "
                              f"min={min_v:.4f} max={max_v:.4f}")


_LIVE_COUNTRY_SOURCES      = {"FRED/ICE-No2", "CEPEA/ESALQ"}
_ESTIMATED_COUNTRY_SOURCES = {"ICE/Brazil-basis", "ICE/Pakistan-domestic", "ICE/Turkey-import"}
_REGIONAL_COUNTRY_SOURCES  = {
    "regional/EastAfrica", "regional/WestAfrica-CIV",
    "regional/EastAfrica-SDN", "regional/WestAfrica",
}
# Kept for backwards compatibility with existing callers
_REAL_COUNTRY_SOURCES = _LIVE_COUNTRY_SOURCES | _ESTIMATED_COUNTRY_SOURCES | _REGIONAL_COUNTRY_SOURCES


def check_country_cotton(url: str, key: str) -> None:
    _section("Country Cotton Data")

    today = date.today()

    # total row count
    count, err = _count(url, key, "cotton_country_prices")
    if err:
        _log("FAIL", f"cotton_country_prices: {err}")
    elif count == 0:
        _log("FAIL", "cotton_country_prices: EMPTY - run scripts/ingest_cotton_countries.py")
    else:
        _log("OK", f"cotton_country_prices: {count:,} rows")

    # Per-country source audit (fetch recent rows with source field)
    rows, _ = _select(url, key, "cotton_country_prices",
                      {"select": "country,source,date", "limit": "2000",
                       "order": "date.desc"})
    if rows:
        by_country: dict[str, dict] = {}
        for r in rows:
            c = r.get("country", "")
            if c and c not in by_country:
                by_country[c] = r   # first occurrence = latest date per country

        live_countries: list[str] = []
        estimated_countries: list[str] = []
        regional_countries: list[str] = []
        sample_countries: list[str] = []

        for country, latest_row in sorted(by_country.items()):
            src = str(latest_row.get("source", ""))
            dt_str = str(latest_row.get("date", ""))[:10]
            try:
                age = (today - datetime.strptime(dt_str, "%Y-%m-%d").date()).days
                age_label = f"{age}d ago"
            except ValueError:
                age_label = dt_str

            if src in _LIVE_COUNTRY_SOURCES:
                live_countries.append(country)
                _log("OK",   f"  {country}: LIVE [{src}], latest {dt_str} ({age_label})")
            elif src in _ESTIMATED_COUNTRY_SOURCES:
                estimated_countries.append(country)
                _log("OK",   f"  {country}: ESTIMATED [{src}], latest {dt_str} ({age_label})")
            elif src in _REGIONAL_COUNTRY_SOURCES:
                regional_countries.append(country)
                _log("OK",   f"  {country}: REGIONAL [{src}], latest {dt_str} ({age_label})")
            else:
                sample_countries.append(country)
                _log("WARN", f"  {country}: SAMPLE [{src}], latest {dt_str}")

        non_sample = live_countries + estimated_countries + regional_countries
        _log("OK" if non_sample else "WARN",
             f"Live: {live_countries}  Estimated: {estimated_countries}  Regional: {regional_countries}")
        if sample_countries:
            _log("WARN", f"Still SAMPLE (no connector yet): {sample_countries}")

    # cotton_country_predictions
    pred_count, err2 = _count(url, key, "cotton_country_predictions")
    if err2:
        _log("FAIL", f"cotton_country_predictions: {err2}")
    elif pred_count == 0:
        _log("WARN", "cotton_country_predictions: empty - run scripts/run_cotton_country_forecasts.py")
    else:
        _log("OK", f"cotton_country_predictions: {pred_count:,} rows")


def check_deployment_files() -> None:
    _section("Deployment Readiness")

    required_files = [
        BASE_DIR / "streamlit_app.py",
        BASE_DIR / "requirements.txt",
        BASE_DIR / ".streamlit" / "secrets.toml",
        BASE_DIR / "scripts" / "run_ingestion.py",
        BASE_DIR / "scripts" / "run_forecasts.py",
        BASE_DIR / "scripts" / "run_cotton_country_forecasts.py",
        BASE_DIR / "scripts" / "ingest_cotton_countries.py",
        BASE_DIR / "scripts" / "ingest_viscose.py",
        BASE_DIR / "scripts" / "ingest_country_sources.py",
        BASE_DIR / "country_sources" / "usa_cotton.py",
        BASE_DIR / "country_sources" / "_common.py",
        BASE_DIR / "country_sources" / "brazil.py",
        BASE_DIR / "country_sources" / "pakistan.py",
        BASE_DIR / "country_sources" / "turkey.py",
        BASE_DIR / "country_sources" / "tanzania.py",
        BASE_DIR / "country_sources" / "ivory_coast.py",
        BASE_DIR / "country_sources" / "sudan.py",
        BASE_DIR / "country_sources" / "west_africa.py",
        BASE_DIR / "src" / "processing" / "units.py",
    ]

    for f in required_files:
        rel = f.relative_to(BASE_DIR)
        if f.exists():
            _log("OK", str(rel))
        else:
            _log("FAIL", f"{rel} -FILE MISSING")

    # Warn about locally-only files that block production rendering
    local_only = [
        BASE_DIR / "data" / "processed" / "purchases_clean" / "purchases_cotton.csv",
    ]
    for f in local_only:
        rel = f.relative_to(BASE_DIR)
        if f.exists():
            _log("OK", f"{rel} (local only - not required in production)")
        else:
            # This is expected to be absent in production; not a failure
            pass


def check_evaluation_artifacts() -> None:
    _section("Forecast Evaluation Artifacts")

    eval_path = BASE_DIR / "artifacts" / "forecast_evaluation.json"
    if not eval_path.exists():
        _log("WARN", "artifacts/forecast_evaluation.json missing - run: python scripts/evaluate_forecasts.py")
        return

    import json as _json
    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            data = _json.load(f)
    except Exception as exc:
        _log("FAIL", f"Cannot parse forecast_evaluation.json: {exc}")
        return

    generated_at = data.get("generated_at", "")
    if generated_at:
        try:
            from datetime import datetime, timezone
            gen_dt  = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - gen_dt).days
            if age_days > 30:
                _log("WARN", f"Evaluation results are {age_days}d old (re-run evaluate_forecasts.py)")
            else:
                _log("OK",   f"Evaluation file present, {age_days}d old")
        except Exception:
            _log("OK", "Evaluation file present (age unknown)")
    else:
        _log("OK", "Evaluation file present")

    commodities = data.get("commodities", {})
    for commodity, cdata in commodities.items():
        if "error" in cdata:
            _log("WARN", f"  {commodity}: evaluation error - {cdata['error']}")
            continue
        # Check if any ML model beats the last_value baseline at h1
        h1 = cdata.get("h1", {})
        baseline_mae = h1.get("last_value", {}).get("mae")
        ml_models    = ["linear_ridge", "random_forest", "ridge_returns", "rf_returns"]
        any_beats    = any(
            h1.get(m, {}).get("mae") is not None
            and h1.get(m, {}).get("mae") < (baseline_mae or float("inf"))
            for m in ml_models
        )
        mape_best = min(
            (h1.get(m, {}).get("mape") or 999 for m in ml_models),
            default=None,
        )
        if baseline_mae is None:
            _log("WARN", f"  {commodity}: no h1 evaluation data")
        elif any_beats:
            label = f"MAPE {mape_best:.1f}%" if mape_best and mape_best < 900 else ""
            _log("OK",   f"  {commodity}: ML beats baseline at h1  {label}".rstrip())
        else:
            _log("WARN", f"  {commodity}: no ML model beats last-value baseline at h1 (baseline MAE {baseline_mae:.4f})")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> int:
    global _USE_COLOR

    parser = argparse.ArgumentParser(
        description="Production health check for the commodity intelligence platform"
    )
    parser.add_argument("--no-color",  action="store_true", help="Disable ANSI colors")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop after first FAIL instead of running all checks")
    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        _USE_COLOR = False

    print()
    print(_c("=" * 60, _BOLD))
    print(_c("  Commodity Intelligence Platform - Production Bootstrap", _BOLD))
    print(_c(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", _BOLD))
    print(_c("=" * 60, _BOLD))

    # Environment (also returns credentials for downstream checks)
    result = check_environment()
    url, key = result if isinstance(result, tuple) else (None, None)

    if args.fail_fast and _COUNTERS.get("FAIL", 0):
        _print_summary()
        return 1

    if not url or not key:
        print()
        print(_c("  Cannot continue without Supabase credentials.", _RED))
        _print_summary()
        return 1

    reachable = check_connectivity(url, key)

    if args.fail_fast and _COUNTERS.get("FAIL", 0):
        _print_summary()
        return 1

    if reachable:
        check_tables(url, key)

        if not (args.fail_fast and _COUNTERS.get("FAIL", 0)):
            check_commodity_data(url, key)

        if not (args.fail_fast and _COUNTERS.get("FAIL", 0)):
            check_forecasts(url, key)

        if not (args.fail_fast and _COUNTERS.get("FAIL", 0)):
            check_country_cotton(url, key)

    check_deployment_files()
    check_evaluation_artifacts()

    _print_summary()
    return 1 if _COUNTERS.get("FAIL", 0) else 0


def _print_summary() -> None:
    ok   = _COUNTERS.get("OK",   0)
    warn = _COUNTERS.get("WARN", 0)
    fail = _COUNTERS.get("FAIL", 0)

    print()
    print(_c("-" * 60, _BOLD))
    parts = [
        _c(f"{ok} OK",     _GREEN  + _BOLD),
        _c(f"{warn} WARN", _YELLOW + _BOLD),
        _c(f"{fail} FAIL", _RED    + _BOLD),
    ]
    print("  " + "   ".join(parts))
    print(_c("-" * 60, _BOLD))

    if fail:
        print(_c("  Platform has CRITICAL issues - fix FAIL items before deploying.", _RED + _BOLD))
    elif warn:
        print(_c("  Platform is operational with warnings.", _YELLOW))
    else:
        print(_c("  Platform looks healthy.", _GREEN + _BOLD))
    print()


if __name__ == "__main__":
    sys.exit(main())
