"""
Procurement signal computation.
All functions read from Supabase only.
No simulation. No hardcoded values.

supabase_cfg: tuple (url, key) as returned by get_supabase_config(), or None.
"""

import requests


def compute_mom_change(commodity_key: str, supabase_cfg) -> dict | None:
    """
    Compute month-over-month % change from commodity_prices.

    Returns dict or None if insufficient data.
    {
        "change_pct": float,
        "direction": "up" | "down" | "flat",
        "current_value": float,
        "previous_value": float,
        "current_date": str,
        "previous_date": str,
        "unit": str
    }
    """
    if not supabase_cfg:
        return None
    url, key = supabase_cfg
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }
    try:
        params = {
            "select": "date,value,unit",
            "commodity": f"eq.{commodity_key}",
            "order": "date.desc",
            "limit": "3",
        }
        resp = requests.get(
            f"{url}/rest/v1/commodity_prices",
            headers=headers,
            params=params,
            timeout=30,
        )
        if not resp.ok:
            return None
        rows = resp.json()
        if not isinstance(rows, list) or len(rows) < 2:
            return None

        rows = sorted(rows, key=lambda x: x["date"], reverse=True)
        current = rows[0]
        previous = rows[1]

        curr_val = float(current["value"])
        prev_val = float(previous["value"])

        if prev_val == 0:
            return None

        change_pct = ((curr_val - prev_val) / prev_val) * 100

        if change_pct > 0.5:
            direction = "up"
        elif change_pct < -0.5:
            direction = "down"
        else:
            direction = "flat"

        return {
            "change_pct": round(change_pct, 2),
            "direction": direction,
            "current_value": curr_val,
            "previous_value": prev_val,
            "current_date": current["date"],
            "previous_date": previous["date"],
            "unit": current.get("unit", ""),
        }
    except Exception as e:
        print(f"[signals] compute_mom_change error for {commodity_key}: {e}")
        return None


def compute_forecast_signal(
    commodity_key: str,
    horizon_months: int,
    supabase_cfg,
    current_value: float = None,
) -> dict | None:
    """
    Compute expected % change from current price to ML forecast.
    Fetches by commodity + horizon only — works with any model stored in DB.

    Returns dict or None if no forecast available.
    {
        "forecast_change_pct": float,
        "direction": "up" | "down" | "flat",
        "forecast_value": float,
        "model_name": str,
        "target_date": str,
        "horizon_months": int
    }
    """
    if not supabase_cfg:
        return None
    url, key = supabase_cfg
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }
    try:
        params = {
            "select": "*",
            "commodity": f"eq.{commodity_key}",
            "horizon_months": f"eq.{horizon_months}",
            "is_demo": "eq.false",
            "order": "created_at.desc",
            "limit": "1",
        }
        resp = requests.get(
            f"{url}/rest/v1/prediction_records",
            headers=headers,
            params=params,
            timeout=30,
        )
        if not resp.ok or not isinstance(resp.json(), list) or not resp.json():
            return None

        row = resp.json()[0]
        forecast_val = float(row["predicted_value"])

        if current_value and current_value > 0:
            change_pct = ((forecast_val - current_value) / current_value) * 100
        else:
            change_pct = 0.0

        if change_pct > 0.5:
            direction = "up"
        elif change_pct < -0.5:
            direction = "down"
        else:
            direction = "flat"

        return {
            "forecast_change_pct": round(change_pct, 2),
            "direction": direction,
            "forecast_value": forecast_val,
            "model_name": row["model_name"],
            "target_date": row["target_date"],
            "horizon_months": horizon_months,
        }
    except Exception as e:
        print(f"[signals] compute_forecast_signal error for {commodity_key}: {e}")
        return None


def get_procurement_signal(
    mom: dict | None,
    forecast: dict | None,
) -> tuple[str, str]:
    """
    Rule-based procurement signal from MoM + forecast data.

    Returns (signal_label, explanation).

    Rules:
    - Both trending UP   >2%: "BUY NOW"
    - Both trending DOWN >2%: "WAIT"
    - Mixed or <2% change:   "MONITOR"
    - Either input None:     "INSUFFICIENT DATA"
    """
    if mom is None or forecast is None:
        return (
            "INSUFFICIENT DATA",
            "Not enough historical or forecast data to generate signal.",
        )

    mom_up   = mom["direction"] == "up"   and abs(mom["change_pct"]) >= 2
    mom_down = mom["direction"] == "down" and abs(mom["change_pct"]) >= 2
    fc_up    = forecast["direction"] == "up"   and abs(forecast["forecast_change_pct"]) >= 2
    fc_down  = forecast["direction"] == "down" and abs(forecast["forecast_change_pct"]) >= 2

    if mom_up and fc_up:
        return (
            "BUY NOW",
            "Both recent trend and forecast indicate rising prices. "
            "Consider forward buying to lock in current rates.",
        )
    elif mom_down and fc_down:
        return (
            "WAIT",
            "Both recent trend and forecast indicate falling prices. "
            "Delay purchases where operationally possible.",
        )
    else:
        return (
            "MONITOR",
            "Mixed or unclear price signals. "
            "Review again next month before committing to large purchases.",
        )
