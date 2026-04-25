"""Centralized unit + currency conversion helpers.

All conversion constants live here so ingestion/update scripts do not drift.
"""

# 1 maund = 40 kg (Pakistan standard) = 88.1849 lb
# Source: 40 * 2.20462262185
LB_PER_MAUND = 88.1849

KG_PER_TON = 1000.0

# Canonical (display/decision) units for the system.
# These are the units the dashboard *intends* to show, even if some raw files
# are stored in alternative units and converted at load time.
COMMODITY_UNITS = {
    "cotton_usd": "USD/lb",
    "cotton_pkr": "PKR/maund",
    "crude_oil_usd": "USD/barrel",
    "crude_oil_pkr": "PKR/barrel",
    "natural_gas_usd": "USD/MMBTU",
    "natural_gas_pkr": "PKR/MMBTU",
    "polyester_usd": "USD/kg",
    "polyester_pkr": "PKR/kg",
    "viscose_usd": "USD/kg",
    "viscose_pkr": "PKR/kg",
}


def usd_to_pkr(value, fx_rate):
    return value * fx_rate


def lb_to_maund(value):
    """Convert a per-lb price into a per-maund price."""
    return value * LB_PER_MAUND


def ton_to_kg(value):
    """Convert a per-ton price into a per-kg price."""
    return value / KG_PER_TON


def kg_to_ton(value):
    """Convert a per-kg price into a per-ton price, OR a kg quantity to ton quantity."""
    return value / KG_PER_TON


def kg_to_ton_price(value):
    """Convert a per-kg price into a per-ton price."""
    return value * KG_PER_TON


def ton_to_kg_qty(value):
    """Convert a ton quantity to kg quantity."""
    return value * KG_PER_TON


def rmb_to_usd(value, fx_rate):
    """Convert RMB/CNY to USD.

    fx_rate = USD/CNY rate (CNY per 1 USD)
    """
    return value / fx_rate


def get_unit(commodity_key):
    return COMMODITY_UNITS.get(commodity_key, "UNKNOWN")

