"""Demo script: End-to-end decision layer flow.

Demonstrates:
1. Load elasticity config
2. Attribute impacts from base ML forecast
3. Run scenarios
4. Generate buy/hold/delay signals
"""

import argparse
from pathlib import Path

import pandas as pd

from forecasting.decision import ElasticityModel, ScenarioSimulator, SignalGenerator
from forecasting.decision.config import ElasticityConfig, load_elasticity_config
from forecasting.decision.scenario import ScenarioInput


def demo_impact_attribution(elasticity_path: Path, ml_forecast: float, timestamp_str: str, variable_changes: dict):
    """Demo 1: Impact attribution."""
    print("\n" + "=" * 80)
    print("DEMO 1: Impact Attribution")
    print("=" * 80)

    if not elasticity_path.exists():
        print(f"Elasticity config not found at {elasticity_path}")
        return

    cfg: ElasticityConfig = load_elasticity_config(elasticity_path)
    print(f"\nLoaded elasticity config for {cfg.asset_id}")
    print(f"Variables: {[v.name for v in cfg.variables]}")

    model = ElasticityModel(cfg)

    timestamp = pd.Timestamp(timestamp_str)
    horizon = 1

    attribution = model.attribute_impacts(
        timestamp=timestamp,
        horizon=horizon,
        base_price_forecast=ml_forecast,
        variable_changes=variable_changes,
    )

    print(f"\nBase Forecast: {ml_forecast:.4f}")
    print(f"Total Elasticity-Based Impact: {attribution.total_impact_pct:.2f}%")
    print(f"Absolute Impact: {attribution.total_impact_abs:.4f}")
    
    implied_price = ml_forecast * (1 + attribution.total_impact_pct / 100.0)
    print(f"Implied Price (base + elasticity impact): {implied_price:.4f}")

    print("\nDetailed Impacts (top 10 by magnitude):")
    top_impacts = model.rank_impacts(attribution, top_n=10)
    print(top_impacts.to_string(index=False))

    return attribution


def demo_scenario_simulation(elasticity_path: Path, ml_forecast: float, asset: str):
    """Demo 2: Scenario simulation."""
    print("\n" + "=" * 80)
    print("DEMO 2: Scenario Simulation (What-If Analysis)")
    print("=" * 80)

    cfg: ElasticityConfig = load_elasticity_config(elasticity_path)
    simulator = ScenarioSimulator(cfg)

    # Asset-specific scenarios
    if asset == "eurusd":
        scenarios = [
            ScenarioInput(
                name="Baseline",
                description="No major macro moves",
                variable_changes_pct={
                    "usd_index": 0.0,
                    "eur_inflation": 0.0,
                    "us_inflation": 0.0,
                    "brent_crude": 0.0,
                    "us_10y_yield": 0.0,
                    "eur_stocks": 0.0,
                },
            ),
            ScenarioInput(
                name="Bull (EUR Strength)",
                description="EUR strengthens: lower yields, rising stocks, oil rally",
                variable_changes_pct={
                    "usd_index": -1.5,
                    "eur_inflation": -0.5,
                    "us_inflation": 0.0,
                    "brent_crude": 2.0,
                    "us_10y_yield": -0.50,
                    "eur_stocks": 2.0,
                },
            ),
            ScenarioInput(
                name="Bear (USD Strength)",
                description="USD rallies: strong economic data, higher yields",
                variable_changes_pct={
                    "usd_index": 2.0,
                    "eur_inflation": 0.5,
                    "us_inflation": 0.8,
                    "brent_crude": -3.0,
                    "us_10y_yield": 0.75,
                    "eur_stocks": -2.0,
                },
            ),
        ]
    else:  # cotton
        scenarios = [
            ScenarioInput(
                name="Baseline",
                description="Stable supply/demand",
                variable_changes_pct={
                    var: 0.0 for var in cfg.get_elasticity_dict().keys()
                },
            ),
            ScenarioInput(
                name="Bull (Tight Supply)",
                description="Production shortfalls, strong demand, expensive synthetics",
                variable_changes_pct={
                    "india_production": -3.0,  # supply shock
                    "global_supply_shock": -2.0,  # drought/pest
                    "china_demand": 2.0,  # demand surge
                    "apparel_demand": 1.5,
                    "synthetic_price": 2.0,  # synthetics rise
                    "india_monsoon": 80.0,  # dry monsoon
                    "oil_price": 3.0,
                    "freight_index": 1.0,
                    "usd_strength": -1.0,
                    "global_inflation": 0.0,
                    "china_production": 0.0,
                    "usa_production": 0.0,
                },
            ),
            ScenarioInput(
                name="Bear (Supply Glut)",
                description="Bumper crops, weak demand, cheap synthetics",
                variable_changes_pct={
                    "india_production": 3.0,  # record harvest
                    "usa_production": 2.0,
                    "china_production": 1.0,
                    "global_supply_shock": 0.0,
                    "china_demand": -2.0,  # weak demand
                    "apparel_demand": -2.5,  # recession fears
                    "synthetic_price": -4.0,  # synthetics cheap
                    "india_monsoon": 110.0,  # wet monsoon
                    "oil_price": -5.0,  # energy cheap
                    "freight_index": -2.0,
                    "usd_strength": 2.0,  # strong USD
                    "global_inflation": -0.5,
                },
            ),
        ]

    results_df = simulator.run_scenarios(ml_forecast, scenarios)
    print(f"\nBase Forecast: {ml_forecast:.4f}")
    print("\nScenario Results:")
    print(results_df.to_string(index=False))

    # Sensitivity: vary key variable based on asset
    sensitivity_var = "usd_index" if asset == "eurusd" else "india_production"
    print(f"\n\nOne-Way Sensitivity: {sensitivity_var}")
    print("-" * 60)
    sensitivity = simulator.sensitivity_analysis(
        base_price=ml_forecast,
        variable=sensitivity_var,
        pct_changes=[-2.0, -1.0, 0.0, 1.0, 2.0],
    )
    print(sensitivity.to_string(index=False))

    return results_df


def demo_signal_generation(buy_threshold: float = 2.0, sell_threshold: float = -1.5):
    """Demo 3: Signal generation."""
    print("\n" + "=" * 80)
    print("DEMO 3: Signal Generation")
    print("=" * 80)

    signal_gen = SignalGenerator(
        buy_threshold_pct=buy_threshold,
        sell_threshold_pct=sell_threshold,
        min_confidence=0.6,
    )

    ml_forecast = 1.1100

    # From bull scenario
    bull_scenario_price = 1.1230
    signal_bull = signal_gen.generate_signal(
        timestamp="2026-02-28",
        ml_forecast_price=ml_forecast,
        scenario_price=bull_scenario_price,
        scenario_name="Bull (EUR Strength)",
        top_drivers=["eur_stocks", "us_10y_yield"],
        elasticity_coverage=0.85,
    )
    print(f"\nBull Scenario Signal:")
    print(f"  Signal: {signal_bull.signal.value}")
    print(f"  Price Forecast: {signal_bull.price_forecast:.4f}")
    print(f"  Confidence: {signal_bull.confidence:.1%}")
    print(f"  Rationale: {signal_bull.rationale}")

    # From bear scenario
    bear_scenario_price = 1.0850
    signal_bear = signal_gen.generate_signal(
        timestamp="2026-02-28",
        ml_forecast_price=ml_forecast,
        scenario_price=bear_scenario_price,
        scenario_name="Bear (USD Strength)",
        top_drivers=["usd_index", "us_10y_yield"],
        elasticity_coverage=0.82,
    )
    print(f"\nBear Scenario Signal:")
    print(f"  Signal: {signal_bear.signal.value}")
    print(f"  Price Forecast: {signal_bear.price_forecast:.4f}")
    print(f"  Confidence: {signal_bear.confidence:.1%}")
    print(f"  Rationale: {signal_bear.rationale}")

    # From baseline
    baseline_scenario_price = ml_forecast
    signal_base = signal_gen.generate_signal(
        timestamp="2026-02-28",
        ml_forecast_price=ml_forecast,
        scenario_price=baseline_scenario_price,
        scenario_name="Baseline",
        top_drivers=["usd_index"],
        elasticity_coverage=0.70,
    )
    print(f"\nBaseline Scenario Signal:")
    print(f"  Signal: {signal_base.signal.value}")
    print(f"  Price Forecast: {signal_base.price_forecast:.4f}")
    print(f"  Confidence: {signal_base.confidence:.1%}")
    print(f"  Rationale: {signal_base.rationale}")


def main():
    ap = argparse.ArgumentParser(description="Decision layer demo.")
    ap.add_argument("--demo", choices=["all", "impact", "scenario", "signal"], default="all")
    ap.add_argument(
        "--asset",
        choices=["eurusd", "cotton"],
        default="eurusd",
        help="Asset to demonstrate (eurusd, cotton)",
    )
    args = ap.parse_args()
    
    # Determine elasticity config and scenario/signal parameters based on asset
    if args.asset == "eurusd":
        elasticity_path = Path("data/decision/eurusd_elasticity.yml")
        base_forecast = 1.1100
        demo_timestamp = "2026-02-28"
        demo_variable_changes = {
            "usd_index": 1.0,
            "eur_inflation": -0.5,
            "us_inflation": 0.3,
            "brent_crude": -2.0,
            "us_10y_yield": 0.30,
            "eur_stocks": -1.0,
        }
        buy_threshold = 2.0
        sell_threshold = -1.5
    elif args.asset == "cotton":
        elasticity_path = Path("data/decision/cotton_elasticity.yml")
        base_forecast = 75.0  # USD/cwt (typical cotton price)
        demo_timestamp = "2026-02-28"
        demo_variable_changes = {
            "india_production": 2.0,      # +2% increase
            "china_demand": 1.5,          # +1.5% demand growth
            "apparel_demand": -1.0,       # -1% demand contraction
            "synthetic_price": -3.0,      # -3% (polyester cheaper)
            "oil_price": 5.0,             # +5% higher energy costs
            "freight_index": 2.0,         # +2% shipping
            "india_monsoon": 90.0,        # 90% of normal (dry)
            "global_inflation": 0.5,      # +0.5% inflation
            "usd_strength": 1.0,          # +1% USD strength
            "china_production": 0.0,
            "usa_production": 0.0,
            "global_supply_shock": 0.0,
        }
        buy_threshold = 3.0
        sell_threshold = -2.0
    else:
        raise ValueError(f"Unknown asset: {args.asset}")
    
    # Pass these to demo functions
    if args.demo in ["all", "impact"]:
        demo_impact_attribution(elasticity_path, base_forecast, demo_timestamp, demo_variable_changes)
    
    if args.demo in ["all", "scenario"]:
        demo_scenario_simulation(elasticity_path, base_forecast, args.asset)
    
    if args.demo in ["all", "signal"]:
        demo_signal_generation(buy_threshold, sell_threshold)

    print("\n" + "=" * 80)
    print("Demo Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
