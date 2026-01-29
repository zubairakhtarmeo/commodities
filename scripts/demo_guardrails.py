"""Guardrails demo: showing how economic constraints protect business-safe outputs."""

import pandas as pd

from forecasting.decision import GuardrailConfig, GuardrailEngine


def demo_guardrails():
    """Demonstrate guardrails in action."""
    print("\n" + "=" * 90)
    print("GUARDRAILS DEMO: Economic Protection for Decision Layer")
    print("=" * 90)

    # Cotton guardrails
    print("\n1. COTTON COMMODITY - Guardrail Examples")
    print("-" * 90)

    cfg_cotton = GuardrailConfig.for_cotton()
    engine_cotton = GuardrailEngine(cfg_cotton)

    base_price = 75.0  # USD/cwt

    test_scenarios = [
        ("Normal Supply Tightening", 70.0),  # -6.7% (within normal)
        ("Significant Supply Shock", 52.5),  # -30% (EXTREME, gets dampened)
        ("Catastrophic Supply Failure", 20.0),  # -73% (IMPLAUSIBLE → floor applied)
        ("Demand Boom", 82.5),  # +10% (normal upside)
        ("Extreme Demand Shock", 120.0),  # +60% (IMPLAUSIBLE)
    ]

    print(f"\nBase Price: ${base_price:.2f}/cwt\n")
    print(
        "Scenario                           | Original | Original % | Guardrailed | GRD % | Flag       | Adjustments"
    )
    print("-" * 120)

    for scenario_name, scenario_price in test_scenarios:
        result = engine_cotton.apply_guardrails(base_price, scenario_price, scenario_name)
        pct_str = f"{result.price_change_pct:+.1f}%"
        grd_pct = f"{result.guardrailed_pct:+.1f}%"
        adjustments_str = " | ".join(result.adjustments) if result.adjustments else "None"
        print(
            f"{scenario_name:32} | ${scenario_price:7.2f} | {pct_str:>9} | ${result.guardrailed_price:7.2f}    | {grd_pct:>5} | {result.extreme_flag.value:10} | {adjustments_str[:50]}"
        )

    # FX guardrails
    print("\n\n2. FX (EURUSD) - Guardrail Examples")
    print("-" * 90)

    cfg_eurusd = GuardrailConfig.for_eurusd()
    engine_eurusd = GuardrailEngine(cfg_eurusd)

    base_price_fx = 1.1100

    test_scenarios_fx = [
        ("Normal EUR Weakness", 1.0800),  # -2.7%
        ("Major USD Strength", 1.0350),  # -6.8% (EXTREME)
        ("Extreme Financial Crisis", 0.8000),  # -28% (IMPLAUSIBLE)
        ("EUR Strength", 1.1500),  # +3.6% (EXTREME but realistic)
    ]

    print(f"\nBase Price: ${base_price_fx:.4f}\n")
    print(
        "Scenario                           | Original | Original % | Guardrailed | GRD %  | Flag       | Adjustments"
    )
    print("-" * 120)

    for scenario_name, scenario_price in test_scenarios_fx:
        result = engine_eurusd.apply_guardrails(base_price_fx, scenario_price, scenario_name)
        pct_str = f"{result.price_change_pct:+.1f}%"
        grd_pct = f"{result.guardrailed_pct:+.1f}%"
        adjustments_str = " | ".join(result.adjustments) if result.adjustments else "None"
        print(
            f"{scenario_name:32} | ${scenario_price:7.4f} | {pct_str:>9} | ${result.guardrailed_price:7.4f}   | {grd_pct:>6} | {result.extreme_flag.value:10} | {adjustments_str[:50]}"
        )

    # Nonlinear dampening illustration
    print("\n\n3. NONLINEAR DAMPENING - How Tail Shocks Are Moderated")
    print("-" * 90)

    print(
        f"\nCotton Config: dampening_threshold={cfg_cotton.dampening_threshold_pct}%, factor={cfg_cotton.dampening_factor}"
    )
    print("\nMove Size   | Undampened | Dampened | Reduction | Flag")
    print("-" * 55)

    move_sizes = [-50, -30, -20, -15, 0, 15, 20, 30, 50]
    for move in move_sizes:
        orig_price = base_price * (1 + move / 100.0)
        result = engine_cotton.apply_guardrails(base_price, orig_price, "test")
        reduction = move - result.guardrailed_pct
        flag = result.extreme_flag.value
        print(
            f"{move:+6.0f}%    | {move:+8.1f}%  | {result.guardrailed_pct:+8.1f}% | {reduction:+8.1f}% | {flag:10}"
        )

    # Summary
    print("\n\n4. GUARDRAIL DESIGN PRINCIPLES")
    print("-" * 90)
    print("""
    ✓ PROTECTION WITHOUT BIAS:
      - Guardrails are symmetric (upside/downside equally constrained)
      - Nonlinear dampening preserves signal, just moderates extremes
      - Price floor prevents impossible outcomes (e.g., negative prices)

    ✓ TRANSPARENCY:
      - Every adjustment is logged and visible
      - Extreme flag warns management for review
      - Original vs. guardrailed prices always reported

    ✓ CONFIGURABILITY:
      - Per-asset guardrail profiles (e.g., cotton vs. FX)
      - Easy to adjust thresholds as market regimes change
      - Can be disabled entirely for analysis-only runs

    ✓ BUSINESS ALIGNMENT:
      - Prevents unrealistic P&L scenarios
      - Maintains analytical integrity (only dampens, never removes signals)
      - Management confidence in system outputs
    """)

    print("\n" + "=" * 90)
    print("Guardrails ensure decision layer outputs are management-safe without weakening logic.")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    demo_guardrails()
