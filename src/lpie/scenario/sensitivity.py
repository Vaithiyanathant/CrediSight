"""One-at-a-time sensitivity — the tornado chart.

Shock each macro variable alone, hold the others at base, and rank the variables
by portfolio impact. OAT is the honest first cut and it is what most risk teams
actually read; its blind spot is interaction effects, which is why the exact
Shapley decomposition in `shapley_macro.py` is reported alongside it rather than
instead of it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from lpie.core.logging import get_logger
from lpie.scenario.transmission import MacroShock

log = get_logger(__name__)

MACRO_VARIABLES = (
    "gdp_growth_pct",
    "unemployment_rate_pct",
    "hpi_change_pct",
    "interest_rate_shock_bps",
    "credit_spread_shock_bps",
    "prepayment_cpr_assumption_pct",
)

MULTIPLIER_VARIABLES = (
    "default_rate_multiplier",
    "delinquency_rate_multiplier",
    "prepayment_rate_multiplier",
)


def _with_variable(base: MacroShock, target: MacroShock, variable: str) -> MacroShock:
    """Base shock with exactly one variable taken from the target scenario."""
    values = base.to_dict()
    values[variable] = getattr(target, variable)
    values["scenario_name"] = f"{base.scenario_name}+{variable}"
    return MacroShock(**{k: v for k, v in values.items() if k != "description"},
                      description=f"OAT: {variable} from {target.scenario_name}")


def tornado(
    evaluate: Callable[[MacroShock], float],
    base: MacroShock,
    target: MacroShock,
    *,
    variables: tuple[str, ...] = MACRO_VARIABLES + MULTIPLIER_VARIABLES,
    metric_name: str = "cumulative_default_rate",
) -> dict[str, Any]:
    """Rank macro variables by their individual portfolio impact."""
    base_value = evaluate(base)
    full_value = evaluate(target)

    rows: list[dict[str, Any]] = []
    for variable in variables:
        if getattr(base, variable) == getattr(target, variable):
            continue
        value = evaluate(_with_variable(base, target, variable))
        rows.append(
            {
                "variable": variable,
                "base_setting": getattr(base, variable),
                "scenario_setting": getattr(target, variable),
                "metric": round(float(value), 8),
                "impact": round(float(value - base_value), 8),
                "abs_impact": round(abs(float(value - base_value)), 8),
                "share_of_full_effect": (
                    round(float((value - base_value) / (full_value - base_value)), 6)
                    if abs(full_value - base_value) > 1e-12
                    else None
                ),
            }
        )
    rows.sort(key=lambda r: -r["abs_impact"])

    oat_total = sum(r["impact"] for r in rows)
    return {
        "metric": metric_name,
        "base_value": round(float(base_value), 8),
        "scenario_value": round(float(full_value), 8),
        "total_effect": round(float(full_value - base_value), 8),
        "variables": rows,
        "oat_sum": round(float(oat_total), 8),
        "interaction_residual": round(float((full_value - base_value) - oat_total), 8),
        "note": (
            "The residual is the part of the total effect that one-at-a-time analysis "
            "cannot attribute — it is carried by interactions between macro variables. "
            "The exact Shapley decomposition attributes it."
        ),
    }
