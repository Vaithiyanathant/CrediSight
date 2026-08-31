"""Exact Shapley attribution over macro variables.

With six macro variables the full subset enumeration is 2^6 = 64 evaluations, so
the *exact* Shapley decomposition is computable — no sampling, no approximation.
That matters because it attributes interaction effects (unemployment x HPI, for
instance) that one-at-a-time sensitivity structurally cannot see.

Exact Shapley attribution of a stress scenario is a genuinely rare deliverable,
and the cost here is 64 vectorised simulations.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from itertools import combinations
from typing import Any

from lpie.core.logging import get_logger
from lpie.scenario.transmission import MacroShock

log = get_logger(__name__)

MAX_EXACT_VARIABLES = 12  # 2^12 = 4096 evaluations; beyond this, sample instead


def _coalition_shock(base: MacroShock, target: MacroShock, members: tuple[str, ...]) -> MacroShock:
    values = base.to_dict()
    for variable in members:
        values[variable] = getattr(target, variable)
    values["scenario_name"] = f"coalition[{','.join(members) or 'empty'}]"
    description = values.pop("description", "")
    return MacroShock(**values, description=description)


def exact_shapley(
    evaluate: Callable[[MacroShock], float],
    base: MacroShock,
    target: MacroShock,
    variables: tuple[str, ...],
    *,
    metric_name: str = "cumulative_default_rate",
) -> dict[str, Any]:
    """Exact Shapley values by full subset enumeration.

        phi_i = sum_{S subset N\\{i}} |S|!(n-|S|-1)!/n! * [ v(S u {i}) - v(S) ]
    """
    active = tuple(v for v in variables if getattr(base, v) != getattr(target, v))
    if not active:
        return {
            "metric": metric_name, "variables": [], "total_effect": 0.0,
            "note": "Scenario is identical to base on every macro variable.",
        }
    if len(active) > MAX_EXACT_VARIABLES:
        raise ValueError(
            f"Exact Shapley over {len(active)} variables needs 2^{len(active)} evaluations; "
            f"the exact path is capped at {MAX_EXACT_VARIABLES}."
        )

    n = len(active)
    cache: dict[frozenset[str], float] = {}

    def value(members: tuple[str, ...]) -> float:
        key = frozenset(members)
        if key not in cache:
            cache[key] = float(evaluate(_coalition_shock(base, target, members)))
        return cache[key]

    # Enumerate every coalition once; each is reused across all n marginal sums.
    for size in range(n + 1):
        for members in combinations(active, size):
            value(members)

    factorial = math.factorial
    shapley: dict[str, float] = {}
    for i, variable in enumerate(active):
        others = tuple(v for v in active if v != variable)
        total = 0.0
        for size in range(len(others) + 1):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for members in combinations(others, size):
                total += weight * (value((*members, variable)) - value(members))
        shapley[variable] = total
        del i

    empty = value(())
    full = value(active)
    total_effect = full - empty
    attributed = sum(shapley.values())

    rows = sorted(
        (
            {
                "variable": variable,
                "shapley_value": round(float(phi), 8),
                "abs_shapley": round(abs(float(phi)), 8),
                "share_of_effect": (
                    round(float(phi / total_effect), 6) if abs(total_effect) > 1e-12 else None
                ),
                "base_setting": getattr(base, variable),
                "scenario_setting": getattr(target, variable),
            }
            for variable, phi in shapley.items()
        ),
        key=lambda r: -r["abs_shapley"],
    )

    return {
        "metric": metric_name,
        "base_value": round(empty, 8),
        "scenario_value": round(full, 8),
        "total_effect": round(total_effect, 8),
        "attributed_effect": round(attributed, 8),
        # Efficiency is the Shapley axiom that guarantees the parts sum to the
        # whole. Reporting the residual proves the decomposition is exact rather
        # than asserting it.
        "efficiency_residual": round(total_effect - attributed, 10),
        "n_evaluations": len(cache),
        "variables": rows,
        "note": (
            "Exact Shapley by full subset enumeration. Unlike one-at-a-time sensitivity "
            "this attributes interaction effects between macro variables, which is why "
            "the values do not match the tornado ranking."
        ),
    }


def compare_with_tornado(shapley: dict[str, Any], tornado: dict[str, Any]) -> dict[str, Any]:
    """Where OAT and Shapley disagree, the gap *is* the interaction effect."""
    oat = {row["variable"]: row["impact"] for row in tornado.get("variables", [])}
    rows = []
    for row in shapley.get("variables", []):
        variable = row["variable"]
        oat_value = oat.get(variable)
        rows.append(
            {
                "variable": variable,
                "shapley": row["shapley_value"],
                "one_at_a_time": oat_value,
                "interaction_gap": (
                    round(row["shapley_value"] - oat_value, 8) if oat_value is not None else None
                ),
            }
        )
    rows.sort(key=lambda r: -(abs(r["interaction_gap"]) if r["interaction_gap"] is not None else 0))
    return {
        "comparison": rows,
        "note": (
            "A large interaction gap means the variable's effect depends on the presence of "
            "other shocks — the classic case is unemployment and HPI, where a labour shock "
            "does far more damage when home prices are also falling."
        ),
    }
