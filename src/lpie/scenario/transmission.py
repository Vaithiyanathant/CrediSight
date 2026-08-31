"""Macro -> feature -> hazard transmission.

`macro_scenarios.csv` conveniently supplies `default_rate_multiplier` and
friends. Applying those to a headline rate is a one-line answer, and it is weak:
it cannot produce segment-differentiated results, which is exactly what a stress
test is for.

So shocks transmit in two layers.

**Layer 1 — structural (macro -> features).** Rate shocks move the refinance
incentive, so they automatically hit high-rate loans hardest. HPI moves the
effective LTV, so they hit high-LTV and recent vintages hardest. Segment
differentiation comes out for free, from the mechanism rather than from a
hand-built segment table.

**Layer 2 — hazard multipliers in log-odds space.**

    logit(h_j') = logit(h_j) + log(multiplier_j)

Applying multipliers on the log-odds rather than multiplying probabilities is
not a stylistic choice: `2.8 x 0.45 = 1.26` is not a probability. Log-odds keeps
every result in [0,1] and preserves the transition simplex.

Layer 2 is then **re-anchored**: we solve for the multiplier scaling that makes
the portfolio-aggregate 12-month default rate match `base_rate x
default_rate_multiplier`, so the scenario respects the stated assumption while
still distributing the impact realistically across segments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from lpie.core.config import Settings, get_settings
from lpie.core.exceptions import ScenarioNotFoundError
from lpie.core.logging import get_logger

log = get_logger(__name__)

DELINQUENT_STATES = ("30DPD", "60DPD", "90DPD")
EPS = 1e-9


@dataclass
class MacroShock:
    scenario_name: str
    description: str = ""
    gdp_growth_pct: float = 0.0
    unemployment_rate_pct: float = 0.0
    hpi_change_pct: float = 0.0
    interest_rate_shock_bps: float = 0.0
    credit_spread_shock_bps: float = 0.0
    prepayment_cpr_assumption_pct: float = 0.0
    default_rate_multiplier: float = 1.0
    delinquency_rate_multiplier: float = 1.0
    prepayment_rate_multiplier: float = 1.0

    @classmethod
    def from_row(cls, row: pd.Series) -> MacroShock:
        def num(key: str, default: float = 0.0) -> float:
            value = row.get(key)
            try:
                v = float(value)
            except (TypeError, ValueError):
                return default
            return default if not np.isfinite(v) else v

        return cls(
            scenario_name=str(row.get("scenario_name", "custom")),
            description=str(row.get("description", "") or ""),
            gdp_growth_pct=num("gdp_growth_pct"),
            unemployment_rate_pct=num("unemployment_rate_pct"),
            hpi_change_pct=num("hpi_change_pct"),
            interest_rate_shock_bps=num("interest_rate_shock_bps"),
            credit_spread_shock_bps=num("credit_spread_shock_bps"),
            prepayment_cpr_assumption_pct=num("prepayment_cpr_assumption_pct"),
            default_rate_multiplier=num("default_rate_multiplier", 1.0),
            delinquency_rate_multiplier=num("delinquency_rate_multiplier", 1.0),
            prepayment_rate_multiplier=num("prepayment_rate_multiplier", 1.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "description": self.description,
            "gdp_growth_pct": self.gdp_growth_pct,
            "unemployment_rate_pct": self.unemployment_rate_pct,
            "hpi_change_pct": self.hpi_change_pct,
            "interest_rate_shock_bps": self.interest_rate_shock_bps,
            "credit_spread_shock_bps": self.credit_spread_shock_bps,
            "prepayment_cpr_assumption_pct": self.prepayment_cpr_assumption_pct,
            "default_rate_multiplier": self.default_rate_multiplier,
            "delinquency_rate_multiplier": self.delinquency_rate_multiplier,
            "prepayment_rate_multiplier": self.prepayment_rate_multiplier,
        }


def load_scenarios(macro: pd.DataFrame) -> dict[str, MacroShock]:
    return {str(row["scenario_name"]): MacroShock.from_row(row) for _, row in macro.iterrows()}


def get_scenario(macro: pd.DataFrame, name: str) -> MacroShock:
    scenarios = load_scenarios(macro)
    if name not in scenarios:
        raise ScenarioNotFoundError(
            f"Unknown scenario '{name}'",
            details={"requested": name, "available": sorted(scenarios)},
        )
    return scenarios[name]


def baseline_shock(name: str = "Base") -> MacroShock:
    return MacroShock(scenario_name=name, description="No shock applied")


# --------------------------------------------------------------------------- #
# Layer 1 — macro -> features
# --------------------------------------------------------------------------- #
def transmit_to_features(
    X: pd.DataFrame, shock: MacroShock, base: MacroShock | None = None,
    *, settings: Settings | None = None,
) -> pd.DataFrame:
    """Shift the shock-sensitive features. Returns a copy; X is never mutated."""
    s = settings or get_settings()
    cfg = s.section("scenario").get("transmission", {})
    base = base or baseline_shock()
    out = X.copy()

    d_rate_bps = shock.interest_rate_shock_bps - base.interest_rate_shock_bps
    d_hpi = (shock.hpi_change_pct - base.hpi_change_pct) / 100.0
    d_spread_bps = shock.credit_spread_shock_bps - base.credit_spread_shock_bps

    # Rates up => existing loans become *more* attractive to keep => prepayment
    # collapses. Automatically hits high-rate loans hardest.
    if "refi_incentive" in out.columns and d_rate_bps:
        delta = float(cfg.get("rate_shock_bps_to_refi_incentive", -0.01)) * d_rate_bps
        out["refi_incentive"] = pd.to_numeric(out["refi_incentive"], errors="coerce") + delta
        for interaction, partner in (
            ("refi_incentive_x_seasoning", "seasoning_frac"),
            ("refi_incentive_x_ltv", "ltv_band_ord"),
            ("burnout_x_incentive", "burnout"),
        ):
            if interaction in out.columns and partner in out.columns:
                out[interaction] = out["refi_incentive"] * pd.to_numeric(out[partner], errors="coerce")

    # Falling home prices raise effective LTV => higher severity and lower refi
    # ability. Hits high-LTV and recent vintages hardest.
    if d_hpi and "ltv_band_ord" in out.columns:
        ltv = pd.to_numeric(out["ltv_band_ord"], errors="coerce")
        # A 10% HPI fall moves a loan roughly one LTV band; bands are ~10pp wide.
        out["ltv_band_ord"] = (ltv - d_hpi * 10.0).clip(lower=0, upper=6)
        if "credit_x_ltv" in out.columns and "credit_score_band_ord" in out.columns:
            out["credit_x_ltv"] = out["ltv_band_ord"] * pd.to_numeric(
                out["credit_score_band_ord"], errors="coerce"
            )
        if "dpd_x_ltv" in out.columns and "dpd" in out.columns:
            out["dpd_x_ltv"] = out["ltv_band_ord"] * pd.to_numeric(out["dpd"], errors="coerce")

    # Spread widening closes the refinance window for weak credit first.
    if d_spread_bps and "refi_incentive" in out.columns and "credit_score_band_ord" in out.columns:
        credit = pd.to_numeric(out["credit_score_band_ord"], errors="coerce").fillna(3.0)
        weakness = (5.0 - credit).clip(lower=0.0) / 5.0
        out["refi_incentive"] = out["refi_incentive"] - (
            abs(float(cfg.get("credit_spread_bps_to_refi_logodds", -0.0022)))
            * d_spread_bps
            * weakness
        )
    return out


# --------------------------------------------------------------------------- #
# Layer 2 — hazard multipliers in log-odds space
# --------------------------------------------------------------------------- #
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def apply_hazard_multipliers(
    M: np.ndarray,
    states: list[str],
    shock: MacroShock,
    base: MacroShock | None = None,
    *,
    scaling: float = 1.0,
    settings: Settings | None = None,
) -> np.ndarray:
    """Shift destination hazards in log-odds space, then renormalise each row.

    Absorbing rows are left untouched: `Prepaid -> Prepaid = 1.0` is a logical
    certainty, and no macro scenario may erode it.
    """
    s = settings or get_settings()
    cfg = s.section("scenario").get("transmission", {})
    base = base or baseline_shock()
    index = {name: i for i, name in enumerate(states)}

    log_shift = np.zeros(len(states), dtype="float64")

    def add(state: str, value: float) -> None:
        if state in index:
            log_shift[index[state]] += value

    for state in DELINQUENT_STATES:
        add(state, scaling * np.log(max(shock.delinquency_rate_multiplier, EPS)))
    add("Default", scaling * np.log(max(shock.default_rate_multiplier, EPS)))
    add("Prepaid", scaling * np.log(max(shock.prepayment_rate_multiplier, EPS)))

    # Unemployment: the canonical macro -> credit channel, additive on log-odds.
    d_unemployment = shock.unemployment_rate_pct - base.unemployment_rate_pct
    if d_unemployment:
        delta = scaling * float(cfg.get("unemployment_delta_to_delinq_logodds", 0.28)) * d_unemployment
        for state in DELINQUENT_STATES:
            add(state, delta)
        add("Default", delta * 0.5)

    # Recoveries are pro-cyclical: GDP growth modulates the cure hazard.
    d_gdp = shock.gdp_growth_pct - base.gdp_growth_pct
    if d_gdp:
        add("Current", scaling * float(cfg.get("gdp_delta_to_cure_logodds", 0.09)) * d_gdp)

    d_hpi = shock.hpi_change_pct - base.hpi_change_pct
    if d_hpi:
        add("Default", scaling * float(cfg.get("hpi_change_to_default_logodds", -0.035)) * d_hpi)

    if not np.any(log_shift):
        return M

    shifted = _sigmoid(_logit(M) + log_shift[None, None, :])
    # Zero out destinations the base matrix ruled impossible: the legal mask is
    # domain law and a macro shock cannot make an illegal transition legal.
    shifted = np.where(M > 0.0, shifted, 0.0)
    totals = shifted.sum(axis=2, keepdims=True)
    shifted = np.divide(shifted, totals, out=np.zeros_like(shifted), where=totals > EPS)

    # Restore any row the shift degenerated, and every absorbing row exactly.
    degenerate = (totals.squeeze(-1) <= EPS)
    if degenerate.any():
        shifted[degenerate] = M[degenerate]
    for terminal in ("Prepaid", "Closed"):
        if terminal in index:
            i = index[terminal]
            shifted[:, i, :] = M[:, i, :]
    return shifted


def reanchor_scaling(
    predict_default_rate,
    shock: MacroShock,
    base_rate: float,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Solve for the multiplier scaling that reproduces the stated aggregate rate.

    `predict_default_rate(scaling) -> float` runs the simulation at a given
    scaling. Bisection on a monotone response: the scenario respects the
    organiser's stated assumption while the *distribution* of impact across
    segments stays driven by the structural transmission. Best of both.
    """
    s = settings or get_settings()
    cfg = s.section("scenario").get("reanchor", {})
    if not cfg.get("enabled", True):
        return {"enabled": False, "scaling": 1.0}

    target = base_rate * shock.default_rate_multiplier
    tolerance = float(cfg.get("tolerance", 0.005))
    max_iter = int(cfg.get("max_iterations", 25))
    tol_abs = tolerance * max(target, EPS)

    lo, hi = 0.0, 3.0
    achieved = predict_default_rate(1.0)
    if abs(achieved - target) <= tol_abs:
        return {"enabled": True, "scaling": 1.0, "target_rate": target,
                "achieved_rate": achieved, "iterations": 0, "converged": True,
                "bracketed": True}

    # Bracket the root before bisecting. `scaling` scales EVERY channel, so
    # scaling=0 reproduces (near enough) the unshocked default rate: for any
    # default_rate_multiplier < 1 the target sits below f(0) and no root exists
    # in [0, 3]. Bisecting anyway walked scaling to 0.0 and that scaling was
    # then applied, zeroing the prepayment and unemployment channels too — which
    # is how the High-Prepayment scenario came out byte-identical to Base.
    f_lo, f_hi = predict_default_rate(lo), predict_default_rate(hi)
    if not (min(f_lo, f_hi) - tol_abs <= target <= max(f_lo, f_hi) + tol_abs):
        return {
            "enabled": True,
            "scaling": 1.0,
            "target_rate": round(target, 8),
            "achieved_rate": round(achieved, 8),
            "iterations": 0,
            "converged": False,
            "bracketed": False,
            "search_bounds": {"lo": lo, "hi": hi,
                              "rate_at_lo": round(f_lo, 8), "rate_at_hi": round(f_hi, 8)},
            "note": (
                f"The stated default_rate_multiplier implies an aggregate default rate of "
                f"{target:.6f}, which is unreachable under this transmission map "
                f"(attainable range {min(f_lo, f_hi):.6f}-{max(f_lo, f_hi):.6f} over "
                f"scaling {lo}-{hi}). Falling back to scaling=1.0 and reporting the "
                "structural scenario as specified. The multipliers are applied in full; "
                "only the aggregate re-anchoring to the organiser's stated rate is skipped."
            ),
        }

    # f may be increasing or decreasing in `scaling` (prepayment competes with
    # default), so take the direction from the bracket rather than assuming it.
    increasing = f_hi >= f_lo
    scaling = 1.0
    for i in range(max_iter):
        scaling = 0.5 * (lo + hi)
        achieved = predict_default_rate(scaling)
        if abs(achieved - target) <= tol_abs:
            return {"enabled": True, "scaling": round(scaling, 6), "target_rate": round(target, 8),
                    "achieved_rate": round(achieved, 8), "iterations": i + 1, "converged": True,
                    "bracketed": True}
        if (achieved < target) == increasing:
            lo = scaling
        else:
            hi = scaling
    return {
        "enabled": True, "scaling": round(scaling, 6), "target_rate": round(target, 8),
        "achieved_rate": round(achieved, 8), "iterations": max_iter, "converged": False,
        "bracketed": True,
        "note": "Re-anchoring did not converge within the iteration budget; reporting the last scaling.",
    }
