"""Monte-Carlo portfolio simulation.

A point estimate says "default rises to 4.2%". A risk system says
"4.2% [90% CI: 3.6% - 4.9%]". The difference is whether a reviewer can size the
uncertainty, and it is the difference between a forecast and a stress test.

Vectorised over the whole book: `n_paths x horizon` steps on a
(n_loans x K) state array, sampled with the inverse-CDF trick. At 10,000 loans,
1,000 paths and 24 months this is seconds, not minutes.

Every run is validated against hard invariants before any number is returned —
probabilities in [0,1], rows summing to 1, absorbing states never exiting. A
violation fails loudly rather than producing a pretty, wrong chart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from lpie.core.config import Settings, get_settings
from lpie.core.exceptions import ScenarioInvariantError
from lpie.core.logging import get_logger
from lpie.core.timing import Timer, utcnow_iso

log = get_logger(__name__)

TOLERANCE = 1e-6


@dataclass
class SimulationResult:
    scenario: str
    n_paths: int
    horizon: int
    states: list[str]
    # (n_paths, horizon+1, K) portfolio state counts
    state_counts: np.ndarray
    balances: np.ndarray                  # (n_paths, horizon+1) outstanding balance
    cumulative_defaults: np.ndarray       # (n_paths, horizon+1) count
    cumulative_prepayments: np.ndarray
    losses: np.ndarray                    # (n_paths, horizon+1) cumulative loss USD
    n_loans: int = 0
    seed: int = 0
    elapsed_ms: float = 0.0
    invariants: dict[str, Any] = field(default_factory=dict)


def _validate_matrix(M: np.ndarray, states: list[str], terminal: tuple[str, ...]) -> dict[str, Any]:
    if M.size == 0:
        return {"n_matrices": 0}
    row_sums = M.sum(axis=2)
    checks = {
        "min_probability": float(M.min()),
        "max_probability": float(M.max()),
        "max_row_sum_error": float(np.abs(row_sums - 1.0).max()),
    }
    index = {s: i for i, s in enumerate(states)}
    for state in terminal:
        if state in index:
            i = index[state]
            checks[f"absorbing_{state}"] = float(M[:, i, i].min())

    problems = []
    if checks["min_probability"] < -TOLERANCE:
        problems.append("negative transition probability")
    if checks["max_probability"] > 1.0 + TOLERANCE:
        problems.append("transition probability above 1")
    if checks["max_row_sum_error"] > 1e-4:
        problems.append(f"transition rows do not sum to 1 (max error {checks['max_row_sum_error']:.2e})")
    for state in terminal:
        key = f"absorbing_{state}"
        if key in checks and checks[key] < 1.0 - 1e-4:
            problems.append(f"{state} is not absorbing (min self-transition {checks[key]:.6f})")

    if problems:
        raise ScenarioInvariantError(
            f"{len(problems)} transition-matrix invariant violation(s)",
            details={"problems": problems, "checks": checks},
        )
    return checks


def simulate(
    M: np.ndarray,
    start_state: np.ndarray,
    balances: np.ndarray,
    loss_severity: np.ndarray,
    *,
    states: list[str],
    scenario: str = "Base",
    n_paths: int = 1000,
    horizon: int = 24,
    seed: int = 0,
    terminal_states: tuple[str, ...] = ("Prepaid", "Closed"),
    settings: Settings | None = None,
) -> SimulationResult:
    """Sample `n_paths` portfolio trajectories through the shocked state machine."""
    settings or get_settings()
    timer = Timer()

    n_loans, K, _ = M.shape
    invariants = _validate_matrix(M, states, terminal_states)

    index = {name: i for i, name in enumerate(states)}
    default_idx = index.get("Default", -1)
    prepaid_idx = index.get("Prepaid", -1)

    # Inverse-CDF sampling: one uniform draw per (path, loan, month), compared
    # against the cumulative row of that loan's current state.
    cdf = np.cumsum(M, axis=2)
    cdf[:, :, -1] = 1.0

    rng = np.random.default_rng(seed)
    state_counts = np.zeros((n_paths, horizon + 1, K), dtype="int32")
    balance_paths = np.zeros((n_paths, horizon + 1), dtype="float64")
    cum_defaults = np.zeros((n_paths, horizon + 1), dtype="int32")
    cum_prepayments = np.zeros((n_paths, horizon + 1), dtype="int32")
    loss_paths = np.zeros((n_paths, horizon + 1), dtype="float64")

    bal = np.asarray(balances, dtype="float64")
    sev = np.asarray(loss_severity, dtype="float64")
    loan_rows = np.arange(n_loans)

    for path in range(n_paths):
        state = np.asarray(start_state, dtype="int64").copy()
        ever_default = state == default_idx
        ever_prepaid = state == prepaid_idx
        outstanding = bal.copy()
        cumulative_loss = float(np.sum(bal[ever_default] * sev[ever_default])) if default_idx >= 0 else 0.0

        state_counts[path, 0] = np.bincount(state, minlength=K)
        balance_paths[path, 0] = float(outstanding.sum())
        cum_defaults[path, 0] = int(ever_default.sum())
        cum_prepayments[path, 0] = int(ever_prepaid.sum())
        loss_paths[path, 0] = cumulative_loss

        for month in range(1, horizon + 1):
            draws = rng.random(n_loans)
            state = (draws[:, None] > cdf[loan_rows, state]).sum(axis=1)
            state = np.clip(state, 0, K - 1)

            if default_idx >= 0:
                newly_defaulted = (state == default_idx) & ~ever_default
                if newly_defaulted.any():
                    cumulative_loss += float(np.sum(outstanding[newly_defaulted] * sev[newly_defaulted]))
                ever_default |= state == default_idx
            if prepaid_idx >= 0:
                ever_prepaid |= state == prepaid_idx

            # Terminal loans carry no outstanding balance.
            gone = np.zeros(n_loans, dtype=bool)
            for terminal in terminal_states:
                if terminal in index:
                    gone |= state == index[terminal]
            outstanding = np.where(gone, 0.0, outstanding)

            state_counts[path, month] = np.bincount(state, minlength=K)
            balance_paths[path, month] = float(outstanding.sum())
            cum_defaults[path, month] = int(ever_default.sum())
            cum_prepayments[path, month] = int(ever_prepaid.sum())
            loss_paths[path, month] = cumulative_loss

    result = SimulationResult(
        scenario=scenario, n_paths=n_paths, horizon=horizon, states=list(states),
        state_counts=state_counts, balances=balance_paths,
        cumulative_defaults=cum_defaults, cumulative_prepayments=cum_prepayments,
        losses=loss_paths, n_loans=n_loans, seed=seed,
        elapsed_ms=round(timer.stop(), 2), invariants=invariants,
    )
    _validate_result(result)
    log.info("scenario.simulated", scenario=scenario, paths=n_paths,
             horizon=horizon, elapsed_ms=result.elapsed_ms)
    return result


def _validate_result(result: SimulationResult) -> None:
    totals = result.state_counts.sum(axis=2)
    if not np.all(totals == result.n_loans):
        raise ScenarioInvariantError(
            "Simulated state counts do not sum to the portfolio size at every step",
            details={"expected": result.n_loans, "observed_min": int(totals.min()),
                     "observed_max": int(totals.max())},
        )
    if np.any(np.diff(result.cumulative_defaults, axis=1) < 0):
        raise ScenarioInvariantError("Cumulative defaults decreased — an absorbing state was exited")
    if np.any(np.diff(result.cumulative_prepayments, axis=1) < 0):
        raise ScenarioInvariantError("Cumulative prepayments decreased — an absorbing state was exited")
    if np.any(result.balances < -TOLERANCE):
        raise ScenarioInvariantError("Negative outstanding balance in a simulated path")


# --------------------------------------------------------------------------- #
# summarisation
# --------------------------------------------------------------------------- #
def fan_chart(paths: np.ndarray, levels: tuple[int, ...] = (5, 50, 95)) -> dict[str, Any]:
    """Mean plus percentile bands over the path dimension."""
    return {
        "mean": [round(float(v), 6) for v in paths.mean(axis=0)],
        **{
            f"p{level}": [round(float(v), 6) for v in np.percentile(paths, level, axis=0)]
            for level in levels
        },
        "months": list(range(paths.shape[1])),
    }


def summarise(
    result: SimulationResult,
    *,
    levels: tuple[int, ...] = (5, 50, 95),
    var_level: int = 95,
) -> dict[str, Any]:
    n = max(result.n_loans, 1)
    delinquent = [
        i for i, name in enumerate(result.states) if name in ("30DPD", "60DPD", "90DPD")
    ]
    delinquency_rate = result.state_counts[:, :, delinquent].sum(axis=2) / n
    default_rate = result.cumulative_defaults / n
    prepay_rate = result.cumulative_prepayments / n

    terminal_loss = result.losses[:, -1]
    var = float(np.percentile(terminal_loss, var_level))
    tail = terminal_loss[terminal_loss >= var]
    expected_shortfall = float(tail.mean()) if tail.size else var

    horizon = result.horizon
    initial_balance = float(result.balances[:, 0].mean())
    final_balance = float(result.balances[:, -1].mean())

    return {
        "scenario": result.scenario,
        "n_paths": result.n_paths,
        "horizon_months": horizon,
        "n_loans": result.n_loans,
        "seed": result.seed,
        "elapsed_ms": result.elapsed_ms,
        "computed_at": utcnow_iso(),
        "fan_charts": {
            "delinquency_rate": fan_chart(delinquency_rate, levels),
            "cumulative_default_rate": fan_chart(default_rate, levels),
            "cumulative_prepayment_rate": fan_chart(prepay_rate, levels),
            "outstanding_balance": fan_chart(result.balances, levels),
            "cumulative_loss": fan_chart(result.losses, levels),
        },
        "state_occupancy": {
            "states": result.states,
            "mean_share": [
                [round(float(v), 6) for v in row]
                for row in (result.state_counts.mean(axis=0) / n)
            ],
            "months": list(range(horizon + 1)),
        },
        "terminal": {
            "default_rate": _dist(default_rate[:, -1], levels),
            "prepayment_rate": _dist(prepay_rate[:, -1], levels),
            "delinquency_rate": _dist(delinquency_rate[:, -1], levels),
            "expected_loss": _dist(terminal_loss, levels),
            "outstanding_balance": _dist(result.balances[:, -1], levels),
        },
        "risk_measures": {
            f"var_{var_level}": round(var, 2),
            f"expected_shortfall_{var_level}": round(expected_shortfall, 2),
            "mean_expected_loss": round(float(terminal_loss.mean()), 2),
            "loss_as_pct_of_initial_balance": (
                round(float(terminal_loss.mean()) / initial_balance * 100.0, 6)
                if initial_balance > 0 else None
            ),
        },
        "portfolio_metrics": {
            "weighted_average_life_months": _weighted_average_life(result.balances),
            "cpr_annualised_pct": _annualised_cpr(prepay_rate[:, -1].mean(), horizon),
            "cdr_annualised_pct": _annualised_cpr(default_rate[:, -1].mean(), horizon),
            "initial_balance": round(initial_balance, 2),
            "final_balance": round(final_balance, 2),
            "balance_runoff_pct": (
                round((1.0 - final_balance / initial_balance) * 100.0, 4)
                if initial_balance > 0 else None
            ),
        },
        "path_dependent": {
            "p_default_rate_above_5pct": round(float((default_rate[:, -1] > 0.05).mean()), 6),
            "p_prepay_20pct_within_12m": (
                round(float((prepay_rate[:, min(12, horizon)] > 0.20).mean()), 6)
            ),
            "p_loss_above_mean": round(float((terminal_loss > terminal_loss.mean()).mean()), 6),
        },
        "invariants": result.invariants,
    }


def _dist(values: np.ndarray, levels: tuple[int, ...]) -> dict[str, float]:
    return {
        "mean": round(float(values.mean()), 6),
        "std": round(float(values.std(ddof=1)) if values.size > 1 else 0.0, 6),
        **{f"p{level}": round(float(np.percentile(values, level)), 6) for level in levels},
    }


def _weighted_average_life(balances: np.ndarray) -> float | None:
    """WAL = sum(t * principal repaid at t) / total principal repaid."""
    mean_balance = balances.mean(axis=0)
    repaid = -np.diff(mean_balance)
    total = repaid.sum()
    if total <= 0:
        return None
    months = np.arange(1, len(repaid) + 1)
    return round(float((months * repaid).sum() / total), 4)


def _annualised_cpr(cumulative_rate: float, horizon: int) -> float | None:
    """Convert a cumulative rate over `horizon` months to an annualised speed."""
    if horizon <= 0 or cumulative_rate <= 0:
        return 0.0
    survivor = max(1.0 - cumulative_rate, 1e-9)
    monthly = 1.0 - survivor ** (1.0 / horizon)
    return round(float((1.0 - (1.0 - monthly) ** 12) * 100.0), 6)


def segment_impact(
    scenario_cif: np.ndarray,
    base_cif: np.ndarray,
    segments: pd.Series,
    *,
    balances: np.ndarray | None = None,
    loss_severity: np.ndarray | None = None,
    min_segment_size: int = 25,
) -> list[dict[str, Any]]:
    """Per-segment delta in expected default incidence vs Base.

    Computed from the per-loan cumulative incidence functions rather than by
    re-running Monte Carlo per segment: the CIF *is* the expectation of the
    default indicator, so the segment means are exact and cost nothing, while
    Monte Carlo remains what produces the portfolio-level distribution.

    Portfolio-level rate multipliers cannot produce this cut at all; only the
    structural transmission of Layer 1 can, which is the whole argument for it.
    """
    seg = pd.Series(segments).astype(str).to_numpy()
    scenario_cif = np.asarray(scenario_cif, dtype="float64")
    base_cif = np.asarray(base_cif, dtype="float64")

    rows: list[dict[str, Any]] = []
    for value in pd.unique(seg):
        mask = seg == value
        n = int(mask.sum())
        if n < min_segment_size:
            continue
        scenario_rate = float(scenario_cif[mask].mean())
        base_rate = float(base_cif[mask].mean())
        entry: dict[str, Any] = {
            "segment": str(value),
            "n_loans": n,
            "share_of_portfolio": round(n / max(len(seg), 1), 6),
            "base_default_rate": round(base_rate, 6),
            "scenario_default_rate": round(scenario_rate, 6),
            "delta": round(scenario_rate - base_rate, 6),
            "relative_delta": (
                round((scenario_rate - base_rate) / base_rate, 6) if base_rate > 1e-9 else None
            ),
        }
        if balances is not None:
            bal = np.asarray(balances, dtype="float64")[mask]
            sev = (
                np.asarray(loss_severity, dtype="float64")[mask]
                if loss_severity is not None
                else np.full(n, 0.35)
            )
            entry["base_expected_loss"] = round(float((base_cif[mask] * bal * sev).sum()), 2)
            entry["scenario_expected_loss"] = round(float((scenario_cif[mask] * bal * sev).sum()), 2)
            entry["expected_loss_delta"] = round(
                entry["scenario_expected_loss"] - entry["base_expected_loss"], 2
            )
            entry["outstanding_balance"] = round(float(bal.sum()), 2)
        rows.append(entry)

    rows.sort(key=lambda r: -(r["delta"] or 0.0))
    return rows
