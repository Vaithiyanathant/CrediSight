"""Survival baselines: Kaplan-Meier, Cox PH, and a first-order Markov chain.

The rubric asks for one baseline; three are more informative because they fail
in different ways:

* **Kaplan-Meier** is the non-parametric truth with no covariates — the honest
  floor, plus log-rank tests for whether segments genuinely separate.
* **Cox PH** gives interpretable hazard ratios (`exp(beta)` per feature, directly
  quotable in a reviewer note) and a Schoenfeld test of the proportional-hazards
  assumption. We *expect* PH to fail for `loan_age`, and that failure is precisely
  the argument for the discrete-time model.
* **First-order Markov** on the measured transition matrix, no covariates at all,
  shows how much the covariates actually add over "the portfolio's average
  transition behaviour".

Censoring is handled explicitly: loans observed to the panel edge without an
event are censored there. The discrete-time likelihood contributes only the
months actually observed, so censored loans contribute correctly rather than
being dropped (which biases hazards downward) or treated as negatives (which
biases them downward harder).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from lpie.core.config import Settings, get_settings
from lpie.core.logging import get_logger

log = get_logger(__name__)

EVENT_STATES = ("Default", "Prepaid", "Closed")
ACTIVE_STATES = ("Current", "30DPD", "60DPD", "90DPD")


@dataclass
class SurvivalDataset:
    """One row per loan: duration to first event (or censoring), plus the cause."""

    frame: pd.DataFrame
    origin: str = "loan_age_months"
    n_censored: int = 0
    n_events: dict[str, int] = field(default_factory=dict)


def build_survival_dataset(
    panel: pd.DataFrame, *, origin: str = "loan_age_months"
) -> SurvivalDataset:
    work = panel.sort_values(["loan_id", "month_index"], kind="mergesort")
    rows: list[dict[str, Any]] = []

    for loan_id, group in work.groupby("loan_id", sort=False):
        statuses = group["current_status"].astype("object").to_numpy()
        ages = pd.to_numeric(group[origin], errors="coerce").to_numpy(dtype="float64")
        entry_age = float(np.nanmin(ages)) if np.isfinite(ages).any() else 0.0

        event_position = None
        event_cause = None
        for i, status in enumerate(statuses):
            if status in EVENT_STATES:
                event_position = i
                event_cause = status
                break

        if event_position is None:
            duration = float(np.nanmax(ages)) - entry_age
            rows.append(
                {"loan_id": loan_id, "entry_age": entry_age, "duration": max(duration, 0.0),
                 "event": 0, "cause": "censored"}
            )
        else:
            duration = float(ages[event_position]) - entry_age
            rows.append(
                {"loan_id": loan_id, "entry_age": entry_age, "duration": max(duration, 0.0),
                 "event": 1, "cause": event_cause}
            )

    frame = pd.DataFrame(rows)
    return SurvivalDataset(
        frame=frame,
        origin=origin,
        n_censored=int((frame["event"] == 0).sum()),
        n_events=frame.loc[frame["event"] == 1, "cause"].value_counts().to_dict(),
    )


def kaplan_meier(
    dataset: SurvivalDataset,
    covariates: pd.DataFrame | None = None,
    *,
    segment: str | None = None,
    cause: str | None = None,
    max_time: int = 24,
) -> dict[str, Any]:
    """KM survival curves, optionally by segment, with a log-rank test."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    frame = dataset.frame.copy()
    if cause is not None:
        # Cause-specific: other causes are treated as censoring at their time.
        frame["event"] = ((frame["event"] == 1) & (frame["cause"] == cause)).astype(int)

    if segment and covariates is not None and segment in covariates.columns:
        merged = frame.merge(
            covariates[["loan_id", segment]].drop_duplicates("loan_id"), on="loan_id", how="left"
        )
        merged[segment] = merged[segment].astype(str).fillna("Unknown")
    else:
        merged = frame.assign(_all="portfolio")
        segment = "_all"

    curves = []
    for value, group in merged.groupby(segment, sort=True):
        if len(group) < 20:
            continue
        fitter = KaplanMeierFitter()
        fitter.fit(group["duration"], group["event"], label=str(value))
        timeline = np.arange(0, max_time + 1)
        survival = fitter.survival_function_at_times(timeline).to_numpy()
        curves.append(
            {
                "segment": str(value),
                "n": int(len(group)),
                "n_events": int(group["event"].sum()),
                "timeline": [int(t) for t in timeline],
                "survival": [round(float(v), 6) for v in survival],
                "median_survival": (
                    None if not np.isfinite(fitter.median_survival_time_)
                    else round(float(fitter.median_survival_time_), 4)
                ),
            }
        )

    logrank: dict[str, Any] = {}
    if len(curves) > 1:
        try:
            test = multivariate_logrank_test(
                merged["duration"], merged[segment], merged["event"]
            )
            logrank = {
                "test_statistic": round(float(test.test_statistic), 6),
                "p_value": round(float(test.p_value), 8),
                "significant_at_0_05": bool(test.p_value < 0.05),
                "interpretation": (
                    "Segment curves separate significantly."
                    if test.p_value < 0.05
                    else "No significant separation between segment curves."
                ),
            }
        except Exception as exc:  # pragma: no cover - lifelines edge case
            logrank = {"error": str(exc)}

    return {
        "method": "kaplan_meier",
        "cause": cause or "any_event",
        "segment_by": None if segment == "_all" else segment,
        "curves": curves,
        "log_rank": logrank,
        "n_censored": dataset.n_censored,
        "censoring_note": (
            "Loans still active at the panel edge are censored there, not treated as "
            "non-events. Treating them as negatives would bias every hazard downward."
        ),
    }


def cox_proportional_hazards(
    dataset: SurvivalDataset,
    covariates: pd.DataFrame,
    *,
    features: list[str] | None = None,
    cause: str = "Default",
    max_rows: int = 20_000,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Cause-specific Cox PH with hazard ratios and a Schoenfeld PH test."""
    from lifelines import CoxPHFitter

    s = settings or get_settings()
    frame = dataset.frame.copy()
    frame["event"] = ((frame["event"] == 1) & (frame["cause"] == cause)).astype(int)

    default_features = [
        "credit_score_band_ord", "ltv_band_ord", "dti_band_ord", "interest_rate",
        "log_original_balance", "loan_age_months", "dpd_max_12m", "cure_count",
        "bal_ratio", "refi_incentive",
    ]
    use = [f for f in (features or default_features) if f in covariates.columns]
    if not use:
        return {"method": "cox_ph", "available": False, "reason": "no usable covariates"}

    # One row per loan, taken at first observation, so the covariates are
    # baseline values rather than time-varying ones the model cannot consume.
    baseline = (
        covariates.sort_values("month_index", kind="mergesort")
        .drop_duplicates("loan_id", keep="first")[["loan_id", *use]]
    )
    merged = frame.merge(baseline, on="loan_id", how="inner").dropna(subset=use)
    merged = merged[merged["duration"] > 0]
    if len(merged) < 200 or merged["event"].sum() < 20:
        return {"method": "cox_ph", "available": False,
                "reason": f"insufficient events for cause {cause}"}
    if len(merged) > max_rows:
        merged = merged.sample(max_rows, random_state=s.seed)

    fitter = CoxPHFitter(penalizer=0.1)
    try:
        fitter.fit(merged[["duration", "event", *use]], duration_col="duration", event_col="event")
    except Exception as exc:
        return {"method": "cox_ph", "available": False, "reason": str(exc)}

    summary = fitter.summary
    ratios = [
        {
            "feature": name,
            "coefficient": round(float(row["coef"]), 6),
            "hazard_ratio": round(float(row["exp(coef)"]), 6),
            "ci_low": round(float(row["exp(coef) lower 95%"]), 6),
            "ci_high": round(float(row["exp(coef) upper 95%"]), 6),
            "p_value": round(float(row["p"]), 8),
            "significant": bool(row["p"] < 0.05),
            "interpretation": (
                f"A one-unit increase multiplies the {cause.lower()} hazard by "
                f"{float(row['exp(coef)']):.3f}."
            ),
        }
        for name, row in summary.iterrows()
    ]
    ratios.sort(key=lambda r: -abs(r["coefficient"]))

    ph_test: dict[str, Any] = {}
    try:
        results = fitter.check_assumptions(
            merged[["duration", "event", *use]], p_value_threshold=0.05, show_plots=False
        )
        violations = []
        for entry in results:
            table = getattr(entry, "summary", None)
            if table is None:
                continue
            for name, row in table.iterrows():
                if float(row.get("p", 1.0)) < 0.05:
                    violations.append({"feature": str(name), "p_value": round(float(row["p"]), 8)})
        ph_test = {
            "schoenfeld_violations": violations,
            "assumption_holds": not violations,
            "interpretation": (
                "The proportional-hazards assumption is violated for the listed covariates. "
                "That is expected for loan age — the seasoning ramp is genuinely non-proportional "
                "— and it is precisely the argument for the discrete-time hazard model, which "
                "makes no proportionality assumption at all."
                if violations
                else "No proportional-hazards violation detected at p < 0.05."
            ),
        }
    except Exception as exc:  # pragma: no cover - lifelines plotting path
        ph_test = {"error": str(exc)}

    return {
        "method": "cox_ph",
        "available": True,
        "cause": cause,
        "n": int(len(merged)),
        "n_events": int(merged["event"].sum()),
        "concordance_index": round(float(fitter.concordance_index_), 6),
        "log_likelihood": round(float(fitter.log_likelihood_), 4),
        "hazard_ratios": ratios,
        "proportional_hazards_test": ph_test,
        "features": use,
    }


def markov_baseline(
    panel: pd.DataFrame, *, states: list[str], horizon: int = 24
) -> dict[str, Any]:
    """First-order Markov chain from the empirical transition matrix, no covariates.

    The structural baseline: it shows how much covariates actually add over the
    portfolio's average transition behaviour.
    """
    sub = panel.dropna(subset=["current_status", "next_state"])
    if sub.empty:
        return {"method": "markov", "available": False, "reason": "no transitions observed"}

    counts = pd.crosstab(sub["current_status"], sub["next_state"]).reindex(
        index=states, columns=states, fill_value=0
    )
    totals = counts.sum(axis=1).replace(0, np.nan)
    matrix = counts.div(totals, axis=0).fillna(0.0)
    for state in states:
        if matrix.loc[state].sum() == 0:
            matrix.loc[state, state] = 1.0
    M = matrix.to_numpy(dtype="float64")

    index = {s: i for i, s in enumerate(states)}
    absorbing = M.copy()
    for state in EVENT_STATES:
        if state in index:
            i = index[state]
            absorbing[i, :] = 0.0
            absorbing[i, i] = 1.0

    start_counts = panel["current_status"].astype("object").value_counts()
    pi = np.array([float(start_counts.get(s, 0)) for s in states])
    pi = pi / max(pi.sum(), 1.0)

    occupancy = np.zeros((horizon + 1, len(states)))
    occupancy[0] = pi
    for m in range(1, horizon + 1):
        occupancy[m] = occupancy[m - 1] @ absorbing

    active_idx = [index[s] for s in ACTIVE_STATES if s in index]
    return {
        "method": "markov",
        "available": True,
        "transition_matrix": {
            src: {dst: round(float(matrix.loc[src, dst]), 6) for dst in states} for src in states
        },
        "states": states,
        "timeline": list(range(horizon + 1)),
        "survival": [round(float(occupancy[m, active_idx].sum()), 6) for m in range(horizon + 1)],
        "cif_default": [
            round(float(occupancy[m, index["Default"]]), 6) for m in range(horizon + 1)
        ] if "Default" in index else [],
        "cif_prepaid": [
            round(float(occupancy[m, index["Prepaid"]]), 6) for m in range(horizon + 1)
        ] if "Prepaid" in index else [],
        "n_transitions": int(len(sub)),
        "note": "No covariates. Any lift the hazard model shows over this is what features buy.",
    }


# --------------------------------------------------------------------------- #
# survival-specific evaluation
# --------------------------------------------------------------------------- #
def concordance_index(
    durations: np.ndarray, predicted_risk: np.ndarray, events: np.ndarray
) -> float | None:
    """Harrell's C-index. Higher predicted risk should mean shorter time to event."""
    from lifelines.utils import concordance_index as _ci

    try:
        return round(float(_ci(durations, -np.asarray(predicted_risk), events)), 6)
    except Exception:
        return None


def integrated_brier_score(
    survival_curves: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    timeline: np.ndarray,
) -> dict[str, Any]:
    """IPCW-weighted integrated Brier score.

    The single best overall survival score: it measures accuracy *and*
    calibration, and the inverse-probability-of-censoring weights correct for
    the fact that later time points have fewer uncensored observations.
    """
    from lifelines import KaplanMeierFitter

    durations = np.asarray(durations, dtype="float64")
    events = np.asarray(events, dtype="float64")
    timeline = np.asarray(timeline, dtype="float64")
    if survival_curves.size == 0 or len(durations) == 0:
        return {"ibs": None, "n": 0}

    censoring = KaplanMeierFitter()
    censoring.fit(durations, 1 - events)

    scores, weights_used = [], []
    for j, t in enumerate(timeline):
        if j >= survival_curves.shape[1]:
            break
        S = survival_curves[:, j]
        g_t = float(censoring.survival_function_at_times(t).iloc[0])
        g_ti = censoring.survival_function_at_times(np.minimum(durations, t)).to_numpy()

        term = np.zeros(len(durations), dtype="float64")
        # Case 1: event before t -> the loan should have S(t) near 0.
        early = (durations <= t) & (events == 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            term[early] = (S[early] ** 2) / np.maximum(g_ti[early], 1e-9)
            # Case 2: still at risk at t -> the loan should have S(t) near 1.
            late = durations > t
            term[late] = ((1.0 - S[late]) ** 2) / max(g_t, 1e-9)
        scores.append(float(np.mean(term)))
        weights_used.append(float(t))

    if not scores:
        return {"ibs": None, "n": int(len(durations))}
    ibs = float(np.trapezoid(scores, weights_used) / max(weights_used[-1] - weights_used[0], 1.0))
    return {
        "ibs": round(ibs, 6),
        "brier_by_time": [
            {"t": round(t, 2), "brier": round(s, 6)} for t, s in zip(weights_used, scores, strict=False)
        ],
        "n": int(len(durations)),
        "note": "IPCW-corrected; lower is better. 0.25 is the uninformative reference.",
    }


def time_dependent_auc(
    predicted_risk: np.ndarray, durations: np.ndarray, events: np.ndarray, horizons: tuple[int, ...]
) -> list[dict[str, Any]]:
    """AUC(t): discrimination as a function of horizon."""
    from sklearn.metrics import roc_auc_score

    durations = np.asarray(durations, dtype="float64")
    events = np.asarray(events, dtype="float64")
    risk = np.asarray(predicted_risk, dtype="float64")

    out = []
    for t in horizons:
        label = ((durations <= t) & (events == 1)).astype("float64")
        # Loans censored before t carry no information about the event by t.
        usable = ~((durations < t) & (events == 0))
        if usable.sum() < 50 or len(np.unique(label[usable])) < 2:
            out.append({"horizon": int(t), "auc": None, "n": int(usable.sum())})
            continue
        out.append(
            {
                "horizon": int(t),
                "auc": round(float(roc_auc_score(label[usable], risk[usable])), 6),
                "n": int(usable.sum()),
                "n_events": int(label[usable].sum()),
            }
        )
    return out
