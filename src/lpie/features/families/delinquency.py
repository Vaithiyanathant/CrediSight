"""F3 — Delinquency path (22). The highest-value family for the default head.

The measured transition matrix shows cure paths are real (90DPD -> Current is
15.8%), so this is not a monotone deterioration process and a plain roll-rate
model is wrong. These features therefore capture *path shape* — recency, peaks,
instability, cures, entropy — rather than just the current bucket.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lpie.features.registry import FeatureSpec, spec

FAMILY = "delinquency"

STATUS_SEVERITY = {
    "Current": 0, "Prepaid": 0, "Closed": 0,
    "30DPD": 1, "60DPD": 2, "90DPD": 3, "Default": 4,
}
STATUS_TO_DPD = {"Current": 0.0, "30DPD": 30.0, "60DPD": 60.0, "90DPD": 90.0, "Default": 120.0}
DELINQUENT_STATES = ("30DPD", "60DPD", "90DPD", "Default")

SPECS: list[FeatureSpec] = [
    spec("dpd", FAMILY, "Days past due, imputed from current_status where null via the measured mapping",
         ["days_past_due", "current_status"]),
    spec("dpd_was_imputed", FAMILY, "days_past_due was null and recovered from current_status",
         ["days_past_due"]),
    spec("current_status", FAMILY, "Loan status at month end (native categorical)", ["current_status"],
         dtype="category", categorical=True),
    spec("status_severity", FAMILY, "Ordinal severity of current_status (Current 0 .. Default 4)",
         ["current_status"], ordinal=True),
    *[
        spec(f"dpd_lag_{k}", FAMILY, f"days_past_due {k} month(s) ago", ["days_past_due", "current_status"],
             temporal_offset=-k)
        for k in (1, 2, 3, 6, 12)
    ],
    spec("dpd_delta_1", FAMILY, "Change in DPD vs last month", ["days_past_due", "current_status"],
         temporal_offset=-1),
    spec("dpd_delta_3", FAMILY, "Change in DPD vs 3 months ago", ["days_past_due", "current_status"],
         temporal_offset=-3),
    *[
        spec(f"dpd_max_{w}m", FAMILY, f"Worst DPD over the trailing {w} months",
             ["days_past_due", "current_status"], temporal_offset=-w)
        for w in (3, 6, 12)
    ],
    spec("dpd_max_life", FAMILY, "Worst DPD observed to date (expanding, shifted)",
         ["days_past_due", "current_status"], temporal_offset=-1),
    spec("dpd_mean_3m", FAMILY, "Mean DPD over the trailing 3 months", ["days_past_due", "current_status"],
         temporal_offset=-3),
    spec("dpd_mean_6m", FAMILY, "Mean DPD over the trailing 6 months", ["days_past_due", "current_status"],
         temporal_offset=-6),
    spec("n_delinquent_months_6m", FAMILY, "Delinquent months in the trailing 6", ["current_status"],
         temporal_offset=-6),
    spec("n_delinquent_months_12m", FAMILY, "Delinquent months in the trailing 12", ["current_status"],
         temporal_offset=-12),
    spec("n_delinquent_months_life", FAMILY, "Delinquent months to date", ["current_status"],
         temporal_offset=-1),
    spec("months_since_first_delinquency", FAMILY, "Months since this loan first went delinquent",
         ["current_status"], temporal_offset=-1),
    spec("months_since_last_current", FAMILY, "Cure recency — months since the loan was last Current",
         ["current_status"], temporal_offset=-1),
    spec("worst_status_life", FAMILY, "Worst status severity observed to date", ["current_status"],
         temporal_offset=-1, ordinal=True),
    spec("n_status_changes_12m", FAMILY, "Status transitions in the trailing 12 months — instability",
         ["current_status"], temporal_offset=-12),
    spec("roll_rate_own_12m", FAMILY,
         "This loan's own historical P(deteriorate | its bucket) over the trailing 12 months",
         ["current_status"], temporal_offset=-12,
         leakage_risk="low",
         justification=(
             "Computed only from this loan's own months strictly before t via a shifted "
             "rolling window; the truncation test in tests/test_leakage.py asserts "
             "equality against a recomputation from a truncated panel."
         )),
    spec("consecutive_current_streak", FAMILY, "Consecutive months Current, ending last month",
         ["current_status"], temporal_offset=-1),
    spec("ever_60plus", FAMILY, "Has ever been 60DPD or worse before now", ["current_status"],
         temporal_offset=-1),
    spec("ever_90plus", FAMILY, "Has ever been 90DPD or worse before now", ["current_status"],
         temporal_offset=-1),
    spec("cure_count", FAMILY, "Number of transitions back to Current from delinquency, to date",
         ["current_status"], temporal_offset=-1),
    spec("delinquency_entropy_12m", FAMILY,
         "Shannon entropy of the status distribution over the trailing 12 months",
         ["current_status"], temporal_offset=-12),
    spec("is_terminal", FAMILY, "Loan is in an absorbing state (Prepaid or Closed)", ["current_status"]),
    spec("default_flag", FAMILY, "Contemporaneous default indicator", ["default_flag"]),
]


def impute_dpd(panel: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Recover null days_past_due from current_status.

    Profiling measures `current_status -> days_past_due` as a functional
    dependency with strength 1.0. Imputing through a measured dependency
    recovers real information; mean imputation would destroy it and invent a
    value no loan ever had.
    """
    raw = pd.to_numeric(panel.get("days_past_due"), errors="coerce")
    status = panel.get("current_status")
    if status is None:
        return raw, pd.Series(0.0, index=panel.index)
    recovered = status.map(STATUS_TO_DPD).astype("float64")
    was_null = raw.isna() & recovered.notna()
    return raw.where(raw.notna(), recovered), was_null.astype("float64")


def build(panel: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=panel.index)
    loan = panel["loan_id"]
    status = panel["current_status"].astype("object")

    dpd, imputed = impute_dpd(panel)
    out["dpd"] = dpd
    out["dpd_was_imputed"] = imputed
    out["current_status"] = status
    severity = status.map(STATUS_SEVERITY).astype("float64")
    out["status_severity"] = severity

    dpd_by_loan = dpd.groupby(loan, sort=False)
    for k in (1, 2, 3, 6, 12):
        out[f"dpd_lag_{k}"] = dpd_by_loan.shift(k)
    out["dpd_delta_1"] = dpd - out["dpd_lag_1"]
    out["dpd_delta_3"] = dpd - out["dpd_lag_3"]

    # Rolling windows include the current row: DPD at t is observed at t, it is
    # not a forward value. Only *derived path history* is shifted.
    for w in (3, 6, 12):
        out[f"dpd_max_{w}m"] = (
            dpd_by_loan.rolling(w, min_periods=1).max().reset_index(level=0, drop=True)
        )
    out["dpd_mean_3m"] = dpd_by_loan.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    out["dpd_mean_6m"] = dpd_by_loan.rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)
    out["dpd_max_life"] = dpd_by_loan.expanding().max().reset_index(level=0, drop=True)

    is_delinq = status.isin(DELINQUENT_STATES).astype("float64")
    dq_by_loan = is_delinq.groupby(loan, sort=False)
    out["n_delinquent_months_6m"] = dq_by_loan.rolling(6, min_periods=1).sum().reset_index(level=0, drop=True)
    out["n_delinquent_months_12m"] = dq_by_loan.rolling(12, min_periods=1).sum().reset_index(level=0, drop=True)
    out["n_delinquent_months_life"] = dq_by_loan.expanding().sum().reset_index(level=0, drop=True)

    out["months_since_first_delinquency"] = _months_since_first(is_delinq > 0, loan)
    out["months_since_last_current"] = _months_since_last(status == "Current", loan)

    sev_by_loan = severity.groupby(loan, sort=False)
    out["worst_status_life"] = sev_by_loan.expanding().max().reset_index(level=0, drop=True)

    changed = (status != status.groupby(loan, sort=False).shift(1)).astype("float64")
    changed.iloc[:] = np.where(status.groupby(loan, sort=False).shift(1).isna(), 0.0, changed)
    out["n_status_changes_12m"] = (
        changed.groupby(loan, sort=False).rolling(12, min_periods=1).sum().reset_index(level=0, drop=True)
    )

    # roll_rate_own: share of the loan's trailing transitions that deteriorated.
    prev_sev = sev_by_loan.shift(1)
    deteriorated = ((severity > prev_sev) & prev_sev.notna()).astype("float64")
    observed = prev_sev.notna().astype("float64")
    det_roll = deteriorated.groupby(loan, sort=False).rolling(12, min_periods=1).sum().reset_index(level=0, drop=True)
    obs_roll = observed.groupby(loan, sort=False).rolling(12, min_periods=1).sum().reset_index(level=0, drop=True)
    # Shift by one so the current transition never contributes to its own rate.
    out["roll_rate_own_12m"] = (
        (det_roll / obs_roll.replace(0.0, np.nan)).groupby(loan, sort=False).shift(1)
    )

    out["consecutive_current_streak"] = _consecutive_streak(status == "Current", loan)
    ever60 = (severity >= 2).astype("float64").groupby(loan, sort=False).cummax()
    ever90 = (severity >= 3).astype("float64").groupby(loan, sort=False).cummax()
    out["ever_60plus"] = ever60.groupby(loan, sort=False).shift(1).fillna(0.0)
    out["ever_90plus"] = ever90.groupby(loan, sort=False).shift(1).fillna(0.0)

    cured = ((status == "Current") & (prev_sev > 0)).astype("float64")
    out["cure_count"] = (
        cured.groupby(loan, sort=False).cumsum().groupby(loan, sort=False).shift(1).fillna(0.0)
    )

    out["delinquency_entropy_12m"] = _rolling_entropy(severity, loan, window=12)

    out["is_terminal"] = status.isin(["Prepaid", "Closed"]).astype("float64")
    out["default_flag"] = pd.to_numeric(panel.get("default_flag"), errors="coerce").fillna(0.0)
    return out


def _months_since_first(condition: pd.Series, groups: pd.Series) -> pd.Series:
    cond = condition.fillna(False).astype(bool)
    position = cond.groupby(groups, sort=False).cumcount()
    first = position.where(cond, np.nan).groupby(groups, sort=False).cummin()
    first = first.groupby(groups, sort=False).ffill()
    return (position - first).astype("float64")


def _months_since_last(condition: pd.Series, groups: pd.Series) -> pd.Series:
    cond = condition.fillna(False).astype(bool)
    position = cond.groupby(groups, sort=False).cumcount()
    last = position.where(cond, np.nan).groupby(groups, sort=False).ffill()
    return (position - last).astype("float64")


def _consecutive_streak(condition: pd.Series, groups: pd.Series) -> pd.Series:
    """Length of the run of consecutive True values ending at t-1."""
    cond = condition.fillna(False).astype(bool)
    prev = cond.groupby(groups, sort=False).shift(1).astype("boolean").fillna(False).astype(bool)
    block = (~prev).groupby(groups, sort=False).cumsum()
    streak = prev.groupby([groups, block], sort=False).cumsum()
    return streak.astype("float64")


def _rolling_entropy(severity: pd.Series, groups: pd.Series, window: int = 12) -> pd.Series:
    """Shannon entropy of the severity distribution over a trailing window.

    High entropy means the loan bounces between buckets — instability that a
    single current-status feature cannot express.
    """
    def _entropy(values: np.ndarray) -> float:
        v = values[~np.isnan(values)]
        if v.size == 0:
            return np.nan
        _, counts = np.unique(v, return_counts=True)
        p = counts / counts.sum()
        return float(-(p * np.log(p)).sum())

    return (
        severity.groupby(groups, sort=False)
        .rolling(window, min_periods=2)
        .apply(_entropy, raw=True)
        .reset_index(level=0, drop=True)
    )
