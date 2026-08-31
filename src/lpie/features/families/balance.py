"""F2 — Amortisation and balance dynamics (18).

The centrepiece is `amort_residual = current_balance - scheduled_balance`, the
single most interpretable balance-anomaly feature: it separates curtailment
(negative residual), negative amortisation (positive, with a modification), and
plain data error (positive, without one). `SMM` and `CPR` are the industry
standard prepayment speeds and make the prepayment head speak the language a
credit-risk reviewer already uses.

Every lag and rolling window is a backward `groupby(loan_id).shift(k)` /
`.rolling(w)` — never `center=True`, never a negative shift.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lpie.features.registry import FeatureSpec, spec

FAMILY = "balance"

SPECS: list[FeatureSpec] = [
    spec("current_balance", FAMILY, "Current unpaid principal balance", ["current_balance"]),
    spec("bal_ratio", FAMILY, "current_balance / original_balance", ["current_balance", "original_balance"]),
    spec("scheduled_balance", FAMILY,
         "Contractual balance from the standard annuity formula given rate, term and age",
         ["original_balance", "interest_rate", "loan_term_months", "loan_age_months"]),
    spec("amort_residual", FAMILY,
         "current_balance - scheduled_balance: negative = curtailment, positive = neg-am or data error",
         ["current_balance", "original_balance", "interest_rate", "loan_term_months", "loan_age_months"]),
    spec("amort_residual_pct", FAMILY, "amort_residual as a share of the scheduled balance",
         ["current_balance", "original_balance", "interest_rate", "loan_term_months", "loan_age_months"]),
    spec("bal_diff_1", FAMILY, "Balance change vs 1 month ago", ["current_balance"], temporal_offset=-1),
    spec("bal_diff_3", FAMILY, "Balance change vs 3 months ago", ["current_balance"], temporal_offset=-3),
    spec("bal_diff_6", FAMILY, "Balance change vs 6 months ago", ["current_balance"], temporal_offset=-6),
    spec("bal_diff_12", FAMILY, "Balance change vs 12 months ago", ["current_balance"], temporal_offset=-12),
    spec("pay_rate", FAMILY, "Monthly principal reduction as a share of original balance",
         ["current_balance", "original_balance"], temporal_offset=-1, winsorised=True),
    spec("smm", FAMILY, "Single Monthly Mortality = -dBalance / previous balance",
         ["current_balance"], temporal_offset=-1, winsorised=True),
    spec("cpr", FAMILY, "Conditional Prepayment Rate = 1 - (1 - SMM)^12, the industry speed measure",
         ["current_balance"], temporal_offset=-1, winsorised=True),
    spec("bal_volatility_6m", FAMILY, "Std of the last 6 monthly balance changes", ["current_balance"],
         temporal_offset=-6),
    spec("bal_trend_6m", FAMILY, "Mean of the last 6 monthly balance changes", ["current_balance"],
         temporal_offset=-6),
    spec("months_since_balance_change", FAMILY, "Months since the balance last moved (a frozen feed signal)",
         ["current_balance"], temporal_offset=-1),
    spec("balance_zero_flag", FAMILY, "Balance is exactly zero", ["current_balance"]),
    spec("balance_exceeds_original_flag", FAMILY,
         "current_balance > 1.05 x original_balance — VR-001's condition as a model input",
         ["current_balance", "original_balance"]),
    spec("modification_flag", FAMILY, "Loan has been modified", ["modification_flag"]),
    spec("months_since_modification", FAMILY, "Months since the modification flag first appeared",
         ["modification_flag"], temporal_offset=-1),
    spec("bal_z_self_12m", FAMILY,
         "Robust z-score of the balance against this loan's own trailing 12-month median/MAD",
         ["current_balance"], temporal_offset=-12),
]

WINSOR_LOW, WINSOR_HIGH = 0.001, 0.999
WINSORISED_FEATURES = ("pay_rate", "smm")


def scheduled_balance(
    original: pd.Series, annual_rate: pd.Series, term: pd.Series, age: pd.Series
) -> pd.Series:
    """Standard annuity remaining balance:  B_n = B0 * (( (1+r)^N - (1+r)^n ) / ( (1+r)^N - 1 )).

    Zero-rate loans degrade to straight-line amortisation rather than dividing by
    zero — a small correctness detail that keeps the residual meaningful for the
    handful of 0% rows a real feed will contain.
    """
    r = pd.to_numeric(annual_rate, errors="coerce") / 1200.0
    n = pd.to_numeric(age, errors="coerce").clip(lower=0)
    N = pd.to_numeric(term, errors="coerce").clip(lower=1)
    B0 = pd.to_numeric(original, errors="coerce")

    n_eff = np.minimum(n, N)
    straight = B0 * (1.0 - n_eff / N)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        growth_N = np.power(1.0 + r, N)
        growth_n = np.power(1.0 + r, n_eff)
        denom = growth_N - 1.0
        annuity = B0 * (growth_N - growth_n) / denom

    out = pd.Series(np.where(r.abs() > 1e-12, annuity, straight), index=original.index)
    return out.replace([np.inf, -np.inf], np.nan).clip(lower=0.0)


def fit_winsor_bounds(series: pd.Series) -> tuple[float, float] | None:
    finite = series.replace([np.inf, -np.inf], np.nan)
    lo, hi = finite.quantile(WINSOR_LOW), finite.quantile(WINSOR_HIGH)
    if pd.isna(lo) or pd.isna(hi):
        return None
    return float(lo), float(hi)


def _winsorise(series: pd.Series, bounds: tuple[float, float] | None) -> pd.Series:
    """Clip division-by-near-zero blowups using *fitted* bounds.

    The bounds are learned on the training window and then held fixed, exactly
    like any other model parameter. Recomputing quantiles over whatever batch
    happens to arrive would make the same loan-month score differently depending
    on its neighbours — and would silently read the future during backtesting.
    """
    finite = series.replace([np.inf, -np.inf], np.nan)
    if bounds is None:
        return finite
    return finite.clip(lower=bounds[0], upper=bounds[1])


def build(
    panel: pd.DataFrame,
    static_features: pd.DataFrame,
    fit: dict[str, Any] | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame(index=panel.index)
    grp = panel.groupby("loan_id", sort=False)

    bal = pd.to_numeric(panel["current_balance"], errors="coerce")
    orig = pd.to_numeric(static_features["original_balance"], errors="coerce")
    rate = pd.to_numeric(static_features["interest_rate"], errors="coerce")
    term = pd.to_numeric(static_features["loan_term_months"], errors="coerce")
    age = pd.to_numeric(panel["loan_age_months"], errors="coerce")

    out["current_balance"] = bal
    out["bal_ratio"] = bal / orig.replace(0.0, np.nan)

    sched = scheduled_balance(orig, rate, term, age)
    out["scheduled_balance"] = sched
    out["amort_residual"] = bal - sched
    out["amort_residual_pct"] = out["amort_residual"] / sched.replace(0.0, np.nan)

    bal_by_loan = bal.groupby(panel["loan_id"], sort=False)
    for k in (1, 3, 6, 12):
        out[f"bal_diff_{k}"] = bal - bal_by_loan.shift(k)

    fit = fit or {}
    prev = bal_by_loan.shift(1)
    raw_pay_rate = -(bal - prev) / orig.replace(0.0, np.nan)
    raw_smm = -(bal - prev) / prev.replace(0.0, np.nan)
    out["pay_rate"] = _winsorise(raw_pay_rate, fit.get("pay_rate_bounds"))
    smm = _winsorise(raw_smm, fit.get("smm_bounds"))
    out["smm"] = smm
    out["cpr"] = 1.0 - np.power((1.0 - smm.clip(-0.99, 0.99)), 12)

    diffs = (bal - prev).groupby(panel["loan_id"], sort=False)
    out["bal_volatility_6m"] = diffs.rolling(6, min_periods=2).std().reset_index(level=0, drop=True)
    out["bal_trend_6m"] = diffs.rolling(6, min_periods=2).mean().reset_index(level=0, drop=True)

    changed = (bal - prev).abs() > 1e-6
    out["months_since_balance_change"] = _months_since(changed, panel["loan_id"])

    out["balance_zero_flag"] = (bal.abs() < 1e-6).astype("float64")
    out["balance_exceeds_original_flag"] = (bal > orig * 1.05).astype("float64")

    mod = pd.to_numeric(panel.get("modification_flag"), errors="coerce").fillna(0.0)
    out["modification_flag"] = mod
    out["months_since_modification"] = _months_since(mod > 0, panel["loan_id"])

    # Per-loan self-referential z-score: "is this balance weird *for this loan*",
    # which a global outlier test misses entirely on a panel.
    med = bal_by_loan.rolling(12, min_periods=3).median().reset_index(level=0, drop=True).groupby(
        panel["loan_id"], sort=False
    ).shift(1)
    mad = (
        bal_by_loan.rolling(12, min_periods=3)
        .apply(lambda w: np.nanmedian(np.abs(w - np.nanmedian(w))), raw=True)
        .reset_index(level=0, drop=True)
        .groupby(panel["loan_id"], sort=False)
        .shift(1)
    )
    out["bal_z_self_12m"] = ((bal - med) / (1.4826 * mad).replace(0.0, np.nan)).clip(-25, 25)

    del grp
    return out


def _months_since(condition: pd.Series, groups: pd.Series) -> pd.Series:
    """Months since `condition` was last True, counted strictly in the past.

    Implemented as a cumulative-max of the position where the event last
    occurred, shifted by one month so the current row never sees its own event.
    """
    cond = condition.fillna(False).astype(bool)
    position = cond.groupby(groups, sort=False).cumcount()
    marked = position.where(cond, np.nan)
    last = marked.groupby(groups, sort=False).ffill()
    last = last.groupby(groups, sort=False).shift(1)
    prev_position = position.groupby(groups, sort=False).shift(1)
    return (prev_position - last + 1).astype("float64")
