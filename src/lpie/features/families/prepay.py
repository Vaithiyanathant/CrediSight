"""F4 — Prepayment / refinance incentive (12). Domain-specific by design.

`burnout` is the one to notice: the cumulative unexercised refinance incentive.
A loan that has sat on a 200bp incentive for eighteen months and *still* has not
refinanced has revealed something about the borrower that no static attribute
captures — and it is the canonical explanation for why high-rate seasoned loans
prepay slower than their incentive implies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lpie.features.registry import FeatureSpec, spec

FAMILY = "prepay"

SPECS: list[FeatureSpec] = [
    spec("refi_incentive", FAMILY,
         "Note rate minus the portfolio median rate observed at t (point-in-time)",
         ["interest_rate"], leakage_risk="low",
         justification=(
             "The portfolio median is computed within month t across loans, which is "
             "information available at scoring time for the whole book; it uses no "
             "future month and no outcome column."
         )),
    spec("refi_incentive_x_ltv", FAMILY, "Refi incentive interacted with LTV band ordinal",
         ["interest_rate", "ltv_band"]),
    spec("burnout", FAMILY,
         "Cumulative unexercised refinance incentive: sum of max(0, incentive) over months < t",
         ["interest_rate"], temporal_offset=-1),
    spec("burnout_months", FAMILY, "Count of prior months with a positive refinance incentive",
         ["interest_rate"], temporal_offset=-1),
    spec("seasoning_frac", FAMILY, "age / (age + remaining_term) — position along the amortisation curve",
         ["loan_age_months", "remaining_term_months"]),
    spec("age_bucket", FAMILY, "Seasoning ramp bucket: 0-12 / 13-36 / 37-60 / 60+",
         ["loan_age_months"], ordinal=True),
    spec("loan_size_decile", FAMILY,
         "Original balance decile — large loans refinance faster (closing costs amortise better)",
         ["original_balance"]),
    spec("is_refi_purpose", FAMILY, "Loan purpose is a refinance", ["loan_purpose"]),
    spec("months_to_maturity", FAMILY, "Remaining term in months", ["remaining_term_months"]),
    spec("rate_percentile_in_vintage", FAMILY, "Rate percentile within the loan's origination vintage",
         ["interest_rate", "vintage_year"]),
    spec("sato", FAMILY, "Spread at origination: note rate minus the vintage median rate",
         ["interest_rate", "vintage_year"]),
    spec("loan_age_months", FAMILY, "Months since origination", ["loan_age_months"]),
    spec("prepayment_flag", FAMILY, "Contemporaneous prepayment indicator", ["prepayment_flag"]),
]

AGE_BUCKET_EDGES = [-np.inf, 12, 36, 60, np.inf]


def build(
    panel: pd.DataFrame,
    static_features: pd.DataFrame,
    fit: dict[str, Any] | None = None,
) -> pd.DataFrame:
    fit = fit or {}
    out = pd.DataFrame(index=panel.index)
    month = panel["month_index"]
    loan = panel["loan_id"]
    rate = pd.to_numeric(static_features["interest_rate"], errors="coerce")

    # Cross-sectional median within the same month: available to a scorer that
    # holds the whole book at t, and uses no future month.
    portfolio_median = rate.groupby(month, sort=False).transform("median")
    incentive = rate - portfolio_median
    out["refi_incentive"] = incentive
    out["refi_incentive_x_ltv"] = incentive * pd.to_numeric(
        static_features.get("ltv_band_ord"), errors="coerce"
    )

    positive = incentive.clip(lower=0.0)
    by_loan = positive.groupby(loan, sort=False)
    out["burnout"] = by_loan.cumsum().groupby(loan, sort=False).shift(1)
    out["burnout_months"] = (
        (positive > 0).astype("float64").groupby(loan, sort=False).cumsum()
        .groupby(loan, sort=False).shift(1)
    )

    age = pd.to_numeric(panel["loan_age_months"], errors="coerce")
    rem = pd.to_numeric(panel["remaining_term_months"], errors="coerce")
    out["loan_age_months"] = age
    out["months_to_maturity"] = rem
    out["seasoning_frac"] = age / (age + rem).replace(0.0, np.nan)
    out["age_bucket"] = pd.cut(age, bins=AGE_BUCKET_EDGES, labels=False).astype("float64")

    orig = pd.to_numeric(static_features["original_balance"], errors="coerce")
    out["loan_size_decile"] = _apply_bins(orig, fit.get("loan_size_edges"))

    purpose = static_features.get("loan_purpose")
    out["is_refi_purpose"] = (
        purpose.astype("string").str.contains("Refi", case=False, na=False).astype("float64")
        if purpose is not None
        else pd.Series(np.nan, index=panel.index)
    )

    # Vintage statistics are loan-level facts about origination, fitted once on
    # the training book and then held fixed. Computing them per batch would make
    # a loan's SATO depend on which other loans happened to be scored with it.
    vintage = pd.to_numeric(static_features.get("vintage_year_num"), errors="coerce")
    medians = fit.get("vintage_rate_median") or {}
    vintage_median = vintage.map(medians).astype("float64")
    out["sato"] = rate - vintage_median
    out["rate_percentile_in_vintage"] = _vintage_percentile(rate, vintage, fit.get("vintage_rate_values"))

    out["prepayment_flag"] = pd.to_numeric(panel.get("prepayment_flag"), errors="coerce").fillna(0.0)
    return out


def fit_params(static_features: pd.DataFrame) -> dict[str, Any]:
    """Learn the loan-level constants this family needs from the training book."""
    orig = pd.to_numeric(static_features["original_balance"], errors="coerce").dropna()
    try:
        edges = np.unique(np.quantile(orig, np.linspace(0, 1, 11))).tolist() if len(orig) else None
    except (ValueError, TypeError):
        edges = None

    rate = pd.to_numeric(static_features["interest_rate"], errors="coerce")
    vintage = pd.to_numeric(static_features.get("vintage_year_num"), errors="coerce")
    grouped = rate.groupby(vintage, sort=True)
    return {
        "loan_size_edges": edges,
        "vintage_rate_median": {float(k): float(v) for k, v in grouped.median().items() if pd.notna(k)},
        "vintage_rate_values": {
            float(k): np.sort(v.dropna().to_numpy()).tolist()
            for k, v in grouped
            if pd.notna(k) and v.notna().any()
        },
    }


def _apply_bins(series: pd.Series, edges: list[float] | None) -> pd.Series:
    if not edges or len(edges) < 3:
        return pd.Series(np.nan, index=series.index)
    bins = np.array(edges, dtype="float64")
    bins[0], bins[-1] = -np.inf, np.inf
    return pd.Series(
        np.digitize(series.to_numpy(dtype="float64"), bins[1:-1], right=True), index=series.index
    ).where(series.notna(), np.nan).astype("float64")


def _vintage_percentile(
    rate: pd.Series, vintage: pd.Series, reference: dict[float, list[float]] | None
) -> pd.Series:
    """Percentile of the note rate within its vintage, against the fitted reference."""
    if not reference:
        return pd.Series(np.nan, index=rate.index)
    out = pd.Series(np.nan, index=rate.index, dtype="float64")
    for v, values in reference.items():
        mask = vintage == v
        if not mask.any() or not values:
            continue
        arr = np.asarray(values, dtype="float64")
        out.loc[mask] = np.searchsorted(arr, rate[mask].to_numpy(dtype="float64"), side="right") / len(arr)
    return out
