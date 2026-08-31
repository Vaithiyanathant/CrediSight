"""F5 — Cohort and peer-relative features (10).

This is the technically riskiest family and where most submissions leak. Every
aggregate is an **expanding window over months strictly before t**, built as
`expanding().mean().shift(1)` on a month-level aggregate and then broadcast back
to rows. The loan's own value is then expressed as a z-score against its peer
group.

`servicer_dq_rate_to_date` is the most valuable member: servicer operational
quality is a real, learnable signal that no single-loan feature can express.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lpie.features.registry import FeatureSpec, spec

FAMILY = "cohort"

_JUSTIFY = (
    "Peer aggregates are expanding means over months strictly less than t "
    "(monthly aggregate -> expanding().mean() -> shift(1)), so no row can see its "
    "own month or any later month. tests/test_leakage.py::test_cohort_features_are_"
    "point_in_time asserts this by recomputation from a truncated panel."
)

SPECS: list[FeatureSpec] = [
    spec("peer_vintage_credit_dpd_mean", FAMILY,
         "Mean DPD of the (vintage, credit band) cohort over months < t",
         ["days_past_due", "current_status", "vintage_year", "credit_score_band"],
         temporal_offset=-1, leakage_risk="medium", justification=_JUSTIFY),
    spec("peer_vintage_credit_delinq_rate", FAMILY,
         "Delinquency rate of the (vintage, credit band) cohort over months < t",
         ["current_status", "vintage_year", "credit_score_band"],
         temporal_offset=-1, leakage_risk="medium", justification=_JUSTIFY),
    spec("peer_state_delinq_rate", FAMILY, "Delinquency rate of the loan's state over months < t",
         ["current_status", "state"], temporal_offset=-1, leakage_risk="medium", justification=_JUSTIFY),
    spec("peer_servicer_delinq_rate", FAMILY, "Delinquency rate of the loan's servicer over months < t",
         ["current_status", "servicer_name"], temporal_offset=-1, leakage_risk="medium", justification=_JUSTIFY),
    spec("peer_servicer_prepay_rate", FAMILY, "Prepayment rate of the loan's servicer over months < t",
         ["prepayment_flag", "servicer_name"], temporal_offset=-1, leakage_risk="medium", justification=_JUSTIFY),
    spec("servicer_dq_rate_to_date", FAMILY,
         "Share of the servicer's records violating any rule, over months < t — operational quality",
         ["servicer_name"], temporal_offset=-1, leakage_risk="medium", justification=_JUSTIFY),
    spec("dpd_z_vs_peer", FAMILY, "This loan's DPD as a z-score against its (vintage, credit band) cohort",
         ["days_past_due", "current_status", "vintage_year", "credit_score_band"],
         temporal_offset=-1, leakage_risk="medium", justification=_JUSTIFY),
    spec("bal_ratio_z_vs_peer", FAMILY, "Balance ratio as a z-score against the cohort",
         ["current_balance", "original_balance", "vintage_year", "credit_score_band"],
         temporal_offset=-1, leakage_risk="medium", justification=_JUSTIFY),
    spec("portfolio_delinq_rate_to_date", FAMILY, "Whole-portfolio delinquency rate over months < t",
         ["current_status"], temporal_offset=-1, leakage_risk="low", justification=_JUSTIFY),
    spec("portfolio_prepay_rate_to_date", FAMILY, "Whole-portfolio prepayment rate over months < t",
         ["prepayment_flag"], temporal_offset=-1, leakage_risk="low", justification=_JUSTIFY),
]

DELINQUENT_STATES = ("30DPD", "60DPD", "90DPD", "Default")


def _expanding_prior_mean(
    values: pd.Series, month: pd.Series, keys: list[pd.Series] | None = None
) -> pd.Series:
    """Mean of `values` over all months strictly before each row's month.

    Aggregated to (key, month) first, then expanded and shifted — so the result
    for month t is a function of months 1..t-1 only, for every row in month t.
    """
    frame = pd.DataFrame({"_m": month.to_numpy(), "_v": pd.to_numeric(values, errors="coerce").to_numpy()})
    group_cols: list[str] = []
    if keys:
        for i, k in enumerate(keys):
            col = f"_k{i}"
            frame[col] = k.astype("object").to_numpy()
            group_cols.append(col)

    agg = frame.groupby([*group_cols, "_m"], dropna=False, sort=True)["_v"].agg(["sum", "count"])
    agg = agg.reset_index()

    if group_cols:
        grp = agg.groupby(group_cols, dropna=False, sort=False)
        cum_sum = grp["sum"].cumsum() - agg["sum"]
        cum_cnt = grp["count"].cumsum() - agg["count"]
    else:
        cum_sum = agg["sum"].cumsum() - agg["sum"]
        cum_cnt = agg["count"].cumsum() - agg["count"]

    agg["_prior_mean"] = (cum_sum / cum_cnt.replace(0, np.nan)).to_numpy()

    merge_keys = [*group_cols, "_m"]
    merged = frame[merge_keys].merge(agg[[*merge_keys, "_prior_mean"]], on=merge_keys, how="left")
    return pd.Series(merged["_prior_mean"].to_numpy(), index=values.index)


def _expanding_prior_std(
    values: pd.Series, month: pd.Series, keys: list[pd.Series] | None = None
) -> pd.Series:
    """Population std over months strictly before t, via the sum-of-squares identity."""
    v = pd.to_numeric(values, errors="coerce")
    mean = _expanding_prior_mean(v, month, keys)
    mean_sq = _expanding_prior_mean(v * v, month, keys)
    var = (mean_sq - mean * mean).clip(lower=0.0)
    return np.sqrt(var)


def build(
    panel: pd.DataFrame,
    static_features: pd.DataFrame,
    delinquency_features: pd.DataFrame,
    balance_features: pd.DataFrame,
    dq_flags: pd.Series | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame(index=panel.index)
    month = panel["month_index"]
    status = panel["current_status"].astype("object")

    is_delinq = status.isin(DELINQUENT_STATES).astype("float64")
    prepaid = pd.to_numeric(panel.get("prepayment_flag"), errors="coerce").fillna(0.0)
    dpd = delinquency_features["dpd"]
    bal_ratio = balance_features["bal_ratio"]

    vintage = static_features.get("vintage_year_num").astype("object")
    credit = static_features.get("credit_score_band").astype("object")
    state = static_features.get("state").astype("object")
    servicer = static_features.get("servicer_name").astype("object")

    cohort_keys = [vintage, credit]

    out["peer_vintage_credit_dpd_mean"] = _expanding_prior_mean(dpd, month, cohort_keys)
    out["peer_vintage_credit_delinq_rate"] = _expanding_prior_mean(is_delinq, month, cohort_keys)
    out["peer_state_delinq_rate"] = _expanding_prior_mean(is_delinq, month, [state])
    out["peer_servicer_delinq_rate"] = _expanding_prior_mean(is_delinq, month, [servicer])
    out["peer_servicer_prepay_rate"] = _expanding_prior_mean(prepaid, month, [servicer])

    if dq_flags is not None:
        out["servicer_dq_rate_to_date"] = _expanding_prior_mean(
            pd.to_numeric(dq_flags, errors="coerce"), month, [servicer]
        )
    else:
        out["servicer_dq_rate_to_date"] = np.nan

    dpd_std = _expanding_prior_std(dpd, month, cohort_keys)
    out["dpd_z_vs_peer"] = (
        (dpd - out["peer_vintage_credit_dpd_mean"]) / dpd_std.replace(0.0, np.nan)
    ).clip(-25, 25)

    bal_mean = _expanding_prior_mean(bal_ratio, month, cohort_keys)
    bal_std = _expanding_prior_std(bal_ratio, month, cohort_keys)
    out["bal_ratio_z_vs_peer"] = ((bal_ratio - bal_mean) / bal_std.replace(0.0, np.nan)).clip(-25, 25)

    out["portfolio_delinq_rate_to_date"] = _expanding_prior_mean(is_delinq, month)
    out["portfolio_prepay_rate_to_date"] = _expanding_prior_mean(prepaid, month)
    return out
