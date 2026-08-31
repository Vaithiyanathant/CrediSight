"""F9 — Interactions (12), selected by domain reasoning rather than brute force.

Every member here answers a question a credit analyst would actually ask: does a
delinquency mean something different for a weak-credit borrower than a strong
one; does refinance incentive matter more early or late in the seasoning curve;
is an amortisation anomaly benign when the loan has been modified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lpie.features.registry import FeatureSpec, spec

FAMILY = "interactions"

SPECS: list[FeatureSpec] = [
    spec("credit_x_ltv", FAMILY, "Credit band ordinal x LTV band ordinal",
         ["credit_score_band", "ltv_band"]),
    spec("dpd_x_credit", FAMILY, "Days past due x credit band ordinal — same DPD, different meaning",
         ["days_past_due", "credit_score_band"]),
    spec("dpd_x_ltv", FAMILY, "Days past due x LTV band ordinal — negative equity amplifies default",
         ["days_past_due", "ltv_band"]),
    spec("refi_incentive_x_seasoning", FAMILY, "Refi incentive x seasoning fraction",
         ["interest_rate", "loan_age_months", "remaining_term_months"]),
    spec("amort_residual_x_modification", FAMILY,
         "Amortisation residual x modification flag — a modified loan may legitimately neg-am",
         ["current_balance", "modification_flag"]),
    spec("dq_score_x_status_severity", FAMILY, "Data quality x delinquency severity",
         ["*rules*", "current_status"]),
    spec("servicer_gap_x_dpd", FAMILY, "Servicer balance gap x days past due — two-source stress",
         ["reported_balance", "days_past_due"]),
    spec("balance_ratio_x_seasoning", FAMILY, "Balance ratio x seasoning — curtailment detector",
         ["current_balance", "loan_age_months"]),
    spec("burnout_x_incentive", FAMILY, "Burnout x current incentive — the S-curve shape term",
         ["interest_rate"]),
    spec("dpd_max_12m_x_cure_count", FAMILY, "Peak DPD x number of cures — chronic vs one-off",
         ["days_past_due", "current_status"]),
    spec("doc_severity_x_source_manual", FAMILY,
         "Document gap x manual data entry — the mined association rule, as a feature",
         ["document_status", "source_system"]),
    spec("age_x_credit", FAMILY, "Loan age x credit band ordinal", ["loan_age_months", "credit_score_band"]),
]


def build(
    static_features: pd.DataFrame,
    balance_features: pd.DataFrame,
    delinquency_features: pd.DataFrame,
    prepay_features: pd.DataFrame,
    dq_features: pd.DataFrame,
    servicer_features: pd.DataFrame,
) -> pd.DataFrame:
    out = pd.DataFrame(index=static_features.index)

    credit = pd.to_numeric(static_features["credit_score_band_ord"], errors="coerce")
    ltv = pd.to_numeric(static_features["ltv_band_ord"], errors="coerce")
    dpd = pd.to_numeric(delinquency_features["dpd"], errors="coerce")

    out["credit_x_ltv"] = credit * ltv
    out["dpd_x_credit"] = dpd * credit
    out["dpd_x_ltv"] = dpd * ltv
    out["refi_incentive_x_seasoning"] = (
        pd.to_numeric(prepay_features["refi_incentive"], errors="coerce")
        * pd.to_numeric(prepay_features["seasoning_frac"], errors="coerce")
    )
    out["amort_residual_x_modification"] = (
        pd.to_numeric(balance_features["amort_residual"], errors="coerce")
        * pd.to_numeric(balance_features["modification_flag"], errors="coerce")
    )
    out["dq_score_x_status_severity"] = (
        pd.to_numeric(dq_features["dq_score"], errors="coerce")
        * pd.to_numeric(delinquency_features["status_severity"], errors="coerce")
    )
    out["servicer_gap_x_dpd"] = (
        pd.to_numeric(servicer_features["servicer_bal_gap_pct"], errors="coerce").abs() * dpd
    )
    out["balance_ratio_x_seasoning"] = (
        pd.to_numeric(balance_features["bal_ratio"], errors="coerce")
        * pd.to_numeric(prepay_features["seasoning_frac"], errors="coerce")
    )
    out["burnout_x_incentive"] = (
        pd.to_numeric(prepay_features["burnout"], errors="coerce")
        * pd.to_numeric(prepay_features["refi_incentive"], errors="coerce")
    )
    out["dpd_max_12m_x_cure_count"] = (
        pd.to_numeric(delinquency_features["dpd_max_12m"], errors="coerce")
        * pd.to_numeric(delinquency_features["cure_count"], errors="coerce")
    )
    doc_sev = pd.to_numeric(dq_features["document_status_severity"], errors="coerce")
    is_manual = (dq_features["source_system"].astype("object") == "Manual-Entry").astype("float64")
    out["doc_severity_x_source_manual"] = doc_sev * is_manual
    out["age_x_credit"] = pd.to_numeric(prepay_features["loan_age_months"], errors="coerce") * credit

    return out.replace([np.inf, -np.inf], np.nan)
