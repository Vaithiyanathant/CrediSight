"""F1 — Static / origination features (14).

Ordinal encoding for the banded fields is deliberate: `<620 < 620-659 < ... <
780+` carries real order that one-hot would discard and target-encoding would
leak. With five servicers and fifty states there is nothing target encoding
could buy, and out-of-fold target encoding over a time panel is a classic
leakage vector — so it is banned by the contract, not merely unused.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lpie.features.registry import FeatureSpec, spec

FAMILY = "static"

CREDIT_BANDS = ["<620", "620-659", "660-699", "700-739", "740-779", "780+"]
LTV_BANDS = ["<=60", "61-70", "71-80", "81-90", "91-95", "96-100", ">100"]
DTI_BANDS = ["<=28", "29-36", "37-43", "44-50", ">50"]

CREDIT_ORDINAL = {b: i for i, b in enumerate(CREDIT_BANDS)}
LTV_ORDINAL = {b: i for i, b in enumerate(LTV_BANDS)}
DTI_ORDINAL = {b: i for i, b in enumerate(DTI_BANDS)}

CATEGORICAL_SOURCES = (
    "state", "loan_purpose", "occupancy_type", "property_type", "servicer_name",
)

SPECS: list[FeatureSpec] = [
    spec("original_balance", FAMILY, "Original unpaid principal balance (USD)", ["original_balance"]),
    spec("log_original_balance", FAMILY, "log1p of original balance — tames the right tail", ["original_balance"]),
    spec("interest_rate", FAMILY, "Annual note rate (%)", ["interest_rate"]),
    spec("loan_term_months", FAMILY, "Full contractual term in months", ["loan_term_months"]),
    spec("credit_score_band_ord", FAMILY, "Credit band as an ordinal 0..5 (order is real)",
         ["credit_score_band"], ordinal=True),
    spec("ltv_band_ord", FAMILY, "LTV band as an ordinal 0..6", ["ltv_band"], ordinal=True),
    spec("dti_band_ord", FAMILY, "DTI band as an ordinal 0..4", ["dti_band"], ordinal=True),
    spec("credit_score_band", FAMILY, "Credit band (native categorical)", ["credit_score_band"],
         dtype="category", categorical=True),
    spec("state", FAMILY, "Property state (native categorical)", ["state"], dtype="category", categorical=True),
    spec("loan_purpose", FAMILY, "Loan purpose", ["loan_purpose"], dtype="category", categorical=True),
    spec("occupancy_type", FAMILY, "Occupancy type", ["occupancy_type"], dtype="category", categorical=True),
    spec("property_type", FAMILY, "Property type", ["property_type"], dtype="category", categorical=True),
    spec("servicer_name", FAMILY, "Servicer (native categorical)", ["servicer_name"],
         dtype="category", categorical=True),
    spec("vintage_year_num", FAMILY, "Origination year as an integer", ["vintage_year", "origination_month"]),
    spec("origination_month_num", FAMILY, "Origination calendar month 1-12 (seasonality of underwriting)",
         ["origination_month"]),
    spec("orig_rate_vs_vintage_median", FAMILY,
         "Note rate minus the median rate of the same vintage year — spread at origination",
         ["interest_rate", "vintage_year"]),
]


def build(panel: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    """Static features. Backfilled from `loan_static_attributes.csv` where the
    monthly panel is null — the static file is a legitimate second source and
    using it is exactly the cross-file reconciliation the problem rewards."""
    out = pd.DataFrame(index=panel.index)
    s = static.set_index("loan_id")
    loan = panel["loan_id"]

    def backfilled(col: str) -> pd.Series:
        monthly = panel[col] if col in panel.columns else pd.Series(np.nan, index=panel.index)
        if col not in s.columns:
            return monthly
        static_values = loan.map(s[col])
        static_values.index = panel.index
        return monthly.where(monthly.notna(), static_values)

    orig_bal = pd.to_numeric(backfilled("original_balance"), errors="coerce")
    out["original_balance"] = orig_bal
    out["log_original_balance"] = np.log1p(orig_bal.clip(lower=0))
    out["interest_rate"] = pd.to_numeric(backfilled("interest_rate"), errors="coerce")

    term = loan.map(s["loan_term_months"]) if "loan_term_months" in s.columns else np.nan
    out["loan_term_months"] = pd.to_numeric(pd.Series(term, index=panel.index), errors="coerce")

    credit = backfilled("credit_score_band")
    ltv = backfilled("ltv_band")
    dti = backfilled("dti_band")
    out["credit_score_band_ord"] = credit.map(CREDIT_ORDINAL).astype("float64")
    out["ltv_band_ord"] = ltv.map(LTV_ORDINAL).astype("float64")
    out["dti_band_ord"] = dti.map(DTI_ORDINAL).astype("float64")
    out["credit_score_band"] = credit.astype("object")

    for col in CATEGORICAL_SOURCES:
        out[col] = backfilled(col).astype("object")

    orig_month = backfilled("origination_month").astype("string")
    year = pd.to_numeric(orig_month.str.slice(0, 4), errors="coerce")
    if "vintage_year" in s.columns:
        vintage = pd.to_numeric(pd.Series(loan.map(s["vintage_year"]), index=panel.index), errors="coerce")
        year = year.where(year.notna(), vintage)
    out["vintage_year_num"] = year
    out["origination_month_num"] = pd.to_numeric(orig_month.str.slice(5, 7), errors="coerce")

    # Vintage median rate is a *static* aggregate over origination-time facts.
    # It uses no monthly outcome and therefore carries no temporal leakage.
    vintage_median = out.groupby(out["vintage_year_num"])["interest_rate"].transform("median")
    out["orig_rate_vs_vintage_median"] = out["interest_rate"] - vintage_median

    return out
