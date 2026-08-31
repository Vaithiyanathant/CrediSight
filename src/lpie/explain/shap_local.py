"""Local explanation for one loan-month.

Every number in the narrative comes from the SHAP computation; the prose comes
from a deterministic template. The LLM is never in this path. That separation is
the governing principle of the whole system stated at the smallest scale: the ML
computes numbers, language layers describe them.

The narrative also distinguishes association from prediction from decision. A
SHAP value says "this feature moved the model's score", not "this feature caused
the outcome", and the wording enforces that.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lpie.core.logging import get_logger

log = get_logger(__name__)


def local_shap(
    artifact: Any, row: pd.DataFrame, *, top_k: int = 8
) -> dict[str, Any]:
    """SHAP waterfall for a single row."""
    import shap

    from lpie.models.heads import _pin_categories

    encode_numeric = artifact.algorithm == "xgboost"
    prepared = _pin_categories(
        row, artifact.feature_names, artifact.categorical_features,
        artifact.category_levels, encode_numeric=encode_numeric,
    )
    if artifact.algorithm == "catboost":
        for c in artifact.categorical_features:
            if c in prepared.columns:
                prepared[c] = prepared[c].astype("object").where(prepared[c].notna(), "Unknown").astype(str)

    explainer = shap.TreeExplainer(artifact.model)
    values = explainer.shap_values(prepared)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values, dtype="float64")
    if values.ndim == 3:
        values = values[:, :, -1]
    values = values[0]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.ravel(base_value)[-1])
    base_value = float(base_value)

    order = np.argsort(-np.abs(values))[:top_k]
    contributions = [
        {
            "feature": artifact.feature_names[i],
            "value": _display(row.iloc[0].get(artifact.feature_names[i])),
            "shap": round(float(values[i]), 8),
            "direction": "increases risk" if values[i] > 0 else "decreases risk",
            "abs_shap": round(abs(float(values[i])), 8),
        }
        for i in order
    ]
    from scipy.special import expit as _sigmoid
    output_logit = base_value + float(values.sum())
    return {
        "base_value": round(base_value, 8),
        "sum_shap": round(float(values.sum()), 8),
        "output_logit": round(output_logit, 8),
        "predicted_probability": round(float(_sigmoid(output_logit)), 6),
        "top_contributions": contributions,
        "contributions": contributions,  # alias for backwards compat
        "n_features": len(artifact.feature_names),
    }


def _display(value: Any) -> Any:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return str(value)


def narrative(
    contributions: list[dict[str, Any]],
    probability: float,
    *,
    head: str,
    base_rate: float | None = None,
    max_drivers: int = 3,
) -> str:
    """Deterministic prose from SHAP values. No LLM, no invented numbers."""
    if not contributions:
        return (
            f"No feature-level attribution is available for {head}; the model produced "
            f"a probability of {probability:.3f} but its drivers could not be computed."
        )

    increasing = [c for c in contributions if c["shap"] > 0][:max_drivers]
    decreasing = [c for c in contributions if c["shap"] < 0][:max_drivers]

    parts = [
        f"The calibrated probability for {head.replace('_', ' ')} is {probability:.3f}"
    ]
    if base_rate is not None:
        ratio = probability / base_rate if base_rate > 1e-9 else None
        parts.append(
            f", against a portfolio base rate of {base_rate:.3f}"
            + (f" ({ratio:.1f}x)" if ratio is not None else "")
        )
    parts.append(". ")

    if increasing:
        drivers = ", ".join(f"{c['feature']} = {c['value']}" for c in increasing)
        parts.append(
            f"The score is raised most by {drivers}. "
        )
    if decreasing:
        drivers = ", ".join(f"{c['feature']} = {c['value']}" for c in decreasing)
        parts.append(f"It is reduced by {drivers}. ")

    parts.append(
        "These are associations the model relies on, not established causes, and the "
        "output is a recommendation for a human reviewer rather than a decision."
    )
    return "".join(parts)


def peer_comparison(
    row: pd.Series, cohort: pd.DataFrame, features: list[str], *, top_k: int = 8
) -> list[dict[str, Any]]:
    """This loan's drivers against its (vintage, credit band) cohort average."""
    out = []
    for feature in features[:top_k]:
        if feature not in cohort.columns:
            continue
        values = pd.to_numeric(cohort[feature], errors="coerce")
        if values.notna().sum() < 5:
            continue
        own = pd.to_numeric(pd.Series([row.get(feature)]), errors="coerce").iloc[0]
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        out.append(
            {
                "feature": feature,
                "this_loan": _display(own),
                "cohort_mean": round(mean, 6),
                "cohort_p50": round(float(values.median()), 6),
                "z_score": (
                    round(float((own - mean) / std), 4)
                    if std > 1e-9 and own is not None and np.isfinite(own)
                    else None
                ),
                "n_cohort": int(values.notna().sum()),
            }
        )
    return out


def history_strip(panel: pd.DataFrame, loan_id: str, month_index: int, months: int = 12) -> dict[str, Any]:
    """Twelve-month DPD and balance trace — how the model's view of this loan evolved."""
    history = panel[
        (panel["loan_id"] == loan_id)
        & (pd.to_numeric(panel["month_index"], errors="coerce") <= month_index)
    ].sort_values("month_index").tail(months)
    return {
        "month_index": [int(m) for m in history["month_index"]],
        "current_status": [str(v) for v in history.get("current_status", [])],
        "days_past_due": [_display(v) for v in history.get("days_past_due", [])],
        "current_balance": [_display(v) for v in history.get("current_balance", [])],
    }
