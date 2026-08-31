"""Global explainability: TreeSHAP, family attribution, permutation importance.

Feature-family attribution is the part that lands with a business audience:
aggregating 145 SHAP bars into nine families answers "how much did delinquency
path contribute versus static credit attributes?" in one chart instead of
requiring someone to read a beeswarm.

SHAP and permutation importance are shown side by side deliberately. They
measure different things — SHAP measures contribution to *the model*,
permutation measures contribution to *generalisation* — and where they disagree
that is a finding, not an inconsistency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lpie.core.config import Settings, get_settings
from lpie.core.logging import get_logger

log = get_logger(__name__)


def tree_shap_values(
    artifact: Any, X: pd.DataFrame, *, max_rows: int = 20_000, seed: int = 0
) -> tuple[np.ndarray, pd.DataFrame]:
    """Exact TreeSHAP for a GBDT head. Returns (values, the rows explained)."""
    import shap

    from lpie.models.heads import _pin_categories

    sample = X if len(X) <= max_rows else X.sample(max_rows, random_state=seed)
    encode_numeric = artifact.algorithm == "xgboost"
    prepared = _pin_categories(
        sample, artifact.feature_names, artifact.categorical_features,
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
    return values, prepared


def global_importance(
    shap_values: np.ndarray, feature_names: list[str], *, top_k: int = 30
) -> list[dict[str, Any]]:
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    order = np.argsort(-mean_abs)[:top_k]
    return [
        {
            "feature": feature_names[i],
            "mean_abs_shap": round(float(mean_abs[i]), 8),
            "mean_signed_shap": round(float(mean_signed[i]), 8),
            "direction": "increases risk" if mean_signed[i] > 0 else "decreases risk",
            "rank": int(rank + 1),
        }
        for rank, i in enumerate(order)
    ]


def family_attribution(
    shap_values: np.ndarray, feature_names: list[str], registry
) -> list[dict[str, Any]]:
    """Aggregate |SHAP| by the nine feature families. One groupby, high signal."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for i, name in enumerate(feature_names):
        family = registry.family_of(name) or "unregistered"
        totals[family] = totals.get(family, 0.0) + float(mean_abs[i])
        counts[family] = counts.get(family, 0) + 1

    grand = sum(totals.values()) or 1.0
    rows = [
        {
            "family": family,
            "total_mean_abs_shap": round(value, 8),
            "share": round(value / grand, 6),
            "n_features": counts[family],
            "mean_per_feature": round(value / counts[family], 8),
        }
        for family, value in totals.items()
    ]
    rows.sort(key=lambda r: -r["share"])
    return rows


def permutation_importance_out_of_time(
    artifact: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_repeats: int = 3,
    max_rows: int = 20_000,
    top_k: int = 30,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Permutation importance on an out-of-time window.

    Deliberately *not* in-fold gain, which is biased toward high-cardinality
    features and measures how much the model used a feature rather than whether
    the feature helps on unseen time.
    """
    from sklearn.metrics import average_precision_score

    from lpie.models.heads import predict_head

    s = settings or get_settings()
    rng = np.random.default_rng(s.seed)
    sample = X if len(X) <= max_rows else X.sample(max_rows, random_state=s.seed)
    y_sample = pd.to_numeric(y.loc[sample.index], errors="coerce")
    mask = y_sample.notna()
    sample, y_sample = sample[mask.to_numpy()], y_sample[mask]
    if len(sample) < 200 or y_sample.nunique() < 2:
        return []

    baseline = float(average_precision_score(y_sample, predict_head(artifact, sample)))
    rows = []
    for name in artifact.feature_names:
        if name not in sample.columns:
            continue
        drops = []
        for _ in range(n_repeats):
            shuffled = sample.copy()
            shuffled[name] = shuffled[name].to_numpy()[rng.permutation(len(shuffled))]
            drops.append(baseline - float(average_precision_score(y_sample, predict_head(artifact, shuffled))))
        rows.append(
            {
                "feature": name,
                "importance": round(float(np.mean(drops)), 8),
                "std": round(float(np.std(drops)), 8),
            }
        )
    rows.sort(key=lambda r: -r["importance"])
    return rows[:top_k]


def monotonicity_audit(
    artifact: Any,
    X: pd.DataFrame,
    *,
    expectations: dict[str, str] | None = None,
    n_grid: int = 6,
    max_rows: int = 5000,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Check the model learned domain-sane directions.

    Worse credit band should mean higher default risk. Where it does not, we
    either accept it with evidence or apply a monotone constraint and report the
    metric cost — an accuracy-versus-trust trade-off made visibly rather than
    assumed away.
    """
    from lpie.models.heads import predict_head

    s = settings or get_settings()
    expectations = expectations or {
        "credit_score_band_ord": "decreasing",
        "ltv_band_ord": "increasing",
        "dti_band_ord": "increasing",
        "dpd": "increasing",
        "dpd_max_12m": "increasing",
        "status_severity": "increasing",
    }
    sample = X if len(X) <= max_rows else X.sample(max_rows, random_state=s.seed)

    rows = []
    for feature, expected in expectations.items():
        if feature not in sample.columns:
            continue
        values = pd.to_numeric(sample[feature], errors="coerce").dropna()
        if values.nunique() < 2:
            continue
        grid = np.linspace(values.quantile(0.05), values.quantile(0.95), n_grid)
        means = []
        for value in grid:
            probe = sample.copy()
            probe[feature] = value
            means.append(float(predict_head(artifact, probe).mean()))
        diffs = np.diff(means)
        observed = (
            "increasing" if (diffs > 0).mean() >= 0.8
            else "decreasing" if (diffs < 0).mean() >= 0.8
            else "non_monotone"
        )
        rows.append(
            {
                "feature": feature,
                "expected_direction": expected,
                "observed_direction": observed,
                "consistent": observed == expected,
                "grid": [round(float(g), 4) for g in grid],
                "mean_prediction": [round(m, 6) for m in means],
                "range": round(float(max(means) - min(means)), 6),
            }
        )
    return rows
