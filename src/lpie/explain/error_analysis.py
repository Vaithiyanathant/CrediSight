"""Error forensics: FP/FN profiling, automatic slice discovery, segment metrics.

Error-slice discovery fits a shallow decision tree on `is_error ~ features` so
the worst-performing cohorts surface *automatically* rather than being found by
eyeballing a segment table. Sample sizes are always reported alongside segment
metrics, so small-cell noise is visible rather than hidden.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lpie.core.config import Settings, get_settings
from lpie.core.logging import get_logger
from lpie.evaluation.metrics import binary_metrics

log = get_logger(__name__)

DEFAULT_SEGMENTS = ("vintage_year_num", "credit_score_band", "state", "servicer_name", "dq_grade")


def confusion_profile(
    X: pd.DataFrame,
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    *,
    profile_columns: list[str] | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    """Profile the false positives and false negatives at the operating threshold.

    Terminal rows (Prepaid/Closed) are excluded before computing the confusion
    matrix. Terminal rows always have forward-target = 0 but receive non-zero
    model scores from their pre-absorption history. Counting them as false
    positives massively inflates the FP count and misrepresents the model's
    actual failure modes on the active population.
    """
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")

    # Gate: exclude absorbing/terminal rows before computing confusion matrix
    TERMINAL = {"Prepaid", "Closed"}
    if "current_status" in X.columns:
        active_mask = ~X["current_status"].astype(str).isin(TERMINAL)
        X = X[active_mask].reset_index(drop=True)
        y = y[active_mask.values]
        p = p[active_mask.values]

    pred = (p >= threshold).astype("float64")

    groups = {
        "true_positive": (y == 1) & (pred == 1),
        "false_positive": (y == 0) & (pred == 1),
        "true_negative": (y == 0) & (pred == 0),
        "false_negative": (y == 1) & (pred == 0),
    }
    columns = profile_columns or [
        c for c in ("current_status", "credit_score_band", "state", "servicer_name",
                    "dq_grade", "document_status_severity", "dpd", "bal_ratio")
        if c in X.columns
    ]

    out: dict[str, Any] = {"threshold": round(float(threshold), 6), "n": int(len(y))}
    for name, mask in groups.items():
        subset = X[mask]
        entry: dict[str, Any] = {
            "n": int(mask.sum()),
            "share": round(float(mask.mean()), 6),
            "mean_probability": round(float(p[mask].mean()), 6) if mask.any() else None,
        }
        profile = {}
        for column in columns[:top_k]:
            if not mask.any():
                continue
            values = subset[column]
            if pd.api.types.is_numeric_dtype(values):
                profile[column] = {
                    "mean": _round(pd.to_numeric(values, errors="coerce").mean()),
                    "median": _round(pd.to_numeric(values, errors="coerce").median()),
                }
            else:
                top = values.astype(str).value_counts(normalize=True).head(3)
                profile[column] = {str(k): round(float(v), 4) for k, v in top.items()}
        entry["profile"] = profile
        out[name] = entry

    fp, fn = groups["false_positive"], groups["false_negative"]
    out["narrative_examples"] = {
        "false_positives": _examples(X, p, fp),
        "false_negatives": _examples(X, p, fn),
    }
    out["cost_note"] = (
        "A false negative is a missed default; a false positive is wasted reviewer "
        "capacity. The operating threshold encodes which the business is willing to trade."
    )
    return out


def _examples(X: pd.DataFrame, p: np.ndarray, mask: np.ndarray, n: int = 3) -> list[dict[str, Any]]:
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    order = idx[np.argsort(-p[idx])][:n]
    columns = [c for c in ("loan_id", "month_index", "current_status", "dpd", "bal_ratio", "dq_score")
               if c in X.columns]
    return [
        {**{c: _display(X.iloc[i].get(c)) for c in columns}, "probability": round(float(p[i]), 6)}
        for i in order
    ]


def error_slices(
    X: pd.DataFrame,
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    *,
    max_depth: int = 3,
    min_samples_leaf: int = 200,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Fit a shallow tree on `is_error ~ features` to surface the worst cohorts."""
    from sklearn.tree import DecisionTreeClassifier, export_text

    s = settings or get_settings()
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    is_error = ((p >= threshold).astype("float64") != y).astype("int64")
    if is_error.sum() < min_samples_leaf or is_error.mean() in (0.0, 1.0):
        return {"available": False, "reason": "too few errors to segment"}

    numeric = [
        c for c in X.columns
        if pd.api.types.is_numeric_dtype(X[c]) and not c.startswith("_") and X[c].notna().any()
    ][:60]
    if not numeric:
        return {"available": False, "reason": "no numeric features available"}

    values = X[numeric].to_numpy(dtype="float64")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    tree = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=s.seed
    )
    tree.fit(values, is_error)

    leaves = tree.apply(values)
    rows = []
    for leaf in np.unique(leaves):
        mask = leaves == leaf
        rows.append(
            {
                "leaf": int(leaf),
                "n": int(mask.sum()),
                "error_rate": round(float(is_error[mask].mean()), 6),
                "lift_over_overall": round(
                    float(is_error[mask].mean() / max(is_error.mean(), 1e-9)), 4
                ),
                "mean_probability": round(float(p[mask].mean()), 6),
                "positive_rate": round(float(y[mask].mean()), 6),
            }
        )
    rows.sort(key=lambda r: -r["error_rate"])
    return {
        "available": True,
        "overall_error_rate": round(float(is_error.mean()), 6),
        "worst_slices": rows[:8],
        "tree_rules": export_text(tree, feature_names=numeric, max_depth=max_depth),
        "note": (
            "Slices are discovered by the tree, not chosen by hand, so a cohort nobody "
            "thought to look at can still surface."
        ),
    }


def segment_performance(
    X: pd.DataFrame,
    y: np.ndarray,
    p: np.ndarray,
    *,
    segments: tuple[str, ...] = DEFAULT_SEGMENTS,
    min_rows: int = 100,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Per-segment metrics with sample sizes attached."""
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    out: dict[str, Any] = {}
    for segment in segments:
        if segment not in X.columns:
            continue
        keys = X[segment].astype(str).to_numpy()
        rows = []
        for value in pd.unique(keys):
            mask = keys == value
            if mask.sum() < min_rows:
                continue
            metrics = binary_metrics(y[mask], p[mask], threshold=threshold, label=str(value))
            rows.append(
                {
                    "value": str(value),
                    "n": int(mask.sum()),
                    "base_rate": metrics.get("base_rate"),
                    "roc_auc": metrics.get("roc_auc"),
                    "pr_auc": metrics.get("pr_auc"),
                    "brier": metrics.get("brier"),
                    "ece": metrics.get("ece"),
                }
            )
        rows.sort(key=lambda r: -(r["n"] or 0))
        if rows:
            out[segment] = rows
    return out


def calibration_by_segment(
    X: pd.DataFrame,
    y: np.ndarray,
    p: np.ndarray,
    *,
    row_key: str = "vintage_year_num",
    column_key: str = "credit_score_band",
    min_rows: int = 100,
) -> list[dict[str, Any]]:
    """The vintage x band reliability grid — where calibration is weakest.

    Terminal rows (Prepaid/Closed) are excluded: they carry label=0 by
    construction for all forward-looking heads, but the model assigns them
    non-zero probabilities from their historical delinquency paths. Including
    them inflates the bias metric and mis-attributes the error to the
    calibration layer rather than the absorbing-state gate.
    """
    from lpie.models.calibration import expected_calibration_error

    if row_key not in X.columns or column_key not in X.columns:
        return []
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")

    # Exclude absorbing/terminal states from calibration diagnostics
    TERMINAL = {"Prepaid", "Closed"}
    if "current_status" in X.columns:
        active_mask = ~X["current_status"].astype(str).isin(TERMINAL)
        X = X[active_mask]
        y = y[active_mask.values]
        p = p[active_mask.values]
    keys = (X[row_key].astype(str) + "|" + X[column_key].astype(str)).to_numpy()

    rows = []
    for value in pd.unique(keys):
        mask = keys == value
        if mask.sum() < min_rows:
            continue
        vintage, band = value.split("|", 1)
        rows.append(
            {
                "vintage": vintage,
                "credit_band": band,
                "n": int(mask.sum()),
                "mean_predicted": round(float(p[mask].mean()), 6),
                "observed_rate": round(float(y[mask].mean()), 6),
                "ece": expected_calibration_error(p[mask], y[mask]),
                "bias": round(float(p[mask].mean() - y[mask].mean()), 6),
            }
        )
    rows.sort(key=lambda r: -abs(r["bias"]))
    return rows


def _round(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, 6)


def _display(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return _round(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return None if value is None else str(value)
