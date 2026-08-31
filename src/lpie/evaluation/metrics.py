"""Metric computation, always reported twice.

Every discrimination metric is computed **overall** and **active-conditional**
(terminal rows removed). On this data pack that distinction is not a nicety:
61-66% of test rows are already `Prepaid` or `Closed`, both perfectly absorbing,
and a model that learns only "is this row terminal?" scores AUC 0.886 on
prepayment while knowing nothing about credit risk.

PR-AUC is primary rather than ROC-AUC. At a 1.86% default rate ROC-AUC is
dominated by the enormous negative class and stays flat while precision
collapses; PR-AUC tracks what a reviewer actually experiences. It is also the
leakage-robust choice here: PR-AUC is invariant to removing terminal rows, which
is itself the proof that the ROC-AUC uplift is the trivial terminal-vs-active
split.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from lpie.models.calibration import (
    brier_decomposition,
    expected_calibration_error,
    maximum_calibration_error,
)

TERMINAL_STATES = ("Prepaid", "Closed")
STATE_ORDER = ("Current", "30DPD", "60DPD", "90DPD", "Default", "Prepaid", "Closed")


def _safe(fn, *args, **kwargs) -> float | None:
    try:
        value = float(fn(*args, **kwargs))
        return None if not np.isfinite(value) else round(value, 6)
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def recall_at_precision(
    y: np.ndarray, p: np.ndarray, min_precision: float = 0.30
) -> dict[str, Any]:
    """Maximum recall achievable at or above a precision floor.

    The floor encodes a real servicing-capacity constraint: a queue at 15%
    precision wastes five reviewer-hours for every real find.
    """
    from sklearn.metrics import precision_recall_curve

    if len(np.unique(y)) < 2:
        return {"recall": None, "precision": None, "threshold": None, "achievable": False}
    precision, recall, thresholds = precision_recall_curve(y, p)
    feasible = precision[:-1] >= min_precision
    if not feasible.any():
        best = int(np.argmax(precision[:-1])) if len(thresholds) else 0
        return {
            "recall": round(float(recall[best]), 6) if len(recall) else None,
            "precision": round(float(precision[best]), 6) if len(precision) else None,
            "threshold": round(float(thresholds[best]), 6) if len(thresholds) else None,
            "achievable": False,
            "min_precision": min_precision,
            "note": f"precision >= {min_precision} is not achievable on this slice",
        }
    idx = np.flatnonzero(feasible)
    best = idx[int(np.argmax(recall[:-1][idx]))]
    return {
        "recall": round(float(recall[best]), 6),
        "precision": round(float(precision[best]), 6),
        "threshold": round(float(thresholds[best]), 6),
        "achievable": True,
        "min_precision": min_precision,
    }


def capture_rate_at_k(y: np.ndarray, p: np.ndarray, k: float = 0.10) -> float | None:
    """Share of all positives captured in the top-k fraction by score."""
    n = len(y)
    if n == 0 or y.sum() == 0:
        return None
    take = max(int(round(n * k)), 1)
    order = np.argsort(-p, kind="mergesort")[:take]
    return round(float(y[order].sum() / y.sum()), 6)


def lift_at_decile(y: np.ndarray, p: np.ndarray, decile: int = 1) -> float | None:
    n = len(y)
    base = float(y.mean()) if n else 0.0
    if n == 0 or base == 0:
        return None
    take = max(int(round(n * 0.1 * decile)), 1)
    order = np.argsort(-p, kind="mergesort")[:take]
    return round(float(y[order].mean() / base), 6)


def binary_metrics(
    y: np.ndarray | pd.Series,
    p: np.ndarray | pd.Series,
    *,
    threshold: float | None = None,
    min_precision: float = 0.30,
    label: str = "",
) -> dict[str, Any]:
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]

    if len(y) == 0:
        return {"label": label, "n": 0, "note": "no scored rows"}

    n_pos = int(y.sum())
    out: dict[str, Any] = {
        "label": label,
        "n": int(len(y)),
        "n_positive": n_pos,
        "base_rate": round(float(y.mean()), 6),
        "mean_predicted": round(float(p.mean()), 6),
    }
    if len(np.unique(y)) < 2:
        out["note"] = "single-class slice; discrimination metrics are undefined"
        out["brier"] = _safe(brier_score_loss, y, p)
        return out

    out["roc_auc"] = _safe(roc_auc_score, y, p)
    out["pr_auc"] = _safe(average_precision_score, y, p)
    out["brier"] = _safe(brier_score_loss, y, p)
    out["log_loss"] = _safe(log_loss, y, np.clip(p, 1e-9, 1 - 1e-9))
    out["ece"] = expected_calibration_error(p, y)
    out["mce"] = maximum_calibration_error(p, y)
    out["brier_decomposition"] = brier_decomposition(p, y)
    out["recall_at_precision"] = recall_at_precision(y, p, min_precision)
    out["capture_rate_top_10pct"] = capture_rate_at_k(y, p, 0.10)
    out["lift_decile_1"] = lift_at_decile(y, p, 1)

    if threshold is not None:
        pred = (p >= threshold).astype("float64")
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0
        )
        out["operating_point"] = {
            "threshold": round(float(threshold), 6),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
            "n_flagged": int(pred.sum()),
        }
    return out


def dual_binary_metrics(
    y: np.ndarray | pd.Series,
    p: np.ndarray | pd.Series,
    current_status: pd.Series,
    *,
    threshold: float | None = None,
    min_precision: float = 0.30,
) -> dict[str, Any]:
    """Overall and active-conditional metrics, side by side, always.

    The `pr_auc_invariance` field is the diagnostic that proves the point: if
    PR-AUC is (near-)identical with and without terminal rows, the model ranks
    every terminal row below every active row, and the entire ROC-AUC uplift is
    the trivial "is it terminal?" signal.
    """
    status = pd.Series(current_status).astype("object").to_numpy()
    active = ~np.isin(status, TERMINAL_STATES)

    overall = binary_metrics(y, p, threshold=threshold, min_precision=min_precision, label="overall")
    conditional = binary_metrics(
        np.asarray(y)[active], np.asarray(p)[active],
        threshold=threshold, min_precision=min_precision, label="active_conditional",
    )

    pr_o, pr_a = overall.get("pr_auc"), conditional.get("pr_auc")
    auc_o, auc_a = overall.get("roc_auc"), conditional.get("roc_auc")
    return {
        "overall": overall,
        "active_conditional": conditional,
        "n_terminal_rows": int((~active).sum()),
        "terminal_share": round(float((~active).mean()), 6) if len(active) else None,
        "roc_auc_inflation": (
            round(auc_o - auc_a, 6) if auc_o is not None and auc_a is not None else None
        ),
        "pr_auc_invariance": (
            {
                "overall": pr_o,
                "active_conditional": pr_a,
                "delta": round(abs(pr_o - pr_a), 6),
                "interpretation": (
                    "PR-AUC is near-identical with and without terminal rows, so the model "
                    "ranks terminal rows below active ones and the ROC-AUC uplift is the "
                    "trivial terminal-vs-active split."
                    if abs(pr_o - pr_a) < 0.02
                    else "PR-AUC changes materially when terminal rows are removed."
                ),
            }
            if pr_o is not None and pr_a is not None
            else None
        ),
        "headline": "active_conditional",
        "headline_note": (
            "The active-conditional figure is the headline. The overall figure is reported "
            "alongside and is absorbing-state inflated."
        ),
    }


def multiclass_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    proba: np.ndarray | None = None,
    *,
    labels: tuple[str, ...] = STATE_ORDER,
    label: str = "",
) -> dict[str, Any]:
    """Next-state metrics including quadratic-weighted kappa.

    The states are ordered by severity, so confusing `Current` with `Default` is
    far worse than confusing `30DPD` with `60DPD`. Accuracy and macro-F1 are both
    blind to that; quadratic-weighted kappa is not.
    """
    y_true = pd.Series(y_true).astype("object")
    y_pred = pd.Series(y_pred).astype("object")
    mask = y_true.notna() & y_pred.notna()
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.empty:
        return {"label": label, "n": 0, "note": "no labelled rows"}

    present = [c for c in labels if c in set(y_true) | set(y_pred)]
    out: dict[str, Any] = {
        "label": label,
        "n": int(len(y_true)),
        "accuracy": _safe(accuracy_score, y_true, y_pred),
        "balanced_accuracy": _safe(balanced_accuracy_score, y_true, y_pred),
        "macro_f1": _safe(f1_score, y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": _safe(f1_score, y_true, y_pred, average="weighted", zero_division=0),
        "quadratic_weighted_kappa": _safe(
            cohen_kappa_score, y_true, y_pred, labels=present, weights="quadratic"
        ),
        "labels": present,
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=present, zero_division=0
    )
    out["per_class"] = [
        {
            "state": present[i],
            "precision": round(float(precision[i]), 6),
            "recall": round(float(recall[i]), 6),
            "f1": round(float(f1[i]), 6),
            "support": int(support[i]),
        }
        for i in range(len(present))
    ]
    cm = confusion_matrix(y_true, y_pred, labels=present)
    out["confusion_matrix"] = {
        "labels": present,
        "matrix": [[int(v) for v in row] for row in cm],
    }

    if proba is not None and proba.shape[0] == len(mask):
        p = np.asarray(proba, dtype="float64")[mask.to_numpy()]
        if p.shape[1] == len(labels):
            top2 = np.argsort(-p, axis=1)[:, :2]
            truth_idx = y_true.map({s: i for i, s in enumerate(labels)}).to_numpy()
            out["top_2_accuracy"] = round(
                float(np.mean([truth_idx[i] in top2[i] for i in range(len(truth_idx))])), 6
            )
    return out


def dual_multiclass_metrics(
    y_true: pd.Series, y_pred: pd.Series, current_status: pd.Series, proba: np.ndarray | None = None
) -> dict[str, Any]:
    status = pd.Series(current_status).astype("object")
    active = ~status.isin(TERMINAL_STATES)
    overall = multiclass_metrics(y_true, y_pred, proba, label="overall")
    conditional = multiclass_metrics(
        y_true[active.to_numpy()], y_pred[active.to_numpy()],
        proba[active.to_numpy()] if proba is not None else None,
        label="active_conditional",
    )
    return {
        "overall": overall,
        "active_conditional": conditional,
        "n_terminal_rows": int((~active).sum()),
        "terminal_share": round(float((~active).mean()), 6) if len(active) else None,
        "headline": "active_conditional",
        "headline_note": (
            "A constant predictor that says next_state = current_state scores about 0.90 "
            "overall accuracy on this pack, because 61-66% of rows are absorbing. "
            "The active-conditional figure is the one that measures modelling."
        ),
    }


def fold_summary(fold_metrics: list[dict[str, Any]], key_path: tuple[str, ...]) -> dict[str, Any]:
    """Mean +/- std across folds. The std is itself a reported stability metric."""
    values = []
    for m in fold_metrics:
        node: Any = m
        for k in key_path:
            node = (node or {}).get(k) if isinstance(node, dict) else None
        if isinstance(node, (int, float)) and np.isfinite(node):
            values.append(float(node))
    if not values:
        return {"mean": None, "std": None, "n_folds": 0, "values": []}
    return {
        "mean": round(float(np.mean(values)), 6),
        "std": round(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 6),
        "min": round(float(np.min(values)), 6),
        "max": round(float(np.max(values)), 6),
        "n_folds": len(values),
        "values": [round(v, 6) for v in values],
    }


def bootstrap_ci(
    y: np.ndarray, p: np.ndarray, metric: str = "pr_auc", *, n_boot: int = 200, seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Bootstrap CI on a headline metric — answers "is this improvement real?"."""
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    if len(y) < 50 or len(np.unique(y)) < 2:
        return {"metric": metric, "point": None, "ci": None, "n_boot": 0}

    fn = average_precision_score if metric == "pr_auc" else roc_auc_score
    rng = np.random.default_rng(seed)
    point = _safe(fn, y, p)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        value = _safe(fn, y[idx], p[idx])
        if value is not None:
            samples.append(value)
    if not samples:
        return {"metric": metric, "point": point, "ci": None, "n_boot": 0}
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return {
        "metric": metric,
        "point": point,
        "ci": [round(float(lo), 6), round(float(hi), 6)],
        "n_boot": len(samples),
        "confidence": round(1 - alpha, 4),
    }
