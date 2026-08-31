"""Fusion — rank-averaged unsupervised score, max-fused with rule severity.

    anomaly_score = max(
        0.60 + 0.40 * rule_severity   if any ERROR rule fires,
        0.30 + 0.30 * rule_severity   if only WARNING rules fire,
        0.60 * rank_avg_unsupervised
    )

Max-fusion rather than a weighted sum, so a hard ERROR always outranks a soft
statistical outlier. A reviewer who sees 0.95 must be able to trust that it
means "a deterministic rule fired", not "four detectors mildly disagreed".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lpie.core.config import Settings, get_settings


def fuse(
    unsupervised_rank: np.ndarray,
    rule_severity: np.ndarray | pd.Series,
    worst_severity: np.ndarray | pd.Series,
    *,
    settings: Settings | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(anomaly_score in [0,1], tier label)."""
    s = settings or get_settings()
    cfg = s.section("anomaly").get("fusion", {})
    error_base = float(cfg.get("error_base", 0.60))
    error_span = float(cfg.get("error_span", 0.40))
    warning_base = float(cfg.get("warning_base", 0.30))
    warning_span = float(cfg.get("warning_span", 0.30))
    unsup_cap = float(cfg.get("unsupervised_cap", 0.60))

    rank = np.nan_to_num(np.asarray(unsupervised_rank, dtype="float64"), nan=0.0)
    severity = np.nan_to_num(np.asarray(rule_severity, dtype="float64"), nan=0.0)
    worst = np.asarray(pd.Series(worst_severity).astype(str))

    rule_component = np.where(
        worst == "ERROR",
        error_base + error_span * severity,
        np.where(worst == "WARNING", warning_base + warning_span * severity, 0.0),
    )
    unsup_component = unsup_cap * rank
    score = np.maximum(rule_component, unsup_component)

    tier = np.where(
        (rule_component > 0) & (rule_component >= unsup_component),
        np.where(worst == "ERROR", "rule:ERROR", "rule:WARNING"),
        np.where(rule_component > 0, "rule+ml", "unsupervised"),
    )
    return np.clip(score, 0.0, 1.0), tier


def precision_at_k(
    scores: np.ndarray, ground_truth: np.ndarray, k: int
) -> dict[str, Any]:
    """Precision in the top-k by anomaly score, against the rule ground truth.

    The right metric for an unsupervised ranker: a reviewer works a queue from
    the top, so what matters is how many of the first k are real.
    """
    scores = np.asarray(scores, dtype="float64")
    truth = np.asarray(ground_truth, dtype="float64")
    n = min(int(k), len(scores))
    if n == 0:
        return {"k": int(k), "precision": None, "n_evaluated": 0}
    order = np.argsort(-scores, kind="mergesort")[:n]
    return {
        "k": int(k),
        "precision": round(float(truth[order].mean()), 6),
        "n_evaluated": n,
        "n_true_positives": int(truth[order].sum()),
        "base_rate": round(float(truth.mean()), 6),
        "lift": round(float(truth[order].mean() / truth.mean()), 4) if truth.mean() > 0 else None,
    }


def novel_catch_rate(
    scores: np.ndarray, rule_flagged: np.ndarray, *, top_k: int = 500
) -> dict[str, Any]:
    """Share of the top-k that the rules did *not* already flag.

    This is the honest measure of what the unsupervised tier adds: rediscovering
    rule hits is worth nothing; surfacing records no rule can see is the entire
    point of having Tier 3.
    """
    scores = np.asarray(scores, dtype="float64")
    flagged = np.asarray(rule_flagged, dtype="float64")
    n = min(int(top_k), len(scores))
    if n == 0:
        return {"top_k": int(top_k), "novel_rate": None}
    order = np.argsort(-scores, kind="mergesort")[:n]
    novel = float((flagged[order] == 0).mean())
    return {
        "top_k": n,
        "novel_rate": round(novel, 6),
        "n_novel": int((flagged[order] == 0).sum()),
        "note": (
            "Records in the top-k that no deterministic rule flagged. These are what "
            "Tier 3 contributes; the rest is rule rediscovery."
        ),
    }
