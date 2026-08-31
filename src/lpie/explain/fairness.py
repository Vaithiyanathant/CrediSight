"""Fairness diagnostics, and an honest statement of their limits.

This data pack contains **no protected attributes** — it is HMDA-derived but
demographics were excluded. So a demographic fairness assessment is not
possible, and claiming one would be worse than declining to.

What we can do is measure disparate model *behaviour* across proxy-adjacent
dimensions — `state`, `credit_score_band`, `servicer_name` — as group-wise TPR /
FPR parity, calibration parity, and selection-rate parity.

These are reported as **diagnostics, not optimisation targets**. Optimising a
fairness metric against a proxy, without the real attribute, can increase
real-world harm: it moves decision boundaries for a group defined by something
other than the protected class, and the two do not coincide. Saying so is part
of the deliverable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lpie.core.config import Settings, get_settings
from lpie.models.calibration import expected_calibration_error

LIMITATION_STATEMENT = (
    "This data pack contains no protected attributes, so demographic fairness cannot be "
    "assessed. The dimensions analysed here (state, credit band, servicer) are "
    "proxy-adjacent operational cuts, not protected classes. They are reported as "
    "diagnostics only and are deliberately not optimised against: tuning a model to "
    "equalise a metric across a proxy, without the real attribute, can increase "
    "real-world disparate impact rather than reduce it. A production deployment would "
    "require protected attributes collected under a lawful basis, a fair-lending review, "
    "and disparate-impact testing under ECOA."
)


def group_parity(
    groups: pd.Series,
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    *,
    min_rows: int = 100,
) -> list[dict[str, Any]]:
    """TPR / FPR / selection rate / calibration, per group, with sample sizes."""
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    pred = (p >= threshold).astype("float64")
    keys = pd.Series(groups).astype(str).to_numpy()

    rows = []
    for value in pd.unique(keys):
        mask = keys == value
        n = int(mask.sum())
        if n < min_rows:
            continue
        yg, pg, predg = y[mask], p[mask], pred[mask]
        positives = yg == 1
        negatives = yg == 0
        rows.append(
            {
                "group": str(value),
                "n": n,
                "base_rate": round(float(yg.mean()), 6),
                "selection_rate": round(float(predg.mean()), 6),
                "tpr": round(float(predg[positives].mean()), 6) if positives.any() else None,
                "fpr": round(float(predg[negatives].mean()), 6) if negatives.any() else None,
                "precision": (
                    round(float(yg[predg == 1].mean()), 6) if (predg == 1).any() else None
                ),
                "mean_predicted": round(float(pg.mean()), 6),
                "calibration_bias": round(float(pg.mean() - yg.mean()), 6),
                "ece": expected_calibration_error(pg, yg),
            }
        )
    rows.sort(key=lambda r: -(r["n"] or 0))
    return rows


def parity_gaps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Max-min gaps across groups. A gap is a finding, not a verdict."""
    def gap(metric: str) -> dict[str, Any]:
        values = [(r["group"], r[metric]) for r in rows if r.get(metric) is not None]
        if len(values) < 2:
            return {"gap": None, "max_group": None, "min_group": None}
        values.sort(key=lambda kv: kv[1])
        return {
            "gap": round(float(values[-1][1] - values[0][1]), 6),
            "max_group": values[-1][0],
            "max_value": round(float(values[-1][1]), 6),
            "min_group": values[0][0],
            "min_value": round(float(values[0][1]), 6),
        }

    return {
        "tpr_parity_gap": gap("tpr"),
        "fpr_parity_gap": gap("fpr"),
        "selection_rate_parity_gap": gap("selection_rate"),
        "calibration_parity_gap": gap("calibration_bias"),
    }


def fairness_report(
    X: pd.DataFrame,
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    *,
    dimensions: tuple[str, ...] | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    s = settings or get_settings()
    dims = dimensions or tuple(s.get("explain.fairness.proxy_dimensions", []) or ())

    out: dict[str, Any] = {
        "threshold": round(float(threshold), 6),
        "n": int(len(y)),
        "dimensions": {},
        "protected_attributes_available": False,
        "limitation": LIMITATION_STATEMENT,
        "interpretation": "diagnostic_only_not_an_optimisation_target",
    }
    for dimension in dims:
        if dimension not in X.columns:
            continue
        rows = group_parity(X[dimension], y, p, threshold)
        if not rows:
            continue
        out["dimensions"][dimension] = {"groups": rows, "gaps": parity_gaps(rows)}
    return out
