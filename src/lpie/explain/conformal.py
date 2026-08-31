"""Split-conformal prediction intervals, Mondrian on `current_status`.

The only uncertainty method here with a finite-sample statistical guarantee:
marginal coverage is at least 1 - alpha regardless of whether the model is
correct. Conditioning on `current_status` (Mondrian) means the guarantee holds
*within each state* rather than only on average across a book that is 60%
absorbing — which is what makes it usable for a per-loan interval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from lpie.core.logging import get_logger

log = get_logger(__name__)

MIN_GROUP_ROWS = 100


@dataclass
class ConformalArtifact:
    alpha: float = 0.10
    global_quantile: float = 0.5
    group_quantiles: dict[str, float] = field(default_factory=dict)
    group_counts: dict[str, int] = field(default_factory=dict)
    n_calibration: int = 0
    achieved_coverage: dict[str, Any] = field(default_factory=dict)

    @property
    def is_fitted(self) -> bool:
        return self.n_calibration > 0


class MondrianConformal:
    """Split-conformal with per-group nonconformity quantiles."""

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = float(alpha)
        self.artifact = ConformalArtifact(alpha=self.alpha)

    def fit(
        self, p: np.ndarray, y: np.ndarray, groups: pd.Series | None = None
    ) -> ConformalArtifact:
        """Nonconformity is |y - p|: how far the calibrated probability was from truth."""
        p = np.asarray(p, dtype="float64")
        y = np.asarray(y, dtype="float64")
        finite = np.isfinite(p) & np.isfinite(y)
        p, y = p[finite], y[finite]
        if len(p) < 50:
            log.warning("conformal.insufficient_data", n=int(len(p)))
            return self.artifact

        scores = np.abs(y - p)
        n = len(scores)
        # The finite-sample-valid quantile level: ceil((n+1)(1-alpha))/n.
        level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.artifact.global_quantile = float(np.quantile(scores, level))
        self.artifact.n_calibration = n

        if groups is not None:
            group_values = pd.Series(groups).astype(str).to_numpy()[finite]
            for key in pd.unique(group_values):
                mask = group_values == key
                count = int(mask.sum())
                self.artifact.group_counts[str(key)] = count
                if count < MIN_GROUP_ROWS:
                    continue
                group_level = min(np.ceil((count + 1) * (1 - self.alpha)) / count, 1.0)
                self.artifact.group_quantiles[str(key)] = float(
                    np.quantile(scores[mask], group_level)
                )

        self.artifact.achieved_coverage = self.evaluate(p, y, groups[finite] if groups is not None else None)
        return self.artifact

    def interval(
        self, p: np.ndarray, groups: pd.Series | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) with the group quantile where one is available."""
        p = np.asarray(p, dtype="float64")
        width = np.full(len(p), self.artifact.global_quantile, dtype="float64")
        if groups is not None and self.artifact.group_quantiles:
            keys = pd.Series(groups).astype(str).to_numpy()
            for key, q in self.artifact.group_quantiles.items():
                width[keys == key] = q
        return np.clip(p - width, 0.0, 1.0), np.clip(p + width, 0.0, 1.0)

    def evaluate(
        self, p: np.ndarray, y: np.ndarray, groups: pd.Series | None = None
    ) -> dict[str, Any]:
        """Achieved coverage versus nominal — reported, not assumed."""
        lower, upper = self.interval(p, groups)
        y = np.asarray(y, dtype="float64")
        covered = (y >= lower - 1e-9) & (y <= upper + 1e-9)
        out: dict[str, Any] = {
            "nominal_coverage": round(1 - self.alpha, 4),
            "achieved_coverage": round(float(covered.mean()), 6),
            "mean_width": round(float((upper - lower).mean()), 6),
            "n": int(len(y)),
        }
        if groups is not None:
            keys = pd.Series(groups).astype(str).to_numpy()
            out["by_group"] = [
                {
                    "group": str(key),
                    "n": int((keys == key).sum()),
                    "coverage": round(float(covered[keys == key].mean()), 6),
                    "mean_width": round(float((upper - lower)[keys == key].mean()), 6),
                }
                for key in pd.unique(keys)
            ]
        return out
