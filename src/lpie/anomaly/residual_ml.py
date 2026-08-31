"""Tier 2 — the supervised residual head.

Trained on `exception_required` **with the rule outputs as features**, so it can
only add value where the rules are silent or wrong. That framing is the point:
an ML model trained without the rule columns would spend its capacity
rediscovering a deterministic function it can only approximate.

The measured residual population is the ~2,750 positives whose
`document_status` is null. Only 15.4% of null-doc rows are true exceptions, so
the expected-F1-optimal threshold will usually *decline* to flag them — a
correct and defensible outcome, not a failure. And the test split has zero null
`document_status`, so on the actual submission the rules reach near-full recall
and this head is close to inert. We say that rather than claiming ML lift the
test data cannot express.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from lpie.core.logging import get_logger
from lpie.models.heads import HeadArtifact, predict_head

log = get_logger(__name__)


@dataclass
class ResidualArtifact:
    model: HeadArtifact | None = None
    threshold: float = 0.5
    rule_columns: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    ceiling: dict[str, Any] = field(default_factory=dict)


class ResidualExceptionHead:
    def __init__(self, artifact: ResidualArtifact | None = None) -> None:
        self.artifact = artifact or ResidualArtifact()

    @property
    def is_loaded(self) -> bool:
        return self.artifact.model is not None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_loaded:
            return np.zeros(len(X), dtype="float64")
        return predict_head(self.artifact.model, X)

    def combine(
        self,
        rule_required: pd.Series,
        rule_type: pd.Series,
        residual_proba: np.ndarray,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Rules win; ML only adds records the rules did not already flag.

        Returns (exception_required, exception_type, source). `source` is the
        provenance a reviewer needs: a record flagged by VR-006 is explainable in
        one sentence, one flagged by the residual head is not.
        """
        required = pd.Series(rule_required).astype("int64").copy()
        kind = pd.Series(rule_type).astype(object).copy()
        source = pd.Series("none", index=required.index, dtype=object)
        source[required.to_numpy() > 0] = "rule"

        if self.is_loaded and len(residual_proba) == len(required):
            ml_flag = (residual_proba >= self.artifact.threshold) & (required.to_numpy() == 0)
            if ml_flag.any():
                required[ml_flag] = 1
                kind[ml_flag] = "doc_gap"
                source[ml_flag] = "residual_ml"
        return required, kind, source
