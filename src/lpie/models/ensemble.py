"""Stacked meta-learner over the direct heads, the hazard core and the baseline.

    final_p = sigma( w0 + w1*logit(p_direct_lgb) + w2*logit(p_direct_xgb)
                        + w3*logit(p_direct_cat) + w4*logit(p_hazard_compounded)
                        + w5*logit(p_baseline_state_rate) )

The meta-learner is an L2 logistic regression fit on **out-of-fold predictions
from the walk-forward folds only** — never in-fold — with the calibration slice
held out from it entirely. Blending a *structural* model (the hazard chain) with
*discriminative* ones is the classic way to get both calibration and
discrimination, and it gives a free consistency check: a large disagreement
between `p_direct` and `p_hazard` is itself an anomaly signal worth surfacing to
a reviewer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from lpie.core.exceptions import PredictionError
from lpie.core.logging import get_logger

log = get_logger(__name__)

EPS = 1e-6
BASE_LEARNERS = ("lightgbm", "xgboost", "catboost", "hazard", "baseline_state")


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype="float64"), -35.0, 35.0)))


@dataclass
class StackArtifact:
    head: str
    learners: list[str]
    model: LogisticRegression | None = None
    coefficients: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    n_training_rows: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    fallback_weights: dict[str, float] = field(default_factory=dict)

    @property
    def is_fitted(self) -> bool:
        return self.model is not None or bool(self.fallback_weights)


class StackedBlender:
    """L2 logistic stack over base-learner logits."""

    def __init__(self, head: str, learners: list[str] | None = None, C: float = 1.0) -> None:
        self.head = head
        self.learners = list(learners or BASE_LEARNERS)
        self.C = float(C)
        self.artifact = StackArtifact(head=head, learners=self.learners)

    # ------------------------------------------------------------------ #
    def _design(self, oof: dict[str, np.ndarray], n: int) -> tuple[np.ndarray, list[str]]:
        columns, names = [], []
        for name in self.learners:
            values = oof.get(name)
            if values is None:
                continue
            arr = np.asarray(values, dtype="float64")
            if arr.shape[0] != n:
                raise PredictionError(
                    f"Base learner '{name}' produced {arr.shape[0]} predictions for {n} rows",
                    details={"head": self.head, "learner": name},
                )
            columns.append(logit(arr))
            names.append(name)
        if not columns:
            raise PredictionError(
                f"No base-learner predictions available to stack for head '{self.head}'"
            )
        return np.column_stack(columns), names

    def fit(
        self, oof_predictions: dict[str, np.ndarray], y: np.ndarray | pd.Series
    ) -> StackArtifact:
        y_arr = np.asarray(y, dtype="float64")
        X, names = self._design(oof_predictions, len(y_arr))

        finite = np.isfinite(X).all(axis=1) & np.isfinite(y_arr)
        X, y_fit = X[finite], y_arr[finite]

        if len(y_fit) < 50 or len(np.unique(y_fit)) < 2:
            # Not enough out-of-fold signal to fit a meta-learner honestly.
            # Fall back to an equal-weight logit average and say so — a stack
            # fitted on 40 rows would be worse than no stack at all.
            weights = {n: 1.0 / len(names) for n in names}
            self.artifact = StackArtifact(
                head=self.head, learners=names, fallback_weights=weights,
                n_training_rows=int(len(y_fit)),
                metrics={"note": "insufficient out-of-fold rows; using equal-weight logit average"},
            )
            log.warning("stack.fallback", head=self.head, n=int(len(y_fit)))
            return self.artifact

        # L2 by default in scikit-learn; passing `penalty=` explicitly is
        # deprecated from 1.8, so the regularisation strength is set via C alone.
        # Non-negative constrained stack: fit unconstrained first, then
        # clip negative weights to zero and renormalise. This prevents the
        # meta-learner from "anti-blending" (subtracting a learner's prediction)
        # which can happen when base learners are correlated. A negative weight
        # means the stack is degrading, not improving, the blended signal.
        model = LogisticRegression(C=self.C, max_iter=1000, solver="lbfgs", random_state=0)
        model.fit(X, y_fit)

        # Clip negative coefficients to 0 (non-negative constraint post-hoc)
        coef = model.coef_[0].copy()
        had_negatives = (coef < 0).any()
        coef = np.maximum(coef, 0.0)
        total = coef.sum()
        if total > EPS:
            coef = coef * (model.coef_[0].sum() / total if model.coef_[0].sum() > EPS else 1.0)
        else:
            coef = np.ones_like(coef) / max(len(coef), 1)
        model.coef_[0] = coef
        if had_negatives:
            log.info(
                "stack.negative_weights_clipped",
                head=self.head,
                note="One or more base learner coefficients were negative and clipped to 0. "
                     "This prevents anti-blending where correlated learners cancel each other.",
            )

        self.artifact = StackArtifact(
            head=self.head,
            learners=names,
            model=model,
            coefficients={n: round(float(c), 6) for n, c in zip(names, model.coef_[0], strict=False)},
            intercept=round(float(model.intercept_[0]), 6),
            n_training_rows=int(len(y_fit)),
        )
        return self.artifact

    # ------------------------------------------------------------------ #
    def predict(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        art = self.artifact
        if not art.is_fitted:
            raise PredictionError(f"Stack for head '{self.head}' is not fitted")

        len(next(iter(predictions.values())))
        columns, names = [], []
        for name in art.learners:
            values = predictions.get(name)
            if values is None:
                # A missing base learner at serving time is a degraded but usable
                # state: drop its column and renormalise the remaining weights,
                # rather than refusing to score.
                continue
            columns.append(logit(np.asarray(values, dtype="float64")))
            names.append(name)
        if not columns:
            raise PredictionError(f"No base-learner predictions supplied for head '{self.head}'")
        X = np.column_stack(columns)

        if art.model is not None and names == art.learners:
            return np.clip(np.asarray(art.model.predict_proba(X)[:, 1], dtype="float64"), 0.0, 1.0)

        if art.model is not None:
            coefs = np.array([art.coefficients.get(n, 0.0) for n in names], dtype="float64")
            total = float(np.abs(coefs).sum())
            if total > EPS:
                z = art.intercept + X @ coefs
                return np.clip(sigmoid(z), 0.0, 1.0)

        weights = np.array(
            [art.fallback_weights.get(n, 1.0 / len(names)) for n in names], dtype="float64"
        )
        weights = weights / weights.sum()
        return np.clip(sigmoid(X @ weights), 0.0, 1.0)

    # ------------------------------------------------------------------ #
    @staticmethod
    def disagreement(predictions: dict[str, np.ndarray]) -> np.ndarray:
        """Std of logit(p) across base learners — the epistemic uncertainty term.

        High disagreement means the model is out of its depth. It is essentially
        free (the predictions already exist) and it feeds `model_confidence`.
        """
        arrays = [logit(np.asarray(v, dtype="float64")) for v in predictions.values() if v is not None]
        if len(arrays) < 2:
            return np.zeros(len(arrays[0]) if arrays else 0, dtype="float64")
        return np.std(np.column_stack(arrays), axis=1)


def out_of_fold_matrix(
    fold_predictions: list[dict[str, Any]], n_rows: int, learners: list[str]
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Assemble OOF predictions from per-fold validation slices.

    Each entry is `{"indices": ndarray, "y": ndarray, "predictions": {learner: ndarray}}`.
    Rows never validated by any fold stay NaN and are excluded from the stack fit —
    which is the whole point: the meta-learner must never see an in-fold prediction.
    """
    oof = {name: np.full(n_rows, np.nan, dtype="float64") for name in learners}
    y_oof = np.full(n_rows, np.nan, dtype="float64")
    covered = np.zeros(n_rows, dtype=bool)

    for entry in fold_predictions:
        idx = np.asarray(entry["indices"], dtype="int64")
        y_oof[idx] = np.asarray(entry["y"], dtype="float64")
        covered[idx] = True
        for name, values in entry["predictions"].items():
            if name in oof:
                oof[name][idx] = np.asarray(values, dtype="float64")

    keep = covered & np.isfinite(y_oof)
    for name in list(oof):
        keep &= np.isfinite(oof[name])
    return {name: values[keep] for name, values in oof.items()}, y_oof[keep], np.flatnonzero(keep)
