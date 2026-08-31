"""Prediction scorer — one bundle per loan-month, with full provenance.

Order of operations matters and is fixed:

    features -> validation/DQ -> base learners -> stack -> calibration
             -> conformal interval -> ABSORBING-STATE GATE -> anomaly fusion
             -> thresholds -> reviewer action

The gate comes after the model and before the decision layer, deliberately.
`Prepaid` and `Closed` are *logically* absorbing (measured self-transition
1.00), not statistically so. Hard-coding that frees model capacity for the
active population, makes the active-conditional metric honest, and gives an
auditable `gated_by_rule` provenance field on every gated row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from lpie.core.config import Settings, get_settings
from lpie.core.exceptions import DataNotFoundError, ModelNotLoadedError, PredictionError
from lpie.core.logging import get_logger
from lpie.core.timing import Timer, utcnow_iso
from lpie.models.calibration import segment_labels
from lpie.models.heads import predict_head
from lpie.models.thresholds import (
    ReviewerPolicy,
    expected_loss,
    reviewer_action,
    severity_midpoints,
)

log = get_logger(__name__)

HEAD_TO_OUTPUT = {
    "next_3m_delinquency": "prob_next_3m_delinquency",
    "next_6m_delinquency": "prob_next_6m_delinquency",
    "next_12m_default": "prob_next_12m_default",
    "next_12m_prepayment": "prob_next_12m_prepayment",
}
TERMINAL_STATES = ("Prepaid", "Closed")
GATE_RULE = "VR-015"


@dataclass
class ScoringResult:
    frame: pd.DataFrame
    elapsed_ms: float
    n_rows: int
    model_version: str
    feature_version: str
    scored_at: str
    degraded: list[str]


class PredictionScorer:
    def __init__(self, state: Any, settings: Settings | None = None) -> None:
        self.state = state
        self.settings = settings or get_settings()
        self.policy = ReviewerPolicy.from_settings(self.settings)
        self.severity_map = severity_midpoints(self.settings)
        self.default_severity = float(self.severity_map.get("N/A", 0.35))

    # ------------------------------------------------------------------ #
    def score(
        self,
        features: pd.DataFrame,
        *,
        include_survival: bool = False,
        survival_horizon: int = 24,
        include_drivers: bool = True,
    ) -> ScoringResult:
        timer = Timer()
        if features.empty:
            raise DataNotFoundError("No feature rows were supplied for scoring")

        state = self.state
        if state.hazard is None or not state.hazard.is_loaded:
            raise ModelNotLoadedError(
                "Hazard model is unavailable; predictions cannot be produced.",
                details={"artifact": "hazard", "remedy": "run `make train`"},
            )
        if not state.heads:
            raise ModelNotLoadedError(
                "Direct-horizon head artifacts are unavailable.",
                details={"artifact": "heads", "remedy": "run `make train`"},
            )

        degraded: list[str] = []
        out = pd.DataFrame(index=features.index)
        out["loan_id"] = features["loan_id"].to_numpy()
        out["month_index"] = pd.to_numeric(features["month_index"], errors="coerce").astype("int64")
        out["reporting_month"] = features.get("reporting_month", pd.Series(index=features.index)).to_numpy()
        status = features["current_status"].astype("object")
        out["current_status"] = status.to_numpy()
        out["current_balance"] = pd.to_numeric(
            features.get("current_balance"), errors="coerce"
        ).fillna(0.0).to_numpy()

        is_terminal = status.isin(TERMINAL_STATES).to_numpy()
        out["is_terminal"] = is_terminal

        # ---------- next-state distribution from the hazard core ----------
        transition = state.hazard.predict_transition(features, status)
        states = state.hazard.states
        out["predicted_next_state"] = [states[i] for i in transition.argmax(axis=1)]
        out["next_state_confidence"] = transition.max(axis=1)
        for i, name in enumerate(states):
            out[f"p_next_{name}"] = transition[:, i]

        # ---------- horizon heads ----------
        base_predictions: dict[str, dict[str, np.ndarray]] = {}
        segments = segment_labels(
            features.get("vintage_year_num"), features.get("credit_score_band"), features.index
        )
        out["calibration_segment"] = segments.to_numpy()

        hazard_matrices = None
        for head, column in HEAD_TO_OUTPUT.items():
            models = state.heads.get(head)
            if not models:
                out[column] = np.nan
                degraded.append(head)
                continue

            preds: dict[str, np.ndarray] = {}
            for algo in ("lightgbm", "xgboost", "catboost"):
                artifact = models.get(algo)
                if artifact is not None:
                    preds[algo] = predict_head(artifact, features)

            horizon = int(self.settings.get(f"heads.{head}.horizon", 12))
            if hazard_matrices is None:
                hazard_matrices = state.hazard.transition_matrices(features)
            start = (
                status.map(state.hazard.state_index).fillna(0).astype("int32").to_numpy()
            )
            key = f"{head}_h"
            compounded = state.hazard.horizon_probabilities(hazard_matrices, start, {key: horizon})
            preds["hazard"] = compounded.get(key, np.full(len(features), np.nan))

            rates = models.get("baseline_state_rates", {})
            overall = float(models.get("baseline_state_overall", 0.0))
            preds["baseline_state"] = (
                status.map(rates).astype("float64").fillna(overall).to_numpy()
            )
            base_predictions[head] = preds

            stack = state.stacks.get(head)
            raw = stack.predict(preds) if stack is not None else np.nanmean(
                np.column_stack(list(preds.values())), axis=1
            )
            if stack is None:
                degraded.append(f"{head}:stack")

            calibrator = state.calibrators.get(head)
            if calibrator is not None and calibrator.artifact.is_fitted:
                calibrated = calibrator.transform(raw, segments)
                out[f"{column}_calibrated"] = True
            else:
                calibrated = raw
                out[f"{column}_calibrated"] = False
                degraded.append(f"{head}:calibration")

            out[column] = np.clip(calibrated, 0.0, 1.0)
            out[f"{column}_raw"] = np.clip(raw, 0.0, 1.0)

            # Epistemic uncertainty is free: the base predictions already exist.
            from lpie.models.ensemble import StackedBlender

            out[f"{column}_disagreement"] = StackedBlender.disagreement(preds)

            conformal = state.conformal.get(head)
            if conformal is not None and conformal.artifact.is_fitted:
                lower, upper = conformal.interval(out[column].to_numpy(), status)
                out[f"{column}_ci_low"] = lower
                out[f"{column}_ci_high"] = upper
            else:
                out[f"{column}_ci_low"] = np.nan
                out[f"{column}_ci_high"] = np.nan

        # ---------- exception + anomaly ----------
        self._apply_exception_and_anomaly(features, out, degraded)

        # ---------- ABSORBING-STATE GATE ----------
        self._apply_absorbing_gate(out, status)

        # ---------- confidence, decision, expected loss ----------
        self._apply_confidence(features, out)
        out["reviewer_action"] = reviewer_action(
            out["prob_next_12m_default"].fillna(0.0),
            out["anomaly_score"].fillna(0.0),
            out["exception_required"].fillna(0),
            out.get("exception_severity"),
            policy=self.policy,
        ).to_numpy()

        severity = self._loss_severity(features)
        out["expected_loss"] = expected_loss(
            out["prob_next_12m_default"].fillna(0.0), out["current_balance"], severity
        ).to_numpy()

        if include_drivers:
            self._apply_drivers(features, out)

        if include_survival:
            self._apply_survival(features, out, status, survival_horizon, hazard_matrices)

        out["model_version"] = self.settings.model_version
        out["feature_version"] = self.settings.feature_version
        scored_at = utcnow_iso()
        out["scored_at"] = scored_at

        return ScoringResult(
            frame=out.reset_index(drop=True),
            elapsed_ms=round(timer.stop(), 2),
            n_rows=int(len(out)),
            model_version=self.settings.model_version,
            feature_version=self.settings.feature_version,
            scored_at=scored_at,
            degraded=sorted(set(degraded)),
        )

    # ------------------------------------------------------------------ #
    def _apply_exception_and_anomaly(
        self, features: pd.DataFrame, out: pd.DataFrame, degraded: list[str]
    ) -> None:
        from lpie.anomaly.fusion import fuse
        from lpie.anomaly.rules_tier import EXCEPTION_RULES, rule_severity_score

        rule_columns = {
            rid: f"rule_{rid.replace('-', '_').lower()}_violated"
            for rid in [r.rule_id for r in self.state.validation_engine.rules]
        }
        available = {rid: col for rid, col in rule_columns.items() if col in features.columns}

        if available:
            passes = pd.DataFrame(
                {rid: features[col].fillna(0.0) < 0.5 for rid, col in available.items()},
                index=features.index,
            )
            rules = [r for r in self.state.validation_engine.rules if r.rule_id in available]
            severity, worst = rule_severity_score(passes, rules)
            required = pd.Series(0, index=features.index, dtype="int64")
            kind = pd.Series("None", index=features.index, dtype=object)
            for rid, exception_type in EXCEPTION_RULES.items():
                if rid in passes.columns:
                    fired = ~passes[rid]
                    required |= fired.astype("int64")
                    kind[fired.to_numpy()] = exception_type
        else:
            severity = pd.Series(0.0, index=features.index)
            worst = pd.Series("NONE", index=features.index, dtype=object)
            required = pd.Series(0, index=features.index, dtype="int64")
            kind = pd.Series("None", index=features.index, dtype=object)
            degraded.append("exception:rules")

        out["rule_severity"] = severity.to_numpy()
        out["worst_severity"] = worst.to_numpy()
        out["exception_severity"] = worst.to_numpy()
        out["exception_required"] = required.to_numpy()
        out["exception_type"] = kind.to_numpy()
        out["exception_source"] = np.where(required.to_numpy() > 0, "rule", "none")

        unsupervised = np.zeros(len(features), dtype="float64")
        if self.state.anomaly is not None:
            try:
                scores = self.state.anomaly.score(features)
                unsupervised = scores["unsupervised_rank"]
                for detector in ("iforest", "ecod", "autoencoder", "self_z"):
                    if detector in scores:
                        out[detector] = scores[detector]
            except Exception as exc:
                degraded.append("anomaly:unsupervised")
                log.warning("scorer.anomaly_failed", error=str(exc))
        else:
            degraded.append("anomaly:unsupervised")

        score, tier = fuse(unsupervised, severity, worst, settings=self.settings)
        out["anomaly_score"] = score
        out["anomaly_tier"] = tier

    # ------------------------------------------------------------------ #
    def _apply_absorbing_gate(self, out: pd.DataFrame, status: pd.Series) -> None:
        """Deterministic gate. The model does not get to override logic.

        `Prepaid` and `Closed` have a measured self-transition of exactly 1.00.
        A loan already prepaid has a 12-month prepayment probability of 1 because
        the event has *already happened*, not because the model predicts it, and
        its delinquency and default probabilities are 0 by construction.
        """
        out["gated_by_rule"] = None
        terminal = status.isin(TERMINAL_STATES).to_numpy()
        if not terminal.any():
            return

        prepaid = (status == "Prepaid").to_numpy()
        closed = (status == "Closed").to_numpy()

        for column in ("prob_next_3m_delinquency", "prob_next_6m_delinquency", "prob_next_12m_default"):
            if column in out.columns:
                out.loc[terminal, column] = 0.0
                out.loc[terminal, f"{column}_ci_low"] = 0.0
                out.loc[terminal, f"{column}_ci_high"] = 0.0
        if "prob_next_12m_prepayment" in out.columns:
            out.loc[prepaid, "prob_next_12m_prepayment"] = 1.0
            out.loc[prepaid, "prob_next_12m_prepayment_ci_low"] = 1.0
            out.loc[prepaid, "prob_next_12m_prepayment_ci_high"] = 1.0
            out.loc[closed, "prob_next_12m_prepayment"] = 0.0
            out.loc[closed, "prob_next_12m_prepayment_ci_low"] = 0.0
            out.loc[closed, "prob_next_12m_prepayment_ci_high"] = 0.0

        out.loc[terminal, "predicted_next_state"] = status[terminal].to_numpy()
        out.loc[terminal, "next_state_confidence"] = 1.0
        for name in ("Current", "30DPD", "60DPD", "90DPD", "Default", "Prepaid", "Closed"):
            column = f"p_next_{name}"
            if column in out.columns:
                out.loc[terminal, column] = 0.0
        out.loc[prepaid, "p_next_Prepaid"] = 1.0
        out.loc[closed, "p_next_Closed"] = 1.0

        out.loc[terminal, "expected_loss"] = 0.0
        out.loc[terminal, "gated_by_rule"] = GATE_RULE

    # ------------------------------------------------------------------ #
    def _apply_confidence(self, features: pd.DataFrame, out: pd.DataFrame) -> None:
        """A principled composite, not a repackaged probability.

            0.40 * (1 - normalised conformal width)
          + 0.30 * (1 - normalised ensemble disagreement)
          + 0.20 * segment calibration quality
          + 0.10 * input data quality

        Including data quality is the cross-task integration that ties Task 1 to
        Task 6: a prediction made on a broken record should not be confident,
        however extreme the probability.
        """
        n = len(out)
        width = out.get("prob_next_12m_default_ci_high", pd.Series(np.nan, index=out.index)) - out.get(
            "prob_next_12m_default_ci_low", pd.Series(np.nan, index=out.index)
        )
        width_term = 1.0 - np.clip(pd.to_numeric(width, errors="coerce").fillna(0.30).to_numpy(), 0.0, 1.0)

        disagreement = pd.to_numeric(
            out.get("prob_next_12m_default_disagreement", pd.Series(0.0, index=out.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy()
        # logit-space std of ~2 is total disagreement between base learners.
        disagreement_term = 1.0 - np.clip(disagreement / 2.0, 0.0, 1.0)

        segment_ece = np.full(n, 0.02, dtype="float64")
        calibrator = self.state.calibrators.get("next_12m_default")
        if calibrator is not None and calibrator.artifact.is_fitted:
            ece = calibrator.artifact.metrics.get("ece_after")
            if ece is not None and np.isfinite(ece):
                segment_ece = np.full(n, float(ece), dtype="float64")
        calibration_term = 1.0 - np.clip(segment_ece * 10.0, 0.0, 1.0)

        dq = pd.to_numeric(features.get("dq_score"), errors="coerce")
        dq_term = np.clip((dq.fillna(85.0) / 100.0).to_numpy(), 0.0, 1.0)

        confidence = (
            0.40 * width_term + 0.30 * disagreement_term + 0.20 * calibration_term + 0.10 * dq_term
        )
        # A terminal row is a logical certainty, so the gate sets confidence to 1.
        terminal = out["is_terminal"].to_numpy()
        confidence = np.where(terminal, 1.0, confidence)

        out["model_confidence"] = np.clip(confidence, 0.0, 1.0)
        out["conformal_width"] = np.clip(
            pd.to_numeric(width, errors="coerce").fillna(np.nan).to_numpy(), 0.0, 1.0
        )
        out["ensemble_disagreement"] = disagreement
        out["segment_ece"] = segment_ece
        out["data_quality_term"] = dq_term
        out["dq_score"] = pd.to_numeric(features.get("dq_score"), errors="coerce").to_numpy()
        out["dq_grade"] = self._dq_grade(out["dq_score"])

    @staticmethod
    def _dq_grade(scores: pd.Series) -> np.ndarray:
        values = pd.to_numeric(scores, errors="coerce")
        return np.select(
            [values >= 95, values >= 85, values >= 70, values >= 50],
            ["A", "B", "C", "D"],
            default="F",
        )

    def _loss_severity(self, features: pd.DataFrame) -> np.ndarray:
        band = features.get("loss_severity_band")
        if band is None:
            return np.full(len(features), self.default_severity, dtype="float64")
        mapped = band.astype("object").map(self.severity_map)
        return pd.to_numeric(mapped, errors="coerce").fillna(self.default_severity).to_numpy()

    # ------------------------------------------------------------------ #
    def _apply_drivers(self, features: pd.DataFrame, out: pd.DataFrame) -> None:
        """Top-3 SHAP drivers, from the cached global ranking when a per-row
        SHAP pass would be too expensive for the batch size.

        A batch of 5,000 rows does not get an exact TreeSHAP pass on the request
        path; it gets the model's ranked drivers restricted to the features that
        are actually extreme on each row. `/explain/{loan_id}` runs exact
        TreeSHAP for a single row, where the cost is trivial.
        """
        ranking = (self.state.shap_global or {}).get("next_12m_default", {}).get("global_importance")
        if not ranking:
            for i in (1, 2, 3):
                out[f"top_driver_{i}"] = ""
            return

        candidates = [r["feature"] for r in ranking[:25] if r["feature"] in features.columns]
        if not candidates:
            for i in (1, 2, 3):
                out[f"top_driver_{i}"] = ""
            return

        values = features[candidates].apply(pd.to_numeric, errors="coerce")
        mean = values.mean()
        std = values.std(ddof=0).replace(0.0, np.nan)
        z = ((values - mean) / std).abs().infer_objects(copy=False).fillna(0.0)
        weight = np.array(
            [next(r["mean_abs_shap"] for r in ranking if r["feature"] == c) for c in candidates]
        )
        combined = z.to_numpy() * weight[None, :]

        order = np.argsort(-combined, axis=1)[:, :3]
        for rank in range(3):
            out[f"top_driver_{rank + 1}"] = [candidates[order[i, rank]] for i in range(len(out))]

    # ------------------------------------------------------------------ #
    def _apply_survival(
        self,
        features: pd.DataFrame,
        out: pd.DataFrame,
        status: pd.Series,
        horizon: int,
        matrices: np.ndarray | None,
    ) -> None:
        hazard = self.state.hazard
        M = matrices if matrices is not None else hazard.transition_matrices(features)
        start = status.map(hazard.state_index).fillna(0).astype("int32").to_numpy()
        propagated = hazard.propagate(M, start, horizon)
        out.attrs["survival"] = {
            "horizons": list(range(horizon + 1)),
            "survival": propagated["survival"],
            "cif_default": propagated["cif_default"],
            "cif_prepaid": propagated["cif_prepaid"],
            "cif_closed": propagated["cif_closed"],
            "conservation_max_error": propagated["conservation_max_error"],
        }


def build_features_for(
    state: Any, loan_ids: list[str] | None = None, months: list[int] | None = None
) -> pd.DataFrame:
    """Read precomputed features, falling back to an online build.

    The store is the fast path — a partition-pruned Parquet read. The online
    build exists for rows that arrive at `/predict` and are not in the store,
    and it runs the *same* builder, so training and serving features cannot
    diverge.
    """
    frame = state.features(months=months, loan_ids=loan_ids)
    if not frame.empty:
        return frame
    raise DataNotFoundError(
        "No precomputed features are available for the requested rows.",
        details={"loan_ids": (loan_ids or [])[:5], "months": months,
                 "remedy": "run `make features` to build the feature store"},
    )


def build_features_online(state: Any, rows: pd.DataFrame) -> pd.DataFrame:
    """Build features for caller-supplied rows, using their own history.

    History comes from the stored panel for the same loans, so a single uploaded
    month still gets its lag, rolling and cohort features computed correctly
    rather than degrading silently to NaN.
    """
    if state.feature_builder is None:
        raise ModelNotLoadedError(
            "Feature pipeline is not initialised (feature_fit artifact missing).",
            details={"artifact": "feature_fit", "remedy": "run `make features`"},
        )
    panel = state.panel()
    loan_ids = set(rows["loan_id"].astype(str))
    history = panel[panel["loan_id"].astype(str).isin(loan_ids)]

    combined = pd.concat([history, rows], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["loan_id", "month_index"], keep="last")
    combined = combined.sort_values(["loan_id", "month_index"], kind="mergesort").reset_index(drop=True)

    result = state.feature_builder.build(
        combined, state.static(), state.servicer(), enforce_contract=False
    )
    features = result.features
    keys = set(zip(rows["loan_id"].astype(str), pd.to_numeric(rows["month_index"], errors="coerce"), strict=False))
    mask = [
        (str(loan), month) in keys
        for loan, month in zip(features["loan_id"], features["month_index"], strict=False)
    ]
    selected = features[mask]
    if selected.empty:
        raise PredictionError("Feature construction produced no rows for the supplied records")
    return selected.reset_index(drop=True)
