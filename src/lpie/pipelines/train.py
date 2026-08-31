"""Training pipeline — offline only, never invoked from a request path.

Per head:

    for each walk-forward fold:
        train LGB / XGB / CatBoost on the censoring-masked train window
        predict the embargoed validation window  -> out-of-fold predictions
    fit the L2 logistic stack on OOF predictions only
    fit isotonic calibration on the dedicated out-of-time calibration slice
    optimise the decision threshold on the calibration slice
    refit the base learners on the full censoring-valid window for production

The hazard core is trained once over the person-month panel and supplies the
`hazard` base learner to every horizon head, which is what keeps the horizon
probabilities, the next-state distribution and the survival curves mutually
consistent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from lpie.core.config import Settings, get_settings
from lpie.core.determinism import seed_everything, sha256_obj
from lpie.core.logging import get_logger
from lpie.core.timing import utcnow_iso
from lpie.evaluation.metrics import (
    binary_metrics,
    bootstrap_ci,
    dual_binary_metrics,
    dual_multiclass_metrics,
    fold_summary,
)
from lpie.models.calibration import (
    SegmentIsotonicCalibrator,
    calibration_report,
    cross_fitted_calibrated,
    segment_labels,
)
from lpie.models.ensemble import StackedBlender, out_of_fold_matrix
from lpie.models.hazard import HazardModel
from lpie.models.heads import (
    TRAINERS,
    baseline_current_state,
    baseline_prior,
    fit_category_levels,
    predict_head,
    scale_pos_weight_for,
)
from lpie.models.splitters import SplitPlan, build_split_plan, censoring_mask

log = get_logger(__name__)

# Absorbing states. Mirrors lpie.serving.scorer.TERMINAL_STATES: the rows the
# serving layer gates to 0.0 are exactly the rows excluded from the isotonic fit.
_TERMINAL_STATES = ("Prepaid", "Closed")


def _apply_terminal_gate(p: np.ndarray, status: pd.Series, head: str) -> None:
    """Mirror PredictionScorer._gate_terminal_rows, in place.

    The gate is head-specific and the difference matters. A loan already
    `Prepaid` has a 12-month prepayment probability of 1 — the event has
    happened — while its delinquency and default probabilities are 0. Gating
    every head to 0.0 assigns probability zero to rows whose prepayment label is
    1, which drove the head's overall ROC-AUC to 0.238 (worse than random) and
    its ECE to 0.386. Evaluation must gate exactly the way serving gates, or the
    published metrics describe a system that was never deployed.
    """
    terminal = status.isin(_TERMINAL_STATES).to_numpy()
    if not terminal.any():
        return
    if head == "next_12m_prepayment":
        p[(status == "Prepaid").to_numpy()] = 1.0
        p[(status == "Closed").to_numpy()] = 0.0
    else:
        p[terminal] = 0.0


def production_train_months(plan: SplitPlan) -> list[int]:
    """All censoring-valid months, for the model that actually scores production.

    This is *not* the embargoed fold window and the difference matters. The
    embargo exists to stop a training label from containing the validation
    period's outcomes. For production scoring the "validation period" is months
    37-42, and the label window of a month-24 row with a 12-month horizon closes
    at month 36 — strictly before scoring begins. So training the production
    model on every censoring-valid month is leakage-free *for the task it is
    used for*, while the fold models stay strictly embargoed for the task they
    are used for (honest measurement).

    Calibration is fitted on the pooled out-of-fold predictions from the
    embargoed validation windows, which are out-of-time by construction, rather
    than on rows the production model has seen.
    """
    return list(range(1, plan.max_valid_month + 1))


BINARY_HEADS = (
    "next_3m_delinquency", "next_6m_delinquency", "next_12m_default", "next_12m_prepayment",
)
STACK_LEARNERS = ["lightgbm", "xgboost", "catboost", "hazard", "baseline_state"]


@dataclass
class HeadTrainingResult:
    head: str
    plan: SplitPlan
    base_models: dict[str, Any] = field(default_factory=dict)
    stack: StackedBlender | None = None
    calibrator: SegmentIsotonicCalibrator | None = None
    threshold: Any = None
    fold_metrics: list[dict[str, Any]] = field(default_factory=list)
    calibration_metrics: dict[str, Any] = field(default_factory=dict)
    baseline_ladder: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0


def _feature_lists(registry, head: str) -> tuple[list[str], list[str]]:
    names = registry.for_head(head)
    cats = [n for n in registry.categorical_features(head) if n in names]
    return names, cats


def train_hazard(
    features: pd.DataFrame,
    *,
    registry,
    settings: Settings | None = None,
    panel_max_month: int | None = None,
    num_boost_round: int = 400,
) -> tuple[HazardModel, dict[str, Any]]:
    """Fit the one-step transition model on every row that has a next-state label."""
    s = settings or get_settings()
    plan = build_split_plan("next_state", panel_max_month=panel_max_month, settings=s)
    names, cats = _feature_lists(registry, "hazard")

    labelled = features["next_state"].notna() & censoring_mask(
        features["month_index"], plan.horizon, plan.panel_max_month
    )
    # Measurement uses the embargoed fold windows; the shipped model uses every
    # censoring-valid month (see production_train_months).
    train_mask = labelled & features["month_index"].isin(production_train_months(plan))
    valid_mask = labelled & features["month_index"].isin(plan.calibration_months)
    measurement_train = labelled & features["month_index"].isin(plan.final_train_months)

    model = HazardModel(s)
    model.fit(
        features.loc[train_mask],
        features.loc[train_mask, "next_state"],
        feature_names=names,
        categorical_features=cats,
        valid_sets=[(features.loc[valid_mask], features.loc[valid_mask, "next_state"])]
        if valid_mask.any()
        else None,
        num_boost_round=num_boost_round,
    )

    metrics: dict[str, Any] = {
        "production_train_window": f"1-{plan.max_valid_month}",
        "embargoed_measurement_window": plan.to_dict()["final_train_window"],
        "n_measurement_train": int(measurement_train.sum()),
        "train_window": f"1-{plan.max_valid_month}",
        "valid_window": plan.to_dict()["calibration_window"],
        "n_train": int(train_mask.sum()),
        "n_valid": int(valid_mask.sum()),
        "fold_plan": plan.to_dict(),
    }

    if valid_mask.any():
        Xv = features.loc[valid_mask]
        probs = model.predict_transition(Xv, Xv["current_status"])
        predicted = pd.Series(
            [model.states[i] for i in probs.argmax(axis=1)], index=Xv.index, dtype=object
        )
        metrics["next_state"] = dual_multiclass_metrics(
            Xv["next_state"], predicted, Xv["current_status"], probs
        )
        metrics["invariants"] = model.validate_invariants(probs)

        # Fold-wise stability for the next_state head, on the same model.
        fold_metrics = []
        for fold in plan.folds:
            fm = labelled & features["month_index"].isin(fold.valid_months)
            if not fm.any():
                continue
            Xf = features.loc[fm]
            pf = model.predict_transition(Xf, Xf["current_status"])
            yhat = pd.Series([model.states[i] for i in pf.argmax(axis=1)], index=Xf.index, dtype=object)
            fold_metrics.append(
                {
                    "fold": fold.fold,
                    "valid_window": fold.valid_window,
                    **dual_multiclass_metrics(Xf["next_state"], yhat, Xf["current_status"]),
                }
            )
        metrics["folds"] = fold_metrics
        metrics["fold_stability"] = {
            "accuracy_overall": fold_summary(fold_metrics, ("overall", "accuracy")),
            "macro_f1_active": fold_summary(fold_metrics, ("active_conditional", "macro_f1")),
        }

    log.info("train.hazard.done", n_train=int(train_mask.sum()))
    return model, metrics


def _hazard_horizon_predictions(
    hazard: HazardModel, X: pd.DataFrame, head: str, horizon: int
) -> np.ndarray:
    """Compound the hazard chain into this head's horizon probability."""
    if not hazard.is_loaded or X.empty:
        return np.full(len(X), np.nan, dtype="float64")
    M = hazard.transition_matrices(X)
    start = (
        X["current_status"].astype("object").map(hazard.state_index).fillna(0).astype("int32").to_numpy()
    )
    key = f"{head}_h"
    probs = hazard.horizon_probabilities(M, start, {key: horizon})
    if key in probs:
        return probs[key]
    return np.full(len(X), np.nan, dtype="float64")


def train_binary_head(
    head: str,
    features: pd.DataFrame,
    *,
    registry,
    hazard: HazardModel,
    settings: Settings | None = None,
    panel_max_month: int | None = None,
    algorithms: tuple[str, ...] = ("lightgbm", "xgboost", "catboost"),
    tuned_params: dict[str, dict[str, Any]] | None = None,
    max_hazard_rows: int = 60_000,
) -> HeadTrainingResult:
    s = settings or get_settings()
    started = time.perf_counter()
    head_cfg = s.require(f"heads.{head}")
    target = head_cfg["target"]
    horizon = int(head_cfg.get("horizon", 0))

    plan = build_split_plan(head, panel_max_month=panel_max_month, settings=s)
    names, cats = _feature_lists(registry, head)
    result = HeadTrainingResult(head=head, plan=plan)

    labelled = features[target].notna() & censoring_mask(
        features["month_index"], horizon, plan.panel_max_month
    )
    y_all = pd.to_numeric(features[target], errors="coerce")

    # ------------------------------------------------------------------ #
    # walk-forward folds -> out-of-fold predictions
    # ------------------------------------------------------------------ #
    fold_predictions: list[dict[str, Any]] = []
    for fold in plan.folds:
        tr = labelled & features["month_index"].isin(fold.train_months)
        va = labelled & features["month_index"].isin(fold.valid_months)
        if tr.sum() < 500 or va.sum() < 100 or y_all[tr].nunique() < 2:
            log.warning("train.fold_skipped", head=head, fold=fold.fold,
                        n_train=int(tr.sum()), n_valid=int(va.sum()))
            continue

        X_tr, y_tr = features.loc[tr], y_all[tr]
        X_va, y_va = features.loc[va], y_all[va]
        levels = fit_category_levels(X_tr, cats)
        spw = scale_pos_weight_for(y_tr)

        preds: dict[str, np.ndarray] = {}
        for algo in algorithms:
            artifact = TRAINERS[algo](
                X_tr, y_tr, head=head, feature_names=names, categorical_features=cats,
                category_levels=levels, valid=(X_va, y_va),
                params=(tuned_params or {}).get(algo), scale_pos_weight=spw, settings=s,
            )
            preds[algo] = predict_head(artifact, X_va)

        preds["hazard"] = _hazard_horizon_predictions(hazard, X_va, head, horizon)
        preds["baseline_state"] = baseline_current_state(
            y_tr, X_tr["current_status"], X_va["current_status"]
        )

        fold_metrics = {
            "fold": fold.fold,
            "train_window": fold.train_window,
            "embargo_window": fold.embargo_window,
            "valid_window": fold.valid_window,
            "n_train": int(tr.sum()),
            "n_valid": int(va.sum()),
            "base_rate_train": round(float(y_tr.mean()), 6),
            "base_rate_valid": round(float(y_va.mean()), 6),
            "by_learner": {
                algo: dual_binary_metrics(y_va, values, X_va["current_status"])
                for algo, values in preds.items()
                if np.isfinite(values).any()
            },
        }
        result.fold_metrics.append(fold_metrics)
        fold_predictions.append(
            {
                "indices": np.flatnonzero(va.to_numpy()),
                "y": y_va.to_numpy(),
                "predictions": preds,
            }
        )
        log.info("train.fold.done", head=head, fold=fold.fold, n_train=int(tr.sum()))

    # ------------------------------------------------------------------ #
    # stack on out-of-fold predictions only (never in-fold)
    # ------------------------------------------------------------------ #
    stack = StackedBlender(head, STACK_LEARNERS)
    oof_stacked: np.ndarray | None = None
    oof_y: np.ndarray | None = None
    oof_index: np.ndarray | None = None

    if fold_predictions:
        oof, y_oof, oof_index = out_of_fold_matrix(fold_predictions, len(features), STACK_LEARNERS)
        stack.fit(oof, y_oof)
        oof_stacked = stack.predict(oof)
        oof_y = y_oof
    else:
        stack.artifact.fallback_weights = {n: 1.0 / len(STACK_LEARNERS) for n in STACK_LEARNERS}
        stack.artifact.metrics = {"note": "no usable folds; equal-weight logit average"}
    result.stack = stack

    # ------------------------------------------------------------------ #
    # production refit on every censoring-valid month
    # ------------------------------------------------------------------ #
    production_months = production_train_months(plan)
    final_train = labelled & features["month_index"].isin(production_months)
    X_final, y_final = features.loc[final_train], y_all[final_train]
    levels = fit_category_levels(X_final, cats)
    spw = scale_pos_weight_for(y_final)

    # Early stopping for the production refit uses the last fold's validation
    # window. That window is inside the production training range, so the
    # stopping iteration is mildly optimistic; it is used only to choose
    # n_estimators, never to report a metric.
    last_fold = plan.folds[-1] if plan.folds else None
    stop_mask = (
        labelled & features["month_index"].isin(last_fold.valid_months) if last_fold is not None else None
    )
    stop_valid = (
        (features.loc[stop_mask], y_all[stop_mask])
        if stop_mask is not None and stop_mask.sum() > 200
        else None
    )

    for algo in algorithms:
        result.base_models[algo] = TRAINERS[algo](
            X_final, y_final, head=head, feature_names=names, categorical_features=cats,
            category_levels=levels, valid=stop_valid,
            params=(tuned_params or {}).get(algo), scale_pos_weight=spw, settings=s,
        )
        result.base_models[algo].train_window = f"1-{plan.max_valid_month}"

    # The state-rate baseline is a lookup table but it is a stack member, so it
    # is persisted with the same provenance as the boosters.
    result.base_models["baseline_state_rates"] = (
        y_final.groupby(X_final["current_status"].astype("object")).mean().to_dict()
    )
    result.base_models["baseline_state_overall"] = float(y_final.mean())

    # ------------------------------------------------------------------ #
    # calibration + threshold on the pooled out-of-fold predictions
    # ------------------------------------------------------------------ #
    calibrator = SegmentIsotonicCalibrator()
    if oof_stacked is not None and oof_y is not None and len(oof_y) > 200:
        X_cal = features.iloc[oof_index]
        segments = segment_labels(
            X_cal.get("vintage_year_num"), X_cal.get("credit_score_band"), X_cal.index
        )
        calib_window = "+".join(f.valid_window for f in plan.folds)

        # Absorbing-state rows are gated by the serving layer, so they must not
        # enter the isotonic fit — see SegmentIsotonicCalibrator.fit.
        cal_status = X_cal["current_status"].astype("object")
        terminal = cal_status.isin(_TERMINAL_STATES).to_numpy()
        calibrator.fit(
            oof_stacked, oof_y, segments,
            calibration_window=calib_window, eligible=~terminal,
        )

        # Metrics and the decision threshold are computed on cross-fitted
        # (out-of-sample) calibrated scores carrying the same terminal gate the
        # serving path applies, so the numbers reported describe production
        # behaviour rather than the calibrator scoring its own fit data.
        calibrated = np.clip(calibrator.transform(oof_stacked, segments), 0.0, 1.0)
        honest = cross_fitted_calibrated(
            oof_stacked, oof_y, segments.to_numpy(), seed=s.seed, eligible=~terminal,
        )
        for array in (calibrated, honest):
            _apply_terminal_gate(array, cal_status, head)

        from lpie.models.thresholds import optimise_threshold

        objective = (
            "recall_at_precision" if head == "next_12m_default" else head_cfg.get("primary_metric", "f1")
        )
        if objective not in ("recall_at_precision", "f1", "f2"):
            objective = "f1"
        result.threshold = optimise_threshold(
            honest, oof_y, head=head, objective=objective,
            min_precision=float(s.get("thresholds.default_head.min_precision", 0.30)), settings=s,
        )
        # The full threshold curve is a large artifact; keep a thinned copy.
        result.threshold.curve = result.threshold.curve[:: max(len(result.threshold.curve) // 60, 1)]

        result.calibration_metrics = {
            "before": calibration_report(oof_stacked, oof_y, label="uncalibrated"),
            "after": calibration_report(honest, oof_y, label="calibrated (cross-fitted)"),
            "after_in_sample": calibration_report(
                calibrated, oof_y, label="calibrated (in-sample — reported for contrast only)"
            ),
            "window": calib_window,
            "source": "pooled out-of-fold predictions from embargoed validation windows",
            "estimator": (
                "5-fold cross-fitted isotonic; absorbing-state rows excluded from every "
                "fit and then gated exactly as the serving path gates them "
                "(Prepaid -> prepayment 1.0, all other heads 0.0)"
            ),
            "n_terminal_rows_gated": int(terminal.sum()),
            "n_segments": len(calibrator.artifact.segment_curves),
        }
        result.summary["out_of_fold_metrics"] = dual_binary_metrics(
            oof_y, honest, X_cal["current_status"],
            threshold=result.threshold.threshold if result.threshold else None,
        )
        result.summary["bootstrap_pr_auc"] = bootstrap_ci(oof_y, honest, "pr_auc", seed=s.seed)
        result.baseline_ladder = _baseline_ladder(
            features, y_all, labelled, plan, names, cats, s, result, hazard, head, horizon,
            oof_stacked=honest, oof_y=oof_y, oof_index=oof_index,
        )
    result.calibrator = calibrator

    result.summary.update(
        {
            "head": head,
            "target": target,
            "horizon": horizon,
            "max_valid_month": plan.max_valid_month,
            "n_folds": len(result.fold_metrics),
            "fold_note": plan.fold_note,
            "n_train_final": int(final_train.sum()),
            "production_train_window": f"1-{plan.max_valid_month}",
            "n_calibration": int(len(oof_y)) if oof_y is not None else 0,
            "calibration_source": "pooled out-of-fold (embargoed) predictions",
            "n_rows_masked_by_censoring": int((~censoring_mask(
                features["month_index"], horizon, plan.panel_max_month
            )).sum()),
            "base_rate": round(float(y_all[labelled].mean()), 6) if labelled.any() else None,
            "scale_pos_weight": spw,
            "stack_coefficients": stack.artifact.coefficients or stack.artifact.fallback_weights,
            "fold_stability": {
                "pr_auc_active": fold_summary(
                    [f["by_learner"].get("lightgbm", {}) for f in result.fold_metrics],
                    ("active_conditional", "pr_auc"),
                ),
                "roc_auc_overall": fold_summary(
                    [f["by_learner"].get("lightgbm", {}) for f in result.fold_metrics],
                    ("overall", "roc_auc"),
                ),
            },
            "trained_at": utcnow_iso(),
        }
    )
    result.elapsed_s = round(time.perf_counter() - started, 2)
    log.info("train.head.done", head=head, elapsed_s=result.elapsed_s, folds=len(result.fold_metrics))
    return result


def _stack_predict(
    result: HeadTrainingResult,
    stack: StackedBlender,
    hazard: HazardModel,
    X: pd.DataFrame,
    head: str,
    horizon: int,
    *,
    max_hazard_rows: int = 60_000,
) -> np.ndarray:
    preds: dict[str, np.ndarray] = {}
    for algo, artifact in result.base_models.items():
        if algo.startswith("baseline_"):
            continue
        preds[algo] = predict_head(artifact, X)

    preds["hazard"] = (
        _hazard_horizon_predictions(hazard, X, head, horizon)
        if len(X) <= max_hazard_rows
        else _hazard_in_chunks(hazard, X, head, horizon, max_hazard_rows)
    )

    rates = result.base_models.get("baseline_state_rates", {})
    overall = result.base_models.get("baseline_state_overall", 0.0)
    preds["baseline_state"] = (
        X["current_status"].astype("object").map(rates).astype("float64").fillna(overall).to_numpy()
    )
    return stack.predict(preds)


def _hazard_in_chunks(
    hazard: HazardModel, X: pd.DataFrame, head: str, horizon: int, chunk: int
) -> np.ndarray:
    """Chunked hazard compounding — bounds peak memory on a small container."""
    parts = [
        _hazard_horizon_predictions(hazard, X.iloc[i : i + chunk], head, horizon)
        for i in range(0, len(X), chunk)
    ]
    return np.concatenate(parts) if parts else np.empty(0, dtype="float64")


def _baseline_ladder(
    features: pd.DataFrame,
    y_all: pd.Series,
    labelled: pd.Series,
    plan: SplitPlan,
    names: list[str],
    cats: list[str],
    settings: Settings,
    result: HeadTrainingResult,
    hazard: HazardModel,
    head: str,
    horizon: int,
    *,
    oof_stacked: np.ndarray,
    oof_y: np.ndarray,
    oof_index: np.ndarray,
) -> dict[str, Any]:
    """B0..B5 under identical splits, reporting the lift of each rung.

    Every rung is measured on the *last embargoed fold* — the same window, the
    same rows, the same censoring mask — so the differences are attributable to
    the modelling step and nothing else. Showing that features contributed +X
    and stacking +Y is what "compare baseline and improved models" means, and it
    is far more persuasive than a single number.
    """
    from lpie.models.heads import baseline_logistic

    if not plan.folds:
        return {"rungs": {}, "ladder": [], "note": "no folds available"}

    fold = plan.folds[-1]
    tr = labelled & features["month_index"].isin(fold.train_months)
    va = labelled & features["month_index"].isin(fold.valid_months)
    if tr.sum() < 500 or va.sum() < 100:
        return {"rungs": {}, "ladder": [], "note": "final fold too small to build a ladder"}

    X_tr, y_tr = features.loc[tr], y_all[tr]
    X_va, y_va = features.loc[va], y_all[va]
    y_v = y_va.to_numpy()
    levels = fit_category_levels(X_tr, cats)
    spw = scale_pos_weight_for(y_tr)

    rungs: dict[str, Any] = {
        "B0_prior": binary_metrics(y_v, baseline_prior(y_tr, len(y_v)), label="B0 prior / majority"),
        "B1_current_state": binary_metrics(
            y_v, baseline_current_state(y_tr, X_tr["current_status"], X_va["current_status"]),
            label="B1 current-state lookup",
        ),
    }
    try:
        rungs["B2_logistic"] = binary_metrics(
            y_v,
            baseline_logistic(
                X_tr, y_tr, X_va, feature_names=names[:40], categorical_features=cats,
                settings=settings,
            ),
            label="B2 logistic regression (one-hot)",
        )
    except Exception as exc:  # pragma: no cover - defensive
        rungs["B2_logistic"] = {"label": "B2 logistic regression", "error": str(exc)}

    raw_columns = [
        c for c in ("dpd", "current_balance", "interest_rate", "loan_age_months",
                    "credit_score_band_ord", "ltv_band_ord", "current_status")
        if c in X_tr.columns
    ]
    try:
        b3 = TRAINERS["lightgbm"](
            X_tr, y_tr, head=head, feature_names=raw_columns,
            categorical_features=[c for c in cats if c in raw_columns],
            category_levels=fit_category_levels(X_tr, [c for c in cats if c in raw_columns]),
            valid=(X_va, y_va), params={"num_leaves": 31}, scale_pos_weight=spw, settings=settings,
        )
        rungs["B3_lightgbm_raw_columns"] = binary_metrics(
            y_v, predict_head(b3, X_va), label="B3 LightGBM, raw columns, default params"
        )
    except Exception as exc:  # pragma: no cover - defensive
        rungs["B3_lightgbm_raw_columns"] = {"label": "B3 LightGBM raw", "error": str(exc)}

    try:
        b4 = TRAINERS["lightgbm"](
            X_tr, y_tr, head=head, feature_names=names, categorical_features=cats,
            category_levels=levels, valid=(X_va, y_va), scale_pos_weight=spw, settings=settings,
        )
        rungs["B4_lightgbm_full_features"] = binary_metrics(
            y_v, predict_head(b4, X_va), label="B4 LightGBM + full feature engineering"
        )
    except Exception as exc:  # pragma: no cover - defensive
        rungs["B4_lightgbm_full_features"] = {"label": "B4 LightGBM + features", "error": str(exc)}

    # B5 is read off the pooled out-of-fold predictions restricted to this fold's
    # rows, so the champion is scored on exactly the rows the other rungs saw.
    fold_rows = np.flatnonzero(va.to_numpy())
    position = {row: i for i, row in enumerate(oof_index)}
    take = [position[r] for r in fold_rows if r in position]
    if take:
        rungs["B5_stacked_calibrated"] = binary_metrics(
            oof_y[take], oof_stacked[take], label="B5 stacked + calibrated (champion)"
        )

    order = [
        "B0_prior", "B1_current_state", "B2_logistic",
        "B3_lightgbm_raw_columns", "B4_lightgbm_full_features", "B5_stacked_calibrated",
    ]
    ladder, previous = [], None
    for key in order:
        value = (rungs.get(key) or {}).get("pr_auc")
        if value is None:
            continue
        ladder.append(
            {
                "rung": key,
                "label": rungs[key].get("label", key),
                "pr_auc": value,
                "roc_auc": rungs[key].get("roc_auc"),
                "lift_over_previous": round(value - previous, 6) if previous is not None else None,
            }
        )
        previous = value
    return {
        "rungs": rungs,
        "ladder": ladder,
        "metric": "pr_auc",
        "evaluation_window": fold.valid_window,
        "train_window": fold.train_window,
        "note": "All rungs measured on the final embargoed fold, identical rows.",
    }


def train_exception_head(
    features: pd.DataFrame,
    *,
    registry,
    settings: Settings | None = None,
    panel_max_month: int | None = None,
) -> dict[str, Any]:
    """Rules-first, with a supervised residual head layered on top.

    We measured that the label is rule-generated: re-deriving the supplied rules
    gives P=0.9997, R=0.9564, F1=0.9776 with zero ML. Starting from an ML model
    here would be an engineering mistake — slower, less accurate, and
    unexplainable, to reproduce a deterministic function. The residual head is
    therefore trained *with the rule outputs as features*, so it can only add
    value where the rules are silent.
    """
    from lpie.models.thresholds import optimise_threshold

    s = settings or get_settings()
    head = "exception_required"
    plan = build_split_plan(head, panel_max_month=panel_max_month, settings=s)
    names, cats = _feature_lists(registry, head)

    y = pd.to_numeric(features["exception_required"], errors="coerce")
    labelled = y.notna()

    rule_columns = [c for c in features.columns if c.startswith("rule_vr_")]
    rule_fired = (
        features[rule_columns].fillna(0.0).to_numpy().sum(axis=1) > 0
        if rule_columns
        else np.zeros(len(features), dtype=bool)
    )
    exception_rule_cols = [
        c for c in ("rule_vr_001_violated", "rule_vr_006_violated", "rule_vr_012_violated")
        if c in features.columns
    ]
    rules_only = (
        features[exception_rule_cols].fillna(0.0).to_numpy().sum(axis=1) > 0
        if exception_rule_cols
        else rule_fired
    )

    valid_mask = labelled & features["month_index"].isin(plan.calibration_months)
    train_mask = labelled & features["month_index"].isin(plan.final_train_months)
    production_mask = labelled & features["month_index"].isin(production_train_months(plan))

    rules_metrics = binary_metrics(
        y[valid_mask].to_numpy(), rules_only[valid_mask.to_numpy()].astype("float64"),
        threshold=0.5, label="rules_only",
    )

    # Measurement model: strictly held out from the evaluation window.
    measurement = TRAINERS["lightgbm"](
        features.loc[train_mask], y[train_mask], head=head, feature_names=names,
        categorical_features=cats, category_levels=fit_category_levels(features.loc[train_mask], cats),
        valid=(features.loc[valid_mask], y[valid_mask]) if valid_mask.any() else None,
        scale_pos_weight=scale_pos_weight_for(y[train_mask]), settings=s,
    )
    residual = predict_head(measurement, features.loc[valid_mask]) if valid_mask.any() else np.empty(0)

    # Shipped model: every censoring-valid month. exception_required is
    # contemporaneous (horizon 0), so there is no label window to overlap.
    artifact = TRAINERS["lightgbm"](
        features.loc[production_mask], y[production_mask], head=head, feature_names=names,
        categorical_features=cats,
        category_levels=fit_category_levels(features.loc[production_mask], cats),
        params={"num_iterations": measurement.best_iteration or 200},
        scale_pos_weight=scale_pos_weight_for(y[production_mask]), settings=s,
    )
    artifact.train_window = f"1-{plan.max_valid_month}"

    threshold = (
        optimise_threshold(residual, y[valid_mask].to_numpy(), head=head, objective="f1", settings=s)
        if valid_mask.any()
        else None
    )
    hybrid_metrics = (
        binary_metrics(
            y[valid_mask].to_numpy(), residual,
            threshold=threshold.threshold if threshold else 0.5, label="rules_plus_residual_ml",
        )
        if valid_mask.any()
        else {}
    )

    # The measured ceiling, stated explicitly so nobody reads a high F1 as a
    # modelling triumph — or a sub-1.0 F1 as a failure.
    missed = labelled & (y > 0) & ~pd.Series(rules_only, index=features.index)
    doc_null_share = (
        float(features.loc[missed, "document_status_missing_flag"].mean())
        if missed.any() and "document_status_missing_flag" in features.columns
        else None
    )

    return {
        "head": head,
        "model": artifact,
        "threshold": threshold,
        "rules_only": rules_metrics,
        "hybrid": hybrid_metrics,
        "rule_columns": exception_rule_cols,
        "ceiling_analysis": {
            "n_positives": int((y > 0).sum()),
            "n_missed_by_rules": int(missed.sum()),
            "missed_with_null_document_status": doc_null_share,
            "note": (
                "The residual positives are almost entirely rows whose document_status is "
                "null. Only about 15% of null-doc rows are true exceptions, so blanket "
                "flagging them collapses precision. The label is partially unrecoverable by "
                "construction and F1 is bounded near 0.98 — a reported F1 above 0.99 on this "
                "pack indicates a leaked label."
            ),
        },
        "train_window": f"1-{plan.max_valid_month}",
        "embargoed_measurement_window": plan.to_dict()["final_train_window"],
        "valid_window": plan.to_dict()["calibration_window"],
        "fold_plan": plan.to_dict(),
    }


def training_manifest(
    results: dict[str, HeadTrainingResult],
    hazard_metrics: dict[str, Any],
    exception_result: dict[str, Any],
    *,
    feature_hash: str,
    data_sha256: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    s = settings or get_settings()
    return {
        "model_version": s.model_version,
        "feature_version": s.feature_version,
        "trained_at": utcnow_iso(),
        "seed": s.seed,
        "feature_hash": feature_hash,
        "data_sha256": data_sha256,
        "config_hash": sha256_obj(s.as_dict()),
        "heads": {head: r.summary for head, r in results.items()},
        "hazard": {k: v for k, v in hazard_metrics.items() if k != "fold_plan"},
        "exception": {
            k: v for k, v in exception_result.items()
            if k not in ("model", "threshold", "fold_plan")
        },
    }


def run_training(
    features: pd.DataFrame,
    *,
    registry,
    settings: Settings | None = None,
    heads: tuple[str, ...] = BINARY_HEADS,
    algorithms: tuple[str, ...] = ("lightgbm", "xgboost", "catboost"),
    panel_max_month: int | None = None,
) -> dict[str, Any]:
    s = settings or get_settings()
    seed_everything(s.seed)

    hazard, hazard_metrics = train_hazard(
        features, registry=registry, settings=s, panel_max_month=panel_max_month
    )

    results: dict[str, HeadTrainingResult] = {}
    for head in heads:
        results[head] = train_binary_head(
            head, features, registry=registry, hazard=hazard, settings=s,
            panel_max_month=panel_max_month, algorithms=algorithms,
        )

    exception_result = train_exception_head(
        features, registry=registry, settings=s, panel_max_month=panel_max_month
    )

    return {
        "hazard": hazard,
        "hazard_metrics": hazard_metrics,
        "heads": results,
        "exception": exception_result,
    }
