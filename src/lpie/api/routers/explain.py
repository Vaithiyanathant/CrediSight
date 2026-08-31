"""Explainability endpoints."""
from __future__ import annotations

from typing import Annotated

import numpy as np
from fastapi import APIRouter, Query

from lpie.api.deps import PredictionDep
from lpie.api.metrics import METRICS
from lpie.api.schemas import (
    CounterfactualRequest,
    CounterfactualResponse,
    ErrorAnalysisResponse,
    GlobalExplainResponse,
    LocalExplainResponse,
)
from lpie.core.exceptions import DataNotFoundError, InvalidRequestError
from lpie.core.logging import get_logger
from lpie.core.timing import Timer
from lpie.serving.scorer import build_features_for

log = get_logger(__name__)
router = APIRouter(prefix="/api/v1/explain", tags=["explainability"])

VALID_HEADS = {"next_3m_delinquency","next_6m_delinquency","next_12m_default","next_12m_prepayment","next_state","exception_required"}


def _shap_capable_model(state, head: str):
    """The single-output tree model for `head`, or None if TreeSHAP cannot apply.

    Heads are not all stored in `state.heads`: `exception_required` is the
    residual-ML model inside exception.joblib, and `next_state` is the 7-class
    hazard softmax, which has no single-output attribution.
    """
    artifact = (state.heads.get(head) or {}).get("lightgbm")
    if artifact is not None:
        return artifact
    if head == "exception_required":
        residual = getattr(state, "exception_head", None)
        residual = residual.get("artifact", residual) if isinstance(residual, dict) else residual
        return getattr(residual, "model", None)
    return None


def _global_response(head, state, shap_data, top_k, timer) -> GlobalExplainResponse:
    METRICS.increment("lpie_explain_global_requests_total")
    return GlobalExplainResponse(
        head=head,
        model_version=state.settings.model_version,
        n_explained=shap_data.get("n_explained", 0),
        global_importance=shap_data.get("importance", shap_data.get("global_importance", []))[:top_k],
        family_attribution=shap_data.get("families", shap_data.get("family_attribution", [])),
        permutation_importance=shap_data.get("permutation_importance", []),
        monotonicity_audit=shap_data.get("monotonicity_audit", []),
        interpretation=shap_data.get(
            "interpretation",
            f"Top {top_k} features by mean |SHAP| for head `{head}`. "
            "Family attribution groups the feature set into its registered families.",
        ),
        elapsed_ms=round(timer.stop(), 2),
    )


@router.get(
    "/global",
    response_model=GlobalExplainResponse,
    summary="SHAP global importance, family attribution, permutation importance",
)
def global_explain(
    state: PredictionDep,
    head: Annotated[str, Query()] = "next_12m_default",
    top_k: Annotated[int, Query(ge=5, le=50)] = 20,
) -> GlobalExplainResponse:
    timer = Timer()
    if head not in VALID_HEADS:
        from lpie.core.exceptions import InvalidRequestError
        raise InvalidRequestError(f"Unknown head: {head}. Valid: {sorted(VALID_HEADS)}")

    shap_data = state.shap_global.get(head, {})
    # Handle wrapped artifact structure: {'artifact': {head: {...}}, 'metadata': {...}}
    if not shap_data and isinstance(state.shap_global.get("artifact"), dict):
        shap_data = state.shap_global["artifact"].get(head, {})
    if not shap_data:
        store_months = state.feature_store_months()
        if not store_months:
            raise DataNotFoundError("Feature store is empty; cannot compute SHAP")
        features = build_features_for(state, months=[max(store_months)])
        if features.empty:
            raise DataNotFoundError("No features available for global explain")
        artifact = _shap_capable_model(state, head)
        if artifact is None:
            # `exception_required` lives in exception.joblib and `next_state` in
            # the multiclass hazard core, so neither is in `state.heads`. Both
            # used to reach here and raise PredictionError -> 500 for a head the
            # endpoint's own validator advertises as valid. An unavailable
            # explanation is an empty explanation with a reason, not a server error.
            log.info("explain.global_unavailable", head=head)
            shap_data = {
                "importance": [],
                "families": [],
                "interpretation": (
                    f"No precomputed SHAP artifact exists for '{head}', and this head has no "
                    "single-output tree model that TreeSHAP can attribute directly "
                    "(`next_state` is a 7-class softmax; per-class attribution is served by "
                    "the local endpoint). Run `make train` to regenerate SHAP artifacts."
                ),
            }
            return _global_response(head, state, shap_data, top_k, timer)
        from lpie.explain.shap_global import (
            family_attribution,
            global_importance,
            tree_shap_values,
        )
        try:
            values, prepared = tree_shap_values(artifact, features, max_rows=5000)
            importance = global_importance(values, artifact.feature_names, top_k=top_k)
            families = family_attribution(values, artifact.feature_names, state.registry)
            shap_data = {"importance": importance, "families": families}
        except Exception as exc:
            log.warning("explain.global_shap_error", error=str(exc))
            shap_data = {"importance": [], "families": []}

    return _global_response(head, state, shap_data, top_k, timer)


@router.get(
    "/errors",
    response_model=ErrorAnalysisResponse,
    summary="FP/FN analysis, error slices, segment metrics",
)
def error_analysis(
    state: PredictionDep,
    head: Annotated[str, Query()] = "next_12m_default",
) -> ErrorAnalysisResponse:
    timer = Timer()
    eval_data = state.evaluation.get("error_analysis", {}).get(head, {})
    threshold = 0.5
    thresholds = state.thresholds
    if thresholds and head in thresholds:
        threshold = float(getattr(thresholds[head], "threshold", threshold))

    return ErrorAnalysisResponse(
        head=head,
        threshold=threshold,
        confusion_profile=eval_data.get("confusion_profile", {}),
        error_slices=eval_data.get("error_slices", {}),
        segment_performance=eval_data.get("segment_performance", {}),
        calibration_by_segment=eval_data.get("calibration_by_segment", []),
        fairness=eval_data.get("fairness", {}),
        elapsed_ms=round(timer.stop(), 2),
    )


@router.post(
    "/counterfactual",
    response_model=CounterfactualResponse,
    summary="Minimal actionable change to flip the outcome",
)
def counterfactual(request: CounterfactualRequest, state: PredictionDep) -> CounterfactualResponse:
    timer = Timer()
    features = build_features_for(state, loan_ids=[request.loan_id], months=[request.month_index])
    if features.empty:
        raise DataNotFoundError(f"No features for {request.loan_id} at month {request.month_index}")

    artifact = _shap_capable_model(state, request.head)
    if artifact is None:
        # `next_state` is a multiclass softmax and has no single scalar
        # probability to drive toward a target, so a counterfactual search over
        # it is undefined. That is a bad request, not a server fault.
        raise InvalidRequestError(
            f"Counterfactual search is not defined for head '{request.head}'. "
            "Supported heads: next_3m_delinquency, next_6m_delinquency, "
            "next_12m_default, next_12m_prepayment, exception_required."
        )

    from lpie.explain.counterfactual import actionable_features, search
    from lpie.models.heads import predict_head

    original_prob = None
    found = False
    cf = None
    alternatives = []
    n_evaluated = 0
    reason = None
    try:
        # Build a predict_fn wrapper around the artifact so search() can call it
        # with modified feature DataFrames
        def _predict_fn(X):
            """Predict for a single row or a batch. Always returns ndarray."""
            import numpy as np
            p = predict_head(artifact, X)
            return np.asarray(p, dtype="float64")

        # Use the latest month features as the reference population
        latest_months = state.feature_store_months()
        reference = build_features_for(
            state, months=[max(latest_months)] if latest_months else [request.month_index]
        ) if latest_months else features
        if reference.empty:
            reference = features

        result = search(
            _predict_fn, features.iloc[[0]], reference,
            max_changes=request.max_changes,
            target_probability=request.target_probability,
            settings=state.settings,
        )
        original_prob = result.get("original_probability")
        found = result.get("found", False)
        cf = result.get("counterfactual")
        alternatives = result.get("alternatives", [])
        n_evaluated = result.get("n_evaluated", 0)
        reason = result.get("reason")
    except Exception as exc:
        reason = str(exc)
        get_logger(__name__).warning("counterfactual.error", error=str(exc))

    actionable = actionable_features(state.settings)
    return CounterfactualResponse(
        loan_id=request.loan_id,
        month_index=request.month_index,
        head=request.head,
        found=found,
        original_probability=original_prob,
        target_probability=request.target_probability,
        counterfactual=cf,
        alternatives=alternatives[:3],
        actionable_features=actionable,
        forbidden_features=[],
        narrative=None,
        governance="Counterfactual is for decision support only. Not for adverse action.",
        reason=reason,
        n_evaluated=n_evaluated,
        elapsed_ms=round(timer.stop(), 2),
    )


@router.get(
    "/{loan_id}",
    response_model=LocalExplainResponse,
    summary="SHAP waterfall, narrative, and conformal interval for one loan",
)
def local_explain(
    loan_id: str,
    state: PredictionDep,
    head: Annotated[str, Query()] = "next_12m_default",
    month_index: Annotated[int | None, Query(ge=1, le=600)] = None,
) -> LocalExplainResponse:
    timer = Timer()
    if head not in VALID_HEADS:
        raise InvalidRequestError(f"Unknown head: {head}. Valid: {sorted(VALID_HEADS)}")
    store_months = state.feature_store_months()
    if not store_months:
        raise DataNotFoundError("Feature store is empty")
    month = month_index or max(store_months)
    features = build_features_for(state, loan_ids=[loan_id], months=[month])
    if features.empty:
        raise DataNotFoundError(f"No features for {loan_id} at month {month}")

    row = features.iloc[[0]]
    artifact = _shap_capable_model(state, head)
    if artifact is None:
        # Only `next_state` reaches this: a 7-class softmax has no single
        # probability for a waterfall to decompose.
        raise InvalidRequestError(
            f"Local SHAP is not defined for head '{head}' (multiclass). "
            "Use /api/v1/predict for its per-class next-state distribution."
        )

    from lpie.explain.shap_local import local_shap, narrative
    shap_result = local_shap(artifact, row)
    prob = float(shap_result.get("predicted_probability", 0.0))
    narr = narrative(shap_result.get("contributions", []), prob, head=head)

    conformal_interval = None
    row["current_status"].iloc[0] if "current_status" in row.columns else "Current"
    conformal = state.conformal.get(head)
    if conformal is not None:
        try:
            from lpie.models.calibration import segment_labels
            segs = segment_labels(row.get("vintage_year_num"), row.get("credit_score_band"), row.index)
            lo, hi = conformal.interval(np.array([prob]), segs)
            conformal_interval = [float(lo[0]), float(hi[0])]
        except Exception:
            pass

    METRICS.increment("lpie_explain_local_requests_total")
    return LocalExplainResponse(
        loan_id=loan_id,
        month_index=month,
        head=head,
        model_version=state.settings.model_version,
        probability=prob,
        base_value=float(shap_result.get("base_value", 0.0)),
        top_contributions=shap_result.get("top_contributions", shap_result.get("contributions", [])),
        narrative=narr,
        peer_comparison=[],
        history_strip={},
        conformal_interval=conformal_interval,
        confidence={"conformal_width": (conformal_interval[1] - conformal_interval[0]) if conformal_interval else None},
        semantics={"model": "LightGBM", "head": head, "method": "TreeSHAP (exact)"},
        elapsed_ms=round(timer.stop(), 2),
    )


