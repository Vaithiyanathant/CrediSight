"""Prediction endpoints."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from fastapi import APIRouter

from lpie.api.deps import DemoDep, PredictionDep
from lpie.api.metrics import METRICS
from lpie.api.schemas import (
    AnomalyBlock,
    ConfidenceBlock,
    DataQualityBlock,
    DriverBlock,
    ExceptionBlock,
    ExplanationBlock,
    NextStatePrediction,
    PredictionBundle,
    PredictionRequest,
    PredictionResponse,
    Predictions,
    PredictionValue,
    SurvivalBlock,
)
from lpie.core.exceptions import DataNotFoundError, InvalidRequestError, PredictionError
from lpie.core.logging import get_logger
from lpie.core.timing import Timer
from lpie.serving.scorer import PredictionScorer, build_features_for, build_features_online

log = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["prediction"])
STATES = ("Current", "30DPD", "60DPD", "90DPD", "Default", "Prepaid", "Closed")
VALID_ACTIONS = {"No Action", "Flag", "Escalate"}
VALID_STATES = set(STATES)


def _sf(v, d=0.0):
    try:
        f = float(v)
        return d if not np.isfinite(f) else f
    except (TypeError, ValueError):
        return d


def _si(v, d=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return d


def _ss(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v)


def _coerce_action(v):
    s = _ss(v)
    return s if s in VALID_ACTIONS else "No Action"


def _coerce_state(v):
    s = _ss(v)
    return s if s in VALID_STATES else "Current"


def _cache_key(loan_ids, months):
    import hashlib
    payload = json.dumps({"l": sorted(loan_ids), "m": sorted(months)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _pval(row, col):
    prob = max(0.0, min(1.0, _sf(row.get(col), 0.0)))
    ci = None
    lo, hi = row.get(f"{col}_ci_low"), row.get(f"{col}_ci_high")
    if lo is not None and hi is not None:
        try:
            ci = [float(lo), float(hi)]
        except (TypeError, ValueError):
            pass
    return PredictionValue(
        value=prob, ci=ci,
        calibrated=bool(row.get(f"{col}_calibrated", True)),
        raw=_sf(row.get(f"{col}_raw")) or None,
        ensemble_disagreement=_sf(row.get(f"{col}_disagreement")) or None,
    )


def _next_state(row):
    raw = row.get("next_state_probs")
    probs: dict[str, float] = {}
    if isinstance(raw, str):
        try:
            probs = {k: float(v) for k, v in json.loads(raw).items()}
        except Exception:
            pass
    elif isinstance(raw, dict):
        probs = {k: float(v) for k, v in raw.items()}
    if not probs:
        for s in STATES:
            col = f"p_next_{s}"
            if col in row and row[col] is not None:
                probs[s] = _sf(row[col], 0.0)
    conf = _sf(row.get("next_state_confidence"), 0.0)
    return NextStatePrediction(
        predicted=_coerce_state(row.get("predicted_next_state")),
        probs=probs,
        confidence=conf,
        legal_mask_applied=True,
    )


def _row_to_bundle(row, survival_attrs=None):
    idx = int(row.get("_survival_idx", -1))
    surv = None
    if survival_attrs and idx >= 0:
        try:
            h = list(survival_attrs.get("horizons", []))
            def _arr(k):
                a = survival_attrs.get(k)
                return [_sf(x) for x in np.asarray(a)[idx].tolist()] if a is not None else []
            surv = SurvivalBlock(
                horizons_m=h,
                survival=_arr("survival"),
                cif_default=_arr("cif_default"),
                cif_prepay=_arr("cif_prepaid"),
                cif_closed=_arr("cif_closed"),
                conservation_max_error=_sf(survival_attrs.get("conservation_max_error"), 0.0),
            )
        except Exception:
            pass

    drivers = [DriverBlock(feature=_ss(row.get(f"top_driver_{i}"))) for i in range(1, 4) if _ss(row.get(f"top_driver_{i}"))]

    return PredictionBundle(
        loan_id=_ss(row.get("loan_id")),
        reporting_month=_ss(row.get("reporting_month")) or None,
        month_index=_si(row.get("month_index"), 0),
        model_version=_ss(row.get("model_version")) or "unknown",
        feature_version=_ss(row.get("feature_version")) or "unknown",
        scored_at=_ss(row.get("scored_at")) or "",
        current_status=_coerce_state(row.get("current_status")),
        current_balance=_sf(row.get("current_balance"), 0.0),
        is_terminal=bool(row.get("is_terminal", False)),
        gated_by_rule=_ss(row.get("gated_by_rule")) or None,
        calibration_segment=_ss(row.get("calibration_segment")) or None,
        predictions=Predictions(
            prob_next_3m_delinquency=_pval(row, "prob_next_3m_delinquency"),
            prob_next_6m_delinquency=_pval(row, "prob_next_6m_delinquency"),
            prob_next_12m_default=_pval(row, "prob_next_12m_default"),
            prob_next_12m_prepayment=_pval(row, "prob_next_12m_prepayment"),
            next_state=_next_state(row),
        ),
        survival=surv,
        anomaly=AnomalyBlock(
            score=max(0.0, min(1.0, _sf(row.get("anomaly_score"), 0.0))),
            tier=_ss(row.get("anomaly_tier")) or "unsupervised",
            rule_severity=_sf(row.get("rule_severity"), 0.0),
            worst_severity=_ss(row.get("worst_severity")) or "NONE",
            drivers=[d for d in [_ss(row.get(f"top_driver_{i}")) for i in range(1, 4)] if d],
        ),
        exception=ExceptionBlock(
            required=_si(row.get("exception_required"), 0),
            type=_ss(row.get("exception_type")) or "none",
            source=_ss(row.get("exception_source")) or "rule",
        ),
        explanation=ExplanationBlock(top_drivers=drivers),
        confidence=ConfidenceBlock(
            model_confidence=max(0.0, min(1.0, _sf(row.get("model_confidence"), 0.5))),
            conformal_width=_sf(row.get("conformal_width")) or None,
            ensemble_disagreement=_sf(row.get("ensemble_disagreement")) or None,
            segment_ece=_sf(row.get("segment_ece")) or None,
            data_quality=_sf(row.get("data_quality_term")) or None,
        ),
        expected_loss=_sf(row.get("expected_loss"), 0.0),
        reviewer_action=_coerce_action(row.get("reviewer_action")),
        data_quality=DataQualityBlock(
            dq_score=_sf(row.get("dq_score")) or None,
            dq_grade=_ss(row.get("dq_grade")) or None,
        ),
    )


def _frame_to_bundles(result_frame, survival_attrs=None):
    frame = result_frame.where(pd.notnull(result_frame), other=None)
    bundles = []
    for idx, (_, row) in enumerate(frame.iterrows()):
        d = row.to_dict()
        d["_survival_idx"] = idx
        try:
            bundles.append(_row_to_bundle(d, survival_attrs))
        except Exception as exc:
            log.warning("prediction.bundle_build_error", loan_id=d.get("loan_id"), error=str(exc))
    return bundles


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Batch scoring",
)
def predict_batch(request: PredictionRequest, state: PredictionDep, demo: DemoDep) -> PredictionResponse:
    timer = Timer()
    scorer = PredictionScorer(state)
    if request.records is not None:
        if not request.records:
            raise InvalidRequestError("`records` list is empty")
        frame_rows = pd.DataFrame(request.records)
        for col in ("loan_id", "month_index"):
            if col not in frame_rows.columns:
                raise InvalidRequestError(f"Supplied records must include `{col}`")
        frame_rows["month_index"] = pd.to_numeric(frame_rows["month_index"], errors="coerce")
        try:
            features = build_features_online(state, frame_rows)
        except DataNotFoundError:
            raise
        except Exception as exc:
            raise PredictionError(f"Online feature build failed: {exc}") from exc
    else:
        months = request.months
        loan_ids = request.loan_ids
        if months is None:
            # Default to the latest available month whether or not loan_ids are given.
            # Scoring across all 42 months is expensive and rarely what the caller wants;
            # the API contract is: omit months -> score the latest snapshot.
            store_months = state.feature_store_months()
            if not store_months:
                raise DataNotFoundError("Feature store is empty")
            months = [max(store_months)]
        try:
            features = build_features_for(state, loan_ids=list(loan_ids) if loan_ids else None, months=months)
        except DataNotFoundError:
            raise
        except Exception as exc:
            raise PredictionError(f"Feature store read failed: {exc}") from exc
    if features.empty:
        raise DataNotFoundError("No features found for the requested rows")
    max_rows = int(state.settings.get("api.max_predict_rows", 5000))
    if len(features) > max_rows:
        features = features.head(max_rows)
    try:
        result = scorer.score(
            features,
            include_survival=request.include_survival,
            survival_horizon=request.survival_horizon,
            include_drivers=request.include_drivers,
        )
    except (DataNotFoundError, PredictionError):
        raise
    except Exception as exc:
        raise PredictionError(f"Scoring failed: {exc}") from exc
    survival_attrs = result.frame.attrs.get("survival")
    bundles = _frame_to_bundles(result.frame, survival_attrs)
    METRICS.increment("lpie_predictions_total", len(bundles))
    METRICS.observe("lpie_scoring_duration_ms", result.elapsed_ms)
    log.info("predict.batch", n_rows=len(bundles), elapsed_ms=round(timer.stop(), 2))
    return PredictionResponse(
        predictions=bundles,
        n_rows=len(bundles),
        model_version=result.model_version,
        feature_version=result.feature_version,
        scored_at=result.scored_at,
        degraded_components=result.degraded,
        elapsed_ms=round(timer.stop(), 2),
    )


@router.get(
    "/predict/{loan_id}",
    response_model=PredictionBundle,
    summary="Latest prediction bundle for one loan",
)
def predict_single(loan_id: str, state: PredictionDep) -> PredictionBundle:
    timer = Timer()
    store_months = state.feature_store_months()
    if not store_months:
        raise DataNotFoundError(f"Feature store is empty; cannot score loan {loan_id}")
    latest = max(store_months)
    try:
        features = build_features_for(state, loan_ids=[loan_id], months=[latest])
    except DataNotFoundError as exc:
        raise DataNotFoundError(f"No features found for loan {loan_id} at month {latest}") from exc
    except Exception as exc:
        raise PredictionError(f"Feature store read failed for {loan_id}: {exc}") from exc
    if features.empty:
        raise DataNotFoundError(f"No features found for loan {loan_id}")
    scorer = PredictionScorer(state)
    try:
        result = scorer.score(features, include_survival=False)
    except Exception as exc:
        raise PredictionError(f"Scoring failed for {loan_id}: {exc}") from exc
    bundles = _frame_to_bundles(result.frame)
    METRICS.increment("lpie_predictions_total", len(bundles))
    matched = [b for b in bundles if b.loan_id == loan_id]
    if not matched:
        raise DataNotFoundError(f"Loan {loan_id} not found in scored output")
    log.info("predict.single", loan_id=loan_id, elapsed_ms=round(timer.stop(), 2))
    return matched[0]
