"""Survival analysis endpoints."""
from __future__ import annotations

from typing import Annotated

import numpy as np
from fastapi import APIRouter, Query

from lpie.api.deps import PredictionDep
from lpie.api.metrics import METRICS
from lpie.api.schemas import (
    SegmentSurvivalRequest,
    SegmentSurvivalResponse,
    StateOccupancyResponse,
    SurvivalResponse,
)
from lpie.core.exceptions import DataNotFoundError
from lpie.core.logging import get_logger
from lpie.core.timing import Timer
from lpie.serving.scorer import build_features_for

log = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["survival"])


def _sf(v, d=0.0):
    try:
        f = float(v)
        return d if not np.isfinite(f) else f
    except (TypeError, ValueError):
        return d


def _get_loan_features(state, loan_id: str):
    store_months = state.feature_store_months()
    if not store_months:
        raise DataNotFoundError("Feature store is empty")
    features = build_features_for(state, loan_ids=[loan_id], months=[max(store_months)])
    if features.empty:
        raise DataNotFoundError(f"No features found for loan {loan_id}")
    return features


@router.post(
    "/survival/segment",
    response_model=SegmentSurvivalResponse,
    summary="KM + model curves by segment",
)
def survival_segment(request: SegmentSurvivalRequest, state: PredictionDep) -> SegmentSurvivalResponse:
    timer = Timer()
    store_months = state.feature_store_months()
    if not store_months:
        raise DataNotFoundError("Feature store is empty")
    features = build_features_for(state, months=[max(store_months)])
    if features.empty:
        raise DataNotFoundError("No features in store")

    seg_col = request.segment_by
    if seg_col not in features.columns:
        seg_col = "current_status"

    hazard = state.hazard
    model_curves = []
    groups = features.groupby(seg_col, dropna=False)
    for seg_val, grp in groups:
        if len(grp) < 5:
            continue
        if request.values and str(seg_val) not in request.values:
            continue
        grp = grp.head(request.max_loans)
        try:
            M = hazard.transition_matrices(grp)
            starts = grp["current_status"].map(hazard.state_index).fillna(0).astype("int32").to_numpy()
            prop = hazard.propagate(M, starts, request.horizon)
            s_arr = np.asarray(prop.get("survival", np.ones((len(grp), request.horizon))))
            mean_s = s_arr.mean(axis=0).tolist()
            model_curves.append({
                "segment": str(seg_val),
                "n_loans": len(grp),
                "survival": [_sf(x) for x in mean_s],
                "horizons_m": list(range(1, request.horizon + 1)),
            })
        except Exception as exc:
            log.warning("survival.segment_error", segment=str(seg_val), error=str(exc))

    return SegmentSurvivalResponse(
        segment_by=request.segment_by,
        horizon=request.horizon,
        n_loans=len(features),
        model_curves=model_curves,
        elapsed_ms=round(timer.stop(), 2),
    )


@router.get(
    "/survival/state-occupancy",
    response_model=StateOccupancyResponse,
    summary="24-month stacked state-occupancy projection",
)
def state_occupancy(
    state: PredictionDep,
    horizon: Annotated[int, Query(ge=1, le=60)] = 24,
    max_loans: Annotated[int, Query(ge=100, le=20000)] = 10000,
) -> StateOccupancyResponse:
    timer = Timer()
    store_months = state.feature_store_months()
    if not store_months:
        raise DataNotFoundError("Feature store is empty")
    latest = max(store_months)
    features = build_features_for(state, months=[latest])
    if features.empty:
        raise DataNotFoundError("No features in store")
    if len(features) > max_loans:
        features = features.sample(max_loans, random_state=state.settings.seed)

    hazard = state.hazard
    states = hazard.states
    M = hazard.transition_matrices(features)
    start = features["current_status"].map(hazard.state_index).fillna(0).astype("int32").to_numpy()
    # occupancy shape: (n_loans, horizon+1, K)
    occ_arr = hazard.state_occupancy(M, start, horizon=horizon)
    # mean across loans: (horizon+1, K)
    mean_occ = occ_arr.mean(axis=0)  # (horizon+1, K)
    mean_share = [[_sf(mean_occ[m, k]) for k in range(len(states))] for m in range(1, horizon + 1)]

    return StateOccupancyResponse(
        as_of_month_index=latest,
        horizon=horizon,
        states=states,
        months=list(range(1, horizon + 1)),
        mean_share=mean_share,
        n_loans=len(features),
        model_version=state.settings.model_version,
        elapsed_ms=round(timer.stop(), 2),
    )
@router.get(
    "/survival/{loan_id}",
    response_model=SurvivalResponse,
    summary="Survival curve and CIFs for one loan",
)
def survival_loan(
    loan_id: str,
    state: PredictionDep,
    horizon: Annotated[int, Query(ge=1, le=60)] = 24,
) -> SurvivalResponse:
    timer = Timer()
    features = _get_loan_features(state, loan_id)
    row = features.iloc[[0]]
    hazard = state.hazard
    status = row["current_status"].iloc[0]
    is_terminal = status in ("Prepaid", "Closed")

    if is_terminal:
        horizons = list(range(1, horizon + 1))
        n = len(horizons)
        return SurvivalResponse(
            loan_id=loan_id,
            month_index=int(row["month_index"].iloc[0]),
            current_status=status,
            model_version=state.settings.model_version,
            horizons_m=horizons,
            survival=[1.0 if status == "Prepaid" else 0.0] * n,
            cif_default=[0.0] * n,
            cif_prepay=[1.0 if status == "Prepaid" else 0.0] * n,
            cif_closed=[1.0 if status == "Closed" else 0.0] * n,
            conservation_max_error=0.0,
            conservation_holds=True,
            is_terminal=True,
            gated_by_rule="VR-015",
            elapsed_ms=round(timer.stop(), 2),
        )

    M = hazard.transition_matrices(row)
    start_idx = hazard.state_index.get(status, 0)
    start = np.array([start_idx], dtype="int32")
    propagated = hazard.propagate(M, start, horizon)
    horizons = list(range(1, horizon + 1))

    def _extract(key, idx):
        arr = propagated.get(key)
        if arr is None:
            return [0.0] * len(horizons)
        a = np.asarray(arr)
        if a.ndim == 2:
            return [_sf(a[0, i]) for i in range(len(horizons))]
        return [_sf(a[i]) for i in range(len(horizons))]

    s_vals = _extract("survival", 0)
    d_vals = _extract("cif_default", 0)
    pp_vals = _extract("cif_prepaid", 0)
    cl_vals = _extract("cif_closed", 0)
    err = _sf(propagated.get("conservation_max_error"), 0.0)

    METRICS.increment("lpie_survival_requests_total")
    return SurvivalResponse(
        loan_id=loan_id,
        month_index=int(row["month_index"].iloc[0]),
        current_status=status,
        model_version=state.settings.model_version,
        horizons_m=horizons,
        survival=s_vals,
        cif_default=d_vals,
        cif_prepay=pp_vals,
        cif_closed=cl_vals,
        conservation_max_error=err,
        conservation_holds=err < 1e-4,
        is_terminal=False,
        gated_by_rule=None,
        elapsed_ms=round(timer.stop(), 2),
    )


