"""Data profiling endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from lpie.api.deps import StateDep
from lpie.api.metrics import METRICS
from lpie.api.schemas import ProfileRequest, ProfileResponse
from lpie.core.exceptions import DataNotFoundError
from lpie.core.timing import Timer

router = APIRouter(prefix="/api/v1", tags=["data intelligence"])


@router.post(
    "/profile",
    response_model=ProfileResponse,
    summary="Four-stage data profile",
    description=(
        "Structural, distributional, missingness and relationship profiling over the "
        "requested slice.\n\n"
        "**Stage A** dtypes, cardinality, null counts and rates, top-k frequencies, "
        "constant and degenerate detection.\n"
        "**Stage B** mean/std/skew/kurtosis, percentiles, zero-inflation, histograms; "
        "entropy, Gini and rare-level counts for categoricals.\n"
        "**Stage C** missingness co-occurrence and MCAR/MAR triage — for each null-bearing "
        "column a classifier predicts `is_null` from the rest; AUC near 0.5 means MCAR, "
        "well above means MAR with the drivers named.\n"
        "**Stage D** correlation, mutual information, Cramér's V, correlation ratio, "
        "functional-dependency mining, and temporal integrity — which is where the "
        "duplicate `(loan_id, reporting_month)` corruption surfaces.\n\n"
        "The state machine and the censoring cliffs are re-derived from the data at "
        "request time, never read from configuration."
    ),
    responses={404: {"description": "No rows matched the requested slice"}},
)
def profile(request: ProfileRequest, state: StateDep) -> ProfileResponse:
    from lpie.profiling.profiler import profile_frame

    timer = Timer()
    panel = state.panel()
    train_max = int(state.settings.get("data.train_month_max", 36))

    if request.split == "train":
        frame = panel[panel["month_index"] <= train_max]
    elif request.split == "test":
        frame = panel[panel["month_index"] > train_max]
    else:
        frame = panel

    if request.months:
        frame = frame[frame["month_index"].isin(request.months)]
    if request.columns:
        keep = [c for c in request.columns if c in frame.columns]
        if not keep:
            raise DataNotFoundError(
                "None of the requested columns exist in the panel",
                details={"requested": request.columns, "available": sorted(frame.columns)[:40]},
            )
        frame = frame[keep]

    if frame.empty:
        raise DataNotFoundError(
            "No rows matched the requested slice",
            details={"split": request.split, "months": request.months},
        )

    report = profile_frame(
        frame,
        name=f"{request.split}_panel",
        top_k=request.top_k,
        sample_rows=request.sample_rows,
        include_relationships=request.include_relationships,
        include_missingness=request.include_missingness,
        settings=state.settings,
    )
    METRICS.increment("lpie_profile_requests_total")
    report["schema"] = report.pop("schema")
    report["elapsed_ms"] = round(timer.stop(), 2)
    return ProfileResponse(**report)
