"""Validation engine endpoint — all 18 rules, DQ scores, batch verdict."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter

from lpie.api.deps import StateDep
from lpie.api.metrics import METRICS
from lpie.api.schemas import ValidationRequest, ValidationResponse
from lpie.core.exceptions import DataNotFoundError, InvalidRequestError
from lpie.core.timing import Timer

router = APIRouter(prefix="/api/v1", tags=["data intelligence"])


@router.post(
    "/validate",
    response_model=ValidationResponse,
    summary="Run the 18-rule validation engine",
    description=(
        "Evaluates every rule in `config/validation_rules.json` — the 12 supplied with the "
        "data pack plus 6 derived from profiling (VR-013 duplicate loan-month, VR-014 "
        "balance monotonicity, VR-015 terminal-state persistence, VR-016 age/term identity, "
        "VR-017 static-join consistency, VR-018 unseen category).\n\n"
        "Returns per-record rule results, the violation ledger with observed-versus-expected "
        "values, six-dimension record DQ scores, and the batch score.\n\n"
        "The DQ score is a weighted product, not a mean: "
        "`100 x PROD_d (1 - w_d * penalty_d)`. One ERROR cannot be averaged away by many "
        "passes. Grades: A>=95, B>=85, C>=70, D>=50, F<50.\n\n"
        "Supply `records` to validate arbitrary rows, or a slice selector to validate stored "
        "data. Rules needing history (VR-014, VR-015) use the stored panel for prior months, "
        "so a single-month batch is still judged correctly."
    ),
    responses={
        400: {"description": "Both `records` and a slice selector were supplied"},
        404: {"description": "No rows matched the requested slice"},
    },
)
def validate(request: ValidationRequest, state: StateDep) -> ValidationResponse:
    timer = Timer()
    engine = state.validation_engine
    panel = state.panel()

    if request.records is not None:
        if len(request.records) > request.limit:
            raise InvalidRequestError(
                f"{len(request.records)} records exceeds the limit of {request.limit}",
                details={"n_records": len(request.records), "limit": request.limit},
            )
        frame = pd.DataFrame(request.records)
        if frame.empty:
            raise InvalidRequestError("`records` is empty")
        for column in ("loan_id", "month_index"):
            if column not in frame.columns:
                raise InvalidRequestError(
                    f"Supplied records must include `{column}`",
                    details={"missing": column, "supplied": sorted(frame.columns)},
                )
        frame["month_index"] = pd.to_numeric(frame["month_index"], errors="coerce")
        history = panel[panel["loan_id"].isin(set(frame["loan_id"].astype(str)))]
    else:
        train_max = int(state.settings.get("data.train_month_max", 36))
        frame = (
            panel[panel["month_index"] <= train_max] if request.split == "train"
            else panel[panel["month_index"] > train_max] if request.split == "test"
            else panel
        )
        if request.months:
            frame = frame[frame["month_index"].isin(request.months)]
        if request.loan_ids:
            frame = frame[frame["loan_id"].isin(request.loan_ids)]
        if frame.empty:
            raise DataNotFoundError(
                "No rows matched the requested slice",
                details={"split": request.split, "months": request.months,
                         "loan_ids": (request.loan_ids or [])[:5]},
            )
        frame = frame.head(request.limit)
        history = panel

    if not engine.vocabulary:
        train_max = int(state.settings.get("data.train_month_max", 36))
        engine.fit_vocabulary(panel[panel["month_index"] <= train_max])

    result = engine.run(
        frame,
        static=state.static(),
        servicer=state.servicer(),
        history=history,
        batch_id=request.batch_id,
        max_violation_rows=request.max_violations,
    )

    METRICS.increment("lpie_validation_requests_total")
    METRICS.increment("lpie_validation_rows_total", float(len(frame)))

    record_results = (
        result.record_results.to_dict(orient="records") if request.include_records else []
    )
    return ValidationResponse(
        record_results=record_results,
        violations=result.violations.replace({float("nan"): None}).to_dict(orient="records"),
        dq_score=result.record_scores.head(request.limit).to_dict(orient="records"),
        batch_score=result.batch_score,
        summary=result.summary,
        elapsed_ms=round(timer.stop(), 2),
    )
