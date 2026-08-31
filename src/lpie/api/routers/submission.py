"""Submission generation and validation endpoints."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from lpie.api.deps import PredictionDep, StateDep
from lpie.api.schemas import SubmissionGenerateRequest, SubmissionResponse, SubmissionValidateResponse
from lpie.core.exceptions import DataNotFoundError, PredictionError
from lpie.core.logging import get_logger
from lpie.core.timing import Timer
from lpie.submit.validator import read_submission_csv

log = get_logger(__name__)
router = APIRouter(prefix="/api/v1/submission", tags=["submission"])


@router.post(
    "/generate",
    response_model=SubmissionResponse,
    summary="Build submission.csv from the latest predictions",
)
def generate_submission(request: SubmissionGenerateRequest, state: PredictionDep) -> SubmissionResponse:
    timer = Timer()
    from lpie.pipelines.submit import stage_submit
    try:
        result = stage_submit(
            state.settings,
            as_of_month=request.as_of_month,
        )
    except Exception as exc:
        raise PredictionError(f"Submission generation failed: {exc}") from exc

    path = result.get("outputs", {}).get("submission") or str(state.settings.get("paths.submission_path", ""))
    n_rows = int(result.get("n_rows", result.get("row_counts", {}).get("submission", 0)))
    validation = result.get("validation", {})
    manifest = result.get("manifest")
    preview = []
    try:
        df = read_submission_csv(path).head(5)
        preview = df.to_dict(orient="records")
    except Exception:
        pass

    return SubmissionResponse(
        valid=validation.get("valid", False),
        path=path,
        n_rows=n_rows,
        n_loans=int(result.get("validation", {}).get("n_loans", result.get("row_counts", {}).get("loans", 0))),
        reporting_month=str(state.settings.get("data.submission_reporting_month", "2024-01")),
        validation=validation,
        manifest=manifest,
        preview=preview,
        elapsed_ms=round(timer.stop(), 2),
    )


@router.get(
    "/validate",
    response_model=SubmissionValidateResponse,
    summary="Validate the existing submission.csv",
)
def validate_submission_endpoint(state: StateDep) -> SubmissionValidateResponse:
    Timer()

    from lpie.submit.validator import validate_submission

    sub_path = str(state.settings.get("paths.submission_path", "artifacts/submission.csv"))
    template_path = str(state.settings.get("paths.dataset_dir", "dataset")) + "/" + str(
        state.settings.get("data.files.submission_template", "submission_template.csv")
    )

    if not Path(sub_path).exists():
        raise DataNotFoundError(
            "submission.csv does not exist. Run `make submit` or `POST /api/v1/submission/generate`.",
            details={"path": sub_path},
        )

    try:
        df = read_submission_csv(sub_path)
    except Exception as exc:
        raise PredictionError(f"Cannot read submission file: {exc}") from exc

    template = None
    try:
        if Path(template_path).exists():
            template = read_submission_csv(template_path)
    except Exception:
        pass

    result = validate_submission(df, template, settings=state.settings)
    n_errors = len(result.get("errors", []))
    n_warnings = len(result.get("warnings", []))

    return SubmissionValidateResponse(
        valid=result.get("valid", False),
        path=sub_path,
        n_rows=len(df),
        n_loans=int(df["loan_id"].nunique()) if "loan_id" in df.columns else 0,
        n_errors=n_errors,
        n_warnings=n_warnings,
        errors=result.get("errors", []),
        warnings=result.get("warnings", []),
        summary=result.get("summary", {}),
        template=template_path,
    )
