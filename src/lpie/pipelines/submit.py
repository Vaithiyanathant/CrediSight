"""Submission stage: score the as-of month, write submission.csv and its manifest."""

from __future__ import annotations

import json
from typing import Any

from lpie.core.config import Settings, get_settings
from lpie.core.determinism import seed_everything
from lpie.core.logging import get_logger
from lpie.core.timing import Timer

log = get_logger(__name__)


def stage_submit(settings: Settings | None = None, *, as_of_month: int | None = None) -> dict[str, Any]:
    from lpie.data.ingest import data_sha256, dataset_hashes, load_submission_template
    from lpie.serving.scorer import PredictionScorer
    from lpie.serving.state import AppState
    from lpie.submit.builder import build_manifest, build_submission, write_manifest, write_submission
    from lpie.submit.validator import validate_submission

    s = settings or get_settings()
    seed_everything(s.seed)
    timer = Timer()

    state = AppState(s)
    state.startup()

    months = state.feature_store_months()
    if not months:
        raise SystemExit("Feature store is empty. Run `make features` first.")

    # The submission is keyed at the template's as-of month. The template's
    # reporting_month (2024-01) corresponds to the first test month; month_index
    # is used to select it because reporting_month itself is corrupted (VR-013).
    target_month = int(as_of_month or s.get("data.test_month_min", 37))
    if target_month not in months:
        target_month = max(months)

    features = state.features(months=[target_month])
    if features.empty:
        raise SystemExit(f"No feature rows at month_index {target_month}")

    scorer = PredictionScorer(state, s)
    result = scorer.score(features, include_drivers=True)
    scored = result.frame

    template = load_submission_template(s)
    reporting_month = str(s.get("data.submission_reporting_month", "2024-01"))
    submission = build_submission(scored, template, reporting_month=reporting_month, settings=s)

    path = write_submission(submission, settings=s)
    report = validate_submission(submission, template, settings=s)

    manifest = build_manifest(
        path,
        data_sha256=data_sha256(),
        file_hashes=dataset_hashes(s.path("dataset_dir"), s.require("data.files")),
        model_versions=state.model_versions(),
        feature_hash=state.registry.contract_hash(),
        n_rows=int(len(submission)),
        n_loans=int(submission["loan_id"].nunique()),
        reporting_month=reporting_month,
        settings=s,
        extra={
            "as_of_month_index": target_month,
            "validation": {k: v for k, v in report.items() if k not in ("errors", "warnings")},
            "n_validation_errors": report["n_errors"],
            "scoring_degraded_components": result.degraded,
        },
    )
    write_manifest(manifest, settings=s)

    payload = {
        "status": "ok" if report["valid"] else "invalid",
        "path": str(path),
        "n_rows": int(len(submission)),
        "validation": report,
        "manifest": manifest,
    }
    (s.path("reports_dir") / "submission_report.json").write_text(
        json.dumps(payload, indent=2, default=str)
    )
    timer.stop()
    log.info("stage.submit.done", rows=len(submission), valid=report["valid"],
             elapsed_ms=timer.elapsed_ms)
    if not report["valid"]:
        raise SystemExit(f"Submission failed validation: {report['errors'][:3]}")
    return payload
