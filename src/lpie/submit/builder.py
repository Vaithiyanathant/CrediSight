"""Submission builder.

Produces `submission.csv` keyed `(loan_id, reporting_month)` at the as-of month
declared in the template, plus a manifest that pins every input to the output:
data SHA256, git SHA, model versions, feature hash, config hash, requirements
hash, and the generation timestamp. That manifest is what makes "every reported
number is regenerable" a checkable claim rather than an assertion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from lpie.core.config import Settings, get_settings
from lpie.core.determinism import git_sha, sha256_file, sha256_obj
from lpie.core.logging import get_logger
from lpie.core.timing import utcnow_iso
from lpie.data.contracts import SUBMISSION_CONTRACT

log = get_logger(__name__)

SUBMISSION_COLUMNS = [c.name for c in SUBMISSION_CONTRACT.columns]
PROBABILITY_COLUMNS = (
    "prob_next_3m_delinquency", "prob_next_6m_delinquency",
    "prob_next_12m_default", "prob_next_12m_prepayment",
)


def build_submission(
    predictions: pd.DataFrame,
    template: pd.DataFrame,
    *,
    reporting_month: str | None = None,
    settings: Settings | None = None,
    strict_row_set: bool = False,
) -> pd.DataFrame:
    """Assemble the submission frame in template column order.

    The supplied template carries 20 illustrative rows; the real submission is
    every loan at the as-of month. `strict_row_set` restricts output to exactly
    the template's loan IDs, for the case where the organiser scores only those.
    """
    s = settings or get_settings()
    month = reporting_month or str(s.get("data.submission_reporting_month", "2024-01"))

    work = predictions.copy()
    if "reporting_month" not in work.columns:
        work["reporting_month"] = month
    work["reporting_month"] = month

    if strict_row_set and "loan_id" in template.columns:
        wanted = set(template["loan_id"].astype(str))
        work = work[work["loan_id"].astype(str).isin(wanted)]

    for column in SUBMISSION_COLUMNS:
        if column not in work.columns:
            work[column] = _default_for(column)

    for column in PROBABILITY_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    work["anomaly_score"] = pd.to_numeric(work["anomaly_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    work["model_confidence"] = (
        pd.to_numeric(work["model_confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    )
    work["exception_required"] = (
        pd.to_numeric(work["exception_required"], errors="coerce").fillna(0).astype("int64").clip(0, 1)
    )
    for column in ("predicted_next_state", "exception_type", "reviewer_action"):
        work[column] = work[column].astype("object").where(work[column].notna(), _default_for(column))
    for column in ("top_driver_1", "top_driver_2", "top_driver_3"):
        work[column] = work[column].astype("object").where(work[column].notna(), "")

    out = work[SUBMISSION_COLUMNS].drop_duplicates(subset=["loan_id", "reporting_month"], keep="first")
    return out.sort_values("loan_id", kind="mergesort").reset_index(drop=True)


def _default_for(column: str) -> Any:
    if column in PROBABILITY_COLUMNS or column in ("anomaly_score", "model_confidence"):
        return 0.0
    if column == "exception_required":
        return 0
    if column == "predicted_next_state":
        return "Current"
    if column == "exception_type":
        return "None"
    if column == "reviewer_action":
        return "No Action"
    return ""


def write_submission(
    frame: pd.DataFrame, *, settings: Settings | None = None, path: Path | None = None
) -> Path:
    s = settings or get_settings()
    target = path or s.path("submission_path")
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, float_format="%.6f")
    log.info("submission.written", path=str(target), rows=len(frame))
    return target


def build_manifest(
    submission_path: Path,
    *,
    data_sha256: str,
    file_hashes: dict[str, str],
    model_versions: dict[str, str],
    feature_hash: str,
    n_rows: int,
    n_loans: int,
    reporting_month: str,
    settings: Settings | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything a third party needs to reproduce this exact file."""
    s = settings or get_settings()
    requirements = s.root / "requirements.lock"
    pyproject = s.root / "pyproject.toml"

    manifest = {
        "generated_at": utcnow_iso(),
        "submission_file": str(submission_path.name),
        "submission_sha256": sha256_file(submission_path) if submission_path.exists() else None,
        "n_rows": int(n_rows),
        "n_loans": int(n_loans),
        "reporting_month": reporting_month,
        "data_sha256": data_sha256,
        "input_file_hashes": file_hashes,
        "git_sha": git_sha(s.root),
        "model_version": s.model_version,
        "feature_version": s.feature_version,
        "model_versions_by_head": model_versions,
        "feature_hash": feature_hash,
        "config_hash": sha256_obj(s.as_dict()),
        "config_path": str(s.config_path.relative_to(s.root)) if s.config_path.is_relative_to(s.root) else str(s.config_path),
        "requirements_hash": sha256_file(requirements) if requirements.exists() else None,
        "pyproject_hash": sha256_file(pyproject) if pyproject.exists() else None,
        "seed": s.seed,
        "reproduce_command": "make all",
        **(extra or {}),
    }
    return manifest


def write_manifest(manifest: dict[str, Any], *, settings: Settings | None = None) -> Path:
    import json

    s = settings or get_settings()
    target = s.path("submission_manifest_path")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, default=str))
    log.info("submission.manifest_written", path=str(target))
    return target
