"""Submission validation against `submission_template.csv`.

Checks are structural (columns, order, key uniqueness), semantic (probability
ranges, allowed state and action values), and completeness (no NaN where the
schema forbids one). A submission that fails here is reported with the exact
offending rows rather than a bare boolean.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lpie.core.config import Settings, get_settings
from lpie.core.logging import get_logger
from lpie.data.contracts import STATUSES, SUBMISSION_CONTRACT, check_contract
from lpie.submit.builder import PROBABILITY_COLUMNS, SUBMISSION_COLUMNS

log = get_logger(__name__)

ALLOWED_ACTIONS = ("No Action", "Flag", "Escalate")
NON_NULL_COLUMNS = (
    "loan_id", "reporting_month", *PROBABILITY_COLUMNS, "predicted_next_state",
    "anomaly_score", "exception_required", "exception_type", "reviewer_action",
    "model_confidence",
)


def validate_submission(
    frame: pd.DataFrame,
    template: pd.DataFrame | None = None,
    *,
    expected_loan_ids: set[str] | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    s = settings or get_settings()
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    # --- structure -----------------------------------------------------
    missing = [c for c in SUBMISSION_COLUMNS if c not in frame.columns]
    if missing:
        errors.append({"check": "required_columns", "missing": missing})
    extra = [c for c in frame.columns if c not in SUBMISSION_COLUMNS]
    if extra:
        warnings.append({"check": "unexpected_columns", "columns": extra})

    if template is not None:
        template_columns = list(template.columns)
        if list(frame.columns)[: len(template_columns)] != template_columns:
            warnings.append(
                {
                    "check": "column_order",
                    "detail": "Column order differs from the template.",
                    "expected": template_columns,
                    "observed": list(frame.columns),
                }
            )

    if missing:
        return _report(frame, errors, warnings, s)

    # --- keys ----------------------------------------------------------
    duplicates = int(frame.duplicated(subset=["loan_id", "reporting_month"]).sum())
    if duplicates:
        errors.append({"check": "unique_key", "n_duplicates": duplicates,
                       "key": ["loan_id", "reporting_month"]})

    bad_ids = frame.loc[~frame["loan_id"].astype(str).str.match(r"^LN\d{7}$", na=False), "loan_id"]
    if not bad_ids.empty:
        errors.append({"check": "loan_id_format", "n": int(len(bad_ids)),
                       "examples": bad_ids.head(5).astype(str).tolist()})

    months = frame["reporting_month"].astype(str)
    bad_months = months[~months.str.match(r"^\d{4}-(0[1-9]|1[0-2])$", na=False)]
    if not bad_months.empty:
        errors.append({"check": "reporting_month_format", "n": int(len(bad_months)),
                       "examples": bad_months.head(5).tolist()})
    if months.nunique() > 1:
        warnings.append({"check": "multiple_reporting_months", "values": sorted(months.unique())[:10]})

    # --- ranges --------------------------------------------------------
    for column in (*PROBABILITY_COLUMNS, "anomaly_score", "model_confidence"):
        values = pd.to_numeric(frame[column], errors="coerce")
        n_nan = int(values.isna().sum())
        out_of_range = int(((values < 0) | (values > 1)).sum())
        if n_nan:
            errors.append({"check": "no_nan", "column": column, "n": n_nan})
        if out_of_range:
            errors.append({"check": "probability_range", "column": column, "n": out_of_range,
                           "min": _f(values.min()), "max": _f(values.max())})

    exception_required = pd.to_numeric(frame["exception_required"], errors="coerce")
    bad_flag = int((~exception_required.isin([0, 1])).sum())
    if bad_flag:
        errors.append({"check": "exception_required_binary", "n": bad_flag})

    bad_state = frame.loc[~frame["predicted_next_state"].astype(str).isin(STATUSES), "predicted_next_state"]
    if not bad_state.empty:
        errors.append({"check": "predicted_next_state_domain", "n": int(len(bad_state)),
                       "examples": sorted(set(bad_state.astype(str)))[:5], "allowed": list(STATUSES)})

    bad_action = frame.loc[~frame["reviewer_action"].astype(str).isin(ALLOWED_ACTIONS), "reviewer_action"]
    if not bad_action.empty:
        errors.append({"check": "reviewer_action_domain", "n": int(len(bad_action)),
                       "examples": sorted(set(bad_action.astype(str)))[:5],
                       "allowed": list(ALLOWED_ACTIONS)})

    # --- completeness --------------------------------------------------
    for column in NON_NULL_COLUMNS:
        n_null = int(frame[column].isna().sum())
        if n_null:
            errors.append({"check": "no_nan", "column": column, "n": n_null})

    # --- coverage ------------------------------------------------------
    if expected_loan_ids:
        present = set(frame["loan_id"].astype(str))
        missing_loans = expected_loan_ids - present
        extra_loans = present - expected_loan_ids
        if missing_loans:
            errors.append({"check": "loan_coverage", "n_missing": len(missing_loans),
                           "examples": sorted(missing_loans)[:5]})
        if extra_loans:
            warnings.append({"check": "unexpected_loans", "n": len(extra_loans),
                             "examples": sorted(extra_loans)[:5]})

    # --- sanity: an all-constant column means the pipeline did not run ---
    for column in PROBABILITY_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        if len(values) > 10 and values.nunique() <= 1:
            errors.append(
                {
                    "check": "degenerate_predictions",
                    "column": column,
                    "detail": (
                        "Every row carries an identical probability. This is what an "
                        "unpopulated template looks like, not a scored submission."
                    ),
                    "value": _f(values.iloc[0]),
                }
            )

    return _report(frame, errors, warnings, s)


def _report(
    frame: pd.DataFrame, errors: list[dict[str, Any]], warnings: list[dict[str, Any]], s: Settings
) -> dict[str, Any]:
    contract: dict[str, Any] = {}
    try:
        contract = check_contract(frame, SUBMISSION_CONTRACT, strict_values=True)
    except Exception as exc:
        errors.append({"check": "contract", "detail": str(exc)})

    summary: dict[str, Any] = {}
    if not frame.empty and all(c in frame.columns for c in PROBABILITY_COLUMNS):
        summary = {
            "mean_probabilities": {
                c: _f(pd.to_numeric(frame[c], errors="coerce").mean()) for c in PROBABILITY_COLUMNS
            },
            "predicted_next_state_distribution": (
                frame["predicted_next_state"].astype(str).value_counts().to_dict()
                if "predicted_next_state" in frame.columns else {}
            ),
            "reviewer_action_distribution": (
                frame["reviewer_action"].astype(str).value_counts().to_dict()
                if "reviewer_action" in frame.columns else {}
            ),
            "exception_rate": (
                _f(pd.to_numeric(frame["exception_required"], errors="coerce").mean())
                if "exception_required" in frame.columns else None
            ),
            "mean_model_confidence": (
                _f(pd.to_numeric(frame["model_confidence"], errors="coerce").mean())
                if "model_confidence" in frame.columns else None
            ),
        }

    return {
        "valid": not errors,
        "n_rows": int(len(frame)),
        "n_loans": int(frame["loan_id"].nunique()) if "loan_id" in frame.columns else 0,
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "contract": {k: v for k, v in contract.items() if k != "columns_present"},
        "summary": summary,
        "template": str(s.dataset_file("submission_template").name),
    }


def _f(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, 6)



# `exception_type` legitimately carries the literal string "None" (the value the
# supplied template uses for "no exception"), and "None" is in pandas' default
# NA list. Reading the submission with default na_values therefore turned 8,350
# valid categorical values into nulls and failed the file for "nulls in a
# non-nullable column" — the generator and the validator disagreed about a file
# neither had corrupted. Categorical columns are read as literal strings.
SUBMISSION_STRING_COLUMNS = [
    "loan_id", "reporting_month", "predicted_next_state", "exception_type",
    "top_driver_1", "top_driver_2", "top_driver_3", "reviewer_action",
]


def read_submission_csv(path) -> pd.DataFrame:
    """Read a submission/template CSV without coercing "None" to NaN."""
    return pd.read_csv(
        path,
        keep_default_na=True,
        na_values=[],
        converters={c: lambda v: v for c in SUBMISSION_STRING_COLUMNS},
    )

def validate_submission_file(
    path: Path | str | None = None, *, settings: Settings | None = None
) -> dict[str, Any]:
    s = settings or get_settings()
    target = Path(path) if path else s.path("submission_path")
    if not target.exists():
        return {
            "valid": False,
            "n_errors": 1,
            "errors": [{"check": "file_exists", "path": str(target),
                        "detail": "submission.csv has not been generated. Run `make submit`."}],
            "warnings": [],
            "n_rows": 0,
        }
    frame = read_submission_csv(target)
    template_path = s.dataset_file("submission_template")
    template = read_submission_csv(template_path) if template_path.exists() else None
    report = validate_submission(frame, template, settings=s)
    report["path"] = str(target)
    return report
