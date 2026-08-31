"""Declared schema contracts for every input file.

A contract failure is raised loudly at ingest. Silent coercion is what turns a
schema change into a plausible wrong answer six weeks later, so we never do it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from lpie.core.exceptions import DataContractError

DType = Literal["string", "int", "float", "bool", "date", "month"]

MONTH_RE = r"^\d{4}-(0[1-9]|1[0-2])$"
DATE_RE = r"^\d{4}-\d{2}-\d{2}"
LOAN_ID_RE = r"^LN\d{7}$"


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: DType
    required: bool = True
    nullable: bool = True
    allowed: tuple[str, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    pattern: str | None = None
    description: str = ""


@dataclass(frozen=True)
class TableContract:
    name: str
    columns: tuple[ColumnSpec, ...]
    primary_key: tuple[str, ...] = ()
    # Columns that may legitimately be absent (targets on the test split).
    optional_groups: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def spec(self, name: str) -> ColumnSpec | None:
        for c in self.columns:
            if c.name == name:
                return c
        return None


STATUSES = ("Current", "30DPD", "60DPD", "90DPD", "Default", "Prepaid", "Closed")
DOC_STATUSES = ("Complete", "Missing-Income", "Missing-Appraisal", "Missing-ID", "Stale")
SOURCE_SYSTEMS = ("LOS", "Servicer-Portal", "Manual-Entry", "Batch-Upload")
LOSS_SEVERITY = ("0%", "1-10%", "11-25%", "26-50%", ">50%", "N/A")
EXCEPTION_TYPES = (
    "None", "balance_anomaly", "doc_gap", "missing_modification", "status_dpd_mismatch",
    "date_inconsistency", "servicer_conflict", "rate_out_of_range", "term_inconsistency",
    "stale_record", "loss_sev_inconsistency", "duplicate_record", "state_reversal",
    "static_mismatch", "unseen_category",
)

_STATIC_COLUMNS = (
    ColumnSpec("loan_id", "string", nullable=False, pattern=LOAN_ID_RE, description="Unique loan identifier"),
    ColumnSpec("origination_month", "month", pattern=MONTH_RE, description="Month originated"),
    ColumnSpec("original_balance", "float", minimum=0, description="Original UPB (USD)"),
    ColumnSpec("interest_rate", "float", minimum=0, maximum=40, description="Annual note rate (%)"),
    ColumnSpec("loan_term_months", "int", minimum=1, maximum=600, description="Full loan term"),
    ColumnSpec("credit_score_band", "string", description="Credit score band"),
    ColumnSpec("ltv_band", "string", description="LTV band"),
    ColumnSpec("dti_band", "string", description="DTI band"),
    ColumnSpec("state", "string", description="Property state"),
    ColumnSpec("loan_purpose", "string", description="Loan purpose"),
    ColumnSpec("occupancy_type", "string", description="Occupancy"),
    ColumnSpec("property_type", "string", description="Property type"),
    ColumnSpec("servicer_name", "string", description="Loan servicer"),
    ColumnSpec("vintage_year", "string", description="Origination year"),
)

_MONTHLY_BASE_COLUMNS = (
    ColumnSpec("loan_id", "string", nullable=False, pattern=LOAN_ID_RE),
    ColumnSpec("month_index", "int", nullable=False, minimum=1, maximum=600,
               description="Sequential panel month — the only trustworthy clock"),
    ColumnSpec("reporting_month", "month", pattern=MONTH_RE,
               description="Calendar month (corrupted in the supplied pack — see VR-013)"),
    ColumnSpec("origination_month", "month", pattern=MONTH_RE),
    ColumnSpec("loan_age_months", "int", minimum=0, maximum=1200),
    ColumnSpec("remaining_term_months", "int", minimum=0, maximum=600),
    ColumnSpec("original_balance", "float", minimum=0),
    ColumnSpec("current_balance", "float", minimum=0),
    ColumnSpec("interest_rate", "float", minimum=0, maximum=40),
    ColumnSpec("credit_score_band", "string"),
    ColumnSpec("ltv_band", "string"),
    ColumnSpec("dti_band", "string"),
    ColumnSpec("state", "string"),
    ColumnSpec("loan_purpose", "string"),
    ColumnSpec("occupancy_type", "string"),
    ColumnSpec("property_type", "string"),
    ColumnSpec("servicer_name", "string"),
    ColumnSpec("current_status", "string", allowed=STATUSES),
    ColumnSpec("days_past_due", "float", minimum=0, maximum=400),
    ColumnSpec("modification_flag", "int", minimum=0, maximum=1),
    ColumnSpec("prepayment_flag", "int", minimum=0, maximum=1),
    ColumnSpec("default_flag", "int", minimum=0, maximum=1),
    ColumnSpec("loss_severity_band", "string", allowed=LOSS_SEVERITY),
    ColumnSpec("last_updated_at", "date", pattern=DATE_RE),
    ColumnSpec("source_system", "string", allowed=SOURCE_SYSTEMS),
    ColumnSpec("document_status", "string", allowed=DOC_STATUSES),
)

_TARGET_COLUMNS = (
    ColumnSpec("next_3m_delinquency_flag", "int", required=False, minimum=0, maximum=1),
    ColumnSpec("next_6m_delinquency_flag", "int", required=False, minimum=0, maximum=1),
    ColumnSpec("next_12m_default_flag", "int", required=False, minimum=0, maximum=1),
    ColumnSpec("next_12m_prepayment_flag", "int", required=False, minimum=0, maximum=1),
    ColumnSpec("next_state", "string", required=False, allowed=STATUSES),
    ColumnSpec("exception_required", "int", required=False, minimum=0, maximum=1),
    ColumnSpec("exception_type", "string", required=False, allowed=EXCEPTION_TYPES),
)

_SERVICER_COLUMNS = (
    ColumnSpec("loan_id", "string", nullable=False, pattern=LOAN_ID_RE),
    ColumnSpec("update_date", "date", nullable=False, pattern=DATE_RE),
    ColumnSpec("servicer_name", "string"),
    ColumnSpec("reported_balance", "float", minimum=0),
    ColumnSpec("reported_status", "string", allowed=STATUSES),
    ColumnSpec("reported_rate", "float", minimum=0, maximum=40),
    ColumnSpec("source_system", "string", allowed=SOURCE_SYSTEMS),
    ColumnSpec("conflict_type", "string",
               allowed=("balance_mismatch", "status_conflict", "stale_record", "rate_discrepancy", "none")),
    ColumnSpec("stale_flag", "int", minimum=0, maximum=1),
    ColumnSpec("notes", "string"),
)

_MACRO_COLUMNS = (
    ColumnSpec("scenario_name", "string", nullable=False),
    ColumnSpec("description", "string"),
    ColumnSpec("gdp_growth_pct", "float"),
    ColumnSpec("unemployment_rate_pct", "float", minimum=0, maximum=100),
    ColumnSpec("hpi_change_pct", "float"),
    ColumnSpec("interest_rate_shock_bps", "float"),
    ColumnSpec("credit_spread_shock_bps", "float"),
    ColumnSpec("prepayment_cpr_assumption_pct", "float", minimum=0, maximum=100),
    ColumnSpec("default_rate_multiplier", "float", minimum=0),
    ColumnSpec("delinquency_rate_multiplier", "float", minimum=0),
    ColumnSpec("prepayment_rate_multiplier", "float", minimum=0),
)

_SUBMISSION_COLUMNS = (
    ColumnSpec("loan_id", "string", nullable=False, pattern=LOAN_ID_RE),
    ColumnSpec("reporting_month", "month", nullable=False, pattern=MONTH_RE),
    ColumnSpec("prob_next_3m_delinquency", "float", nullable=False, minimum=0.0, maximum=1.0),
    ColumnSpec("prob_next_6m_delinquency", "float", nullable=False, minimum=0.0, maximum=1.0),
    ColumnSpec("prob_next_12m_default", "float", nullable=False, minimum=0.0, maximum=1.0),
    ColumnSpec("prob_next_12m_prepayment", "float", nullable=False, minimum=0.0, maximum=1.0),
    ColumnSpec("predicted_next_state", "string", nullable=False, allowed=STATUSES),
    ColumnSpec("anomaly_score", "float", nullable=False, minimum=0.0, maximum=1.0),
    ColumnSpec("exception_required", "int", nullable=False, minimum=0, maximum=1),
    ColumnSpec("exception_type", "string", nullable=False),
    ColumnSpec("top_driver_1", "string"),
    ColumnSpec("top_driver_2", "string"),
    ColumnSpec("top_driver_3", "string"),
    ColumnSpec("reviewer_action", "string", nullable=False,
               allowed=("No Action", "Flag", "Escalate")),
    ColumnSpec("model_confidence", "float", nullable=False, minimum=0.0, maximum=1.0),
)

STATIC_CONTRACT = TableContract("loan_static_attributes", _STATIC_COLUMNS, primary_key=("loan_id",))
MONTHLY_TRAIN_CONTRACT = TableContract(
    "loan_monthly_performance_train",
    _MONTHLY_BASE_COLUMNS + _TARGET_COLUMNS,
    primary_key=("loan_id", "month_index"),
    optional_groups={"targets": tuple(c.name for c in _TARGET_COLUMNS)},
)
MONTHLY_TEST_CONTRACT = TableContract(
    "loan_monthly_performance_test",
    _MONTHLY_BASE_COLUMNS,
    primary_key=("loan_id", "month_index"),
)
SERVICER_CONTRACT = TableContract("servicer_updates", _SERVICER_COLUMNS)
MACRO_CONTRACT = TableContract("macro_scenarios", _MACRO_COLUMNS, primary_key=("scenario_name",))
SUBMISSION_CONTRACT = TableContract(
    "submission", _SUBMISSION_COLUMNS, primary_key=("loan_id", "reporting_month")
)

CONTRACTS: dict[str, TableContract] = {
    c.name: c
    for c in (
        STATIC_CONTRACT, MONTHLY_TRAIN_CONTRACT, MONTHLY_TEST_CONTRACT,
        SERVICER_CONTRACT, MACRO_CONTRACT, SUBMISSION_CONTRACT,
    )
}

# Pandas dtypes used at read time. Strings stay `object` so NaN survives; numerics
# use nullable Float64/Int64 so "missing" is never silently 0.
PANDAS_DTYPES: dict[DType, str] = {
    "string": "object",
    "int": "Float64",   # read as float; integrality is asserted, not coerced
    "float": "Float64",
    "bool": "boolean",
    "date": "object",
    "month": "object",
}


def check_contract(
    df: pd.DataFrame,
    contract: TableContract,
    *,
    strict_values: bool = True,
    allow_missing_optional_groups: bool = True,
    sample_rows: int | None = None,
) -> dict[str, Any]:
    """Validate `df` against `contract`.

    Returns a structured report. Raises `DataContractError` on any *structural*
    violation (missing required column, wrong key cardinality, non-nullable null),
    because those cannot be recovered from. Value-domain violations are reported
    as findings — they are exactly what the validation engine is for.
    """
    findings: list[dict[str, Any]] = []
    errors: list[str] = []

    optional_all = {n for group in contract.optional_groups.values() for n in group}
    present = set(df.columns)

    for spec in contract.columns:
        if spec.name in present:
            continue
        if not spec.required:
            continue
        if allow_missing_optional_groups and spec.name in optional_all:
            continue
        errors.append(f"missing required column '{spec.name}'")

    extras = sorted(present - set(contract.column_names))
    if extras:
        findings.append({"kind": "extra_columns", "columns": extras})

    if errors:
        raise DataContractError(
            f"{contract.name}: {len(errors)} structural contract violation(s)",
            details={"violations": errors, "table": contract.name},
        )

    sample = df if sample_rows is None or len(df) <= sample_rows else df.sample(sample_rows, random_state=0)

    for spec in contract.columns:
        if spec.name not in present:
            continue
        col = sample[spec.name]
        n_null = int(col.isna().sum())
        if not spec.nullable and n_null:
            raise DataContractError(
                f"{contract.name}.{spec.name}: {n_null} null(s) in a non-nullable column",
                details={"table": contract.name, "column": spec.name, "n_null": n_null},
            )
        if n_null:
            findings.append({"kind": "nulls", "column": spec.name, "n": n_null,
                             "rate": round(n_null / max(len(sample), 1), 6)})

        if not strict_values:
            continue
        non_null = col.dropna()
        if non_null.empty:
            findings.append({"kind": "all_null", "column": spec.name})
            continue

        if spec.dtype in ("float", "int"):
            numeric = pd.to_numeric(non_null, errors="coerce")
            n_bad = int(numeric.isna().sum())
            if n_bad:
                findings.append({"kind": "non_numeric", "column": spec.name, "n": n_bad})
            numeric = numeric.dropna()
            if spec.minimum is not None:
                n_low = int((numeric < spec.minimum).sum())
                if n_low:
                    findings.append({"kind": "below_minimum", "column": spec.name,
                                     "n": n_low, "minimum": spec.minimum})
            if spec.maximum is not None:
                n_high = int((numeric > spec.maximum).sum())
                if n_high:
                    findings.append({"kind": "above_maximum", "column": spec.name,
                                     "n": n_high, "maximum": spec.maximum})
            if spec.dtype == "int" and len(numeric):
                frac = np.abs(numeric.to_numpy(dtype="float64") % 1.0)
                n_frac = int((frac > 1e-9).sum())
                if n_frac:
                    findings.append({"kind": "non_integral", "column": spec.name, "n": n_frac})
        else:
            as_str = non_null.astype(str)
            if spec.allowed is not None:
                bad = sorted(set(as_str.unique()) - set(spec.allowed))
                if bad:
                    findings.append({"kind": "unexpected_levels", "column": spec.name,
                                     "levels": bad[:25], "n_levels": len(bad)})
            if spec.pattern is not None:
                n_bad = int((~as_str.str.match(spec.pattern, na=False)).sum())
                if n_bad:
                    findings.append({"kind": "pattern_mismatch", "column": spec.name,
                                     "n": n_bad, "pattern": spec.pattern})

    pk_report: dict[str, Any] = {}
    if contract.primary_key and set(contract.primary_key).issubset(present):
        dup = int(df.duplicated(subset=list(contract.primary_key)).sum())
        pk_report = {"key": list(contract.primary_key), "n_duplicates": dup}
        if dup:
            findings.append({"kind": "duplicate_primary_key",
                             "key": list(contract.primary_key), "n": dup})

    return {
        "table": contract.name,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns_present": sorted(present),
        "columns_missing_optional": sorted(
            n for n in optional_all if n not in present
        ),
        "primary_key": pk_report,
        "findings": findings,
        "passed": not any(f["kind"] in {"duplicate_primary_key"} for f in findings) and not errors,
    }
