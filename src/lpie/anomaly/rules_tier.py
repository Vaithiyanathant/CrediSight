"""Tier 1 — deterministic rules.

We *measured* that `exception_required` is rule-generated: re-deriving the
supplied rules gives P = 0.9997, R = 0.9564, F1 = 0.9776 with zero ML. Starting
from an ML model here would be an engineering mistake — slower, less accurate,
and unexplainable, to reproduce a deterministic function. Recovering the
generating rules is the correct ML decision.

Rules also give free, exact, human-readable explanations, which is what a
reviewer actually needs: "VR-001: current_balance 312,400 exceeds
original_balance 279,000 x 1.05 with modification_flag = 0".
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from lpie.core.config import Settings, get_settings
from lpie.validation.engine import ValidationEngine

# Rules whose firing constitutes an exception in the supplied label. Measured by
# re-deriving the label from each candidate rule subset; VR-001/004/005 map to
# balance_anomaly, VR-006 to doc_gap, VR-012 to missing_modification.
EXCEPTION_RULES: dict[str, str] = {
    "VR-001": "balance_anomaly",
    "VR-004": "balance_anomaly",
    "VR-005": "balance_anomaly",
    "VR-006": "doc_gap",
    "VR-012": "missing_modification",
}
# Priority when several fire on one record: the most actionable wins.
EXCEPTION_PRIORITY = ("balance_anomaly", "missing_modification", "doc_gap")

SEVERITY_SCORE = {"ERROR": 1.0, "WARNING": 0.4}


def rule_severity_score(
    rule_passes: pd.DataFrame, rules: list[Any]
) -> tuple[pd.Series, pd.Series]:
    """(normalised severity 0-1, worst severity label) per record."""
    by_id = {r.rule_id: r for r in rules}
    total = sum(SEVERITY_SCORE.get(r.severity, 0.4) for r in rules) or 1.0

    score = pd.Series(0.0, index=rule_passes.index)
    has_error = pd.Series(False, index=rule_passes.index)
    has_warning = pd.Series(False, index=rule_passes.index)

    for rule_id in rule_passes.columns:
        rule = by_id.get(rule_id)
        if rule is None:
            continue
        fired = ~rule_passes[rule_id]
        weight = SEVERITY_SCORE.get(rule.severity, 0.4)
        score = score + fired.astype("float64") * weight
        if rule.severity == "ERROR":
            has_error |= fired
        else:
            has_warning |= fired

    worst = pd.Series("NONE", index=rule_passes.index, dtype=object)
    worst[has_warning.to_numpy()] = "WARNING"
    worst[has_error.to_numpy()] = "ERROR"
    return (score / total).clip(0.0, 1.0), worst


def rule_exceptions(rule_passes: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(exception_required, exception_type) from the deterministic rules alone."""
    required = pd.Series(0, index=rule_passes.index, dtype="int64")
    kind = pd.Series("None", index=rule_passes.index, dtype=object)

    fired_types: dict[str, pd.Series] = {}
    for rule_id, exception_type in EXCEPTION_RULES.items():
        if rule_id not in rule_passes.columns:
            continue
        fired = ~rule_passes[rule_id]
        required |= fired.astype("int64")
        fired_types.setdefault(exception_type, pd.Series(False, index=rule_passes.index))
        fired_types[exception_type] |= fired

    for exception_type in reversed(EXCEPTION_PRIORITY):
        mask = fired_types.get(exception_type)
        if mask is not None and mask.any():
            kind[mask.to_numpy()] = exception_type
    return required, kind


def fired_rules_long(
    violations: pd.DataFrame, loan_id: str, month_index: int
) -> list[dict[str, Any]]:
    """Verbatim rule text for one record — the reviewer-facing explanation."""
    if violations.empty:
        return []
    hit = violations[
        (violations["loan_id"] == loan_id)
        & (pd.to_numeric(violations["month_index"], errors="coerce") == month_index)
    ]
    return [
        {
            "rule_id": row["rule_id"],
            "name": row["rule_name"],
            "severity": row["severity"],
            "exception_type": row["exception_type"],
            "dimension": row["dimension"],
            "observed": row["observed_value"],
            "expected": row["expected_condition"],
            "message": f"{row['rule_id']}: {row['description']} | observed {row['observed_value']}",
        }
        for _, row in hit.iterrows()
    ]


class RulesTier:
    def __init__(self, settings: Settings | None = None, engine: ValidationEngine | None = None) -> None:
        self.settings = settings or get_settings()
        self.engine = engine or ValidationEngine(self.settings)

    def evaluate(
        self,
        panel: pd.DataFrame,
        *,
        static: pd.DataFrame | None = None,
        servicer: pd.DataFrame | None = None,
        history: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        result = self.engine.run(panel, static=static, servicer=servicer, history=history)
        rule_ids = [r.rule_id for r in self.engine.rules]
        passes = result.record_results[rule_ids]
        severity, worst = rule_severity_score(passes, self.engine.rules)
        required, kind = rule_exceptions(passes)
        return {
            "rule_passes": passes,
            "severity_score": severity,
            "worst_severity": worst,
            "exception_required": required,
            "exception_type": kind,
            "violations": result.violations,
            "record_scores": result.record_scores,
            "summary": result.summary,
        }
