"""Evidence packet builder — deterministic Python, zero LLM.

Every number the copilot is allowed to say must exist here first, and each one
carries a `source` field naming the component that computed it. The verifier
then checks generated text against this object, which is only possible because
the packet is typed, flat, and complete.

The packet is also the cache key: `hash(evidence_packet)` identifies a response,
so an identical question about an identical loan-month never costs a second API
call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from lpie.core.determinism import sha256_obj
from lpie.core.timing import utcnow_iso


@dataclass
class EvidencePacket:
    task: str
    subject: dict[str, Any] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)
    numbers: dict[str, dict[str, Any]] = field(default_factory=dict)
    rules_fired: list[dict[str, Any]] = field(default_factory=list)
    field_names: list[str] = field(default_factory=list)
    rule_ids: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    model_version: str = ""
    feature_version: str = ""
    built_at: str = field(default_factory=utcnow_iso)

    def add_number(
        self, key: str, value: Any, *, source: str, unit: str = "", description: str = ""
    ) -> None:
        numeric = _to_float(value)
        if numeric is None:
            return
        self.numbers[key] = {
            "value": numeric,
            "source": source,
            "unit": unit,
            "description": description,
        }

    def allowed_values(self) -> list[float]:
        return [entry["value"] for entry in self.numbers.values()]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "subject": self.subject,
            "facts": self.facts,
            "numbers": self.numbers,
            "rules_fired": self.rules_fired,
            "field_names": sorted(set(self.field_names)),
            "rule_ids": sorted(set(self.rule_ids)),
            "citations": self.citations,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "built_at": self.built_at,
        }

    def hash(self) -> str:
        payload = self.to_dict()
        payload.pop("built_at", None)
        return sha256_obj(payload)

    def render(self, *, max_chars: int = 6000) -> str:
        """Compact, unambiguous text form for the prompt."""
        lines = [f"TASK: {self.task}", f"MODEL VERSION: {self.model_version}"]
        if self.subject:
            lines.append("SUBJECT:")
            lines.extend(f"  {k}: {v}" for k, v in self.subject.items())
        if self.numbers:
            lines.append("NUMERIC FACTS (these are the ONLY numbers you may state):")
            for key, entry in self.numbers.items():
                unit = f" {entry['unit']}" if entry["unit"] else ""
                note = f"  -- {entry['description']}" if entry["description"] else ""
                lines.append(f"  {key} = {entry['value']}{unit}  [source: {entry['source']}]{note}")
        if self.facts:
            lines.append("CATEGORICAL FACTS:")
            lines.extend(f"  {k}: {v}" for k, v in self.facts.items())
        if self.rules_fired:
            lines.append("VALIDATION RULES FIRED (quote these verbatim):")
            for rule in self.rules_fired:
                lines.append(
                    f"  {rule.get('rule_id')} [{rule.get('severity')}] {rule.get('name')}: "
                    f"observed {rule.get('observed')} | expected {rule.get('expected')}"
                )
        if self.field_names:
            lines.append("VALID FIELD NAMES: " + ", ".join(sorted(set(self.field_names))))
        if self.citations:
            lines.append("RETRIEVED CONTEXT:")
            for c in self.citations:
                lines.append(f"  [{c.get('citation')}] {str(c.get('text', ''))[:500]}")
        text = "\n".join(lines)
        return text[:max_chars]


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, 6)


# --------------------------------------------------------------------------- #
def build_loan_packet(
    prediction: dict[str, Any],
    *,
    task: str = "reviewer_note",
    citations: list[dict[str, Any]] | None = None,
    dictionary_fields: list[str] | None = None,
    rule_ids: list[str] | None = None,
    segment_benchmarks: dict[str, Any] | None = None,
) -> EvidencePacket:
    """Turn one prediction bundle into a verifiable evidence packet."""
    packet = EvidencePacket(
        task=task,
        model_version=str(prediction.get("model_version", "")),
        feature_version=str(prediction.get("feature_version", "")),
    )
    packet.subject = {
        "loan_id": prediction.get("loan_id"),
        "reporting_month": prediction.get("reporting_month"),
        "month_index": prediction.get("month_index"),
        "current_status": prediction.get("current_status"),
    }

    predictions = prediction.get("predictions", {}) or {}
    for key, entry in predictions.items():
        if isinstance(entry, dict) and "value" in entry:
            packet.add_number(
                key, entry["value"], source="prediction_core",
                description=f"calibrated probability, {key}",
            )
            ci = entry.get("ci")
            if isinstance(ci, (list, tuple)) and len(ci) == 2:
                packet.add_number(f"{key}_ci_low", ci[0], source="conformal_interval")
                packet.add_number(f"{key}_ci_high", ci[1], source="conformal_interval")

    next_state = predictions.get("next_state") or {}
    if isinstance(next_state, dict):
        packet.facts["predicted_next_state"] = next_state.get("predicted")
        for state, probability in (next_state.get("probs") or {}).items():
            packet.add_number(f"p_next_state_{state}", probability, source="hazard_model")

    anomaly = prediction.get("anomaly") or {}
    packet.add_number("anomaly_score", anomaly.get("score"), source="anomaly_ensemble")
    packet.facts["anomaly_tier"] = anomaly.get("tier")

    for rule in anomaly.get("rules_fired") or []:
        packet.rules_fired.append(rule)
        if rule.get("rule_id"):
            packet.rule_ids.append(rule["rule_id"])

    exception = prediction.get("exception") or {}
    packet.facts["exception_required"] = exception.get("required")
    packet.facts["exception_type"] = exception.get("type")
    packet.facts["exception_source"] = exception.get("source")

    dq = prediction.get("data_quality") or {}
    packet.add_number("dq_score", dq.get("dq_score"), source="data_quality_engine", unit="/100")
    packet.facts["dq_grade"] = dq.get("dq_grade")

    confidence = prediction.get("confidence") or {}
    for key, value in confidence.items():
        packet.add_number(key, value, source="uncertainty_module")

    explanation = prediction.get("explanation") or {}
    for driver in (explanation.get("top_drivers") or [])[:5]:
        name = driver.get("feature")
        if not name:
            continue
        packet.field_names.append(name)
        packet.add_number(
            f"shap_{name}", driver.get("shap"), source="treeshap",
            description=f"SHAP contribution of {name}",
        )
        packet.add_number(f"value_{name}", driver.get("value"), source="feature_pipeline")
        packet.facts[f"driver_{name}_direction"] = driver.get("direction")

    packet.facts["reviewer_action"] = prediction.get("reviewer_action")
    packet.facts["is_terminal"] = prediction.get("is_terminal")
    packet.facts["gated_by_rule"] = prediction.get("gated_by_rule")

    if segment_benchmarks:
        for key, value in segment_benchmarks.items():
            packet.add_number(f"segment_{key}", value, source="segment_benchmark")

    if dictionary_fields:
        packet.field_names.extend(dictionary_fields)
    if rule_ids:
        packet.rule_ids.extend(rule_ids)
    packet.citations = citations or []
    return packet


def build_scenario_packet(
    summary: dict[str, Any],
    *,
    scenario_name: str,
    assumptions: dict[str, Any] | None = None,
    citations: list[dict[str, Any]] | None = None,
) -> EvidencePacket:
    packet = EvidencePacket(task="scenario_summary")
    packet.subject = {"scenario": scenario_name, "n_paths": summary.get("n_paths"),
                      "horizon_months": summary.get("horizon_months"),
                      "n_loans": summary.get("n_loans")}

    for key, value in (assumptions or {}).items():
        packet.add_number(f"assumption_{key}", value, source="macro_scenarios.csv")

    terminal = summary.get("terminal", {}) or {}
    for metric, distribution in terminal.items():
        if not isinstance(distribution, dict):
            continue
        for stat, value in distribution.items():
            packet.add_number(f"{metric}_{stat}", value, source="monte_carlo_simulation")

    for key, value in (summary.get("risk_measures") or {}).items():
        packet.add_number(key, value, source="monte_carlo_simulation", unit="USD")
    for key, value in (summary.get("portfolio_metrics") or {}).items():
        packet.add_number(key, value, source="monte_carlo_simulation")
    for key, value in (summary.get("path_dependent") or {}).items():
        packet.add_number(key, value, source="monte_carlo_simulation")

    packet.citations = citations or []
    return packet


def build_query_packet(
    question: str,
    *,
    citations: list[dict[str, Any]],
    query_results: pd.DataFrame | None = None,
    sql: str | None = None,
    dictionary_fields: list[str] | None = None,
    rule_ids: list[str] | None = None,
) -> EvidencePacket:
    packet = EvidencePacket(task="grounded_qa")
    packet.subject = {"question": question}
    if sql:
        packet.facts["executed_sql"] = sql
    if query_results is not None and not query_results.empty:
        head = query_results.head(50)
        packet.facts["query_row_count"] = int(len(query_results))
        packet.facts["query_columns"] = list(head.columns)
        for column in head.columns:
            series = pd.to_numeric(head[column], errors="coerce")
            if series.notna().any():
                for i, value in series.dropna().head(20).items():
                    packet.add_number(f"{column}_row{i}", value, source="duckdb_query")
            packet.field_names.append(str(column))
        packet.facts["query_preview"] = head.head(10).to_dict(orient="records")
    packet.citations = citations
    if dictionary_fields:
        packet.field_names.extend(dictionary_fields)
    if rule_ids:
        packet.rule_ids.extend(rule_ids)
    return packet
