"""F6 — Data-quality and provenance features (14).

Records that are badly maintained often belong to loans that are badly serviced.
Feeding the DQ score into the risk model — rather than only reporting it — is a
genuine cross-task insight, and it is what lets `model_confidence` legitimately
fall when the underlying record is broken.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lpie.features.registry import FeatureSpec, spec

FAMILY = "dq"

DOC_SEVERITY = {
    "Complete": 0.0, "Stale": 1.0,
    "Missing-Income": 2.0, "Missing-Appraisal": 2.0, "Missing-ID": 3.0,
}
SOURCE_SYSTEMS = ("LOS", "Servicer-Portal", "Manual-Entry", "Batch-Upload")
RULE_FLAG_IDS = ("VR-001", "VR-002", "VR-006", "VR-007", "VR-010", "VR-012", "VR-013", "VR-014")

SPECS: list[FeatureSpec] = [
    spec("dq_score", FAMILY, "Record data-quality score 0-100", ["*rules*"]),
    spec("dq_completeness", FAMILY, "Completeness sub-score", ["*rules*"]),
    spec("dq_validity", FAMILY, "Validity sub-score", ["*rules*"]),
    spec("dq_consistency", FAMILY, "Consistency sub-score", ["*rules*"]),
    spec("dq_timeliness", FAMILY, "Timeliness sub-score", ["*rules*"]),
    spec("dq_uniqueness", FAMILY, "Uniqueness sub-score", ["*rules*"]),
    spec("dq_cross_source", FAMILY, "Cross-source sub-score", ["*rules*"]),
    spec("n_rules_violated", FAMILY, "Count of validation rules this record violates", ["*rules*"]),
    *[
        spec(f"rule_{rid.replace('-', '_').lower()}_violated", FAMILY,
             f"{rid} fired on this record", ["*rules*"])
        for rid in RULE_FLAG_IDS
    ],
    spec("days_since_last_update", FAMILY, "Staleness of the master record at month end",
         ["last_updated_at", "month_index"]),
    spec("source_system", FAMILY, "Source system of the record (native categorical)",
         ["source_system"], dtype="category", categorical=True),
    spec("source_system_changed_flag", FAMILY, "Source system differs from last month",
         ["source_system"], temporal_offset=-1),
    spec("n_source_switches_life", FAMILY, "Number of source-system switches to date",
         ["source_system"], temporal_offset=-1),
    spec("document_status_severity", FAMILY, "Document status as an ordinal severity",
         ["document_status"], ordinal=True),
    spec("document_status_missing_flag", FAMILY, "document_status itself is null",
         ["document_status"]),
]


def build(
    panel: pd.DataFrame,
    record_scores: pd.DataFrame,
    rule_passes: pd.DataFrame,
    days_since_update: pd.Series,
) -> pd.DataFrame:
    out = pd.DataFrame(index=panel.index)
    loan = panel["loan_id"]

    scores = record_scores.reset_index(drop=True)
    scores.index = panel.index
    out["dq_score"] = pd.to_numeric(scores.get("dq_score"), errors="coerce")
    for dim in ("completeness", "validity", "consistency", "timeliness", "uniqueness", "cross_source"):
        out[f"dq_{dim}"] = pd.to_numeric(scores.get(dim), errors="coerce")
    out["n_rules_violated"] = pd.to_numeric(scores.get("n_rules_violated"), errors="coerce")

    passes = rule_passes.reset_index(drop=True)
    passes.index = panel.index
    for rid in RULE_FLAG_IDS:
        col = f"rule_{rid.replace('-', '_').lower()}_violated"
        out[col] = (~passes[rid]).astype("float64") if rid in passes.columns else np.nan

    out["days_since_last_update"] = pd.to_numeric(days_since_update, errors="coerce")

    source = panel.get("source_system")
    if source is None:
        source = pd.Series(pd.NA, index=panel.index, dtype="object")
    out["source_system"] = source.astype("object")
    prev_source = source.groupby(loan, sort=False).shift(1)
    changed = ((source != prev_source) & prev_source.notna()).astype("float64")
    out["source_system_changed_flag"] = changed
    out["n_source_switches_life"] = (
        changed.groupby(loan, sort=False).cumsum().groupby(loan, sort=False).shift(1).fillna(0.0)
    )

    doc = panel.get("document_status")
    if doc is None:
        doc = pd.Series(pd.NA, index=panel.index, dtype="object")
    out["document_status_severity"] = doc.map(DOC_SEVERITY).astype("float64")
    out["document_status_missing_flag"] = doc.isna().astype("float64")
    return out
