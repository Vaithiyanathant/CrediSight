"""Reviewer cards — the full payload behind one anomaly.

A card is everything a human needs to make a decision in under a minute:
identity and state, the score with tier attribution, the rules fired *verbatim*
with observed vs expected, the top SHAP drivers, the nearest-normal contrast,
a 12-month balance/DPD sparkline, the servicer-vs-master reconciliation, and
the data-quality panel. The LLM-drafted note is attached separately and always
carries the recommendation banner.

Cards are stratified across every exception type and every detection tier when
generated in bulk — twenty cards that are all the same failure mode teach a
reviewer nothing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lpie.anomaly.rules_tier import fired_rules_long
from lpie.core.logging import get_logger

log = get_logger(__name__)

SPARKLINE_MONTHS = 12


def build_card(
    loan_id: str,
    month_index: int,
    *,
    scored: pd.DataFrame,
    panel: pd.DataFrame,
    violations: pd.DataFrame,
    record_scores: pd.DataFrame | None = None,
    shap_drivers: list[dict[str, Any]] | None = None,
    nearest_normal: list[dict[str, Any]] | None = None,
    reconstruction: list[dict[str, Any]] | None = None,
    servicer: pd.DataFrame | None = None,
) -> dict[str, Any]:
    row = scored[
        (scored["loan_id"] == loan_id)
        & (pd.to_numeric(scored["month_index"], errors="coerce") == month_index)
    ]
    if row.empty:
        return {}
    r = row.iloc[0]

    history = panel[
        (panel["loan_id"] == loan_id)
        & (pd.to_numeric(panel["month_index"], errors="coerce") <= month_index)
    ].sort_values("month_index").tail(SPARKLINE_MONTHS)

    dq: dict[str, Any] = {}
    if record_scores is not None and not record_scores.empty:
        hit = record_scores[
            (record_scores["loan_id"] == loan_id)
            & (pd.to_numeric(record_scores["month_index"], errors="coerce") == month_index)
        ]
        if not hit.empty:
            entry = hit.iloc[0]
            dq = {
                "dq_score": _num(entry.get("dq_score")),
                "dq_grade": entry.get("dq_grade"),
                "n_rules_violated": int(entry.get("n_rules_violated") or 0),
                "dimensions": {
                    d: _num(entry.get(d))
                    for d in ("completeness", "validity", "consistency",
                              "timeliness", "uniqueness", "cross_source")
                },
            }

    return {
        "loan_id": loan_id,
        "month_index": int(month_index),
        "reporting_month": r.get("reporting_month"),
        "current_status": r.get("current_status"),
        "current_balance": _num(r.get("current_balance")),
        "anomaly": {
            "score": _num(r.get("anomaly_score")),
            "tier": r.get("anomaly_tier"),
            "detector_scores": {
                d: _num(r.get(d)) for d in ("iforest", "ecod", "autoencoder", "self_z")
                if d in scored.columns
            },
            "rule_severity": _num(r.get("rule_severity")),
            "worst_severity": r.get("worst_severity"),
        },
        "exception": {
            "required": int(r.get("exception_required") or 0),
            "type": r.get("exception_type"),
            "source": r.get("exception_source"),
        },
        "rules_fired": fired_rules_long(violations, loan_id, month_index),
        "shap_drivers": shap_drivers or [],
        "reconstruction_breakdown": reconstruction or [],
        "nearest_normal": nearest_normal or [],
        "history": {
            "month_index": [int(m) for m in history["month_index"]],
            "current_balance": [_num(v) for v in history.get("current_balance", [])],
            "days_past_due": [_num(v) for v in history.get("days_past_due", [])],
            "current_status": [str(v) for v in history.get("current_status", [])],
        },
        "servicer_reconciliation": _servicer_panel(servicer, loan_id, month_index),
        "data_quality": dq,
        "governance": {
            "note": "Model output is a recommendation for a human reviewer, not a decision.",
            "actions": ["Confirm", "Reject", "Escalate"],
        },
    }


def _servicer_panel(
    servicer: pd.DataFrame | None, loan_id: str, month_index: int
) -> list[dict[str, Any]]:
    if servicer is None or servicer.empty:
        return []
    hits = servicer[servicer["loan_id"] == loan_id].sort_values("update_date").tail(5)
    return [
        {
            "update_date": str(row.get("update_date"))[:10],
            "servicer_name": row.get("servicer_name"),
            "reported_balance": _num(row.get("reported_balance")),
            "reported_status": row.get("reported_status"),
            "reported_rate": _num(row.get("reported_rate")),
            "conflict_type": row.get("conflict_type"),
            "stale_flag": int(row.get("stale_flag") or 0),
            "notes": row.get("notes"),
        }
        for _, row in hits.iterrows()
    ]


def _num(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, 6)


def stratified_card_selection(
    scored: pd.DataFrame, n: int = 20, *, seed: int = 0
) -> pd.DataFrame:
    """Pick n anomalies covering every exception type and detection tier.

    Taking the twenty highest scores would return twenty copies of the same
    failure mode. A reviewer queue that demonstrates coverage is worth more than
    one that demonstrates severity.
    """
    if scored.empty:
        return scored

    work = scored.sort_values("anomaly_score", ascending=False, kind="mergesort")
    groups: list[pd.DataFrame] = []
    seen: set[tuple[Any, Any]] = set()

    strata = []
    if "exception_type" in work.columns:
        strata.append("exception_type")
    if "anomaly_tier" in work.columns:
        strata.append("anomaly_tier")

    if strata:
        for _, group in work.groupby(strata, dropna=False, sort=False):
            take = group.head(max(1, n // max(work.groupby(strata, dropna=False).ngroups, 1)))
            groups.append(take)
            seen.update(zip(take["loan_id"], take["month_index"], strict=False))

    selected = pd.concat(groups, ignore_index=True) if groups else work.head(0)
    if len(selected) < n:
        remainder = work[
            ~work.set_index(["loan_id", "month_index"]).index.isin(seen)
        ].head(n - len(selected))
        selected = pd.concat([selected, remainder], ignore_index=True)
    return selected.sort_values("anomaly_score", ascending=False, kind="mergesort").head(n).reset_index(drop=True)
