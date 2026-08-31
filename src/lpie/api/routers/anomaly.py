"""Anomaly and exception endpoints."""
from __future__ import annotations

from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Query

from lpie.anomaly.cards import build_card
from lpie.api.deps import AnomalyDep
from lpie.api.metrics import METRICS
from lpie.api.schemas import AnomalyCardResponse, AnomalyEntry, AnomalyListResponse
from lpie.core.exceptions import DataNotFoundError
from lpie.core.logging import get_logger
from lpie.core.timing import Timer
from lpie.serving.scorer import PredictionScorer, build_features_for

log = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["anomaly"])


def _sf(v, d=0.0):
    try:
        f = float(v)
        return d if not __import__("numpy").isfinite(f) else f
    except (TypeError, ValueError):
        return d


@router.get(
    "/anomalies",
    response_model=AnomalyListResponse,
    summary="Ranked anomaly list",
)
def list_anomalies(
    state: AnomalyDep,
    limit: Annotated[int, Query(ge=1, le=1000)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    anomaly_type: Annotated[str | None, Query(alias="type")] = None,
    min_score: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
) -> AnomalyListResponse:
    timer = Timer()
    store_months = state.feature_store_months()
    if not store_months:
        raise DataNotFoundError("Feature store is empty")
    latest = max(store_months)
    features = build_features_for(state, months=[latest])
    if features.empty:
        raise DataNotFoundError("No features in store")

    scorer = PredictionScorer(state)
    result = scorer.score(features, include_survival=False)
    frame = result.frame

    filtered = frame[frame["anomaly_score"].fillna(0) >= min_score]
    if anomaly_type:
        filtered = filtered[filtered["anomaly_tier"].fillna("").str.contains(anomaly_type, case=False)]
    ranked = filtered.nlargest(limit + offset, "anomaly_score", keep="all").iloc[offset:offset + limit]
    total = len(filtered)

    VALID_STATES = {"Current", "30DPD", "60DPD", "90DPD", "Default", "Prepaid", "Closed"}
    entries = []
    for _, row in ranked.iterrows():
        status = str(row.get("current_status", "Current"))
        action = str(row.get("reviewer_action", "No Action"))
        entries.append(AnomalyEntry(
            loan_id=str(row["loan_id"]),
            month_index=int(row["month_index"]),
            current_status=status if status in VALID_STATES else "Current",
            anomaly_score=min(1.0, max(0.0, float(row.get("anomaly_score", 0.0) or 0.0))),
            anomaly_tier=str(row.get("anomaly_tier", "unsupervised") or "unsupervised"),
            exception_required=int(row.get("exception_required", 0) or 0),
            exception_type=str(row.get("exception_type", "") or ""),
            rule_severity=float(row.get("rule_severity", 0.0) or 0.0),
            worst_severity=str(row.get("worst_severity", "NONE") or "NONE"),
            current_balance=float(row.get("current_balance", 0.0) or 0.0),
            dq_score=float(row.get("dq_score", 0.0) or 0.0) or None,
            dq_grade=str(row.get("dq_grade", "") or "") or None,
            reviewer_action=action if action in {"No Action", "Flag", "Escalate"} else "No Action",
            rules_fired=[r for r in [str(row.get(f"rule_{i}", "") or "") for i in range(1, 4)] if r],
        ))

    detectors = {
        "isolation_forest": state.anomaly is not None,
        "ecod": state.anomaly is not None,
        "autoencoder": state.anomaly is not None,
        "mahalanobis": state.anomaly is not None,
    }
    METRICS.increment("lpie_anomaly_list_requests_total")
    return AnomalyListResponse(
        n=len(entries),
        total_matching=total,
        filters={"limit": limit, "offset": offset, "type": anomaly_type, "min_score": min_score},
        entries=entries,
        detectors_available=detectors,
        elapsed_ms=round(timer.stop(), 2),
    )


@router.get(
    "/anomalies/{loan_id}/{month_index}",
    response_model=AnomalyCardResponse,
    summary="Full reviewer card for one anomaly",
)
def anomaly_card(loan_id: str, month_index: int, state: AnomalyDep) -> AnomalyCardResponse:
    timer = Timer()
    features = build_features_for(state, loan_ids=[loan_id], months=[month_index])
    if features.empty:
        raise DataNotFoundError(f"No features for {loan_id} at month {month_index}")

    scorer = PredictionScorer(state)
    result = scorer.score(features, include_survival=False)

    panel = state.panel()
    violations = pd.DataFrame()
    if state.duckdb.row_count("dq_rule_results") > 0:
        violations = state.duckdb.query(
            "SELECT * FROM dq_rule_results WHERE loan_id = ? AND month_index = ?",
            [loan_id, month_index],
        )

    card = build_card(
        loan_id=loan_id,
        month_index=month_index,
        scored=result.frame,
        panel=panel,
        violations=violations,
    )
    if not card:
        raise DataNotFoundError(f"Could not build reviewer card for {loan_id} at month {month_index}")

    METRICS.increment("lpie_anomaly_card_requests_total")
    return AnomalyCardResponse(
        loan_id=loan_id,
        month_index=month_index,
        elapsed_ms=round(timer.stop(), 2),
        **{k: v for k, v in card.items() if k not in ("loan_id", "month_index")},
    )
