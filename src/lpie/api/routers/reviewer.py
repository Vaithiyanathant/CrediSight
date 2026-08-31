"""Reviewer decision endpoint (HITL loop)."""
from __future__ import annotations

from fastapi import APIRouter

from lpie.api.deps import StateDep
from lpie.api.schemas import ReviewerDecisionRequest, ReviewerDecisionResponse
from lpie.core.logging import get_logger
from lpie.core.timing import utcnow_iso

log = get_logger(__name__)
router = APIRouter(prefix="/api/v1/reviewer", tags=["reviewer"])


@router.post(
    "/decision",
    response_model=ReviewerDecisionResponse,
    summary="Record a human reviewer decision (HITL loop)",
    description=(
        "Persists the reviewer decision in the SQLite app store and closes the "
        "human-in-the-loop governance loop. `agreed_with_model` is computed "
        "automatically by comparing the human decision to the model recommendation."
    ),
)
def record_decision(request: ReviewerDecisionRequest, state: StateDep) -> ReviewerDecisionResponse:
    model_rec = request.model_recommendation
    agreed: bool | None = None
    if model_rec is not None:
        model_flag = model_rec in ("Flag", "Escalate")
        human_flag = request.human_decision in ("Confirm", "Escalate")
        agreed = model_flag == human_flag

    payload = {
        "loan_id": request.loan_id,
        "month_index": request.month_index,
        "model_recommendation": model_rec,
        "human_decision": request.human_decision,
        "rationale": request.rationale,
        "reviewer": request.reviewer,
        "decided_at": utcnow_iso(),
        "agreed_with_model": agreed,
        "anomaly_score": float(request.anomaly_score) if request.anomaly_score is not None else None,
        "exception_type": request.exception_type,
    }
    decision_id = state.app_store.record_reviewer_decision(payload)
    stats = state.app_store.reviewer_agreement_stats()

    log.info(
        "reviewer.decision",
        id=decision_id,
        loan_id=request.loan_id,
        human=request.human_decision,
        model=model_rec,
        agreed=agreed,
    )
    return ReviewerDecisionResponse(
        id=decision_id,
        loan_id=request.loan_id,
        month_index=request.month_index,
        human_decision=request.human_decision,
        model_recommendation=model_rec,
        agreed_with_model=agreed,
        decided_at=payload["decided_at"],
        agreement_stats=stats,
    )
