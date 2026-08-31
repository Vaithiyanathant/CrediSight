"""Portfolio endpoints."""
from __future__ import annotations

import time
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Query

from lpie.api.deps import PredictionDep
from lpie.api.metrics import METRICS
from lpie.api.schemas import (
    LoanState,
    PortfolioSummaryResponse,
    WatchlistEntry,
    WatchlistResponse,
)
from lpie.core.exceptions import DataNotFoundError
from lpie.core.logging import get_logger
from lpie.core.timing import Timer, utcnow_iso
from lpie.serving.scorer import PredictionScorer, build_features_for

log = get_logger(__name__)
router = APIRouter(prefix="/api/v1/portfolio", tags=["portfolio"])

ACTIVE_STATES = {"Current", "30DPD", "60DPD", "90DPD", "Default"}
TERMINAL_STATES = {"Prepaid", "Closed"}

# ── In-memory TTL cache (5 minutes) ─────────────────────────────────────────
_PORTFOLIO_CACHE: dict = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes


def _sf(v, d=0.0):
    try:
        f = float(v)
        return d if not np.isfinite(f) else f
    except (TypeError, ValueError):
        return d


def _score_portfolio(state, month_index: int | None = None):
    """Score all loans at `month_index` (default: latest), with a 5-min cache.

    The cache is keyed by month. It previously held a single unkeyed entry, so
    the month a caller asked for was silently ignored and whatever month was
    scored first was returned for every subsequent request.
    """
    store_months = state.feature_store_months()
    if not store_months:
        raise DataNotFoundError("Feature store is empty. Run `make features`.")
    if month_index is None:
        month = max(store_months)
    elif month_index in store_months:
        month = int(month_index)
    else:
        raise DataNotFoundError(
            f"No features at month_index {month_index}. "
            f"Available: {min(store_months)}-{max(store_months)}."
        )

    now = time.time()
    entry = _PORTFOLIO_CACHE.get(month)
    if entry is not None and (now - entry["ts"]) < _CACHE_TTL_SECONDS:
        return entry["result"], month

    features = build_features_for(state, months=[month])
    if features.empty:
        raise DataNotFoundError(f"No features at month {month}")
    scorer = PredictionScorer(state)
    result = scorer.score(features, include_survival=False)

    _PORTFOLIO_CACHE[month] = {"result": result, "ts": now}
    # Bound the cache: one scored panel month is large.
    if len(_PORTFOLIO_CACHE) > 4:
        oldest = min(_PORTFOLIO_CACHE, key=lambda k: _PORTFOLIO_CACHE[k]["ts"])
        _PORTFOLIO_CACHE.pop(oldest, None)
    return result, month


@router.get(
    "/summary",
    response_model=PortfolioSummaryResponse,
    summary="Aggregate portfolio risk summary",
)
def portfolio_summary(
    state: PredictionDep,
    month_index: Annotated[int | None, Query(ge=1, le=600)] = None,
) -> PortfolioSummaryResponse:
    timer = Timer()
    result, latest = _score_portfolio(state, month_index)
    frame = result.frame

    n_total = len(frame)
    is_terminal = frame["is_terminal"].fillna(False)
    n_active = int((~is_terminal).sum())
    n_terminal = int(is_terminal.sum())

    total_balance = float(frame["current_balance"].fillna(0.0).sum())
    active_frame = frame[~is_terminal]

    def _rate(col):
        return float(active_frame[col].fillna(0.0).mean()) if not active_frame.empty else 0.0

    delinquency_rate = float(
        active_frame["current_status"].isin({"30DPD", "60DPD", "90DPD"}).mean()
    ) if not active_frame.empty else 0.0

    action_dist = frame["reviewer_action"].value_counts().to_dict()
    state_dist = frame["current_status"].value_counts().to_dict()

    risk_dist = {"low": 0, "medium": 0, "high": 0}
    for _, val in frame["prob_next_12m_default"].fillna(0.0).items():
        if val < 0.1:
            risk_dist["low"] += 1
        elif val < 0.3:
            risk_dist["medium"] += 1
        else:
            risk_dist["high"] += 1

    dq_dist: dict[str, int] = {}
    if "dq_grade" in frame.columns:
        dq_dist = frame["dq_grade"].fillna("unknown").value_counts().to_dict()

    el = float(frame["expected_loss"].fillna(0.0).sum())
    el_pct = (el / total_balance * 100) if total_balance > 0 else None

    # Segment breakdown by credit_score_band from features
    segments: dict[str, list[dict[str, Any]]] = {}
    features = state.features(months=[latest])
    if "credit_score_band" in features.columns:
        merged = frame.merge(features[["loan_id", "month_index", "credit_score_band"]], on=["loan_id", "month_index"], how="left")
        for band, grp in merged.groupby("credit_score_band", dropna=False):
            segments.setdefault("credit_score_band", []).append({
                "value": str(band),
                "n_loans": len(grp),
                "mean_default_prob": float(grp["prob_next_12m_default"].fillna(0).mean()),
                "expected_loss": float(grp["expected_loss"].fillna(0).sum()),
            })

    METRICS.increment("lpie_portfolio_summary_requests_total")
    return PortfolioSummaryResponse(
        as_of_month_index=latest,
        reporting_month=str(state.settings.get("data.submission_reporting_month", "")),
        model_version=result.model_version,
        total_loans=n_total,
        total_balance=total_balance,
        active_loans=n_active,
        terminal_loans=n_terminal,
        delinquency_rate=delinquency_rate,
        projected_default_rate=_rate("prob_next_12m_default"),
        projected_prepayment_rate=_rate("prob_next_12m_prepayment"),
        expected_loss=el,
        expected_loss_pct_of_balance=el_pct,
        risk_distribution=risk_dist,
        reviewer_action_distribution=action_dist,
        dq_distribution=dq_dist,
        state_distribution=state_dist,
        confidence_distribution={"mean": float(frame["model_confidence"].fillna(0.5).mean())},
        segments=segments,
        computed_at=utcnow_iso(),
        elapsed_ms=round(timer.stop(), 2),
    )


@router.get(
    "/watchlist",
    response_model=WatchlistResponse,
    summary="Capacity-constrained watchlist by expected loss",
)
def watchlist(
    state: PredictionDep,
    n: Annotated[int, Query(ge=1, le=1000)] = 50,
    min_default_prob: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
    status_filter: Annotated[LoanState | None, Query()] = None,
    month_index: Annotated[int | None, Query(ge=1, le=600)] = None,
) -> WatchlistResponse:
    timer = Timer()
    result, latest = _score_portfolio(state, month_index)
    frame = result.frame

    active = frame[~frame["is_terminal"].fillna(False)]
    if status_filter:
        active = active[active["current_status"] == status_filter]
    if min_default_prob > 0:
        active = active[active["prob_next_12m_default"].fillna(0) >= min_default_prob]

    # keep="all" returned MORE than n rows on ties, and expected_loss ties are
    # the norm rather than the exception here, so the capacity contract was not
    # held. Rank explicitly instead, breaking ties on the *raw* stacked score:
    # isotonic calibration is a step function, so many loans share a calibrated
    # probability that the raw score still separates. Calibration is monotone,
    # so the raw score refines the calibrated ordering and never contradicts it.
    # loan_id is the final tie-break, purely so the ranking is deterministic.
    sort_cols = ["expected_loss", "prob_next_12m_default"]
    if "prob_next_12m_default_raw" in active.columns:
        sort_cols.append("prob_next_12m_default_raw")
    sort_cols.append("loan_id")
    ranked = (
        active.sort_values(
            sort_cols,
            ascending=[False] * (len(sort_cols) - 1) + [True],
            kind="mergesort",
        )
        .head(n)
        .reset_index(drop=True)
    )

    portfolio_el = float(frame["expected_loss"].fillna(0).sum())
    watchlist_el = float(ranked["expected_loss"].fillna(0).sum())
    el_share = watchlist_el / portfolio_el if portfolio_el > 0 else None

    entries = []
    for i, row in ranked.iterrows():
        drivers = [str(row.get(f"top_driver_{j}", "") or "") for j in range(1, 4) if row.get(f"top_driver_{j}")]
        entries.append(WatchlistEntry(
            rank=int(i) + 1,
            loan_id=str(row["loan_id"]),
            month_index=int(row["month_index"]),
            current_status=str(row["current_status"]) if str(row.get("current_status")) in {"Current","30DPD","60DPD","90DPD","Default","Prepaid","Closed"} else "Current",
            current_balance=float(row.get("current_balance", 0.0) or 0.0),
            prob_next_12m_default=min(1.0, max(0.0, float(row.get("prob_next_12m_default", 0.0) or 0.0))),
            expected_loss=float(row.get("expected_loss", 0.0) or 0.0),
            anomaly_score=min(1.0, max(0.0, float(row.get("anomaly_score", 0.0) or 0.0))),
            exception_required=int(row.get("exception_required", 0) or 0),
            exception_type=str(row.get("exception_type", "") or ""),
            reviewer_action=str(row.get("reviewer_action", "No Action")),
            model_confidence=min(1.0, max(0.0, float(row.get("model_confidence", 0.5) or 0.5))),
            dq_grade=str(row.get("dq_grade", "") or "") or None,
            top_drivers=drivers,
        ))

    return WatchlistResponse(
        n=len(entries),
        ranked_by="expected_loss",
        capacity=n,
        total_expected_loss_in_watchlist=watchlist_el,
        share_of_portfolio_expected_loss=el_share,
        filters={"min_default_prob": min_default_prob, "status_filter": status_filter},
        entries=entries,
        computed_at=utcnow_iso(),
        elapsed_ms=round(timer.stop(), 2),
    )
