"""Drift monitoring endpoint."""

from __future__ import annotations

import re
from typing import Annotated

from fastapi import APIRouter, Query

from lpie.api.deps import StateDep
from lpie.api.metrics import METRICS
from lpie.api.schemas import DriftResponse
from lpie.core.exceptions import DataNotFoundError, InvalidRequestError
from lpie.core.timing import Timer

router = APIRouter(prefix="/api/v1", tags=["data intelligence"])

WINDOW_PATTERN = re.compile(r"^\d{1,3}-\d{1,3}$")


def _parse_window(window: str, label: str) -> tuple[int, int]:
    if not WINDOW_PATTERN.match(window):
        raise InvalidRequestError(
            f"`{label}` must be a month_index range like '31-36'",
            details={label: window},
        )
    lo, hi = (int(x) for x in window.split("-"))
    if lo > hi:
        raise InvalidRequestError(f"`{label}` start is after its end", details={label: window})
    return lo, hi


@router.get(
    "/drift",
    response_model=DriftResponse,
    summary="Feature drift between two month windows",
    description=(
        "PSI, KS, Jensen-Shannon divergence and missingness delta per feature, plus an "
        "adversarial multivariate drift AUC.\n\n"
        "PSI verdicts: `< 0.10` KEEP, `0.10-0.25` MONITOR, `> 0.25` DROP_OR_ROBUSTIFY.\n\n"
        "**Missingness delta is a first-class metric here, not an afterthought.** On this "
        "pack the null rates of `credit_score_band`, `document_status` and `days_past_due` "
        "collapse between the train tail and the test window, which makes naive `is_null` "
        "indicators the most drifted signals in the dataset. A value-only drift report "
        "would miss the failure mode most likely to break the model in production, so a "
        "feature's verdict is the worse of its value verdict and its missingness verdict.\n\n"
        "The retraining trigger fires on PSI > 0.25 across at least 3 features, or an "
        "adversarial AUC above 0.80. `month_index`, `reporting_month` and `loan_id` are "
        "excluded from every computation: they define the split, so including them would "
        "score AUC 1.0 by construction and mean nothing."
    ),
    responses={404: {"description": "One of the requested windows contains no rows"}},
)
def drift(
    state: StateDep,
    ref: Annotated[str, Query(description="Reference month_index window, e.g. '31-36'")] = "31-36",
    cur: Annotated[str, Query(description="Current month_index window, e.g. '37-42'")] = "37-42",
    adversarial: Annotated[bool, Query(description="Include the adversarial drift classifier")] = True,
) -> DriftResponse:
    from lpie.profiling.drift import drift_report

    timer = Timer()
    ref_lo, ref_hi = _parse_window(ref, "ref")
    cur_lo, cur_hi = _parse_window(cur, "cur")

    panel = state.panel()
    reference = panel[panel["month_index"].between(ref_lo, ref_hi)]
    current = panel[panel["month_index"].between(cur_lo, cur_hi)]

    if reference.empty or current.empty:
        raise DataNotFoundError(
            "One of the requested windows contains no rows",
            details={"ref": ref, "n_reference": int(len(reference)),
                     "cur": cur, "n_current": int(len(current)),
                     "available_months": [int(panel['month_index'].min()),
                                          int(panel['month_index'].max())]},
        )

    columns = [c for c in current.columns if c in reference.columns]
    report = drift_report(
        reference, current, ref_window=ref, cur_window=cur, columns=columns,
        include_adversarial=adversarial, settings=state.settings,
    )

    METRICS.increment("lpie_drift_requests_total")
    METRICS.set_gauge("lpie_drift_max_psi", float(report.get("max_psi") or 0.0))
    METRICS.set_gauge(
        "lpie_drift_retrain_required",
        1.0 if report["retraining_trigger"]["retrain_required"] else 0.0,
    )

    report["elapsed_ms"] = round(timer.stop(), 2)
    # Sanitise NaN/Inf floats to None so Python json.dumps doesn't crash
    import math
    def _clean(obj):
        if isinstance(obj, float):
            return None if not math.isfinite(obj) else obj
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj
    report = _clean(report)
    return DriftResponse(**report)
