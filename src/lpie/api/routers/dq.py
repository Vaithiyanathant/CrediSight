"""Data-quality summary endpoint."""

from __future__ import annotations

from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Query

from lpie.api.deps import StateDep
from lpie.api.schemas import DQSummaryResponse
from lpie.core.exceptions import DataNotFoundError
from lpie.core.timing import utcnow_iso

router = APIRouter(prefix="/api/v1/dq", tags=["data intelligence"])


@router.get(
    "/summary",
    response_model=DQSummaryResponse,
    summary="Data-quality grade distribution and top violated rules",
    description=(
        "Batch-level data quality: mean and median record score, grade distribution "
        "(A>=95, B>=85, C>=70, D>=50, F<50), per-dimension means across the six dimensions "
        "(completeness, validity, consistency, timeliness, uniqueness, cross-source), and "
        "the most-violated rules.\n\n"
        "Reads the persisted `dq_record_scores` table when it exists (populated by "
        "`make profile`) and recomputes on the fly otherwise, so the endpoint is never a "
        "dead link waiting on a batch job."
    ),
)
def dq_summary(
    state: StateDep,
    months: Annotated[str | None, Query(description="Comma-separated month_index values")] = None,
    by_month: Annotated[bool, Query(description="Include a per-month breakdown")] = True,
) -> DQSummaryResponse:
    wanted = (
        [int(m.strip()) for m in months.split(",") if m.strip().isdigit()] if months else None
    )

    scores = pd.DataFrame()
    if state.duckdb.row_count("dq_record_scores") > 0:
        clause = ""
        params: list[int] = []
        if wanted:
            clause = f"WHERE month_index IN ({', '.join(['?'] * len(wanted))})"
            params = wanted
        scores = state.duckdb.query(f"SELECT * FROM dq_record_scores {clause}", params)

    engine = state.validation_engine
    pd.DataFrame()
    summary_rules: list[dict] = []

    if scores.empty:
        panel = state.panel()
        train_max = int(state.settings.get("data.train_month_max", 36))
        frame = panel[panel["month_index"] <= train_max]
        if wanted:
            frame = frame[frame["month_index"].isin(wanted)]
        if frame.empty:
            raise DataNotFoundError("No rows available to score", details={"months": wanted})
        frame = frame.head(120_000)
        if not engine.vocabulary:
            engine.fit_vocabulary(panel[panel["month_index"] <= train_max])
        result = engine.run(
            frame, static=state.static(), servicer=state.servicer(),
            batch_id="dq_summary", emit_violations=False,
        )
        scores = result.record_scores
        summary_rules = result.summary["per_rule"][:10]
        batch = result.batch_score
    else:
        rule_counts = (
            state.duckdb.query(
                "SELECT rule_id, severity, count(*) AS n FROM dq_rule_results "
                "GROUP BY rule_id, severity ORDER BY n DESC LIMIT 10"
            )
            if state.duckdb.row_count("dq_rule_results") > 0
            else pd.DataFrame()
        )
        summary_rules = [
            {"rule_id": r["rule_id"], "severity": r["severity"], "n_violations": int(r["n"])}
            for _, r in rule_counts.iterrows()
        ]
        batch = engine.scorer.score_batch(
            scores, pd.DataFrame(index=scores.index), batch_id="stored"
        )

    counts = scores["dq_grade"].value_counts()
    n = int(len(scores))

    per_month = []
    if by_month and "month_index" in scores.columns:
        grouped = scores.groupby("month_index")["dq_score"]
        for month, values in grouped:
            month_scores = scores[scores["month_index"] == month]
            per_month.append(
                {
                    "month_index": int(month),
                    "n": int(len(values)),
                    "mean_dq": round(float(values.mean()), 4),
                    "pct_grade_a": round(
                        100.0 * float((month_scores["dq_grade"] == "A").mean()), 4
                    ),
                }
            )
        per_month.sort(key=lambda r: r["month_index"])

    dimensions = ("completeness", "validity", "consistency", "timeliness", "uniqueness", "cross_source")
    return DQSummaryResponse(
        n_records=n,
        mean_dq=round(float(scores["dq_score"].mean()), 4) if n else None,
        median_dq=round(float(scores["dq_score"].median()), 4) if n else None,
        grade=batch.get("grade"),
        grade_distribution={str(k): int(v) for k, v in counts.items()},
        pct_grade_a=round(100.0 * float(counts.get("A", 0)) / n, 4) if n else None,
        pct_grade_f=round(100.0 * float(counts.get("F", 0)) / n, 4) if n else None,
        mean_dimension_scores={
            d: round(float(scores[d].mean()), 4) for d in dimensions if d in scores.columns
        },
        top_violated_rules=summary_rules,
        n_error_violations=int(batch.get("n_error_violations", 0) or 0),
        n_warning_violations=int(batch.get("n_warning_violations", 0) or 0),
        by_month=per_month,
        computed_at=utcnow_iso(),
    )
