"""Copilot (LLM governance) endpoints."""
from __future__ import annotations

from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Query

from lpie.api.deps import CopilotDep, StateDep
from lpie.api.metrics import METRICS
from lpie.api.schemas import (
    CopilotRequest,
    CopilotResponseModel,
    PromptLogEntry,
    PromptLogResponse,
    ReviewerNoteRequest,
    ScenarioSummaryRequest,
    VerifierResult,
)
from lpie.core.config import get_settings
from lpie.core.exceptions import DataNotFoundError
from lpie.core.logging import get_logger
from lpie.core.timing import Timer

log = get_logger(__name__)
router = APIRouter(prefix="/api/v1/copilot", tags=["copilot"])


def _citations(rag_results) -> list[dict]:
    """Shape RAG hits into the keys BOTH consumers read.

    EvidencePacket.render() emits `[{citation}] {text}` and the routers were
    building `{"chunk": ..., "source": ...}` — neither key. Every retrieved
    passage therefore rendered as the literal line "[None] " and the model was
    told, in effect, that retrieval had returned nothing. It said so: grounded
    answers came back as "the retrieved context is empty" while the API response
    advertised six citations. RAG was wired to the response but never to the
    prompt. `chunk`/`source` are kept for the response body and the fallback
    renderer, which read those names.
    """
    root = str(get_settings().root)
    out = []
    for r in rag_results or []:
        text = str(r.get("text", ""))
        # source_path is absolute on disk; a client has no use for the server's
        # filesystem layout and should not be shown it.
        source = str(r.get("source_path") or r.get("title") or r.get("doc_id") or "")
        if source.startswith(root):
            source = source[len(root):].lstrip("/")
        out.append({
            "citation": r.get("citation") or f"{r.get('doc_id', '?')}#{r.get('chunk_id', '?')}",
            "text": text[:500],
            "chunk": text[:200],
            "source": source,
            "title": r.get("title", ""),
            "score": r.get("score"),
        })
    return out


def _to_copilot_response(
    resp,
    task: str,
    *,
    sql: str | None = None,
    query_preview: list[dict] | None = None,
    sql_error: str | None = None,
) -> CopilotResponseModel:
    verif = resp.verification or {}
    return CopilotResponseModel(
        task=task,
        answer=resp.answer,
        banner=resp.banner,
        verifier=VerifierResult(
            verdict=resp.verdict,
            passed=resp.verdict in ("PASS", "REGENERATED"),
            regenerated=resp.regenerated,
            used_fallback=resp.used_fallback,
            n_failures=len(verif.get("failures", [])),
            failures=verif.get("failures", []),
            numbers_checked=verif.get("numbers_checked", 0),
            numbers_matched=verif.get("numbers_matched", 0),
            fields_checked=verif.get("fields_checked", 0),
            rules_checked=verif.get("rules_checked", 0),
        ),
        citations=resp.citations,
        evidence_hash=resp.evidence_hash,
        model=resp.model,
        provider=resp.provider,
        latency_ms=resp.latency_ms,
        prompt_log_id=resp.prompt_log_id,
        sql=sql if sql is not None else resp.sql,
        query_preview=query_preview if query_preview is not None else resp.query_preview,
        llm_error_code=getattr(resp, "llm_error_code", None),
        llm_error=getattr(resp, "llm_error", None),
        governance={
            "banner_required": True,
            "llm_writes_prose_only": True,
            **({"sql_error": sql_error} if sql_error else {}),
        },
    )


def _load_query_results_into_packet(packet, df) -> None:
    """Register a generated-SQL result frame as evidence, row-aware.

    The original version keyed every number by bare column name
    (`packet.add_number(col, val)`), so a multi-row ranking result — "which
    state has the highest delinquency rate?" — overwrote the same key on every
    row and kept only the last one. A row's numbers must be scoped to whatever
    identifies that row, or a ranking answer is structurally impossible: the
    model has no way to say "VT" is the state a given rate belongs to once the
    label has been discarded.

    Also required: `source` on every `add_number` call. It is a mandatory
    keyword-only argument; omitting it raises immediately rather than silently
    degrading, so a row that fails to register is visible in the response
    (`sql_error`/logs) instead of vanishing into a caught TypeError.
    """
    if df is None or df.empty:
        return
    frame = df.head(10)
    numeric_cols = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    label_cols = [c for c in frame.columns if c not in numeric_cols]

    if len(frame) == 1 or not label_cols:
        # A single row (an aggregate) needs no row label — the column name is
        # unambiguous on its own.
        row = frame.iloc[0]
        for col in frame.columns:
            val = row[col]
            if pd.notna(val):
                try:
                    packet.add_number(str(col), float(val), source="generated_sql")
                except (TypeError, ValueError):
                    packet.facts[str(col)] = str(val)
        return

    label_col = label_cols[0]
    summary_lines = []
    for _, row in frame.iterrows():
        label = str(row[label_col])
        parts = [label]
        for col in numeric_cols:
            val = row[col]
            if pd.isna(val):
                continue
            key = f"{label}__{col}"
            try:
                packet.add_number(key, float(val), source="generated_sql")
                parts.append(f"{col}={val}")
            except (TypeError, ValueError):
                pass
        summary_lines.append(", ".join(parts))
    # A compact table the model can quote directly, since row-scoped number
    # keys (`VT__delinquency_rate_pct`) are not names a reviewer would write.
    packet.facts["query_result_rows"] = " | ".join(summary_lines)


@router.post(
    "/ask",
    response_model=CopilotResponseModel,
    summary="Grounded Q&A with RAG + numeric verifier",
)
def copilot_ask(request: CopilotRequest, state: CopilotDep) -> CopilotResponseModel:
    Timer()
    from lpie.copilot.evidence import build_query_packet
    rag_results = state.copilot.retrieve(request.question, k=request.top_k)
    citations = _citations(rag_results)
    packet = build_query_packet(
        request.question,
        citations=citations,
    )
    sql: str | None = None
    sql_error: str | None = None
    preview: list[dict] = []
    if request.use_sql:
        sql, df, sql_error = state.copilot.natural_language_query(request.question)
        if df is not None:
            preview = df.head(20).to_dict(orient="records")
            _load_query_results_into_packet(packet, df)
    resp = state.copilot.run(
        task="grounded_qa",
        packet=packet,
        question=request.question,
        use_cache=request.use_cache,
    )
    # The response contract advertises `sql` and `query_preview`. Dropping them
    # here left every use_sql=True request indistinguishable from use_sql=False:
    # the query that produced the evidence was unauditable, which defeats the
    # point of generating a re-runnable query instead of a number.
    # These are passed through rather than assigned onto `resp`, which may be a
    # shared cached object.
    METRICS.increment("lpie_copilot_requests_total")
    log.info(
        "copilot.ask", verdict=resp.verdict, latency_ms=resp.latency_ms,
        used_sql=request.use_sql, sql_rows=len(preview), sql_error=sql_error,
    )
    return _to_copilot_response(resp, "grounded_qa", sql=sql, query_preview=preview,
                                sql_error=sql_error)


@router.post(
    "/reviewer-note",
    response_model=CopilotResponseModel,
    summary="LLM-drafted reviewer note for a specific anomaly",
)
def reviewer_note(request: ReviewerNoteRequest, state: CopilotDep) -> CopilotResponseModel:
    Timer()
    from lpie.copilot.evidence import build_loan_packet
    from lpie.serving.scorer import PredictionScorer, build_features_for
    store_months = state.feature_store_months()
    if not store_months:
        raise DataNotFoundError("Feature store is empty")
    features = build_features_for(state, loan_ids=[request.loan_id], months=[request.month_index])
    scorer = PredictionScorer(state)
    result = scorer.score(features, include_survival=False) if not features.empty else None
    scored_row = result.frame.iloc[0].to_dict() if result and not result.frame.empty else {}
    violations = []
    if state.duckdb.row_count("dq_rule_results") > 0:
        df = state.duckdb.query(
            "SELECT rule_id, severity FROM dq_rule_results WHERE loan_id = ? AND month_index = ?",
            [request.loan_id, request.month_index]
        )
        violations = df.to_dict(orient="records") if not df.empty else []
    rag_results = state.copilot.retrieve(f"loan {request.loan_id} exception anomaly")
    citations = _citations(rag_results)
    rule_ids = [v["rule_id"] for v in violations if "rule_id" in v]
    packet = build_loan_packet(
        scored_row if scored_row else {"loan_id": request.loan_id, "month_index": request.month_index},
        task="reviewer_note",
        citations=citations,
        rule_ids=rule_ids if rule_ids else None,
    )
    resp = state.copilot.run("reviewer_note", packet, use_cache=request.use_cache)
    METRICS.increment("lpie_copilot_requests_total")
    return _to_copilot_response(resp, "reviewer_note")


@router.post(
    "/scenario-summary",
    response_model=CopilotResponseModel,
    summary="Business narrative for a scenario run",
)
def scenario_summary_copilot(request: ScenarioSummaryRequest, state: CopilotDep) -> CopilotResponseModel:
    Timer()
    from lpie.copilot.evidence import build_scenario_packet
    from lpie.serving.scenario_runner import ScenarioRunner
    runner = ScenarioRunner(state)
    # A scenario that does not exist, or a simulation that cannot run, must
    # surface as an error status. Swallowing it into the evidence packet
    # returned 200 with a fluent narrative written about an error string —
    # the misleading-success case this endpoint most needs to avoid.
    run = runner.run(
        scenario=request.scenario,
        n_paths=request.n_paths,
        horizon=request.horizon,
    )
    scenario_data = {
        "scenario": run.scenario,
        "summary": run.summary,
        "assumptions": run.assumptions,
    }
    rag_results = state.copilot.retrieve(f"scenario {request.scenario} macro stress")
    citations = _citations(rag_results)
    packet = build_scenario_packet(
        scenario_data.get("summary", scenario_data),
        scenario_name=request.scenario,
        assumptions=scenario_data.get("assumptions"),
        citations=citations,
    )
    resp = state.copilot.run("scenario_summary", packet, use_cache=request.use_cache)
    METRICS.increment("lpie_copilot_requests_total")
    return _to_copilot_response(resp, "scenario_summary")


@router.get(
    "/prompt-log",
    response_model=PromptLogResponse,
    summary="Full prompt audit trail",
)
def prompt_log(
    state: StateDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    verdict: Annotated[str | None, Query()] = None,
) -> PromptLogResponse:
    entries_raw = state.app_store.list_prompt_log(limit=limit, offset=offset, verdict=verdict)
    stats = state.app_store.prompt_log_stats()
    def _as_list(value: Any) -> list[dict[str, Any]]:
        return value if isinstance(value, list) else []

    entries = []
    for e in entries_raw:
        try:
            entries.append(PromptLogEntry(
                id=e["id"],
                ts=str(e.get("ts", "")),
                task=e.get("task"),
                model=e.get("model"),
                provider=e.get("provider"),
                verifier_verdict=e.get("verifier_verdict"),
                accepted=e.get("accepted"),
                latency_ms=e.get("latency_ms"),
                # The audit trail is only an audit trail if the prompt and the
                # response travel with it — these were being read out of SQLite
                # and then dropped on the floor here, so the UI could only ever
                # show a verdict with no way to see what was actually asked or
                # answered.
                system_prompt=e.get("system_prompt"),
                user_prompt=e.get("user_prompt"),
                raw_output=e.get("raw_output"),
                regenerated_output=e.get("regenerated_output"),
                final_output=e.get("final_output"),
                verifier_failures=_as_list(e.get("verifier_failures")),
                retrieved_context=_as_list(e.get("retrieved_context")),
                evidence_packet=e.get("evidence_packet") if isinstance(e.get("evidence_packet"), dict) else None,
                input_tokens=e.get("input_tokens"),
                output_tokens=e.get("output_tokens"),
                request_id=e.get("request_id"),
            ))
        except Exception:
            continue
    rejection_gallery = [
        e for e in entries_raw
        if e.get("verifier_verdict") in ("FALLBACK", "FAIL")
    ][:10]
    return PromptLogResponse(
        entries=entries,
        stats=stats,
        rejection_gallery=rejection_gallery,
        n=len(entries),
    )
