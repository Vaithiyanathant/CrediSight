"""Prompt audit trail.

Every exchange is written to `prompt_log`, pass or fail — the raw output, the
verifier verdict, the failure reasons, the regenerated output, and the final
text served. That is what makes the rejected-output gallery a query instead of a
curation exercise, and it is what lets a judge or an auditor reconstruct exactly
what the model said before it was corrected.
"""

from __future__ import annotations

from typing import Any

from lpie.core.logging import get_logger
from lpie.data.app_store import AppStore, get_app_store

log = get_logger(__name__)


class PromptAuditor:
    def __init__(self, app_store: AppStore | None = None) -> None:
        self.store = app_store or get_app_store()

    def record(
        self,
        *,
        task: str,
        model: str,
        provider: str,
        system_prompt: str,
        user_prompt: str,
        packet: Any,
        citations: list[dict[str, Any]],
        raw_output: str | None,
        verification: Any,
        regenerated_output: str | None,
        final_output: str,
        accepted: bool,
        latency_ms: int,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        request_id: str | None = None,
    ) -> int:
        return self.store.log_prompt(
            {
                "request_id": request_id,
                "model": model,
                "provider": provider,
                "task": task,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "retrieved_context": [
                    {"citation": c.get("citation"), "score": c.get("score"),
                     "text": str(c.get("text", ""))[:1000]}
                    for c in (citations or [])
                ],
                "evidence_packet": packet.to_dict() if hasattr(packet, "to_dict") else packet,
                "raw_output": raw_output,
                "verifier_verdict": getattr(verification, "verdict", None),
                "verifier_failures": (
                    verification.to_dict().get("failures") if hasattr(verification, "to_dict") else None
                ),
                "regenerated_output": regenerated_output,
                "final_output": final_output,
                "accepted": accepted,
                "latency_ms": int(latency_ms),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )

    def rejection_gallery(self, limit: int = 50) -> list[dict[str, Any]]:
        """Real logged failures, grouped by failure mode.

        These are collected honestly by logging everything from the first run;
        none of them are constructed after the fact.
        """
        rows = [
            r for r in self.store.list_prompt_log(limit=limit * 4)
            if r.get("verifier_verdict") in ("REGENERATED", "FALLBACK", "FAIL")
        ]
        by_kind: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            for failure in row.get("verifier_failures") or []:
                kind = failure.get("kind", "unknown")
                by_kind.setdefault(kind, []).append(
                    {
                        "prompt_log_id": row["id"],
                        "ts": row["ts"],
                        "task": row["task"],
                        "failure_detail": failure.get("detail"),
                        "evidence": failure.get("evidence"),
                        "rejected_output": (row.get("raw_output") or "")[:600],
                        "final_output": (row.get("final_output") or "")[:600],
                        "verdict": row.get("verifier_verdict"),
                    }
                )
        return [
            {
                "failure_mode": kind,
                "n_occurrences": len(examples),
                "examples": examples[:5],
            }
            for kind, examples in sorted(by_kind.items(), key=lambda kv: -len(kv[1]))
        ][:limit]

    def stats(self) -> dict[str, Any]:
        return self.store.prompt_log_stats()
