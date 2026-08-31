"""Groq LLM client — primary AI provider.

Uses the Groq API (https://console.groq.com) with the model configured in
`copilot.model` (default: qwen/qwen3.8-27b).  Falls back to Anthropic if
GROQ_API_KEY is absent and ANTHROPIC_API_KEY is set.  If neither key is
configured the service still works end-to-end via the deterministic template
fallback — the governance path is exercised identically either way.

Error codes returned to callers:
  GROQ_API_KEY_EXHAUSTED  — 429 / quota exceeded on the Groq side
  GROQ_MODEL_NOT_FOUND    — 404 from Groq (model decommissioned / wrong ID)
  GROQ_AUTH_ERROR         — 401 from Groq (invalid key)
  LLM_UNAVAILABLE         — no provider configured or all providers failed
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from lpie.copilot.audit import PromptAuditor
from lpie.copilot.evidence import EvidencePacket
from lpie.copilot.prompts import (
    QA_PROMPT,
    REVIEWER_NOTE_PROMPT,
    RISK_NARRATIVE_PROMPT,
    SCENARIO_SUMMARY_PROMPT,
    SQL_GENERATION_PROMPT,
    build_prompt,
    build_regeneration_prompt,
)
from lpie.copilot.rag import RAGIndex
from lpie.copilot.verifier import (
    VERDICT_FALLBACK,
    VERDICT_PASS,
    VERDICT_REGENERATED,
    NumericVerifier,
    fallback_answer,
    fallback_reviewer_note,
    fallback_scenario_summary,
)
from lpie.core.config import Settings, get_settings
from lpie.core.logging import get_logger
from lpie.data.duckdb_store import READ_ONLY_ALLOWLIST, DuckDBStore, validate_read_only_sql

log = get_logger(__name__)

TASK_TEMPLATES = {
    "reviewer_note": REVIEWER_NOTE_PROMPT,
    "risk_narrative": RISK_NARRATIVE_PROMPT,
    "scenario_summary": SCENARIO_SUMMARY_PROMPT,
    "grounded_qa": QA_PROMPT,
}
# The "name at least N fields" rule exists so a REVIEWER NOTE is actionable
# without opening another screen. Grounded Q&A has no reviewer to act: a correct
# answer to "what does days_past_due mean" names exactly one field, and the rule
# was rejecting those and serving the template fallback instead. Scoping the
# rule to the tasks it was written for is a correction, not a relaxation —
# every substantive check (unsupported numbers, unknown fields, unfired rule
# IDs, forbidden phrases, causal claims, the banner) still applies to all tasks.
SPECIFICITY_REQUIRED_TASKS = frozenset({"reviewer_note", "risk_narrative", "scenario_summary"})

FALLBACKS = {
    "reviewer_note": fallback_reviewer_note,
    "risk_narrative": fallback_reviewer_note,
    "scenario_summary": fallback_scenario_summary,
    "grounded_qa": fallback_answer,
}


@dataclass
class CopilotResponse:
    task: str
    answer: str
    verdict: str
    verification: dict[str, Any]
    citations: list[dict[str, Any]] = field(default_factory=list)
    evidence_hash: str = ""
    model: str = ""
    provider: str = ""
    latency_ms: int = 0
    prompt_log_id: int | None = None
    regenerated: bool = False
    used_fallback: bool = False
    llm_error: str | None = None
    llm_error_code: str | None = None
    raw_output: str | None = None
    sql: str | None = None
    query_preview: list[dict[str, Any]] = field(default_factory=list)
    banner: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "answer": self.answer,
            "banner": self.banner,
            "verifier": {
                "verdict": self.verdict,
                "regenerated": self.regenerated,
                "used_fallback": self.used_fallback,
                **self.verification,
            },
            "llm_error": self.llm_error,
            "llm_error_code": self.llm_error_code,
            "citations": self.citations,
            "evidence_hash": self.evidence_hash,
            "model": self.model,
            "provider": self.provider,
            "latency_ms": self.latency_ms,
            "prompt_log_id": self.prompt_log_id,
            "sql": self.sql,
            "query_preview": self.query_preview,
        }


class GroqClient:
    """Primary LLM client — Groq first, Anthropic as fallback.

    Graceful degradation:
      1. Groq  (GROQ_API_KEY set)
      2. Anthropic (ANTHROPIC_API_KEY set)
      3. Deterministic fallback (no provider key)
    """

    GROQ_DEFAULT_MODEL = "qwen/qwen3.8-27b"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        cfg = self.settings.section("copilot")
        self.model = str(cfg.get("model", self.GROQ_DEFAULT_MODEL))
        self.temperature = float(cfg.get("temperature", 0.2))
        self.max_tokens = int(cfg.get("max_tokens", 1200))
        self.timeout = float(cfg.get("timeout_s", 45))
        self._groq_client = None
        self._anthropic_client = None
        self._last_error: str | None = None

    @property
    def groq_key(self) -> str | None:
        return self.settings.groq_api_key

    @property
    def anthropic_key(self) -> str | None:
        return self.settings.anthropic_api_key

    @property
    def available(self) -> bool:
        return bool(self.groq_key) or bool(self.anthropic_key)

    @property
    def provider(self) -> str:
        if self.groq_key:
            return "groq"
        if self.anthropic_key:
            return "anthropic"
        return "deterministic_fallback"

    def _ensure_groq(self):
        if self._groq_client is None:
            import groq as _groq_sdk
            self._groq_client = _groq_sdk.Groq(api_key=self.groq_key, timeout=self.timeout)
        return self._groq_client

    def _ensure_anthropic(self):
        if self._anthropic_client is None:
            import anthropic as _anthropic_sdk
            self._anthropic_client = _anthropic_sdk.Anthropic(
                api_key=self.anthropic_key, timeout=self.timeout
            )
        return self._anthropic_client

    # ── public generate ───────────────────────────────────────────────────
    def generate(self, system: str, user: str) -> dict[str, Any]:
        """Try Groq, fall back to Anthropic, return structured error on failure."""
        if not self.available:
            return {
                "text": None,
                "error": (
                    "No LLM provider configured. "
                    "Set GROQ_API_KEY in your .env file to enable AI-generated responses."
                ),
                "error_code": "LLM_UNAVAILABLE",
                "input_tokens": None,
                "output_tokens": None,
            }

        if self.groq_key:
            result = self._call_groq(system, user)
            if result["text"] is not None:
                return result
            log.warning("copilot.groq_failed", error=result.get("error"),
                        error_code=result.get("error_code"))
            if not self.anthropic_key:
                return result

        if self.anthropic_key:
            return self._call_anthropic(system, user)

        return {"text": None, "error": "All LLM providers failed.",
                "error_code": "LLM_UNAVAILABLE",
                "input_tokens": None, "output_tokens": None}

    def _call_groq(self, system: str, user: str) -> dict[str, Any]:
        try:
            client = self._ensure_groq()
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            text = resp.choices[0].message.content or ""
            usage = resp.usage
            return {
                "text": text,
                "error": None,
                "error_code": None,
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
            }
        except Exception as exc:
            error_code, message = _classify_groq_error(exc)
            self._last_error = message
            log.warning("copilot.generation_failed", provider="groq",
                        error_code=error_code, error=message)
            return {"text": None, "error": message, "error_code": error_code,
                    "input_tokens": None, "output_tokens": None}

    def _call_anthropic(self, system: str, user: str) -> dict[str, Any]:
        try:
            client = self._ensure_anthropic()
            message = client.messages.create(
                model="claude-sonnet-5",
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(
                b.text for b in message.content if getattr(b, "type", "") == "text"
            )
            return {
                "text": text,
                "error": None,
                "error_code": None,
                "input_tokens": getattr(message.usage, "input_tokens", None),
                "output_tokens": getattr(message.usage, "output_tokens", None),
            }
        except Exception as exc:
            message = _redact(str(exc))
            self._last_error = message
            log.warning("copilot.generation_failed", provider="anthropic", error=message)
            return {"text": None, "error": message, "error_code": "ANTHROPIC_ERROR",
                    "input_tokens": None, "output_tokens": None}

    def health(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "groq_key_configured": bool(self.groq_key),
            "anthropic_key_configured": bool(self.anthropic_key),
            "last_error": self._last_error,
            "note": (
                None if self.available
                else (
                    "Set GROQ_API_KEY in your .env file to enable AI-generated responses. "
                    "Deterministic fallback responses are served in the meantime."
                )
            ),
        }


_KEY_RE = re.compile(r"(gsk_|sk-ant-)[A-Za-z0-9_\-]{8,}")


def _redact(text: str) -> str:
    """Provider errors can echo request context. Never let a key reach a client."""
    return _KEY_RE.sub(lambda m: f"{m.group(1)}<redacted>", text)


def _classify_groq_error(exc: Exception) -> tuple[str, str]:
    """Map a Groq exception → (error_code, human_readable_message)."""
    msg = _redact(str(exc))
    low = msg.lower()
    if "429" in msg or "rate_limit" in low or "quota" in low or "exhausted" in low:
        return ("GROQ_API_KEY_EXHAUSTED",
                f"Groq API quota exhausted or rate-limited — please wait and retry. ({msg})")
    if "401" in msg or "invalid_api_key" in low or ("auth" in low and "error" in low):
        return ("GROQ_AUTH_ERROR",
                f"Groq API key is invalid or revoked — update GROQ_API_KEY in .env. ({msg})")
    if "404" in msg or "model_not_found" in low or "decommissioned" in low:
        model_hint = msg.split("`")[1] if "`" in msg else "unknown"
        return ("GROQ_MODEL_NOT_FOUND",
                f"Groq model '{model_hint}' is not available. "
                "Update copilot.model in config/config.yaml. "
                f"Available: qwen/qwen3.8-27b, groq/compound-mini. ({msg})")
    if "400" in msg:
        return ("GROQ_BAD_REQUEST", f"Groq rejected the request: {msg}")
    if "timeout" in low or "timed out" in low:
        return ("GROQ_TIMEOUT", f"Groq API request timed out: {msg}")
    return ("GROQ_ERROR", f"Groq API error: {msg}")


# ── backwards-compat alias so any code referencing AnthropicClient still works ──
AnthropicClient = GroqClient


class CopilotService:
    """Governed copilot: evidence → RAG → generation → numeric verifier → audit log."""

    def __init__(
        self,
        settings: Settings | None = None,
        rag: RAGIndex | None = None,
        client: GroqClient | None = None,
        auditor: PromptAuditor | None = None,
        store: DuckDBStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.rag = rag or RAGIndex(self.settings)
        self.client = client or GroqClient(self.settings)
        self.verifier = NumericVerifier(self.settings)
        self.auditor = auditor or PromptAuditor()
        self.store = store
        self.banner = str(self.settings.get("copilot.banner", "AI RECOMMENDATION — NOT A DECISION"))
        self.max_regenerations = int(self.settings.get("copilot.verifier.max_regenerations", 1))
        self._cache: dict[str, CopilotResponse] = {}
        self._cache_limit = 256

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        if not self.rag.is_built:
            return []
        return self.rag.search(query, k)

    # ------------------------------------------------------------------ #
    def run(
        self,
        task: str,
        packet: EvidencePacket,
        *,
        question: str | None = None,
        request_id: str | None = None,
        use_cache: bool = True,
        require_banner: bool = True,
        require_specificity: bool | None = None,
    ) -> CopilotResponse:
        """Generate, verify, regenerate once on failure, fall back, and log."""
        if require_specificity is None:
            require_specificity = task in SPECIFICITY_REQUIRED_TASKS
        cache_key = f"{task}:{packet.hash()}:{question or ''}"
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            log.info("copilot.cache_hit", task=task)
            return cached

        started = time.perf_counter()
        template = TASK_TEMPLATES.get(task, QA_PROMPT)
        packet_text = packet.render()
        kwargs = {"question": question} if "{question}" in template else {}
        user_prompt = build_prompt(template, packet_text, banner=self.banner, **kwargs)
        system_prompt = f"Respond as a governed risk analyst. Begin every response with: {self.banner}"

        raw_output: str | None = None
        regenerated_output: str | None = None
        verdict = VERDICT_FALLBACK
        input_tokens = output_tokens = None

        generation = self.client.generate(system_prompt, user_prompt)
        raw_output = generation["text"]
        input_tokens, output_tokens = generation["input_tokens"], generation["output_tokens"]
        # A fallback caused by an exhausted or revoked key looks identical to a
        # fallback caused by a verifier rejection unless the provider error is
        # carried out to the caller. Operators cannot fix what they cannot see.
        llm_error = generation.get("error")
        llm_error_code = generation.get("error_code")

        verification = None
        final_output: str | None = None

        if raw_output:
            verification = self.verifier.verify(
                raw_output, packet, require_banner=require_banner,
                require_specificity=require_specificity,
            )
            if verification.passed:
                verdict, final_output = VERDICT_PASS, raw_output
            elif self.max_regenerations > 0:
                log.info("copilot.regenerating", task=task,
                         failures=[f.kind for f in verification.failures])
                retry_prompt = build_regeneration_prompt(
                    user_prompt, raw_output, verification.violation_summary()
                )
                retry = self.client.generate(system_prompt, retry_prompt)
                regenerated_output = retry["text"]
                if regenerated_output:
                    retry_verification = self.verifier.verify(
                        regenerated_output, packet, require_banner=require_banner,
                        require_specificity=require_specificity,
                    )
                    if retry_verification.passed:
                        verdict, final_output = VERDICT_REGENERATED, regenerated_output
                        verification = retry_verification
                    else:
                        verification = retry_verification

        if final_output is None:
            # Never surface unverified text. The fallback is built from the same
            # packet, so it is correct by construction, and it is verified too.
            fallback_fn = FALLBACKS.get(task, fallback_answer)
            final_output = fallback_fn(packet, self.banner)
            verdict = VERDICT_FALLBACK
            if verification is None:
                verification = self.verifier.verify(
                    final_output, packet, require_banner=False, require_specificity=False
                )

        latency_ms = int((time.perf_counter() - started) * 1000)
        prompt_log_id = self.auditor.record(
            task=task,
            model=self.client.model,
            provider=self.client.provider,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            packet=packet,
            citations=packet.citations,
            raw_output=raw_output,
            verification=verification,
            regenerated_output=regenerated_output,
            final_output=final_output,
            accepted=verdict in (VERDICT_PASS, VERDICT_REGENERATED),
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            request_id=request_id,
        )

        response = CopilotResponse(
            task=task,
            answer=final_output,
            verdict=verdict,
            verification=verification.to_dict() if verification else {},
            citations=packet.citations,
            evidence_hash=packet.hash(),
            model=self.client.model,
            provider=self.client.provider,
            latency_ms=latency_ms,
            prompt_log_id=prompt_log_id,
            regenerated=verdict == VERDICT_REGENERATED,
            used_fallback=verdict == VERDICT_FALLBACK,
            llm_error=llm_error,
            llm_error_code=llm_error_code,
            raw_output=raw_output,
            banner=self.banner,
        )
        if use_cache:
            if len(self._cache) >= self._cache_limit:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = response
        return response

    # ------------------------------------------------------------------ #
    def natural_language_query(
        self, question: str, *, max_rows: int = 200
    ) -> tuple[str | None, pd.DataFrame | None, str | None]:
        """LLM writes SQL; the gate validates it; DuckDB executes it read-only.

        The model never invents an aggregate — it writes a query that is parsed,
        allowlisted, executed, and whose results are then handed back as evidence.
        A generated query is verifiable and re-runnable; a generated number is not.
        """
        if self.store is None:
            return None, None, "No analytical store is attached to the copilot service."
        if not self.client.available:
            return None, None, "SQL generation requires a configured LLM provider."

        schema = self._schema_text()
        prompt = SQL_GENERATION_PROMPT.format(
            tables=", ".join(sorted(READ_ONLY_ALLOWLIST)),
            max_rows=max_rows,
            schema=schema,
            question=question,
        )
        generation = self.client.generate("You write read-only DuckDB SQL and nothing else.", prompt)
        if not generation["text"]:
            return None, None, generation.get("error") or "SQL generation returned no text"

        candidate = generation["text"].strip()
        candidate = candidate.removeprefix("```sql").removeprefix("```").removesuffix("```").strip()
        try:
            sql = validate_read_only_sql(candidate)
        except Exception as exc:
            log.warning("copilot.sql_rejected", error=str(exc))
            return candidate, None, f"Generated SQL was rejected by the safety gate: {exc}"

        try:
            frame = self.store.query(f"SELECT * FROM ({sql}) LIMIT {int(max_rows)}")
        except Exception as exc:
            return sql, None, f"Generated SQL failed to execute: {exc}"
        return sql, frame, None

    def _schema_text(self) -> str:
        if self.store is None:
            return ""
        lines = []
        for table in sorted(READ_ONLY_ALLOWLIST):
            if not self.store.table_exists(table):
                continue
            try:
                columns = self.store.query(f"SELECT * FROM {table} LIMIT 0").columns.tolist()
            except Exception:
                continue
            lines.append(f"{table}({', '.join(columns[:40])})")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    def health(self) -> dict[str, Any]:
        return {
            "llm": self.client.health(),
            "rag": self.rag.health(),
            "verifier": {
                "numeric_rel_tolerance": self.verifier.rel_tolerance,
                "numeric_abs_tolerance": self.verifier.abs_tolerance,
                "max_regenerations": self.max_regenerations,
                "banner": self.banner,
                "n_forbidden_phrases": len(self.verifier.forbidden),
            },
            "audit": self.auditor.stats(),
            "response_cache_size": len(self._cache),
        }
