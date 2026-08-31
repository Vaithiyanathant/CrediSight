"""The numeric verifier — deterministic, and the reason this is a governed system.

Nothing generated ever reaches a client unverified. The verifier extracts every
numeral, every field reference and every rule ID from the model's output and
checks each against the evidence packet. It is plain Python with no model in it,
which is the whole point: the component that decides whether an LLM may be
believed must not itself be an LLM.

Failure handling is fixed and non-negotiable:

    FAIL -> regenerate once, with the violations appended to the prompt
    FAIL again -> deterministic template fallback
    never -> surface unverified text

Every attempt is written to `prompt_log` whether it passed or failed, so the
rejected-output gallery is a query rather than a curation exercise.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from lpie.core.config import Settings, get_settings
from lpie.core.logging import get_logger

log = get_logger(__name__)

VERDICT_PASS = "PASS"
VERDICT_REGENERATED = "REGENERATED"
VERDICT_FALLBACK = "FALLBACK"
VERDICT_FAIL = "FAIL"

# Numerals, including percentages, currency and thousands separators.
# The pattern has three alternatives (in priority order):
#   1. Comma-formatted large numbers: 1,234,567 or $1,234,567.89
#   2. Plain integers and decimals of ANY length: 2134411.40031, 174412.2, 2024, 42, 0.087
#   3. Leading-decimal floats: .087
#
# The integer run MUST stay unbounded. Two earlier bounded variants both
# corrupted the extracted value rather than failing loudly:
#   \d{1,3} split "2024" into "202" + a "4" the lookbehind then dropped;
#   \d{1,6} split "2134411.40031" into "213441", which the verifier then
#           reported as a hallucinated figure absent from the packet.
# A truncated numeral is worse than an unparsed one: it turns every currency
# value at or above 1,000,000 (expected_loss, balances) into a false rejection,
# which is why scenario summaries could never pass verification.
NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.])[-+]?\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?"   # comma-formatted
    r"|(?<![A-Za-z0-9_.])[-+]?\$?\d+(?:\.\d+)?%?"                  # plain int/decimal, any length
    r"|(?<![A-Za-z0-9_.])[-+]?\.\d+%?"                             # leading-decimal: .087
)
# Spans that look numeric but assert nothing about the data, and so must not be
# checked against the packet:
#   dates      — "2024-05" is a quoted identifier, not a measurement;
#   ordinals   — "95th percentile" names a statistic, it does not claim 95;
#   month refs — "month 37" / "month_index 42" index the panel clock.
# Left intact each produces a token the packet cannot support, and the output is
# rejected for being correct.
NOT_A_CLAIM_PATTERN = re.compile(
    r"\b\d{4}-\d{2}(?:-\d{2})?\b"                 # ISO date / year-month
    r"|\b\d+(?:st|nd|rd|th)\b"                    # ordinal: 95th, 1st
    r"|\bP\d{1,2}\b"                              # percentile shorthand: P95
    r"|\bmonth(?:_index)?\s+\d{1,3}\b"            # month 37, month_index 42
    r"|\[[A-Za-z0-9_.\-]+#[A-Za-z0-9_.\-]+\]",     # citation: [doc_id#chunk_id]
    re.IGNORECASE,
)

# The QA prompt *requires* citations of the form [doc_id#chunk_id]. Their slugs
# carry digits and underscores, so unguarded they were mined for numbers ("31",
# "36" out of "...train-months-31-36-0") and for field names ("system_design"),
# and the model was penalised for obeying its own instructions.
CITATION_PATTERN = re.compile(r"\[[A-Za-z0-9_.\-]+#[A-Za-z0-9_.\-]+\]")
RULE_PATTERN = re.compile(r"\bVR-\d{3}\b")
FIELD_PATTERN = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`|\b([a-z][a-z0-9]*(?:_[a-z0-9]+){1,5})\b")

# Numbers a sentence may legitimately contain that are not claims about the data:
# ordinals, small counts, years, and the 0/1 of a boolean flag.
ALWAYS_ALLOWED = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 12.0, 24.0, 100.0}

# Canonical column names from the data dictionary. Referring to a real column of
# the dataset is always legitimate even when that column is not in this packet.
DICTIONARY_FIELDS: frozenset[str] = frozenset({
    "loan_id", "month_index", "reporting_month", "origination_month", "loan_age_months",
    "remaining_term_months", "original_balance", "current_balance", "interest_rate",
    "credit_score_band", "ltv_band", "dti_band", "loan_purpose", "occupancy_type",
    "property_type", "servicer_name", "current_status", "days_past_due",
    "modification_flag", "prepayment_flag", "default_flag", "loss_severity_band",
    "last_updated_at", "source_system", "document_status", "loan_term_months",
    "vintage_year", "reported_balance", "reported_status", "reported_rate",
    "conflict_type", "stale_flag", "update_date", "anomaly_score", "exception_type",
    "exception_required", "reviewer_action", "model_confidence", "dq_score", "dq_grade",
    "next_state", "scenario_name", "gdp_growth_pct", "unemployment_rate_pct",
    "hpi_change_pct", "interest_rate_shock_bps", "credit_spread_shock_bps",
    "prepayment_cpr_assumption_pct", "default_rate_multiplier",
    "delinquency_rate_multiplier", "prepayment_rate_multiplier",
})

# Decision language and causal assertion are matched by pattern, not by exact
# phrase: "recommend immediate foreclosure proceedings" must be caught by the
# same rule that catches "recommend foreclosure".
DECISION_PATTERNS = (
    r"\b(recommend|initiate|proceed\s+with|begin|start|pursue)\b[^.]{0,40}\bforeclos",
    r"\bforeclos\w*\b[^.]{0,30}\b(immediately|now|proceedings)\b",
    r"\b(loan|application)\s+(is|has\s+been)\s+(approved|denied|rejected)\b",
    r"\bwe\s+(approve|deny|reject|decline)\b",
    r"\b(approve|deny|decline|reject)\s+(the|this)\s+(loan|application|borrower)\b",
    r"\badverse\s+action\b",
    r"\b(will|is)\s+(definitely|certainly|guaranteed\s+to)\s+default\b",
    r"\bwe\s+(will|shall)\s+foreclos",
)

CAUSAL_PATTERNS = (
    r"\bcaus(?:ed|ing|es)\b",
    r"\bbecause\s+(?:the\s+)?borrower\b",
    r"\bdue\s+to\s+(?:the\s+)?borrower\b",
    r"\bled\s+to\b",
    r"\bresulted\s+in\b",
    r"\btriggered\s+by\b",
    r"\bas\s+a\s+result\s+of\s+(?:unemployment|job\s+loss|illness|divorce)\b",
    r"\b(?:lost|losing)\s+(?:their|his|her)\s+job\b",
    r"\bthe\s+reason\s+(?:for|the\s+borrower)\b",
)

FAILURE_KINDS = (
    "unsupported_number",
    "unknown_field",
    "unknown_rule_id",
    "forbidden_decision_language",
    "unsupported_causal_claim",
    "insufficient_specificity",
    "missing_governance_banner",
)


@dataclass
class VerificationFailure:
    kind: str
    detail: str
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    verdict: str
    failures: list[VerificationFailure] = field(default_factory=list)
    numbers_checked: int = 0
    numbers_matched: int = 0
    fields_checked: int = 0
    rules_checked: int = 0

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "passed": self.passed,
            "n_failures": len(self.failures),
            "failures": [f.to_dict() for f in self.failures],
            "numbers_checked": self.numbers_checked,
            "numbers_matched": self.numbers_matched,
            "fields_checked": self.fields_checked,
            "rules_checked": self.rules_checked,
        }

    def violation_summary(self) -> str:
        return "\n".join(f"- [{f.kind}] {f.detail}" for f in self.failures)


def extract_numbers(text: str) -> list[tuple[str, float]]:
    """Every numeral in the text, normalised to float, with its raw span."""
    out: list[tuple[str, float]] = []
    # Blank these out first so "2024-05" / "95th percentile" contribute no claims.
    text = NOT_A_CLAIM_PATTERN.sub(lambda m: " " * len(m.group(0)), text)
    for match in NUMBER_PATTERN.finditer(text):
        raw = match.group(0)
        cleaned = raw.replace(",", "").replace("$", "").strip()
        is_percent = cleaned.endswith("%")
        cleaned = cleaned.rstrip("%")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        out.append((raw, value / 100.0 if is_percent else value))
        if is_percent:
            # A model may write "8.7%" for a probability stored as 0.087 or as
            # 8.7. Both readings are offered; matching either is acceptable.
            out.append((raw, value))
    return out


def extract_rule_ids(text: str) -> list[str]:
    return sorted(set(RULE_PATTERN.findall(text)))


def extract_field_references(text: str) -> list[str]:
    """Snake_case tokens and backticked identifiers — candidate field names."""
    out: set[str] = set()
    text = CITATION_PATTERN.sub(" ", text)
    for backticked, snake in FIELD_PATTERN.findall(text):
        token = backticked or snake
        if token and "_" in token and len(token) > 3:
            out.add(token)
    return sorted(out)


class NumericVerifier:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        cfg = self.settings.section("copilot")
        vcfg = cfg.get("verifier", {})
        self.rel_tolerance = float(vcfg.get("numeric_rel_tolerance", 0.02))
        self.abs_tolerance = float(vcfg.get("numeric_abs_tolerance", 0.005))
        self.min_named_fields = int(vcfg.get("require_min_named_fields", 2))
        self.banner = str(cfg.get("banner", "AI RECOMMENDATION — NOT A DECISION"))
        self.forbidden = [p.lower() for p in (cfg.get("forbidden_phrases") or [])]
        self.causal_markers = [p.lower() for p in (cfg.get("causal_markers") or [])]

    # ------------------------------------------------------------------ #
    def verify(
        self,
        text: str,
        packet: Any,
        *,
        require_banner: bool = True,
        require_specificity: bool = True,
    ) -> VerificationResult:
        failures: list[VerificationFailure] = []
        lowered = text.lower()

        allowed_values = list(packet.allowed_values())
        allowed_values.extend(_embedded_numbers(packet.subject))
        allowed_values.extend(_embedded_numbers(packet.facts))
        for rule in packet.rules_fired:
            allowed_values.extend(_embedded_numbers(rule))
        # The retrieved passages are part of the evidence. The QA prompt tells the
        # model to answer from them and cite them, so a figure it quotes out of a
        # cited passage ("a 5% null rate") is grounded by definition. Checking
        # only packet.numbers rejected the model for doing exactly as instructed.
        allowed_values.extend(_citation_numbers(packet))
        allowed_fields_from_context = _citation_fields(packet)
        allowed_fields = {f.lower() for f in packet.field_names}
        allowed_fields |= {str(k).lower() for k in packet.facts}
        allowed_fields |= {str(k).lower() for k in packet.numbers}
        allowed_fields |= {str(k).lower() for k in packet.subject}
        allowed_fields |= DICTIONARY_FIELDS
        allowed_fields |= allowed_fields_from_context
        # Field names quoted inside a fired rule's observed/expected text are
        # legitimate references — the packet put them there.
        for rule in packet.rules_fired:
            for value in rule.values():
                allowed_fields |= {m.lower() for m in extract_field_references(str(value))}
        allowed_rules = {r.upper() for r in packet.rule_ids}

        # 1-3. Every numeral must appear in the packet within tolerance.
        numbers = extract_numbers(text)
        matched = 0
        seen_raw: set[str] = set()
        for raw, value in numbers:
            if raw in seen_raw:
                continue
            if self._matches(value, allowed_values):
                matched += 1
                seen_raw.add(raw)
                continue
            # Try the other reading of a percentage before failing it.
            if any(self._matches(v, allowed_values) for _, v in numbers if _ == raw):
                matched += 1
                seen_raw.add(raw)
                continue
            if value in ALWAYS_ALLOWED:
                seen_raw.add(raw)
                continue
            seen_raw.add(raw)
            failures.append(
                VerificationFailure(
                    "unsupported_number",
                    f"The value {raw} does not appear in the evidence packet.",
                    evidence=_context(text, raw),
                )
            )

        # 4. Field names must exist.
        fields = extract_field_references(text)
        for name in fields:
            if name.lower() not in allowed_fields and not self._is_derived_name(name, allowed_fields):
                failures.append(
                    VerificationFailure(
                        "unknown_field",
                        f"'{name}' is not a field in the evidence packet or the data dictionary.",
                        evidence=_context(text, name),
                    )
                )

        # 5. Rule IDs.
        #
        # "Did this rule fire?" and "does this rule exist?" are different
        # questions, and only the first has a record to check against. A packet
        # with no loan subject — grounded Q&A — has an empty rules_fired list by
        # construction, so the fired-check rejected every correct documentation
        # answer: asking "which rule covers balance exceeding original balance?"
        # and being told VR-001 is right, and was being failed for it.
        #
        # With a record present the packet's fired list stays authoritative and
        # nothing here is relaxed.
        rules = extract_rule_ids(text)
        has_record = bool(getattr(packet, "rules_fired", None)) or "loan_id" in (
            getattr(packet, "subject", None) or {}
        )
        catalogue = allowed_rules | (
            set() if has_record else {r.upper() for r in _catalogue_rule_ids(packet)}
        )
        for rule_id in rules:
            if rule_id not in catalogue:
                failures.append(
                    VerificationFailure(
                        "unknown_rule_id",
                        (
                            f"{rule_id} was cited but did not fire on this record."
                            if has_record
                            else f"{rule_id} does not appear in the retrieved context."
                        ),
                        evidence=_context(text, rule_id),
                    )
                )

        # 6-8. Forbidden decision language and unsupported causal claims.
        for phrase in self.forbidden:
            if phrase in lowered:
                failures.append(
                    VerificationFailure(
                        "forbidden_decision_language",
                        f"Output contains decision language: '{phrase}'. The system recommends; "
                        "it does not decide.",
                        evidence=_context(text, phrase),
                    )
                )
        for pattern in DECISION_PATTERNS:
            match = re.search(pattern, lowered)
            if match:
                failures.append(
                    VerificationFailure(
                        "forbidden_decision_language",
                        f"Output states or implies a decision: '{match.group(0)}'. The system "
                        "recommends for human review; it does not decide.",
                        evidence=_context(text, match.group(0)),
                    )
                )
                break

        for marker in self.causal_markers:
            if marker in lowered:
                failures.append(
                    VerificationFailure(
                        "unsupported_causal_claim",
                        f"Output asserts causation ('{marker}'). The data supports association only.",
                        evidence=_context(text, marker),
                    )
                )
                break
        else:
            for pattern in CAUSAL_PATTERNS:
                match = re.search(pattern, lowered)
                if match:
                    failures.append(
                        VerificationFailure(
                            "unsupported_causal_claim",
                            f"Output asserts causation ('{match.group(0)}'). This pack contains no "
                            "employment, income or intent data; only association is supportable.",
                            evidence=_context(text, match.group(0)),
                        )
                    )
                    break

        # 9. Specificity: a useful note names fields, not vibes.
        if require_specificity and len(fields) < self.min_named_fields:
            failures.append(
                VerificationFailure(
                    "insufficient_specificity",
                    f"Output references {len(fields)} named field(s); at least "
                    f"{self.min_named_fields} are required for a reviewer to act on it.",
                )
            )

        # 10. The governance banner must survive.
        if require_banner and self.banner.lower() not in lowered:
            failures.append(
                VerificationFailure(
                    "missing_governance_banner",
                    f"Output is missing the required banner: '{self.banner}'.",
                )
            )

        return VerificationResult(
            verdict=VERDICT_PASS if not failures else VERDICT_FAIL,
            failures=failures,
            numbers_checked=len(seen_raw),
            numbers_matched=matched,
            fields_checked=len(fields),
            rules_checked=len(rules),
        )

    # ------------------------------------------------------------------ #
    def _matches(self, value: float, allowed: list[float]) -> bool:
        """Absolute tolerance for probabilities, relative for large magnitudes.

        Relative tolerance on a probability is far too permissive: at 2%, a
        fabricated 0.73 would "match" a real 0.72 and the whole verifier would be
        theatre. Anything below 1.0 is therefore matched on absolute distance
        only; relative tolerance applies to balances, counts and currency, where
        rounding for readability is legitimate.
        """
        for candidate in allowed:
            if abs(value - candidate) <= self.abs_tolerance:
                return True
            if abs(candidate) >= 1.0 and abs(value) >= 1.0:
                scale = max(abs(candidate), 1e-9)
                if abs(value - candidate) / scale <= self.rel_tolerance:
                    return True
            # A packet probability of 0.087 may legitimately be written as 8.7.
            if abs(candidate) < 1.0 and abs(value - candidate * 100.0) <= self.abs_tolerance * 100.0:
                return True
            if abs(value) < 1.0 and abs(value * 100.0 - candidate) <= self.abs_tolerance * 100.0:
                return True
        return False

    @staticmethod
    def _is_derived_name(name: str, allowed: set[str]) -> bool:
        """Accept `p_next_state_30dpd` when `p_next_state_Current` is in the packet."""
        for candidate in allowed:
            if name.startswith(candidate[: max(len(candidate) - 6, 6)]):
                return True
            if candidate.startswith(name[: max(len(name) - 6, 6)]):
                return True
        return False


def _citation_text(packet: Any) -> str:
    """All retrieved-passage text on the packet, concatenated."""
    parts = []
    for citation in getattr(packet, "citations", None) or []:
        if not isinstance(citation, dict):
            continue
        for key in ("text", "chunk", "passage"):
            value = citation.get(key)
            if value:
                parts.append(str(value))
                break
    return "\n".join(parts)


def _catalogue_rule_ids(packet: Any) -> set[str]:
    """Rule IDs named in the retrieved documentation.

    Only consulted for packets with no loan record, where a rule ID is a
    reference to the rule catalogue rather than a claim about a specific row.
    """
    return set(RULE_PATTERN.findall(_citation_text(packet)))


def _citation_numbers(packet: Any) -> list[float]:
    """Numerals appearing in the retrieved context, which is evidence too."""
    return [value for _, value in extract_numbers(_citation_text(packet))]


def _citation_fields(packet: Any) -> set[str]:
    """Field names named in the retrieved context."""
    return {name.lower() for name in extract_field_references(_citation_text(packet))}


def _embedded_numbers(payload: Any) -> list[float]:
    """Numerals carried inside categorical values, e.g. `60DPD` -> 60.

    A note that says "the loan is 60DPD" is quoting the packet's own status
    string, not fabricating a figure, and must not be rejected for it.
    """
    out: list[float] = []
    if not isinstance(payload, dict):
        return out
    for value in payload.values():
        if isinstance(value, bool):
            out.append(float(value))
        elif isinstance(value, (int, float)) and np.isfinite(value):
            out.append(float(value))
        elif isinstance(value, str):
            for token in re.findall(r"\d+(?:\.\d+)?", value):
                try:
                    out.append(float(token))
                except ValueError:
                    continue
    return out


def _context(text: str, needle: str, width: int = 60) -> str:
    idx = text.lower().find(needle.lower())
    if idx < 0:
        return ""
    start = max(0, idx - width // 2)
    return text[start : start + width + len(needle)].replace("\n", " ").strip()


# --------------------------------------------------------------------------- #
# deterministic fallbacks — never an apology, always the real numbers
# --------------------------------------------------------------------------- #
def fallback_reviewer_note(packet: Any, banner: str) -> str:
    numbers = packet.numbers
    facts = packet.facts
    subject = packet.subject

    def number(key: str, fmt: str = "{:.3f}") -> str:
        entry = numbers.get(key)
        return fmt.format(entry["value"]) if entry else "not available"

    lines = [
        banner,
        "",
        f"Loan {subject.get('loan_id')} at reporting month {subject.get('reporting_month')} "
        f"(panel month {subject.get('month_index')}) is currently {subject.get('current_status')}.",
        f"Calibrated 12-month default probability is {number('prob_next_12m_default')}; "
        f"3-month delinquency probability is {number('prob_next_3m_delinquency')}.",
        f"The anomaly score is {number('anomaly_score')} and the record's data-quality score is "
        f"{number('dq_score', '{:.1f}')} (grade {facts.get('dq_grade', 'n/a')}).",
    ]
    if packet.rules_fired:
        fired = "; ".join(
            f"{r.get('rule_id')} [{r.get('severity')}] {r.get('name')} (observed {r.get('observed')})"
            for r in packet.rules_fired[:4]
        )
        lines.append(f"Validation rules fired: {fired}.")
    else:
        lines.append("No deterministic validation rule fired on this record.")

    drivers = [k[len("value_"):] for k in numbers if k.startswith("value_")]
    if drivers:
        lines.append("Top model drivers: " + ", ".join(drivers[:3]) + ".")

    lines.append(
        f"Recommended reviewer action: {facts.get('reviewer_action', 'No Action')}. "
        "This is a recommendation for a human reviewer, produced from model output and "
        "deterministic rules; it is not a decision and asserts association, not causation."
    )
    lines.append("")
    lines.append(
        "[Generated by the deterministic fallback template because the language model's "
        "output did not pass numeric verification.]"
    )
    return "\n".join(lines)


def fallback_scenario_summary(packet: Any, banner: str) -> str:
    numbers = packet.numbers
    subject = packet.subject

    def number(key: str, fmt: str = "{:.4f}") -> str:
        entry = numbers.get(key)
        return fmt.format(entry["value"]) if entry else "not available"

    return "\n".join(
        [
            banner,
            "",
            f"Scenario {subject.get('scenario')} was simulated over {subject.get('n_paths')} "
            f"Monte-Carlo paths and {subject.get('horizon_months')} months across "
            f"{subject.get('n_loans')} loans.",
            f"Terminal cumulative default rate: mean {number('default_rate_mean')} "
            f"[P5 {number('default_rate_p5')}, P95 {number('default_rate_p95')}].",
            f"Terminal cumulative prepayment rate: mean {number('prepayment_rate_mean')}.",
            f"Mean expected loss: {number('mean_expected_loss', '{:,.0f}')} USD; "
            f"VaR(95) {number('var_95', '{:,.0f}')} USD; "
            f"Expected Shortfall(95) {number('expected_shortfall_95', '{:,.0f}')} USD.",
            "",
            "[Generated by the deterministic fallback template because the language model's "
            "output did not pass numeric verification.]",
        ]
    )


def fallback_answer(packet: Any, banner: str) -> str:
    lines = [banner, ""]
    if packet.citations:
        lines.append("The indexed documentation contains the following relevant passages:")
        for i, citation in enumerate(packet.citations[:3], 1):
            # Support both {chunk, source} and {text, citation} key formats
            text = (
                citation.get("chunk")
                or citation.get("text")
                or citation.get("passage")
                or ""
            )
            src = citation.get("source") or citation.get("title") or f"source {i}"
            excerpt = str(text).strip()[:250].replace("\n", " ")
            # The source was resolved but never rendered, so the fallback's
            # citations were unattributable — the one thing a citation is for.
            lines.append(f"[{i}] ({src}) {excerpt}")
    else:
        lines.append(
            "No indexed passage matched this question closely enough to answer it from "
            "the documentation."
        )
    lines.append("")
    lines.append(
        "[Note: A language model API key (GROQ_API_KEY or ANTHROPIC_API_KEY) is not configured "
        "for this request, or the LLM provider returned no usable output. "
        "The passages above are retrieved verbatim from the indexed corpus. "
        "Set GROQ_API_KEY in your .env file to enable full AI-generated answers.]"
    )
    return "\n".join(lines)
