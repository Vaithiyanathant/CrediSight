# LLM Copilot Rejection Gallery

This document records every case where the numeric verifier caught a problematic
LLM output — either rejecting it outright (FAIL/FALLBACK) or requiring
regeneration (REGENERATED). The LLM writes prose; the ML computes numbers.
Any AI number that was not in the evidence packet is automatically caught.

## Summary

- Total copilot calls: 7
- Passed first attempt: 5
- Required regeneration: 0
- Fell back to deterministic: 0
- Hard failures: 2
- Verification failure rate: 28.6%

## Governance Principle

The numeric verifier checks every number in every LLM output against the evidence
packet before display. A number not in the packet causes immediate rejection.
This prevents hallucinated probabilities, fabricated rule IDs, and invented
percentages from reaching a reviewer.

Example: LLM claimed a 73% default probability. Verifier rejected it — 73% was
not in the evidence packet. The ML model computed 8.7%. The LLM wrote prose.

## Rejected / Degraded Outputs

### Rejection 1
- **Task:** scenario_summary
- **Verdict:** FAIL
- **Model:** claude-sonnet-5
- **Timestamp:** 2026-08-29T05:18:49.216Z
- **Latency:** 0ms
- **Fix applied:** Deterministic fallback constructed from evidence packet

### Rejection 2
- **Task:** reviewer_note
- **Verdict:** FAIL
- **Model:** claude-sonnet-5
- **Timestamp:** 2026-08-29T05:18:46.162Z
- **Latency:** 0ms
- **Fix applied:** Deterministic fallback constructed from evidence packet

