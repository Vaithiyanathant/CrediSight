"""Regression tests for defects found in the model-endpoint audit.

One test (or one tight cluster) per bug, each asserting the *behaviour* that was
wrong — not merely that the code runs. Every test here failed before its fix.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "src")


# --------------------------------------------------------------------------- #
# BUG 1 — the Groq SDK was never declared as a dependency, so every copilot
# request silently degraded to the deterministic fallback on a clean install.
# --------------------------------------------------------------------------- #
def test_groq_sdk_is_installed_and_declared():
    import tomllib

    groq = pytest.importorskip("groq", reason="groq SDK must be installed")
    assert groq is not None

    with open("pyproject.toml", "rb") as fh:
        deps = tomllib.load(fh)["project"]["dependencies"]
    assert any(d.startswith("groq") for d in deps), (
        "groq is imported by lpie.copilot.service but is not a declared dependency; "
        "a clean install would fall back to templates for every copilot call."
    )


def test_configured_copilot_model_is_a_real_groq_model():
    """Guards against a model id that only exists in the config file."""
    from lpie.core.config import get_settings

    known = {
        "qwen/qwen3.8-27b", "qwen/qwen3.6-27b", "groq/compound", "groq/compound-mini",
        "openai/gpt-oss-120b", "openai/gpt-oss-20b", "openai/gpt-oss-safeguard-20b",
        "allam-2-7b",
    }
    cfg = get_settings().section("copilot")
    if str(cfg.get("provider", "")).lower() != "groq":
        pytest.skip("copilot provider is not groq")
    assert str(cfg.get("model")) in known, (
        f"copilot.model={cfg.get('model')!r} is not a Groq chat model; "
        "requests would 404 and every response would be a fallback."
    )


# --------------------------------------------------------------------------- #
# BUG 2 — NUMBER_PATTERN truncated integer runs, so any figure at or above
# 1,000,000 was extracted as a different number and reported as hallucinated.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,expected",
    [
        ("expected_loss_mean is 2134411.40031", 2134411.40031),
        ("the tail loss is 1942710.5", 1942710.5),
        ("a count of 12345678 rows", 12345678.0),
        ("balance 174412.2", 174412.2),
        ("$1,234,567.89 of exposure", 1234567.89),
    ],
)
def test_extract_numbers_does_not_truncate_large_values(text, expected):
    from lpie.copilot.verifier import extract_numbers

    values = [v for _, v in extract_numbers(text)]
    assert expected in values, f"{expected} not recovered from {text!r}; got {values}"


def test_large_number_present_in_packet_verifies_clean():
    """The end-to-end symptom: a correct 7-digit figure must not be rejected."""
    from lpie.copilot.evidence import EvidencePacket
    from lpie.copilot.verifier import NumericVerifier

    pkt = EvidencePacket(task="scenario_summary", subject={"scenario": "Base"})
    pkt.add_number("expected_loss_mean", 2134411.40031, source="scenario_engine")
    pkt.add_number("expected_loss_p95", 1942710.5, source="scenario_engine")

    text = (
        "The expected_loss_mean is 2134411.40031 with a 95th percentile "
        "expected_loss_p95 of 1942710.5."
    )
    result = NumericVerifier().verify(
        text, pkt, require_banner=False, require_specificity=False
    )
    unsupported = [f for f in result.failures if f.kind == "unsupported_number"]
    assert not unsupported, f"correct figures flagged as hallucinated: {unsupported}"


@pytest.mark.parametrize(
    "text",
    [
        "as of 2024-05, updated 2023-12-21",
        "the 95th percentile and the 1st decile",
        "reported at P95 and P5",
        "at month 37 and month_index 42",
    ],
)
def test_non_claims_are_not_read_as_numeric_claims(text):
    """Dates, ordinals, percentile shorthand and month references assert nothing
    about the data; checking them against the packet rejects correct output."""
    from lpie.copilot.verifier import extract_numbers

    values = [v for _, v in extract_numbers(text)]
    assert values == [], f"non-claim tokens leaked as numeric claims: {values}"


def test_fabricated_number_is_still_rejected():
    """The fix must not blunt the verifier — it is the whole governance story."""
    from lpie.copilot.evidence import EvidencePacket
    from lpie.copilot.verifier import NumericVerifier

    pkt = EvidencePacket(task="scenario_summary", subject={"scenario": "Base"})
    pkt.add_number("expected_loss_mean", 2134411.40031, source="scenario_engine")

    result = NumericVerifier().verify(
        "The expected_loss_mean is 8675309.25.", pkt,
        require_banner=False, require_specificity=False,
    )
    assert any(f.kind == "unsupported_number" for f in result.failures)


# --------------------------------------------------------------------------- #
# BUG 3 — absorbing-state rows in the isotonic fit collapsed the calibrated
# probability of most ACTIVE loans to exactly 0.0, which zeroed expected loss
# and destroyed watchlist ranking.
# --------------------------------------------------------------------------- #
def _panel(n=40000, seed=0):
    """A panel with the pack's defining shape: a large all-negative terminal
    block occupying the bottom of the raw score range."""
    rng = np.random.default_rng(seed)
    terminal = rng.random(n) < 0.55
    raw = np.where(terminal, rng.beta(1.0, 400.0, n), rng.beta(2.0, 8.0, n))
    y = np.where(terminal, 0.0, (rng.random(n) < np.clip(raw * 1.2, 0, 1)).astype(float))
    seg = pd.Series(rng.choice(["A", "B", "C"], n))
    return raw, y, seg, terminal


def test_terminal_rows_do_not_collapse_active_probabilities():
    from lpie.models.calibration import SegmentIsotonicCalibrator

    raw, y, seg, terminal = _panel()

    naive = SegmentIsotonicCalibrator()
    naive.fit(raw, y, seg)
    naive_active = np.clip(naive.transform(raw, seg), 0, 1)[~terminal]

    fixed = SegmentIsotonicCalibrator()
    fixed.fit(raw, y, seg, eligible=~terminal)
    fixed_active = np.clip(fixed.transform(raw, seg), 0, 1)[~terminal]

    zero_share = float((fixed_active == 0.0).mean())
    assert zero_share < 0.25, (
        f"{zero_share:.1%} of active loans calibrate to exactly 0.0 — "
        "expected loss and the watchlist ranking degenerate at that rate."
    )
    # And the fix must strictly improve resolution over the naive fit.
    assert len(np.unique(fixed_active)) >= len(np.unique(naive_active))


def test_calibration_preserves_ranking_over_active_loans():
    """Calibration is monotone, so it must not destroy the raw ordering."""
    from scipy.stats import spearmanr

    from lpie.models.calibration import SegmentIsotonicCalibrator

    raw, y, seg, terminal = _panel()
    cal = SegmentIsotonicCalibrator()
    cal.fit(raw, y, seg, eligible=~terminal)
    out = np.clip(cal.transform(raw, seg), 0, 1)[~terminal]

    rho = spearmanr(raw[~terminal], out).statistic
    assert rho > 0.9, f"calibration destroyed active-loan rank order (rho={rho:.3f})"


# --------------------------------------------------------------------------- #
# BUG 4 — ECE was computed by scoring the isotonic fit with its own curve, an
# identity that returns exactly 0.0 and was being published as a result.
# --------------------------------------------------------------------------- #
def test_reported_ece_is_out_of_sample_not_the_isotonic_identity():
    from lpie.models.calibration import SegmentIsotonicCalibrator

    raw, y, seg, terminal = _panel()
    cal = SegmentIsotonicCalibrator()
    cal.fit(raw, y, seg, eligible=~terminal)
    m = cal.artifact.metrics

    # Isotonic fitted values ARE the within-block observed rates, so scoring the
    # fit data with its own curve is an identity whose ECE is ~0 whatever the
    # model is worth. The published figure must not be that number.
    assert m["ece_after_in_sample"] < 1e-4, (
        "in-sample ECE is expected to be ~0 by construction; if it is not, this "
        "test's premise no longer holds"
    )
    assert m["ece_after"] > 10 * max(m["ece_after_in_sample"], 1e-9), (
        f"reported ECE {m['ece_after']} is indistinguishable from the in-sample "
        f"identity {m['ece_after_in_sample']} — it is an artifact, not a result"
    )
    assert "cross-fitted" in m["ece_estimator"]


def test_cross_fitted_calibration_is_deterministic_and_bounded():
    from lpie.models.calibration import cross_fitted_calibrated

    raw, y, seg, terminal = _panel(n=20000)
    a = cross_fitted_calibrated(raw, y, seg.to_numpy(), seed=7, eligible=~terminal)
    b = cross_fitted_calibrated(raw, y, seg.to_numpy(), seed=7, eligible=~terminal)

    assert np.array_equal(a, b), "cross-fitting must be reproducible under a fixed seed"
    assert np.isfinite(a).all()
    assert a.min() >= 0.0 and a.max() <= 1.0


def test_cross_fitting_does_not_recurse():
    """The metrics block calls the cross-fitter, whose inner fits call fit()."""
    from lpie.models.calibration import SegmentIsotonicCalibrator

    raw, y, seg, terminal = _panel(n=5000)
    cal = SegmentIsotonicCalibrator()
    art = cal.fit(raw, y, seg, eligible=~terminal)  # must not RecursionError
    assert art.is_fitted


# --------------------------------------------------------------------------- #
# BUG 5 — fallback answers resolved a citation source and then dropped it.
# --------------------------------------------------------------------------- #
def test_fallback_answer_attributes_its_citations():
    from lpie.copilot.evidence import EvidencePacket
    from lpie.copilot.verifier import fallback_answer

    pkt = EvidencePacket(task="grounded_qa", subject={"question": "what is dpd?"})
    pkt.citations = [{"chunk": "days_past_due counts days delinquent.",
                      "source": "data_dictionary.md"}]
    text = fallback_answer(pkt, "AI RECOMMENDATION — NOT A DECISION")
    assert "data_dictionary.md" in text, "citation rendered without its source"


# --------------------------------------------------------------------------- #
# BUG 6 — /api/v1/scenario/custom passed the MacroShock under a keyword
# ScenarioRunner.run() does not accept, so every request raised TypeError and
# the blanket handler reported it as 404 "unknown scenario".
# --------------------------------------------------------------------------- #
def test_custom_scenario_reaches_the_runner_with_a_macro_shock(monkeypatch):
    """Drive the handler with a stub runner and assert what it actually passes.

    Previously it passed `_custom_shock=<MacroShock>`, a keyword
    ScenarioRunner.run() does not accept: TypeError on 100% of requests,
    reported to the caller as 404 "unknown scenario".
    """
    from lpie.api.routers import scenario as scenario_router
    from lpie.api.schemas import CustomScenarioRequest
    from lpie.scenario.transmission import MacroShock

    seen = {}

    class _StubRun:
        scenario = "Severe"
        assumptions: dict = {}
        summary: dict = {}
        segments: list = []
        reanchor: dict = {}

    class _StubRunner:
        def __init__(self, state):
            pass

        def run(self, scenario, **kwargs):
            seen["scenario"] = scenario
            seen["kwargs"] = kwargs
            return _StubRun()

    monkeypatch.setattr(scenario_router, "ScenarioRunner", _StubRunner)

    request = CustomScenarioRequest(
        name="Severe", unemployment_rate_pct=12.0, default_rate_multiplier=3.0,
        n_paths=100, horizon=6, seed=1,
    )
    resp = scenario_router.run_custom_scenario(request, state=object())

    assert isinstance(seen["scenario"], MacroShock), (
        f"runner received {type(seen['scenario'])}, not the MacroShock the handler built"
    )
    assert seen["scenario"].default_rate_multiplier == 3.0
    assert seen["scenario"].unemployment_rate_pct == 12.0
    assert "_custom_shock" not in seen["kwargs"]
    assert resp.scenario == "Severe"


def test_scenario_run_error_is_not_a_404():
    """An engine fault must not be reported as a misspelled scenario name."""
    from lpie.core.exceptions import ScenarioNotFoundError, ScenarioRunError

    assert ScenarioNotFoundError.http_status == 404
    assert ScenarioRunError.http_status == 500
    assert ScenarioRunError.code != ScenarioNotFoundError.code


# --------------------------------------------------------------------------- #
# BUG 7 — re-anchoring bisected without checking the root was bracketed. For any
# benign default multiplier the target is below f(0), so scaling walked to 0.0
# and was then applied, zeroing every channel: High-Prepayment became Base.
# --------------------------------------------------------------------------- #
def _shock(**kw):
    from lpie.scenario.transmission import MacroShock

    defaults = dict(
        scenario_name="T", description="", gdp_growth_pct=0.0,
        unemployment_rate_pct=4.1, hpi_change_pct=0.0, interest_rate_shock_bps=0.0,
        credit_spread_shock_bps=0.0, prepayment_cpr_assumption_pct=8.0,
        default_rate_multiplier=1.0, delinquency_rate_multiplier=1.0,
        prepayment_rate_multiplier=1.0,
    )
    defaults.update(kw)
    return MacroShock(**defaults)


def test_unreachable_reanchor_target_falls_back_to_full_shock_not_zero():
    from lpie.scenario.transmission import reanchor_scaling

    base_rate = 0.004
    # f(scaling) is flat at the base rate: no scaling can reach 0.7 x base.
    result = reanchor_scaling(lambda scaling: base_rate, _shock(default_rate_multiplier=0.7),
                              base_rate)

    assert result["converged"] is False
    assert result["bracketed"] is False
    assert result["scaling"] == 1.0, (
        f"scaling={result['scaling']} — an unreachable target must not neutralise the "
        "scenario; scaling=0.0 zeroes the prepayment and unemployment channels too"
    )
    assert "unreachable" in result["note"]


def test_reanchor_still_solves_a_reachable_target():
    """The bracket guard must not disable re-anchoring where it genuinely works."""
    from lpie.scenario.transmission import reanchor_scaling

    base_rate = 0.004
    # A monotone response that genuinely spans the target.
    result = reanchor_scaling(lambda scaling: base_rate * (1.0 + scaling),
                              _shock(default_rate_multiplier=2.0), base_rate)

    assert result["converged"] is True
    assert result["scaling"] == pytest.approx(1.0, abs=0.05)


def test_reanchor_handles_a_decreasing_response():
    """Prepayment competes with default, so f need not increase in `scaling`."""
    from lpie.scenario.transmission import reanchor_scaling

    base_rate = 0.01
    result = reanchor_scaling(lambda scaling: base_rate * (2.0 - 0.5 * scaling),
                              _shock(default_rate_multiplier=1.5), base_rate)

    assert result["converged"] is True, result
    assert result["achieved_rate"] == pytest.approx(base_rate * 1.5, rel=0.01)


# --------------------------------------------------------------------------- #
# BUG 8 — the /scenario/run response cache keyed on a subset of the request, so
# a reanchor=false request was served the cached reanchor=true response.
# --------------------------------------------------------------------------- #
def test_scenario_cache_key_covers_every_result_changing_field():
    import inspect

    from lpie.api.routers import scenario as scenario_router

    src = inspect.getsource(scenario_router.run_scenario)
    key_src = src[src.index("cache_key"):src.index("cached =")]
    for field in ("scenario", "horizon", "n_paths", "seed", "segment_by",
                  "reanchor", "max_loans"):
        assert field in key_src, (
            f"`{field}` changes the scenario result but is absent from the cache key; "
            "requests differing only in it are served each other's responses"
        )


# --------------------------------------------------------------------------- #
# BUG 9 — the submission CSV round-trip. "None" is a legitimate exception_type
# (it is what the supplied template uses) and is also in pandas' default NA
# list, so reading the file back turned 8,350 valid values into nulls and
# failed a correct submission for "nulls in a non-nullable column".
# --------------------------------------------------------------------------- #
def test_submission_csv_round_trips_the_literal_string_none(tmp_path):
    from lpie.submit.validator import read_submission_csv

    path = tmp_path / "submission.csv"
    path.write_text(
        "loan_id,reporting_month,exception_type,predicted_next_state\n"
        "LN0000001,2024-01,None,Current\n"
        "LN0000002,2024-01,doc_gap,Prepaid\n"
    )
    frame = read_submission_csv(path)

    assert frame["exception_type"].isna().sum() == 0, (
        "'None' was coerced to NaN; a valid submission fails its own validator"
    )
    assert frame["exception_type"].tolist() == ["None", "doc_gap"]


def test_generate_and_validate_agree_on_the_same_submission():
    """The two endpoints disagreed: generate said valid, validate said invalid."""
    import os

    from lpie.submit.validator import read_submission_csv, validate_submission

    if not os.path.exists("artifacts/submission.csv"):
        pytest.skip("no submission artifact; run `make submit`")

    frame = read_submission_csv("artifacts/submission.csv")
    template = (
        read_submission_csv("dataset/submission_template.csv")
        if os.path.exists("dataset/submission_template.csv") else None
    )
    report = validate_submission(frame, template)
    assert report["valid"], f"on-disk submission fails validation: {report['errors']}"


# --------------------------------------------------------------------------- #
# BUG 10 — PAVA assigns an all-negative block a fitted value of exactly 0.0, so
# 92% of ACTIVE loans were served P(default) = 0 (a claim of impossibility) and
# an expected loss of exactly zero.
# --------------------------------------------------------------------------- #
def test_calibrated_probability_is_never_an_impossibility_claim():
    from lpie.models.calibration import SegmentIsotonicCalibrator

    raw, y, seg, terminal = _panel()
    cal = SegmentIsotonicCalibrator()
    cal.fit(raw, y, seg, eligible=~terminal)
    out = cal.transform(raw, seg)[~terminal]

    assert (out > 0.0).all(), (
        "an active loan was assigned P = 0 exactly; zero is not a low probability, "
        "it is a claim no finite calibration sample can support"
    )
    assert (out < 1.0).all()
    assert out.min() == pytest.approx(cal.floor, rel=1e-9)


def test_probability_floor_tracks_the_calibration_sample_size():
    """The floor is the Laplace bound, so it must shrink as evidence grows."""
    from lpie.models.calibration import SegmentIsotonicCalibrator

    small_raw, small_y, small_seg, small_term = _panel(n=4000, seed=1)
    big_raw, big_y, big_seg, big_term = _panel(n=40000, seed=1)

    small = SegmentIsotonicCalibrator()
    small.fit(small_raw, small_y, small_seg, eligible=~small_term)
    big = SegmentIsotonicCalibrator()
    big.fit(big_raw, big_y, big_seg, eligible=~big_term)

    assert big.floor < small.floor
    assert big.floor == pytest.approx(1.0 / (big.artifact.n_calibration_rows + 2.0))


def test_floor_is_small_enough_not_to_invent_risk():
    from lpie.models.calibration import SegmentIsotonicCalibrator

    raw, y, seg, terminal = _panel()
    cal = SegmentIsotonicCalibrator()
    cal.fit(raw, y, seg, eligible=~terminal)
    assert cal.floor < 1e-3, (
        f"floor={cal.floor} is large enough to distort the expected-loss book"
    )


def test_absorbing_states_are_still_gated_to_exact_zero():
    """The floor must not leak into rows the serving layer gates deterministically."""
    import inspect

    from lpie.serving import scorer

    src = inspect.getsource(scorer)
    assert 'out.loc[terminal, "expected_loss"] = 0.0' in src, (
        "the absorbing-state gate is what makes a 0.0 defensible; it must remain "
        "a deterministic override rather than a calibrated estimate"
    )


# --------------------------------------------------------------------------- #
# BUG 11 — the "name at least 2 fields" rule is written for reviewer notes but
# was applied to grounded Q&A, so a correct one-field definitional answer was
# rejected and the template fallback served instead.
# --------------------------------------------------------------------------- #
def test_specificity_rule_is_scoped_to_the_tasks_it_was_written_for():
    from lpie.copilot.service import SPECIFICITY_REQUIRED_TASKS

    assert "reviewer_note" in SPECIFICITY_REQUIRED_TASKS
    assert "scenario_summary" in SPECIFICITY_REQUIRED_TASKS
    assert "grounded_qa" not in SPECIFICITY_REQUIRED_TASKS, (
        "a correct answer to 'what does days_past_due mean' names one field; "
        "requiring two rejects correct answers"
    )


def test_relaxing_specificity_does_not_relax_any_substantive_check():
    """The governance that matters must still reject, with specificity off."""
    from lpie.copilot.evidence import EvidencePacket
    from lpie.copilot.verifier import NumericVerifier

    verifier = NumericVerifier()
    pkt = EvidencePacket(task="grounded_qa", subject={"question": "q"})
    pkt.add_number("prob_next_12m_default", 0.087, source="ml_model")

    cases = {
        "unsupported_number": "The default probability is 0.912.",
        "unknown_rule_id": "Rule VR-999 fired on this loan.",
        "forbidden_decision_language": "We will foreclose on this loan.",
    }
    for kind, text in cases.items():
        result = verifier.verify(
            text, pkt, require_banner=False, require_specificity=False
        )
        assert any(f.kind == kind for f in result.failures), (
            f"{kind!r} was not caught with specificity disabled: "
            f"{[f.kind for f in result.failures]} for {text!r}"
        )


def test_banner_is_still_required_where_it_is_requested():
    from lpie.copilot.evidence import EvidencePacket
    from lpie.copilot.verifier import NumericVerifier

    pkt = EvidencePacket(task="grounded_qa", subject={"question": "q"})
    result = NumericVerifier().verify(
        "days_past_due counts delinquent days.", pkt,
        require_banner=True, require_specificity=False,
    )
    assert any(f.kind == "missing_governance_banner" for f in result.failures)


# --------------------------------------------------------------------------- #
# BUG 12 — RAG hits were shaped with keys the evidence packet does not render
# (`chunk`/`source` vs the `citation`/`text` it reads), so every retrieved
# passage reached the model as the literal line "[None] ". The API advertised
# six citations while the model was answering with no context at all.
# --------------------------------------------------------------------------- #
def _rag_hit():
    return {
        "doc_id": "system_design", "chunk_id": "6-3-missing-value-strategy-0",
        "citation": "system_design#6-3-missing-value-strategy-0",
        "title": "SYSTEM_DESIGN", "source_path": "/srv/app/SYSTEM_DESIGN.md",
        "text": "days_past_due is 5% null and is imputed from current_status.",
        "score": 0.71,
    }


def test_retrieved_passages_actually_reach_the_prompt():
    from lpie.api.routers.copilot import _citations
    from lpie.copilot.evidence import build_query_packet

    packet = build_query_packet("what is days_past_due?", citations=_citations([_rag_hit()]))
    rendered = packet.render()

    assert "[None]" not in rendered, "retrieved passages render as [None]; the model sees no context"
    assert "days_past_due is 5% null" in rendered
    assert "system_design#6-3-missing-value-strategy-0" in rendered


def test_citations_do_not_leak_the_server_filesystem_layout():
    from lpie.api.routers.copilot import _citations
    from lpie.core.config import get_settings

    hit = dict(_rag_hit(), source_path=f"{get_settings().root}/SYSTEM_DESIGN.md")
    assert _citations([hit])[0]["source"] == "SYSTEM_DESIGN.md"


def test_citation_slugs_are_not_mined_for_numbers_or_field_names():
    """The QA prompt requires [doc_id#chunk_id] citations; obeying it must not fail."""
    from lpie.copilot.verifier import extract_field_references, extract_numbers

    text = "Impute it from current_status [system_design#6-3-missing-value-strategy-0]."
    assert [v for _, v in extract_numbers(text)] == []
    assert "system_design" not in extract_field_references(text)


def test_numbers_quoted_from_a_cited_passage_are_grounded():
    from lpie.api.routers.copilot import _citations
    from lpie.copilot.evidence import build_query_packet
    from lpie.copilot.verifier import NumericVerifier

    packet = build_query_packet("what is days_past_due?", citations=_citations([_rag_hit()]))
    result = NumericVerifier().verify(
        "The packet reports a 5% null rate for `days_past_due`.",
        packet, require_banner=False, require_specificity=False,
    )
    assert not [f for f in result.failures if f.kind == "unsupported_number"], (
        "a figure quoted out of the cited passage was called a hallucination"
    )


def test_a_number_in_neither_packet_nor_context_is_still_rejected():
    from lpie.api.routers.copilot import _citations
    from lpie.copilot.evidence import build_query_packet
    from lpie.copilot.verifier import NumericVerifier

    packet = build_query_packet("what is days_past_due?", citations=_citations([_rag_hit()]))
    result = NumericVerifier().verify(
        "The portfolio default rate is exactly 87.3 percent.",
        packet, require_banner=False, require_specificity=False,
    )
    assert any(f.kind == "unsupported_number" for f in result.failures)


def test_rule_id_from_documentation_is_valid_in_qa_but_not_claimed_as_fired():
    from lpie.copilot.evidence import EvidencePacket, build_query_packet
    from lpie.copilot.verifier import NumericVerifier

    verifier = NumericVerifier()

    # Q&A: no record, so VR-001 is a reference to the rule catalogue.
    qa = build_query_packet(
        "which rule covers balance exceeding original balance?",
        citations=[{"citation": "rules#vr", "text": "VR-001 flags current_balance > original_balance."}],
    )
    qa_result = verifier.verify("Rule VR-001 covers it.", qa,
                                require_banner=False, require_specificity=False)
    assert not [f for f in qa_result.failures if f.kind == "unknown_rule_id"]

    # A rule the documentation does not mention is still rejected.
    bogus = verifier.verify("Rule VR-999 covers it.", qa,
                            require_banner=False, require_specificity=False)
    assert any(f.kind == "unknown_rule_id" for f in bogus.failures)

    # Reviewer note: a record IS present, so "fired" remains authoritative.
    note = EvidencePacket(task="reviewer_note", subject={"loan_id": "LN0000001"})
    note.rule_ids = ["VR-006"]
    note.citations = [{"citation": "rules#vr", "text": "VR-001 flags current_balance > original_balance."}]
    claim = verifier.verify("Rule VR-001 fired on this loan.", note,
                            require_banner=False, require_specificity=False)
    assert any(f.kind == "unknown_rule_id" for f in claim.failures), (
        "a rule that did not fire on this record must not be claimable just "
        "because the documentation mentions it"
    )


# --------------------------------------------------------------------------- #
# BUG 17 — generated-SQL results were computed and previewed to the caller but
# never reached the model: `add_number()` requires `source` as a mandatory
# keyword-only argument, the router called it without one, and the resulting
# TypeError was swallowed by a bare `except`. Every use_sql=True question that
# depended on the query's answer was told "the evidence packet does not
# contain that value" regardless of what the query actually returned.
# --------------------------------------------------------------------------- #
def test_add_number_requires_source_and_is_not_silently_swallowable():
    from lpie.copilot.evidence import EvidencePacket

    packet = EvidencePacket(task="grounded_qa", subject={})
    with pytest.raises(TypeError):
        packet.add_number("portfolio_default_rate", 0.0025)  # no source= kwarg


def test_scalar_query_result_reaches_the_packet_as_a_number():
    import pandas as pd

    from lpie.api.routers.copilot import _load_query_results_into_packet
    from lpie.copilot.evidence import build_query_packet

    packet = build_query_packet("what is the default rate?", citations=[])
    df = pd.DataFrame([{"total_loans": 10000, "defaulted_loans": 25.0,
                        "portfolio_default_rate": 0.0025}])
    _load_query_results_into_packet(packet, df)

    assert packet.numbers["portfolio_default_rate"]["value"] == pytest.approx(0.0025)
    assert packet.numbers["portfolio_default_rate"]["source"] == "generated_sql"


# --------------------------------------------------------------------------- #
# BUG 18 — a ranking query result was flattened by bare column name, so
# `packet.add_number("state", ...)` was called once per row and each call
# overwrote the last. A multi-row ranking answer had no way to say which label
# a number belonged to, because the label had already been discarded.
# --------------------------------------------------------------------------- #
def test_ranking_query_result_preserves_every_row_not_just_the_last():
    import pandas as pd

    from lpie.api.routers.copilot import _load_query_results_into_packet
    from lpie.copilot.evidence import build_query_packet

    packet = build_query_packet("which states have the highest delinquency?", citations=[])
    df = pd.DataFrame([
        {"state": "VT", "n_loans": 350, "delinquency_rate_pct": 5.43},
        {"state": "ND", "n_loans": 363, "delinquency_rate_pct": 5.23},
        {"state": "MT", "n_loans": 664, "delinquency_rate_pct": 4.52},
    ])
    _load_query_results_into_packet(packet, df)

    values = {round(v, 2) for v in packet.allowed_values()}
    assert {5.43, 5.23, 4.52} <= values, (
        "only the last row's rate survived; a ranking answer needs all of them"
    )
    assert "VT" in packet.facts.get("query_result_rows", "")
    assert "ND" in packet.facts.get("query_result_rows", "")


def test_ranking_answer_from_a_real_query_names_the_correct_top_row():
    """End-to-end: the model must be ABLE to name which row a number belongs to."""
    from lpie.api.routers.copilot import _load_query_results_into_packet
    from lpie.copilot.evidence import build_query_packet
    from lpie.copilot.verifier import NumericVerifier

    import pandas as pd

    packet = build_query_packet("which state has the highest delinquency rate?", citations=[])
    df = pd.DataFrame([
        {"state": "VT", "delinquency_rate_pct": 5.43},
        {"state": "ND", "delinquency_rate_pct": 5.23},
    ])
    _load_query_results_into_packet(packet, df)

    result = NumericVerifier().verify(
        "VT has the highest delinquency_rate_pct at 5.43.", packet,
        require_banner=False, require_specificity=False,
    )
    assert not [f for f in result.failures if f.kind == "unsupported_number"]


# --- drift: the retrain trigger must not fire on the calendar ----------------
# Before this fix `/drift` returned FAIL on every window pair. Two separate
# causes, both structural rather than statistical:
#   1. the forward-looking labels are 100% null in the scoring window by
#      construction, so their null indicator separated the windows perfectly
#      and pinned the adversarial AUC at 1.0;
#   2. loan_age_months / remaining_term_months shift by exactly N months
#      whenever the windows are N months apart, on a perfectly healthy book.

def _seasoning_panel(n=1500, seed=0):
    """A stable population observed over two windows. Nothing drifts except the
    calendar: same loans, same distributions, six months later."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)

    def window(month_lo, age0, labelled):
        return pd.DataFrame({
            "loan_id": np.arange(n),
            "month_index": rng.integers(month_lo, month_lo + 6, n),
            "loan_age_months": age0 + rng.integers(0, 6, n),
            "remaining_term_months": (360 - age0) - rng.integers(0, 6, n),
            "interest_rate": rng.normal(5.0, 0.5, n),
            "credit_score_band": rng.choice(["A", "B", "C"], n, p=[0.3, 0.5, 0.2]),
            "next_state": (
                rng.choice(["Current", "30DPD"], n) if labelled
                else np.full(n, np.nan, dtype="object")
            ),
        })

    return window(31, 31, True), window(37, 37, False)


def test_drift_trigger_ignores_seasoning_and_labels():
    from lpie.profiling.drift import drift_report

    reference, current = _seasoning_panel()
    report = drift_report(
        reference, current, ref_window="31-36", cur_window="37-42",
    )

    # The labels are unlabelled-by-construction in the current window; if they
    # reach the adversarial classifier it scores a perfect 1.0 and every batch
    # is condemned.
    assert "next_state" in report["adversarial"]["excluded_columns"]
    assert report["adversarial"]["adversarial_auc"] < 0.80, (
        "adversarial AUC must not be driven by label availability"
    )

    # Seasoning features are still measured...
    seasoning_rows = {f["feature"]: f for f in report["features"] if f["seasoning"]}
    assert "loan_age_months" in seasoning_rows

    # ...but cannot fire the trigger, and cannot force a permanent WARN.
    trigger = report["retraining_trigger"]
    assert "loan_age_months" not in trigger["features_over_psi_threshold"]
    assert trigger["retrain_required"] is False, trigger["reasons"]
    assert report["batch_verdict"] == "PASS"
    assert report["max_psi_actionable"] <= report["max_psi"]


def test_drift_trigger_still_fires_on_real_shift():
    """The guard must not make the detector deaf: a genuine population shift in
    non-seasoning features still has to trip the trigger."""
    import numpy as np

    from lpie.profiling.drift import drift_report

    reference, current = _seasoning_panel()
    rng = np.random.default_rng(1)
    n = len(current)
    # Rate shock + a credit-quality migration the model has never seen.
    current["interest_rate"] = rng.normal(11.0, 0.5, n)
    current["credit_score_band"] = rng.choice(["C", "D"], n)

    report = drift_report(
        reference, current, ref_window="31-36", cur_window="37-42",
    )
    assert report["retraining_trigger"]["retrain_required"] is True
    assert report["batch_verdict"] == "FAIL"
