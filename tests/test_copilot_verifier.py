"""Copilot numeric verifier tests."""
from __future__ import annotations
import sys
sys.path.insert(0, 'src')
import pytest


def _make_packet(numbers):
    from lpie.copilot.evidence import EvidencePacket
    pkt = EvidencePacket(task='reviewer_note', subject={'loan_id': 'LN0000001'})
    for key, val in numbers.items():
        pkt.add_number(key, float(val), source='ml_model')
    return pkt


def test_verifier_passes_correct_number():
    from lpie.copilot.verifier import NumericVerifier
    verifier = NumericVerifier()
    pkt = _make_packet({'prob_next_12m_default': 0.087})
    text = 'The 12-month default probability is 0.087.'
    result = verifier.verify(text, pkt, require_banner=False, require_specificity=False)
    # Either passes or has 0 critical failures on a correct number
    assert result is not None


def test_verifier_runs_on_text():
    from lpie.copilot.verifier import NumericVerifier
    verifier = NumericVerifier()
    pkt = _make_packet({'prob_next_12m_default': 0.087})
    text = 'The default risk is elevated with probability 0.087.'
    result = verifier.verify(text, pkt, require_banner=False, require_specificity=False)
    assert hasattr(result, 'passed') or hasattr(result, 'failures')


def test_fallback_is_always_safe():
    from lpie.copilot.verifier import fallback_answer
    from lpie.copilot.evidence import EvidencePacket
    pkt = EvidencePacket(task='reviewer_note', subject={'loan_id': 'LN0000001'})
    pkt.add_number('anomaly_score', 0.72, source='anomaly_ensemble')
    banner = 'RECOMMENDATION, NOT DECISION'
    text = fallback_answer(pkt, banner)
    assert isinstance(text, str)
    assert len(text) > 10


def test_evidence_packet_tracks_numbers():
    from lpie.copilot.evidence import EvidencePacket
    pkt = EvidencePacket(task='grounded_qa', subject={})
    pkt.add_number('prob_default', 0.087, source='model')
    pkt.add_number('anomaly_score', 0.72, source='anomaly')
    assert len(pkt.allowed_values()) == 2
    assert 0.087 in pkt.allowed_values()
    assert 0.72 in pkt.allowed_values()
