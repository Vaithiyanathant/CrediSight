"""Validation rule engine tests."""
from __future__ import annotations
import sys
sys.path.insert(0, 'src')
import pandas as pd
import pytest


def _make_minimal_frame():
    return pd.DataFrame([{
        'loan_id': 'LN0000001', 'month_index': 1,
        'current_balance': 100000.0, 'original_balance': 120000.0,
        'current_status': 'Current', 'days_past_due': 0.0,
        'modification_flag': 0, 'prepayment_flag': 0, 'default_flag': 0,
        'document_status': 'Complete', 'loan_age_months': 12,
        'remaining_term_months': 348, 'interest_rate': 0.05,
        'credit_score_band': '700-739', 'state': 'CA',
        'reporting_month': '2021-01', 'source_system': 'SYSTEM_A',
        'servicer_name': 'Servicer A', 'ltv_band': '60-70',
        'dti_band': '20-30', 'loan_purpose': 'Purchase',
        'occupancy_type': 'Primary', 'property_type': 'Single-Family',
        'origination_month': '2019-01',
        'last_updated_at': '2021-01-31', 'loss_severity_band': None,
    }])


def test_vr001_in_rule_set():
    from lpie.validation.rules import load_rules
    rules = load_rules()
    vr001 = next((r for r in rules if r.rule_id == 'VR-001'), None)
    assert vr001 is not None, 'VR-001 must be in the rule set'


def test_rule_count_is_18():
    from lpie.validation.rules import load_rules
    rules = load_rules()
    assert len(rules) == 18, f'Expected 18 rules, got {len(rules)}'


def test_engine_runs_on_minimal_frame():
    from lpie.core.config import get_settings
    from lpie.validation.engine import ValidationEngine
    s = get_settings()
    engine = ValidationEngine(s)
    frame = _make_minimal_frame()
    result = engine.run(
        frame,
        static=pd.DataFrame(),
        servicer=pd.DataFrame(),
        batch_id='test_batch',
    )
    assert result is not None
    assert hasattr(result, 'batch_score')
    assert result.batch_score is not None
    assert isinstance(result.batch_score, dict)
    assert 0 <= result.batch_score["mean_dq"] <= 100


def test_rule_ids_are_unique():
    from lpie.validation.rules import load_rules
    rules = load_rules()
    ids = [r.rule_id for r in rules]
    assert len(ids) == len(set(ids))


def test_all_rules_have_severity():
    from lpie.validation.rules import load_rules
    VALID = {'ERROR', 'WARNING', 'INFO'}
    for r in load_rules():
        assert r.severity in VALID, f'Rule {r.rule_id} has invalid severity: {r.severity}'
