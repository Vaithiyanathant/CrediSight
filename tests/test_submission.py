"""Submission builder and validator tests."""
from __future__ import annotations
import sys
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
import pytest


def _make_pred_frame(n=5):
    return pd.DataFrame({
        'loan_id': [f'LN{i:07d}' for i in range(n)],
        'reporting_month': ['2024-01'] * n,
        'prob_next_3m_delinquency': np.random.uniform(0, 1, n),
        'prob_next_6m_delinquency': np.random.uniform(0, 1, n),
        'prob_next_12m_default': np.random.uniform(0, 1, n),
        'prob_next_12m_prepayment': np.random.uniform(0, 1, n),
        'predicted_next_state': ['Current'] * n,
        'anomaly_score': np.random.uniform(0, 1, n),
        'exception_required': [0] * n,
        'exception_type': ['none'] * n,
        'reviewer_action': ['No Action'] * n,
        'model_confidence': np.random.uniform(0.5, 1.0, n),
        'top_driver_1': ['days_past_due'] * n,
        'top_driver_2': ['loan_age_months'] * n,
        'top_driver_3': ['current_balance'] * n,
    })


def test_build_submission_columns():
    from lpie.submit.builder import build_submission, SUBMISSION_COLUMNS
    preds = _make_pred_frame()
    template = pd.DataFrame({'loan_id': preds['loan_id'], 'reporting_month': '2024-01'})
    result = build_submission(preds, template)
    missing = [c for c in SUBMISSION_COLUMNS if c not in result.columns]
    assert not missing, f'Missing submission columns: {missing}'


def test_probabilities_clipped_to_unit_interval():
    from lpie.submit.builder import build_submission
    preds = _make_pred_frame()
    preds['prob_next_12m_default'] = 1.5  # Out of range
    result = build_submission(preds, pd.DataFrame())
    assert result['prob_next_12m_default'].max() <= 1.0
    assert result['prob_next_12m_default'].min() >= 0.0


def test_validate_submission_flags_bad_ids():
    from lpie.submit.validator import validate_submission
    from lpie.submit.builder import build_submission
    preds = _make_pred_frame()
    # Corrupt one loan_id
    preds.loc[0, 'loan_id'] = 'INVALID'
    result_frame = build_submission(preds, pd.DataFrame())
    report = validate_submission(result_frame)
    assert not report['valid']
