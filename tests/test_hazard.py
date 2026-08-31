"""Hazard model invariant tests."""
from __future__ import annotations
import sys
sys.path.insert(0, 'src')
import numpy as np
import pytest


def test_legal_mask_construction():
    from lpie.models.hazard import build_legal_mask
    states = ['Current', '30DPD', '60DPD', '90DPD', 'Default', 'Prepaid', 'Closed']
    legal = {
        'Current': ['Current', '30DPD', 'Prepaid'],
        '30DPD': ['Current', '30DPD', '60DPD', 'Prepaid'],
        '60DPD': ['Current', '30DPD', '60DPD', '90DPD', 'Prepaid'],
        '90DPD': ['Current', '60DPD', '90DPD', 'Default', 'Prepaid'],
        'Default': ['Default', 'Closed'],
        'Prepaid': ['Prepaid'],
        'Closed': ['Closed'],
    }
    mask = build_legal_mask(states, legal)
    assert mask.shape == (7, 7)
    # Prepaid -> Prepaid only
    prepaid_idx = states.index('Prepaid')
    assert mask[prepaid_idx, prepaid_idx] == True
    assert mask[prepaid_idx].sum() == 1
    # Closed -> Closed only
    closed_idx = states.index('Closed')
    assert mask[closed_idx, closed_idx] == True
    assert mask[closed_idx].sum() == 1
    # Current cannot go to Default directly
    current_idx = states.index('Current')
    default_idx = states.index('Default')
    assert mask[current_idx, default_idx] == False


def test_apply_legal_mask_renormalises():
    from lpie.models.hazard import apply_legal_mask, build_legal_mask
    states = ['Current', '30DPD', '60DPD', '90DPD', 'Default', 'Prepaid', 'Closed']
    legal = {'Current': ['Current', '30DPD', 'Prepaid'],
             '30DPD': ['Current', '30DPD', '60DPD', 'Prepaid'],
             '60DPD': ['Current', '30DPD', '60DPD', '90DPD', 'Prepaid'],
             '90DPD': ['Current', '60DPD', '90DPD', 'Default', 'Prepaid'],
             'Default': ['Default', 'Closed'], 'Prepaid': ['Prepaid'], 'Closed': ['Closed']}
    mask = build_legal_mask(states, legal)
    probs = np.ones((3, 7)) / 7.0
    from_idx = np.array([0, 1, 5])  # Current, 30DPD, Prepaid
    masked = apply_legal_mask(probs, from_idx, mask)
    # Rows must sum to 1
    row_sums = masked.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
    # Prepaid row: only Prepaid allowed
    prepaid_idx = states.index('Prepaid')
    assert masked[2, prepaid_idx] == pytest.approx(1.0)
    for j in range(7):
        if j != prepaid_idx:
            assert masked[2, j] == pytest.approx(0.0)
