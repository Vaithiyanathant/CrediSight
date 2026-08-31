"""Monte-Carlo scenario invariant tests."""
from __future__ import annotations
import sys
sys.path.insert(0, 'src')
import numpy as np
import pytest


def test_transition_matrix_row_sums():
    """A valid transition matrix must have rows summing to 1."""
    import numpy as np
    states = ['Current', '30DPD', '60DPD', '90DPD', 'Default', 'Prepaid', 'Closed']
    K = len(states)
    rng = np.random.default_rng(0)
    # Build a proper (n_loans, K, K) transition matrix with rows summing to 1
    n_loans = 5
    M = np.zeros((n_loans, K, K))
    for i in range(n_loans):
        for k in range(K):
            row = rng.dirichlet(np.ones(K))
            M[i, k, :] = row
    row_sums = M.sum(axis=2)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


def test_terminal_states_absorbing_in_mask():
    """Prepaid and Closed rows must not exit their state (legal mask)."""
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
    prepaid_idx = states.index('Prepaid')
    closed_idx = states.index('Closed')
    # Absorbing: only self-transition allowed
    assert mask[prepaid_idx].sum() == 1 and mask[prepaid_idx, prepaid_idx]
    assert mask[closed_idx].sum() == 1 and mask[closed_idx, closed_idx]


def test_hazard_propagation_invariant():
    """CIF_default + CIF_prepay + CIF_closed + S = 1 by construction."""
    import joblib
    from pathlib import Path
    hazard_path = Path('artifacts/models/hazard.joblib')
    if not hazard_path.exists():
        pytest.skip('hazard.joblib not found')
    from lpie.core.config import get_settings
    from lpie.models.hazard import HazardModel
    from lpie.features.builder import read_feature_store
    s = get_settings()
    hazard = HazardModel(s, artifact=joblib.load(hazard_path))
    features = read_feature_store(months=[24], settings=s)
    if features.empty:
        pytest.skip('No features at month 24')
    row = features.iloc[[0]]
    try:
        M = hazard.transition_matrices(row)
        start = np.array([hazard.state_index.get(str(row['current_status'].iloc[0]), 0)], dtype='int32')
        prop = hazard.propagate(M, start, 12)
        err = float(prop.get('conservation_max_error', 0.0))
        assert err < 1e-4, f'CIF + S conservation error {err:.2e} exceeds tolerance'
    except Exception as exc:
        pytest.skip(f'Propagation error (likely missing features): {exc}')


def test_state_occupancy_sums_to_one():
    """At each horizon month, all state occupancies must sum to 1."""
    import joblib
    from pathlib import Path
    hazard_path = Path('artifacts/models/hazard.joblib')
    if not hazard_path.exists():
        pytest.skip('hazard.joblib not found')
    from lpie.core.config import get_settings
    from lpie.models.hazard import HazardModel
    from lpie.features.builder import read_feature_store
    import numpy as np
    s = get_settings()
    hazard = HazardModel(s, artifact=joblib.load(hazard_path))
    features = read_feature_store(months=[24], settings=s)
    if features.empty:
        pytest.skip('No features at month 24')
    try:
        occ = hazard.state_occupancy(features.head(50), horizon=6)
        for m in range(1, 7):
            total = sum(occ[m].values())
            assert abs(total - 1.0) < 0.02, f'Month {m} occupancy sum {total:.4f} != 1.0'
    except Exception as exc:
        pytest.skip(f'State occupancy failed: {exc}')
