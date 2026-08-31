"""API endpoint smoke tests."""
from __future__ import annotations
import sys
sys.path.insert(0, 'src')
import pytest


@pytest.fixture(scope='module')
def client():
    from fastapi.testclient import TestClient
    from lpie.api.main import create_app
    from lpie.core.config import get_settings
    app = create_app(get_settings())
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_livez(client):
    resp = client.get('/livez')
    assert resp.status_code == 200
    assert resp.json()['status'] == 'alive'


def test_healthz_returns_valid_schema(client):
    resp = client.get('/healthz')
    assert resp.status_code in (200, 503)
    data = resp.json()
    assert 'status' in data
    assert 'ready' in data
    assert 'model_version' in data


def test_readyz(client):
    resp = client.get('/readyz')
    assert resp.status_code in (200, 503)
    data = resp.json()
    assert 'ready' in data


def test_metrics_endpoint(client):
    resp = client.get('/metrics')
    assert resp.status_code == 200
    assert 'lpie_ready' in resp.text


def test_meta_models(client):
    resp = client.get('/api/v1/meta/models')
    assert resp.status_code == 200
    data = resp.json()
    assert 'model_version' in data
    assert 'models' in data


def test_dq_summary(client):
    resp = client.get('/api/v1/dq/summary')
    assert resp.status_code in (200, 404, 503)


def test_scenarios_list(client):
    resp = client.get('/api/v1/scenarios')
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        assert 'scenarios' in resp.json()


def test_portfolio_summary(client):
    resp = client.get('/api/v1/portfolio/summary')
    # May be 503 if artifacts not loaded, but must not be 500
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert 'total_loans' in data


def test_predict_requires_loan_ids_or_records(client):
    # Missing both loan_ids and records -> 422
    resp = client.post('/api/v1/predict', json={})
    assert resp.status_code == 422


def test_submission_validate_missing_file(client):
    resp = client.get('/api/v1/submission/validate')
    # Either 200 (file exists) or 404 (doesn't exist yet)
    assert resp.status_code in (200, 404)


def test_openapi_schema_accessible(client):
    resp = client.get('/openapi.json')
    assert resp.status_code == 200
    data = resp.json()
    assert 'paths' in data
    assert len(data['paths']) > 15  # All 15 routers registered


# ---- Additional endpoint tests added after bug fixes ----

def test_drift_endpoint_no_nan(client):
    """Drift endpoint must not return NaN floats (JSON non-compliant)."""
    resp = client.get('/api/v1/drift')
    # 200 or 404 (if test months not in panel) - never a crash
    assert resp.status_code in (200, 404, 503)
    if resp.status_code == 200:
        import json
        # Must parse cleanly (no NaN)
        data = resp.json()
        assert 'features' in data or 'elapsed_ms' in data


def test_survival_state_occupancy(client):
    """State-occupancy endpoint returns correct structure."""
    resp = client.get('/api/v1/survival/state-occupancy?horizon=6')
    assert resp.status_code in (200, 404, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert 'states' in data
        assert 'mean_share' in data
        # Each month row must have K=7 entries
        assert all(len(row) == len(data['states']) for row in data['mean_share'])


def test_explain_errors_not_captured_by_loan_id_route(client):
    """GET /explain/errors must NOT be captured by /{loan_id} route."""
    resp = client.get('/api/v1/explain/errors?head=next_12m_default')
    # If it returned a loan explanation for loan_id='errors' that is the route ordering bug
    # It should return 200 (evaluation data present) or 404 (not yet computed)
    assert resp.status_code in (200, 404, 503)
    if resp.status_code == 200:
        data = resp.json()
        # Must have ErrorAnalysisResponse fields, not LocalExplainResponse fields
        assert 'head' in data
        assert 'confusion_profile' in data or 'error_slices' in data


def test_explain_local_loan(client):
    """Local explanation for a specific loan."""
    resp = client.get('/api/v1/explain/LN0000001?head=next_12m_default')
    assert resp.status_code in (200, 404, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert data.get('loan_id') == 'LN0000001'
        assert 'probability' in data
        assert 'top_contributions' in data


def test_survival_loan(client):
    """Survival endpoint for a single loan."""
    resp = client.get('/api/v1/survival/LN0000001')
    assert resp.status_code in (200, 404, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert data['loan_id'] == 'LN0000001'
        assert 'survival' in data
        assert 'cif_default' in data
        # CIF + S conservation
        s = data['survival']
        d = data['cif_default']
        pp = data['cif_prepay']
        cl = data['cif_closed']
        for i in range(min(len(s),len(d),len(pp),len(cl))):
            total = s[i] + d[i] + pp[i] + cl[i]
            assert abs(total - 1.0) < 0.01, f'horizon {i}: CIF+S = {total} != 1.0'


def test_anomalies_endpoint(client):
    """Anomaly list returns correct structure."""
    resp = client.get('/api/v1/anomalies?limit=5')
    assert resp.status_code in (200, 503)  # 503 if anomaly artifact not loaded
    if resp.status_code == 200:
        data = resp.json()
        assert 'entries' in data
        assert 'n' in data
        for entry in data['entries']:
            assert 0.0 <= entry['anomaly_score'] <= 1.0


def test_reviewer_decision_records(client):
    """Reviewer decision is persisted."""
    resp = client.post('/api/v1/reviewer/decision', json={
        'loan_id': 'LN0000042',
        'month_index': 37,
        'human_decision': 'Escalate',
        'model_recommendation': 'Flag',
        'rationale': 'High anomaly score with VR-012 violation'
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data['loan_id'] == 'LN0000042'
    assert data['human_decision'] == 'Escalate'
    assert 'agreed_with_model' in data
    assert 'agreement_stats' in data


def test_scenario_run_returns_summary(client):
    """Scenario run returns summary with default rate."""
    resp = client.post('/api/v1/scenario/run', json={
        'scenario': 'Base',
        'n_paths': 20,
        'horizon': 3
    })
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert 'summary' in data
        assert 'scenario' in data
        assert data['scenario'] == 'Base'


def test_portfolio_summary_stats(client):
    """Portfolio summary has sensible aggregate stats."""
    resp = client.get('/api/v1/portfolio/summary')
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert data['total_loans'] > 0
        assert data['total_balance'] > 0
        assert 0.0 <= data['delinquency_rate'] <= 1.0
        assert 0.0 <= data['projected_default_rate'] <= 1.0


def test_predict_batch_response_schema(client):
    """Batch predict returns correct PredictionResponse schema."""
    resp = client.post('/api/v1/predict', json={'loan_ids': ['LN0000001', 'LN0000002']})
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert 'predictions' in data
        assert 'n_rows' in data
        assert data['n_rows'] >= 1
        for bundle in data['predictions']:
            assert 'loan_id' in bundle
            assert 'is_terminal' in bundle
            assert 'predictions' in bundle
            # Probabilities in [0,1]
            preds = bundle['predictions']
            for key in ['prob_next_3m_delinquency','prob_next_6m_delinquency',
                        'prob_next_12m_default','prob_next_12m_prepayment']:
                val = preds[key]['value']
                assert 0.0 <= val <= 1.0, f'{key} = {val} out of [0,1]'
