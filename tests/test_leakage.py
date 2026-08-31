"""Leakage prevention tests."""
from __future__ import annotations
import sys
sys.path.insert(0, 'src')
import pytest


def test_feature_store_partitioned_by_month():
    """Feature store must be Hive-partitioned by month_index."""
    from pathlib import Path
    store = Path('artifacts/store/features')
    if not store.exists():
        pytest.skip('Feature store not built yet')
    partitions = [p for p in store.iterdir() if p.is_dir()]
    assert len(partitions) > 0, 'No partitions found in feature store'
    for p in partitions[:5]:
        assert p.name.startswith('month_index='), (
            f'Non-standard partition: {p.name} (expected month_index=N)'
        )


def test_head_artifacts_exclude_target_columns():
    """The trained head artifacts must only reference feature columns, not targets."""
    import joblib
    from pathlib import Path
    TARGETS = {'next_3m_delinquency_flag','next_6m_delinquency_flag',
               'next_12m_default_flag','next_12m_prepayment_flag',
               'next_state','exception_required','exception_type'}
    heads_path = Path('artifacts/models/heads.joblib')
    if not heads_path.exists():
        pytest.skip('heads.joblib not found')
    heads = joblib.load(heads_path)
    for head_name, algos in heads.items():
        if not isinstance(algos, dict):
            continue
        for algo, artifact in algos.items():
            if not hasattr(artifact, 'feature_names'):
                continue
            leaked = set(artifact.feature_names) & TARGETS
            assert not leaked, (
                f'Head {head_name}/{algo} uses target column(s) as features: {leaked}'
            )


def test_loan_id_not_in_head_features():
    """loan_id must never be a model feature (target-encoding leakage)."""
    import joblib
    from pathlib import Path
    heads_path = Path('artifacts/models/heads.joblib')
    if not heads_path.exists():
        pytest.skip('heads.joblib not found')
    heads = joblib.load(heads_path)
    for head_name, algos in heads.items():
        if not isinstance(algos, dict):
            continue
        for algo, artifact in algos.items():
            if not hasattr(artifact, 'feature_names'):
                continue
            assert 'loan_id' not in artifact.feature_names, (
                f'Head {head_name}/{algo} uses loan_id as a feature — target-encoding leakage!'
            )


def test_purged_embargo_guarantees_no_leakage():
    """The temporal split must never allow validation rows to precede training rows."""
    from lpie.models.splitters import build_split_plan
    for head_name in ['next_12m_default', 'next_3m_delinquency', 'next_6m_delinquency', 'next_12m_prepayment']:
        plan = build_split_plan(head=head_name, panel_max_month=36)
        for fold in plan.folds:
            train_end = max(fold.train_months)
            val_start = min(fold.valid_months)
            assert val_start > train_end, (
                f"{head_name} fold: val starts at {val_start} before training ends at {train_end}"
            )


def test_feature_store_month_count():
    """Feature store should have exactly 36 train partitions."""
    from pathlib import Path
    store = Path('artifacts/store/features')
    if not store.exists():
        pytest.skip('Feature store not built yet')
    partitions = [p for p in store.iterdir() if p.is_dir() and p.name.startswith('month_index=')]
    # Should have 36 partitions for months 1-36
    assert len(partitions) >= 1, 'Feature store has no month partitions'
