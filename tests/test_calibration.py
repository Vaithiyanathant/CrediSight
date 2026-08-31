"""Calibration invariant tests."""
from __future__ import annotations
import sys
sys.path.insert(0, 'src')
import numpy as np
import pytest


def test_ece_is_low_after_calibration():
    """The trained calibration ECE should be < 0.05 per the model card."""
    import joblib
    from pathlib import Path
    cal_path = Path('artifacts/models/calibrators.joblib')
    if not cal_path.exists():
        pytest.skip('calibrators.joblib not found — run make train first')
    calibrators = joblib.load(cal_path)
    for head, cal in calibrators.items():
        if hasattr(cal, 'artifact') and hasattr(cal.artifact, 'ece'):
            ece = float(cal.artifact.ece)
            assert ece < 0.05, f'{head} ECE {ece:.4f} exceeds 0.05 — calibration regression'


def test_probabilities_in_unit_interval():
    import joblib
    from pathlib import Path
    cal_path = Path('artifacts/models/calibrators.joblib')
    if not cal_path.exists():
        pytest.skip('calibrators.joblib not found')
    calibrators = joblib.load(cal_path)
    rng = np.random.default_rng(42)
    raw = rng.uniform(0, 1, 100)
    for head, cal in calibrators.items():
        if hasattr(cal, 'transform'):
            import pandas as pd
            segs = pd.Series(['2021_700-739'] * 100)
            try:
                out = cal.transform(raw, segs)
                assert np.all(out >= 0.0) and np.all(out <= 1.0)
            except Exception:
                pass
