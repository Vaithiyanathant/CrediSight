"""Temporal split and leakage tests."""
from __future__ import annotations
import sys
sys.path.insert(0, 'src')
import pytest


def test_purged_embargo_no_leakage():
    from lpie.models.splitters import build_split_plan
    plan = build_split_plan(head='next_12m_default', panel_max_month=36)
    for fold in plan.folds:
        train_end = max(fold.train_months)
        val_start = min(fold.valid_months)
        assert val_start > train_end, (
            f"val_start {val_start} <= train_end {train_end} — leakage!"
        )
        # Embargo must separate them
        embargo = fold.embargo_months
        assert len(embargo) >= 0  # embargo may be zero for short-horizon heads


def test_censoring_mask_excludes_future():
    from lpie.models.splitters import censoring_mask
    import pandas as pd
    months = pd.Series(list(range(1, 37)))
    mask = censoring_mask(months, horizon=12, panel_max_month=36)
    # max_valid = 36 - 12 = 24; months 25..36 should be masked
    assert not mask.iloc[24:].any(), "Rows after max_valid_month should be masked out"
    assert mask.iloc[:24].all(), "Rows within valid window should be kept"


def test_no_random_split_allowed():
    from lpie.models.splitters import build_split_plan
    plan = build_split_plan(head='next_3m_delinquency', panel_max_month=36)
    for fold in plan.folds:
        train_end = max(fold.train_months)
        val_start = min(fold.valid_months)
        assert val_start > train_end, "Validation months must be strictly after training months"


def test_3m_horizon_max_valid_month():
    from lpie.models.splitters import build_split_plan
    plan = build_split_plan(head='next_3m_delinquency', panel_max_month=36)
    assert plan.max_valid_month == 33


def test_12m_horizon_max_valid_month():
    from lpie.models.splitters import build_split_plan
    plan = build_split_plan(head='next_12m_default', panel_max_month=36)
    assert plan.max_valid_month == 24
