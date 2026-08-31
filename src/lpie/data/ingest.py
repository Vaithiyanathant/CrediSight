"""Ingest: CSV -> typed pandas -> contract check -> DuckDB raw zone.

The raw zone is immutable and as-ingested. Nothing here repairs data; defects
are the *product* of Task 1 and must survive to the validation engine intact.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lpie.core.config import Settings, get_settings
from lpie.core.determinism import combined_data_sha256, dataset_hashes, sha256_file
from lpie.core.exceptions import DataNotFoundError
from lpie.core.logging import get_logger
from lpie.data.contracts import (
    MACRO_CONTRACT,
    MONTHLY_TEST_CONTRACT,
    MONTHLY_TRAIN_CONTRACT,
    SERVICER_CONTRACT,
    STATIC_CONTRACT,
    TableContract,
    check_contract,
)

log = get_logger(__name__)

_NUMERIC_MONTHLY = [
    "month_index", "loan_age_months", "remaining_term_months", "original_balance",
    "current_balance", "interest_rate", "days_past_due", "modification_flag",
    "prepayment_flag", "default_flag",
]
_TARGET_NUMERIC = [
    "next_3m_delinquency_flag", "next_6m_delinquency_flag",
    "next_12m_default_flag", "next_12m_prepayment_flag", "exception_required",
]


@dataclass
class IngestResult:
    frames: dict[str, pd.DataFrame]
    contract_reports: dict[str, dict[str, Any]]
    file_hashes: dict[str, str]
    data_sha256: str


def _read_csv(path: Path, *, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise DataNotFoundError(
            f"Required input file is missing: {path.name}",
            details={"path": str(path)},
        )
    df = pd.read_csv(path, low_memory=False)
    for col in parse_dates or []:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _normalise_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace on object columns; empty string becomes NaN.

    This is presentation hygiene, not repair: it does not change any value's
    meaning and prevents ' Current' from becoming an unseen category.
    """
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].astype("string").str.strip()
            df[c] = s.replace({"": pd.NA}).astype(object)
    return df


def load_static(settings: Settings | None = None) -> pd.DataFrame:
    s = settings or get_settings()
    df = _read_csv(s.dataset_file("static"))
    df = _normalise_strings(df)
    df = _to_numeric(df, ["original_balance", "interest_rate", "loan_term_months"])
    df["vintage_year"] = df["vintage_year"].astype("string").astype(object)
    return df


def load_monthly(split: str, settings: Settings | None = None) -> pd.DataFrame:
    """Load the train or test panel. `_split` marks provenance for the raw zone."""
    s = settings or get_settings()
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    df = _read_csv(s.dataset_file(split), parse_dates=["last_updated_at"])
    df = _normalise_strings(df)
    df = _to_numeric(df, _NUMERIC_MONTHLY + _TARGET_NUMERIC)
    df["month_index"] = df["month_index"].astype("int32")
    df["_split"] = split
    return df.sort_values(["loan_id", "month_index"], kind="mergesort").reset_index(drop=True)


def load_servicer(settings: Settings | None = None) -> pd.DataFrame:
    s = settings or get_settings()
    df = _read_csv(s.dataset_file("servicer"), parse_dates=["update_date"])
    df = _normalise_strings(df)
    df = _to_numeric(df, ["reported_balance", "reported_rate", "stale_flag"])
    return df.sort_values(["loan_id", "update_date"], kind="mergesort").reset_index(drop=True)


def load_macro(settings: Settings | None = None) -> pd.DataFrame:
    s = settings or get_settings()
    df = _read_csv(s.dataset_file("macro"))
    df = _normalise_strings(df)
    numeric = [c for c in df.columns if c not in {"scenario_name", "description"}]
    return _to_numeric(df, numeric)


def load_submission_template(settings: Settings | None = None) -> pd.DataFrame:
    s = settings or get_settings()
    return _read_csv(s.dataset_file("submission_template"))


def load_panel(settings: Settings | None = None) -> pd.DataFrame:
    """Train + test concatenated into one continuous panel, sorted by (loan, month).

    Feature engineering runs over this combined panel so a test row at month 37
    can legitimately see its own history at months 1..36. That is point-in-time
    correct — the history is genuinely in the past — and it is the whole reason
    the problem is panel forecasting rather than cold start.
    """
    s = settings or get_settings()
    train = load_monthly("train", s)
    test = load_monthly("test", s)
    for col in s.require("data.target_columns"):
        if col not in test.columns:
            test[col] = np.nan
    panel = pd.concat([train, test], ignore_index=True, sort=False)
    return panel.sort_values(["loan_id", "month_index"], kind="mergesort").reset_index(drop=True)


def ingest_all(settings: Settings | None = None, *, strict: bool = True) -> IngestResult:
    s = settings or get_settings()
    frames: dict[str, pd.DataFrame] = {
        "static": load_static(s),
        "train": load_monthly("train", s),
        "test": load_monthly("test", s),
        "servicer": load_servicer(s),
        "macro": load_macro(s),
    }
    contracts: dict[str, TableContract] = {
        "static": STATIC_CONTRACT,
        "train": MONTHLY_TRAIN_CONTRACT,
        "test": MONTHLY_TEST_CONTRACT,
        "servicer": SERVICER_CONTRACT,
        "macro": MACRO_CONTRACT,
    }
    reports = {
        key: check_contract(frames[key], contracts[key], strict_values=strict)
        for key in frames
    }
    hashes = dataset_hashes(s.path("dataset_dir"), s.require("data.files"))
    result = IngestResult(
        frames=frames,
        contract_reports=reports,
        file_hashes=hashes,
        data_sha256=combined_data_sha256(hashes),
    )
    log.info(
        "ingest.complete",
        rows={k: int(len(v)) for k, v in frames.items()},
        data_sha256=result.data_sha256[:16],
    )
    return result


@lru_cache(maxsize=1)
def data_sha256() -> str:
    """Combined SHA256 of the input pack — stamped into every artifact."""
    s = get_settings()
    return combined_data_sha256(dataset_hashes(s.path("dataset_dir"), s.require("data.files")))


def file_sha256(key: str) -> str | None:
    s = get_settings()
    p = s.dataset_file(key)
    return sha256_file(p) if p.exists() else None
