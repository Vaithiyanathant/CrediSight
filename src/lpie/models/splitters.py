"""Purged + embargoed walk-forward validation on `month_index`.

Two distinct controls, both necessary:

**Censoring mask.** A row at month m with an h-month forward label only has a
complete label window if `m <= panel_max - h`. Beyond that the label decays to
zero not because nothing happened but because the panel ended. Those rows are
*masked out of the loss*, never fed as negatives.

**Embargo.** A training row at month 30 with a 12-month label has its outcome
determined by months 31-42. If validation is months 31-36, the training label
already contains the validation period's outcomes. So the embargo is exactly the
horizon length: months (T, T+h] are purged between train and validation.

Random row splits and GroupKFold on `loan_id` are both wrong here and are
implemented nowhere in this codebase.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from lpie.core.config import Settings, get_settings
from lpie.core.exceptions import InvalidRequestError
from lpie.core.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Fold:
    fold: int
    train_months: tuple[int, ...]
    embargo_months: tuple[int, ...]
    valid_months: tuple[int, ...]

    @property
    def train_window(self) -> str:
        return _window(self.train_months)

    @property
    def embargo_window(self) -> str:
        return _window(self.embargo_months)

    @property
    def valid_window(self) -> str:
        return _window(self.valid_months)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "train_months": list(self.train_months),
            "embargo_months": list(self.embargo_months),
            "valid_months": list(self.valid_months),
            "train_window": self.train_window,
            "embargo_window": self.embargo_window,
            "valid_window": self.valid_window,
        }


@dataclass
class SplitPlan:
    head: str
    horizon: int
    panel_max_month: int
    max_valid_month: int
    folds: list[Fold]
    calibration_months: tuple[int, ...]
    final_train_months: tuple[int, ...]
    scoring_months: tuple[int, ...] = field(default_factory=tuple)
    fold_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["folds"] = [f.to_dict() for f in self.folds]
        d["calibration_window"] = _window(self.calibration_months)
        d["final_train_window"] = _window(self.final_train_months)
        d["scoring_window"] = _window(self.scoring_months)
        for key in ("calibration_months", "final_train_months", "scoring_months"):
            d[key] = list(d[key])
        return d

    def __iter__(self) -> Iterator[Fold]:
        return iter(self.folds)


def _window(months: tuple[int, ...] | list[int]) -> str:
    if not months:
        return "none"
    lo, hi = min(months), max(months)
    return f"{lo}-{hi}" if lo != hi else str(lo)


def censoring_mask(
    month_index: pd.Series, horizon: int, panel_max_month: int
) -> pd.Series:
    """True where the label window is complete. Everything else leaves the loss."""
    max_valid = panel_max_month - int(horizon)
    return pd.to_numeric(month_index, errors="coerce") <= max_valid


def build_split_plan(
    head: str,
    *,
    horizon: int | None = None,
    panel_max_month: int | None = None,
    scoring_months: list[int] | None = None,
    settings: Settings | None = None,
) -> SplitPlan:
    """Expanding-window walk-forward with a horizon-length embargo.

    An expanding window is correct here: there is no regime change to forget and
    the earliest data is only 36 months old, so discarding it would throw away
    signal for no benefit.
    """
    s = settings or get_settings()
    head_cfg = s.get(f"heads.{head}")
    if head_cfg is None:
        raise InvalidRequestError(f"Unknown head '{head}'", details={"head": head})

    h = int(horizon if horizon is not None else head_cfg.get("horizon", 0))
    panel_max = int(panel_max_month if panel_max_month is not None else s.get("data.train_month_max", 36))
    max_valid = max(int(panel_max - h), 1)

    split_cfg = s.section("validation_split")
    override = (split_cfg.get("per_head") or {}).get(head, {})

    n_folds = int(override.get("n_folds", split_cfg.get("n_folds", 4)))
    step = int(override.get("step_months", split_cfg.get("step_months", 3)))
    valid_len = int(override.get("valid_window_months", split_cfg.get("valid_window_months", 6)))
    min_train = int(override.get("min_train_months", split_cfg.get("min_train_months", 15)))
    min_train_floor = int(split_cfg.get("min_train_months_floor", 6))

    # Folds are laid out *backward* from the censoring bound, so the last fold is
    # always the most recent honest window available. With a 12-month horizon on
    # a 36-month panel this is binding: max_valid is 24 and the embargo consumes
    # 12 more months, so at most a handful of folds exist. We report the count
    # rather than manufacturing folds the data cannot support.
    folds: list[Fold] = []
    for k in range(n_folds):
        valid_end = max_valid - k * step
        valid_start = valid_end - valid_len + 1
        if valid_start < 1:
            break
        train_end = valid_start - h - 1
        if train_end < min_train_floor:
            break
        folds.append(
            Fold(
                fold=0,
                train_months=tuple(range(1, train_end + 1)),
                embargo_months=tuple(range(train_end + 1, train_end + h + 1)),
                valid_months=tuple(range(valid_start, valid_end + 1)),
            )
        )
    folds = [
        Fold(fold=i + 1, train_months=f.train_months,
             embargo_months=f.embargo_months, valid_months=f.valid_months)
        for i, f in enumerate(reversed(folds))
    ]
    # Drop folds whose training window is below the configured comfort minimum,
    # but never drop the last one — a single honest fold beats zero validation.
    trimmed = [f for f in folds if len(f.train_months) >= min_train]
    if trimmed:
        folds = [
            Fold(fold=i + 1, train_months=f.train_months,
                 embargo_months=f.embargo_months, valid_months=f.valid_months)
            for i, f in enumerate(trimmed)
        ]

    if not folds:
        train_end = max(max_valid - valid_len - h, min_train_floor)
        valid_start = min(train_end + h + 1, max_valid)
        folds = [
            Fold(
                fold=1,
                train_months=tuple(range(1, train_end + 1)),
                embargo_months=tuple(range(train_end + 1, train_end + h + 1)),
                valid_months=tuple(range(valid_start, max_valid + 1)),
            )
        ]

    # The calibration slice sits after the final training window and before
    # production scoring, so calibration is learned genuinely out-of-time —
    # mirroring how the production model meets months 37-42.
    calib_len = int(s.get("validation_split.calibration_slice_months", 6))
    calib_end = max_valid
    calib_start = max(calib_end - calib_len + 1, 1)
    calibration = tuple(range(calib_start, calib_end + 1))

    # The final model trains on everything censoring-valid, with the calibration
    # slice held out so the isotonic fit never sees its own training rows.
    final_train_end = max(calib_start - 1 - h, 1)
    final_train = tuple(range(1, final_train_end + 1))

    scoring = tuple(scoring_months or range(
        int(s.get("data.test_month_min", 37)), int(s.get("data.test_month_max", 42)) + 1
    ))

    note = ""
    if len(folds) < n_folds:
        note = (
            f"Only {len(folds)} of {n_folds} requested folds are supported: with a "
            f"{h}-month horizon on a {panel_max}-month panel the censoring bound is "
            f"month {max_valid} and the embargo consumes {h} further months. "
            "Reported metrics carry this fold count explicitly."
        )

    plan = SplitPlan(
        head=head,
        horizon=h,
        panel_max_month=panel_max,
        max_valid_month=max_valid,
        folds=folds,
        calibration_months=calibration,
        final_train_months=final_train,
        scoring_months=scoring,
        fold_note=note,
    )
    _assert_no_overlap(plan)
    return plan


def _assert_no_overlap(plan: SplitPlan) -> None:
    """Guard the guard: assert the embargo really separates train from validation."""
    for fold in plan.folds:
        train, valid = set(fold.train_months), set(fold.valid_months)
        if train & valid:
            raise InvalidRequestError(
                f"Fold {fold.fold} for head '{plan.head}' has overlapping train/valid months",
                details=fold.to_dict(),
            )
        # Every training row's label window must end before validation begins.
        if train and valid:
            latest_label_end = max(train) + plan.horizon
            if latest_label_end >= min(valid):
                raise InvalidRequestError(
                    f"Fold {fold.fold} for head '{plan.head}': training labels reach month "
                    f"{latest_label_end}, which is not before validation start {min(valid)}",
                    details=fold.to_dict(),
                )
    calib = set(plan.calibration_months)
    final_train = set(plan.final_train_months)
    if calib & final_train:
        raise InvalidRequestError(
            f"Head '{plan.head}': calibration slice overlaps the final training window",
            details=plan.to_dict(),
        )
    if final_train and calib:
        if max(final_train) + plan.horizon >= min(calib):
            raise InvalidRequestError(
                f"Head '{plan.head}': final training labels reach into the calibration slice",
                details=plan.to_dict(),
            )


def split_indices(
    month_index: pd.Series, fold: Fold, *, horizon: int, panel_max_month: int
) -> tuple[np.ndarray, np.ndarray]:
    """(train_idx, valid_idx) with the censoring mask already applied to both."""
    months = pd.to_numeric(month_index, errors="coerce")
    valid_label = censoring_mask(months, horizon, panel_max_month)
    train = months.isin(fold.train_months) & valid_label
    valid = months.isin(fold.valid_months) & valid_label
    return np.flatnonzero(train.to_numpy()), np.flatnonzero(valid.to_numpy())


def all_plans(
    *, panel_max_month: int | None = None, settings: Settings | None = None
) -> dict[str, SplitPlan]:
    s = settings or get_settings()
    return {
        head: build_split_plan(head, panel_max_month=panel_max_month, settings=s)
        for head in (s.section("heads") or {})
    }
