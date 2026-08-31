"""Four-stage data profiler.

Stage A structural, Stage B distributional, Stage C missingness intelligence,
Stage D relationships and temporal integrity. Everything the design asserts as
"measured" is measured here at runtime, so a different data pack re-derives its
own censoring cliffs and state machine rather than inheriting ours.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from lpie.core.config import Settings, get_settings
from lpie.core.logging import get_logger
from lpie.core.timing import utcnow_iso
from lpie.profiling.missingness import missingness_intelligence
from lpie.profiling.relationships import relationship_intelligence, temporal_integrity

log = get_logger(__name__)

PERCENTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _gini(counts: np.ndarray) -> float:
    """Gini impurity of the level distribution (0 = single level, ->1 = uniform)."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    return float(1.0 - (p**2).sum())


def profile_column(series: pd.Series, *, top_k: int = 10) -> dict[str, Any]:
    n = int(len(series))
    n_null = int(series.isna().sum())
    non_null = series.dropna()
    n_unique = int(non_null.nunique())

    prof: dict[str, Any] = {
        "column": series.name,
        "dtype": str(series.dtype),
        "n": n,
        "n_null": n_null,
        "null_rate": round(n_null / n, 6) if n else None,
        "n_unique": n_unique,
        "cardinality_ratio": round(n_unique / max(n - n_null, 1), 6),
        "is_constant": n_unique <= 1,
        "is_degenerate": n_null == n,
        "memory_bytes": int(series.memory_usage(deep=True)),
    }

    is_numeric = pd.api.types.is_numeric_dtype(series)
    if is_numeric and not non_null.empty:
        values = pd.to_numeric(non_null, errors="coerce").dropna().astype("float64")
        prof["kind"] = "numeric"
        if values.empty:
            prof["kind"] = "numeric_empty"
            return prof
        desc = values.describe(percentiles=list(PERCENTILES))
        prof.update(
            {
                "mean": _f(desc.get("mean")),
                "std": _f(desc.get("std")),
                "min": _f(values.min()),
                "max": _f(values.max()),
                "percentiles": {
                    f"p{int(q * 100)}": _f(values.quantile(q)) for q in PERCENTILES
                },
                "skew": _f(values.skew()),
                "kurtosis": _f(values.kurtosis()),
                "zero_inflation_rate": round(float((values == 0).mean()), 6),
                "negative_rate": round(float((values < 0).mean()), 6),
                "iqr": _f(values.quantile(0.75) - values.quantile(0.25)),
                "mad": _f((values - values.median()).abs().median()),
                "histogram": _histogram(values),
            }
        )
    else:
        prof["kind"] = "categorical"
        counts = non_null.astype(str).value_counts()
        prof["top_categories"] = [
            {"value": str(k), "count": int(v), "share": round(float(v) / max(n, 1), 6)}
            for k, v in counts.head(top_k).items()
        ]
        arr = counts.to_numpy(dtype="float64")
        prof["entropy"] = round(_entropy(arr), 6)
        prof["normalised_entropy"] = (
            round(_entropy(arr) / math.log(len(arr)), 6) if len(arr) > 1 else 0.0
        )
        prof["gini"] = round(_gini(arr), 6)
        rare = counts[counts / max(n, 1) < 0.005]
        prof["n_rare_levels"] = int(len(rare))
        prof["rare_levels"] = [str(k) for k in rare.index[:20]]
        prof["mode"] = str(counts.index[0]) if len(counts) else None
        prof["mode_share"] = round(float(counts.iloc[0]) / max(n, 1), 6) if len(counts) else None
    return prof


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(v) else round(v, 6)


def _histogram(values: pd.Series, bins: int = 20) -> dict[str, list[float]]:
    lo, hi = float(values.min()), float(values.max())
    if not math.isfinite(lo) or not math.isfinite(hi) or lo == hi:
        return {"edges": [lo, hi], "counts": [int(len(values))]}
    counts, edges = np.histogram(values.to_numpy(), bins=bins, range=(lo, hi))
    return {"edges": [round(float(e), 6) for e in edges], "counts": [int(c) for c in counts]}


def unseen_categories(
    train: pd.DataFrame, current: pd.DataFrame, columns: list[str] | None = None
) -> dict[str, list[str]]:
    """Levels present in `current` but absent from `train` — VR-018's evidence."""
    cols = columns or [
        c for c in current.columns
        if c in train.columns and not pd.api.types.is_numeric_dtype(current[c])
    ]
    out: dict[str, list[str]] = {}
    for c in cols:
        seen = set(train[c].dropna().astype(str).unique())
        now = set(current[c].dropna().astype(str).unique())
        diff = sorted(now - seen)
        if diff:
            out[c] = diff[:50]
    return out


def derive_state_machine(df: pd.DataFrame) -> dict[str, Any]:
    """Re-derive the empirical transition matrix and absorbing set from the data.

    The config ships a legal-transition mask, but the mask must be *earned* from
    whatever pack is loaded — a different pack has a different state machine.
    """
    if "next_state" not in df.columns or "current_status" not in df.columns:
        return {"available": False}
    sub = df.dropna(subset=["current_status", "next_state"])
    if sub.empty:
        return {"available": False}
    counts = pd.crosstab(sub["current_status"], sub["next_state"])
    matrix = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)
    absorbing = [s for s in matrix.index if s in matrix.columns and matrix.loc[s, s] >= 0.999]
    legal = {
        state: sorted(matrix.columns[matrix.loc[state] > 0].tolist())
        for state in matrix.index
    }
    return {
        "available": True,
        "states": sorted(set(matrix.index) | set(matrix.columns)),
        "transition_matrix": {
            src: {dst: round(float(v), 6) for dst, v in row.items()} for src, row in matrix.iterrows()
        },
        "transition_counts": {
            src: {dst: int(v) for dst, v in row.items()} for src, row in counts.iterrows()
        },
        "absorbing_states": absorbing,
        "legal_transitions": legal,
    }


def derive_censoring(
    df: pd.DataFrame, targets: dict[str, int], *, panel_max: int | None = None
) -> dict[str, Any]:
    """Measure each forward target's censoring behaviour at the panel edge.

    This is Insight 1 of the design, computed rather than assumed. Two distinct
    things are measured and reported separately, because they are different
    defects:

    * **collapse** — the trailing months where the label window runs past the
      end of the panel, so the rate decays to exactly zero. The *declared*
      horizon is what determines how many rows to mask (`panel_max - horizon`);
      the measured collapse is the evidence that the mask is necessary.
    * **frozen plateau** — a run of months carrying an identical non-zero rate.
      `next_12m_prepayment_flag` freezes at 0.5504 from month 25, which means it
      stopped being a 12-month-forward label and became "prepaid at any point
      through end-of-panel". A frozen label must not be consumed directly; the
      hazard core derives the horizon probability from monthly transitions
      instead.
    """
    if "month_index" not in df.columns:
        return {"available": False}

    months = pd.to_numeric(df["month_index"], errors="coerce")
    pmax = int(panel_max if panel_max is not None else months.max())
    out: dict[str, Any] = {"available": True, "panel_max_month": pmax, "targets": {}}

    for target, horizon in targets.items():
        if target not in df.columns:
            continue
        series = pd.to_numeric(df[target], errors="coerce")
        if series.notna().sum() == 0:
            continue
        by_month = series.groupby(months).mean().sort_index()
        values = by_month.to_numpy(dtype="float64")
        index = by_month.index.to_numpy()

        # trailing run of exact zeros
        zero_run = 0
        for v in values[::-1]:
            if v <= 1e-12:
                zero_run += 1
            else:
                break
        first_zero_month = int(index[len(values) - zero_run]) if zero_run else None

        plateau_from, plateau_value = _frozen_plateau(index, values)

        declared_max_valid = max(int(pmax - horizon), 1)
        out["targets"][target] = {
            "declared_horizon_months": int(horizon),
            "overall_rate": round(float(series.mean()), 6),
            "rate_by_month": {int(k): round(float(v), 6) for k, v in by_month.items()},
            "trailing_zero_months": zero_run,
            "first_all_zero_month": first_zero_month,
            "frozen_plateau": plateau_from is not None,
            "frozen_plateau_from_month": plateau_from,
            "frozen_plateau_value": plateau_value,
            "max_valid_month": declared_max_valid,
            "n_rows_masked": int((months > declared_max_valid).sum()),
            "verdict": _censoring_verdict(zero_run, plateau_from),
        }
    return out


def _frozen_plateau(
    index: np.ndarray, values: np.ndarray, *, min_run: int = 4, tol: float = 1e-6
) -> tuple[int | None, float | None]:
    """Longest trailing run of a repeated non-zero rate, ignoring the zero tail."""
    end = len(values)
    while end > 0 and values[end - 1] <= 1e-12:
        end -= 1
    if end == 0:
        return None, None
    last = values[end - 1]
    start = end - 1
    while start > 0 and abs(values[start - 1] - last) <= tol:
        start -= 1
    if end - start < min_run:
        return None, None
    return int(index[start]), round(float(last), 6)


def _censoring_verdict(zero_run: int, plateau_from: int | None) -> str:
    if zero_run and plateau_from is not None:
        return "censored_and_frozen"
    if zero_run:
        return "censored_at_panel_edge"
    if plateau_from is not None:
        return "frozen_plateau"
    return "uncensored"


def profile_frame(
    df: pd.DataFrame,
    *,
    name: str = "frame",
    top_k: int = 10,
    max_relationship_columns: int = 24,
    sample_rows: int | None = 120_000,
    include_relationships: bool = True,
    include_missingness: bool = True,
    settings: Settings | None = None,
) -> dict[str, Any]:
    s = settings or get_settings()
    seed = s.seed
    work = df
    sampled = False
    if sample_rows is not None and len(df) > sample_rows:
        work = df.sample(sample_rows, random_state=seed)
        sampled = True

    columns = [profile_column(df[c], top_k=top_k) for c in df.columns]

    report: dict[str, Any] = {
        "name": name,
        "computed_at": utcnow_iso(),
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1e6, 3),
        "sampled_for_expensive_stages": sampled,
        "sample_rows": int(len(work)) if sampled else int(len(df)),
        "schema": {c: str(df[c].dtype) for c in df.columns},
        "columns": columns,
        "degenerate_columns": [c["column"] for c in columns if c["is_degenerate"]],
        "constant_columns": [c["column"] for c in columns if c["is_constant"]],
        "high_null_columns": [
            {"column": c["column"], "null_rate": c["null_rate"]}
            for c in columns
            if (c["null_rate"] or 0) > 0.01
        ],
    }

    if include_missingness:
        report["missingness"] = missingness_intelligence(work, seed=seed)
    if include_relationships:
        report["relationships"] = relationship_intelligence(
            work, max_columns=max_relationship_columns, seed=seed
        )
    report["temporal_integrity"] = temporal_integrity(df)

    if "next_state" in df.columns:
        report["state_machine"] = derive_state_machine(df)
    horizons = {
        spec["target"]: int(spec.get("horizon", 0))
        for spec in (s.section("heads") or {}).values()
        if spec.get("target") in df.columns and pd.api.types.is_numeric_dtype(df[spec["target"]])
    }
    if horizons:
        report["censoring"] = derive_censoring(df, horizons)

    return report
