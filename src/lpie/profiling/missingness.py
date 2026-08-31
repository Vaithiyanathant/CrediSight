"""Missingness intelligence: patterns, co-occurrence, and MCAR/MAR triage.

Rates alone are not intelligence. What matters is whether nulls arrive in blocks
(a broken feed) or independently (injected noise), and whether missingness is
*predictable from the other columns* — because a predictable null is information
the model will happily learn, and a drifting one is a production failure.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

MIN_NULLS_FOR_TRIAGE = 200
MAX_TRIAGE_ROWS = 40_000


def missingness_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if df[c].isna().any()]
    return df[cols].isna() if cols else pd.DataFrame(index=df.index)


def cooccurrence(mask: pd.DataFrame) -> dict[str, Any]:
    """Jaccard co-occurrence between null indicators + the top joint patterns."""
    cols = list(mask.columns)
    if not cols:
        return {"columns": [], "jaccard": {}, "top_patterns": []}

    arr = mask.to_numpy(dtype=bool)
    inter = arr.T.astype("int64") @ arr.astype("int64")
    counts = arr.sum(axis=0).astype("int64")
    union = counts[:, None] + counts[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(union > 0, inter / union, 0.0)

    patterns = (
        mask.astype(int).astype(str).agg("".join, axis=1).value_counts().head(12)
    )
    return {
        "columns": cols,
        "jaccard": {
            cols[i]: {cols[j]: round(float(jac[i, j]), 6) for j in range(len(cols))}
            for i in range(len(cols))
        },
        "top_patterns": [
            {
                "pattern": p,
                "columns_null": [cols[i] for i, ch in enumerate(p) if ch == "1"],
                "count": int(n),
                "share": round(float(n) / max(len(mask), 1), 6),
            }
            for p, n in patterns.items()
        ],
        "independent_null_hypothesis": _independence_check(arr, counts, len(mask)),
    }


def _independence_check(arr: np.ndarray, counts: np.ndarray, n: int) -> dict[str, Any]:
    """Compare observed joint-null counts to the product of marginals.

    Ratio ~1 across the board means nulls were injected independently (MCAR-like
    mechanism); ratios >> 1 mean nulls arrive in blocks, i.e. a broken feed.
    """
    if n == 0 or len(counts) < 2:
        return {"max_lift": None, "verdict": "insufficient_data"}
    inter = arr.T.astype("int64") @ arr.astype("int64")
    expected = np.outer(counts, counts) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where(expected > 0, inter / expected, np.nan)
    np.fill_diagonal(lift, np.nan)
    max_lift = float(np.nanmax(lift)) if np.isfinite(lift).any() else None
    verdict = (
        "insufficient_data" if max_lift is None
        else "independent_injection" if max_lift < 1.5
        else "block_structured"
    )
    return {"max_lift": round(max_lift, 4) if max_lift is not None else None, "verdict": verdict}


def mcar_mar_triage(
    df: pd.DataFrame, *, seed: int = 0, max_rows: int = MAX_TRIAGE_ROWS
) -> dict[str, Any]:
    """For each null-bearing column, fit `is_null ~ other columns`.

    AUC ~ 0.5 -> MCAR (nulls unrelated to observed data).
    AUC >> 0.5 -> MAR, and the top permutation drivers name *what* predicts it.
    MNAR is not testable from observed data; we say so rather than guessing.
    """
    out: dict[str, Any] = {}
    null_cols = [c for c in df.columns if df[c].isna().sum() >= MIN_NULLS_FOR_TRIAGE]
    if not null_cols:
        return out

    work = df if len(df) <= max_rows else df.sample(max_rows, random_state=seed)

    for target_col in null_cols:
        y = work[target_col].isna().astype(int).to_numpy()
        if y.mean() in (0.0, 1.0):
            continue
        feature_cols = [c for c in work.columns if c != target_col and not c.startswith("_")]
        X = _encode(work[feature_cols])
        if X.shape[1] == 0:
            continue
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, random_state=seed, stratify=y
            )
            model = HistGradientBoostingClassifier(
                max_iter=60, max_depth=4, learning_rate=0.15, random_state=seed
            )
            model.fit(X_tr, y_tr)
            auc = float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
        except Exception:  # pragma: no cover - degenerate column
            continue

        verdict = "MCAR" if auc < 0.60 else ("weakly_MAR" if auc < 0.75 else "MAR")
        out[target_col] = {
            "null_rate": round(float(y.mean()), 6),
            "is_null_auc": round(auc, 4),
            "verdict": verdict,
            "note": (
                "MNAR cannot be tested from observed data; a MAR verdict here means "
                "missingness is predictable from other observed columns."
            ),
            "top_drivers": _top_drivers(model, X_te, y_te, feature_cols, seed=seed) if auc >= 0.60 else [],
        }
    return out


def _encode(df: pd.DataFrame) -> np.ndarray:
    """Numeric passthrough + integer codes for categoricals. NaN preserved."""
    parts: list[np.ndarray] = []
    for c in df.columns:
        col = df[c]
        if pd.api.types.is_numeric_dtype(col):
            parts.append(pd.to_numeric(col, errors="coerce").to_numpy(dtype="float64"))
        elif pd.api.types.is_datetime64_any_dtype(col):
            parts.append(col.astype("int64").where(col.notna(), np.nan).to_numpy(dtype="float64"))
        else:
            codes = pd.Categorical(col.astype("object")).codes.astype("float64")
            codes[codes < 0] = np.nan
            parts.append(codes)
    if not parts:
        return np.empty((len(df), 0))
    return np.column_stack(parts)


def _top_drivers(
    model: Any, X: np.ndarray, y: np.ndarray, names: list[str], *, seed: int, k: int = 5
) -> list[dict[str, Any]]:
    from sklearn.inspection import permutation_importance

    try:
        r = permutation_importance(
            model, X, y, n_repeats=3, random_state=seed, scoring="roc_auc", n_jobs=1
        )
    except Exception:  # pragma: no cover
        return []
    order = np.argsort(-r.importances_mean)[:k]
    return [
        {"feature": names[i], "importance": round(float(r.importances_mean[i]), 6)}
        for i in order
        if r.importances_mean[i] > 0
    ]


def missingness_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    """Per-column change in null rate between two windows.

    This is the design's headline finding: `credit_score_band` and
    `document_status` null rates collapse between the train tail and the test
    window, which makes naive `is_null` indicators the most drifted features in
    the dataset.
    """
    cols = sorted(set(reference.columns) & set(current.columns))
    rows = []
    for c in cols:
        ref_rate = float(reference[c].isna().mean())
        cur_rate = float(current[c].isna().mean())
        delta = cur_rate - ref_rate
        rows.append(
            {
                "column": c,
                "ref_null_rate": round(ref_rate, 6),
                "cur_null_rate": round(cur_rate, 6),
                "delta": round(delta, 6),
                "abs_delta": round(abs(delta), 6),
                "verdict": (
                    "stable" if abs(delta) < 0.01
                    else "moderate_shift" if abs(delta) < 0.03
                    else "severe_shift"
                ),
            }
        )
    rows.sort(key=lambda r: -r["abs_delta"])
    return {
        "columns": rows,
        "n_severe": sum(1 for r in rows if r["verdict"] == "severe_shift"),
        "worst": rows[:10],
    }


def missingness_intelligence(df: pd.DataFrame, *, seed: int = 0) -> dict[str, Any]:
    mask = missingness_matrix(df)
    return {
        "null_bearing_columns": list(mask.columns),
        "n_complete_rows": int((~mask.any(axis=1)).sum()) if not mask.empty else int(len(df)),
        "complete_row_rate": round(
            float((~mask.any(axis=1)).mean()) if not mask.empty else 1.0, 6
        ),
        "cooccurrence": cooccurrence(mask),
        "triage": mcar_mar_triage(df, seed=seed),
    }
