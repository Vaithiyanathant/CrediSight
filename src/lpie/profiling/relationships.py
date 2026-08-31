"""Relationship and consistency intelligence (profiler Stage D).

Correlation is the easy half. The parts that earn their keep here are functional
dependency mining (which discovers `current_status -> days_past_due` and
quantifies exactly how often it breaks — that break *is* VR-002), association
rule mining over binned features, and temporal integrity checks that surface the
duplicate `(loan_id, reporting_month)` corruption.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

MAX_CATEGORICAL_LEVELS = 60


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 1]


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        n = df[c].nunique(dropna=True)
        if 1 < n <= MAX_CATEGORICAL_LEVELS:
            out.append(c)
    return out


def correlations(df: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    if len(cols) < 2:
        return {"pearson": {}, "spearman": {}, "high_pairs": []}
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    pearson = sub.corr(method="pearson")
    spearman = sub.corr(method="spearman")
    pairs = []
    for a, b in combinations(cols, 2):
        p = pearson.loc[a, b]
        s = spearman.loc[a, b]
        if pd.notna(p) and abs(p) >= 0.85:
            pairs.append(
                {"a": a, "b": b, "pearson": round(float(p), 6),
                 "spearman": round(float(s), 6) if pd.notna(s) else None}
            )
    pairs.sort(key=lambda r: -abs(r["pearson"]))
    return {
        "pearson": _matrix_to_dict(pearson),
        "spearman": _matrix_to_dict(spearman),
        "high_pairs": pairs[:25],
    }


def _matrix_to_dict(m: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    return {
        str(a): {
            str(b): (None if pd.isna(v) else round(float(v), 6)) for b, v in row.items()
        }
        for a, row in m.iterrows()
    }


def cramers_v(a: pd.Series, b: pd.Series) -> float | None:
    """Bias-corrected Cramér's V for categorical<->categorical association."""
    table = pd.crosstab(a, b)
    if table.size == 0 or table.shape[0] < 2 or table.shape[1] < 2:
        return None
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    if n == 0:
        return None
    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
    kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return None
    return float(math.sqrt(phi2corr / denom))


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float | None:
    """eta-squared: categorical -> numeric association."""
    mask = categories.notna() & values.notna()
    if mask.sum() < 10:
        return None
    cats, vals = categories[mask], pd.to_numeric(values[mask], errors="coerce")
    grand_mean = vals.mean()
    ss_total = float(((vals - grand_mean) ** 2).sum())
    if ss_total <= 0:
        return None
    ss_between = 0.0
    for _, group in vals.groupby(cats, observed=True):
        ss_between += len(group) * (group.mean() - grand_mean) ** 2
    return float(math.sqrt(max(ss_between, 0.0) / ss_total))


def functional_dependencies(
    df: pd.DataFrame, columns: list[str], *, min_strength: float = 0.80
) -> list[dict[str, Any]]:
    """Mine A -> B dependencies via normalised conditional entropy.

    strength = 1 - H(B|A)/H(B). A strength near 1 with a non-zero break rate is
    a rule candidate: the relationship is real, and the exceptions are defects.
    """
    out: list[dict[str, Any]] = []
    for a, b in combinations(columns, 2):
        for src, dst in ((a, b), (b, a)):
            res = _dependency_strength(df[src], df[dst])
            if res and res["strength"] >= min_strength:
                out.append({"determinant": src, "dependent": dst, **res})
    out.sort(key=lambda r: (-r["strength"], r["break_rate"]))
    return out[:30]


def _dependency_strength(src: pd.Series, dst: pd.Series) -> dict[str, Any] | None:
    mask = src.notna() & dst.notna()
    if mask.sum() < 50:
        return None
    s, d = src[mask].astype(str), dst[mask].astype(str)
    counts = d.value_counts(normalize=True).to_numpy()
    h_dst = float(-(counts * np.log(counts)).sum())
    if h_dst <= 1e-9:
        return None

    # A near-unique determinant "explains" anything; that is an artefact of
    # cardinality, not a dependency. Require the key space to be well below n.
    n_keys = int(s.nunique())
    if n_keys > max(20, 0.05 * mask.sum()):
        return None

    joint = pd.crosstab(s, d)
    row_totals = joint.sum(axis=1)
    probs = joint.div(row_totals, axis=0).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probs > 0, -probs * np.log(probs), 0.0)
    h_cond_rows = terms.sum(axis=1)
    weights = (row_totals / row_totals.sum()).to_numpy()
    h_cond = float((weights * h_cond_rows).sum())

    strength = float(1.0 - h_cond / h_dst)
    # Break rate: share of rows not matching the modal dependent value for their key.
    modal = joint.idxmax(axis=1)
    predicted = s.map(modal)
    break_rate = float((predicted != d).mean())
    return {
        "strength": round(strength, 6),
        "break_rate": round(break_rate, 6),
        "n_observed": int(mask.sum()),
        "mapping_size": int(len(modal)),
        "is_rule_candidate": bool(strength >= 0.95 and 0.0 < break_rate < 0.10),
    }


def association_rules(
    df: pd.DataFrame,
    columns: list[str],
    consequent: str,
    *,
    min_support: float = 0.005,
    min_confidence: float = 0.6,
    max_antecedent_size: int = 2,
    max_items: int = 40,
) -> list[dict[str, Any]]:
    """Mine `{feature=level, ...} => consequent=1` rules with support/confidence/lift.

    Implemented directly rather than via a dependency: at 2 items over ~40
    candidate literals the search space is trivial and an exact enumeration is
    both faster and easier to audit than an FP-growth library call.
    """
    if consequent not in df.columns:
        return []
    target = pd.to_numeric(df[consequent], errors="coerce")
    mask = target.notna()
    if mask.sum() < 100:
        return []
    y = (target[mask] > 0).to_numpy()
    base_rate = float(y.mean())
    if base_rate <= 0:
        return []
    n = len(y)

    literals: list[tuple[str, str, np.ndarray]] = []
    for c in columns:
        if c == consequent or c not in df.columns:
            continue
        col = df.loc[mask, c]
        if pd.api.types.is_numeric_dtype(col):
            binned = pd.qcut(pd.to_numeric(col, errors="coerce"), 4, duplicates="drop")
            levels = binned.astype(str)
        else:
            levels = col.astype(str)
        vc = levels.value_counts()
        for level, count in vc.head(8).items():
            if level in {"nan", "<NA>"} or count / n < min_support:
                continue
            literals.append((c, str(level), (levels == level).to_numpy()))
        if len(literals) >= max_items * 3:
            break

    rules: list[dict[str, Any]] = []

    def _emit(items: list[tuple[str, str]], indicator: np.ndarray) -> None:
        support = float(indicator.mean())
        if support < min_support:
            return
        conf = float(y[indicator].mean()) if indicator.any() else 0.0
        if conf < min_confidence:
            return
        rules.append(
            {
                "antecedent": [f"{c}={v}" for c, v in items],
                "consequent": f"{consequent}=1",
                "support": round(support, 6),
                "confidence": round(conf, 6),
                "lift": round(conf / base_rate, 4),
                "n_matched": int(indicator.sum()),
            }
        )

    for c, v, ind in literals:
        _emit([(c, v)], ind)
    if max_antecedent_size >= 2:
        for (c1, v1, i1), (c2, v2, i2) in combinations(literals, 2):
            if c1 == c2:
                continue
            _emit([(c1, v1), (c2, v2)], i1 & i2)

    rules.sort(key=lambda r: (-r["lift"], -r["support"]))
    return rules[:max_items]


def mutual_information(
    df: pd.DataFrame, target: str, columns: list[str], *, seed: int = 0, max_rows: int = 40_000
) -> list[dict[str, Any]]:
    """MI against a target — catches non-monotone dependence correlation misses."""
    if target not in df.columns:
        return []
    work = df if len(df) <= max_rows else df.sample(max_rows, random_state=seed)
    y_raw = work[target]
    cols = [c for c in columns if c != target and c in work.columns]
    if not cols:
        return []

    X = np.column_stack(
        [
            pd.to_numeric(work[c], errors="coerce").to_numpy(dtype="float64")
            if pd.api.types.is_numeric_dtype(work[c])
            else pd.Categorical(work[c].astype("object")).codes.astype("float64")
            for c in cols
        ]
    )
    finite = np.isfinite(X).all(axis=1)
    X = X[finite]
    y_raw = y_raw[finite]
    if len(X) < 100:
        return []

    try:
        if pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique() > 10:
            scores = mutual_info_regression(X, pd.to_numeric(y_raw).to_numpy(), random_state=seed)
        else:
            y = pd.Categorical(y_raw.astype("object")).codes
            keep = y >= 0
            scores = mutual_info_classif(X[keep], y[keep], random_state=seed)
    except Exception:  # pragma: no cover
        return []

    out = [{"feature": c, "mutual_information": round(float(s), 6)} for c, s in zip(cols, scores, strict=False)]
    out.sort(key=lambda r: -r["mutual_information"])
    return out


def temporal_integrity(df: pd.DataFrame) -> dict[str, Any]:
    """Panel-shape checks: gaps, duplicates, and the reporting_month corruption."""
    out: dict[str, Any] = {}
    if not {"loan_id", "month_index"}.issubset(df.columns):
        return {"available": False}

    grp = df.groupby("loan_id")["month_index"]
    sizes = grp.size()
    spans = grp.max() - grp.min() + 1
    gaps = (spans - sizes)
    out["available"] = True
    out["n_entities"] = int(sizes.shape[0])
    out["months_per_entity"] = {
        "min": int(sizes.min()), "max": int(sizes.max()), "mean": round(float(sizes.mean()), 4),
    }
    out["n_entities_with_gaps"] = int((gaps > 0).sum())
    out["entities_with_gaps_sample"] = gaps[gaps > 0].index[:10].tolist()
    out["duplicate_loan_month_index"] = int(df.duplicated(subset=["loan_id", "month_index"]).sum())

    if "reporting_month" in df.columns:
        dup_pairs = df.duplicated(subset=["loan_id", "reporting_month"], keep=False)
        out["duplicate_loan_reporting_month_rows"] = int(dup_pairs.sum())
        out["duplicate_loan_reporting_month_pairs"] = int(
            df.loc[dup_pairs, ["loan_id", "reporting_month"]].drop_duplicates().shape[0]
        )
        collisions = (
            df.groupby("reporting_month")["month_index"].nunique().sort_values(ascending=False)
        )
        colliding = collisions[collisions > 1]
        out["reporting_months_mapping_multiple_month_index"] = {
            str(k): int(v) for k, v in colliding.head(10).items()
        }
        out["reporting_month_is_trustworthy"] = bool(colliding.empty)
    return out


def relationship_intelligence(
    df: pd.DataFrame, *, max_columns: int = 24, seed: int = 0
) -> dict[str, Any]:
    numeric = _numeric_columns(df)[:max_columns]
    categorical = _categorical_columns(df)[:max_columns]

    cat_assoc: list[dict[str, Any]] = []
    for a, b in combinations(categorical, 2):
        v = cramers_v(df[a], df[b])
        if v is not None and v >= 0.30:
            cat_assoc.append({"a": a, "b": b, "cramers_v": round(v, 6)})
    cat_assoc.sort(key=lambda r: -r["cramers_v"])

    eta: list[dict[str, Any]] = []
    for c in categorical:
        for n in numeric:
            r = correlation_ratio(df[c], df[n])
            if r is not None and r >= 0.30:
                eta.append({"categorical": c, "numeric": n, "eta": round(r, 6)})
    eta.sort(key=lambda r: -r["eta"])

    return {
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "correlations": correlations(df, numeric),
        "categorical_associations": cat_assoc[:25],
        "correlation_ratios": eta[:25],
        "functional_dependencies": functional_dependencies(df, categorical + numeric[:8]),
    }
