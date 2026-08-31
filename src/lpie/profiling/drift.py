"""Drift monitoring: PSI, KS, Jensen-Shannon, missingness delta, adversarial AUC.

Missingness delta is a first-class metric here, not an afterthought. In this data
pack the null rates of `credit_score_band`, `document_status` and `days_past_due`
collapse between the train tail and the test window — which makes naive
`is_null` indicators the *most drifted* features in the dataset. A drift report
that only looked at value distributions would miss the single failure mode most
likely to break the model in production.

Two classes of column are held out of the comparison so the remaining signal is
actionable. *Non-model columns* (targets, banned features) are dropped outright:
the forward-looking labels are 100% null in the scoring window by construction,
which pins the adversarial AUC at 1.0 regardless of population stability.
*Seasoning features* (`loan_age_months`, `remaining_term_months`) are measured
and reported but cannot fire the retrain trigger: on a healthy amortizing book
they shift by exactly N whenever the windows are N months apart. Without both
holdouts this report returns FAIL on every window pair forever.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lpie.core.config import Settings, get_settings
from lpie.core.logging import get_logger
from lpie.core.timing import utcnow_iso

log = get_logger(__name__)

EPS = 1e-6

# Columns excluded from every drift computation. These are the *definition* of
# the split (month_index, reporting_month) or pure identity (loan_id): an
# adversarial classifier that sees them scores AUC 1.0 by construction and tells
# you nothing. Excluding them is what makes the remaining AUC meaningful.
DRIFT_EXCLUDED_COLUMNS: frozenset[str] = frozenset({
    "loan_id", "month_index", "reporting_month", "origination_month",
    "last_updated_at", "_split", "_ingest_id", "_row_hash",
})


# Features that move deterministically as the book seasons: compare any two
# windows N months apart and they shift by exactly N, on a perfectly healthy
# panel. They are still measured and reported (flagged `seasoning: True`), but
# they cannot *fire* the retrain trigger and they are hidden from the adversarial
# classifier — `loan_age_months` alone separates two consecutive windows almost
# perfectly, which pushes the AUC past 0.80 by construction and turns the retrain
# alert into noise. Overridable via `drift.seasoning_features`.
DEFAULT_SEASONING_FEATURES: frozenset[str] = frozenset({
    "loan_age_months", "remaining_term_months",
})


def non_model_columns(settings: Settings | None = None) -> frozenset[str]:
    """Targets and banned columns — never seen by the model, never drift signal.

    The forward-looking labels (`next_state`, `next_3m_delinquency_flag`, ...) are
    100% null in the scoring window by construction: the look-ahead does not exist
    yet. Their null indicator separates the windows *perfectly*, so leaving them in
    pins the adversarial AUC at 1.0 no matter how stable the population is —
    exactly the `month_index` failure mode under a different name. `loss_severity_band`
    is the same story via `banned_feature_columns`. Drift here answers "does the
    model need retraining"; a column the model never sees cannot answer it.
    """
    s = settings or get_settings()
    cols: set[str] = set()
    for key in ("data.target_columns", "data.banned_feature_columns"):
        for c in s.get(key, []) or []:
            cols.add(str(c))
    return frozenset(cols)


def seasoning_features(settings: Settings | None = None) -> frozenset[str]:
    s = settings or get_settings()
    configured = s.get("drift.seasoning_features", None)
    if configured is None:
        return DEFAULT_SEASONING_FEATURES
    return frozenset(str(c) for c in configured)


def _drift_columns(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str] | None,
    settings: Settings | None = None,
) -> list[str]:
    cols = columns or sorted(set(reference.columns) & set(current.columns))
    dropped = DRIFT_EXCLUDED_COLUMNS | non_model_columns(settings)
    return [c for c in cols if not c.startswith("_") and c not in dropped]


def psi(
    reference: pd.Series, current: pd.Series, *, bins: int = 10, categorical: bool | None = None
) -> tuple[float, list[dict[str, Any]]]:
    """Population Stability Index with quantile bins taken from the reference.

    Returns (psi, per-bin detail). Bins are derived from the reference window
    only — deriving them jointly would hide exactly the shift we are measuring.
    """
    ref = reference.dropna()
    cur = current.dropna()
    if ref.empty or cur.empty:
        return float("nan"), []

    is_cat = categorical if categorical is not None else not pd.api.types.is_numeric_dtype(ref)

    if is_cat:
        levels = sorted(set(ref.astype(str).unique()) | set(cur.astype(str).unique()))
        ref_p = ref.astype(str).value_counts(normalize=True).reindex(levels).fillna(0.0)
        cur_p = cur.astype(str).value_counts(normalize=True).reindex(levels).fillna(0.0)
        labels = levels
    else:
        ref_num = pd.to_numeric(ref, errors="coerce").dropna()
        cur_num = pd.to_numeric(cur, errors="coerce").dropna()
        if ref_num.empty or cur_num.empty:
            return float("nan"), []
        quantiles = np.unique(np.quantile(ref_num, np.linspace(0, 1, bins + 1)))
        if len(quantiles) < 3:
            return 0.0, []
        edges = quantiles.copy()
        edges[0], edges[-1] = -np.inf, np.inf
        ref_counts = np.histogram(ref_num, bins=edges)[0].astype("float64")
        cur_counts = np.histogram(cur_num, bins=edges)[0].astype("float64")
        ref_p = pd.Series(ref_counts / max(ref_counts.sum(), 1))
        cur_p = pd.Series(cur_counts / max(cur_counts.sum(), 1))
        labels = [f"[{quantiles[i]:.4g}, {quantiles[i + 1]:.4g})" for i in range(len(quantiles) - 1)]

    r = np.clip(ref_p.to_numpy(dtype="float64"), EPS, None)
    c = np.clip(cur_p.to_numpy(dtype="float64"), EPS, None)
    contributions = (c - r) * np.log(c / r)
    detail = [
        {
            "bin": str(labels[i]),
            "ref_share": round(float(ref_p.iloc[i]), 6),
            "cur_share": round(float(cur_p.iloc[i]), 6),
            "contribution": round(float(contributions[i]), 6),
        }
        for i in range(len(labels))
    ]
    return float(contributions.sum()), detail


def ks_statistic(reference: pd.Series, current: pd.Series) -> tuple[float | None, float | None]:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if len(ref) < 10 or len(cur) < 10:
        return None, None
    stat, pvalue = stats.ks_2samp(ref, cur)
    return float(stat), float(pvalue)


def jensen_shannon(
    reference: pd.Series, current: pd.Series, *, bins: int = 20
) -> float | None:
    """JS divergence in bits. Symmetric, bounded [0, 1] — safe for categoricals."""
    ref = reference.dropna()
    cur = current.dropna()
    if ref.empty or cur.empty:
        return None

    if pd.api.types.is_numeric_dtype(ref):
        ref_num = pd.to_numeric(ref, errors="coerce").dropna()
        cur_num = pd.to_numeric(cur, errors="coerce").dropna()
        lo = float(min(ref_num.min(), cur_num.min()))
        hi = float(max(ref_num.max(), cur_num.max()))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return 0.0
        edges = np.linspace(lo, hi, bins + 1)
        p = np.histogram(ref_num, bins=edges)[0].astype("float64")
        q = np.histogram(cur_num, bins=edges)[0].astype("float64")
    else:
        levels = sorted(set(ref.astype(str).unique()) | set(cur.astype(str).unique()))
        p = ref.astype(str).value_counts().reindex(levels).fillna(0.0).to_numpy(dtype="float64")
        q = cur.astype(str).value_counts().reindex(levels).fillna(0.0).to_numpy(dtype="float64")

    p = p / max(p.sum(), 1.0)
    q = q / max(q.sum(), 1.0)
    return float(stats.entropy(p + EPS, q + EPS, base=2) * 0 + _js(p, q))


def _js(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    p, q, m = p + EPS, q + EPS, m + EPS
    kl_pm = float((p * np.log2(p / m)).sum())
    kl_qm = float((q * np.log2(q / m)).sum())
    return max(0.0, 0.5 * (kl_pm + kl_qm))


def verdict_for_psi(value: float | None, keep: float, monitor: float) -> str:
    if value is None or not np.isfinite(value):
        return "UNKNOWN"
    if value < keep:
        return "KEEP"
    if value <= monitor:
        return "MONITOR"
    return "DROP_OR_ROBUSTIFY"


def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    s = settings or get_settings()
    bins = int(s.get("drift.psi_bins", 10))
    keep = float(s.get("drift.thresholds.keep", 0.10))
    monitor = float(s.get("drift.thresholds.monitor", 0.25))
    seasoning = seasoning_features(s)

    cols = _drift_columns(reference, current, columns, s)

    rows: list[dict[str, Any]] = []
    for c in cols:
        ref_col, cur_col = reference[c], current[c]
        is_num = pd.api.types.is_numeric_dtype(ref_col) and pd.api.types.is_numeric_dtype(cur_col)
        value_psi, _ = psi(ref_col, cur_col, bins=bins, categorical=not is_num)
        ks, ks_p = ks_statistic(ref_col, cur_col) if is_num else (None, None)
        js = jensen_shannon(ref_col, cur_col)
        ref_null = float(ref_col.isna().mean())
        cur_null = float(cur_col.isna().mean())
        missing_delta = cur_null - ref_null

        # Missingness drift alone can condemn a feature even when its observed
        # values are perfectly stable — an is_null indicator built on it would be
        # the most drifted signal in the model.
        value_verdict = verdict_for_psi(value_psi, keep, monitor)
        missing_verdict = (
            "KEEP" if abs(missing_delta) < 0.01
            else "MONITOR" if abs(missing_delta) < 0.03
            else "DROP_OR_ROBUSTIFY"
        )
        order = {"KEEP": 0, "MONITOR": 1, "DROP_OR_ROBUSTIFY": 2, "UNKNOWN": 0}
        verdict = value_verdict if order[value_verdict] >= order[missing_verdict] else missing_verdict

        rows.append(
            {
                "feature": c,
                "kind": "numeric" if is_num else "categorical",
                "psi": None if not np.isfinite(value_psi) else round(float(value_psi), 6),
                "ks_stat": None if ks is None else round(ks, 6),
                "ks_pvalue": None if ks_p is None else round(ks_p, 8),
                "js_div": None if js is None else round(js, 6),
                "ref_null_rate": round(ref_null, 6),
                "cur_null_rate": round(cur_null, 6),
                "missing_delta": round(missing_delta, 6),
                "value_verdict": value_verdict,
                "missingness_verdict": missing_verdict,
                "verdict": verdict,
                "driver": "missingness" if missing_verdict != "KEEP" and order[missing_verdict] >= order[value_verdict] else "values",
                # Reported, but never allowed to fire the retrain trigger: this
                # column shifts by construction when the windows are apart in time.
                "seasoning": c in seasoning,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["verdict", "psi"], ascending=[False, False], kind="mergesort"
    ).reset_index(drop=True)


def adversarial_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    seed: int = 0,
    max_rows: int = 40_000,
    exclude: frozenset[str] | set[str] | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Train a classifier to tell reference from current.

    Its AUC is a single scalar multivariate drift score; its top permutation
    features name *which* columns drive the shift. AUC ~ 0.5 means the two
    windows are indistinguishable.

    `exclude` drops deterministic seasoning columns before fitting. Left in,
    `loan_age_months` is a near-perfect separator of any two windows that are
    apart in time, and the AUC measures the calendar rather than the population.
    """
    excluded_seasoning = set(exclude or ())
    cols = [
        c for c in _drift_columns(reference, current, columns, settings)
        if c not in excluded_seasoning
    ]
    if not cols:
        return {"available": False, "reason": "no shared comparable columns"}

    ref = reference[cols].sample(min(len(reference), max_rows), random_state=seed)
    cur = current[cols].sample(min(len(current), max_rows), random_state=seed)
    X_df = pd.concat([ref, cur], ignore_index=True)
    y = np.r_[np.zeros(len(ref)), np.ones(len(cur))]

    # Null indicators are added explicitly: the missingness shift is the finding,
    # and a value-only encoding would hide it from the adversary.
    parts, names = [], []
    for c in cols:
        col = X_df[c]
        if pd.api.types.is_numeric_dtype(col):
            parts.append(pd.to_numeric(col, errors="coerce").to_numpy(dtype="float64"))
        elif pd.api.types.is_datetime64_any_dtype(col):
            parts.append(col.astype("int64").where(col.notna(), np.nan).to_numpy(dtype="float64"))
        else:
            codes = pd.Categorical(col.astype("object")).codes.astype("float64")
            codes[codes < 0] = np.nan
            parts.append(codes)
        names.append(c)
        if col.isna().any():
            parts.append(col.isna().to_numpy(dtype="float64"))
            names.append(f"{c}__isnull")

    X = np.column_stack(parts)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    model = HistGradientBoostingClassifier(
        max_iter=120, max_depth=5, learning_rate=0.1, random_state=seed
    )
    model.fit(X_tr, y_tr)
    auc = float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))

    try:
        imp = permutation_importance(
            model, X_te, y_te, n_repeats=3, random_state=seed, scoring="roc_auc", n_jobs=1
        )
        order = np.argsort(-imp.importances_mean)[:12]
        drivers = [
            {"feature": names[i], "importance": round(float(imp.importances_mean[i]), 6)}
            for i in order
            if imp.importances_mean[i] > 0
        ]
    except Exception:  # pragma: no cover
        drivers = []

    return {
        "available": True,
        "adversarial_auc": round(auc, 6),
        "n_reference": int(len(ref)),
        "n_current": int(len(cur)),
        "excluded_columns": sorted(
            (DRIFT_EXCLUDED_COLUMNS | non_model_columns(settings))
            & (set(reference.columns) | set(current.columns))
        ),
        "excluded_seasoning_columns": sorted(
            excluded_seasoning & (set(reference.columns) | set(current.columns))
        ),
        "interpretation": (
            "indistinguishable" if auc < 0.60
            else "mild multivariate shift" if auc < 0.75
            else "strong multivariate shift"
        ),
        "top_drivers": drivers,
    }


def prediction_drift(reference_scores: pd.Series, current_scores: pd.Series, bins: int = 10) -> dict[str, Any]:
    """PSI of the score distribution — the metric a production monitor alerts on
    when labels have not matured yet."""
    value, detail = psi(reference_scores, current_scores, bins=bins, categorical=False)
    return {
        "psi": None if not np.isfinite(value) else round(float(value), 6),
        "verdict": verdict_for_psi(value, 0.10, 0.25),
        "bins": detail,
    }


def retraining_trigger(
    drift_table: pd.DataFrame, adversarial: dict[str, Any], settings: Settings | None = None
) -> dict[str, Any]:
    """PSI > 0.25 on >= 3 features OR adversarial AUC > 0.80 -> retrain.

    Seasoning features are counted separately and never fire the trigger. On a
    healthy amortizing panel `loan_age_months` and `remaining_term_months` blow
    past any PSI threshold whenever the two windows are apart in time, so
    including them makes the trigger fire on every window pair forever.
    """
    s = settings or get_settings()
    psi_threshold = float(s.get("drift.retrain_trigger.psi_threshold", 0.25))
    min_features = int(s.get("drift.retrain_trigger.min_features_over_threshold", 3))
    auc_threshold = float(s.get("drift.retrain_trigger.adversarial_auc_threshold", 0.80))
    seasoning = seasoning_features(s)

    if drift_table.empty:
        over_all: list[str] = []
    else:
        over_all = drift_table.loc[
            drift_table["psi"].notna() & (drift_table["psi"] > psi_threshold), "feature"
        ].tolist()
    over = [f for f in over_all if f not in seasoning]
    over_seasoning = [f for f in over_all if f in seasoning]

    auc = adversarial.get("adversarial_auc")
    psi_fired = len(over) >= min_features
    auc_fired = auc is not None and auc > auc_threshold

    reasons = []
    if psi_fired:
        reasons.append(f"PSI > {psi_threshold} on {len(over)} features (threshold {min_features})")
    if auc_fired:
        reasons.append(f"adversarial AUC {auc:.4f} > {auc_threshold}")

    notes = []
    if over_seasoning:
        notes.append(
            f"{len(over_seasoning)} seasoning feature(s) exceeded PSI {psi_threshold} "
            f"({', '.join(over_seasoning)}) and were excluded — they shift by "
            "construction when the windows are apart in time."
        )

    return {
        "retrain_required": bool(psi_fired or auc_fired),
        "reasons": reasons,
        "notes": notes,
        "features_over_psi_threshold": over,
        "n_features_over_threshold": len(over),
        "seasoning_features_over_psi_threshold": over_seasoning,
        "adversarial_auc": auc,
        "thresholds": {
            "psi": psi_threshold,
            "min_features": min_features,
            "adversarial_auc": auc_threshold,
        },
    }


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    ref_window: str,
    cur_window: str,
    columns: list[str] | None = None,
    include_adversarial: bool = True,
    settings: Settings | None = None,
) -> dict[str, Any]:
    s = settings or get_settings()
    seasoning = seasoning_features(s)
    table = feature_drift(reference, current, columns=columns, settings=s)
    adversarial = (
        adversarial_drift(
            reference, current, columns=columns, seed=s.seed,
            exclude=seasoning, settings=s,
        )
        if include_adversarial
        else {"available": False, "reason": "disabled for this request"}
    )
    trigger = retraining_trigger(table, adversarial, settings=s)

    counts = table["verdict"].value_counts().to_dict() if not table.empty else {}
    max_psi = float(table["psi"].max()) if not table.empty and table["psi"].notna().any() else None

    # The batch verdict reads the actionable max only. `max_psi` stays the true
    # maximum across every feature so the seasoning shift is still visible.
    if table.empty:
        actionable = table
    else:
        actionable = table[~table["seasoning"] & table["psi"].notna()]
    max_psi_actionable = (
        float(actionable["psi"].max()) if not actionable.empty else None
    )
    batch_verdict = (
        "FAIL" if trigger["retrain_required"]
        else "WARN" if (
            max_psi_actionable is not None
            and max_psi_actionable > float(s.get("drift.thresholds.keep", 0.10))
        )
        else "PASS"
    )

    return {
        "ref_window": ref_window,
        "cur_window": cur_window,
        "n_reference_rows": int(len(reference)),
        "n_current_rows": int(len(current)),
        "computed_at": utcnow_iso(),
        "features": table.to_dict(orient="records"),
        "verdict_counts": {str(k): int(v) for k, v in counts.items()},
        "max_psi": None if max_psi is None else round(max_psi, 6),
        "max_psi_actionable": None if max_psi_actionable is None else round(max_psi_actionable, 6),
        "seasoning_features": sorted(seasoning & set(table["feature"]) if not table.empty else seasoning),
        "missingness_drift_leaders": (
            table.reindex(table["missing_delta"].abs().sort_values(ascending=False).index)
            .head(8)[["feature", "ref_null_rate", "cur_null_rate", "missing_delta", "missingness_verdict"]]
            .to_dict(orient="records")
            if not table.empty
            else []
        ),
        "adversarial": adversarial,
        "retraining_trigger": trigger,
        "batch_verdict": batch_verdict,
    }
