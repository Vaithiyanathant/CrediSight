"""Counterfactual explanations over *actionable* features only.

"If days_past_due returned to 0 and the modification flag were set, P(default)
would fall from 0.34 to 0.11" is a sentence a servicer can act on. "If the loan
were in a different state" is not — it is also the kind of counterfactual that
edges toward disparate treatment, so geography, credit band and every static
borrower attribute are excluded from the search space by construction rather
than by convention.

Search is a bounded random/grid hybrid over the declared actionable set. DiCE is
supported when installed; the in-house search is the default because it is
deterministic, has no extra dependency, and respects the actionability
constraint exactly.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from lpie.core.config import Settings, get_settings
from lpie.core.exceptions import InvalidRequestError
from lpie.core.logging import get_logger

log = get_logger(__name__)

# Never searchable. Changing these would produce a "counterfactual" that is
# either meaningless or a disparate-treatment suggestion.
FORBIDDEN_FEATURES = frozenset({
    "state", "credit_score_band", "credit_score_band_ord", "vintage_year_num",
    "loan_purpose", "occupancy_type", "property_type", "servicer_name",
    "original_balance", "log_original_balance", "loan_term_months",
    "origination_month_num", "month_index", "months_since_panel_start",
})


def actionable_features(settings: Settings | None = None) -> list[str]:
    s = settings or get_settings()
    declared = list(s.get("explain.counterfactual.actionable_features", []) or [])
    return [f for f in declared if f not in FORBIDDEN_FEATURES]


def _candidate_values(X: pd.DataFrame, feature: str, *, n: int = 5) -> list[Any]:
    values = pd.to_numeric(X[feature], errors="coerce").dropna()
    if values.empty:
        return []
    if values.nunique() <= n:
        return sorted(values.unique().tolist())
    quantiles = np.quantile(values, np.linspace(0.05, 0.95, n))
    return sorted({round(float(q), 6) for q in quantiles} | {0.0})


def search(
    predict_fn,
    row: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    features: list[str] | None = None,
    target_probability: float | None = None,
    max_changes: int = 3,
    max_candidates: int = 4000,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Find the minimal actionable change that most reduces the predicted risk.

    `predict_fn(DataFrame) -> ndarray` must be the same calibrated scoring path
    the API uses, so the counterfactual describes the deployed model rather than
    a base learner.
    """
    s = settings or get_settings()
    if len(row) != 1:
        raise InvalidRequestError("Counterfactual search expects exactly one row")

    candidates_features = [
        f for f in (features or actionable_features(s))
        if f in row.columns and f not in FORBIDDEN_FEATURES
    ]
    if not candidates_features:
        return {
            "found": False,
            "reason": "No actionable features are available for this model configuration.",
            "actionable_features": [],
        }

    original = float(predict_fn(row)[0])
    target = target_probability if target_probability is not None else original * 0.5

    grids = {f: _candidate_values(reference, f) for f in candidates_features}
    grids = {f: v for f, v in grids.items() if v}

    evaluated = 0
    results: list[dict[str, Any]] = []

    for size in range(1, max_changes + 1):
        for combo in combinations(grids, size):
            variants: list[pd.DataFrame] = []
            descriptions: list[list[dict[str, Any]]] = []
            for values in _product(*[grids[f] for f in combo]):
                if evaluated >= max_candidates:
                    break
                variant = row.copy()
                changes = []
                for feature, value in zip(combo, values, strict=False):
                    current = row.iloc[0].get(feature)
                    if pd.notna(current) and np.isclose(
                        pd.to_numeric(pd.Series([current]), errors="coerce").iloc[0], value
                    ):
                        continue
                    variant[feature] = value
                    changes.append(
                        {
                            "feature": feature,
                            "from": _display(current),
                            "to": _display(value),
                        }
                    )
                if not changes:
                    continue
                variants.append(variant)
                descriptions.append(changes)
                evaluated += 1
            if not variants:
                continue

            batch = pd.concat(variants, ignore_index=True)
            probabilities = predict_fn(batch)
            for changes, probability in zip(descriptions, probabilities, strict=False):
                results.append(
                    {
                        "changes": changes,
                        "n_changes": len(changes),
                        "probability": round(float(probability), 6),
                        "delta": round(float(probability - original), 6),
                    }
                )
            if evaluated >= max_candidates:
                break
        # A one-feature fix that already reaches the target is preferable to a
        # three-feature one, so stop widening as soon as the target is met.
        if any(r["probability"] <= target for r in results):
            break

    if not results:
        return {
            "found": False,
            "original_probability": round(original, 6),
            "reason": "No actionable variation changed the prediction.",
            "actionable_features": candidates_features,
            "n_evaluated": evaluated,
        }

    achieving = [r for r in results if r["probability"] <= target]
    pool = achieving or results
    pool.sort(key=lambda r: (r["n_changes"], r["probability"]))
    best = pool[0]

    return {
        "found": bool(achieving),
        "original_probability": round(original, 6),
        "target_probability": round(float(target), 6),
        "counterfactual": best,
        "alternatives": pool[1:6],
        "actionable_features": candidates_features,
        "forbidden_features": sorted(FORBIDDEN_FEATURES),
        "n_evaluated": evaluated,
        "narrative": _narrative(best, original),
        "governance": (
            "Counterfactuals are restricted to operationally actionable fields. Static "
            "borrower and geographic attributes are excluded by construction: varying them "
            "would describe a different borrower, not a different outcome for this one."
        ),
    }


def _product(*iterables):
    from itertools import product

    yield from product(*iterables)


def _display(value: Any) -> Any:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return None if not np.isfinite(v) else round(v, 6)


def _narrative(best: dict[str, Any], original: float) -> str:
    changes = ", ".join(
        f"{c['feature']} moved from {c['from']} to {c['to']}" for c in best["changes"]
    )
    return (
        f"If {changes}, the model's probability would move from {original:.3f} to "
        f"{best['probability']:.3f}. This describes the model's response surface, not a "
        f"causal claim about the borrower."
    )
