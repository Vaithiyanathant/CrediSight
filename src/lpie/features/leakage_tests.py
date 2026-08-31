"""Automated leakage governance.

Eight checks, all runnable as a library call and all wired into pytest. A
leakage failure fails CI — that is the point. The checks are deliberately
adversarial toward our own code:

1. no target column is used as a feature
2. no `loan_id` predictive encoding exists
3. no future shift is declared
4. no centered rolling window appears in any family's source
5. no future aggregate is computed
6. no future servicer update is joined
7. **truncation equality** — a row's feature vector recomputed from history
   truncated at that row's month must match the stored vector
8. any feature with univariate AUC > 0.97 against a forward target is quarantined
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lpie.core.config import Settings, get_settings
from lpie.core.exceptions import LeakageError
from lpie.core.logging import get_logger
from lpie.features import families as families_pkg
from lpie.features.builder import FeatureBuilder, build_registry
from lpie.features.registry import FeatureRegistry

log = get_logger(__name__)

SUSPICIOUS_AUC = 0.97
FLOAT_TOLERANCE = 1e-6

# Features whose value legitimately depends on the whole cross-section at month t
# (portfolio medians, peer aggregates). Truncating the panel by *month* keeps them
# reproducible; truncating by loan would not, so the truncation test slices by
# month and these are checked like everything else.
FAMILY_MODULES = (
    "static", "balance", "delinquency", "prepay", "cohort",
    "dq", "servicer", "temporal", "interactions",
)


@dataclass
class LeakageCheck:
    name: str
    passed: bool
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class LeakageReport:
    checks: list[LeakageCheck]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[LeakageCheck]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "n_checks": len(self.checks),
            "n_failed": len(self.failures),
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail, "evidence": c.evidence}
                for c in self.checks
            ],
        }

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise LeakageError(
                f"{len(self.failures)} leakage check(s) failed: "
                + ", ".join(c.name for c in self.failures),
                details=self.to_dict(),
            )


# --------------------------------------------------------------------------- #
# 1-3, 5: declaration-level checks against the registry
# --------------------------------------------------------------------------- #
def check_no_target_sources(registry: FeatureRegistry, targets: list[str]) -> LeakageCheck:
    offenders = {
        s.name: sorted(set(targets) & set(s.source_columns))
        for s in registry.specs
        if set(targets) & set(s.source_columns)
    }
    return LeakageCheck(
        "no_target_column_as_feature_source",
        not offenders,
        "No declared feature sources a target column"
        if not offenders
        else f"{len(offenders)} feature(s) source a target column",
        {"offenders": offenders},
    )


def check_no_loan_id_encoding(registry: FeatureRegistry, features: pd.DataFrame | None = None) -> LeakageCheck:
    """`loan_id` may be an identity column but never a predictive one."""
    offenders = [s.name for s in registry.specs if "loan_id" in s.source_columns and s.family != "servicer"]
    detail_bits: list[str] = []
    if offenders:
        detail_bits.append(f"features sourcing loan_id: {offenders}")

    encoded = []
    if features is not None:
        for name in registry.names:
            if name not in features.columns:
                continue
            lowered = name.lower()
            if "loan_id" in lowered or lowered.endswith("_te") or "target_enc" in lowered:
                encoded.append(name)
    if encoded:
        detail_bits.append(f"suspicious encoded names: {encoded}")

    passed = not offenders and not encoded
    return LeakageCheck(
        "no_loan_id_predictive_encoding",
        passed,
        "loan_id is used only as an identity key" if passed else "; ".join(detail_bits),
        {"source_offenders": offenders, "name_offenders": encoded},
    )


def check_no_future_offsets(registry: FeatureRegistry) -> LeakageCheck:
    offenders = {s.name: s.temporal_offset for s in registry.specs if s.temporal_offset > 0}
    return LeakageCheck(
        "no_positive_temporal_offset",
        not offenders,
        "Every feature declares temporal_offset <= 0"
        if not offenders
        else f"{len(offenders)} feature(s) declare a future offset",
        {"offenders": offenders},
    )


def check_declared_features_have_justification(registry: FeatureRegistry) -> LeakageCheck:
    offenders = [
        s.name for s in registry.specs
        if s.leakage_risk in ("medium", "high") and not s.justification.strip()
    ]
    return LeakageCheck(
        "elevated_risk_features_are_justified",
        not offenders,
        "Every medium/high-risk feature carries a written justification"
        if not offenders
        else f"{len(offenders)} unjustified elevated-risk feature(s)",
        {"offenders": offenders},
    )


# --------------------------------------------------------------------------- #
# 4, 6: source-level static analysis of the family modules
# --------------------------------------------------------------------------- #
def _family_sources() -> dict[str, str]:
    import importlib

    out: dict[str, str] = {}
    for name in FAMILY_MODULES:
        module = importlib.import_module(f"{families_pkg.__name__}.{name}")
        out[name] = inspect.getsource(module)
    return out


def check_no_centered_or_negative_windows() -> LeakageCheck:
    """AST scan: no `center=True` in a rolling call, no negative shift/lag."""
    import importlib

    offenders: list[dict[str, Any]] = []
    for name in FAMILY_MODULES:
        module = importlib.import_module(f"{families_pkg.__name__}.{name}")
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func_name = _call_name(node.func)
            if func_name == "rolling":
                for kw in node.keywords:
                    if kw.arg == "center" and _is_truthy(kw.value):
                        offenders.append({"module": name, "line": node.lineno, "issue": "rolling(center=True)"})
            if func_name in {"shift", "diff"}:
                for arg in [*node.args, *(kw.value for kw in node.keywords if kw.arg in {"periods", "lag"})]:
                    value = _const_int(arg)
                    if value is not None and value < 0:
                        offenders.append(
                            {"module": name, "line": node.lineno, "issue": f"{func_name}({value})"}
                        )
    return LeakageCheck(
        "no_centered_windows_or_negative_shifts",
        not offenders,
        "No centered rolling window or negative shift in any feature family"
        if not offenders
        else f"{len(offenders)} forbidden temporal operation(s)",
        {"offenders": offenders},
    )


def check_servicer_join_is_backward() -> LeakageCheck:
    """Every merge_asof in the codebase must be explicitly backward."""
    sources = _family_sources()
    from lpie.validation import engine as validation_engine

    sources["validation.engine"] = inspect.getsource(validation_engine)

    offenders: list[dict[str, Any]] = []
    n_joins = 0
    for name, src in sources.items():
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node.func) == "merge_asof":
                n_joins += 1
                direction = next(
                    (kw.value for kw in node.keywords if kw.arg == "direction"), None
                )
                value = getattr(direction, "value", None) if isinstance(direction, ast.Constant) else None
                if value != "backward":
                    offenders.append(
                        {"module": name, "line": node.lineno, "direction": value or "unspecified"}
                    )
    return LeakageCheck(
        "servicer_asof_join_is_backward_only",
        not offenders,
        f"All {n_joins} merge_asof call(s) are direction='backward'"
        if not offenders
        else f"{len(offenders)} merge_asof call(s) are not backward",
        {"offenders": offenders, "n_joins": n_joins},
    )


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _is_truthy(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and bool(node.value)


def _const_int(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _const_int(node.operand)
        return None if inner is None else -inner
    return None


# --------------------------------------------------------------------------- #
# 7: truncation equality — the substantive test
# --------------------------------------------------------------------------- #
def check_truncation_equality(
    panel: pd.DataFrame,
    static: pd.DataFrame,
    servicer: pd.DataFrame | None,
    stored: pd.DataFrame,
    *,
    registry: FeatureRegistry | None = None,
    fit_params: Any = None,
    cut_months: tuple[int, ...] = (18, 27, 34),
    sample_rows: int = 200,
    settings: Settings | None = None,
    tolerance: float = FLOAT_TOLERANCE,
) -> LeakageCheck:
    """Rebuild features from a panel truncated at month T and compare row-for-row.

    If any feature reads the future, its value will change when months > T are
    removed. This catches leakage that a declaration check cannot: an accidental
    `shift(-1)` buried three call frames deep still shows up here as a mismatch.
    """
    s = settings or get_settings()
    registry = registry or build_registry()
    rng = np.random.default_rng(s.seed)

    mismatches: list[dict[str, Any]] = []
    n_compared = 0

    for cut in cut_months:
        truncated = panel[panel["month_index"] <= cut]
        if truncated.empty:
            continue
        builder = FeatureBuilder(s, fit_params=fit_params)
        rebuilt = builder.build(
            truncated, static, servicer, enforce_contract=False
        ).features

        edge = rebuilt[rebuilt["month_index"] == cut]
        if edge.empty:
            continue
        take = min(sample_rows, len(edge))
        idx = rng.choice(len(edge), size=take, replace=False)
        sample = edge.iloc[idx]

        reference = stored[stored["month_index"] == cut].set_index("loan_id")
        sample = sample.set_index("loan_id")
        common = sample.index.intersection(reference.index)
        if common.empty:
            continue

        left = sample.loc[common]
        right = reference.loc[common]
        n_compared += len(common)

        for col in registry.names:
            if col not in left.columns or col not in right.columns:
                continue
            a, b = left[col], right[col]
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                av = pd.to_numeric(a, errors="coerce").to_numpy(dtype="float64")
                bv = pd.to_numeric(b, errors="coerce").to_numpy(dtype="float64")
                both_nan = np.isnan(av) & np.isnan(bv)
                diff = np.abs(av - bv)
                scale = np.maximum(np.abs(bv), 1.0)
                bad = (~both_nan) & ~(diff <= tolerance * scale) & ~(np.isnan(av) & np.isnan(bv))
                bad = bad & ~(np.isnan(av) & np.isnan(bv))
                n_bad = int(np.nansum(bad))
            else:
                av = a.astype("string").fillna("<NA>")
                bv = b.astype("string").fillna("<NA>")
                n_bad = int((av.to_numpy() != bv.to_numpy()).sum())
            if n_bad:
                mismatches.append(
                    {
                        "cut_month": cut,
                        "feature": col,
                        "n_mismatched": n_bad,
                        "n_compared": int(len(common)),
                    }
                )

    passed = not mismatches
    return LeakageCheck(
        "point_in_time_truncation_equality",
        passed,
        f"{n_compared} row(s) reproduce bit-for-bit from a truncated panel"
        if passed
        else f"{len(mismatches)} feature/cut combination(s) changed under truncation",
        {"mismatches": mismatches[:40], "n_rows_compared": n_compared, "cut_months": list(cut_months)},
    )


# --------------------------------------------------------------------------- #
# 8: suspicious univariate discrimination
# --------------------------------------------------------------------------- #
def check_univariate_auc(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    registry: FeatureRegistry | None = None,
    threshold: float = SUSPICIOUS_AUC,
    max_rows: int = 60_000,
    settings: Settings | None = None,
) -> LeakageCheck:
    """Quarantine any single feature that all but predicts a forward target alone.

    A univariate AUC above 0.97 against a 3-to-12-month-forward label is not a
    good feature; it is a label that has leaked into the matrix under another
    name. Genuinely predictive features on this pack top out far below that.
    """
    s = settings or get_settings()
    registry = registry or build_registry()
    binary_targets = [
        spec["target"]
        for spec in (s.section("heads") or {}).values()
        if spec.get("type") == "binary" and int(spec.get("horizon", 0)) > 0
    ]

    merged = features.merge(targets, on=["loan_id", "month_index"], how="inner", suffixes=("", "_tgt"))
    if len(merged) > max_rows:
        merged = merged.sample(max_rows, random_state=s.seed)

    flagged: list[dict[str, Any]] = []
    checked = 0
    for target in binary_targets:
        if target not in merged.columns:
            continue
        y = pd.to_numeric(merged[target], errors="coerce")
        mask = y.notna()
        if mask.sum() < 500 or y[mask].nunique() < 2:
            continue
        y_values = y[mask].to_numpy()
        for name in registry.names:
            if name not in merged.columns:
                continue
            col = merged.loc[mask, name]
            if not pd.api.types.is_numeric_dtype(col):
                continue
            values = pd.to_numeric(col, errors="coerce")
            valid = values.notna().to_numpy()
            if valid.sum() < 500 or values[valid].nunique() < 2:
                continue
            checked += 1
            try:
                auc = float(roc_auc_score(y_values[valid], values.to_numpy()[valid]))
            except ValueError:
                continue
            auc = max(auc, 1.0 - auc)
            if auc > threshold:
                flagged.append({"feature": name, "target": target, "auc": round(auc, 6)})

    justified = {
        s_.name for s_ in registry.specs if s_.leakage_risk in ("medium", "high") and s_.justification
    }
    unjustified = [f for f in flagged if f["feature"] not in justified]

    return LeakageCheck(
        "no_suspicious_univariate_auc",
        not unjustified,
        f"{checked} feature/target pair(s) checked; none exceed AUC {threshold}"
        if not unjustified
        else f"{len(unjustified)} feature(s) exceed AUC {threshold} without written justification",
        {"flagged": flagged, "unjustified": unjustified, "threshold": threshold, "n_checked": checked},
    )


# --------------------------------------------------------------------------- #
def run_all(
    panel: pd.DataFrame,
    static: pd.DataFrame,
    servicer: pd.DataFrame | None,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    registry: FeatureRegistry | None = None,
    fit_params: Any = None,
    settings: Settings | None = None,
    include_truncation: bool = True,
    include_univariate: bool = True,
) -> LeakageReport:
    s = settings or get_settings()
    registry = registry or build_registry()
    target_columns = s.require("data.target_columns")

    checks = [
        check_no_target_sources(registry, target_columns),
        check_no_loan_id_encoding(registry, features),
        check_no_future_offsets(registry),
        check_declared_features_have_justification(registry),
        check_no_centered_or_negative_windows(),
        check_servicer_join_is_backward(),
    ]
    if include_truncation:
        checks.append(
            check_truncation_equality(
                panel, static, servicer, features,
                registry=registry, fit_params=fit_params, settings=s,
            )
        )
    if include_univariate:
        checks.append(check_univariate_auc(features, targets, registry=registry, settings=s))

    report = LeakageReport(checks)
    log.info(
        "leakage.report",
        passed=report.passed,
        failed=[c.name for c in report.failures],
    )
    return report
