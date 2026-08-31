"""Feature pipeline orchestration.

Builds all nine families over the combined train+test panel, enforces the
feature contract, and writes a Parquet store partitioned by `month_index`.

Partitioning by the time key is not cosmetic: it makes "train <= T" a partition
prune, so an accidental future read is a schema-visible bug rather than a silent
one. The storage layout enforces time-awareness that a comment could only
request.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lpie.core.config import Settings, get_settings
from lpie.core.determinism import sha256_obj
from lpie.core.exceptions import FeatureContractError
from lpie.core.logging import get_logger
from lpie.core.timing import Timer
from lpie.features.families import (
    balance as f_balance,
)
from lpie.features.families import (
    cohort as f_cohort,
)
from lpie.features.families import (
    delinquency as f_delinquency,
)
from lpie.features.families import (
    dq as f_dq,
)
from lpie.features.families import (
    interactions as f_interactions,
)
from lpie.features.families import (
    prepay as f_prepay,
)
from lpie.features.families import (
    servicer as f_servicer,
)
from lpie.features.families import (
    static as f_static,
)
from lpie.features.families import (
    temporal as f_temporal,
)
from lpie.features.registry import FeatureRegistry
from lpie.validation.engine import ValidationEngine
from lpie.validation.rules import CTX_DAYS_SINCE_UPDATE, CTX_MONTH_END

log = get_logger(__name__)

ID_COLUMNS = ("loan_id", "month_index", "reporting_month", "_split")
PASSTHROUGH_COLUMNS = ("current_status", "current_balance", "loan_age_months")


def build_registry() -> FeatureRegistry:
    """The single authoritative registry, assembled from the family declarations."""
    reg = FeatureRegistry()
    for module in (
        f_static, f_balance, f_delinquency, f_prepay, f_cohort,
        f_dq, f_servicer, f_temporal, f_interactions,
    ):
        reg.extend(module.SPECS)
    return reg


@dataclass
class FeatureFitParams:
    """Constants learned on the training window and then held fixed.

    Anything whose value depends on *which rows happen to be in the batch* has to
    live here, not be recomputed per call: winsorisation bounds, balance deciles,
    vintage rate statistics, and the categorical vocabulary. Recomputing them at
    scoring time would make a loan-month's features depend on its neighbours and
    would quietly read the future during backtesting.
    """

    balance: dict[str, Any] = field(default_factory=dict)
    prepay: dict[str, Any] = field(default_factory=dict)
    vocabulary: dict[str, list[str]] = field(default_factory=dict)
    fitted_on: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "balance": self.balance,
            "prepay": self.prepay,
            "vocabulary": self.vocabulary,
            "fitted_on": self.fitted_on,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeatureFitParams:
        return cls(
            balance=d.get("balance", {}),
            prepay=d.get("prepay", {}),
            vocabulary=d.get("vocabulary", {}),
            fitted_on=d.get("fitted_on", {}),
        )

    def hash(self) -> str:
        return sha256_obj(self.to_dict())


@dataclass
class FeatureBuildResult:
    features: pd.DataFrame
    registry: FeatureRegistry
    targets: pd.DataFrame
    feature_hash: str
    n_rows: int
    n_features: int
    elapsed_ms: float
    fit_params: FeatureFitParams | None = None
    stage_timings: dict[str, float] = field(default_factory=dict)
    contract_violations: list[str] = field(default_factory=list)


class FeatureBuilder:
    """Builds the full feature matrix from the raw panel.

    The builder is used identically offline (`make features`) and online (scoring
    a batch that arrives at the API). Same code path, same result — which is the
    only way the training and serving distributions can be guaranteed to match.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        engine: ValidationEngine | None = None,
        fit_params: FeatureFitParams | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = build_registry()
        self.engine = engine or ValidationEngine(self.settings)
        self.fit_params = fit_params
        if fit_params is not None and fit_params.vocabulary:
            self.engine._vocabulary = {k: set(v) for k, v in fit_params.vocabulary.items()}

    # ------------------------------------------------------------------ #
    def fit(
        self, panel: pd.DataFrame, static: pd.DataFrame, *, train_months: list[int] | None = None
    ) -> FeatureFitParams:
        """Learn the batch-independent constants from the training window only."""
        train = panel
        if train_months is not None:
            train = panel[panel["month_index"].isin(train_months)]
        elif "_split" in panel.columns:
            train = panel[panel["_split"] == "train"]
        if train.empty:
            train = panel

        static_features = f_static.build(train, static)
        balance_features = f_balance.build(train, static_features, fit={})

        bal_fit = {
            "pay_rate_bounds": f_balance.fit_winsor_bounds(balance_features["pay_rate"]),
            "smm_bounds": f_balance.fit_winsor_bounds(balance_features["smm"]),
        }
        # Loan-level constants are fitted on one row per loan so a loan that
        # appears 36 times does not outvote one that appears twice.
        per_loan = static_features.assign(_loan=train["loan_id"].to_numpy()).drop_duplicates("_loan")
        prepay_fit = f_prepay.fit_params(per_loan)

        vocab = self.engine.fit_vocabulary(train)
        params = FeatureFitParams(
            balance=bal_fit,
            prepay=prepay_fit,
            vocabulary={k: sorted(v) for k, v in vocab.items()},
            fitted_on={
                "n_rows": int(len(train)),
                "n_loans": int(train["loan_id"].nunique()),
                "months": [int(train["month_index"].min()), int(train["month_index"].max())],
            },
        )
        self.fit_params = params
        return params

    # ------------------------------------------------------------------ #
    def build(
        self,
        panel: pd.DataFrame,
        static: pd.DataFrame,
        servicer: pd.DataFrame | None = None,
        *,
        history: pd.DataFrame | None = None,
        fit_vocabulary_from: pd.DataFrame | None = None,
        enforce_contract: bool = True,
    ) -> FeatureBuildResult:
        timer = Timer()
        timings: dict[str, float] = {}
        panel = panel.sort_values(["loan_id", "month_index"], kind="mergesort").reset_index(drop=True)

        if self.fit_params is None:
            source = fit_vocabulary_from if fit_vocabulary_from is not None else panel
            self.fit(source, static)
        elif fit_vocabulary_from is not None:
            self.engine.fit_vocabulary(fit_vocabulary_from)

        fit = self.fit_params

        # --- validation first: DQ is an input to the model, not just a report --
        t = Timer()
        work, _ = self.engine.prepare_context(panel, static=static, servicer=servicer, history=history)
        rule_passes = pd.DataFrame(
            {rule.rule_id: rule.evaluate(work, self._ctx(static, servicer)) for rule in self.engine.rules},
            index=work.index,
        )
        record_scores = self.engine.scorer.score_records(work, rule_passes)
        timings["validation"] = t.stop()

        frames: dict[str, pd.DataFrame] = {}

        t = Timer()
        frames["static"] = f_static.build(panel, static)
        timings["static"] = t.stop()

        t = Timer()
        frames["balance"] = f_balance.build(panel, frames["static"], fit=fit.balance if fit else None)
        timings["balance"] = t.stop()

        t = Timer()
        frames["delinquency"] = f_delinquency.build(panel)
        timings["delinquency"] = t.stop()

        t = Timer()
        frames["prepay"] = f_prepay.build(panel, frames["static"], fit=fit.prepay if fit else None)
        timings["prepay"] = t.stop()

        t = Timer()
        frames["dq"] = f_dq.build(panel, record_scores, rule_passes, work[CTX_DAYS_SINCE_UPDATE])
        timings["dq"] = t.stop()

        t = Timer()
        frames["servicer"] = f_servicer.build(panel, servicer, frames["static"], work[CTX_MONTH_END])
        timings["servicer"] = t.stop()

        t = Timer()
        any_violation = (~rule_passes).any(axis=1).astype("float64")
        any_violation.index = panel.index
        frames["cohort"] = f_cohort.build(
            panel, frames["static"], frames["delinquency"], frames["balance"], dq_flags=any_violation
        )
        timings["cohort"] = t.stop()

        t = Timer()
        frames["temporal"] = f_temporal.build(panel)
        timings["temporal"] = t.stop()

        t = Timer()
        frames["interactions"] = f_interactions.build(
            frames["static"], frames["balance"], frames["delinquency"],
            frames["prepay"], frames["dq"], frames["servicer"],
        )
        timings["interactions"] = t.stop()

        features = self._assemble(panel, frames)
        violations = self._enforce_contract(features, enforce=enforce_contract)

        targets = self._extract_targets(panel)

        return FeatureBuildResult(
            features=features,
            registry=self.registry,
            targets=targets,
            feature_hash=self.registry.contract_hash(),
            n_rows=int(len(features)),
            n_features=len(self.registry),
            elapsed_ms=round(timer.stop(), 2),
            fit_params=self.fit_params,
            stage_timings={k: round(v, 2) for k, v in timings.items()},
            contract_violations=violations,
        )

    def _ctx(self, static: pd.DataFrame | None, servicer: pd.DataFrame | None):
        from lpie.validation.rules import RuleContext

        return RuleContext(
            vocabulary=self.engine.vocabulary,
            has_servicer=servicer is not None and not servicer.empty,
            has_static=static is not None and not static.empty,
            has_history=True,
            dq_config=self.settings.section("dq"),
        )

    def _assemble(self, panel: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        declared = set(self.registry.names)
        parts: list[pd.DataFrame] = []

        ids = pd.DataFrame(index=panel.index)
        for c in ID_COLUMNS:
            if c in panel.columns:
                ids[c] = panel[c].to_numpy()
        parts.append(ids)

        # `month_index` is both an identity column and a declared temporal
        # feature. It is emitted once, from the identity block, and marked as
        # already delivered so the family frame does not duplicate the column.
        seen: set[str] = {c for c in ids.columns if c in declared}
        for family_name, frame in frames.items():
            keep = [c for c in frame.columns if c in declared and c not in seen]
            missing = [c for c in frame.columns if c not in declared]
            if missing:
                raise FeatureContractError(
                    f"Family '{family_name}' produced undeclared column(s): {sorted(missing)}. "
                    "Every feature must be declared in the family's SPECS list.",
                    details={"family": family_name, "undeclared": sorted(missing)},
                )
            seen.update(keep)
            parts.append(frame[keep])

        undelivered = declared - seen
        if undelivered:
            raise FeatureContractError(
                f"Declared feature(s) not produced by any family: {sorted(undelivered)}",
                details={"missing": sorted(undelivered)},
            )

        out = pd.concat(parts, axis=1)
        for c in PASSTHROUGH_COLUMNS:
            if c in panel.columns and c not in out.columns:
                out[c] = panel[c].to_numpy()

        # Categorical features carry an explicit "Unknown" level rather than NaN:
        # missingness there is semantically meaningful and must survive to the
        # model as a level, not be silently imputed.
        for name in self.registry.categorical_features():
            if name in out.columns:
                out[name] = out[name].astype("object").where(out[name].notna(), "Unknown").astype(str)

        return out.replace([np.inf, -np.inf], np.nan)

    def _enforce_contract(self, features: pd.DataFrame, *, enforce: bool) -> list[str]:
        targets = self.settings.require("data.target_columns")
        banned = set(self.settings.require("data.banned_feature_columns"))
        violations = self.registry.validate_against_targets(targets)

        present_targets = sorted(set(features.columns) & set(targets))
        if present_targets:
            violations.append(f"target column(s) present in feature matrix: {present_targets}")

        feature_names = set(self.registry.names)
        present_banned = sorted(feature_names & banned)
        if present_banned:
            violations.append(f"banned column(s) declared as features: {present_banned}")

        if violations and enforce:
            raise FeatureContractError(
                f"{len(violations)} feature-contract violation(s)",
                details={"violations": violations},
            )
        return violations

    def _extract_targets(self, panel: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.settings.require("data.target_columns") if c in panel.columns]
        out = pd.DataFrame(index=panel.index)
        for c in ("loan_id", "month_index"):
            out[c] = panel[c].to_numpy()
        for c in cols:
            out[c] = panel[c].to_numpy()
        return out


# --------------------------------------------------------------------------- #
# Parquet feature store
# --------------------------------------------------------------------------- #
def write_feature_store(
    features: pd.DataFrame,
    targets: pd.DataFrame | None = None,
    *,
    settings: Settings | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Write `month_index=<m>/part.parquet` partitions.

    Targets travel with the features so a training run reads one partition set,
    but the *masking* that keeps censored labels out of the loss is applied by
    the splitter, never here — storage must not silently drop rows.
    """
    s = settings or get_settings()
    root: Path = s.path("feature_store_dir")
    if overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    frame = features
    if targets is not None and not targets.empty:
        target_cols = [c for c in targets.columns if c not in ("loan_id", "month_index")]
        frame = features.merge(
            targets[["loan_id", "month_index", *target_cols]],
            on=["loan_id", "month_index"],
            how="left",
            validate="one_to_one",
        )

    row_group = int(s.get("runtime.parquet_row_group_size", 65536))
    written: list[dict[str, Any]] = []
    for month, part in frame.groupby("month_index", sort=True):
        part_dir = root / f"month_index={int(month)}"
        part_dir.mkdir(parents=True, exist_ok=True)
        path = part_dir / "part.parquet"
        table = pa.Table.from_pandas(part.drop(columns=["month_index"]), preserve_index=False)
        pq.write_table(table, path, compression="zstd", row_group_size=row_group)
        written.append({"month_index": int(month), "rows": int(len(part)), "path": str(path)})

    manifest = {
        "root": str(root),
        "n_partitions": len(written),
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "partitions": written,
        "schema_hash": sha256_obj({c: str(frame[c].dtype) for c in sorted(frame.columns)}),
    }
    log.info("feature_store.written", partitions=len(written), rows=len(frame))
    return manifest


def read_feature_store(
    months: list[int] | None = None,
    loan_ids: list[str] | None = None,
    columns: list[str] | None = None,
    *,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Partition-pruned read. Reading months [1..T] physically cannot see T+1."""
    s = settings or get_settings()
    root: Path = s.path("feature_store_dir")
    if not root.exists():
        return pd.DataFrame()

    available = sorted(
        int(p.name.split("=", 1)[1]) for p in root.glob("month_index=*") if p.is_dir()
    )
    wanted = available if months is None else [m for m in available if m in set(months)]
    if not wanted:
        return pd.DataFrame()

    read_cols = None
    if columns is not None:
        read_cols = sorted({*columns, "loan_id"} - {"month_index"})

    parts: list[pd.DataFrame] = []
    for m in wanted:
        path = root / f"month_index={m}" / "part.parquet"
        if not path.exists():
            continue
        filters = [("loan_id", "in", set(loan_ids))] if loan_ids else None
        part = pq.read_table(path, columns=read_cols, filters=filters).to_pandas()
        part.insert(1, "month_index", m)
        parts.append(part)

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    return out.sort_values(["loan_id", "month_index"], kind="mergesort").reset_index(drop=True)


def feature_store_months(settings: Settings | None = None) -> list[int]:
    s = settings or get_settings()
    root: Path = s.path("feature_store_dir")
    if not root.exists():
        return []
    return sorted(int(p.name.split("=", 1)[1]) for p in root.glob("month_index=*") if p.is_dir())
