"""Offline pipeline stages. One command per stage, all reproducible.

    data      ingest -> contract check -> DuckDB raw zone, assert dataset SHA256
    profile   four-stage profiling + drift + validation -> reports/
    features  build the Parquet feature store + run the leakage suite
    train     hazard core, direct heads, stacking, calibration, thresholds
    evaluate  all metrics (overall and active-conditional), survival, explainability
    simulate  Monte-Carlo scenarios, tornado, exact Shapley
    submit    submission.csv + manifest

Nothing here runs inside an API request. `make all` chains them, and every stage
writes a manifest recording inputs, hashes, row counts and duration.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lpie.core.config import Settings, get_settings
from lpie.core.determinism import git_sha, seed_everything
from lpie.core.logging import configure_logging, get_logger
from lpie.core.timing import Timer, utcnow_iso
from lpie.data.app_store import get_app_store
from lpie.data.duckdb_store import get_store
from lpie.data.ingest import ingest_all, load_panel, load_servicer, load_static
from lpie.features.builder import (
    FeatureBuilder,
    read_feature_store,
    write_feature_store,
)
from lpie.models.registry import get_artifact_manager

log = get_logger(__name__)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def _record_run(stage: str, started: str, timer: Timer, payload: dict[str, Any], settings: Settings) -> None:
    get_app_store(settings).record_pipeline_run(
        {
            "run_id": f"{stage}-{uuid.uuid4().hex[:12]}",
            "stage": stage,
            "started_at": started,
            "finished_at": utcnow_iso(),
            "status": payload.get("status", "ok"),
            "inputs": payload.get("inputs"),
            "outputs": payload.get("outputs"),
            "row_counts": payload.get("row_counts"),
            "duration_ms": int(timer.elapsed_ms),
            "git_sha": git_sha(settings.root),
            "data_sha256": payload.get("data_sha256"),
            "error": payload.get("error"),
        }
    )


# --------------------------------------------------------------------------- #
def stage_data(settings: Settings) -> dict[str, Any]:
    started, timer = utcnow_iso(), Timer()
    result = ingest_all(settings)
    store = get_store(settings)
    store.initialise()
    get_app_store(settings).initialise()

    monthly = pd.concat([result.frames["train"], result.frames["test"]], ignore_index=True, sort=False)
    for column in settings.require("data.target_columns"):
        if column not in monthly.columns:
            monthly[column] = np.nan

    schema_columns = [
        "loan_id", "month_index", "reporting_month", "origination_month", "loan_age_months",
        "remaining_term_months", "original_balance", "current_balance", "interest_rate",
        "credit_score_band", "ltv_band", "dti_band", "state", "loan_purpose", "occupancy_type",
        "property_type", "servicer_name", "current_status", "days_past_due", "modification_flag",
        "prepayment_flag", "default_flag", "loss_severity_band", "last_updated_at",
        "source_system", "document_status", *settings.require("data.target_columns"), "_split",
    ]
    store.replace_table("raw_monthly", monthly[[c for c in schema_columns if c in monthly.columns]])
    store.replace_table("raw_static", result.frames["static"])
    store.replace_table("raw_servicer", result.frames["servicer"])
    store.replace_table("raw_macro", result.frames["macro"])

    payload = {
        "status": "ok",
        "data_sha256": result.data_sha256,
        "file_hashes": result.file_hashes,
        "row_counts": {k: int(len(v)) for k, v in result.frames.items()},
        "contract_reports": result.contract_reports,
        "outputs": {"duckdb": str(settings.path("duckdb_path"))},
    }
    _write_json(settings.path("reports_dir") / "ingest_manifest.json", payload)
    timer.stop()
    _record_run("data", started, timer, payload, settings)
    log.info("stage.data.done", rows=payload["row_counts"], elapsed_ms=timer.elapsed_ms)
    return payload


def stage_profile(settings: Settings, *, fast: bool = False) -> dict[str, Any]:
    from lpie.profiling.drift import drift_report
    from lpie.profiling.profiler import profile_frame, unseen_categories
    from lpie.profiling.relationships import association_rules
    from lpie.validation.engine import ValidationEngine

    started, timer = utcnow_iso(), Timer()
    panel = load_panel(settings)
    static = load_static(settings)
    servicer = load_servicer(settings)

    train_max = int(settings.get("data.train_month_max", 36))
    train = panel[panel["month_index"] <= train_max]
    test = panel[panel["month_index"] > train_max]

    sample_rows = 20_000 if fast else 60_000
    profile = profile_frame(
        train, name="train_panel", sample_rows=sample_rows,
        include_relationships=not fast, include_missingness=not fast, settings=settings,
    )

    engine = ValidationEngine(settings)
    engine.fit_vocabulary(train)
    validation = engine.run(train, static=static, servicer=servicer, batch_id="train")

    store = get_store(settings)
    store.replace_table("dq_record_scores", validation.record_scores)
    store.replace_table(
        "dq_rule_results",
        validation.violations[
            ["loan_id", "month_index", "rule_id", "severity", "exception_type",
             "dimension", "observed_value", "expected_condition"]
        ],
    )

    ref_window = settings.get("drift.default_ref_window", "31-36")
    lo, hi = (int(x) for x in str(ref_window).split("-"))
    reference = panel[panel["month_index"].between(lo, hi)]
    shared = [c for c in test.columns if c in reference.columns]
    drift = drift_report(
        reference, test, ref_window=ref_window,
        cur_window=settings.get("drift.default_cur_window", "37-42"),
        columns=shared, include_adversarial=not fast, settings=settings,
    )
    store.replace_table(
        "drift_metrics",
        pd.DataFrame(drift["features"]).assign(
            ref_window=drift["ref_window"], cur_window=drift["cur_window"],
            kind=lambda d: d["kind"], computed_at=pd.Timestamp.utcnow(),
        )[["feature", "ref_window", "cur_window", "psi", "ks_stat", "ks_pvalue",
           "js_div", "missing_delta", "verdict", "kind", "computed_at"]],
    )

    rules = association_rules(
        train.sample(min(len(train), 60_000), random_state=settings.seed),
        columns=[c for c in ("document_status", "source_system", "current_status", "servicer_name",
                             "credit_score_band", "modification_flag", "state") if c in train.columns],
        consequent="exception_required",
    ) if "exception_required" in train.columns else []

    payload = {
        "status": "ok",
        "profile": profile,
        "validation_summary": validation.summary,
        "batch_dq": validation.batch_score,
        "drift": drift,
        "unseen_categories_test_vs_train": unseen_categories(train, test),
        "association_rules": rules,
        "row_counts": {"train": int(len(train)), "test": int(len(test))},
        "outputs": {"report": "reports/data_intelligence.json"},
    }
    _write_json(settings.path("reports_dir") / "data_intelligence.json", payload)
    timer.stop()
    _record_run("profile", started, timer, payload, settings)
    log.info("stage.profile.done", elapsed_ms=timer.elapsed_ms)
    return payload


def stage_features(settings: Settings, *, run_leakage: bool = True) -> dict[str, Any]:
    from lpie.features import leakage_tests

    started, timer = utcnow_iso(), Timer()
    panel = load_panel(settings)
    static = load_static(settings)
    servicer = load_servicer(settings)

    builder = FeatureBuilder(settings)
    fit_params = builder.fit(panel, static)
    result = builder.build(panel, static, servicer)

    manifest = write_feature_store(result.features, result.targets, settings=settings)
    get_artifact_manager(settings).save(
        "feature_fit", fit_params,
        metadata={"feature_hash": result.feature_hash, "n_features": result.n_features},
    )
    get_store(settings).refresh_feature_view()

    leakage: dict[str, Any] = {"skipped": True}
    if run_leakage:
        report = leakage_tests.run_all(
            panel, static, servicer, result.features, result.targets,
            registry=builder.registry, fit_params=fit_params, settings=settings,
        )
        leakage = report.to_dict()

    payload = {
        "status": "ok" if leakage.get("passed", True) else "failed",
        "n_rows": result.n_rows,
        "n_features": result.n_features,
        "feature_hash": result.feature_hash,
        "fit_params_hash": fit_params.hash(),
        "stage_timings_ms": result.stage_timings,
        "families": {k: len(v) for k, v in builder.registry.by_family().items()},
        "store_manifest": manifest,
        "leakage": leakage,
        "row_counts": {"features": result.n_rows},
        "outputs": {"feature_store": manifest["root"]},
    }
    _write_json(settings.path("reports_dir") / "feature_manifest.json", payload)
    _write_feature_contract(settings, builder, result, leakage)
    timer.stop()
    _record_run("features", started, timer, payload, settings)

    if run_leakage and not leakage.get("passed", True):
        failed = [c["name"] for c in leakage["checks"] if not c["passed"]]
        raise SystemExit(f"Leakage checks failed: {failed}")

    log.info("stage.features.done", rows=result.n_rows, features=result.n_features)
    return payload


def _write_feature_contract(settings: Settings, builder, result, leakage: dict[str, Any]) -> None:
    """Emit FEATURE_CONTRACT.md and config/features.yaml from the registry.

    Generated, never hand-written, so the document cannot drift from the code.
    """
    import yaml

    registry = builder.registry
    by_family = registry.by_family()

    lines = [
        "# Feature Contract",
        "",
        "> Generated by `make features` from `lpie.features.registry`. Do not edit by hand.",
        f"> Feature version `{settings.feature_version}` · contract hash `{result.feature_hash[:16]}` "
        f"· {len(registry)} features in {len(by_family)} families.",
        "",
        "## Governance rules",
        "",
        "1. Every feature used by a head must be declared here.",
        "2. No feature may declare `temporal_offset > 0` — that is a future read.",
        "3. No feature may source a target column.",
        "4. Any feature with `leakage_risk` above `low` must carry a written justification.",
        "5. A feature whose univariate AUC against a forward target exceeds 0.97 is quarantined.",
        "",
        "**Explicitly banned as features:** all seven target columns, `loan_id` as a value "
        "(identity only), `loss_severity_band` (100% null in test), `reporting_month` "
        "(corrupted clock — see VR-013), and any aggregate computed over months >= t.",
        "",
        "**Target encoding is banned**, not merely unused: with five servicers and fifty "
        "states there is nothing to gain, and out-of-fold target encoding over a time panel "
        "is a classic leakage vector. Recorded as a considered-and-rejected option.",
        "",
        "## Leakage test results",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for check in leakage.get("checks", []):
        lines.append(
            f"| `{check['name']}` | {'PASS' if check['passed'] else 'FAIL'} | {check['detail']} |"
        )

    for family, specs in by_family.items():
        lines += [
            "",
            f"## Family `{family}` ({len(specs)} features)",
            "",
            "| Feature | dtype | Temporal offset | Allowed heads | Leakage risk | Source columns | Description |",
            "|---|---|---:|---|---|---|---|",
        ]
        for s in specs:
            heads = "all" if len(s.allowed_heads) >= 8 else ", ".join(sorted(s.allowed_heads))
            lines.append(
                f"| `{s.name}` | {s.dtype} | {s.temporal_offset} | {heads} | {s.leakage_risk} | "
                f"{', '.join(f'`{c}`' for c in s.source_columns)} | {s.description} |"
            )
            if s.justification:
                lines.append(f"| | | | | | | **Justification:** {s.justification} |")

    (settings.root / "FEATURE_CONTRACT.md").write_text("\n".join(lines) + "\n")

    features_yaml = {
        "feature_version": settings.feature_version,
        "contract_hash": result.feature_hash,
        "n_features": len(registry),
        "families": {family: [s.name for s in specs] for family, specs in by_family.items()},
        "per_head_allowlist": {
            head: registry.for_head(head)
            for head in ("next_3m_delinquency", "next_6m_delinquency", "next_12m_default",
                         "next_12m_prepayment", "next_state", "exception_required",
                         "hazard", "anomaly")
        },
        "categorical_features": registry.categorical_features(),
        "declarations": registry.to_dicts(),
    }
    (settings.root / "config" / "features.yaml").write_text(
        yaml.safe_dump(features_yaml, sort_keys=False, width=100)
    )


def stage_train(
    settings: Settings, *, algorithms: tuple[str, ...] = ("lightgbm", "xgboost", "catboost")
) -> dict[str, Any]:
    from lpie.anomaly.residual_ml import ResidualArtifact
    from lpie.anomaly.unsupervised import UnsupervisedEnsemble
    from lpie.data.ingest import data_sha256
    from lpie.explain.conformal import MondrianConformal
    from lpie.features.builder import build_registry
    from lpie.models.splitters import build_split_plan, censoring_mask
    from lpie.pipelines.train import (
        BINARY_HEADS,
        train_binary_head,
        train_exception_head,
        train_hazard,
        training_manifest,
    )

    started, timer = utcnow_iso(), Timer()
    seed_everything(settings.seed)

    features = read_feature_store(settings=settings)
    if features.empty:
        raise SystemExit("Feature store is empty. Run `make features` first.")

    registry = build_registry()
    manager = get_artifact_manager(settings)
    sha = data_sha256()

    hazard, hazard_metrics = train_hazard(features, registry=registry, settings=settings)
    manager.save("hazard", hazard.artifact, metadata={"metrics": _slim(hazard_metrics)})
    manager.register(
        head="next_state", algo="lightgbm-multiclass", artifact_name="hazard",
        metrics=_slim(hazard_metrics.get("next_state", {})),
        train_window=str(hazard_metrics.get("train_window", "")),
        valid_window=str(hazard_metrics.get("valid_window", "")), embargo_months=1,
        feature_hash=registry.contract_hash(), data_sha256=sha,
        n_features=len(registry.for_head("hazard")),
    )

    heads: dict[str, Any] = {}
    stacks: dict[str, Any] = {}
    calibrators: dict[str, Any] = {}
    thresholds: dict[str, Any] = {}
    conformal: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    results = {}

    for head in BINARY_HEADS:
        result = train_binary_head(
            head, features, registry=registry, hazard=hazard, settings=settings,
            algorithms=algorithms,
        )
        results[head] = result
        heads[head] = result.base_models
        stacks[head] = result.stack
        calibrators[head] = result.calibrator
        if result.threshold is not None:
            thresholds[head] = result.threshold
        summaries[head] = {
            **result.summary,
            "calibration": result.calibration_metrics,
            "baseline_ladder": result.baseline_ladder,
            "folds": result.fold_metrics,
        }

        # Conformal calibration on the same out-of-fold rows the isotonic map used.
        plan = build_split_plan(head, settings=settings)
        target = settings.require(f"heads.{head}.target")
        mask = features[target].notna() & censoring_mask(
            features["month_index"], plan.horizon, plan.panel_max_month
        )
        fold_months = {m for f in plan.folds for m in f.valid_months}
        mask &= features["month_index"].isin(fold_months)
        if mask.sum() > 200:
            X_cal = features.loc[mask]
            scored = _score_for_conformal(result, hazard, X_cal, head, plan.horizon, settings)
            model = MondrianConformal(float(settings.get("explain.conformal_alpha", 0.10)))
            model.fit(scored, pd.to_numeric(X_cal[target], errors="coerce").to_numpy(),
                      X_cal["current_status"])
            conformal[head] = model

        manager.register(
            head=head, algo="stack(lgb,xgb,cat,hazard,baseline)+isotonic",
            artifact_name="heads", metrics=_slim(result.summary.get("out_of_fold_metrics", {})),
            train_window=str(result.summary.get("production_train_window", "")),
            valid_window="+".join(f.valid_window for f in result.plan.folds),
            embargo_months=result.plan.horizon, feature_hash=registry.contract_hash(),
            data_sha256=sha, n_features=len(registry.for_head(head)),
            notes=result.plan.fold_note,
        )

    exception = train_exception_head(features, registry=registry, settings=settings)
    manager.save("exception", ResidualArtifact(
        model=exception["model"],
        threshold=exception["threshold"].threshold if exception["threshold"] else 0.5,
        rule_columns=exception["rule_columns"],
        metrics={"rules_only": exception["rules_only"], "hybrid": exception["hybrid"]},
        ceiling=exception["ceiling_analysis"],
    ))
    manager.register(
        head="exception_required", algo="rules+residual_lightgbm", artifact_name="exception",
        metrics=_slim({"rules_only": exception["rules_only"], "hybrid": exception["hybrid"]}),
        train_window=str(exception["train_window"]), valid_window=str(exception["valid_window"]),
        embargo_months=0, feature_hash=registry.contract_hash(), data_sha256=sha,
        n_features=len(registry.for_head("exception_required")),
    )

    anomaly_features = [
        f for f in registry.for_head("anomaly")
        if f in features.columns and pd.api.types.is_numeric_dtype(features[f])
    ]
    ensemble = UnsupervisedEnsemble(settings)
    ensemble.fit(features, anomaly_features)
    manager.save("anomaly", ensemble, metadata={"detectors": ensemble.artifact.available})

    manager.save("heads", heads, metadata={"heads": list(heads)})
    manager.save("stacks", stacks)
    manager.save("calibrators", calibrators)
    manager.save("thresholds", thresholds)
    manager.save("conformal", conformal)

    manifest = training_manifest(
        results, hazard_metrics, exception,
        feature_hash=registry.contract_hash(), data_sha256=sha, settings=settings,
    )
    manifest["heads"] = summaries
    manifest["anomaly_detectors"] = ensemble.artifact.available
    manifest["conformal_coverage"] = {
        head: model.artifact.achieved_coverage for head, model in conformal.items()
    }
    _write_json(settings.path("reports_dir") / "model_performance.json", manifest)
    manager.save("evaluation", manifest)

    payload = {"status": "ok", "heads": list(heads), "outputs": {"models": str(settings.path("models_dir"))}}
    timer.stop()
    _record_run("train", started, timer, {**payload, "data_sha256": sha}, settings)
    log.info("stage.train.done", elapsed_ms=timer.elapsed_ms)
    return {**payload, "manifest": manifest}


def _score_for_conformal(result, hazard, X, head, horizon, settings):
    from lpie.models.calibration import segment_labels
    from lpie.pipelines.train import _stack_predict

    raw = _stack_predict(result, result.stack, hazard, X, head, horizon)
    if result.calibrator and result.calibrator.artifact.is_fitted:
        segments = segment_labels(X.get("vintage_year_num"), X.get("credit_score_band"), X.index)
        return result.calibrator.transform(raw, segments)
    return raw


def _slim(payload: Any, max_chars: int = 40_000) -> Any:
    """Drop long curve arrays before a metrics blob goes into the registry."""
    text = json.dumps(payload, default=_json_default)
    if len(text) <= max_chars:
        return payload
    if isinstance(payload, dict):
        return {
            k: v for k, v in payload.items()
            if k not in ("curve", "reliability_curve", "rate_by_month", "folds", "confusion_matrix")
        }
    return {"note": "metrics payload truncated"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lpie", description="LPIE offline pipeline")
    parser.add_argument(
        "stage",
        choices=["data", "profile", "features", "train", "evaluate", "simulate", "submit", "all"],
    )
    parser.add_argument("--fast", action="store_true", help="skip the expensive profiling stages")
    parser.add_argument("--no-leakage", action="store_true", help="skip the leakage suite (CI only)")
    parser.add_argument("--algorithms", default="lightgbm,xgboost,catboost")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.get("logging.level", "INFO"), bool(settings.get("logging.json", True)))
    seed_everything(settings.seed)
    settings.ensure_dirs()

    algorithms = tuple(a.strip() for a in args.algorithms.split(",") if a.strip())
    stages = (
        ["data", "profile", "features", "train", "evaluate", "simulate", "submit"]
        if args.stage == "all"
        else [args.stage]
    )

    overall = time.perf_counter()
    for stage in stages:
        print(f"\n=== stage: {stage} ===", file=sys.stderr, flush=True)
        if stage == "data":
            stage_data(settings)
        elif stage == "profile":
            stage_profile(settings, fast=args.fast)
        elif stage == "features":
            stage_features(settings, run_leakage=not args.no_leakage)
        elif stage == "train":
            stage_train(settings, algorithms=algorithms)
        elif stage == "evaluate":
            from lpie.pipelines.evaluate import stage_evaluate

            stage_evaluate(settings, fast=args.fast)
        elif stage == "simulate":
            from lpie.pipelines.simulate import stage_simulate

            stage_simulate(settings, fast=args.fast)
        elif stage == "submit":
            from lpie.pipelines.submit import stage_submit

            stage_submit(settings)

    print(f"\nPipeline complete in {time.perf_counter() - overall:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
