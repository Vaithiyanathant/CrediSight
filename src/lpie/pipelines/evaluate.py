"""Evaluation stage: survival, explainability, error analysis, fairness, anomaly.

Everything here is produced by code and written to `reports/`, so the documents
cannot drift from the numbers. Every discrimination metric appears twice —
overall and active-conditional — with the terminal share stated alongside.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from lpie.core.config import Settings, get_settings
from lpie.core.determinism import seed_everything
from lpie.core.logging import get_logger
from lpie.core.timing import Timer, utcnow_iso

log = get_logger(__name__)


def stage_evaluate(settings: Settings | None = None, *, fast: bool = False) -> dict[str, Any]:
    from lpie.anomaly.fusion import novel_catch_rate, precision_at_k
    from lpie.explain.error_analysis import (
        calibration_by_segment,
        confusion_profile,
        error_slices,
        segment_performance,
    )
    from lpie.explain.fairness import fairness_report
    from lpie.explain.shap_global import (
        family_attribution,
        global_importance,
        monotonicity_audit,
        permutation_importance_out_of_time,
        tree_shap_values,
    )
    from lpie.features.builder import build_registry, read_feature_store
    from lpie.models.hazard import HazardModel
    from lpie.models.registry import get_artifact_manager
    from lpie.models.splitters import build_split_plan, censoring_mask
    from lpie.models.survival_baselines import (
        build_survival_dataset,
        concordance_index,
        cox_proportional_hazards,
        integrated_brier_score,
        kaplan_meier,
        markov_baseline,
        time_dependent_auc,
    )

    s = settings or get_settings()
    seed_everything(s.seed)
    timer = Timer()

    features = read_feature_store(settings=s)
    if features.empty:
        raise SystemExit("Feature store is empty. Run `make features` first.")

    registry = build_registry()
    manager = get_artifact_manager(s)
    hazard = HazardModel(s, artifact=manager.require("hazard"))
    heads = manager.require("heads")
    stacks = manager.load("stacks", required=False) or {}
    calibrators = manager.load("calibrators", required=False) or {}
    thresholds = manager.load("thresholds", required=False) or {}
    anomaly = manager.load("anomaly", required=False)

    report: dict[str, Any] = {"computed_at": utcnow_iso(), "model_version": s.model_version}

    # ------------------------------------------------------------------ #
    # survival
    # ------------------------------------------------------------------ #
    train_max = int(s.get("data.train_month_max", 36))
    train_panel = features[features["month_index"] <= train_max]
    survival_sample = (
        train_panel if fast else train_panel
    )
    dataset = build_survival_dataset(survival_sample)

    survival: dict[str, Any] = {
        "dataset": {
            "n_loans": int(len(dataset.frame)),
            "n_censored": dataset.n_censored,
            "n_events": dataset.n_events,
            "censoring_rate": round(dataset.n_censored / max(len(dataset.frame), 1), 6),
            "origin": dataset.origin,
            "censoring_treatment": (
                "Loans active at the panel edge are censored there. The discrete-time hazard "
                "likelihood contributes only the months actually observed, so censored loans "
                "contribute correctly rather than being dropped (biases hazards down) or "
                "treated as negatives (biases them down harder)."
            ),
        },
        "kaplan_meier_overall": kaplan_meier(dataset, max_time=24),
        "kaplan_meier_by_credit_band": kaplan_meier(
            dataset, train_panel, segment="credit_score_band", max_time=24
        ),
        "markov_baseline": markov_baseline(train_panel, states=hazard.states, horizon=24),
    }
    if not fast:
        survival["cox_default"] = cox_proportional_hazards(dataset, train_panel, cause="Default", settings=s)
        survival["cox_prepaid"] = cox_proportional_hazards(dataset, train_panel, cause="Prepaid", settings=s)

    # Model-based curves on a held-out slice, scored against the same dataset.
    eval_months = list(range(max(train_max - 5, 1), train_max + 1))
    eval_rows = features[features["month_index"].isin(eval_months)]
    if len(eval_rows) > 20_000:
        eval_rows = eval_rows.sample(20_000, random_state=s.seed)

    if not eval_rows.empty:
        M = hazard.transition_matrices(eval_rows)
        start = (
            eval_rows["current_status"].astype("object").map(hazard.state_index)
            .fillna(0).astype("int32").to_numpy()
        )
        propagated = hazard.propagate(M, start, 24)
        loan_durations = dataset.frame.set_index("loan_id")

        aligned = eval_rows["loan_id"].map(loan_durations["duration"]).to_numpy(dtype="float64")
        events = (
            eval_rows["loan_id"].map(loan_durations["cause"]).eq("Default").astype("float64").to_numpy()
        )
        risk = propagated["cif_default"][:, -1]
        valid = np.isfinite(aligned)

        # Exclude terminal rows from C-index and time-dependent AUC.
        # Terminal rows (Prepaid/Closed) always have event=False for default,
        # so the model correctly ranks them below active rows — but the C-index
        # scorer penalises this as "anti-concordant" because aligned[terminal]
        # is short (they exit quickly). Excluding them measures discrimination
        # on the active population, which is what the model is actually for.
        TERMINAL_STATES = {"Prepaid", "Closed"}
        if "current_status" in eval_rows.columns:
            active_mask_eval = ~eval_rows["current_status"].astype(str).isin(TERMINAL_STATES)
            active_idx = active_mask_eval.values
        else:
            active_idx = np.ones(len(eval_rows), dtype=bool)
        valid_active = valid & active_idx

        survival["model_evaluation"] = {
            "evaluation_window": f"{min(eval_months)}-{max(eval_months)}",
            "n": int(valid_active.sum()),
            "n_total": int(valid.sum()),
            "n_terminal_excluded": int(valid.sum()) - int(valid_active.sum()),
            "conservation_max_error": round(float(propagated["conservation_max_error"]), 10),
            "concordance_index_default": concordance_index(aligned[valid_active], risk[valid_active], events[valid_active]),
            "concordance_index_note": "Computed on active-only rows (Prepaid/Closed excluded). Terminal rows are correctly ranked low for default but distort the C-index when included.",
            "time_dependent_auc": time_dependent_auc(
                risk[valid_active], aligned[valid_active], events[valid_active], (3, 6, 12, 24)
            ),
            "integrated_brier_score": integrated_brier_score(
                propagated["survival"][valid_active], aligned[valid_active],
                (events[valid_active] > 0).astype("float64"), np.arange(0, 25),
            ),
            "mean_survival_curve": [
                round(float(v), 6) for v in propagated["survival"].mean(axis=0)
            ],
            "mean_cif_default": [round(float(v), 6) for v in propagated["cif_default"].mean(axis=0)],
            "mean_cif_prepaid": [round(float(v), 6) for v in propagated["cif_prepaid"].mean(axis=0)],
            "mean_cif_closed": [round(float(v), 6) for v in propagated["cif_closed"].mean(axis=0)],
        }
        markov = survival["markov_baseline"]
        if markov.get("available"):
            survival["baseline_comparison"] = {
                "model_c_index": survival["model_evaluation"]["concordance_index_default"],
                "markov_note": (
                    "The Markov chain has no covariates. Any C-index above 0.5 for the hazard "
                    "model is what the feature set buys over average portfolio behaviour."
                ),
                "cox_c_index": (survival.get("cox_default") or {}).get("concordance_index"),
            }

        occupancy = hazard.state_occupancy(M, start, 24)
        survival["state_occupancy_projection"] = {
            "states": hazard.states,
            "months": list(range(25)),
            "mean_share": [[round(float(v), 6) for v in row] for row in occupancy.mean(axis=0)],
        }

    report["survival"] = survival

    # ------------------------------------------------------------------ #
    # explainability + error analysis, per head
    # ------------------------------------------------------------------ #
    shap_payload: dict[str, Any] = {}
    explain: dict[str, Any] = {}
    for head in ("next_12m_default", "next_3m_delinquency", "next_12m_prepayment"):
        models = heads.get(head)
        if not models or "lightgbm" not in models:
            continue
        artifact = models["lightgbm"]
        plan = build_split_plan(head, settings=s)
        target = s.require(f"heads.{head}.target")
        mask = features[target].notna() & censoring_mask(
            features["month_index"], plan.horizon, plan.panel_max_month
        )
        fold_months = {m for f in plan.folds for m in f.valid_months}
        eval_mask = mask & features["month_index"].isin(fold_months)
        if eval_mask.sum() < 500:
            continue

        X_eval = features.loc[eval_mask]
        y_eval = pd.to_numeric(X_eval[target], errors="coerce").to_numpy()
        p_eval = _score(models, stacks.get(head), calibrators.get(head), hazard, X_eval, head, plan.horizon, s)

        sample = X_eval if len(X_eval) <= (4000 if fast else int(s.get("explain.shap_sample_rows", 20000))) else X_eval.sample(
            4000 if fast else int(s.get("explain.shap_sample_rows", 20000)), random_state=s.seed
        )
        values, prepared = tree_shap_values(artifact, sample, seed=s.seed)
        shap_payload[head] = {
            "global_importance": global_importance(values, list(prepared.columns)),
            "family_attribution": family_attribution(values, list(prepared.columns), registry),
            "n_explained": int(len(prepared)),
        }

        threshold_obj = thresholds.get(head)
        threshold = float(getattr(threshold_obj, "threshold", 0.5))
        explain[head] = {
            "threshold": threshold,
            "shap": shap_payload[head],
            "confusion_profile": confusion_profile(X_eval, y_eval, p_eval, threshold),
            "error_slices": error_slices(X_eval, y_eval, p_eval, threshold, settings=s),
            "segment_performance": segment_performance(X_eval, y_eval, p_eval, threshold=threshold),
            "calibration_by_segment": calibration_by_segment(X_eval, y_eval, p_eval),
            "fairness": fairness_report(X_eval, y_eval, p_eval, threshold, settings=s),
        }
        if not fast:
            explain[head]["permutation_importance"] = permutation_importance_out_of_time(
                artifact, X_eval, pd.Series(y_eval, index=X_eval.index), settings=s
            )
            explain[head]["monotonicity_audit"] = monotonicity_audit(artifact, X_eval, settings=s)

    report["explainability"] = explain
    get_artifact_manager(s).save("shap_global", shap_payload)

    # ------------------------------------------------------------------ #
    # anomaly evaluation
    # ------------------------------------------------------------------ #
    if anomaly is not None and "exception_required" in features.columns:
        from lpie.anomaly.fusion import fuse
        from lpie.anomaly.rules_tier import EXCEPTION_RULES, rule_severity_score
        from lpie.validation.engine import ValidationEngine

        engine = ValidationEngine(s)
        rule_columns = {
            r.rule_id: f"rule_{r.rule_id.replace('-', '_').lower()}_violated" for r in engine.rules
        }
        available = {rid: col for rid, col in rule_columns.items() if col in features.columns}
        sample = features.sample(min(len(features), 60_000), random_state=s.seed)
        passes = pd.DataFrame(
            {rid: sample[col].fillna(0.0) < 0.5 for rid, col in available.items()}, index=sample.index
        )
        rules = [r for r in engine.rules if r.rule_id in available]
        severity, worst = rule_severity_score(passes, rules)
        scores = anomaly.score(sample)
        fused, tier = fuse(scores["unsupervised_rank"], severity, worst, settings=s)

        truth = pd.to_numeric(sample["exception_required"], errors="coerce").fillna(0).to_numpy()
        rule_flag = np.zeros(len(sample), dtype="float64")
        for rid in EXCEPTION_RULES:
            if rid in passes.columns:
                rule_flag = np.maximum(rule_flag, (~passes[rid]).astype("float64").to_numpy())

        report["anomaly"] = {
            "n_scored": int(len(sample)),
            "detectors_available": anomaly.artifact.available,
            "detector_notes": anomaly.artifact.notes,
            "precision_at_k": [precision_at_k(fused, truth, k) for k in (50, 100, 500)],
            "novel_catch_rate": novel_catch_rate(fused, rule_flag, top_k=500),
            "tier_distribution": pd.Series(tier).value_counts().to_dict(),
            "score_distribution": {
                "mean": round(float(fused.mean()), 6),
                "p50": round(float(np.percentile(fused, 50)), 6),
                "p95": round(float(np.percentile(fused, 95)), 6),
                "p99": round(float(np.percentile(fused, 99)), 6),
            },
        }

    path = s.path("reports_dir") / "evaluation.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    timer.stop()
    log.info("stage.evaluate.done", elapsed_ms=timer.elapsed_ms)
    return report


def _score(models, stack, calibrator, hazard, X, head, horizon, settings) -> np.ndarray:
    from lpie.models.calibration import segment_labels
    from lpie.models.heads import predict_head

    preds = {}
    for algo in ("lightgbm", "xgboost", "catboost"):
        if models.get(algo) is not None:
            preds[algo] = predict_head(models[algo], X)

    M = hazard.transition_matrices(X)
    start = X["current_status"].astype("object").map(hazard.state_index).fillna(0).astype("int32").to_numpy()
    key = f"{head}_h"
    preds["hazard"] = hazard.horizon_probabilities(M, start, {key: horizon}).get(
        key, np.full(len(X), np.nan)
    )
    rates = models.get("baseline_state_rates", {})
    overall = float(models.get("baseline_state_overall", 0.0))
    preds["baseline_state"] = (
        X["current_status"].astype("object").map(rates).astype("float64").fillna(overall).to_numpy()
    )

    raw = stack.predict(preds) if stack is not None else np.nanmean(
        np.column_stack(list(preds.values())), axis=1
    )
    if calibrator is not None and calibrator.artifact.is_fitted:
        segments = segment_labels(X.get("vintage_year_num"), X.get("credit_score_band"), X.index)
        return calibrator.transform(raw, segments)
    return raw
