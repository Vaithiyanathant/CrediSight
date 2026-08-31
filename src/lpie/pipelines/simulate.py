"""Simulation stage: every scenario, plus sensitivity and the Base backtest."""

from __future__ import annotations

import json
from typing import Any

from lpie.core.config import Settings, get_settings
from lpie.core.determinism import seed_everything
from lpie.core.logging import get_logger
from lpie.core.timing import Timer, utcnow_iso

log = get_logger(__name__)


def stage_simulate(settings: Settings | None = None, *, fast: bool = False) -> dict[str, Any]:
    from lpie.serving.scenario_runner import ScenarioRunner
    from lpie.serving.state import AppState

    s = settings or get_settings()
    seed_everything(s.seed)
    timer = Timer()

    state = AppState(s)
    state.startup()
    runner = ScenarioRunner(state, s)

    n_paths = 200 if fast else int(s.get("scenario.default_n_paths", 1000))
    horizon = int(s.get("scenario.default_horizon_months", 24))

    report: dict[str, Any] = {
        "computed_at": utcnow_iso(),
        "n_paths": n_paths,
        "horizon_months": horizon,
        "seed": s.seed,
        "scenarios": {},
        "backtest": runner.backtest_base(n_paths=min(n_paths, 200), horizon=12),
    }

    for entry in runner.available_scenarios():
        name = entry["scenario_name"]
        run = runner.run(
            name, n_paths=n_paths, horizon=horizon, segment_by="credit_band", use_cache=False
        )
        report["scenarios"][name] = {
            "summary": run.summary,
            "segments": run.segments,
            "assumptions": run.assumptions,
        }
        log.info("simulate.scenario_done", scenario=name)

    # Monotonicity: a stress must not reduce the risk it is a stress on.
    base = report["scenarios"].get("Base", {}).get("summary", {})
    base_default = (base.get("terminal", {}).get("default_rate") or {}).get("mean")
    checks = []
    for name, entry in report["scenarios"].items():
        if name == "Base" or base_default is None:
            continue
        summary = entry["summary"]
        default_rate = (summary.get("terminal", {}).get("default_rate") or {}).get("mean")
        prepay_rate = (summary.get("terminal", {}).get("prepayment_rate") or {}).get("mean")
        assumptions = entry["assumptions"]
        expected_up = assumptions.get("default_rate_multiplier", 1.0) > 1.0
        checks.append(
            {
                "scenario": name,
                "default_rate": default_rate,
                "base_default_rate": base_default,
                "expected_direction": "higher" if expected_up else "lower_or_equal",
                "monotonicity_ok": (
                    default_rate >= base_default if expected_up else default_rate <= base_default * 1.05
                ),
                "prepayment_rate": prepay_rate,
            }
        )
    report["monotonicity_checks"] = checks

    if not fast:
        for name in report["scenarios"]:
            if name == "Base":
                continue
            report["scenarios"][name]["sensitivity"] = runner.sensitivity(name, max_loans=3000)

    path = s.path("reports_dir") / "scenarios.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    timer.stop()
    log.info("stage.simulate.done", elapsed_ms=timer.elapsed_ms)
    return report
