"""Scenario execution service — used by both the API and the offline stage.

One code path, so a scenario run from `/api/v1/scenario/run` and one from
`make simulate` produce byte-identical numbers for the same seed. Results are
cached by `(scenario, horizon, paths, seed, segment)` because a 1,000-path run
over 10,000 loans is seconds, not milliseconds, and a dashboard will ask for the
same scenario repeatedly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from lpie.core.config import Settings, get_settings
from lpie.core.determinism import sha256_obj
from lpie.core.exceptions import ModelNotLoadedError, ScenarioNotFoundError
from lpie.core.logging import get_logger
from lpie.models.thresholds import severity_midpoints
from lpie.scenario.montecarlo import segment_impact, simulate, summarise
from lpie.scenario.sensitivity import MACRO_VARIABLES, MULTIPLIER_VARIABLES, tornado
from lpie.scenario.shapley_macro import compare_with_tornado, exact_shapley
from lpie.scenario.transmission import (
    MacroShock,
    apply_hazard_multipliers,
    baseline_shock,
    get_scenario,
    load_scenarios,
    reanchor_scaling,
    transmit_to_features,
)

log = get_logger(__name__)

SEGMENT_COLUMNS = {
    "vintage": "vintage_year_num",
    "credit_band": "credit_score_band",
    "state": "state",
    "servicer": "servicer_name",
    "ltv_band": "ltv_band_ord",
}


@dataclass
class ScenarioRun:
    scenario: str
    summary: dict[str, Any]
    segments: list[dict[str, Any]]
    reanchor: dict[str, Any]
    assumptions: dict[str, Any]
    cache_key: str


class ScenarioRunner:
    def __init__(self, state: Any, settings: Settings | None = None) -> None:
        self.state = state
        self.settings = settings or get_settings()
        self.severity_map = severity_midpoints(self.settings)
        self.default_severity = float(self.severity_map.get("N/A", 0.35))

    # ------------------------------------------------------------------ #
    def available_scenarios(self) -> list[dict[str, Any]]:
        return [shock.to_dict() for shock in load_scenarios(self.state.macro()).values()]

    def as_of_features(self, *, max_loans: int | None = None) -> pd.DataFrame:
        """The latest observed month per loan — the simulation's starting point."""
        months = self.state.feature_store_months()
        if not months:
            raise ModelNotLoadedError(
                "Feature store is empty; scenarios cannot be simulated.",
                details={"remedy": "run `make features`"},
            )
        frame = self.state.features(months=[max(months)])
        if frame.empty:
            raise ModelNotLoadedError("No feature rows at the latest month")
        if max_loans is not None and len(frame) > max_loans:
            frame = frame.sample(max_loans, random_state=self.settings.seed)
        return frame.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    def run(
        self,
        scenario: str | MacroShock,
        *,
        n_paths: int | None = None,
        horizon: int | None = None,
        segment_by: str | None = None,
        seed: int | None = None,
        reanchor: bool = True,
        max_loans: int | None = None,
        use_cache: bool = True,
    ) -> ScenarioRun:
        s = self.settings
        n_paths = int(n_paths or s.get("scenario.default_n_paths", 1000))
        horizon = int(horizon or s.get("scenario.default_horizon_months", 24))
        seed = int(seed if seed is not None else s.seed)

        n_paths = min(n_paths, int(s.get("scenario.max_n_paths", 5000)))
        horizon = min(horizon, int(s.get("scenario.max_horizon_months", 60)))

        macro = self.state.macro()
        shock = scenario if isinstance(scenario, MacroShock) else get_scenario(macro, scenario)
        base = self._base_shock(macro)

        cache_key = sha256_obj(
            {
                "scenario": shock.to_dict(), "n_paths": n_paths, "horizon": horizon,
                "seed": seed, "segment_by": segment_by, "reanchor": reanchor,
                "model_version": s.model_version,
            }
        )
        if use_cache:
            cached = self.state.scenario_cache.get(cache_key)
            if cached is not None:
                return cached

        hazard = self.state.hazard
        if hazard is None or not hazard.is_loaded:
            raise ModelNotLoadedError(
                "Hazard model is unavailable; scenarios cannot be simulated.",
                details={"artifact": "hazard", "remedy": "run `make train`"},
            )

        features = self.as_of_features(max_loans=max_loans)
        start = (
            features["current_status"].astype("object").map(hazard.state_index)
            .fillna(0).astype("int32").to_numpy()
        )
        balances = pd.to_numeric(features.get("current_balance"), errors="coerce").fillna(0.0).to_numpy()
        severity = self._severity(features)

        base_matrices = hazard.transition_matrices(features)

        def build_matrices(candidate: MacroShock, scaling: float = 1.0) -> np.ndarray:
            shocked_features = transmit_to_features(features, candidate, base, settings=s)
            matrices = hazard.transition_matrices(shocked_features)
            return apply_hazard_multipliers(
                matrices, hazard.states, candidate, base, scaling=scaling, settings=s
            )

        def default_rate_at(scaling: float) -> float:
            matrices = build_matrices(shock, scaling)
            propagated = hazard.propagate(matrices, start, horizon)
            return float(propagated["cif_default"][:, -1].mean())

        base_propagated = hazard.propagate(base_matrices, start, horizon)
        base_default_rate = float(base_propagated["cif_default"][:, -1].mean())

        anchor = {"enabled": False, "scaling": 1.0}
        if reanchor and abs(shock.default_rate_multiplier - 1.0) > 1e-6:
            anchor = reanchor_scaling(default_rate_at, shock, base_default_rate, settings=s)

        matrices = build_matrices(shock, float(anchor.get("scaling", 1.0)))
        result = simulate(
            matrices, start, balances, severity, states=hazard.states,
            scenario=shock.scenario_name, n_paths=n_paths, horizon=horizon, seed=seed,
            settings=s,
        )
        summary = summarise(
            result,
            levels=tuple(int(x) for x in s.get("scenario.confidence_levels", [5, 50, 95])),
            var_level=int(s.get("scenario.var_level", 95)),
        )
        summary["base_scenario_default_rate"] = round(base_default_rate, 8)
        summary["reanchoring"] = anchor
        summary["assumptions"] = shock.to_dict()
        summary["as_of_month"] = int(features["month_index"].iloc[0])

        segments: list[dict[str, Any]] = []
        if segment_by:
            column = SEGMENT_COLUMNS.get(segment_by, segment_by)
            if column not in features.columns:
                raise ScenarioNotFoundError(
                    f"Unknown segment '{segment_by}'",
                    details={"available": sorted(SEGMENT_COLUMNS)},
                )
            scenario_propagated = hazard.propagate(matrices, start, horizon)
            segments = segment_impact(
                scenario_propagated["cif_default"][:, -1],
                base_propagated["cif_default"][:, -1],
                features[column].astype(str),
                balances=balances,
                loss_severity=severity,
            )

        run = ScenarioRun(
            scenario=shock.scenario_name, summary=summary, segments=segments,
            reanchor=anchor, assumptions=shock.to_dict(), cache_key=cache_key,
        )
        if use_cache:
            self.state.scenario_cache.put(cache_key, run)
        return run

    # ------------------------------------------------------------------ #
    def sensitivity(
        self,
        scenario: str,
        *,
        horizon: int | None = None,
        max_loans: int = 4000,
        include_shapley: bool = True,
    ) -> dict[str, Any]:
        """Tornado plus exact Shapley, on the deterministic CIF rather than MC.

        Sensitivity needs 64+ evaluations. Running Monte Carlo for each would be
        wasteful and would add sampling noise on top of the effect being measured;
        the propagated CIF is the exact expectation, so the attribution is clean.
        """
        s = self.settings
        horizon = int(horizon or s.get("scenario.default_horizon_months", 24))
        macro = self.state.macro()
        shock = get_scenario(macro, scenario)
        base = self._base_shock(macro)

        hazard = self.state.hazard
        if hazard is None or not hazard.is_loaded:
            raise ModelNotLoadedError("Hazard model is unavailable for sensitivity analysis")

        features = self.as_of_features(max_loans=max_loans)
        start = (
            features["current_status"].astype("object").map(hazard.state_index)
            .fillna(0).astype("int32").to_numpy()
        )

        def evaluate(candidate: MacroShock) -> float:
            shocked = transmit_to_features(features, candidate, base, settings=s)
            matrices = apply_hazard_multipliers(
                hazard.transition_matrices(shocked), hazard.states, candidate, base, settings=s
            )
            return float(hazard.propagate(matrices, start, horizon)["cif_default"][:, -1].mean())

        oat = tornado(evaluate, base, shock, metric_name="cumulative_default_rate_24m")
        out: dict[str, Any] = {
            "scenario": scenario,
            "horizon_months": horizon,
            "n_loans": int(len(features)),
            "metric": "cumulative_default_rate_24m",
            "tornado": oat,
        }
        if include_shapley:
            variables = tuple(
                v for v in (*MACRO_VARIABLES, *MULTIPLIER_VARIABLES)
                if getattr(base, v) != getattr(shock, v)
            )
            shapley = exact_shapley(evaluate, base, shock, variables,
                                    metric_name="cumulative_default_rate_24m")
            out["shapley"] = shapley
            out["tornado_vs_shapley"] = compare_with_tornado(shapley, oat)
        return out

    # ------------------------------------------------------------------ #
    def backtest_base(self, *, n_paths: int = 200, horizon: int = 12) -> dict[str, Any]:
        """Validate the simulator against the observed rate — a self-check.

        If the Base scenario cannot reproduce what actually happened, no stressed
        scenario built on it is worth reading.
        """
        run = self.run("Base", n_paths=n_paths, horizon=horizon, reanchor=False, use_cache=False)
        simulated = run.summary["terminal"]["default_rate"]["mean"]

        panel = self.state.panel()
        train_max = int(self.settings.get("data.train_month_max", 36))
        observed_window = panel[panel["month_index"].between(train_max - horizon, train_max)]
        observed = (
            float(observed_window["current_status"].eq("Default").mean())
            if not observed_window.empty else None
        )
        relative_error = (
            abs(simulated - observed) / observed if observed and observed > 1e-9 else None
        )
        return {
            "simulated_default_rate": simulated,
            "observed_default_rate": round(observed, 8) if observed is not None else None,
            "relative_error": round(relative_error, 6) if relative_error is not None else None,
            "tolerance": 0.10,
            "passed": relative_error is not None and relative_error <= 0.10,
            "horizon_months": horizon,
            "note": (
                "The Base scenario must reproduce the observed empirical rate within "
                "tolerance. A simulator that cannot backtest is a chart, not a model."
            ),
        }

    # ------------------------------------------------------------------ #
    def _base_shock(self, macro: pd.DataFrame) -> MacroShock:
        try:
            return get_scenario(macro, "Base")
        except ScenarioNotFoundError:
            return baseline_shock()

    def _severity(self, features: pd.DataFrame) -> np.ndarray:
        band = features.get("loss_severity_band")
        if band is None:
            return np.full(len(features), self.default_severity, dtype="float64")
        mapped = band.astype("object").map(self.severity_map)
        return pd.to_numeric(mapped, errors="coerce").fillna(self.default_severity).to_numpy()
