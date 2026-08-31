"""Scenario endpoints."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from lpie.api.deps import PredictionDep
from lpie.api.metrics import METRICS
from lpie.api.schemas import (
    CustomScenarioRequest,
    ScenarioInfo,
    ScenarioRequest,
    ScenarioResponse,
    ScenariosResponse,
    SensitivityResponse,
)
from lpie.core.determinism import sha256_obj
from lpie.core.exceptions import LPIEError, ScenarioRunError
from lpie.core.logging import get_logger
from lpie.core.timing import Timer
from lpie.serving.scenario_runner import ScenarioRunner

log = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["scenario"])

SEGMENT_OPTIONS = ["vintage", "credit_band", "state", "servicer", "ltv_band"]


@router.get(
    "/scenarios",
    response_model=ScenariosResponse,
    summary="Available scenarios and their macro assumptions",
)
def list_scenarios(state: PredictionDep) -> ScenariosResponse:
    Timer()
    runner = ScenarioRunner(state)
    raw = runner.available_scenarios()
    scenarios = []
    for s in raw:
        try:
            scenarios.append(ScenarioInfo(
                scenario_name=str(s.get("scenario_name", s.get("name", ""))),
                description=str(s.get("description", "")),
                gdp_growth_pct=float(s.get("gdp_growth_pct", 0.0)),
                unemployment_rate_pct=float(s.get("unemployment_rate_pct", 0.0)),
                hpi_change_pct=float(s.get("hpi_change_pct", 0.0)),
                interest_rate_shock_bps=float(s.get("interest_rate_shock_bps", 0.0)),
                credit_spread_shock_bps=float(s.get("credit_spread_shock_bps", 0.0)),
                prepayment_cpr_assumption_pct=float(s.get("prepayment_cpr_assumption_pct", 0.0)),
                default_rate_multiplier=float(s.get("default_rate_multiplier", 1.0)),
                delinquency_rate_multiplier=float(s.get("delinquency_rate_multiplier", 1.0)),
                prepayment_rate_multiplier=float(s.get("prepayment_rate_multiplier", 1.0)),
            ))
        except Exception as exc:
            log.warning("scenarios.parse_error", error=str(exc))
    return ScenariosResponse(
        scenarios=scenarios,
        defaults={"n_paths": 1000, "horizon": 24},
        segment_options=SEGMENT_OPTIONS,
    )


def _run_scenario(state, scenario_name, n_paths, horizon, segment_by, seed, reanchor, max_loans):
    runner = ScenarioRunner(state)
    run = runner.run(
        scenario=scenario_name,
        n_paths=n_paths,
        horizon=horizon,
        segment_by=segment_by,
        seed=seed,
        reanchor=reanchor,
        max_loans=max_loans,
    )
    return run


@router.post(
    "/scenario/run",
    response_model=ScenarioResponse,
    summary="Monte-Carlo scenario run",
)
def run_scenario(request: ScenarioRequest, state: PredictionDep) -> ScenarioResponse:
    timer = Timer()
    # Every field that changes the result must be in the key. `reanchor` and
    # `max_loans` were missing, so a reanchor=false request was served the
    # cached reanchor=true response — the payload contradicted the request.
    cache_key = sha256_obj({
        "scenario": request.scenario, "horizon": request.horizon,
        "n_paths": request.n_paths, "seed": request.seed,
        "segment_by": request.segment_by, "reanchor": request.reanchor,
        "max_loans": request.max_loans, "model_version": state.settings.model_version,
    })
    cached = state.scenario_cache.get(cache_key)
    if cached is not None:
        METRICS.increment("lpie_scenario_cache_hits_total")
        return ScenarioResponse(**cached)
    try:
        run = _run_scenario(
            state, request.scenario, request.n_paths, request.horizon,
            request.segment_by, request.seed, request.reanchor, request.max_loans,
        )
    except LPIEError:
        # Already a typed, correctly-statused error (404 unknown scenario, 503
        # hazard unavailable, 500 invariant violation). Re-wrapping these as
        # SCENARIO_NOT_FOUND told the caller their scenario name was wrong when
        # the real fault was elsewhere.
        raise
    except Exception as exc:
        log.exception("scenario.run_failed", scenario=request.scenario)
        raise ScenarioRunError(f"Scenario run failed: {exc}") from exc
    METRICS.increment("lpie_scenario_runs_total")
    resp = ScenarioResponse(
        scenario=run.scenario,
        assumptions=run.assumptions,
        summary=run.summary,
        segments=run.segments,
        reanchoring=run.reanchor,
        invariants_passed=True,
        cache_key=cache_key,
        elapsed_ms=round(timer.stop(), 2),
    )
    state.scenario_cache.put(cache_key, resp.model_dump())
    return resp


@router.post(
    "/scenario/custom",
    response_model=ScenarioResponse,
    summary="Custom user-defined macro shock",
)
def run_custom_scenario(request: CustomScenarioRequest, state: PredictionDep) -> ScenarioResponse:
    timer = Timer()
    from lpie.scenario.transmission import MacroShock
    shock = MacroShock(
        scenario_name=request.name,
        description=request.description,
        gdp_growth_pct=request.gdp_growth_pct,
        unemployment_rate_pct=request.unemployment_rate_pct,
        hpi_change_pct=request.hpi_change_pct,
        interest_rate_shock_bps=request.interest_rate_shock_bps,
        credit_spread_shock_bps=request.credit_spread_shock_bps,
        prepayment_cpr_assumption_pct=request.prepayment_cpr_assumption_pct,
        default_rate_multiplier=request.default_rate_multiplier,
        delinquency_rate_multiplier=request.delinquency_rate_multiplier,
        prepayment_rate_multiplier=request.prepayment_rate_multiplier,
    )
    runner = ScenarioRunner(state)
    # `run` takes the MacroShock as its first argument. Passing the shock under a
    # `_custom_shock` keyword it does not accept raised TypeError on every single
    # request, which the blanket handler below then reported as a 404 "unknown
    # scenario" — so the endpoint was 100% broken and said nothing useful.
    run = runner.run(
        shock,
        n_paths=request.n_paths,
        horizon=request.horizon,
        segment_by=request.segment_by,
        seed=request.seed,
        reanchor=request.reanchor,
    )
    return ScenarioResponse(
        scenario=run.scenario,
        assumptions=run.assumptions,
        summary=run.summary,
        segments=run.segments,
        reanchoring=run.reanchor,
        invariants_passed=True,
        cache_key=f"custom:{request.name}",
        elapsed_ms=round(timer.stop(), 2),
    )


@router.get(
    "/scenario/sensitivity",
    response_model=SensitivityResponse,
    summary="Tornado + exact Shapley macro attribution",
)
def sensitivity(
    state: PredictionDep,
    scenario: Annotated[str, Query()] = "Base",
    horizon: Annotated[int, Query(ge=1, le=60)] = 12,
    max_loans: Annotated[int, Query(ge=100, le=20000)] = 5000,
) -> SensitivityResponse:
    timer = Timer()
    runner = ScenarioRunner(state)
    try:
        result = runner.sensitivity(scenario=scenario, horizon=horizon, max_loans=max_loans)
    except LPIEError:
        raise
    except Exception as exc:
        log.exception("scenario.sensitivity_failed", scenario=scenario)
        raise ScenarioRunError(f"Sensitivity failed: {exc}") from exc
    return SensitivityResponse(
        scenario=scenario,
        horizon_months=horizon,
        n_loans=result.get("n_loans", 0),
        metric=result.get("metric", "default_rate"),
        tornado=result.get("tornado", {}),
        shapley=result.get("shapley"),
        elapsed_ms=round(timer.stop(), 2),
    )
