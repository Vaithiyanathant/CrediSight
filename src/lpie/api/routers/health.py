"""Liveness, readiness and metrics.

`/healthz` reports real dependency state — DuckDB, SQLite, every artifact, the
feature store — and refuses to say `ok` when a mandatory artifact is missing.
A health check that always returns 200 tells an operator nothing.
"""

from __future__ import annotations

import time
from datetime import UTC
from typing import Any

from fastapi import APIRouter, Response, status

from lpie.api.deps import StateDep
from lpie.api.metrics import METRICS
from lpie.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


def _uptime(state: Any) -> float:
    from datetime import datetime

    started = datetime.fromisoformat(state.started_at.replace("Z", "+00:00"))
    return round((datetime.now(UTC) - started).total_seconds(), 3)


@router.get(
    "/healthz",
    response_model=HealthResponse,
    summary="Service health and readiness",
    description=(
        "Reports service status, database connectivity, artifact inventory, loaded model "
        "versions, application version, git SHA and the input-data SHA256. Returns 503 "
        "when a mandatory artifact or the feature store is missing — the service is up "
        "but cannot serve predictions, and the payload names exactly what is absent."
    ),
    responses={
        200: {"description": "Service is healthy and every mandatory dependency is present"},
        503: {"description": "Service is running but degraded; see `missing_artifacts`"},
    },
)
def healthz(state: StateDep, response: Response) -> HealthResponse:
    from lpie.data.ingest import data_sha256

    readiness = state.readiness()
    detail = readiness.detail

    try:
        sha = data_sha256()
    except Exception:
        sha = None

    if not readiness.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        service_status = "degraded"
    elif readiness.degraded_capabilities:
        service_status = "degraded"
    else:
        service_status = "ok"

    METRICS.set_gauge("lpie_ready", 1.0 if readiness.ready else 0.0)
    METRICS.set_gauge("lpie_missing_artifacts", float(len(readiness.missing_artifacts)))

    return HealthResponse(
        status=service_status,
        ready=readiness.ready,
        service=state.settings.get("project.name", "LPIE"),
        version=state.settings.get("project.model_version", "unknown"),
        model_version=state.settings.model_version,
        feature_version=state.settings.feature_version,
        git_sha=state.git_sha,
        data_sha256=sha,
        started_at=state.started_at,
        uptime_seconds=_uptime(state),
        database=detail.get("duckdb", {}),
        app_store=detail.get("sqlite", {}),
        artifacts=detail.get("artifacts", {}),
        loaded_model_versions=state.model_versions(),
        feature_store_months=int(detail.get("feature_store_months", 0)),
        degraded_capabilities=readiness.degraded_capabilities,
        missing_artifacts=readiness.missing_artifacts,
        startup_errors=state.startup_errors,
    )


@router.get(
    "/livez",
    summary="Liveness probe",
    description="Process is alive and serving. Does not check dependencies — that is /readyz.",
)
def livez() -> dict[str, Any]:
    return {"status": "alive", "timestamp": time.time()}


@router.get(
    "/readyz",
    summary="Readiness probe",
    description="Returns 503 until every mandatory artifact and the feature store are present.",
    responses={200: {"description": "Ready to serve"}, 503: {"description": "Not ready"}},
)
def readyz(state: StateDep, response: Response) -> dict[str, Any]:
    readiness = state.readiness()
    if not readiness.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "ready": readiness.ready,
        "missing_artifacts": readiness.missing_artifacts,
        "degraded_capabilities": readiness.degraded_capabilities,
    }


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description=(
        "Request counts and latency histogram by route, error counts, prediction and "
        "copilot counters, verifier failures, model-load failures, and cache statistics."
    ),
    response_class=Response,
    responses={200: {"content": {"text/plain": {}}, "description": "Prometheus text exposition"}},
)
def metrics(state: StateDep) -> Response:
    readiness = state.readiness()
    gauges = {
        "lpie_ready": 1.0 if readiness.ready else 0.0,
        "lpie_missing_artifacts": float(len(readiness.missing_artifacts)),
        "lpie_feature_store_months": float(len(state.feature_store_months())),
        "lpie_prediction_cache_size": float(state.prediction_cache.stats()["size"]),
        "lpie_prediction_cache_hits": float(state.prediction_cache.hits),
        "lpie_prediction_cache_misses": float(state.prediction_cache.misses),
        "lpie_scenario_cache_size": float(state.scenario_cache.stats()["size"]),
        "lpie_artifact_cache_size": float(state.artifacts.health()["cache_size"]),
        "lpie_startup_ms": float(state.startup_ms),
        "lpie_startup_errors": float(len(state.startup_errors)),
    }
    return Response(content=METRICS.render(gauges), media_type="text/plain; version=0.0.4")
