"""FastAPI application factory and lifespan."""
from __future__ import annotations

try:
    import torch  # noqa: F401 -- ensure torch is loaded before joblib loads anomaly artifacts
except ImportError:
    pass
import contextlib
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lpie.api.exceptions import register_exception_handlers
from lpie.api.metrics import METRICS
from lpie.api.middleware import (
    APIKeyMiddleware,
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
)
from lpie.api.routers import (
    anomaly,
    copilot,
    dq,
    drift,
    explain,
    health,
    meta,
    portfolio,
    prediction,
    profile,
    reviewer,
    scenario,
    submission,
    survival,
    validation,
)
from lpie.core.config import Settings, get_settings
from lpie.core.logging import configure_logging, get_logger
from lpie.serving.state import init_state, reset_state

log = get_logger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    s: Settings = app.state.settings
    configure_logging(
        level=str(s.get("logging.level", "INFO")),
        json_logs=bool(s.get("logging.json", True)),
    )
    log.info("startup.begin", version=s.model_version)
    try:
        state = init_state(s)
        app.state.lpie = state
        METRICS.set_gauge("lpie_startup_ms", float(state.startup_ms))
        log.info("startup.complete", ready=state.readiness().ready, ms=state.startup_ms)
    except Exception as exc:
        log.error("startup.failed", error=str(exc))
    yield
    log.info("shutdown.begin")
    reset_state()
    log.info("shutdown.complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    app = FastAPI(
        title=str(s.get("project.name", "LPIE")),
        version=str(s.get("project.model_version", "lpie-v1.0.0")),
        description="Loan Performance Intelligence Engine. Governed multi-head ML system.",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = s

    origins = s.get("api.cors_origins", ["http://localhost:3000", "http://localhost:5173"])
    if isinstance(origins, str):
        origins = [o.strip() for o in origins.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=bool(s.get("api.cors_allow_credentials", False)),
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time-Ms", "X-Model-Version"],
    )

    max_bytes = int(s.get("api.max_request_bytes", 16 * 1024 * 1024))
    app.add_middleware(RequestContextMiddleware, settings=s)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=max_bytes)

    rate_cfg = s.section("api").get("rate_limit", {})
    if rate_cfg and rate_cfg.get("enabled", False):
        exempt = set(s.get("api.public_paths", ["/healthz", "/livez", "/readyz", "/metrics", "/docs", "/redoc", "/openapi.json"]))
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=int(rate_cfg.get("requests_per_minute", 300)),
            burst=int(rate_cfg.get("burst", 60)),
            exempt_paths=exempt,
        )

    if bool(s.get("api.api_key_enabled", False)):
        api_key = os.environ.get("LPIE_API_KEY", "")
        if api_key:
            public_paths = list(s.get("api.public_paths", ["/healthz", "/livez", "/readyz", "/metrics", "/docs", "/redoc", "/openapi.json"]))
            app.add_middleware(
                APIKeyMiddleware,
                api_key=api_key,
                header=str(s.get("api.api_key_header", "X-API-Key")),
                public_paths=public_paths,
            )

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(meta.router)
    app.include_router(profile.router)
    app.include_router(validation.router)
    app.include_router(drift.router)
    app.include_router(dq.router)
    app.include_router(prediction.router)
    app.include_router(portfolio.router)
    app.include_router(survival.router)
    app.include_router(scenario.router)
    app.include_router(anomaly.router)
    app.include_router(reviewer.router)
    app.include_router(explain.router)
    app.include_router(copilot.router)
    app.include_router(submission.router)
    return app


app = create_app()
