"""FastAPI dependencies.

Everything a router needs is resolved from the process-scoped `AppState`. No
dependency here opens a connection, loads a model, or reads a file — those are
startup concerns. A dependency that guards a capability raises the typed 503
rather than letting a router discover a missing artifact halfway through.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, Query, Request

from lpie.core.config import Settings, get_settings
from lpie.core.exceptions import ModelNotLoadedError
from lpie.data.app_store import AppStore
from lpie.data.duckdb_store import DuckDBStore
from lpie.models.registry import ArtifactManager
from lpie.serving.state import AppState


def get_app_state(request: Request) -> AppState:
    state = getattr(request.app.state, "lpie", None)
    if state is None:
        raise ModelNotLoadedError("Application state is not initialised")
    return state


StateDep = Annotated[AppState, Depends(get_app_state)]


def get_settings_dep() -> Settings:
    return get_settings()


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]


def get_duckdb(state: StateDep) -> DuckDBStore:
    return state.duckdb


DuckDBDep = Annotated[DuckDBStore, Depends(get_duckdb)]


def get_app_store_dep(state: StateDep) -> AppStore:
    return state.app_store


AppStoreDep = Annotated[AppStore, Depends(get_app_store_dep)]


def get_artifacts(state: StateDep) -> ArtifactManager:
    return state.artifacts


ArtifactsDep = Annotated[ArtifactManager, Depends(get_artifacts)]


def get_request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


RequestIdDep = Annotated[str | None, Depends(get_request_id)]


# --------------------------------------------------------------------------- #
# capability guards
# --------------------------------------------------------------------------- #
def require_prediction_stack(state: StateDep) -> AppState:
    if state.hazard is None or not state.hazard.is_loaded:
        raise ModelNotLoadedError(
            "The hazard model is not loaded, so predictions cannot be served.",
            details={"artifact": "hazard", "remedy": "run `make train`",
                     "missing": state.artifacts.missing_mandatory()},
        )
    if not state.heads:
        raise ModelNotLoadedError(
            "Direct-horizon head artifacts are not loaded.",
            details={"artifact": "heads", "remedy": "run `make train`"},
        )
    return state


PredictionDep = Annotated[AppState, Depends(require_prediction_stack)]


def require_feature_store(state: StateDep) -> AppState:
    if not state.feature_store_months():
        raise ModelNotLoadedError(
            "The Parquet feature store is empty.",
            details={"remedy": "run `make features`",
                     "path": str(state.settings.path("feature_store_dir"))},
        )
    return state


FeatureStoreDep = Annotated[AppState, Depends(require_feature_store)]


def require_anomaly(state: StateDep) -> AppState:
    if state.anomaly is None:
        raise ModelNotLoadedError(
            "The anomaly ensemble is not loaded.",
            details={"artifact": "anomaly", "remedy": "run `make train`"},
        )
    return state


AnomalyDep = Annotated[AppState, Depends(require_anomaly)]


def require_copilot(state: StateDep) -> AppState:
    if state.copilot is None:
        raise ModelNotLoadedError("The copilot service is not initialised")
    return state


CopilotDep = Annotated[AppState, Depends(require_copilot)]


# --------------------------------------------------------------------------- #
def demo_mode(
    demo: Annotated[
        int,
        Query(
            ge=0, le=1,
            description=(
                "Demo mode. Serves pre-warmed cached artifacts and smaller simulation "
                "budgets so no screen waits on computation. Results are real generated "
                "artifacts — nothing is hardcoded."
            ),
        ),
    ] = 0,
) -> bool:
    return bool(demo)


DemoDep = Annotated[bool, Depends(demo_mode)]


def pagination(
    limit: Annotated[int, Query(ge=1, le=1000, description="Maximum rows to return")] = 50,
    offset: Annotated[int, Query(ge=0, le=1_000_000, description="Rows to skip")] = 0,
) -> dict[str, int]:
    return {"limit": limit, "offset": offset}


PaginationDep = Annotated[dict[str, int], Depends(pagination)]


def sanitise(value: str | None, *, max_length: int = 128) -> str | None:
    """Strip control characters from a user-controlled string before it is logged
    or echoed. Log injection through a scenario name is a small hole, but a hole."""
    if value is None:
        return None
    cleaned = "".join(ch for ch in str(value) if ch.isprintable())
    return cleaned[:max_length].strip()


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(int(value), upper))


def as_dict(payload: Any) -> dict[str, Any]:
    return payload if isinstance(payload, dict) else {}
