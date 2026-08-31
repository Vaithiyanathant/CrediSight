"""Application state — loaded once at startup, shared by every request.

Nothing here is constructed per request. Models, the feature store handle, the
DuckDB and SQLite connections, the RAG index and the fitted feature parameters
all live for the process lifetime. A request path that touched disk for a model
would dominate latency on a small container and would make throughput a function
of page cache rather than of the work being done.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from lpie.copilot.rag import RAGIndex
from lpie.copilot.service import CopilotService
from lpie.core.config import Settings, get_settings
from lpie.core.determinism import git_sha, seed_everything
from lpie.core.logging import get_logger
from lpie.core.timing import Timer, utcnow_iso
from lpie.data.app_store import AppStore, get_app_store
from lpie.data.duckdb_store import DuckDBStore, get_store
from lpie.features.builder import FeatureBuilder, FeatureFitParams, build_registry, feature_store_months
from lpie.models.hazard import HazardModel
from lpie.models.registry import ArtifactManager, get_artifact_manager
from lpie.validation.engine import ValidationEngine

log = get_logger(__name__)


@dataclass
class ReadinessReport:
    ready: bool
    degraded_capabilities: list[str] = field(default_factory=list)
    missing_artifacts: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


class BoundedCache:
    """Small thread-safe LRU used for per-request scoring results."""

    def __init__(self, capacity: int = 1024) -> None:
        self.capacity = max(int(capacity), 1)
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "size": len(self._data),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 6) if total else None,
        }


class AppState:
    """The process-scoped container the API depends on."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.started_at = utcnow_iso()
        self.git_sha = git_sha(self.settings.root)
        self.registry = build_registry()

        self.duckdb: DuckDBStore = get_store(self.settings)
        self.app_store: AppStore = get_app_store(self.settings)
        self.artifacts: ArtifactManager = get_artifact_manager(self.settings)

        self.validation_engine = ValidationEngine(self.settings)
        self.feature_builder: FeatureBuilder | None = None
        self.hazard: HazardModel | None = None
        self.heads: dict[str, Any] = {}
        self.stacks: dict[str, Any] = {}
        self.calibrators: dict[str, Any] = {}
        self.thresholds: dict[str, Any] = {}
        self.exception_head: Any = None
        self.anomaly: Any = None
        self.conformal: dict[str, Any] = {}
        self.survival_baselines: dict[str, Any] = {}
        self.shap_global: dict[str, Any] = {}
        self.evaluation: dict[str, Any] = {}
        self.feature_fit: FeatureFitParams | None = None

        self.rag = RAGIndex(self.settings)
        self.copilot: CopilotService | None = None

        self.prediction_cache = BoundedCache(int(self.settings.get("runtime.prediction_cache_size", 4096)))
        self.scenario_cache = BoundedCache(64)
        self._panel_cache: pd.DataFrame | None = None
        self._static_cache: pd.DataFrame | None = None
        self._servicer_cache: pd.DataFrame | None = None
        self._macro_cache: pd.DataFrame | None = None
        self._lock = threading.RLock()
        self.startup_errors: list[dict[str, str]] = []
        self.startup_ms: float = 0.0

    # ------------------------------------------------------------------ #
    def startup(self) -> ReadinessReport:
        timer = Timer()
        seed_everything(self.settings.seed)
        self.settings.ensure_dirs()

        self._safely("duckdb", self.duckdb.initialise)
        self._safely("sqlite", self.app_store.initialise)
        self._safely("artifacts", self._load_artifacts)
        self._safely("rag", self._build_rag)

        self.copilot = CopilotService(self.settings, rag=self.rag, store=self.duckdb)

        if bool(self.settings.get("runtime.eager_load_features_on_startup", False)):
            self._safely("feature_store", lambda: self.panel())

        self.startup_ms = round(timer.stop(), 2)
        report = self.readiness()
        log.info(
            "startup.complete",
            ready=report.ready,
            missing_artifacts=report.missing_artifacts,
            elapsed_ms=self.startup_ms,
        )
        if not report.ready and bool(self.settings.get("runtime.fail_startup_on_missing_artifacts", False)):
            from lpie.core.exceptions import ArtifactNotFoundError

            raise ArtifactNotFoundError(
                "Startup aborted: mandatory artifacts are missing and "
                "fail_startup_on_missing_artifacts is enabled.",
                details={"missing": report.missing_artifacts},
            )
        return report

    def _safely(self, name: str, fn) -> None:
        try:
            fn()
        except Exception as exc:  # startup must degrade, not crash
            self.startup_errors.append({"component": name, "error": str(exc)})
            log.error("startup.component_failed", component=name, error=str(exc))

    def _load_artifacts(self) -> None:
        loaded = self.artifacts.warm()
        log.info("startup.artifacts", loaded=loaded)

        self.feature_fit = self.artifacts.load("feature_fit", required=False)
        if self.feature_fit is not None and not isinstance(self.feature_fit, FeatureFitParams):
            self.feature_fit = FeatureFitParams.from_dict(self.feature_fit)

        self.feature_builder = FeatureBuilder(
            self.settings, engine=self.validation_engine, fit_params=self.feature_fit
        )

        hazard_artifact = self.artifacts.load("hazard", required=False)
        if hazard_artifact is not None:
            self.hazard = HazardModel(self.settings, artifact=hazard_artifact)

        self.heads = self.artifacts.load("heads", required=False) or {}
        self.stacks = self.artifacts.load("stacks", required=False) or {}
        self.calibrators = self.artifacts.load("calibrators", required=False) or {}
        self.thresholds = self.artifacts.load("thresholds", required=False) or {}
        self.exception_head = self.artifacts.load("exception", required=False)
        self.anomaly = self.artifacts.load("anomaly", required=False)
        self.conformal = self.artifacts.load("conformal", required=False) or {}
        self.survival_baselines = self.artifacts.load("survival_baselines", required=False) or {}
        self.shap_global = self.artifacts.load("shap_global", required=False) or {}
        self.evaluation = self.artifacts.load("evaluation", required=False) or {}

        if self.feature_fit is not None and self.feature_fit.vocabulary:
            self.validation_engine._vocabulary = {
                k: set(v) for k, v in self.feature_fit.vocabulary.items()
            }

    def _build_rag(self) -> None:
        prefer = bool(self.settings.get("copilot.rag.prefer_embeddings", False))
        self.rag.build(prefer_embeddings=prefer)

    # ------------------------------------------------------------------ #
    # lazily-loaded reference data
    # ------------------------------------------------------------------ #
    def panel(self) -> pd.DataFrame:
        with self._lock:
            if self._panel_cache is None:
                from lpie.data.ingest import load_panel

                self._panel_cache = load_panel(self.settings)
            return self._panel_cache

    def static(self) -> pd.DataFrame:
        with self._lock:
            if self._static_cache is None:
                from lpie.data.ingest import load_static

                self._static_cache = load_static(self.settings)
            return self._static_cache

    def servicer(self) -> pd.DataFrame:
        with self._lock:
            if self._servicer_cache is None:
                from lpie.data.ingest import load_servicer

                self._servicer_cache = load_servicer(self.settings)
            return self._servicer_cache

    def macro(self) -> pd.DataFrame:
        with self._lock:
            if self._macro_cache is None:
                from lpie.data.ingest import load_macro

                self._macro_cache = load_macro(self.settings)
            return self._macro_cache

    def features(
        self, months: list[int] | None = None, loan_ids: list[str] | None = None
    ) -> pd.DataFrame:
        """Partition-pruned read from the Parquet feature store."""
        from lpie.features.builder import read_feature_store

        return read_feature_store(months=months, loan_ids=loan_ids, settings=self.settings)

    def feature_store_months(self) -> list[int]:
        return feature_store_months(self.settings)

    # ------------------------------------------------------------------ #
    def readiness(self) -> ReadinessReport:
        missing = self.artifacts.missing_mandatory()
        degraded: list[str] = []
        if not self.hazard or not self.hazard.is_loaded:
            degraded.extend(["predict", "survival", "scenario"])
        if not self.heads:
            degraded.append("predict")
        if not self.anomaly:
            degraded.append("anomaly")
        if not self.shap_global and not self.heads:
            degraded.append("explain")
        if not self.feature_store_months():
            degraded.append("feature_store")
        if not self.rag.is_built:
            degraded.append("copilot_rag")

        return ReadinessReport(
            ready=not missing and bool(self.feature_store_months()),
            degraded_capabilities=sorted(set(degraded)),
            missing_artifacts=missing,
            detail={
                "duckdb": self.duckdb.health(),
                "sqlite": self.app_store.health(),
                "artifacts": self.artifacts.health(),
                "feature_store_months": len(self.feature_store_months()),
                "startup_errors": self.startup_errors,
            },
        )

    def model_versions(self) -> dict[str, str]:
        versions = self.artifacts.loaded_model_versions()
        if not versions:
            versions = {"default": self.settings.model_version}
        return versions

    def shutdown(self) -> None:
        self.prediction_cache.clear()
        self.scenario_cache.clear()
        self.artifacts.clear()
        self.duckdb.close()
        self.app_store.close()
        self._panel_cache = None
        log.info("shutdown.complete")


_STATE: AppState | None = None
_STATE_LOCK = threading.Lock()


def get_state() -> AppState:
    global _STATE
    if _STATE is None:
        raise RuntimeError("Application state has not been initialised")
    return _STATE


def init_state(settings: Settings | None = None) -> AppState:
    global _STATE
    with _STATE_LOCK:
        _STATE = AppState(settings)
        _STATE.startup()
    return _STATE


def reset_state() -> None:
    global _STATE
    with _STATE_LOCK:
        if _STATE is not None:
            _STATE.shutdown()
        _STATE = None
