"""Artifact manager and model registry.

The API loads artifacts **once**, at startup, into a process-scoped manager with
a bounded LRU. Nothing in a request path ever touches disk to fetch a model, and
nothing ever trains during request handling.

Every artifact is registered with its provenance — train window, embargo, feature
hash, config hash, git SHA, data SHA256 — so a prediction can always answer
"which model, trained on what, from which data". A missing champion is a loud
503, never a silently substituted alternative.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from lpie.core.config import Settings, get_settings
from lpie.core.determinism import git_sha, sha256_file, sha256_obj
from lpie.core.exceptions import ArtifactNotFoundError, ModelNotLoadedError
from lpie.core.logging import get_logger
from lpie.core.timing import utcnow_iso
from lpie.data.app_store import AppStore, get_app_store

log = get_logger(__name__)

# Artifact names the serving layer knows about. `required_for` records which API
# capability degrades to 503 when the artifact is absent.
ARTIFACT_SPECS: dict[str, dict[str, Any]] = {
    "hazard": {"file": "hazard.joblib", "required_for": ["predict", "survival", "scenario"]},
    "heads": {"file": "heads.joblib", "required_for": ["predict"]},
    "stacks": {"file": "stacks.joblib", "required_for": ["predict"]},
    "calibrators": {"file": "calibrators.joblib", "required_for": ["predict"]},
    "thresholds": {"file": "thresholds.joblib", "required_for": ["predict", "anomaly"]},
    "exception": {"file": "exception.joblib", "required_for": ["predict", "anomaly"]},
    "anomaly": {"file": "anomaly.joblib", "required_for": ["anomaly"]},
    "conformal": {"file": "conformal.joblib", "required_for": ["explain"]},
    "feature_fit": {"file": "feature_fit.joblib", "required_for": ["predict"]},
    "survival_baselines": {"file": "survival_baselines.joblib", "required_for": ["survival"]},
    "shap_global": {"file": "shap_global.joblib", "required_for": ["explain"]},
    "evaluation": {"file": "evaluation.joblib", "required_for": []},
}

MANDATORY_FOR_HEALTHY = ("hazard", "heads", "calibrators", "thresholds", "feature_fit")


@dataclass
class ArtifactInfo:
    name: str
    path: Path
    exists: bool
    sha256: str | None = None
    size_bytes: int | None = None
    modified_at: str | None = None
    loaded: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "exists": self.exists,
            "sha256": self.sha256[:16] if self.sha256 else None,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at,
            "loaded": self.loaded,
            "error": self.error,
            "required_for": ARTIFACT_SPECS.get(self.name, {}).get("required_for", []),
        }


class ArtifactManager:
    """Process-scoped, thread-safe, bounded artifact cache."""

    def __init__(self, settings: Settings | None = None, app_store: AppStore | None = None) -> None:
        self.settings = settings or get_settings()
        self.models_dir: Path = self.settings.path("models_dir")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.app_store = app_store or get_app_store(self.settings)
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_size = int(self.settings.get("runtime.lru_model_cache_size", 24))
        self._lock = threading.RLock()
        self._load_errors: dict[str, str] = {}
        self._loaded_at: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    def path_for(self, name: str) -> Path:
        spec = ARTIFACT_SPECS.get(name)
        filename = spec["file"] if spec else f"{name}.joblib"
        return self.models_dir / filename

    def exists(self, name: str) -> bool:
        return self.path_for(name).exists()

    def save(self, name: str, obj: Any, *, metadata: dict[str, Any] | None = None) -> Path:
        path = self.path_for(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact": obj,
            "metadata": {
                "name": name,
                "saved_at": utcnow_iso(),
                "model_version": self.settings.model_version,
                "feature_version": self.settings.feature_version,
                "git_sha": git_sha(self.settings.root),
                **(metadata or {}),
            },
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        joblib.dump(payload, tmp, compress=3)
        tmp.replace(path)
        with self._lock:
            self._cache.pop(name, None)
        log.info("artifact.saved", name=name, path=str(path), bytes=path.stat().st_size)
        return path

    def load(self, name: str, *, required: bool = True) -> Any:
        """LRU-cached load. Never called from a hot request path after startup."""
        with self._lock:
            if name in self._cache:
                self._cache.move_to_end(name)
                return self._cache[name]

        path = self.path_for(name)
        if not path.exists():
            message = (
                f"Artifact '{name}' not found at {path.name}. "
                "Run `make train` to produce it."
            )
            self._load_errors[name] = "missing"
            if required:
                raise ArtifactNotFoundError(
                    message, details={"artifact": name, "path": str(path)}
                )
            return None

        try:
            payload = joblib.load(path)
        except Exception as exc:
            self._load_errors[name] = str(exc)
            log.error("artifact.load_failed", name=name, error=str(exc))
            if required:
                raise ArtifactNotFoundError(
                    f"Artifact '{name}' exists but could not be deserialised",
                    details={"artifact": name, "error": str(exc)},
                ) from exc
            return None

        obj = payload.get("artifact") if isinstance(payload, dict) else payload
        with self._lock:
            self._cache[name] = obj
            self._cache.move_to_end(name)
            while len(self._cache) > self._max_size:
                evicted, _ = self._cache.popitem(last=False)
                log.debug("artifact.evicted", name=evicted)
        self._load_errors.pop(name, None)
        self._loaded_at[name] = utcnow_iso()
        log.info("artifact.loaded", name=name)
        return obj

    def metadata(self, name: str) -> dict[str, Any]:
        path = self.path_for(name)
        if not path.exists():
            return {}
        try:
            payload = joblib.load(path)
        except Exception:
            return {}
        return payload.get("metadata", {}) if isinstance(payload, dict) else {}

    def require(self, name: str) -> Any:
        obj = self.load(name, required=False)
        if obj is None:
            raise ModelNotLoadedError(
                f"Champion artifact '{name}' is unavailable",
                details={
                    "artifact": name,
                    "path": str(self.path_for(name)),
                    "remedy": "run `make train`",
                    "required_for": ARTIFACT_SPECS.get(name, {}).get("required_for", []),
                },
            )
        return obj

    # ------------------------------------------------------------------ #
    def warm(self, names: list[str] | None = None) -> dict[str, bool]:
        """Preload at startup so no request ever pays a first-load cost."""
        wanted = names or list(ARTIFACT_SPECS)
        out: dict[str, bool] = {}
        for name in wanted:
            try:
                out[name] = self.load(name, required=False) is not None
            except Exception as exc:  # pragma: no cover - defensive
                self._load_errors[name] = str(exc)
                out[name] = False
        return out

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    # ------------------------------------------------------------------ #
    def inventory(self) -> list[ArtifactInfo]:
        out = []
        for name in ARTIFACT_SPECS:
            path = self.path_for(name)
            exists = path.exists()
            info = ArtifactInfo(
                name=name,
                path=path,
                exists=exists,
                loaded=name in self._cache,
                error=self._load_errors.get(name),
            )
            if exists:
                stat = path.stat()
                info.size_bytes = int(stat.st_size)
                info.modified_at = (
                    datetime.fromtimestamp(stat.st_mtime, tz=UTC)
                    .isoformat(timespec="seconds")
                    .replace("+00:00", "Z")
                )
            out.append(info)
        return out

    def missing_mandatory(self) -> list[str]:
        return [n for n in MANDATORY_FOR_HEALTHY if not self.exists(n)]

    def is_ready(self) -> bool:
        return not self.missing_mandatory()

    def health(self) -> dict[str, Any]:
        inventory = self.inventory()
        missing = self.missing_mandatory()
        return {
            "status": "ok" if not missing else "degraded",
            "models_dir": str(self.models_dir),
            "artifacts": [i.to_dict() for i in inventory],
            "n_present": sum(1 for i in inventory if i.exists),
            "n_expected": len(inventory),
            "missing_mandatory": missing,
            "cache_size": len(self._cache),
            "cache_capacity": self._max_size,
        }

    # ------------------------------------------------------------------ #
    # registry bridge
    # ------------------------------------------------------------------ #
    def register(
        self,
        *,
        head: str,
        algo: str,
        artifact_name: str,
        metrics: dict[str, Any],
        train_window: str,
        valid_window: str,
        embargo_months: int,
        feature_hash: str,
        data_sha256: str,
        n_features: int,
        model_version: str | None = None,
        status: str = "champion",
        notes: str = "",
    ) -> None:
        path = self.path_for(artifact_name)
        self.app_store.register_model(
            {
                "model_version": model_version or self.settings.model_version,
                "head": head,
                "algo": algo,
                "trained_at": utcnow_iso(),
                "train_window": train_window,
                "valid_window": valid_window,
                "embargo_months": int(embargo_months),
                "metrics": metrics,
                "feature_hash": feature_hash,
                "config_hash": self.config_hash(),
                "code_git_sha": git_sha(self.settings.root),
                "data_sha256": data_sha256,
                "artifact_path": str(path),
                "artifact_sha256": sha256_file(path) if path.exists() else None,
                "n_features": int(n_features),
                "status": status,
                "notes": notes,
            },
            promote=(status == "champion"),
        )

    def config_hash(self) -> str:
        return sha256_obj(self.settings.as_dict())

    def champion(self, head: str) -> dict[str, Any] | None:
        return self.app_store.get_champion(head)

    def list_models(self, head: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        return self.app_store.list_models(head=head, status=status)

    def loaded_model_versions(self) -> dict[str, str]:
        """head -> champion model_version, for the health payload."""
        return {
            entry["head"]: entry["model_version"]
            for entry in self.app_store.list_models(status="champion")
        }


_MANAGER: ArtifactManager | None = None
_LOCK = threading.Lock()


def get_artifact_manager(settings: Settings | None = None) -> ArtifactManager:
    global _MANAGER
    if _MANAGER is None:
        with _LOCK:
            if _MANAGER is None:
                _MANAGER = ArtifactManager(settings)
    return _MANAGER


def reset_artifact_manager() -> None:
    global _MANAGER
    with _LOCK:
        if _MANAGER is not None:
            _MANAGER.clear()
        _MANAGER = None
