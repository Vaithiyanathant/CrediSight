"""Configuration loading.

`config/config.yaml` is the single source of truth for every path, window,
threshold and limit. Secrets never live there — they arrive through the
environment (see `.env.example`). Environment variables override YAML for the
handful of values a deployment legitimately needs to change.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml

_ENV_PREFIX = "LPIE_"


def project_root() -> Path:
    """Repository root: three parents up from this file (src/lpie/core/config.py)."""
    env = os.getenv(f"{_ENV_PREFIX}PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[3]


def _deep_get(cfg: dict, dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _deep_set(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _coerce(raw: str, template: Any) -> Any:
    if isinstance(template, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(template, int) and not isinstance(template, bool):
        return int(raw)
    if isinstance(template, float):
        return float(raw)
    if isinstance(template, list):
        return [p.strip() for p in raw.split(",") if p.strip()]
    return raw


# Environment overrides: LPIE_<NAME> -> dotted config path.
_ENV_OVERRIDES: dict[str, str] = {
    "LOG_LEVEL": "logging.level",
    "LOG_JSON": "logging.json",
    "API_HOST": "api.host",
    "API_PORT": "api.port",
    "API_ROOT_PATH": "api.root_path",
    "CORS_ORIGINS": "api.cors_origins",
    "API_KEY_ENABLED": "api.api_key_enabled",
    "MAX_PREDICT_ROWS": "api.max_predict_rows",
    "RATE_LIMIT_ENABLED": "api.rate_limit.enabled",
    "RATE_LIMIT_RPM": "api.rate_limit.requests_per_minute",
    "DUCKDB_MEMORY_LIMIT": "runtime.duckdb_memory_limit",
    "DUCKDB_THREADS": "runtime.duckdb_threads",
    "LRU_MODEL_CACHE_SIZE": "runtime.lru_model_cache_size",
    "EAGER_LOAD_FEATURES": "runtime.eager_load_features_on_startup",
    "FAIL_STARTUP_ON_MISSING_ARTIFACTS": "runtime.fail_startup_on_missing_artifacts",
    "MODEL_VERSION": "project.model_version",
    "SEED": "project.seed",
    "COPILOT_MODEL": "copilot.model",
    "COPILOT_TEMPERATURE": "copilot.temperature",
    "SCENARIO_MAX_PATHS": "scenario.max_n_paths",
    "DATASET_DIR": "paths.dataset_dir",
    "ARTIFACTS_DIR": "paths.artifacts_dir",
}


class Settings:
    """Immutable-by-convention settings object wrapping the merged config tree."""

    def __init__(self, cfg: dict[str, Any], root: Path, config_path: Path) -> None:
        self._cfg = cfg
        self.root = root
        self.config_path = config_path

    # ---------------- access ----------------
    def get(self, dotted: str, default: Any = None) -> Any:
        value = _deep_get(self._cfg, dotted)
        return default if value is None else value

    def require(self, dotted: str) -> Any:
        value = _deep_get(self._cfg, dotted)
        if value is None:
            raise KeyError(f"Missing required configuration key: {dotted}")
        return value

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.get(name, {}) or {})

    def as_dict(self) -> dict[str, Any]:
        return self._cfg

    # ---------------- paths ----------------
    def path(self, dotted: str) -> Path:
        """Resolve a `paths.*` entry to an absolute path under the project root."""
        raw = self.require(dotted if dotted.startswith("paths.") else f"paths.{dotted}")
        p = Path(raw)
        return p if p.is_absolute() else (self.root / p)

    def dataset_file(self, key: str) -> Path:
        return self.path("dataset_dir") / self.require(f"data.files.{key}")

    # ---------------- secrets ----------------
    @staticmethod
    def secret(name: str, default: str | None = None) -> str | None:
        return os.getenv(name, default)

    @property
    def anthropic_api_key(self) -> str | None:
        return os.getenv("ANTHROPIC_API_KEY") or None

    @property
    def groq_api_key(self) -> str | None:
        """Groq API key — takes priority over Anthropic when both are set."""
        return os.getenv("GROQ_API_KEY") or None

    @property
    def api_key(self) -> str | None:
        return os.getenv("LPIE_API_KEY") or None

    @property
    def api_key_enabled(self) -> bool:
        if os.getenv("LPIE_API_KEY_ENABLED") is not None:
            return os.getenv("LPIE_API_KEY_ENABLED", "").lower() in {"1", "true", "yes", "on"}
        return bool(self.get("api.api_key_enabled", False)) and bool(self.api_key)

    @property
    def model_version(self) -> str:
        return str(self.get("project.model_version", "lpie-v0.0.0"))

    @property
    def feature_version(self) -> str:
        return str(self.get("project.feature_version", "fv-0.0.0"))

    @property
    def seed(self) -> int:
        return int(self.get("project.seed", 20260828))

    def ensure_dirs(self) -> None:
        for key in (
            "artifacts_dir", "models_dir", "store_dir", "feature_store_dir",
            "mlruns_dir", "reports_dir", "logs_dir", "rag_index_dir",
        ):
            self.path(key).mkdir(parents=True, exist_ok=True)


def load_settings(config_path: Path | str | None = None) -> Settings:
    root = project_root()

    # Load .env from the project root before reading anything else so secrets
    # set there (GROQ_API_KEY, ANTHROPIC_API_KEY, LPIE_API_KEY …) are visible
    # to every os.getenv() call made anywhere in the process.
    try:
        from dotenv import load_dotenv
        env_file = root / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=False)
    except ImportError:
        pass  # python-dotenv is optional; CI sets vars directly

    path = Path(config_path) if config_path else Path(os.getenv(f"{_ENV_PREFIX}CONFIG", root / "config" / "config.yaml"))
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise FileNotFoundError(f"LPIE configuration not found at {path}")
    cfg = yaml.safe_load(path.read_text()) or {}

    for env_suffix, dotted in _ENV_OVERRIDES.items():
        raw = os.getenv(f"{_ENV_PREFIX}{env_suffix}")
        if raw is None:
            continue
        _deep_set(cfg, dotted, _coerce(raw, _deep_get(cfg, dotted)))

    return Settings(cfg, root, path)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton. Cheap to call from anywhere."""
    return load_settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
