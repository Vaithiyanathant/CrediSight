"""Model registry and metadata."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query

from lpie.api.deps import ArtifactsDep, StateDep
from lpie.api.schemas import ModelRegistryEntry, ModelsResponse

router = APIRouter(prefix="/api/v1/meta", tags=["metadata"])


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="Model registry",
    description=(
        "Every registered model with its full provenance: algorithm, training and "
        "validation windows, embargo length, metrics, feature hash, config hash, git SHA, "
        "data SHA256, artifact hash, and champion/candidate/archived status. This is what "
        "makes a prediction auditable — a score can always be traced to the exact model, "
        "trained on the exact data, from the exact code."
    ),
)
def list_models(
    state: StateDep,
    artifacts: ArtifactsDep,
    head: Annotated[str | None, Query(description="Filter by head name")] = None,
    status_filter: Annotated[
        str | None, Query(alias="status", description="champion | candidate | archived")
    ] = None,
) -> ModelsResponse:
    from lpie.data.ingest import data_sha256

    entries = artifacts.list_models(head=head, status=status_filter)
    champions = {
        e["head"]: e["model_version"] for e in artifacts.list_models(status="champion")
    }
    try:
        sha = data_sha256()
    except Exception:
        sha = None

    return ModelsResponse(
        model_version=state.settings.model_version,
        feature_version=state.settings.feature_version,
        feature_hash=state.registry.contract_hash(),
        config_hash=artifacts.config_hash(),
        git_sha=state.git_sha,
        data_sha256=sha,
        n_models=len(entries),
        champions=champions,
        models=[ModelRegistryEntry(**_clean(e)) for e in entries],
        artifacts=artifacts.health()["artifacts"],
    )


@router.get(
    "/config",
    summary="Effective runtime configuration",
    description=(
        "The non-secret configuration this process is actually running with — windows, "
        "thresholds, limits, heads, states and the legal-transition mask. Secrets never "
        "appear here; they live only in the environment."
    ),
)
def effective_config(state: StateDep) -> dict[str, Any]:
    cfg = state.settings.as_dict()
    return {
        "project": cfg.get("project", {}),
        "data": {k: v for k, v in (cfg.get("data") or {}).items() if k != "files"},
        "states": cfg.get("states", {}),
        "heads": cfg.get("heads", {}),
        "validation_split": cfg.get("validation_split", {}),
        "dq": cfg.get("dq", {}),
        "drift": cfg.get("drift", {}),
        "thresholds": cfg.get("thresholds", {}),
        "scenario": cfg.get("scenario", {}),
        "explain": cfg.get("explain", {}),
        "api_limits": {
            "max_predict_rows": cfg.get("api", {}).get("max_predict_rows"),
            "max_page_size": cfg.get("api", {}).get("max_page_size"),
            "max_request_bytes": cfg.get("api", {}).get("max_request_bytes"),
            "rate_limit": cfg.get("api", {}).get("rate_limit"),
        },
        "copilot": {
            k: v for k, v in (cfg.get("copilot") or {}).items()
            if k not in ("forbidden_phrases", "causal_markers")
        },
        "config_hash": state.artifacts.config_hash(),
    }


@router.get(
    "/features",
    summary="Feature contract",
    description=(
        "The machine-readable feature contract: every declared feature with its family, "
        "temporal offset, allowed heads, leakage risk and justification. Mirrors "
        "FEATURE_CONTRACT.md, which is generated from this same registry."
    ),
)
def feature_contract(state: StateDep) -> dict[str, Any]:
    registry = state.registry
    return {
        "feature_version": state.settings.feature_version,
        "contract_hash": registry.contract_hash(),
        "n_features": len(registry),
        "families": {family: len(specs) for family, specs in registry.by_family().items()},
        "per_head_counts": {
            head: len(registry.for_head(head))
            for head in ("next_3m_delinquency", "next_6m_delinquency", "next_12m_default",
                         "next_12m_prepayment", "next_state", "exception_required",
                         "hazard", "anomaly")
        },
        "categorical_features": registry.categorical_features(),
        "banned_columns": state.settings.require("data.banned_feature_columns"),
        "target_columns": state.settings.require("data.target_columns"),
        "declarations": registry.to_dicts(),
    }


@router.get(
    "/rules",
    summary="Validation rules",
    description="All 18 deterministic rules: 12 supplied with the pack plus 6 derived from profiling.",
)
def validation_rules(state: StateDep) -> dict[str, Any]:
    rules = state.validation_engine.rules
    return {
        "n_rules": len(rules),
        "n_supplied": sum(1 for r in rules if r.origin == "supplied"),
        "n_added": sum(1 for r in rules if r.origin != "supplied"),
        "dimensions": sorted({r.dimension for r in rules}),
        "rules": [
            {
                "rule_id": r.rule_id, "name": r.name, "description": r.description,
                "field": r.field_name, "severity": r.severity,
                "exception_type": r.exception_type, "condition": r.condition,
                "dimension": r.dimension, "origin": r.origin, "weight": r.weight,
            }
            for r in rules
        ],
    }


def _clean(entry: dict[str, Any]) -> dict[str, Any]:
    allowed = set(ModelRegistryEntry.model_fields)
    return {k: v for k, v in entry.items() if k in allowed}
