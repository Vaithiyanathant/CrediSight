"""Feature registry — the machine-readable feature contract.

Every feature must be declared here before it can enter a model matrix. The
declaration carries what the leakage tests need to check automatically:

* `temporal_offset` — months of look-ahead. Must be <= 0. Anything positive is a
  future read and fails CI.
* `source_columns` — must not intersect the target set.
* `allowed_heads` — per-head policy. `month_index` is legal in the hazard model
  (time is a modelled baseline there) and illegal in the direct-horizon heads,
  where a censored tail would teach "high month_index => no events".
* `leakage_risk` — declared risk level; anything above `low` needs a written
  justification, which the contract test asserts is non-empty.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Family = Literal[
    "static", "balance", "delinquency", "prepay", "cohort", "dq", "servicer",
    "temporal", "interactions",
]
LeakageRisk = Literal["none", "low", "medium", "high"]

ALL_HEADS = (
    "next_3m_delinquency", "next_6m_delinquency", "next_12m_default",
    "next_12m_prepayment", "next_state", "exception_required", "hazard", "anomaly",
)
DIRECT_HORIZON_HEADS = (
    "next_3m_delinquency", "next_6m_delinquency", "next_12m_default", "next_12m_prepayment",
)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: Family
    dtype: str
    source_columns: tuple[str, ...]
    description: str
    temporal_offset: int = 0          # <= 0 always; -1 means "uses month t-1"
    allowed_heads: tuple[str, ...] = ALL_HEADS
    leakage_risk: LeakageRisk = "none"
    justification: str = ""
    categorical: bool = False
    ordinal: bool = False
    winsorised: bool = False
    drift_psi: float | None = None
    drift_verdict: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["source_columns"] = list(self.source_columns)
        d["allowed_heads"] = list(self.allowed_heads)
        return d


class FeatureRegistry:
    """Ordered, deduplicated collection of feature declarations."""

    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> FeatureSpec:
        if spec.temporal_offset > 0:
            raise ValueError(
                f"Feature '{spec.name}' declares temporal_offset={spec.temporal_offset}. "
                "A positive offset is a future read and is banned by the feature contract."
            )
        if spec.leakage_risk in ("medium", "high") and not spec.justification:
            raise ValueError(
                f"Feature '{spec.name}' declares leakage_risk='{spec.leakage_risk}' "
                "without a written justification."
            )
        existing = self._specs.get(spec.name)
        if existing is not None and existing != spec:
            raise ValueError(f"Conflicting declarations for feature '{spec.name}'")
        self._specs[spec.name] = spec
        return spec

    def extend(self, specs: list[FeatureSpec]) -> None:
        for s in specs:
            self.register(s)

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def get(self, name: str) -> FeatureSpec | None:
        return self._specs.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._specs)

    @property
    def specs(self) -> list[FeatureSpec]:
        return list(self._specs.values())

    def by_family(self) -> dict[str, list[FeatureSpec]]:
        out: dict[str, list[FeatureSpec]] = {}
        for s in self._specs.values():
            out.setdefault(s.family, []).append(s)
        return out

    def family_of(self, name: str) -> str | None:
        spec = self._specs.get(name)
        return spec.family if spec else None

    def for_head(self, head: str) -> list[str]:
        """Feature allow-list for one head, in registration order."""
        return [s.name for s in self._specs.values() if head in s.allowed_heads]

    def categorical_features(self, head: str | None = None) -> list[str]:
        return [
            s.name for s in self._specs.values()
            if s.categorical and (head is None or head in s.allowed_heads)
        ]

    def to_dicts(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._specs.values()]

    def contract_hash(self) -> str:
        from lpie.core.determinism import sha256_obj

        return sha256_obj(
            [
                {
                    "name": s.name,
                    "family": s.family,
                    "source_columns": sorted(s.source_columns),
                    "temporal_offset": s.temporal_offset,
                    "allowed_heads": sorted(s.allowed_heads),
                }
                for s in self._specs.values()
            ]
        )

    # ------------------------------------------------------------------ #
    def validate_against_targets(self, target_columns: list[str]) -> list[str]:
        """Return a list of contract violations: features sourcing a target column."""
        targets = set(target_columns)
        problems: list[str] = []
        for s in self._specs.values():
            overlap = targets & set(s.source_columns)
            if overlap:
                problems.append(f"{s.name} sources target column(s) {sorted(overlap)}")
            if s.temporal_offset > 0:
                problems.append(f"{s.name} has temporal_offset {s.temporal_offset} > 0")
        return problems


def spec(
    name: str,
    family: Family,
    description: str,
    source_columns: tuple[str, ...] | list[str],
    *,
    dtype: str = "float64",
    temporal_offset: int = 0,
    allowed_heads: tuple[str, ...] = ALL_HEADS,
    leakage_risk: LeakageRisk = "none",
    justification: str = "",
    categorical: bool = False,
    ordinal: bool = False,
    winsorised: bool = False,
) -> FeatureSpec:
    """Terse constructor so family modules read as declarations, not boilerplate."""
    return FeatureSpec(
        name=name,
        family=family,
        dtype=dtype,
        source_columns=tuple(source_columns),
        description=description,
        temporal_offset=temporal_offset,
        allowed_heads=allowed_heads,
        leakage_risk=leakage_risk,
        justification=justification,
        categorical=categorical,
        ordinal=ordinal,
        winsorised=winsorised,
    )
