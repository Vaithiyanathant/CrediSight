"""Discrete-time competing-risk hazard model — the core of the system.

Rather than four independent binary classifiers we model the monthly transition
distribution

    h_k(t) = P( state_{t+1} = k | state_t, X_t ),  k in the 7-state space

with a multiclass LightGBM over the person-month panel. This *is* a discrete-time
multi-state hazard model, with the GBDT replacing the usual multinomial-logit
link. Two things make it the right choice for this data:

**Legal-transition masking.** The measured transition matrix is sparse and
structurally constrained (`Current` can only reach `{Current, 30DPD, Prepaid}`).
Impossible transitions are zeroed *before* renormalisation. This is hard domain
knowledge, not a statistical regularity, and encoding it strictly improves
calibration.

**Censoring disappears.** A one-month-ahead label exists for every row up to
month 35, so there is no long-horizon censoring problem at all. This is the deep
reason the hazard formulation dominates here: it converts a censored 12-month
problem into an uncensored 1-month problem, and the 12-month probabilities are
then *derived* by compounding, which is exactly what makes them mutually
consistent with the survival curves and the scenario simulation.

Horizon probabilities come from compounding the transition matrix forward, and
the cause-specific CIFs satisfy `sum_j CIF_j(m) + S(m) = 1` by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from lpie.core.config import Settings, get_settings
from lpie.core.exceptions import ModelNotLoadedError, PredictionError
from lpie.core.logging import get_logger

log = get_logger(__name__)

STATUS_FEATURE = "current_status"
EVENT_STATES = ("Default", "Prepaid", "Closed")
ACTIVE_STATES = ("Current", "30DPD", "60DPD", "90DPD")
PROB_TOLERANCE = 1e-6


def build_legal_mask(states: list[str], legal: dict[str, list[str]]) -> np.ndarray:
    """(K, K) boolean mask. mask[i, j] = transition i -> j is possible."""
    index = {s: i for i, s in enumerate(states)}
    mask = np.zeros((len(states), len(states)), dtype=bool)
    for src, destinations in legal.items():
        if src not in index:
            continue
        for dst in destinations:
            if dst in index:
                mask[index[src], index[dst]] = True
    # A state with no declared legal destination would produce an all-zero row
    # and a division by zero on renormalisation. Self-transition is the only
    # safe default and is what an unobserved state means in practice.
    for i, s in enumerate(states):
        if not mask[i].any():
            mask[i, i] = True
            log.warning("hazard.no_legal_transitions", state=s)
    return mask


def apply_legal_mask(
    probs: np.ndarray, from_state_idx: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Zero illegal destinations, then renormalise over the legal set."""
    allowed = mask[from_state_idx]
    masked = np.where(allowed, probs, 0.0)
    totals = masked.sum(axis=1, keepdims=True)
    # If the model put *all* its mass on illegal destinations the row is
    # unusable; fall back to a uniform distribution over the legal set rather
    # than emitting NaN into a risk report.
    degenerate = (totals <= PROB_TOLERANCE).ravel()
    if degenerate.any():
        uniform = allowed[degenerate].astype("float64")
        masked[degenerate] = uniform / uniform.sum(axis=1, keepdims=True)
        totals[degenerate] = 1.0
    return masked / totals


@dataclass
class HazardArtifact:
    """Everything needed to reproduce a hazard prediction, and nothing else."""

    booster: Any
    states: list[str]
    feature_names: list[str]
    categorical_features: list[str]
    legal_mask: np.ndarray
    category_levels: dict[str, list[str]] = field(default_factory=dict)
    model_version: str = ""
    trained_at: str = ""
    train_window: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    class_weights: dict[str, float] = field(default_factory=dict)


class HazardModel:
    """Fit / predict / propagate for the competing-risk hazard core."""

    def __init__(self, settings: Settings | None = None, artifact: HazardArtifact | None = None) -> None:
        self.settings = settings or get_settings()
        self.states: list[str] = list(self.settings.require("states.all"))
        self.state_index = {s: i for i, s in enumerate(self.states)}
        self.legal_mask = build_legal_mask(self.states, self.settings.require("states.legal_transitions"))
        self.artifact = artifact
        self.terminal_states = set(self.settings.require("states.terminal"))

    # ------------------------------------------------------------------ #
    @property
    def is_loaded(self) -> bool:
        return self.artifact is not None and self.artifact.booster is not None

    def _require_loaded(self) -> HazardArtifact:
        if not self.is_loaded:
            raise ModelNotLoadedError(
                "Hazard model artifact is not loaded. Run `make train` to produce it.",
                details={"artifact": "hazard"},
            )
        return self.artifact  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # fitting
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: pd.DataFrame,
        y_next_state: pd.Series,
        *,
        feature_names: list[str],
        categorical_features: list[str],
        params: dict[str, Any] | None = None,
        num_boost_round: int = 400,
        valid_sets: list[tuple[pd.DataFrame, pd.Series]] | None = None,
        early_stopping_rounds: int = 50,
        class_weight: str | None = "balanced",
    ) -> HazardArtifact:
        import lightgbm as lgb

        y = y_next_state.astype("object").map(self.state_index)
        keep = y.notna()
        if keep.sum() == 0:
            raise PredictionError("No labelled rows available to fit the hazard model")

        X_fit = self._prepare(X.loc[keep], feature_names, categorical_features)
        y_fit = y[keep].astype("int32").to_numpy()

        weights = None
        weight_map: dict[str, float] = {}
        if class_weight == "balanced":
            counts = np.bincount(y_fit, minlength=len(self.states)).astype("float64")
            with np.errstate(divide="ignore"):
                inv = np.where(counts > 0, len(y_fit) / (len(self.states) * counts), 0.0)
            weights = inv[y_fit]
            weight_map = {s: round(float(inv[i]), 6) for i, s in enumerate(self.states)}

        base_params = {
            "objective": "multiclass",
            "num_class": len(self.states),
            "metric": "multi_logloss",
            "learning_rate": 0.06,
            "num_leaves": 63,
            "min_child_samples": 100,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbose": -1,
            "seed": self.settings.seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": int(self.settings.get("runtime.duckdb_threads", 4)),
        }
        base_params.update(params or {})

        cats = [c for c in categorical_features if c in X_fit.columns]
        train_set = lgb.Dataset(X_fit, label=y_fit, weight=weights, categorical_feature=cats, free_raw_data=False)

        valid = []
        valid_names = []
        for i, (Xv, yv) in enumerate(valid_sets or []):
            yv_mapped = yv.astype("object").map(self.state_index)
            vkeep = yv_mapped.notna()
            if vkeep.sum() == 0:
                continue
            valid.append(
                lgb.Dataset(
                    self._prepare(Xv.loc[vkeep], feature_names, categorical_features),
                    label=yv_mapped[vkeep].astype("int32").to_numpy(),
                    categorical_feature=cats,
                    reference=train_set,
                    free_raw_data=False,
                )
            )
            valid_names.append(f"valid_{i}")

        callbacks = [lgb.log_evaluation(period=0)]
        if valid:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

        booster = lgb.train(
            base_params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=valid or None,
            valid_names=valid_names or None,
            callbacks=callbacks,
        )

        self.artifact = HazardArtifact(
            booster=booster,
            states=list(self.states),
            feature_names=list(feature_names),
            categorical_features=list(cats),
            legal_mask=self.legal_mask,
            category_levels={c: sorted(X_fit[c].cat.categories.tolist()) for c in cats},
            model_version=self.settings.model_version,
            class_weights=weight_map,
        )
        return self.artifact

    def _prepare(
        self, X: pd.DataFrame, feature_names: list[str], categorical_features: list[str]
    ) -> pd.DataFrame:
        """Select declared features and pin categorical dtypes.

        Categories are pinned to the *fitted* level set at predict time so an
        unseen level maps to NaN (which LightGBM handles) instead of silently
        shifting every other level's integer code.
        """
        missing = [c for c in feature_names if c not in X.columns]
        if missing:
            raise PredictionError(
                f"Feature matrix is missing {len(missing)} declared feature(s)",
                details={"missing": missing[:20]},
            )
        out = X[feature_names].copy()
        levels = self.artifact.category_levels if self.artifact else {}
        for c in categorical_features:
            if c not in out.columns:
                continue
            if c in levels and levels[c]:
                out[c] = pd.Categorical(out[c].astype("object"), categories=levels[c])
            else:
                out[c] = pd.Categorical(out[c].astype("object"))
        for c in out.columns:
            if c not in categorical_features and not pd.api.types.is_numeric_dtype(out[c]):
                out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    # ------------------------------------------------------------------ #
    # prediction
    # ------------------------------------------------------------------ #
    def predict_transition(
        self, X: pd.DataFrame, current_status: pd.Series | None = None
    ) -> np.ndarray:
        """(n, K) next-state distribution, legal-masked and renormalised."""
        art = self._require_loaded()
        status = current_status if current_status is not None else X.get(STATUS_FEATURE)
        if status is None:
            raise PredictionError(
                "current_status is required to apply the legal-transition mask",
                details={"expected_column": STATUS_FEATURE},
            )
        prepared = self._prepare(X, art.feature_names, art.categorical_features)
        raw = art.booster.predict(prepared, num_iteration=art.booster.best_iteration or None)
        raw = np.asarray(raw, dtype="float64").reshape(len(prepared), len(art.states))

        from_idx = (
            pd.Series(status).astype("object").map(self.state_index).fillna(0).astype("int32").to_numpy()
        )
        return apply_legal_mask(raw, from_idx, art.legal_mask)

    def transition_matrices(self, X: pd.DataFrame) -> np.ndarray:
        """(n, K, K) per-loan transition matrix.

        Built by substituting each of the K states into the status features and
        re-predicting, which yields P(next | from=s, X) for every s. That is what
        forward propagation needs: a loan currently `Current` still needs its
        `30DPD -> ?` row to know where it goes in month three.
        """
        art = self._require_loaded()
        n, K = len(X), len(art.states)
        out = np.empty((n, K, K), dtype="float64")
        for i, state in enumerate(art.states):
            substituted = self._substitute_state(X, state)
            probs = self.predict_transition(substituted, pd.Series([state] * n, index=X.index))
            out[:, i, :] = probs
        return out

    def _substitute_state(self, X: pd.DataFrame, state: str) -> pd.DataFrame:
        """Set every status-derived feature to be consistent with `state`.

        Substituting `current_status` alone would leave `dpd` and
        `status_severity` contradicting it, and the model would be scored on a
        row that could not exist. Consistency here is what makes the propagated
        matrix meaningful rather than an artefact.
        """
        from lpie.features.families.delinquency import STATUS_SEVERITY, STATUS_TO_DPD

        out = X.copy()
        if STATUS_FEATURE in out.columns:
            out[STATUS_FEATURE] = state
        if "status_severity" in out.columns:
            out["status_severity"] = float(STATUS_SEVERITY.get(state, 0))
        if "dpd" in out.columns and state in STATUS_TO_DPD:
            out["dpd"] = float(STATUS_TO_DPD[state])
        if "is_terminal" in out.columns:
            out["is_terminal"] = 1.0 if state in self.terminal_states else 0.0
        return out

    # ------------------------------------------------------------------ #
    # propagation
    # ------------------------------------------------------------------ #
    def absorbing_matrices(self, M: np.ndarray) -> np.ndarray:
        """Copy of M with Default / Prepaid / Closed made fully absorbing.

        This is the standard cause-specific first-passage construction. Without
        it a loan that goes `Default -> Closed` would be counted twice, and the
        CIFs would not sum with survival to one.
        """
        A = M.copy()
        for state in EVENT_STATES:
            if state not in self.state_index:
                continue
            i = self.state_index[state]
            A[:, i, :] = 0.0
            A[:, i, i] = 1.0
        return A

    def propagate(
        self,
        M: np.ndarray,
        start_state: np.ndarray,
        horizon: int = 24,
        *,
        absorbing_for_cif: bool = True,
    ) -> dict[str, np.ndarray]:
        """Forward-propagate the chain and derive survival + cause-specific CIFs.

        Returns arrays shaped (n, horizon+1) for the curves and
        (n, horizon+1, K) for the occupancy, with index 0 = "now".
        """
        n, K, _ = M.shape
        if len(start_state) != n:
            raise PredictionError("start_state length does not match the transition matrix batch")

        chain = self.absorbing_matrices(M) if absorbing_for_cif else M
        occupancy = np.zeros((n, horizon + 1, K), dtype="float64")
        occupancy[:, 0, :] = np.eye(K, dtype="float64")[start_state]

        for m in range(1, horizon + 1):
            occupancy[:, m, :] = np.einsum("nk,nkj->nj", occupancy[:, m - 1, :], chain)

        active_idx = [self.state_index[s] for s in ACTIVE_STATES if s in self.state_index]
        survival = occupancy[:, :, active_idx].sum(axis=2)

        cifs = {
            f"cif_{state.lower()}": occupancy[:, :, self.state_index[state]]
            for state in EVENT_STATES
            if state in self.state_index
        }
        # A loan that starts *in* an event state has already experienced it; its
        # CIF is 1 from month 0 and its survival is 0. That is correct, and it is
        # the same fact the absorbing-state gate enforces downstream.
        total = survival + sum(cifs.values())
        max_error = float(np.abs(total - 1.0).max()) if n else 0.0

        return {
            "occupancy": occupancy,
            "survival": survival,
            **cifs,
            "conservation_max_error": max_error,
            "states": np.array(self.states, dtype=object),
        }

    def state_occupancy(
        self, M: np.ndarray, start_state: np.ndarray, horizon: int = 24
    ) -> np.ndarray:
        """(n, horizon+1, K) occupancy under the *real* chain.

        Uses the unmodified matrix so `Default -> Closed` liquidation flows are
        visible. This is the stacked-area projection; it is a different object
        from the CIF construction above and both are correct for their purpose.
        """
        return self.propagate(M, start_state, horizon, absorbing_for_cif=False)["occupancy"]

    # ------------------------------------------------------------------ #
    # derived horizon probabilities
    # ------------------------------------------------------------------ #
    def horizon_probabilities(
        self, M: np.ndarray, start_state: np.ndarray, horizons: dict[str, int]
    ) -> dict[str, np.ndarray]:
        """Compound the chain into the four submission horizons.

        P(delinquent within m) is one minus the probability of never leaving the
        non-delinquent set, computed on a chain where any delinquent state is
        made absorbing — otherwise a loan that goes delinquent and cures would
        not count as having been delinquent, which is not what the label means.
        """
        max_h = max(horizons.values()) if horizons else 0
        result: dict[str, np.ndarray] = {}

        if max_h:
            propagated = self.propagate(M, start_state, max_h, absorbing_for_cif=True)
            for name, h in horizons.items():
                if "default" in name:
                    result[name] = propagated["cif_default"][:, h]
                elif "prepay" in name:
                    result[name] = propagated["cif_prepaid"][:, h]

            delinq = self._delinquency_absorbing(M)
            occ = np.zeros((M.shape[0], max_h + 1, M.shape[1]), dtype="float64")
            occ[:, 0, :] = np.eye(M.shape[1], dtype="float64")[start_state]
            for m in range(1, max_h + 1):
                occ[:, m, :] = np.einsum("nk,nkj->nj", occ[:, m - 1, :], delinq)
            delinq_idx = [self.state_index[s] for s in ("30DPD", "60DPD", "90DPD", "Default")
                          if s in self.state_index]
            ever_delinq = occ[:, :, delinq_idx].sum(axis=2)
            for name, h in horizons.items():
                if "delinquency" in name:
                    result[name] = ever_delinq[:, h]

        return {k: np.clip(v, 0.0, 1.0) for k, v in result.items()}

    def _delinquency_absorbing(self, M: np.ndarray) -> np.ndarray:
        A = M.copy()
        for state in ("30DPD", "60DPD", "90DPD", "Default", "Prepaid", "Closed"):
            if state not in self.state_index:
                continue
            i = self.state_index[state]
            A[:, i, :] = 0.0
            A[:, i, i] = 1.0
        return A

    # ------------------------------------------------------------------ #
    def validate_invariants(self, probs: np.ndarray, *, tolerance: float = 1e-6) -> dict[str, Any]:
        """Assertions the scenario engine and tests both rely on."""
        row_sums = probs.sum(axis=-1)
        return {
            "min_probability": float(probs.min()),
            "max_probability": float(probs.max()),
            "max_row_sum_error": float(np.abs(row_sums - 1.0).max()) if probs.size else 0.0,
            "all_in_unit_interval": bool(probs.min() >= -tolerance and probs.max() <= 1 + tolerance),
            "rows_sum_to_one": bool(np.abs(row_sums - 1.0).max() <= tolerance) if probs.size else True,
        }
