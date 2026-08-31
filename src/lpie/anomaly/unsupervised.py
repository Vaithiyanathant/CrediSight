"""Tier 3 — the unsupervised ensemble.

Four detectors with genuinely different inductive biases, fused by rank
averaging (scale-free, so no one detector's score distribution dominates):

* **Isolation Forest** — global multivariate outliers. Fast, robust, standard.
* **ECOD** — empirical-CDF tail detection. Parameter-free and deterministic, so
  there is no tuning choice to justify, and it explains itself dimension by
  dimension.
* **Autoencoder** — reconstruction error. This is what finds "the balance is
  plausible and the status is plausible but they do not go together", i.e. the
  cross-column breaks a per-column test cannot see.
* **Per-loan self-referential z-score** — value against *this loan's own*
  history via median/MAD. Panel-native; catches step changes a global detector
  calls normal.

Each detector degrades independently: if PyTorch or PyOD is unavailable the
ensemble drops that member and says so, rather than failing the request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from lpie.core.config import Settings, get_settings
from lpie.core.logging import get_logger

log = get_logger(__name__)

DETECTORS = ("iforest", "ecod", "autoencoder", "self_z")


@dataclass
class UnsupervisedArtifact:
    iforest: Any = None
    ecod: Any = None
    autoencoder: Any = None
    scaler_mean: np.ndarray | None = None
    scaler_scale: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)
    available: dict[str, bool] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    self_z_columns: list[str] = field(default_factory=list)


def _standardise(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    out = (X - mean) / np.where(scale > 1e-9, scale, 1.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def rank_normalise(scores: np.ndarray) -> np.ndarray:
    """Map to [0, 1] by rank. Scale-free, so detectors combine without weighting."""
    finite = np.isfinite(scores)
    out = np.full(len(scores), 0.5, dtype="float64")
    if finite.sum() < 2:
        return out
    values = scores[finite]
    order = values.argsort(kind="mergesort").argsort()
    out[finite] = order / max(len(values) - 1, 1)
    return out


class UnsupervisedEnsemble:
    def __init__(self, settings: Settings | None = None, artifact: UnsupervisedArtifact | None = None) -> None:
        self.settings = settings or get_settings()
        self.cfg = self.settings.section("anomaly")
        self.artifact = artifact or UnsupervisedArtifact()

    # ------------------------------------------------------------------ #
    def fit(
        self, X: pd.DataFrame, feature_names: list[str], *, max_rows: int = 120_000
    ) -> UnsupervisedArtifact:
        numeric = [
            c for c in feature_names
            if c in X.columns and pd.api.types.is_numeric_dtype(X[c])
        ]
        work = X[numeric]
        if len(work) > max_rows:
            work = work.sample(max_rows, random_state=self.settings.seed)
        values = work.to_numpy(dtype="float64")

        mean = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        mean = np.nan_to_num(mean, nan=0.0)
        scale = np.nan_to_num(scale, nan=1.0)
        Z = _standardise(values, mean, scale)

        art = UnsupervisedArtifact(
            feature_names=numeric, scaler_mean=mean, scaler_scale=scale, available={}, notes={}
        )

        cfg_if = self.cfg.get("isolation_forest", {})
        art.iforest = IsolationForest(
            n_estimators=int(cfg_if.get("n_estimators", 300)),
            max_samples=min(int(cfg_if.get("max_samples", 8192)), len(Z)),
            contamination=float(cfg_if.get("contamination", 0.05)),
            random_state=self.settings.seed,
            n_jobs=int(self.settings.get("runtime.duckdb_threads", 4)),
        ).fit(Z)
        art.available["iforest"] = True

        if self.cfg.get("ecod", {}).get("enabled", True):
            try:
                from pyod.models.ecod import ECOD

                ecod = ECOD()
                ecod.fit(Z)
                art.ecod = ecod
                art.available["ecod"] = True
            except Exception as exc:
                art.available["ecod"] = False
                art.notes["ecod"] = f"unavailable: {exc}"
                log.warning("anomaly.ecod_unavailable", error=str(exc))
        else:
            art.available["ecod"] = False
            art.notes["ecod"] = "disabled in configuration"

        if self.cfg.get("autoencoder", {}).get("enabled", True):
            try:
                art.autoencoder = self._fit_autoencoder(Z)
                art.available["autoencoder"] = True
            except Exception as exc:
                art.available["autoencoder"] = False
                art.notes["autoencoder"] = f"unavailable: {exc}"
                log.warning("anomaly.autoencoder_unavailable", error=str(exc))
        else:
            art.available["autoencoder"] = False
            art.notes["autoencoder"] = "disabled in configuration"

        art.self_z_columns = [
            c for c in ("bal_z_self_12m", "amort_residual_pct", "dpd_delta_1", "servicer_bal_gap_pct")
            if c in X.columns
        ]
        art.available["self_z"] = bool(art.self_z_columns)

        self.artifact = art
        log.info("anomaly.fitted", detectors={k: v for k, v in art.available.items()})
        return art

    def _fit_autoencoder(self, Z: np.ndarray) -> dict[str, Any]:
        import torch
        from torch import nn

        cfg = self.cfg.get("autoencoder", {})
        hidden = list(cfg.get("hidden", [64, 16, 64]))
        torch.manual_seed(self.settings.seed)
        d = Z.shape[1]

        model = nn.Sequential(
            nn.Linear(d, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], hidden[2]), nn.ReLU(),
            nn.Linear(hidden[2], d),
        )
        optimiser = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 0.003)))
        loss_fn = nn.MSELoss()
        tensor = torch.tensor(np.clip(Z, -10, 10), dtype=torch.float32)
        batch = int(cfg.get("batch_size", 2048))
        epochs = int(cfg.get("epochs", 12))

        model.train()
        for _ in range(epochs):
            perm = torch.randperm(len(tensor))
            for i in range(0, len(tensor), batch):
                chunk = tensor[perm[i : i + batch]]
                optimiser.zero_grad()
                loss = loss_fn(model(chunk), chunk)
                loss.backward()
                optimiser.step()
        model.eval()
        return {"model": model, "input_dim": d}

    # ------------------------------------------------------------------ #
    def score(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Per-detector raw scores plus their rank-averaged fusion."""
        art = self.artifact
        n = len(X)
        available = [c for c in art.feature_names if c in X.columns]
        values = X.reindex(columns=art.feature_names).to_numpy(dtype="float64")
        Z = _standardise(values, art.scaler_mean, art.scaler_scale)

        raw: dict[str, np.ndarray] = {}
        if art.iforest is not None:
            # Higher = more anomalous, so negate the decision function.
            raw["iforest"] = -np.asarray(art.iforest.decision_function(Z), dtype="float64")
        if art.ecod is not None:
            raw["ecod"] = np.asarray(art.ecod.decision_function(Z), dtype="float64")
        if art.autoencoder is not None:
            raw["autoencoder"] = self._autoencoder_error(Z)
        if art.self_z_columns:
            raw["self_z"] = self._self_z(X, art.self_z_columns)

        ranked = {name: rank_normalise(values) for name, values in raw.items()}
        fused = (
            np.mean(np.column_stack(list(ranked.values())), axis=1)
            if ranked
            else np.zeros(n, dtype="float64")
        )
        return {
            **{f"{k}_raw": v for k, v in raw.items()},
            **{k: v for k, v in ranked.items()},
            "unsupervised_rank": fused,
            "n_detectors": np.full(n, len(ranked), dtype="float64"),
            "_missing_features": np.full(n, len(art.feature_names) - len(available), dtype="float64"),
        }

    def _autoencoder_error(self, Z: np.ndarray) -> np.ndarray:
        import torch

        model = self.artifact.autoencoder["model"]
        tensor = torch.tensor(np.clip(Z, -10, 10), dtype=torch.float32)
        with torch.no_grad():
            reconstructed = model(tensor).numpy()
        return np.mean((Z - reconstructed) ** 2, axis=1)

    def reconstruction_breakdown(self, X: pd.DataFrame, row: int) -> list[dict[str, Any]]:
        """Per-feature reconstruction error for one record.

        This is what turns an autoencoder score into a sentence a reviewer can
        act on: "the model expected current_balance around 168,200 given
        everything else; it observed 312,400."
        """
        art = self.artifact
        if art.autoencoder is None:
            return []
        import torch

        values = X.reindex(columns=art.feature_names).to_numpy(dtype="float64")
        Z = _standardise(values, art.scaler_mean, art.scaler_scale)
        tensor = torch.tensor(np.clip(Z[row : row + 1], -10, 10), dtype=torch.float32)
        with torch.no_grad():
            reconstructed = art.autoencoder["model"](tensor).numpy()[0]

        errors = (Z[row] - reconstructed) ** 2
        order = np.argsort(-errors)[:8]
        return [
            {
                "feature": art.feature_names[i],
                "observed": _round(values[row, i]),
                "expected": _round(reconstructed[i] * art.scaler_scale[i] + art.scaler_mean[i]),
                "squared_error": round(float(errors[i]), 6),
            }
            for i in order
            if np.isfinite(errors[i]) and errors[i] > 0
        ]

    def _self_z(self, X: pd.DataFrame, columns: list[str]) -> np.ndarray:
        """Max absolute per-loan self-referential z-score across the tracked columns."""
        parts = []
        for c in columns:
            values = pd.to_numeric(X[c], errors="coerce").abs()
            parts.append(values.fillna(0.0).to_numpy(dtype="float64"))
        return np.max(np.column_stack(parts), axis=1) if parts else np.zeros(len(X))


def _round(value: float) -> float | None:
    return None if not np.isfinite(value) else round(float(value), 4)


def nearest_normal_neighbours(
    X: pd.DataFrame,
    target_row: int,
    normal_mask: np.ndarray,
    feature_names: list[str],
    *,
    k: int = 5,
) -> list[dict[str, Any]]:
    """The k most similar non-anomalous records, with a field-by-field diff.

    The most reviewer-friendly explanation format there is: "here are five loans
    that look like this one and did *not* trip a rule; here is what differs."
    """
    numeric = [c for c in feature_names if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    if not numeric or not normal_mask.any():
        return []

    values = X[numeric].to_numpy(dtype="float64")
    mean = np.nan_to_num(np.nanmean(values, axis=0), nan=0.0)
    scale = np.nan_to_num(np.nanstd(values, axis=0), nan=1.0)
    Z = _standardise(values, mean, scale)

    target = Z[target_row]
    candidates = np.flatnonzero(normal_mask)
    candidates = candidates[candidates != target_row]
    if candidates.size == 0:
        return []
    distances = np.linalg.norm(Z[candidates] - target, axis=1)
    take = candidates[np.argsort(distances)[:k]]

    out = []
    for idx in take:
        diffs = []
        for j, name in enumerate(numeric):
            a, b = values[target_row, j], values[idx, j]
            if not np.isfinite(a) or not np.isfinite(b):
                continue
            gap = abs(a - b) / max(abs(scale[j]), 1e-9)
            if gap > 1.0:
                diffs.append({"feature": name, "this_loan": _round(a), "neighbour": _round(b),
                              "z_gap": round(float(gap), 4)})
        diffs.sort(key=lambda d: -d["z_gap"])
        row = {
            "loan_id": X.iloc[idx].get("loan_id"),
            "month_index": int(X.iloc[idx].get("month_index", 0) or 0),
            "distance": round(float(np.linalg.norm(Z[idx] - target)), 4),
            "differences": diffs[:6],
        }
        out.append(row)
    return out
