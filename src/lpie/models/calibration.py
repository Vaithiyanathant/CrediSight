"""Isotonic calibration with segment-aware shrinkage.

Isotonic rather than Platt: it is monotone and non-parametric, and with ~30k
calibration rows there is ample data — Platt would impose a sigmoid shape the
data has not asked for.

Segment-wise by `vintage_year x credit_score_band`, with a hierarchical
fallback: use the segment curve when n >= 500, otherwise shrink toward the
global curve with weight `n / (n + 500)`. That keeps thin cells from
overfitting while still letting fat cells express genuinely different
calibration.

The calibration slice is a dedicated out-of-time window (see splitters), never
the data used for model selection — fitting calibration on selection data
re-introduces exactly the optimism calibration is meant to remove.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from lpie.core.logging import get_logger

log = get_logger(__name__)

MIN_SEGMENT_ROWS = 500
SHRINKAGE_PRIOR = 500.0
EPS = 1e-9


@dataclass
class CalibrationArtifact:
    global_curve: IsotonicRegression | None = None
    segment_curves: dict[str, IsotonicRegression] = field(default_factory=dict)
    segment_counts: dict[str, int] = field(default_factory=dict)
    segment_keys: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    n_calibration_rows: int = 0
    calibration_window: str = ""

    @property
    def is_fitted(self) -> bool:
        return self.global_curve is not None


def segment_labels(
    vintage: pd.Series | None, credit_band: pd.Series | None, index: pd.Index
) -> pd.Series:
    """`vintage x credit band` label, with a single bucket when either is absent."""
    if vintage is None and credit_band is None:
        return pd.Series("__global__", index=index, dtype=object)
    v = (
        pd.to_numeric(vintage, errors="coerce").astype("Int64").astype("string").fillna("NA")
        if vintage is not None
        else pd.Series("NA", index=index, dtype="string")
    )
    c = (
        credit_band.astype("string").fillna("NA")
        if credit_band is not None
        else pd.Series("NA", index=index, dtype="string")
    )
    return (v + "|" + c).astype(object)


class SegmentIsotonicCalibrator:
    def __init__(self, min_segment_rows: int = MIN_SEGMENT_ROWS, prior: float = SHRINKAGE_PRIOR) -> None:
        self.min_segment_rows = int(min_segment_rows)
        self.prior = float(prior)
        self.artifact = CalibrationArtifact()

    @property
    def floor(self) -> float:
        """Smallest probability the calibration sample can actually resolve.

        PAVA assigns a block with zero observed positives a fitted value of
        exactly 0.0. For the 12-month default head — a 1.86% base-rate event —
        that block covers most of the active population, so 92% of active loans
        were being served P(default) = 0 and an expected loss of exactly zero.

        Zero is not a low probability, it is a claim of impossibility, and no
        finite sample supports it: a block of n all-negative observations has
        posterior mean 1/(n+2) under a uniform prior, not 0. The floor is that
        Laplace bound over the calibration sample, so the number stays honest
        about what the data can resolve without inventing risk that was not
        measured.
        """
        n = int(self.artifact.n_calibration_rows)
        return 1.0 / (n + 2.0) if n > 0 else 0.0

    # ------------------------------------------------------------------ #
    def fit(
        self,
        p_raw: np.ndarray | pd.Series,
        y: np.ndarray | pd.Series,
        segments: pd.Series | None = None,
        *,
        calibration_window: str = "",
        eligible: np.ndarray | pd.Series | None = None,
        compute_metrics: bool = True,
    ) -> CalibrationArtifact:
        """Fit the global and segment curves.

        `eligible` restricts the fit to the rows the calibrator will actually be
        asked to score in production. Absorbing-state rows must be excluded:
        the serving layer overwrites them with a deterministic 0.0 gate, but
        while they sit in the isotonic fit they form one enormous all-negative
        block at the bottom of the score range. PAVA then pools that block with
        the adjacent active rows and assigns the whole region a fitted value of
        exactly 0.0 — which is how 93% of *active* loans ended up with
        `prob_next_12m_default == 0.0` and an expected loss of zero. Excluding
        them costs nothing (their output is gated anyway) and restores the
        calibrator's resolution over the population it is actually used on.
        """
        p = np.clip(np.asarray(p_raw, dtype="float64"), EPS, 1 - EPS)
        y_arr = np.asarray(y, dtype="float64")
        finite = np.isfinite(p) & np.isfinite(y_arr)
        if eligible is not None:
            finite &= np.asarray(eligible, dtype=bool)
        p, y_arr = p[finite], y_arr[finite]

        if len(p) < 20 or len(np.unique(y_arr)) < 2:
            log.warning("calibration.insufficient_data", n=int(len(p)))
            self.artifact = CalibrationArtifact(
                n_calibration_rows=int(len(p)), calibration_window=calibration_window
            )
            return self.artifact

        global_curve = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        global_curve.fit(p, y_arr)

        segment_curves: dict[str, IsotonicRegression] = {}
        segment_counts: dict[str, int] = {}
        if segments is not None:
            seg = pd.Series(segments).to_numpy()[finite]
            for key in pd.unique(seg):
                mask = seg == key
                n = int(mask.sum())
                segment_counts[str(key)] = n
                if n < 50 or len(np.unique(y_arr[mask])) < 2:
                    continue
                curve = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                curve.fit(p[mask], y_arr[mask])
                segment_curves[str(key)] = curve

        self.artifact = CalibrationArtifact(
            global_curve=global_curve,
            segment_curves=segment_curves,
            segment_counts=segment_counts,
            segment_keys=sorted(segment_curves),
            n_calibration_rows=int(len(p)),
            calibration_window=calibration_window,
        )
        if not compute_metrics:
            # Set by the cross-fitter, whose inner fits must not recurse back
            # into cross-fitting to score themselves.
            return self.artifact

        # Honest ECE requires out-of-sample calibrated scores. Isotonic fitted
        # values ARE the within-bin observed means, so scoring the fit data with
        # its own curve drives ECE to exactly 0.0 by construction — the 0.0 this
        # field used to report was an artifact of the identity, not evidence of
        # perfect calibration. Cross-fitting is the cheapest correct estimate.
        seg_fit = pd.Series(segments).to_numpy()[finite] if segments is not None else None
        honest = cross_fitted_calibrated(
            p, y_arr, seg_fit,
            min_segment_rows=self.min_segment_rows, prior=self.prior,
        )
        # `p`/`y_arr` are already restricted to eligible rows by the `finite`
        # mask above, so no further gating is needed here.
        self.artifact.metrics = {
            "ece_before": expected_calibration_error(p, y_arr),
            "ece_after": expected_calibration_error(honest, y_arr),
            "ece_after_in_sample": expected_calibration_error(
                self.transform(p, pd.Series(seg_fit) if seg_fit is not None else None), y_arr
            ),
            "brier_before": float(np.mean((p - y_arr) ** 2)),
            "brier_after": float(np.mean((honest - y_arr) ** 2)),
            "ece_estimator": "5-fold cross-fitted (out-of-sample)",
            "fit_rows_eligible_only": eligible is not None,
            "n_segments_fitted": len(segment_curves),
        }
        return self.artifact

    # ------------------------------------------------------------------ #
    def transform(
        self, p_raw: np.ndarray | pd.Series, segments: pd.Series | None = None
    ) -> np.ndarray:
        """Calibrate, shrinking the segment curve toward the global one by n/(n+prior)."""
        art = self.artifact
        p = np.clip(np.asarray(p_raw, dtype="float64"), EPS, 1 - EPS)
        if art.global_curve is None:
            return p

        out = np.asarray(art.global_curve.predict(p), dtype="float64")
        if segments is None or not art.segment_curves:
            return np.clip(out, self.floor, 1.0 - self.floor)

        seg = pd.Series(segments).astype(str).to_numpy()
        for key, curve in art.segment_curves.items():
            mask = seg == key
            if not mask.any():
                continue
            n = float(art.segment_counts.get(key, 0))
            weight = n / (n + self.prior) if n < self.min_segment_rows else 1.0
            segment_pred = np.asarray(curve.predict(p[mask]), dtype="float64")
            out[mask] = weight * segment_pred + (1.0 - weight) * out[mask]
        return np.clip(out, self.floor, 1.0 - self.floor)

    def segment_for(self, key: str) -> str:
        """Which curve actually applied — surfaced as `calibration_segment` in the API."""
        if key in self.artifact.segment_curves:
            n = self.artifact.segment_counts.get(key, 0)
            return f"{key} (n={n}{'' if n >= self.min_segment_rows else ', shrunk to global'})"
        return "global"


def cross_fitted_calibrated(
    p: np.ndarray,
    y: np.ndarray,
    segments: np.ndarray | None = None,
    *,
    n_splits: int = 5,
    seed: int = 20260828,
    min_segment_rows: int = MIN_SEGMENT_ROWS,
    prior: float = SHRINKAGE_PRIOR,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Out-of-sample calibrated scores: fit on K-1 folds, score the held-out one.

    Every metric that judges the calibration layer — ECE, Brier, and the decision
    threshold swept over calibrated scores — must be computed on these, not on
    the calibrator's own fitted values. Threshold selection is the sharper of the
    two: a threshold tuned against in-sample isotonic output is tuned against the
    observed rates it was built from, so it looks optimal on paper and drifts in
    production.
    """
    p = np.asarray(p, dtype="float64")
    y = np.asarray(y, dtype="float64")
    n = len(p)
    out = np.full(n, np.nan, dtype="float64")
    if n < 5 * n_splits or len(np.unique(y)) < 2:
        return np.clip(p, 0.0, 1.0)

    keep = np.ones(n, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    rng = np.random.default_rng(seed)
    folds = rng.permutation(n) % n_splits
    for k in range(n_splits):
        test = folds == k
        # The inner fits must exclude the same rows the production fit excludes,
        # or the cross-fitted estimate reproduces the collapse it exists to detect.
        train = (~test) & keep
        if not test.any() or train.sum() < 20 or len(np.unique(y[train])) < 2:
            out[test] = p[test]
            continue
        cal = SegmentIsotonicCalibrator(min_segment_rows=min_segment_rows, prior=prior)
        seg_tr = pd.Series(segments[train]) if segments is not None else None
        seg_te = pd.Series(segments[test]) if segments is not None else None
        cal.fit(p[train], y[train], seg_tr, compute_metrics=False)
        out[test] = cal.transform(p[test], seg_te) if cal.artifact.is_fitted else p[test]

    bad = ~np.isfinite(out)
    out[bad] = p[bad]
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# calibration diagnostics
# --------------------------------------------------------------------------- #
def reliability_curve(
    p: np.ndarray, y: np.ndarray, n_bins: int = 10
) -> list[dict[str, Any]]:
    p = np.asarray(p, dtype="float64")
    y = np.asarray(y, dtype="float64")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        out.append(
            {
                "bin": b,
                "lower": round(float(edges[b]), 4),
                "upper": round(float(edges[b + 1]), 4),
                "n": n,
                "mean_predicted": round(float(p[mask].mean()), 6) if n else None,
                "observed_rate": round(float(y[mask].mean()), 6) if n else None,
            }
        )
    return out


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    p = np.asarray(p, dtype="float64")
    y = np.asarray(y, dtype="float64")
    if p.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        total += (n / len(p)) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return round(float(total), 6)


def maximum_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    p = np.asarray(p, dtype="float64")
    y = np.asarray(y, dtype="float64")
    if p.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    worst = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        worst = max(worst, abs(float(p[mask].mean()) - float(y[mask].mean())))
    return round(float(worst), 6)


def brier_decomposition(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    """Brier = reliability - resolution + uncertainty.

    This separates "am I well calibrated?" (reliability) from "am I informative?"
    (resolution) — two questions a single Brier score conflates.
    """
    p = np.asarray(p, dtype="float64")
    y = np.asarray(y, dtype="float64")
    n = len(p)
    if n == 0:
        return {"brier": float("nan"), "reliability": float("nan"),
                "resolution": float("nan"), "uncertainty": float("nan")}

    base = float(y.mean())
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    reliability = resolution = 0.0
    for b in range(n_bins):
        mask = idx == b
        nk = int(mask.sum())
        if nk == 0:
            continue
        p_bar = float(p[mask].mean())
        o_bar = float(y[mask].mean())
        reliability += nk * (p_bar - o_bar) ** 2
        resolution += nk * (o_bar - base) ** 2
    reliability /= n
    resolution /= n
    uncertainty = base * (1.0 - base)

    return {
        "brier": round(float(np.mean((p - y) ** 2)), 6),
        "reliability": round(reliability, 6),
        "resolution": round(resolution, 6),
        "uncertainty": round(uncertainty, 6),
        "identity_check": round(reliability - resolution + uncertainty, 6),
    }


def calibration_report(
    p: np.ndarray, y: np.ndarray, *, n_bins: int = 10, label: str = ""
) -> dict[str, Any]:
    return {
        "label": label,
        "n": int(len(p)),
        "ece": expected_calibration_error(p, y, n_bins),
        "mce": maximum_calibration_error(p, y, n_bins),
        "brier_decomposition": brier_decomposition(p, y, n_bins),
        "reliability_curve": reliability_curve(p, y, n_bins),
        "mean_predicted": round(float(np.mean(p)), 6) if len(p) else None,
        "observed_rate": round(float(np.mean(y)), 6) if len(y) else None,
    }
