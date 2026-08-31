"""Threshold optimisation and the reviewer decision mapping.

Probabilities are for the model; **thresholds are a business decision**. They are
optimised per head on the calibration slice against an explicit objective and
then exposed, rather than being buried inside a `> 0.5`.

The default head maximises recall subject to precision >= 0.30 — a realistic
servicing-capacity constraint. The exception head maximises F1, because review
capacity is the binding constraint there.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from lpie.core.config import Settings, get_settings
from lpie.core.logging import get_logger

log = get_logger(__name__)

Action = Literal["No Action", "Flag", "Escalate"]
ACTIONS: tuple[str, ...] = ("No Action", "Flag", "Escalate")


@dataclass
class ThresholdResult:
    head: str
    objective: str
    threshold: float
    precision: float
    recall: float
    f1: float
    f2: float
    n_positives_predicted: int
    n_positives_actual: int
    constraint_met: bool = True
    note: str = ""
    curve: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _prf(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float, float]:
    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) else 0.0
    return precision, recall, f1, f2


def threshold_curve(
    p: np.ndarray, y: np.ndarray, *, n_points: int = 200
) -> list[dict[str, float]]:
    p = np.asarray(p, dtype="float64")
    y = np.asarray(y, dtype="float64")
    if p.size == 0:
        return []
    candidates = np.unique(np.quantile(p, np.linspace(0.0, 1.0, min(n_points, len(p)))))
    out = []
    for t in candidates:
        pred = (p >= t).astype("float64")
        precision, recall, f1, f2 = _prf(y, pred)
        out.append(
            {
                "threshold": round(float(t), 6),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "f2": round(f2, 6),
                "n_flagged": int(pred.sum()),
            }
        )
    return out


def optimise_threshold(
    p: np.ndarray,
    y: np.ndarray,
    *,
    head: str,
    objective: str = "f1",
    min_precision: float = 0.30,
    settings: Settings | None = None,
) -> ThresholdResult:
    """Pick the operating point that maximises the head's stated objective."""
    p = np.asarray(p, dtype="float64")
    y = np.asarray(y, dtype="float64")
    curve = threshold_curve(p, y)
    n_actual = int(y.sum())

    if not curve:
        return ThresholdResult(head, objective, 0.5, 0.0, 0.0, 0.0, 0.0, 0, n_actual,
                               constraint_met=False, note="No scored rows available")

    if objective == "recall_at_precision":
        feasible = [row for row in curve if row["precision"] >= min_precision and row["n_flagged"] > 0]
        if feasible:
            best = max(feasible, key=lambda r: (r["recall"], r["precision"]))
            note = f"Maximum recall subject to precision >= {min_precision}"
            met = True
        else:
            # The constraint is unreachable on this data. Say so and fall back to
            # the best achievable precision, rather than silently pretending the
            # constraint was met.
            best = max(curve, key=lambda r: (r["precision"], r["recall"]))
            note = (
                f"Precision >= {min_precision} is unreachable on the calibration slice "
                f"(best achievable {best['precision']:.4f}); using the max-precision point."
            )
            met = False
    elif objective == "f2":
        best = max(curve, key=lambda r: r["f2"])
        note = "Maximum F2 (recall-weighted)"
        met = True
    else:
        best = max(curve, key=lambda r: r["f1"])
        note = "Maximum F1"
        met = True

    return ThresholdResult(
        head=head,
        objective=objective,
        threshold=float(best["threshold"]),
        precision=float(best["precision"]),
        recall=float(best["recall"]),
        f1=float(best["f1"]),
        f2=float(best["f2"]),
        n_positives_predicted=int(best["n_flagged"]),
        n_positives_actual=n_actual,
        constraint_met=met,
        note=note,
        curve=curve,
    )


# --------------------------------------------------------------------------- #
# reviewer decision mapping
# --------------------------------------------------------------------------- #
@dataclass
class ReviewerPolicy:
    tau_hi: float = 0.15
    tau_lo: float = 0.05
    anomaly_escalate: float = 0.90
    anomaly_flag: float = 0.70

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> ReviewerPolicy:
        s = settings or get_settings()
        cfg = s.section("thresholds").get("reviewer_action", {})
        return cls(
            tau_hi=float(cfg.get("tau_hi", 0.15)),
            tau_lo=float(cfg.get("tau_lo", 0.05)),
            anomaly_escalate=float(cfg.get("anomaly_escalate", 0.90)),
            anomaly_flag=float(cfg.get("anomaly_flag", 0.70)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reviewer_action(
    p_default: np.ndarray | pd.Series,
    anomaly_score: np.ndarray | pd.Series,
    exception_required: np.ndarray | pd.Series,
    exception_severity: np.ndarray | pd.Series | None = None,
    *,
    policy: ReviewerPolicy | None = None,
) -> pd.Series:
    """Map model output onto the three-level reviewer action.

        Escalate  : p_default >= tau_hi OR anomaly >= 0.90 OR (exception AND ERROR)
        Flag      : p_default >= tau_lo OR anomaly >= 0.70 OR (exception AND WARNING)
        No Action : otherwise
    """
    pol = policy or ReviewerPolicy()
    pd_ = pd.Series(np.asarray(p_default, dtype="float64")).fillna(0.0)
    an = pd.Series(np.asarray(anomaly_score, dtype="float64")).fillna(0.0)
    exc = pd.Series(np.asarray(exception_required, dtype="float64")).fillna(0.0) > 0

    if exception_severity is None:
        sev = pd.Series(["WARNING"] * len(pd_))
    else:
        sev = pd.Series(np.asarray(exception_severity, dtype=object)).fillna("WARNING")

    escalate = (pd_ >= pol.tau_hi) | (an >= pol.anomaly_escalate) | (exc & (sev.to_numpy() == "ERROR"))
    flag = (pd_ >= pol.tau_lo) | (an >= pol.anomaly_flag) | exc

    out = pd.Series("No Action", index=pd_.index, dtype=object)
    out[flag.to_numpy()] = "Flag"
    out[escalate.to_numpy()] = "Escalate"
    return out


def expected_loss(
    p_default: np.ndarray | pd.Series,
    current_balance: np.ndarray | pd.Series,
    loss_severity: np.ndarray | pd.Series | float,
) -> pd.Series:
    """EL = P(default) x balance x E[loss severity]."""
    p = pd.Series(np.asarray(p_default, dtype="float64")).fillna(0.0).clip(0.0, 1.0)
    bal = pd.Series(np.asarray(current_balance, dtype="float64")).fillna(0.0).clip(lower=0.0)
    if np.isscalar(loss_severity):
        sev = pd.Series(float(loss_severity), index=p.index)
    else:
        sev = pd.Series(np.asarray(loss_severity, dtype="float64")).fillna(0.0).clip(0.0, 1.0)
    return (p * bal * sev).astype("float64")


def severity_midpoints(settings: Settings | None = None) -> dict[str, float]:
    s = settings or get_settings()
    return {
        str(k): float(v)
        for k, v in (s.section("thresholds").get("loss_severity_midpoint") or {}).items()
    }


def capacity_constrained_watchlist(
    frame: pd.DataFrame,
    n: int,
    *,
    score_column: str = "expected_loss",
    segment: str | None = None,
    segment_column: str | None = None,
    min_score: float | None = None,
    action_filter: str | None = None,
) -> pd.DataFrame:
    """Given a reviewer budget of n loans, take the top n by expected loss.

    This is what turns a probability into a decision with a stated objective:
    the bank cannot review everything, so the ranking must be by money at risk,
    not by probability alone. A 90%-likely default on a $12k balance costs less
    than a 20%-likely default on a $600k one.
    """
    work = frame
    if segment is not None and segment_column and segment_column in work.columns:
        work = work[work[segment_column].astype(str) == str(segment)]
    if min_score is not None and score_column in work.columns:
        work = work[pd.to_numeric(work[score_column], errors="coerce") >= min_score]
    if action_filter and "reviewer_action" in work.columns:
        work = work[work["reviewer_action"] == action_filter]
    if score_column not in work.columns:
        return work.head(n)
    return work.sort_values(score_column, ascending=False, kind="mergesort").head(int(n))
