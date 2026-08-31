"""Data-quality scoring — record level and batch level.

    DQ(record) = 100 x PROD_d (1 - w_d * penalty_d)
    penalty_d  = clip( SUM_{r in R_d} sev(r)*violated(r) / SUM_{r in R_d} sev(r), 0, 1 )
    sev(ERROR) = 1.0, sev(WARNING) = 0.4

The product form is the point: one ERROR cannot be washed out by many passes,
which is exactly what an arithmetic mean would do. Completeness has no rules
behind it and is driven directly by the null rate on required fields.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lpie.core.config import Settings, get_settings
from lpie.core.timing import utcnow_iso

DIMENSIONS = ("completeness", "validity", "consistency", "timeliness", "uniqueness", "cross_source")


class DQScorer:
    def __init__(self, settings: Settings | None = None, rules: list[Any] | None = None) -> None:
        self.settings = settings or get_settings()
        cfg = self.settings.section("dq")
        self.weights: dict[str, float] = {
            d: float(cfg.get("weights", {}).get(d, 0.0)) for d in DIMENSIONS
        }
        self.severity_weight: dict[str, float] = {
            k: float(v) for k, v in (cfg.get("severity_weight") or {"ERROR": 1.0, "WARNING": 0.4}).items()
        }
        self.grades: dict[str, float] = {
            k: float(v) for k, v in (cfg.get("grades") or {"A": 95, "B": 85, "C": 70, "D": 50}).items()
        }
        self.required_fields: list[str] = list(cfg.get("required_fields") or [])
        self.rules = rules or []
        self._by_dimension: dict[str, list[Any]] = {}
        for r in self.rules:
            self._by_dimension.setdefault(r.dimension, []).append(r)

    # ------------------------------------------------------------------ #
    def completeness_penalty(self, df: pd.DataFrame) -> pd.Series:
        """Fraction of the required fields that are null on this record."""
        fields = [f for f in self.required_fields if f in df.columns]
        if not fields:
            return pd.Series(0.0, index=df.index)
        nulls = df[fields].isna().sum(axis=1).astype("float64")
        return (nulls / len(fields)).clip(0.0, 1.0)

    def dimension_penalty(self, dimension: str, pass_frame: pd.DataFrame) -> pd.Series:
        """Severity-weighted share of the dimension's rules that this record fails."""
        rules = self._by_dimension.get(dimension, [])
        rules = [r for r in rules if r.rule_id in pass_frame.columns]
        if not rules:
            return pd.Series(0.0, index=pass_frame.index)
        total = sum(self.severity_weight.get(r.severity, 1.0) * r.weight for r in rules)
        if total <= 0:
            return pd.Series(0.0, index=pass_frame.index)
        acc = pd.Series(0.0, index=pass_frame.index)
        for r in rules:
            sev = self.severity_weight.get(r.severity, 1.0) * r.weight
            acc = acc + (~pass_frame[r.rule_id]).astype("float64") * sev
        return (acc / total).clip(0.0, 1.0)

    # ------------------------------------------------------------------ #
    def score_records(self, df: pd.DataFrame, pass_frame: pd.DataFrame) -> pd.DataFrame:
        if pass_frame.empty:
            return pd.DataFrame(columns=["loan_id", "month_index", *DIMENSIONS, "dq_score", "dq_grade"])

        penalties: dict[str, pd.Series] = {}
        for dim in DIMENSIONS:
            if dim == "completeness":
                penalties[dim] = self.completeness_penalty(df)
            else:
                penalties[dim] = self.dimension_penalty(dim, pass_frame)

        score = pd.Series(1.0, index=pass_frame.index)
        out: dict[str, Any] = {}
        for dim in DIMENSIONS:
            w = self.weights.get(dim, 0.0)
            factor = (1.0 - w * penalties[dim]).clip(0.0, 1.0)
            score = score * factor
            # Per-dimension sub-score on the same 0-100 scale, for the UI.
            out[dim] = ((1.0 - penalties[dim]) * 100.0).round(4)

        dq = (score * 100.0).clip(0.0, 100.0)
        result = pd.DataFrame(out, index=pass_frame.index)
        result["dq_score"] = dq.round(4)
        result["dq_grade"] = self.grade(dq)
        result["n_rules_violated"] = (~pass_frame).sum(axis=1).astype("int32")

        ids = pd.DataFrame(index=pass_frame.index)
        for c in ("loan_id", "month_index"):
            if c in df.columns:
                ids[c] = df[c].to_numpy()
        return pd.concat([ids, result], axis=1).reset_index(drop=True)

    def grade(self, score: pd.Series) -> pd.Series:
        a, b, c, d = (self.grades.get(k, v) for k, v in (("A", 95), ("B", 85), ("C", 70), ("D", 50)))
        conditions = [score >= a, score >= b, score >= c, score >= d]
        return pd.Series(
            np.select(conditions, ["A", "B", "C", "D"], default="F"), index=score.index, dtype=object
        )

    def grade_scalar(self, score: float) -> str:
        if score >= self.grades.get("A", 95):
            return "A"
        if score >= self.grades.get("B", 85):
            return "B"
        if score >= self.grades.get("C", 70):
            return "C"
        if score >= self.grades.get("D", 50):
            return "D"
        return "F"

    # ------------------------------------------------------------------ #
    def score_batch(
        self,
        record_scores: pd.DataFrame,
        pass_frame: pd.DataFrame,
        *,
        batch_id: str = "adhoc",
        drift_verdict: str | None = None,
    ) -> dict[str, Any]:
        if record_scores.empty:
            return {"batch_id": batch_id, "n_records": 0, "mean_dq": None, "grade": None}

        by_id = {r.rule_id: r for r in self.rules}
        n_error = sum(
            int((~pass_frame[rid]).sum())
            for rid in pass_frame.columns
            if by_id.get(rid) is not None and by_id[rid].severity == "ERROR"
        )
        n_warning = sum(
            int((~pass_frame[rid]).sum())
            for rid in pass_frame.columns
            if by_id.get(rid) is not None and by_id[rid].severity == "WARNING"
        )
        counts = record_scores["dq_grade"].value_counts()
        n = int(len(record_scores))
        mean_dq = float(record_scores["dq_score"].mean())

        per_rule = sorted(
            (
                {
                    "rule_id": rid,
                    "name": by_id[rid].name if rid in by_id else rid,
                    "severity": by_id[rid].severity if rid in by_id else "UNKNOWN",
                    "n_violations": int((~pass_frame[rid]).sum()),
                }
                for rid in pass_frame.columns
            ),
            key=lambda r: -r["n_violations"],
        )

        return {
            "batch_id": batch_id,
            "n_records": n,
            "mean_dq": round(mean_dq, 4),
            "median_dq": round(float(record_scores["dq_score"].median()), 4),
            "p05_dq": round(float(record_scores["dq_score"].quantile(0.05)), 4),
            "grade": self.grade_scalar(mean_dq),
            "grade_distribution": {k: int(v) for k, v in counts.items()},
            "pct_grade_a": round(100.0 * float(counts.get("A", 0)) / n, 4),
            "pct_grade_f": round(100.0 * float(counts.get("F", 0)) / n, 4),
            "n_error_violations": n_error,
            "n_warning_violations": n_warning,
            "mean_dimension_scores": {
                d: round(float(record_scores[d].mean()), 4) for d in DIMENSIONS if d in record_scores
            },
            "top_violated_rules": per_rule[:10],
            "drift_verdict": drift_verdict,
            "computed_at": utcnow_iso(),
        }
