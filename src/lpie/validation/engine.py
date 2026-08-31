"""Validation engine.

Runs all 18 rules over a batch, produces per-record results, a violation ledger,
record-level DQ scores and a batch-level DQ score. It is the single component
Task 1 and the anomaly rules tier both call — so a rule fires identically whether
it is being reported or being fused into an anomaly score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from lpie.core.config import Settings, get_settings
from lpie.core.logging import get_logger
from lpie.core.timing import utcnow_iso
from lpie.profiling.dq_score import DQScorer
from lpie.validation.rules import (
    CATEGORICAL_VOCAB_COLUMNS,
    CONTEXT_COLUMNS,
    CTX_DAYS_SINCE_UPDATE,
    CTX_DUP_COUNT,
    CTX_LOAN_TERM,
    CTX_MONTH_END,
    CTX_PREV_BALANCE,
    CTX_PRIOR_TERMINAL,
    CTX_SERVICER_BALANCE,
    CTX_SERVICER_DAYS,
    CTX_SERVICER_RATE,
    CTX_SERVICER_STALE,
    CTX_SERVICER_STATUS,
    CTX_STATIC_ORIG_BAL,
    CTX_STATIC_RATE,
    CTX_UNSEEN_COLS,
    CTX_UNSEEN_COUNT,
    OBSERVED_COLUMNS,
    TERMINAL_STATES,
    Rule,
    RuleContext,
    load_rules,
)

log = get_logger(__name__)

# Panel month 1 corresponds to this calendar month. reporting_month itself is
# corrupted (months 1 and 2 both say 2021-01), so staleness is measured against
# a month_index-derived clock, which VR-013 exists to flag.
PANEL_ANCHOR = pd.Timestamp("2021-01-01")


def _month_end_from_index(month_index: pd.Series, index: pd.Index) -> pd.Series:
    """month_index -> calendar month end, anchored at the panel start.

    Derived from month_index rather than reporting_month because reporting_month
    is corrupted in the supplied pack (VR-013); month_index is the only clock the
    data supports. Vectorised via period arithmetic — no per-row DateOffset.
    """
    offsets = month_index.fillna(1).astype("int64") - 1
    anchor = PANEL_ANCHOR.to_period("M").ordinal
    periods = pd.PeriodIndex.from_ordinals(anchor + offsets.to_numpy(), freq="M")
    ends = pd.Series(periods.to_timestamp(how="end").normalize(), index=index)
    return ends.where(month_index.notna().to_numpy(), pd.NaT)


@dataclass
class ValidationResult:
    record_results: pd.DataFrame     # one row per input record, one bool column per rule
    violations: pd.DataFrame         # long format: (loan_id, month_index, rule_id, ...)
    record_scores: pd.DataFrame      # DQ dimensions + score + grade per record
    batch_score: dict[str, Any]
    summary: dict[str, Any]

    def n_records(self) -> int:
        return int(len(self.record_results))


class ValidationEngine:
    def __init__(self, settings: Settings | None = None, rules: list[Rule] | None = None) -> None:
        self.settings = settings or get_settings()
        self.rules = rules if rules is not None else load_rules(self.settings)
        self.scorer = DQScorer(self.settings, self.rules)
        self._vocabulary: dict[str, set[str]] | None = None

    # ------------------------------------------------------------------ #
    # vocabulary (for VR-018)
    # ------------------------------------------------------------------ #
    def fit_vocabulary(self, train_df: pd.DataFrame) -> dict[str, set[str]]:
        vocab: dict[str, set[str]] = {}
        for col in CATEGORICAL_VOCAB_COLUMNS:
            if col in train_df.columns:
                values = train_df[col].dropna().astype(str).unique()
                vocab[col] = set(values.tolist())
        self._vocabulary = vocab
        return vocab

    @property
    def vocabulary(self) -> dict[str, set[str]]:
        return self._vocabulary or {}

    # ------------------------------------------------------------------ #
    # context preparation
    # ------------------------------------------------------------------ #
    def prepare_context(
        self,
        df: pd.DataFrame,
        *,
        static: pd.DataFrame | None = None,
        servicer: pd.DataFrame | None = None,
        history: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, RuleContext]:
        """Attach every derived column the rules need. Returns (working_df, ctx).

        `history` supplies prior months when the batch itself is a slice (e.g. a
        single month being scored online). Only rows with month_index strictly
        below the batch minimum are used — a rule must never see the future.
        """
        work = df.copy()
        work["_row_pos"] = np.arange(len(work))

        # --- calendar anchors -----------------------------------------
        month_index = pd.to_numeric(work.get("month_index"), errors="coerce")
        month_end = _month_end_from_index(month_index, work.index)
        work[CTX_MONTH_END] = month_end

        if "last_updated_at" in work.columns:
            updated = pd.to_datetime(work["last_updated_at"], errors="coerce")
            work[CTX_DAYS_SINCE_UPDATE] = (month_end - updated).dt.days.astype("Float64")
        else:
            work[CTX_DAYS_SINCE_UPDATE] = pd.Series(pd.NA, index=work.index, dtype="Float64")

        # --- duplicates (VR-013) --------------------------------------
        if {"loan_id", "reporting_month"}.issubset(work.columns):
            work[CTX_DUP_COUNT] = work.groupby(
                ["loan_id", "reporting_month"], dropna=False
            )["_row_pos"].transform("size").astype("float64")
        else:
            work[CTX_DUP_COUNT] = 1.0

        # --- per-loan history (VR-014, VR-015) ------------------------
        combined = self._combined_history(work, history)
        has_history = combined is not None
        if has_history:
            prev_bal, prior_terminal = self._history_lookups(combined)
            key = list(zip(work["loan_id"], work["month_index"], strict=False))
            work[CTX_PREV_BALANCE] = [prev_bal.get(k, np.nan) for k in key]
            work[CTX_PRIOR_TERMINAL] = [prior_terminal.get(k) for k in key]
        else:
            work[CTX_PREV_BALANCE] = np.nan
            work[CTX_PRIOR_TERMINAL] = None

        # --- static join (VR-009, VR-016, VR-017) ---------------------
        has_static = static is not None and not static.empty
        if has_static:
            s = static.set_index("loan_id")
            work[CTX_LOAN_TERM] = work["loan_id"].map(s["loan_term_months"]).astype("float64")
            work[CTX_STATIC_ORIG_BAL] = work["loan_id"].map(s["original_balance"]).astype("float64")
            work[CTX_STATIC_RATE] = work["loan_id"].map(s["interest_rate"]).astype("float64")
        else:
            for c in (CTX_LOAN_TERM, CTX_STATIC_ORIG_BAL, CTX_STATIC_RATE):
                work[c] = np.nan

        # --- servicer as-of join (VR-007) -----------------------------
        has_servicer = servicer is not None and not servicer.empty
        if has_servicer:
            work = self._asof_servicer(work, servicer)
        else:
            for c in (CTX_SERVICER_BALANCE, CTX_SERVICER_RATE, CTX_SERVICER_DAYS, CTX_SERVICER_STALE):
                work[c] = np.nan
            work[CTX_SERVICER_STATUS] = None

        # --- unseen categories (VR-018) -------------------------------
        vocab = self.vocabulary
        if vocab:
            unseen_count = pd.Series(0, index=work.index, dtype="int64")
            unseen_cols: list[list[str]] = [[] for _ in range(len(work))]
            positions = np.arange(len(work))
            for col, allowed in vocab.items():
                if col not in work.columns:
                    continue
                values = work[col]
                bad = values.notna() & ~values.astype(str).isin(allowed)
                if bad.any():
                    unseen_count += bad.astype("int64")
                    for p in positions[bad.to_numpy()]:
                        unseen_cols[p].append(col)
            work[CTX_UNSEEN_COUNT] = unseen_count.astype("float64")
            work[CTX_UNSEEN_COLS] = ["|".join(c) if c else None for c in unseen_cols]
        else:
            work[CTX_UNSEEN_COUNT] = np.nan
            work[CTX_UNSEEN_COLS] = None

        ctx = RuleContext(
            vocabulary=vocab,
            has_servicer=bool(has_servicer),
            has_static=bool(has_static),
            has_history=bool(has_history),
            dq_config=self.settings.section("dq"),
        )
        return work, ctx

    @staticmethod
    def _combined_history(work: pd.DataFrame, history: pd.DataFrame | None) -> pd.DataFrame | None:
        if "loan_id" not in work.columns or "month_index" not in work.columns:
            return None
        cols = ["loan_id", "month_index", "current_balance", "current_status"]
        available = [c for c in cols if c in work.columns]
        if len(available) < 2:
            return None
        parts = [work[available]]
        if history is not None and not history.empty:
            hist_cols = [c for c in cols if c in history.columns]
            if len(hist_cols) >= 2:
                min_month = pd.to_numeric(work["month_index"], errors="coerce").min()
                past = history[pd.to_numeric(history["month_index"], errors="coerce") < min_month]
                if not past.empty:
                    parts.append(past[hist_cols])
        combined = pd.concat(parts, ignore_index=True, sort=False)
        return combined.drop_duplicates(subset=["loan_id", "month_index"], keep="first")

    @staticmethod
    def _history_lookups(
        combined: pd.DataFrame,
    ) -> tuple[dict[tuple[Any, Any], float], dict[tuple[Any, Any], str | None]]:
        c = combined.sort_values(["loan_id", "month_index"], kind="mergesort")
        grp = c.groupby("loan_id", sort=False)

        prev_bal_series = (
            grp["current_balance"].shift(1) if "current_balance" in c.columns else pd.Series(np.nan, index=c.index)
        )
        if "current_status" in c.columns:
            is_terminal = c["current_status"].isin(TERMINAL_STATES)
            # First terminal status strictly before the current row.
            terminal_value = c["current_status"].where(is_terminal)
            prior_terminal_series = (
                terminal_value.groupby(c["loan_id"], sort=False).shift(1).groupby(c["loan_id"], sort=False).ffill()
            )
        else:
            prior_terminal_series = pd.Series(None, index=c.index, dtype="object")

        keys = list(zip(c["loan_id"], c["month_index"], strict=False))
        prev_bal = dict(zip(keys, prev_bal_series.to_numpy(), strict=False))
        prior_terminal = {
            k: (None if pd.isna(v) else str(v))
            for k, v in zip(keys, prior_terminal_series.to_numpy(), strict=False)
        }
        return prev_bal, prior_terminal

    @staticmethod
    def _asof_servicer(work: pd.DataFrame, servicer: pd.DataFrame) -> pd.DataFrame:
        """Backward merge_asof: only servicer updates dated on/before month end.

        A naive equi-join here would import future updates — the classic
        cross-source leakage vector. `direction='backward'` makes it impossible.
        """
        left = work[["loan_id", CTX_MONTH_END, "_row_pos"]].copy()
        left = left.dropna(subset=[CTX_MONTH_END]).sort_values(CTX_MONTH_END, kind="mergesort")
        wanted = ["loan_id", "update_date", "reported_balance", "reported_status", "reported_rate", "stale_flag"]
        right = servicer[[c for c in wanted if c in servicer.columns]].copy()
        if "stale_flag" not in right.columns:
            right["stale_flag"] = np.nan
        right["update_date"] = pd.to_datetime(right["update_date"], errors="coerce")
        right = right.dropna(subset=["update_date"]).sort_values("update_date", kind="mergesort")

        if left.empty or right.empty:
            for c in (CTX_SERVICER_BALANCE, CTX_SERVICER_RATE, CTX_SERVICER_DAYS, CTX_SERVICER_STALE):
                work[c] = np.nan
            work[CTX_SERVICER_STATUS] = None
            return work

        merged = pd.merge_asof(
            left,
            right,
            left_on=CTX_MONTH_END,
            right_on="update_date",
            by="loan_id",
            direction="backward",
        )
        merged[CTX_SERVICER_DAYS] = (merged[CTX_MONTH_END] - merged["update_date"]).dt.days
        lookup = merged.set_index("_row_pos")
        work[CTX_SERVICER_BALANCE] = work["_row_pos"].map(lookup["reported_balance"]).astype("float64")
        work[CTX_SERVICER_RATE] = work["_row_pos"].map(lookup["reported_rate"]).astype("float64")
        work[CTX_SERVICER_DAYS] = work["_row_pos"].map(lookup[CTX_SERVICER_DAYS]).astype("float64")
        work[CTX_SERVICER_STALE] = work["_row_pos"].map(lookup["stale_flag"]).astype("float64")
        work[CTX_SERVICER_STATUS] = work["_row_pos"].map(lookup["reported_status"])
        return work

    # ------------------------------------------------------------------ #
    # evaluation
    # ------------------------------------------------------------------ #
    def run(
        self,
        df: pd.DataFrame,
        *,
        static: pd.DataFrame | None = None,
        servicer: pd.DataFrame | None = None,
        history: pd.DataFrame | None = None,
        batch_id: str = "adhoc",
        emit_violations: bool = True,
        max_violation_rows: int | None = None,
    ) -> ValidationResult:
        if df.empty:
            return ValidationResult(
                record_results=pd.DataFrame(),
                violations=pd.DataFrame(),
                record_scores=pd.DataFrame(),
                batch_score={"batch_id": batch_id, "n_records": 0},
                summary={"n_records": 0, "rules_evaluated": len(self.rules)},
            )

        work, ctx = self.prepare_context(df, static=static, servicer=servicer, history=history)

        passes: dict[str, pd.Series] = {}
        for rule in self.rules:
            passes[rule.rule_id] = rule.evaluate(work, ctx)

        pass_frame = pd.DataFrame(passes, index=work.index)
        id_cols = [c for c in ("loan_id", "month_index", "reporting_month") if c in work.columns]
        record_results = pd.concat([work[id_cols].reset_index(drop=True), pass_frame.reset_index(drop=True)], axis=1)

        violations = (
            self._build_violations(work, pass_frame, max_rows=max_violation_rows)
            if emit_violations
            else pd.DataFrame()
        )
        record_scores = self.scorer.score_records(work, pass_frame)
        batch_score = self.scorer.score_batch(record_scores, pass_frame, batch_id=batch_id)
        summary = self._summarise(work, pass_frame, record_scores)
        summary["batch_id"] = batch_id
        summary["computed_at"] = utcnow_iso()

        return ValidationResult(
            record_results=record_results,
            violations=violations,
            record_scores=record_scores,
            batch_score=batch_score,
            summary=summary,
        )

    def _build_violations(
        self, work: pd.DataFrame, pass_frame: pd.DataFrame, max_rows: int | None = None
    ) -> pd.DataFrame:
        chunks: list[pd.DataFrame] = []
        by_id = {r.rule_id: r for r in self.rules}
        for rule_id in pass_frame.columns:
            failed = ~pass_frame[rule_id]
            if not failed.any():
                continue
            rule = by_id[rule_id]
            idx = pass_frame.index[failed]
            sub = work.loc[idx]
            chunk = pd.DataFrame(
                {
                    "loan_id": sub.get("loan_id", pd.Series(index=idx, dtype=object)),
                    "month_index": sub.get("month_index", pd.Series(index=idx, dtype="float64")),
                    "rule_id": rule_id,
                    "rule_name": rule.name,
                    "severity": rule.severity,
                    "exception_type": rule.exception_type,
                    "dimension": rule.dimension,
                    "observed_value": self._observed_text(sub, rule_id),
                    "expected_condition": rule.condition,
                    "description": rule.description,
                }
            )
            chunks.append(chunk)
        if not chunks:
            return pd.DataFrame(
                columns=[
                    "loan_id", "month_index", "rule_id", "rule_name", "severity",
                    "exception_type", "dimension", "observed_value",
                    "expected_condition", "description",
                ]
            )
        out = pd.concat(chunks, ignore_index=True)
        out = out.sort_values(["severity", "rule_id", "loan_id", "month_index"], kind="mergesort")
        if max_rows is not None and len(out) > max_rows:
            out = out.head(max_rows)
        return out.reset_index(drop=True)

    @staticmethod
    def _observed_text(sub: pd.DataFrame, rule_id: str) -> pd.Series:
        cols = [c for c in OBSERVED_COLUMNS.get(rule_id, ()) if c in sub.columns]
        if not cols:
            return pd.Series("", index=sub.index)
        parts = []
        for c in cols:
            label = c.replace("_ctx_", "")
            values = sub[c]
            if pd.api.types.is_float_dtype(values):
                text = values.map(lambda v: "null" if pd.isna(v) else f"{v:,.4g}")
            else:
                text = values.map(lambda v: "null" if pd.isna(v) else str(v))
            parts.append(label + "=" + text)
        joined = parts[0]
        for p in parts[1:]:
            joined = joined + ", " + p
        return joined

    def _summarise(
        self, work: pd.DataFrame, pass_frame: pd.DataFrame, record_scores: pd.DataFrame
    ) -> dict[str, Any]:
        by_id = {r.rule_id: r for r in self.rules}
        per_rule = []
        for rule_id in pass_frame.columns:
            n_fail = int((~pass_frame[rule_id]).sum())
            rule = by_id[rule_id]
            per_rule.append(
                {
                    "rule_id": rule_id,
                    "name": rule.name,
                    "severity": rule.severity,
                    "dimension": rule.dimension,
                    "origin": rule.origin,
                    "exception_type": rule.exception_type,
                    "n_violations": n_fail,
                    "violation_rate": round(n_fail / max(len(pass_frame), 1), 6),
                }
            )
        per_rule.sort(key=lambda r: -r["n_violations"])

        n_error = sum(
            int((~pass_frame[r.rule_id]).sum()) for r in self.rules if r.severity == "ERROR"
        )
        n_warning = sum(
            int((~pass_frame[r.rule_id]).sum()) for r in self.rules if r.severity == "WARNING"
        )
        clean = int((pass_frame.all(axis=1)).sum())
        return {
            "n_records": int(len(pass_frame)),
            "rules_evaluated": len(self.rules),
            "n_clean_records": clean,
            "clean_rate": round(clean / max(len(pass_frame), 1), 6),
            "n_error_violations": n_error,
            "n_warning_violations": n_warning,
            "per_rule": per_rule,
            "grade_distribution": (
                record_scores["dq_grade"].value_counts().to_dict() if not record_scores.empty else {}
            ),
        }


def strip_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in (*CONTEXT_COLUMNS, "_row_pos") if c in df.columns]
    return df.drop(columns=drop)
