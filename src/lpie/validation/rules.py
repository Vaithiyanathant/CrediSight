"""The 18 deterministic validation rules.

`config/validation_rules.json` is loaded as *data*: each entry compiles to a
vectorised predicate returning True = PASS. Rules are never hard-coded in the
engine, so adding a rule is a config change plus one predicate registration.

Several rules need context beyond the row itself — the previous month's balance,
the loan's prior terminal state, the as-of servicer record, the static file, the
training vocabulary. `RuleContext` carries exactly those, precomputed once per
batch, so every predicate stays a pure vectorised expression.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from lpie.core.config import Settings, get_settings
from lpie.core.exceptions import ValidationRuleError

Severity = Literal["ERROR", "WARNING"]
Predicate = Callable[[pd.DataFrame, "RuleContext"], pd.Series]

TERMINAL_STATES = ("Prepaid", "Closed")
MISSING_DOC_STATUSES = ("Missing-Income", "Missing-Appraisal", "Missing-ID")

# Columns the rule engine adds to the working frame. Prefixed so they can never
# collide with a real data column and are trivially strippable.
CTX_PREV_BALANCE = "_ctx_prev_balance"
CTX_PRIOR_TERMINAL = "_ctx_prior_terminal"
CTX_LOAN_TERM = "_ctx_loan_term_months"
CTX_STATIC_ORIG_BAL = "_ctx_static_original_balance"
CTX_STATIC_RATE = "_ctx_static_interest_rate"
CTX_SERVICER_BALANCE = "_ctx_servicer_balance"
CTX_SERVICER_STATUS = "_ctx_servicer_status"
CTX_SERVICER_RATE = "_ctx_servicer_rate"
CTX_SERVICER_DAYS = "_ctx_servicer_days_since_update"
CTX_SERVICER_STALE = "_ctx_servicer_stale_flag"
CTX_DUP_COUNT = "_ctx_dup_loan_month_count"
CTX_UNSEEN_COUNT = "_ctx_unseen_category_count"
CTX_UNSEEN_COLS = "_ctx_unseen_category_columns"
CTX_DAYS_SINCE_UPDATE = "_ctx_days_since_last_update"
CTX_MONTH_END = "_ctx_month_end"

CONTEXT_COLUMNS = (
    CTX_PREV_BALANCE, CTX_PRIOR_TERMINAL, CTX_LOAN_TERM, CTX_STATIC_ORIG_BAL,
    CTX_STATIC_RATE, CTX_SERVICER_BALANCE, CTX_SERVICER_STATUS, CTX_SERVICER_RATE,
    CTX_SERVICER_DAYS, CTX_SERVICER_STALE, CTX_DUP_COUNT, CTX_UNSEEN_COUNT, CTX_UNSEEN_COLS,
    CTX_DAYS_SINCE_UPDATE, CTX_MONTH_END,
)

CATEGORICAL_VOCAB_COLUMNS = (
    "current_status", "document_status", "source_system", "credit_score_band",
    "ltv_band", "dti_band", "state", "loan_purpose", "occupancy_type",
    "property_type", "servicer_name", "loss_severity_band",
)


@dataclass
class RuleContext:
    """Batch-level context needed by rules that look beyond a single row."""

    vocabulary: dict[str, set[str]] = field(default_factory=dict)
    has_servicer: bool = False
    has_static: bool = False
    has_history: bool = False
    dq_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Rule:
    rule_id: str
    name: str
    description: str
    field_name: str
    severity: Severity
    exception_type: str
    condition: str
    dimension: str
    origin: str
    weight: float
    predicate: Predicate

    def evaluate(self, df: pd.DataFrame, ctx: RuleContext) -> pd.Series:
        """True = PASS. Rows the rule cannot judge (missing context) PASS.

        A rule that cannot see its inputs must not manufacture a violation —
        that would penalise a record for the pipeline's own gaps.
        """
        try:
            result = self.predicate(df, ctx)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValidationRuleError(
                f"Rule {self.rule_id} ({self.name}) failed to evaluate: {exc}",
                details={"rule_id": self.rule_id},
            ) from exc
        return pd.Series(result, index=df.index).fillna(True).astype(bool)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _str(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="object")
    return df[col]


def _all_pass(df: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=df.index, dtype=bool)


# --------------------------------------------------------------------------- #
# predicates — one per rule_id
# --------------------------------------------------------------------------- #
def _vr001(df: pd.DataFrame, ctx: RuleContext) -> pd.Series:
    """current_balance <= original_balance * 1.05 unless modification_flag = 1."""
    cur, orig = _num(df, "current_balance"), _num(df, "original_balance")
    mod = _num(df, "modification_flag").fillna(0)
    ok = (cur <= orig * 1.05) | (mod == 1)
    return ok.where(cur.notna() & orig.notna(), True)


_DPD_BANDS: dict[str, tuple[float, float]] = {
    "Current": (0.0, 0.0),
    "30DPD": (25.0, 35.0),
    "60DPD": (55.0, 65.0),
    "90DPD": (85.0, 120.0),
}


def _vr002(df: pd.DataFrame, ctx: RuleContext) -> pd.Series:
    """days_past_due must lie in the band implied by current_status."""
    dpd, status = _num(df, "days_past_due"), _str(df, "current_status")
    ok = _all_pass(df)
    for state, (lo, hi) in _DPD_BANDS.items():
        m = (status == state) & dpd.notna()
        if m.any():
            ok.loc[m] = (dpd[m] >= lo) & (dpd[m] <= hi)
    # Terminal and Default rows carry no DPD contract; unknown DPD cannot violate.
    return ok.where(dpd.notna(), True)


def _vr003(df: pd.DataFrame, ctx: RuleContext) -> pd.Series:
    """reporting_month >= origination_month, and loan_age matches the month gap.

    The age identity is checked with a +/-1 month tolerance because
    reporting_month itself is corrupted in this pack (month_index 1 and 2 both
    map to 2021-01, shifting every subsequent month by one). VR-013 owns that
    corruption; VR-003 must not double-count it as 360,000 age violations.
    """
    rep = pd.to_datetime(_str(df, "reporting_month"), format="%Y-%m", errors="coerce")
    orig = pd.to_datetime(_str(df, "origination_month"), format="%Y-%m", errors="coerce")
    known = rep.notna() & orig.notna()
    order_ok = rep >= orig

    gap = (rep.dt.year - orig.dt.year) * 12 + (rep.dt.month - orig.dt.month)
    age = _num(df, "loan_age_months")
    age_ok = ((age - gap).abs() <= 1) | age.isna() | gap.isna()

    return (order_ok & age_ok).where(known, True)


def _vr004(df: pd.DataFrame, ctx: RuleContext) -> pd.Series:
    """prepayment_flag = 1 implies current_balance = 0."""
    flag, bal = _num(df, "prepayment_flag").fillna(0), _num(df, "current_balance")
    ok = (flag != 1) | (bal.abs() < 1e-6)
    return ok.where(bal.notna(), True)


def _vr005(df: pd.DataFrame, ctx: RuleContext) -> pd.Series:
    """current_status = Closed implies current_balance = 0."""
    status, bal = _str(df, "current_status"), _num(df, "current_balance")
    ok = (status != "Closed") | (bal.abs() < 1e-6)
    return ok.where(bal.notna(), True)


def _vr006(df: pd.DataFrame, ctx: RuleContext) -> pd.Series:
    """document_status must not be one of the Missing-* levels."""
    doc = _str(df, "document_status")
    return ~doc.isin(MISSING_DOC_STATUSES)


def _vr007(df: pd.DataFrame, ctx: RuleContext) -> pd.Series:
    """Servicer-reported balance within 10% of master (as-of join, backward only)."""
    if not ctx.has_servicer or CTX_SERVICER_BALANCE not in df.columns:
        return _all_pass(df)
    cur, srv = _num(df, "current_balance"), _num(df, CTX_SERVICER_BALANCE)
    denom = cur.replace(0.0, np.nan)
    gap = (cur - srv).abs() / denom
    ok = gap <= 0.10
    # No servicer record, or a zero master balance, means the rule is silent.
    return ok.where(srv.notna() & denom.notna(), True)


def _vr008(df: pd.DataFrame, ctx: RuleContext) -> pd.Series:
    """1.0 <= interest_rate <= 20.0."""
    rate = _num(df, "interest_rate")
    ok = (rate >= 1.0) & (rate <= 20.0)
    return ok.where(rate.notna(), True)


def _vr009(df: pd.DataFrame, ctx: RuleContext) -> pd.Series:
    """0 <= remaining_term_months <= loan_term_months."""
    rem = _num(df, "remaining_term_months")
    term = _num(df, CTX_LOAN_TERM) if CTX_LOAN_TERM in df.columns else pd.Series(np.nan, index=df.index)
    lower_ok = rem >= 0
    upper_ok = (rem <= term) | term.isna()
    ok = lower_ok & upper_ok
    return ok.where(rem.notna(), True)


def _vr010(df: pd.DataFrame, ctx: RuleContext) -> pd.Series:
    """No source is stale: master record and as-of servicer record both <= 65 days old.

    Covers both staleness channels the pack contains (SYSTEM_DESIGN 1.5 maps
    servicer staleness to VR-007/VR-010): the master `last_updated_at`, and the
    age of the most recent servicer update visible at this month end.
    """
    ok = _all_pass(df)
    if CTX_DAYS_SINCE_UPDATE in df.columns:
        days = _num(df, CTX_DAYS_SINCE_UPDATE)
        ok &= (days <= 65).where(days.notna(), True)
    if ctx.has_servicer and CTX_SERVICER_STALE in df.columns:
        # The servicer file carries its own stale determination; we honour it
        # rather than re-deriving an age, because update cadence varies by
        # servicer and a slow-but-current feed is not a data defect.
        stale = _num(df, CTX_SERVICER_STALE)
        ok &= (stale != 1).where(stale.notna(), True)
    return ok.astype(bool)


def _vr011(df: pd.DataFrame, ctx: RuleContext) -> pd.Series:
    """loss_severity_band is non-N/A only when default_flag = 1."""
    flag = _num(df, "default_flag").fillna(0)
    band = _str(df, "loss_severity_band")
    non_na = band.notna() & (band.astype("string") != "N/A")
    ok = (flag == 1) | ~non_na
    return ok.astype(bool)


def _vr012(df: pd.DataFrame, ctx: RuleContext) -> pd.Series:
    """90DPD / Default without a modification flag requires review."""
    status, mod = _str(df, "current_status"), _num(df, "modification_flag")
    severe = status.isin(["90DPD", "Default"])
    ok = ~(severe & (mod.fillna(0) == 0))
    return ok.astype(bool)


def _vr013(df: pd.DataFrame, ctx: RuleContext) -> pd.Series:
    """(loan_id, reporting_month) must be unique within the batch."""
    if CTX_DUP_COUNT not in df.columns:
        return _all_pass(df)
    return _num(df, CTX_DUP_COUNT).fillna(1) <= 1


def _vr014(df: pd.DataFrame, ctx: RuleContext) -> pd.Series:
    """Balance must not rise above the previous month by >0.1% without a modification."""
    if CTX_PREV_BALANCE not in df.columns:
        return _all_pass(df)
    cur, prev = _num(df, "current_balance"), _num(df, CTX_PREV_BALANCE)
    mod = _num(df, "modification_flag").fillna(0)
    ok = (cur <= prev * 1.001) | (mod == 1)
    return ok.where(cur.notna() & prev.notna(), True)


def _vr015(df: pd.DataFrame, ctx: RuleContext) -> pd.Series:
    """Once Prepaid or Closed, the loan must stay in that terminal state."""
    if CTX_PRIOR_TERMINAL not in df.columns:
        return _all_pass(df)
    prior, status = _str(df, CTX_PRIOR_TERMINAL), _str(df, "current_status")
    has_prior = prior.notna()
    ok = ~has_prior | (status == prior)
    return ok.astype(bool)


def _vr016(df: pd.DataFrame, ctx: RuleContext) -> pd.Series:
    """loan_age + remaining_term == loan_term (+/- 1)."""
    if CTX_LOAN_TERM not in df.columns:
        return _all_pass(df)
    age, rem, term = _num(df, "loan_age_months"), _num(df, "remaining_term_months"), _num(df, CTX_LOAN_TERM)
    ok = (age + rem - term).abs() <= 1
    return ok.where(age.notna() & rem.notna() & term.notna(), True)


def _vr017(df: pd.DataFrame, ctx: RuleContext) -> pd.Series:
    """Monthly original_balance / interest_rate must match the static file."""
    if not ctx.has_static or CTX_STATIC_ORIG_BAL not in df.columns:
        return _all_pass(df)
    bal_ok = (_num(df, "original_balance") - _num(df, CTX_STATIC_ORIG_BAL)).abs() <= 0.01
    rate_ok = (_num(df, "interest_rate") - _num(df, CTX_STATIC_RATE)).abs() <= 0.001
    known = _num(df, CTX_STATIC_ORIG_BAL).notna()
    return (bal_ok & rate_ok).where(known, True)


def _vr018(df: pd.DataFrame, ctx: RuleContext) -> pd.Series:
    """No categorical level outside the training vocabulary."""
    if CTX_UNSEEN_COUNT not in df.columns:
        return _all_pass(df)
    return _num(df, CTX_UNSEEN_COUNT).fillna(0) == 0


PREDICATES: dict[str, Predicate] = {
    "VR-001": _vr001, "VR-002": _vr002, "VR-003": _vr003, "VR-004": _vr004,
    "VR-005": _vr005, "VR-006": _vr006, "VR-007": _vr007, "VR-008": _vr008,
    "VR-009": _vr009, "VR-010": _vr010, "VR-011": _vr011, "VR-012": _vr012,
    "VR-013": _vr013, "VR-014": _vr014, "VR-015": _vr015, "VR-016": _vr016,
    "VR-017": _vr017, "VR-018": _vr018,
}

# The observed value shown to a reviewer for each rule. Deliberately explicit:
# "VR-001: current_balance 312,400 exceeds original_balance 279,000 x 1.05".
OBSERVED_COLUMNS: dict[str, tuple[str, ...]] = {
    "VR-001": ("current_balance", "original_balance", "modification_flag"),
    "VR-002": ("current_status", "days_past_due"),
    "VR-003": ("reporting_month", "origination_month", "loan_age_months"),
    "VR-004": ("prepayment_flag", "current_balance"),
    "VR-005": ("current_status", "current_balance"),
    "VR-006": ("document_status",),
    "VR-007": ("current_balance", CTX_SERVICER_BALANCE),
    "VR-008": ("interest_rate",),
    "VR-009": ("remaining_term_months", CTX_LOAN_TERM),
    "VR-010": (CTX_DAYS_SINCE_UPDATE, "last_updated_at", CTX_SERVICER_STALE, CTX_SERVICER_DAYS),
    "VR-011": ("default_flag", "loss_severity_band"),
    "VR-012": ("current_status", "modification_flag"),
    "VR-013": ("reporting_month", CTX_DUP_COUNT),
    "VR-014": ("current_balance", CTX_PREV_BALANCE, "modification_flag"),
    "VR-015": ("current_status", CTX_PRIOR_TERMINAL),
    "VR-016": ("loan_age_months", "remaining_term_months", CTX_LOAN_TERM),
    "VR-017": ("original_balance", CTX_STATIC_ORIG_BAL, "interest_rate", CTX_STATIC_RATE),
    "VR-018": (CTX_UNSEEN_COLS,),
}


def load_rules(settings: Settings | None = None, path: Path | str | None = None) -> list[Rule]:
    """Compile `config/validation_rules.json` into executable Rule objects."""
    s = settings or get_settings()
    rules_path = Path(path) if path else (s.root / "config" / "validation_rules.json")
    if not rules_path.exists():
        rules_path = s.dataset_file("validation_rules")
    spec = json.loads(rules_path.read_text())

    rules: list[Rule] = []
    for item in spec["rules"]:
        rid = item["rule_id"]
        predicate = PREDICATES.get(rid)
        if predicate is None:
            raise ValidationRuleError(
                f"No predicate registered for rule {rid}. "
                "Every rule in validation_rules.json must have an implementation.",
                details={"rule_id": rid},
            )
        rules.append(
            Rule(
                rule_id=rid,
                name=item["name"],
                description=item["description"],
                field_name=item.get("field", ""),
                severity=item["severity"],
                exception_type=item["exception_type"],
                condition=item.get("condition", ""),
                dimension=item.get("dimension", "consistency"),
                origin=item.get("origin", "supplied"),
                weight=float(item.get("weight", 1.0)),
                predicate=predicate,
            )
        )
    return rules


def rules_by_dimension(rules: list[Rule]) -> dict[str, list[Rule]]:
    out: dict[str, list[Rule]] = {}
    for r in rules:
        out.setdefault(r.dimension, []).append(r)
    return out
