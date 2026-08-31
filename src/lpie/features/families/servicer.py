"""F7 — Cross-source reconciliation (10), from `servicer_updates.csv`.

Join discipline is the whole story: strictly `update_date <= month_end(t)`,
implemented as a backward `merge_asof`. A naive equi-join would import future
servicer updates and leak the outcome the model is trying to predict.

Coverage is only 50% of loans, so `has_servicer_record` is itself a feature —
and because the missingness is MCAR by construction (the file simply covers
5,000 of 10,000 loans), using it is safe rather than a hidden drift bomb.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lpie.features.registry import FeatureSpec, spec

FAMILY = "servicer"

CONFLICT_TYPES = ("balance_mismatch", "status_conflict", "stale_record", "rate_discrepancy")

SPECS: list[FeatureSpec] = [
    spec("has_servicer_record", FAMILY,
         "A servicer update dated on or before this month end exists for this loan",
         ["loan_id", "update_date"], temporal_offset=-1),
    spec("servicer_bal_gap_pct", FAMILY, "(master - servicer) balance gap as a share of master balance",
         ["current_balance", "reported_balance"], temporal_offset=-1),
    spec("servicer_rate_gap_bps", FAMILY, "Master minus servicer rate, in basis points",
         ["interest_rate", "reported_rate"], temporal_offset=-1),
    spec("servicer_status_matches", FAMILY, "Servicer-reported status equals master status",
         ["current_status", "reported_status"], temporal_offset=-1),
    spec("days_since_servicer_update", FAMILY, "Age of the most recent visible servicer update",
         ["update_date", "month_index"], temporal_offset=-1),
    spec("servicer_stale_flag", FAMILY, "The as-of servicer record is flagged stale",
         ["stale_flag"], temporal_offset=-1),
    spec("n_servicer_updates_to_date", FAMILY, "Servicer updates observed for this loan up to t",
         ["update_date"], temporal_offset=-1),
    spec("n_conflicts_life", FAMILY, "Servicer updates with a non-'none' conflict_type up to t",
         ["conflict_type"], temporal_offset=-1),
    *[
        spec(f"servicer_conflict_{c}", FAMILY, f"Most recent visible conflict_type is {c}",
             ["conflict_type"], temporal_offset=-1)
        for c in CONFLICT_TYPES
    ],
]


def build(
    panel: pd.DataFrame,
    servicer: pd.DataFrame | None,
    static_features: pd.DataFrame,
    month_end: pd.Series,
) -> pd.DataFrame:
    out = pd.DataFrame(index=panel.index)
    n = len(panel)

    if servicer is None or servicer.empty:
        out["has_servicer_record"] = 0.0
        for c in (
            "servicer_bal_gap_pct", "servicer_rate_gap_bps", "servicer_status_matches",
            "days_since_servicer_update", "servicer_stale_flag",
        ):
            out[c] = np.nan
        out["n_servicer_updates_to_date"] = 0.0
        out["n_conflicts_life"] = 0.0
        for c in CONFLICT_TYPES:
            out[f"servicer_conflict_{c}"] = 0.0
        return out

    left = pd.DataFrame(
        {"loan_id": panel["loan_id"].to_numpy(), "_month_end": pd.to_datetime(month_end).to_numpy()}
    )
    left["_pos"] = np.arange(n)
    left = left.dropna(subset=["_month_end"]).sort_values("_month_end", kind="mergesort")

    right = servicer.copy()
    right["update_date"] = pd.to_datetime(right["update_date"], errors="coerce")
    right = right.dropna(subset=["update_date"]).sort_values("update_date", kind="mergesort")
    right["_is_conflict"] = (
        right["conflict_type"].notna() & (right["conflict_type"].astype(str) != "none")
    ).astype("float64")
    right["_update_seq"] = right.groupby("loan_id", sort=False).cumcount() + 1
    right["_conflicts_to_date"] = right.groupby("loan_id", sort=False)["_is_conflict"].cumsum()

    merged = pd.merge_asof(
        left,
        right[
            [
                "loan_id", "update_date", "reported_balance", "reported_status", "reported_rate",
                "conflict_type", "stale_flag", "_update_seq", "_conflicts_to_date",
            ]
        ],
        left_on="_month_end",
        right_on="update_date",
        by="loan_id",
        direction="backward",
    ).set_index("_pos")

    def take(col: str) -> pd.Series:
        s = pd.Series(np.nan, index=range(n), dtype="float64")
        if col in merged.columns:
            aligned = merged[col].reindex(range(n))
            s = pd.to_numeric(aligned, errors="coerce")
        s.index = panel.index
        return s

    def take_obj(col: str) -> pd.Series:
        s = merged[col].reindex(range(n)) if col in merged.columns else pd.Series(pd.NA, index=range(n))
        s.index = panel.index
        return s

    rep_bal = take("reported_balance")
    rep_rate = take("reported_rate")
    rep_status = take_obj("reported_status")
    upd_date = pd.to_datetime(merged["update_date"].reindex(range(n)))
    upd_date.index = panel.index

    out["has_servicer_record"] = rep_bal.notna().astype("float64")

    master_bal = pd.to_numeric(panel["current_balance"], errors="coerce")
    out["servicer_bal_gap_pct"] = (
        (master_bal - rep_bal) / master_bal.replace(0.0, np.nan)
    ).clip(-50, 50)

    master_rate = pd.to_numeric(static_features["interest_rate"], errors="coerce")
    out["servicer_rate_gap_bps"] = (master_rate - rep_rate) * 100.0

    out["servicer_status_matches"] = (
        (rep_status.astype("object") == panel["current_status"].astype("object"))
        .where(rep_status.notna(), np.nan)
        .astype("float64")
    )

    me = pd.to_datetime(month_end)
    me.index = panel.index
    out["days_since_servicer_update"] = (me - upd_date).dt.days.astype("float64")
    out["servicer_stale_flag"] = take("stale_flag")
    out["n_servicer_updates_to_date"] = take("_update_seq").fillna(0.0)
    out["n_conflicts_life"] = take("_conflicts_to_date").fillna(0.0)

    conflict = take_obj("conflict_type").astype("object")
    for c in CONFLICT_TYPES:
        out[f"servicer_conflict_{c}"] = (conflict == c).astype("float64")
    return out
