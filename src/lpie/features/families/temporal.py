"""F8 — Temporal / seasonality (8).

`month_index` carries a deliberate per-head policy. It is included in the hazard
model, where calendar time is a modelled baseline and the one-month-ahead label
is uncensored for every row up to month 35. It is **excluded from the
direct-horizon heads**, because those labels collapse to zero in the trailing
months (measured: `next_3m` decays 0.0878 -> 0.0663 -> 0.0421 -> 0.0000 across
months 33-36), so a head that can see `month_index` will learn "recent month =>
no events" and then be scored on months 37-42, where that rule is false.

That is Insight 1 of the design, encoded as an allow-list rather than a comment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lpie.features.registry import ALL_HEADS, DIRECT_HORIZON_HEADS, FeatureSpec, spec

FAMILY = "temporal"

_NON_DIRECT_HEADS = tuple(h for h in ALL_HEADS if h not in DIRECT_HORIZON_HEADS)

SPECS: list[FeatureSpec] = [
    spec("month_index", FAMILY,
         "Panel month. Hazard/anomaly/exception only — banned from the direct-horizon "
         "heads, whose labels are censored at the panel edge.",
         ["month_index"], allowed_heads=_NON_DIRECT_HEADS,
         leakage_risk="medium",
         justification=(
             "Not leakage in the point-in-time sense (month_index is known at scoring "
             "time), but a censoring trap for the direct-horizon heads: their label "
             "rate collapses to zero in months 34-36, so the head would learn a "
             "spurious 'late month => no event' rule and mis-score months 37-42. "
             "Permitted only where the label is uncensored (1-step hazard, "
             "contemporaneous exception, unsupervised anomaly)."
         )),
    spec("calendar_month_sin", FAMILY, "sin(2*pi*month/12) — cyclic calendar encoding",
         ["month_index"]),
    spec("calendar_month_cos", FAMILY, "cos(2*pi*month/12) — cyclic calendar encoding",
         ["month_index"]),
    spec("quarter", FAMILY, "Calendar quarter 1-4", ["month_index"]),
    spec("is_year_end", FAMILY, "Reporting month is December", ["month_index"]),
    spec("months_since_panel_start", FAMILY, "Months elapsed since the first panel month",
         ["month_index"], allowed_heads=_NON_DIRECT_HEADS,
         leakage_risk="medium",
         justification="Monotone in month_index; carries the same censoring trap. Same policy applies."),
    spec("months_observed", FAMILY, "Number of months of history observed for this loan up to t",
         ["month_index"], temporal_offset=-1),
    spec("is_first_observation", FAMILY, "First observed month for this loan — the cold-start indicator",
         ["month_index"]),
]

PANEL_ANCHOR_MONTH = 1  # January


def build(panel: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=panel.index)
    month = pd.to_numeric(panel["month_index"], errors="coerce")
    loan = panel["loan_id"]

    out["month_index"] = month
    panel_start = float(month.min())
    out["months_since_panel_start"] = month - panel_start

    # Calendar month derived from month_index, not reporting_month: reporting_month
    # is corrupted in this pack (VR-013) and would inject the corruption into a
    # seasonality feature.
    calendar_month = ((month - 1 + (PANEL_ANCHOR_MONTH - 1)) % 12) + 1
    out["calendar_month_sin"] = np.sin(2.0 * np.pi * calendar_month / 12.0)
    out["calendar_month_cos"] = np.cos(2.0 * np.pi * calendar_month / 12.0)
    out["quarter"] = np.ceil(calendar_month / 3.0)
    out["is_year_end"] = (calendar_month == 12).astype("float64")

    out["months_observed"] = month.groupby(loan, sort=False).cumcount().astype("float64")
    out["is_first_observation"] = (out["months_observed"] == 0).astype("float64")
    return out
