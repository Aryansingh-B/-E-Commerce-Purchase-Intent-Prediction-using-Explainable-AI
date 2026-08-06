"""Behavioural feature engineering (PRD §4.4).

A handful of intent-capturing ratios added to the raw columns, each with a
plain-English rationale a stakeholder could follow:

- total_pages: total pages viewed across all three page types. Simple
  engagement-depth signal.
- total_duration: total time spent across all three page types.
- avg_time_per_page: total_duration / total_pages. Distinguishes someone
  lingering vs. rapidly skimming.
- product_page_share: ProductRelated / total_pages. Share of the session
  spent actually looking at products vs. admin/info pages — a purchase-intent
  proxy that is easy to explain to a CRO team.
- product_time_share: ProductRelated_Duration / total_duration. Same idea,
  weighted by time instead of page count.
- is_returning_visitor: binary flag from VisitorType == 'Returning_Visitor'.
  Kept alongside the categorical VisitorType column as an explicit, cheap
  signal many tree models split on early.

All ratios guard against divide-by-zero (sessions with zero page views are
common in this dataset per the student hint on skewed durations).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

ENGINEERED_FEATURE_NAMES = [
    "total_pages",
    "total_duration",
    "avg_time_per_page",
    "product_page_share",
    "product_time_share",
    "is_returning_visitor",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with the engineered behavioural columns appended."""
    out = df.copy()

    out["total_pages"] = out["Administrative"] + out["Informational"] + out["ProductRelated"]
    out["total_duration"] = (
        out["Administrative_Duration"]
        + out["Informational_Duration"]
        + out["ProductRelated_Duration"]
    )
    out["avg_time_per_page"] = np.where(
        out["total_pages"] > 0, out["total_duration"] / out["total_pages"], 0.0
    )
    out["product_page_share"] = np.where(
        out["total_pages"] > 0, out["ProductRelated"] / out["total_pages"], 0.0
    )
    out["product_time_share"] = np.where(
        out["total_duration"] > 0,
        out["ProductRelated_Duration"] / out["total_duration"],
        0.0,
    )
    out["is_returning_visitor"] = (out["VisitorType"] == "Returning_Visitor").astype(int)

    logger.info(
        "Added %d engineered features: %s", len(ENGINEERED_FEATURE_NAMES), ENGINEERED_FEATURE_NAMES
    )
    return out
