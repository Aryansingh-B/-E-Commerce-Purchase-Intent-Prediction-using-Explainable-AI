"""Data ingestion layer (PRD §4.1).

A single loader function/class that every notebook, script, the API, and the
dashboard call, so they all read the CSV the same way. Fails loudly with a
clear log message on missing/malformed input.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.settings import Settings, get_settings
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
    "Converted",
]

EXPECTED_DTYPE_KINDS = {
    "Administrative": "i",
    "Administrative_Duration": "f",
    "Informational": "i",
    "Informational_Duration": "f",
    "ProductRelated": "i",
    "ProductRelated_Duration": "f",
    "BounceRates": "f",
    "ExitRates": "f",
    "PageValues": "f",
    "SpecialDay": "f",
    "OperatingSystems": "i",
    "Browser": "i",
    "Region": "i",
    "TrafficType": "i",
    "Converted": "i",
}


class DataValidationError(Exception):
    """Raised when the raw CSV does not match the expected schema."""


class SessionDataLoader:
    """Loads and validates ecommerce_sessions.csv.

    Usage:
        loader = SessionDataLoader()
        df = loader.load()
    """

    def __init__(self, settings: Settings | None = None, path: str | Path | None = None):
        self.settings = settings or get_settings()
        self.path = Path(path) if path else self.settings.paths.resolve("raw_data")

    def load(self, validate: bool = True) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Expected dataset at {self.path} but it does not exist. "
                "Check src/config/config.yaml -> paths.raw_data."
            )
        try:
            df = pd.read_csv(self.path)
        except Exception as exc:  # noqa: BLE001 - surfaced with context, not swallowed
            raise DataValidationError(f"Failed to parse CSV at {self.path}: {exc}") from exc

        logger.info("Loaded %d rows, %d columns from %s", len(df), df.shape[1], self.path)

        if validate:
            self.validate_schema(df)

        return df

    def validate_schema(self, df: pd.DataFrame) -> None:
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            raise DataValidationError(f"Missing expected columns: {sorted(missing)}")

        extra = set(df.columns) - set(EXPECTED_COLUMNS)
        if extra:
            logger.warning(
                "Unexpected extra columns present (ignored downstream): %s", sorted(extra)
            )

        for col, _kind in EXPECTED_DTYPE_KINDS.items():
            if col not in df.columns:
                continue
            actual_kind = df[col].dtype.kind
            if actual_kind not in ("i", "u", "f"):
                raise DataValidationError(
                    f"Column '{col}' expected numeric dtype, got {df[col].dtype}"
                )

        if df.empty:
            raise DataValidationError("Loaded dataframe is empty.")

        n_dupes = df.duplicated().sum()
        if n_dupes:
            logger.warning("%d exact duplicate rows found in raw data.", n_dupes)

        logger.info("Schema validation passed.")

    @staticmethod
    def expected_columns() -> list[str]:
        return list(EXPECTED_COLUMNS)


def load_sessions(settings: Settings | None = None) -> pd.DataFrame:
    """Convenience function wrapping SessionDataLoader for simple call sites."""
    return SessionDataLoader(settings=settings).load()
