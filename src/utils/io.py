"""Small IO helpers for persisting/loading joblib artifacts and JSON reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def save_artifact(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info("Saved artifact -> %s", path)


def load_artifact(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found at {path}")
    return joblib.load(path)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    logger.info("Saved JSON -> %s", path)


def load_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found at {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
