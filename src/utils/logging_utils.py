"""Centralised logging setup.

Every module gets a configured logger via get_logger(__name__) instead of
print(). One place controls format/level for notebooks, scripts, API, and
dashboard alike.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def _configure_root(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with the shared format configured."""
    _configure_root(level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
