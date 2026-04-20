"""Logging helpers for the Trekion processing pipeline."""

from __future__ import annotations

import logging as _logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure consistent console logging."""

    numeric_level = getattr(_logging, level.upper(), _logging.INFO)
    _logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> _logging.Logger:
    return _logging.getLogger(name)
