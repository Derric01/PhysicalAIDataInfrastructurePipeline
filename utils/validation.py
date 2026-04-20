"""Validation and timestamp normalization utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimestampStats:
    stream_name: str
    unit: str
    scale_to_ns: int
    min_ns: int
    max_ns: int
    duration_s: float
    estimated_rate_hz: float


def estimate_sampling_rate(timestamp_ns: np.ndarray) -> float:
    """Estimate sample rate from monotonic nanosecond timestamps."""

    if len(timestamp_ns) < 2:
        return 0.0
    duration_s = (float(timestamp_ns[-1]) - float(timestamp_ns[0])) / 1e9
    if duration_s <= 0:
        return 0.0
    return (len(timestamp_ns) - 1) / duration_s


def detect_timestamp_unit(timestamps: pd.Series | np.ndarray) -> tuple[str, int]:
    """Infer timestamp units from typical adjacent deltas.

    Robotics logs commonly use either microseconds or nanoseconds.  The
    heuristic is based on median positive deltas rather than absolute magnitude
    so relative hardware clocks are handled correctly.
    """

    values = np.asarray(timestamps, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return "ns", 1

    deltas = np.diff(np.sort(values))
    positive = deltas[deltas > 0]
    if len(positive) == 0:
        return "ns", 1

    median_delta = float(np.median(positive))
    if median_delta >= 1_000_000.0:
        return "ns", 1
    if median_delta >= 1_000.0:
        return "us", 1_000
    if median_delta >= 1.0:
        return "ms", 1_000_000
    return "s", 1_000_000_000


def normalize_timestamps(
    df: pd.DataFrame,
    *,
    column: str = "timestamp",
    output_column: str = "timestamp_ns",
    stream_name: str,
) -> tuple[pd.DataFrame, TimestampStats]:
    """Add a normalized nanosecond timestamp column and print debug stats."""

    if column not in df.columns:
        raise KeyError(f"{stream_name} dataframe is missing timestamp column {column!r}")

    unit, scale = detect_timestamp_unit(df[column])
    normalized = df.copy()
    normalized[output_column] = (normalized[column].astype("int64") * scale).astype("int64")
    normalized = normalized.sort_values(output_column).reset_index(drop=True)

    timestamp_ns = normalized[output_column].to_numpy(dtype=np.int64)
    if len(timestamp_ns) > 1 and not np.all(np.diff(timestamp_ns) > 0):
        raise ValueError(f"{stream_name} timestamps are not strictly monotonic after normalization")

    rate_hz = estimate_sampling_rate(timestamp_ns)
    duration_s = (int(timestamp_ns[-1]) - int(timestamp_ns[0])) / 1e9 if len(timestamp_ns) > 1 else 0.0
    stats = TimestampStats(
        stream_name=stream_name,
        unit=unit,
        scale_to_ns=scale,
        min_ns=int(timestamp_ns[0]),
        max_ns=int(timestamp_ns[-1]),
        duration_s=duration_s,
        estimated_rate_hz=rate_hz,
    )

    LOGGER.info(
        "%s timestamps: unit=%s min_ns=%s max_ns=%s duration=%.3fs estimated_rate=%.3fHz",
        stream_name,
        stats.unit,
        stats.min_ns,
        stats.max_ns,
        stats.duration_s,
        stats.estimated_rate_hz,
    )
    return normalized, stats


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{name} dataframe is missing required columns: {missing}")


def assert_finite(df: pd.DataFrame, columns: list[str], name: str) -> None:
    values = df[columns].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} dataframe contains non-finite values in {columns}")
