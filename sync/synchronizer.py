"""Time synchronization between video frames and high-rate IMU samples."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from parsers.imu_parser import SENSOR_COLUMNS
from utils.validation import assert_finite, estimate_sampling_rate, require_columns

LOGGER = logging.getLogger(__name__)


def _nearest_delay_ns(frame_timestamps: np.ndarray, imu_timestamps: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(imu_timestamps, frame_timestamps)
    left_indices = np.clip(indices - 1, 0, len(imu_timestamps) - 1)
    right_indices = np.clip(indices, 0, len(imu_timestamps) - 1)

    left_delta = np.abs(frame_timestamps - imu_timestamps[left_indices])
    right_delta = np.abs(frame_timestamps - imu_timestamps[right_indices])
    return np.minimum(left_delta, right_delta).astype(np.int64)


def synchronize_streams(
    vts_df: pd.DataFrame,
    imu_df: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp_ns",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Interpolate IMU values to each video frame timestamp.

    This intentionally does not use nearest-neighbor matching for the synced
    telemetry.  Nearest raw IMU samples are used only to report delay metrics.
    """

    require_columns(vts_df, ["frame_number", timestamp_column], "VTS")
    require_columns(imu_df, [timestamp_column, *SENSOR_COLUMNS], "IMU")
    assert_finite(imu_df, SENSOR_COLUMNS, "IMU")

    vts_sorted = vts_df.sort_values(timestamp_column).reset_index(drop=True)
    imu_sorted = imu_df.sort_values(timestamp_column).drop_duplicates(timestamp_column).reset_index(drop=True)

    frame_timestamps = vts_sorted[timestamp_column].to_numpy(dtype=np.int64)
    imu_timestamps = imu_sorted[timestamp_column].to_numpy(dtype=np.int64)

    if len(imu_timestamps) < 2:
        raise ValueError("Need at least two IMU samples for interpolation")
    if not np.all(np.diff(frame_timestamps) > 0):
        raise ValueError("VTS timestamps must be strictly monotonic")
    if not np.all(np.diff(imu_timestamps) > 0):
        raise ValueError("IMU timestamps must be strictly monotonic")

    synced = pd.DataFrame(
        {
            "frame_number": vts_sorted["frame_number"].to_numpy(dtype=np.int64),
            "timestamp": frame_timestamps,
            "timestamp_ns": frame_timestamps,
        }
    )

    in_imu_range = (frame_timestamps >= imu_timestamps[0]) & (frame_timestamps <= imu_timestamps[-1])
    synced["in_imu_range"] = in_imu_range

    imu_time_float = imu_timestamps.astype(np.float64)
    frame_time_float = frame_timestamps.astype(np.float64)
    for column in SENSOR_COLUMNS:
        values = imu_sorted[column].to_numpy(dtype=np.float64)
        synced[column] = np.interp(
            frame_time_float,
            imu_time_float,
            values,
            left=np.nan,
            right=np.nan,
        )

    nearest_delay = _nearest_delay_ns(frame_timestamps, imu_timestamps)
    synced["nearest_imu_delay_ns"] = nearest_delay
    synced["nearest_imu_delay_ms"] = nearest_delay.astype(np.float64) / 1e6

    valid_delay = nearest_delay[in_imu_range]
    metrics: dict[str, Any] = {
        "frame_count": int(len(synced)),
        "imu_sample_count": int(len(imu_sorted)),
        "in_range_frame_count": int(np.sum(in_imu_range)),
        "mean_delay_ns": float(np.mean(valid_delay)),
        "median_delay_ns": float(np.median(valid_delay)),
        "max_delay_ns": int(np.max(valid_delay)),
        "mean_delay_ms": float(np.mean(valid_delay) / 1e6),
        "median_delay_ms": float(np.median(valid_delay) / 1e6),
        "max_delay_ms": float(np.max(valid_delay) / 1e6),
        "imu_rate_hz": float(estimate_sampling_rate(imu_timestamps)),
        "camera_rate_hz": float(estimate_sampling_rate(frame_timestamps)),
        "sync_method": "linear_interpolation",
    }

    LOGGER.info(
        "Synchronized %s frames using %s IMU samples; delay mean/median/max = %.3f/%.3f/%.3f ms",
        metrics["frame_count"],
        metrics["imu_sample_count"],
        metrics["mean_delay_ms"],
        metrics["median_delay_ms"],
        metrics["max_delay_ms"],
    )
    return synced, metrics
