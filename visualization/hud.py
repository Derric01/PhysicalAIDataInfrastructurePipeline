"""HUD overlay rendering for synchronized telemetry videos."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pandas as pd

from visualization.plots import MUTED, PANEL, TEXT, put_text


def _value(row: pd.Series, key: str, default: float = float("nan")) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def draw_frame_hud(
    frame: np.ndarray,
    row: pd.Series,
    metrics: dict[str, Any],
    *,
    camera_fps: float,
) -> None:
    """Draw telemetry directly over the camera frame."""

    overlay = frame.copy()
    cv2.rectangle(overlay, (18, 18), (650, 246), PANEL, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (18, 18), (650, 246), (82, 98, 116), 1, cv2.LINE_AA)

    frame_number = int(row["frame_number"])
    timestamp_ns = int(row["timestamp_ns"])
    put_text(frame, "TREKION SENSOR SYNC", (34, 48), scale=0.78, color=(92, 231, 154), thickness=2)
    put_text(frame, f"Frame {frame_number:04d}  |  Timestamp {timestamp_ns} ns", (34, 82), scale=0.54)
    put_text(
        frame,
        f"Accel  [{_value(row, 'accel_x'):6.2f}, {_value(row, 'accel_y'):6.2f}, {_value(row, 'accel_z'):6.2f}] m/s2",
        (34, 118),
        scale=0.52,
    )
    put_text(
        frame,
        f"Gyro   [{_value(row, 'gyro_x'):6.2f}, {_value(row, 'gyro_y'):6.2f}, {_value(row, 'gyro_z'):6.2f}] deg/s",
        (34, 148),
        scale=0.52,
    )
    put_text(
        frame,
        f"Mag    [{_value(row, 'mag_x'):6.2f}, {_value(row, 'mag_y'):6.2f}, {_value(row, 'mag_z'):6.2f}] uT",
        (34, 178),
        scale=0.52,
    )
    put_text(
        frame,
        (
            f"Temp {_value(row, 'temp'):5.2f} C  |  Camera {camera_fps:.2f} FPS  |  "
            f"IMU {metrics.get('imu_rate_hz', 0.0):.2f} Hz"
        ),
        (34, 210),
        scale=0.52,
    )
    put_text(
        frame,
        (
            "Sync delay mean/median/max "
            f"{metrics.get('mean_delay_ms', 0.0):.3f}/"
            f"{metrics.get('median_delay_ms', 0.0):.3f}/"
            f"{metrics.get('max_delay_ms', 0.0):.3f} ms"
        ),
        (34, 236),
        scale=0.46,
        color=MUTED,
    )


def draw_sidebar_summary(
    canvas: np.ndarray,
    x: int,
    y: int,
    row: pd.Series,
    metrics: dict[str, Any],
) -> None:
    cv2.rectangle(canvas, (x, y), (canvas.shape[1], canvas.shape[0]), (13, 15, 19), -1)
    put_text(canvas, "LIVE TELEMETRY", (x + 18, y + 38), scale=0.72, color=(92, 231, 154), thickness=2)
    put_text(canvas, f"Interpolation: {metrics.get('sync_method', 'unknown')}", (x + 18, y + 70), scale=0.46, color=MUTED)
    put_text(canvas, f"Nearest IMU delay: {_value(row, 'nearest_imu_delay_ms'):0.3f} ms", (x + 18, y + 96), scale=0.48, color=TEXT)
