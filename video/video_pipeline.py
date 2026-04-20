"""Video rendering pipelines for synchronized telemetry output."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from parsers.imu_parser import SENSOR_COLUMNS
from visualization.hud import draw_frame_hud, draw_sidebar_summary
from visualization.plots import draw_xyz_plot

LOGGER = logging.getLogger(__name__)


def _video_fps(cap: cv2.VideoCapture, fallback: float = 30.0) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    return fps if np.isfinite(fps) and fps > 1.0 else fallback


def _open_writer(output_path: str | Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer for {output_path}")
    return writer


def render_imu_sync_video(
    *,
    video_path: str | Path,
    telemetry_df: pd.DataFrame,
    metrics: dict[str, Any],
    output_path: str | Path,
    max_frames: int | None = None,
    plot_window_seconds: float = 3.0,
    sidebar_width: int = 560,
) -> int:
    """Render synchronized video + IMU HUD using OpenCV drawing only."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = _video_fps(cap, fallback=float(metrics.get("camera_rate_hz", 30.0) or 30.0))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video dimensions for {video_path}: {width}x{height}")

    writer = _open_writer(output_path, fps, (width + sidebar_width, height))
    telemetry = telemetry_df.reset_index(drop=True)
    window_frames = max(2, int(round(plot_window_seconds * fps)))

    arrays = {column: telemetry[column].to_numpy(dtype=np.float64) for column in SENSOR_COLUMNS}

    frame_index = 0
    LOGGER.info("Rendering IMU HUD video to %s", output_path)
    while True:
        if max_frames is not None and frame_index >= max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break
        if frame_index >= len(telemetry):
            LOGGER.warning("Video stream has more frames than telemetry; stopping at frame %s", frame_index)
            break

        row = telemetry.iloc[frame_index]
        canvas = np.zeros((height, width + sidebar_width, 3), dtype=np.uint8)
        canvas[:, :width] = frame

        draw_frame_hud(canvas[:, :width], row, metrics, camera_fps=fps)
        draw_sidebar_summary(canvas, width, 0, row, metrics)

        start = max(0, frame_index - window_frames + 1)
        stop = frame_index + 1
        panel_x = width + 18
        panel_w = sidebar_width - 36

        draw_xyz_plot(
            canvas,
            {axis: arrays[f"accel_{axis}"][start:stop] for axis in ("x", "y", "z")},
            x=panel_x,
            y=130,
            w=panel_w,
            h=245,
            title="Accelerometer",
            unit="m/s2",
        )
        draw_xyz_plot(
            canvas,
            {axis: arrays[f"gyro_{axis}"][start:stop] for axis in ("x", "y", "z")},
            x=panel_x,
            y=405,
            w=panel_w,
            h=245,
            title="Gyroscope",
            unit="deg/s",
        )
        draw_xyz_plot(
            canvas,
            {axis: arrays[f"mag_{axis}"][start:stop] for axis in ("x", "y", "z")},
            x=panel_x,
            y=680,
            w=panel_w,
            h=245,
            title="Magnetometer",
            unit="uT",
        )

        writer.write(canvas)
        frame_index += 1
        if frame_index % 100 == 0:
            LOGGER.info("Rendered %s IMU HUD frames", frame_index)

    cap.release()
    writer.release()
    LOGGER.info("Finished IMU HUD video: %s frames -> %s", frame_index, output_path)
    return frame_index
