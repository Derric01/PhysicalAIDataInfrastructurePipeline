"""OpenCV plotting primitives used by the HUD video renderer."""

from __future__ import annotations

import cv2
import numpy as np

BG = (16, 18, 22)
PANEL = (28, 32, 39)
GRID = (62, 70, 82)
TEXT = (226, 235, 243)
MUTED = (150, 162, 176)
AXIS_COLORS = {
    "x": (58, 95, 245),
    "y": (82, 210, 122),
    "z": (245, 176, 75),
}


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.55,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_panel(image: np.ndarray, x: int, y: int, w: int, h: int, *, title: str | None = None) -> None:
    cv2.rectangle(image, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(image, (x, y), (x + w, y + h), (74, 88, 104), 1, cv2.LINE_AA)
    if title:
        put_text(image, title, (x + 14, y + 24), scale=0.62, color=TEXT, thickness=1)


def _safe_range(series: list[np.ndarray]) -> tuple[float, float]:
    stacked = np.concatenate([values[np.isfinite(values)] for values in series if len(values)])
    if len(stacked) == 0:
        return -1.0, 1.0
    low = float(np.percentile(stacked, 2))
    high = float(np.percentile(stacked, 98))
    if abs(high - low) < 1e-6:
        low -= 1.0
        high += 1.0
    padding = 0.12 * (high - low)
    return low - padding, high + padding


def draw_xyz_plot(
    image: np.ndarray,
    values: dict[str, np.ndarray],
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    unit: str,
) -> None:
    """Draw a scrolling X/Y/Z line plot into ``image``."""

    draw_panel(image, x, y, w, h, title=title)
    plot_left = x + 46
    plot_right = x + w - 14
    plot_top = y + 42
    plot_bottom = y + h - 32
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top

    for idx in range(5):
        gy = int(plot_top + idx * plot_h / 4)
        cv2.line(image, (plot_left, gy), (plot_right, gy), GRID, 1, cv2.LINE_AA)

    clean_values = {axis: np.asarray(axis_values, dtype=np.float64) for axis, axis_values in values.items()}
    y_min, y_max = _safe_range(list(clean_values.values()))
    put_text(image, f"{y_max: .1f}", (x + 8, plot_top + 6), scale=0.38, color=MUTED)
    put_text(image, f"{y_min: .1f}", (x + 8, plot_bottom), scale=0.38, color=MUTED)
    put_text(image, unit, (x + w - 58, y + 24), scale=0.42, color=MUTED)

    denom = y_max - y_min
    for axis, axis_values in clean_values.items():
        if len(axis_values) < 2:
            continue
        finite = np.isfinite(axis_values)
        if np.sum(finite) < 2:
            continue
        draw_values = np.clip(axis_values, y_min, y_max)
        xs = np.linspace(plot_left, plot_right, len(draw_values))
        ys = plot_bottom - ((draw_values - y_min) / denom) * plot_h
        points = np.column_stack((xs[finite], ys[finite])).astype(np.int32)
        if len(points) > 1:
            cv2.polylines(image, [points], False, AXIS_COLORS.get(axis, TEXT), 2, cv2.LINE_AA)

    legend_x = x + 14
    legend_y = y + h - 10
    for axis in ("x", "y", "z"):
        color = AXIS_COLORS[axis]
        cv2.circle(image, (legend_x, legend_y - 4), 4, color, -1, cv2.LINE_AA)
        put_text(image, axis.upper(), (legend_x + 8, legend_y), scale=0.38, color=color)
        legend_x += 44
