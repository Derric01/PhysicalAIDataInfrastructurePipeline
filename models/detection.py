"""YOLO segmentation/detection video pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
from ultralytics import YOLO

LOGGER = logging.getLogger(__name__)

DEFAULT_SEGMENTATION_WEIGHTS = "yolov8n-seg.pt"


def _ultralytics_device(device: str) -> str | None:
    if device == "auto":
        return None
    return device


def render_segmentation_video(
    *,
    video_path: str | Path,
    output_path: str | Path,
    weights_path: str | Path = DEFAULT_SEGMENTATION_WEIGHTS,
    conf: float = 0.25,
    max_frames: int | None = None,
    device: str = "auto",
) -> int:
    """Render YOLOv8 segmentation masks, boxes, labels, and confidences."""

    weights = Path(weights_path)
    if not weights.exists():
        raise FileNotFoundError(
            f"YOLO weights not found at {weights}. Place yolov8n-seg.pt in the project root or pass --weights."
        )

    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open video writer for {output_path}")

    frame_index = 0
    LOGGER.info("Rendering segmentation video to %s", output_path)
    while True:
        if max_frames is not None and frame_index >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            frame,
            conf=conf,
            verbose=False,
            device=_ultralytics_device(device),
        )
        writer.write(results[0].plot())

        frame_index += 1
        if frame_index % 30 == 0:
            LOGGER.info("Rendered %s segmentation frames", frame_index)

    cap.release()
    writer.release()
    LOGGER.info("Finished segmentation video: %s frames -> %s", frame_index, output_path)
    return frame_index
