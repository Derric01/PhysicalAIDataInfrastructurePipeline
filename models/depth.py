"""Monocular depth estimation video pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import pipeline

LOGGER = logging.getLogger(__name__)

DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def _pipeline_device(device: str) -> int:
    if device == "auto":
        return 0 if torch.cuda.is_available() else -1
    if device == "cpu":
        return -1
    return int(device)


def _load_depth_pipeline(model_name: str, device: str):
    hf_device = _pipeline_device(device)
    LOGGER.info("Loading depth model %s on device=%s", model_name, device)
    model_path = model_name

    if not Path(model_name).exists():
        try:
            model_path = snapshot_download(repo_id=model_name, local_files_only=True)
            LOGGER.info("Using cached HuggingFace snapshot: %s", model_path)
        except Exception as exc:
            LOGGER.warning("Cached depth model snapshot not found (%s); attempting standard HuggingFace load", exc)

    try:
        return pipeline(task="depth-estimation", model=model_path, device=hf_device, local_files_only=True)
    except Exception as exc:
        LOGGER.warning("Local depth model load failed (%s); retrying normal HuggingFace load", exc)
        return pipeline(task="depth-estimation", model=model_name, device=hf_device)


def render_depth_video(
    *,
    video_path: str | Path,
    output_path: str | Path,
    max_frames: int | None = None,
    device: str = "auto",
    model_name: str = DEFAULT_DEPTH_MODEL,
    colormap: int = cv2.COLORMAP_TURBO,
) -> int:
    """Render a side-by-side RGB/depth video."""

    depth_pipe = _load_depth_pipeline(model_name, device)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open video writer for {output_path}")

    frame_index = 0
    LOGGER.info("Rendering depth video to %s", output_path)
    while True:
        if max_frames is not None and frame_index >= max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        depth_result = depth_pipe(Image.fromarray(rgb))
        depth = np.asarray(depth_result["depth"])
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_CUBIC)
        depth_u8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_color = cv2.applyColorMap(depth_u8, colormap)

        writer.write(np.hstack((frame, depth_color)))
        frame_index += 1
        if frame_index % 30 == 0:
            LOGGER.info("Rendered %s depth frames", frame_index)

    cap.release()
    writer.release()
    LOGGER.info("Finished depth video: %s frames -> %s", frame_index, output_path)
    return frame_index
