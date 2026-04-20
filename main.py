"""CLI entrypoint for the Trekion robotics data pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from models.depth import DEFAULT_DEPTH_MODEL, render_depth_video
from models.detection import DEFAULT_SEGMENTATION_WEIGHTS, render_segmentation_video
from parsers.imu_parser import parse_imu_file
from parsers.vts_parser import parse_vts_file
from sync.synchronizer import synchronize_streams
from utils.logging import configure_logging, get_logger
from utils.validation import normalize_timestamps
from video.video_pipeline import render_imu_sync_video

LOGGER = get_logger(__name__)


def _colormap(name: str) -> int:
    colormaps = {
        "turbo": cv2.COLORMAP_TURBO,
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
    }
    try:
        return colormaps[name.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"Unknown colormap {name!r}; choose from {sorted(colormaps)}") from exc


def run_imu_mode(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imu_df = parse_imu_file(args.imu)
    vts_df = parse_vts_file(args.vts)

    imu_df, imu_stats = normalize_timestamps(imu_df, stream_name="IMU")
    vts_df, vts_stats = normalize_timestamps(vts_df, stream_name="VTS")

    synced_df, metrics = synchronize_streams(vts_df, imu_df)
    metrics["imu_timestamp_unit"] = imu_stats.unit
    metrics["vts_timestamp_unit"] = vts_stats.unit

    csv_path = out_dir / "synchronized_telemetry.csv"
    synced_df.to_csv(csv_path, index=False)
    LOGGER.info("Wrote synchronized telemetry CSV to %s", csv_path)

    metrics_path = out_dir / "sync_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Wrote synchronization metrics to %s", metrics_path)

    if args.skip_video:
        return

    render_imu_sync_video(
        video_path=args.video,
        telemetry_df=synced_df,
        metrics=metrics,
        output_path=out_dir / "imu_sync_output.mp4",
        max_frames=args.max_frames,
    )


def run_depth_mode(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_depth_video(
        video_path=args.video,
        output_path=out_dir / "depth_output.mp4",
        max_frames=args.max_frames,
        device=args.device,
        model_name=args.depth_model,
        colormap=_colormap(args.colormap),
    )


def run_seg_mode(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_segmentation_video(
        video_path=args.video,
        output_path=out_dir / "segmentation_output.mp4",
        weights_path=args.weights,
        conf=args.conf,
        max_frames=args.max_frames,
        device=args.device,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trekion multi-modal robotics data pipeline")
    parser.add_argument("--mode", choices=["imu", "depth", "seg", "all"], required=True)
    parser.add_argument("--video", default="Givenfiles/recording2.mp4", help="Path to RGB video")
    parser.add_argument("--imu", default="Givenfiles/recording2.imu", help="Path to binary IMU file")
    parser.add_argument("--vts", default="Givenfiles/recording2.vts", help="Path to binary VTS file")
    parser.add_argument("--out-dir", default=".", help="Directory for generated outputs")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for video rendering")
    parser.add_argument("--device", default="auto", help="Model device: auto, cpu, or GPU index such as 0")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--weights", default=DEFAULT_SEGMENTATION_WEIGHTS, help="YOLO segmentation weights path")
    parser.add_argument("--depth-model", default=DEFAULT_DEPTH_MODEL, help="HuggingFace depth model name")
    parser.add_argument("--colormap", default="turbo", choices=["turbo", "inferno", "magma"])
    parser.add_argument("--skip-video", action="store_true", help="Only parse/sync/write CSV in IMU mode")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    if args.mode in ("imu", "all"):
        run_imu_mode(args)
    if args.mode in ("depth", "all"):
        run_depth_mode(args)
    if args.mode in ("seg", "all"):
        run_seg_mode(args)


if __name__ == "__main__":
    main()
