# Trekion Robotics Sensor Pipeline

Production-oriented Python pipeline for Trekion's multi-modal robotics data
assessment.

It covers the full flow from raw proprietary binaries to analysis-ready outputs:

- IMU binary parsing from TRIMU001 format
- Video timestamp parsing from TRIVTS01 format
- Timestamp normalization and stream synchronization
- Telemetry + video HUD rendering
- Monocular depth rendering
- YOLO segmentation rendering

The primary entrypoint is `main.py`.

## What This Project Produces

For one recording session, the pipeline can generate:

- Synchronized telemetry CSV at video-frame cadence
- Synchronization quality metrics JSON
- IMU HUD video with overlaid telemetry and plots
- Depth video (RGB + colorized depth)
- Segmentation video with masks and labels

## Repository Layout

```text
parsers/
  imu_parser.py          # TRIMU001 parser + layout discovery
  vts_parser.py          # TRIVTS01 parser + layout discovery
sync/
  synchronizer.py        # Frame-wise synchronization and delay metrics
models/
  depth.py               # Depth model inference and rendering
  detection.py           # YOLO segmentation inference and rendering
visualization/
  hud.py                 # HUD primitives and telemetry text overlay
  plots.py               # Scrolling plot rendering
video/
  video_pipeline.py      # IMU synchronized HUD video pipeline
utils/
  validation.py          # Timestamp normalization and sanity checks
  logging.py             # Shared logger configuration
main.py                  # CLI entrypoint
tests/                   # Parser and synchronization tests
```

## Data Requirements

Place the provided dataset files in `Givenfiles/`:

```text
Givenfiles/recording2.mp4
Givenfiles/recording2.imu
Givenfiles/recording2.vts
```

Segmentation defaults to `yolov8n-seg.pt` in the project root unless an
alternative path is passed with `--weights`.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

Generate telemetry CSV + sync metrics + IMU HUD video:

```bash
python main.py --mode imu
```

Generate depth video:

```bash
python main.py --mode depth
```

Generate segmentation video:

```bash
python main.py --mode seg
```

Run all three modes:

```bash
python main.py --mode all
```

Fast smoke runs:

```bash
python main.py --mode imu --max-frames 5 --out-dir smoke_outputs
python main.py --mode depth --max-frames 5 --out-dir smoke_outputs
python main.py --mode seg --max-frames 5 --out-dir smoke_outputs
```

## CLI Notes

Common options:

- `--mode`: `imu`, `depth`, `seg`, or `all`
- `--video`, `--imu`, `--vts`: input file paths
- `--out-dir`: output directory (default is current directory)
- `--max-frames`: optional frame cap for quicker runs
- `--device`: `auto`, `cpu`, or GPU index
- `--conf`: segmentation confidence threshold
- `--weights`: YOLO segmentation weights path
- `--skip-video`: IMU mode only, writes CSV/metrics without HUD video

## Outputs

Typical generated files:

```text
synchronized_telemetry.csv
sync_metrics.json
imu_sync_output.mp4
depth_output.mp4
segmentation_output.mp4
```

## Synchronization Behavior

- Synchronization is performed at each video timestamp.
- IMU channels are linearly interpolated to frame time.
- Nearest raw IMU samples are used only for reporting delay metrics.

This avoids nearest-neighbor quantization artifacts while preserving a clear
quality signal through max/median/mean delay values.

## Validation

```bash
pytest
```

Expected parser/sync facts for the supplied dataset:

```text
IMU rows: 24938
VTS/video frames: 1316
IMU rate: about 568.4 Hz
Camera rate: about 30.0 Hz
Max nearest raw IMU delay: under 1 ms
```

## Limitations

- Fisheye lens undistortion is not applied because calibration parameters were
  not provided.
- OpenCV frame count metadata for this MP4 is unreliable; frame processing is
  streamed to EOF and validated against VTS sequence.
- Segmentation quality depends on the chosen YOLO weights and class coverage.

See `WRITEUP.md` for detailed design decisions and trade-offs.
