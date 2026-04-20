# Trekion Robotics Sensor Pipeline

Production-oriented Python pipeline for Trekion's multi-modal robotics data
assessment: proprietary binary IMU parsing, video timestamp parsing, temporal
synchronization, HUD visualization, monocular depth, and YOLO segmentation.

## Repository Layout

```text
parsers/
  imu_parser.py          # TRIMU001 binary parser and layout discovery
  vts_parser.py          # TRIVTS01 binary parser and layout discovery
sync/
  synchronizer.py        # Linear interpolation sync and delay metrics
models/
  depth.py               # Depth Anything V2 video renderer
  detection.py           # YOLOv8 segmentation renderer
visualization/
  hud.py                 # OpenCV HUD overlay
  plots.py               # OpenCV scrolling plots
video/
  video_pipeline.py      # IMU HUD video renderer
utils/
  validation.py          # Timestamp normalization and checks
  logging.py             # Logging setup
main.py                  # CLI entrypoint
tests/                   # Parser and sync tests
```

Legacy exploratory scripts are kept in the project root for reference, but
`main.py` is the source-of-truth entrypoint.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Place the supplied files in `Givenfiles/`:

```text
Givenfiles/recording2.mp4
Givenfiles/recording2.imu
Givenfiles/recording2.vts
```

The segmentation pipeline expects `yolov8n-seg.pt` in the project root unless
you pass `--weights`.

## Usage

Generate synchronized telemetry CSV and IMU HUD video:

```bash
python main.py --mode imu
```

Generate dense monocular depth video:

```bash
python main.py --mode depth
```

Generate segmentation video:

```bash
python main.py --mode seg
```

Run all outputs:

```bash
python main.py --mode all
```

Useful smoke-test options:

```bash
python main.py --mode imu --max-frames 5 --out-dir smoke_outputs
python main.py --mode depth --max-frames 5 --out-dir smoke_outputs
python main.py --mode seg --max-frames 5 --out-dir smoke_outputs
```

## Outputs

```text
synchronized_telemetry.csv
sync_metrics.json
imu_sync_output.mp4
depth_output.mp4
segmentation_output.mp4
```

## Validation

```bash
pytest
```

Expected parser facts for the provided dataset:

```text
IMU rows: 24938
VTS/video frames: 1316
IMU rate: about 568.4 Hz
Camera rate: about 30.0 Hz
Max nearest raw IMU delay: under 1 ms
```

## Notes

- IMU synchronization uses linear interpolation at each video frame timestamp.
  Nearest raw IMU samples are used only to report delay metrics.
- The fisheye lens is not undistorted because no calibration matrix or
  distortion coefficients were provided. This is documented in `WRITEUP.md`.
- OpenCV frame count metadata is not trusted for this MP4; the renderer streams
  frames until EOF and validates against the VTS frame sequence.
