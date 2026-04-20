# Trekion Robotics Pipeline Write-up

## Binary Reverse Engineering

Both proprietary files begin with an 8-byte magic header. I validated these
before parsing:

- IMU: `TRIMU001`
- VTS: `TRIVTS01`

The final parser does not blindly assume offsets. It scores plausible
little-endian record layouts by checking timestamp monotonicity, realistic
sensor ranges, and expected camera/IMU cadence.

For the supplied dataset, the discovered layouts are:

- IMU: 64-byte metadata/header region followed by 80-byte records.
- IMU record: `<Q18f`, meaning one unsigned 64-bit timestamp plus 18 float32
  values. The first 10 floats map to accel XYZ, gyro XYZ, mag XYZ, and
  temperature. The remaining floats are preserved as `extra_*` diagnostics.
- VTS: 32-byte metadata/header region followed by 24-byte records.
- VTS record: `<IQIQ`, meaning frame number, primary hardware timestamp,
  auxiliary frame-like value, and auxiliary timestamp-like value.

The primary VTS timestamp is already in the same nanosecond clock domain as the
IMU stream. It aligns all 1,316 frames with sub-millisecond nearest-sample
delay. The auxiliary timestamp appears microsecond-like, but it has a worse
edge alignment for this dataset and is not used for synchronization.

## Timestamp Normalization

The pipeline detects timestamp units from median positive sample deltas rather
than from absolute magnitude. This is important because hardware clocks are
often relative clocks instead of UNIX epoch timestamps.

The supplied data is detected as:

- IMU: nanoseconds, about 568.4 Hz
- VTS: nanoseconds, about 30.0 Hz

Normalized timestamps are written as `timestamp_ns`. The CSV also keeps
`timestamp` as a compatibility alias for the frame timestamp in nanoseconds.

## Synchronization Strategy

The synchronized telemetry is produced with vectorized linear interpolation via
NumPy. For every video frame timestamp, each IMU channel is interpolated from
the surrounding raw IMU samples.

Interpolation is used instead of nearest-neighbor matching because the camera
is low-rate (30 Hz) and the IMU is high-rate (about 568 Hz). Nearest-neighbor
matching adds quantization error and can create small step artifacts in plots
or downstream analysis. Linear interpolation gives a better estimate of the
sensor state at the exact exposure timestamp.

Nearest raw IMU samples are still useful as a quality metric. The pipeline
reports mean, median, and max nearest-sample delay, but it does not use nearest
matching for the synchronized values.

## Visualization

The IMU HUD video is rendered with OpenCV drawing primitives only. This avoids
slow Matplotlib-per-frame rendering. Each frame contains:

- frame number and timestamp
- interpolated accel, gyro, mag, and temperature
- estimated IMU and camera rates
- sync delay metrics
- scrolling 2-3 second plots for accel, gyro, and magnetometer XYZ axes

The renderer streams video frames until EOF because OpenCV reports unreliable
container frame-count metadata for this MP4.

## Depth Model

Depth Anything V2 Small is used through HuggingFace Transformers. It gives
strong relative monocular depth quality while remaining practical for a laptop
GPU. The output video is side-by-side RGB and colorized depth using a perceptual
OpenCV colormap such as TURBO.

The fisheye lens is currently not undistorted because no camera matrix or
distortion coefficients were provided. In production, calibration from a
checkerboard sequence should be used to undistort frames before depth or
detection inference.

## Segmentation Model

YOLOv8 nano segmentation is used for speed and simplicity. It draws masks,
bounding boxes, labels, and confidence values. The model is COCO-pretrained, so
domain-specific robotics lab objects may be misclassified. A production version
should fine-tune on Trekion-specific classes and optionally add a dedicated hand
detector or pose model if hands are a target class.

## Limitations and Improvements

- Add camera calibration and fisheye undistortion.
- Add dataset-specific YOLO fine-tuning for lab equipment and robotic hardware.
- Add chunked CSV/parquet writers for very long logs.
- Add output manifests with parser layout, sync metrics, model versions, and
  runtime hardware details.
- Add CI smoke tests using tiny fixture binaries so parser regressions are
  caught without requiring large media files.
