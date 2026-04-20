# Trekion.ai - Robotics Data Engineering Assessment
**Candidate:** [Derric samson]

## Overview
This repository contains a complete data pipeline for parsing, synchronizing, and running visual inference on proprietary multi-modal robotics sensor data. The pipeline is broken down into three main tasks: binary extraction/synchronization, monocular depth estimation, and scene segmentation.

## Task 1: Binary Format Parsing & Synchronization
**Approach:**
The raw `.imu` and `.vts` files utilized proprietary binary formats with no explicit documentation. I utilized hex inspection and Python's `struct` module to reverse-engineer the schemas.
* **IMU File:** Identified an 8-byte `TRIMU001` header followed by a metadata block. The true data rows were discovered starting at offset `28`. The schema was unpacked as `<Q10f` (an 8-byte nanosecond timestamp followed by ten 32-bit floats representing Accel, Gyro, Mag, and Temp).
* **VTS File:** Identified the `TRIVTS01` header, with data rows starting at offset `32`. The schema was unpacked as `<IQ` (a 4-byte frame index and an 8-byte microsecond timestamp).

**Synchronization Challenge:**
While both files shared the same clock domain, they recorded in different units. The VTS recorded in microseconds, while the high-frequency IMU (568 Hz) recorded in nanoseconds. By applying a `1000x` scalar to the VTS timestamps, I was able to utilize `pandas.merge_asof` (nearest direction) to perfectly align the high-frequency telemetry with the 30 FPS camera frames.

## Task 2: Monocular Dense Depth Estimation
**Approach:**
I implemented the `MiDaS_small` model via PyTorch Hub. The small variant was chosen for optimal CPU/GPU inference speed while maintaining highly accurate relative depth maps. 
* **Visualization:** The depth maps were normalized to an 8-bit scale and colorized using OpenCV's `COLORMAP_INFERNO` to create a perceptually intuitive visualization of proximity. The output was horizontally stacked with the original RGB frame for side-by-side comparison.

## Task 3: Object and Scene Segmentation
**Approach:**
I deployed Ultralytics `YOLOv8n-seg` (Nano Segmentation) for zero-shot object detection and pixel-level mask generation. 
* **Model Limitations & Hallucinations:** Because the pre-trained YOLOv8 model is constrained to the 80 classes of the standard COCO dataset, it naturally misclassified domain-specific hardware. For example, the checkerboard calibration pattern was labeled as a "book," and a monitor stand as a "tv". 

## Future Improvements for Production
1.  **Custom Weights:** Fine-tune the YOLOv8 model on a custom dataset representing specific physical AI lab equipment (soldering irons, calibration boards, robotic chassis) rather than relying on COCO.
2.  **Lens Calibration:** The wide-angle fisheye lens introduces notable barrel distortion. In a production environment, I would calculate the camera matrix and distortion coefficients using the visible checkerboard to undistort the frames *before* feeding them into the MiDaS depth model.
3.  **Vectorized Graphing:** The current HUD overlay uses Matplotlib canvas rendering per-frame, which is computationally heavy. For real-time production rendering, the graphing logic should be ported to a vectorized solution or handled asynchronously.