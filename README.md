# Real-Time 3D Motion Tracking System

A Python-based stereo vision and object-detection pipeline developed for an NVIDIA Jetson AGX Orin. The system detects a DJI Tello drone and estimates its approximate position relative to a stereo-camera setup.

Developed as the motion-tracking subsystem for Binghamton University Senior Design Project 117: Brain-Controlled Drone.

## My Contribution

I developed the motion-tracking software, integrating camera capture, stereo rectification, YOLO-based drone detection, disparity processing, and approximate 3D coordinate estimation.

This repository contains the motion-tracking program. The accompanying team report documents the broader project, including its separate biosignal-based drone-control subsystem.

## Hardware and Development Environment

- **Compute:** NVIDIA Jetson AGX Orin
- **Cameras:** Two e-CAM20_CUOAGX MIPI-CSI cameras
- **Mount:** Custom tripod and aluminum-extrusion stereo mount
- **Tracking target:** DJI Tello drone
- **Original software environment:** JetPack 5.0.2 / L4T 35.1.0

## Software Stack

- **Python**
- **OpenCV:** Stereo rectification, disparity estimation, image processing, and visualization
- **NumPy:** Calibration data handling and numerical processing
- **Ultralytics YOLOv8n:** Drone detection using a fine-tuned model
- **CUDA:** GPU-based YOLO inference
- **GStreamer:** Camera-stream capture
- **Python threading:** Independent frame acquisition from both cameras

## How It Works

### 1. Camera Capture and Rectification

The program captures both camera streams through GStreamer pipelines using separate frame-acquisition threads.

Precomputed stereo calibration maps rectify the images so corresponding features align across the left and right views.

### 2. Drone Detection

A fine-tuned YOLOv8n model runs on the rectified left image to locate the drone.

The project’s training dataset was expanded with labeled images from cluttered lab scenes and the final demonstration environment to improve detection under realistic conditions.

### 3. Disparity and Position Estimation

OpenCV’s StereoSGBM algorithm computes a disparity map from the rectified stereo images.

The program samples valid disparity values within the detected drone’s bounding box, filters outliers, and uses the camera calibration parameters to estimate relative X, Y, and Z coordinates. A short history of coordinate estimates helps smooth the output.

### 4. Live Display

The interface displays:

- Drone detection and tracking overlays
- Stereo disparity visualization
- Approximate relative X, Y, and Z coordinates
- Processing frame rate
- Status messages when detection or depth information is unavailable

The program also supports reopening camera streams if they become stale.

## Results and Limitations

The final system demonstrated real-time drone detection and approximate position tracking. Additional training images improved detection in cluttered environments.

Coordinate precision remained a limitation. The displayed coordinates are approximate and rounded to half-foot increments; they should not be interpreted as precise measurements.

Performance depends on camera alignment, calibration quality, scene texture, lighting, distance, and detection reliability.

## Required Hardware and Files

Running the program requires:

- The Jetson and compatible stereo-camera hardware
- Compatible camera drivers
- An OpenCV build with GStreamer support
- NumPy and Ultralytics
- A CUDA-compatible inference environment
- Camera-specific stereo calibration data
- The trained YOLO model weights

The current script expects:

- `stereo_calib_sb_9x6_24mm.npz`
- `models/drone_demo_finetune_best.pt`

The calibration file and model weights are not included in this repository. Calibration data must match the camera geometry and capture resolution.

OpenCV’s `ximgproc` module enables optional WLS disparity filtering. The program falls back to unfiltered StereoSGBM disparity when that functionality is unavailable.

## Running the Program

After configuring the hardware, dependencies, calibration file, and model weights, run:

    python3 motion_tracking_program.py

### Controls

- **Q:** Quit
- **R:** Reopen the camera streams

## Project Documentation

See the [Final Design Report](SDP117_Final_Design_Report.pdf) for the full team project, hardware architecture, implementation details, testing, and operating instructions.
