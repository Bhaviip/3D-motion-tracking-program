# SDP 117: Real-Time 3D Motion Tracking System (NVIDIA Jetson)

An embedded stereo vision and deep learning motion tracking pipeline developed to localize and track an aerial target (DJI Tello) in real time on an NVIDIA Jetson AGX Orin.

---

## 📌 System Architecture & Hardware
- **Compute:** NVIDIA Jetson AGX Orin
- **Sensors:** Dual e-CAM20_CUOAGX MIPI-CSI cameras mounted on an aluminum extrusion stereo rig
- **Target System:** DJI Tello drone

---

## 🛠 Software & Tech Stack
- **Model:** YOLOv8n (fine-tuned on cluttered indoor and flight environments)
- **Computer Vision:** OpenCV (stereo calibration, rectification matrix, disparity estimation, 3D localization)
- **Streaming & Acceleration:** GStreamer pipelines, CUDA, NVIDIA VPI, Linux/Bash

---

## 🚀 Pipeline Overview
1. **Stereo Image Acquisition & Rectification:** Ingests dual CSI camera feeds via GStreamer and applies stereo rectification matrices to align epipolar lines.
2. **Object Detection:** Runs the fine-tuned YOLOv8n detector on the rectified primary feed to generate bounding boxes around the drone.
3. **Disparity & 3D Localization:** Computes stereo disparity to estimate relative $X, Y, Z$ spatial coordinates in real time.

---

## 💻 Getting Started & Execution

### 1. Environment & Dependencies
```bash
# Clone the repository
git clone [https://github.com/bhavi-pate/3D-motion-tracking-program.git](https://github.com/bhavi-patel/3D-motion-tracking-program.git)
cd 3D-motion-tracking-program

# Install dependencies on Jetson
pip install ultralytics opencv-python numpy
