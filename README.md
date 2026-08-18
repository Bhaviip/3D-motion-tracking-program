# Real-Time 3D Motion Tracking System (NVIDIA Jetson AGX Orin)

An embedded stereo vision and deep learning motion tracking pipeline engineered for custom dual-camera hardware on an NVIDIA Jetson AGX Orin. Developed as the motion-tracking subsystem for Senior Design Project 117 (Brain-Controlled Drone).

---

## 📌 Hardware & Rig Architecture
*Designed and configured specifically for custom laboratory hardware:*
- **Edge Compute:** NVIDIA Jetson AGX Orin
- **Sensors:** Dual e-CAM20_CUOAGX MIPI-CSI cameras mounted on a custom aluminum extrusion stereo bar
- **Target Platform:** DJI Tello drone
- **Interfacing:** Linux terminal / Jetson JetPack environment with GStreamer multi-stream ingestion

---

## 🛠 Tech Stack & Implementation
- **Model:** Fine-tuned YOLOv8n (trained on cluttered indoor environments and custom flight frames)
- **Computer Vision:** OpenCV (custom stereo calibration matrices, epipolar rectification, disparity computation, 3D coordinate estimation)
- **Hardware Acceleration:** CUDA, NVIDIA VPI, GStreamer

---

## 🔍 System Pipeline & Technical Workflow

1. **Custom Stereo Calibration & Rectification:**
   - Evaluated and computed stereo rectification matrices for the dual CSI camera configuration.
   - Rectified left and right video streams to align epipolar lines and stabilize feature matching.

2. **Target Detection:**
   - Ingested camera streams via hardware-accelerated GStreamer pipelines.
   - Deployed the custom-trained YOLOv8n model on the primary rectified feed for real-time bounding box prediction.

3. **Disparity & 3D Spatial Localization:**
   - Generated stereo disparity maps across the dual feeds to extract depth information.
   - Estimated real-time approximate relative $(X, Y, Z)$ spatial coordinates of the drone.

---

## 📄 Project Documentation
- **Full Report:** See [`SDP117_Final_Design_Report.pdf`](SDP117_Final_Design_Report.pdf) for in-depth system architecture, calibration data, and experimental evaluation.
