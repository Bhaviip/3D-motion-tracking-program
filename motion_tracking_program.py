# Drone Detection with YOLO
# Written By: Bhavi Patel
# Date: May 1, 2026
# Notes:
# Final version of the 3D motion tracking program.

import cv2
import numpy as np
import time
import threading
import os
from collections import deque

from ultralytics import YOLO

# ==========================
# SETTINGS
# ==========================
# Camera capture resolution
WIDTH, HEIGHT = 1920, 1080

# Processing resolution (keep v2 speed)
PROCESS_W, PROCESS_H = 960, 540

# Display resolution
DISPLAY_W, DISPLAY_H = 1280, 720

LEFT_DEV = 0
RIGHT_DEV = 1

CALIB_FILE = "stereo_calib_sb_9x6_24mm.npz"
MODEL_PATH = "models/drone_demo_finetune_best.pt"

STALE_TIMEOUT = 1.0

# --------------------------
# Stereo disparity settings
# --------------------------
MIN_DISPARITY = 0
NUM_DISPARITIES = 16 * 6
BLOCK_SIZE = 9

# Optional disparity improvements
USE_CLAHE = True
ENABLE_WLS = True
WLS_LAMBDA = 8000.0
WLS_SIGMA = 1.5

# --------------------------
# Detection settings
# --------------------------
CONF_THRESH = 0.20
DETECT_EVERY_N_FRAMES = 1
BOX_HOLD_FRAMES = 2
TARGET_CLASS_ID = 0
BOX_SMOOTH_ALPHA = 0.85   # 1.0 = no smoothing, 0.85 = responsive

# YOLO settings
YOLO_IMGSZ = 800
USE_HALF = True

# Display settings
MIRROR_DISPLAY = True
WINDOW_NAME = "SDP 117: Stereo YOLO Motion Tracking"
FOOTER_TEXT = "Left: Drone Detection and Tracking | Right: Disparity Map"
SHOW_DEBUG_DEPTH_TEXT = False

# --------------------------
# Depth / coordinate settings
# --------------------------
MM_TO_IN = 1.0 / 25.4
IN_TO_FT = 1.0 / 12.0

def round_to_half_foot(v_in):
    v_ft = v_in * IN_TO_FT
    return round(v_ft * 2.0) / 2.0

# Inner ROI inside the bbox for depth sampling
BODY_ROI_W_FRAC = 0.30
BODY_ROI_H_FRAC = 0.35
MIN_BODY_ROI_HALF_W = 8
MIN_BODY_ROI_HALF_H = 8
MAX_BODY_ROI_HALF_W = 20
MAX_BODY_ROI_HALF_H = 24

MIN_VALID_DISP = 2.0
MIN_VALID_DISP_PIXELS = 30
TRIM_PERCENT_LOW = 20
TRIM_PERCENT_HIGH = 80

# Coordinate smoothing
XYZ_HISTORY_LEN = 5

# ==========================
# GSTREAMER PIPELINE
# ==========================
def gstreamer_pipeline(device: int) -> str:
    return (
        f"v4l2src device=/dev/video{device} ! "
        f"video/x-raw,format=UYVY,width={WIDTH},height={HEIGHT} ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink max-buffers=1 drop=true sync=false"
    )

# ==========================
# CAMERA THREAD
# ==========================
class LatestFrameGrabber:
    def __init__(self, dev_index: int):
        self.dev_index = dev_index
        self.cap = None
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.stop_flag = False
        self.thread = None
        self.last_update_time = 0.0

    def open(self):
        self.cap = cv2.VideoCapture(gstreamer_pipeline(self.dev_index), cv2.CAP_GSTREAMER)
        return self.cap.isOpened()

    def start(self):
        self.stop_flag = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_flag:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.05)
                continue

            ret, frm = self.cap.read()
            with self.lock:
                self.ok = bool(ret) and frm is not None
                if self.ok:
                    self.frame = frm
                    self.last_update_time = time.time()
            time.sleep(0.001)

    def get(self):
        with self.lock:
            frame_copy = None if self.frame is None else self.frame.copy()
            return self.ok, frame_copy, self.last_update_time

    def reopen(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        time.sleep(0.25)
        return self.open()

    def stop(self):
        self.stop_flag = True
        try:
            if self.thread is not None:
                self.thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

# ==========================
# HELPERS
# ==========================
def smooth_box(prev_box, new_box, alpha):
    if prev_box is None:
        return [float(v) for v in new_box]
    return [
        (1.0 - alpha) * prev_box[0] + alpha * new_box[0],
        (1.0 - alpha) * prev_box[1] + alpha * new_box[1],
        (1.0 - alpha) * prev_box[2] + alpha * new_box[2],
        (1.0 - alpha) * prev_box[3] + alpha * new_box[3],
    ]

def choose_best_detection(result_boxes, target_class_id=None):
    best = None
    best_conf = -1.0
    best_cls = None

    for b in result_boxes:
        conf = float(b.conf[0].item())
        cls_id = int(b.cls[0].item()) if b.cls is not None else -1

        if target_class_id is not None and cls_id != target_class_id:
            continue

        if conf > best_conf:
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            best = xyxy
            best_conf = conf
            best_cls = cls_id

    if best is None:
        return None, None, None
    return best, best_conf, best_cls

def clamp_box(box, w, h):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    return x1, y1, x2, y2

def mirror_box_coords(x1, y1, x2, y2, img_w):
    mx1 = (img_w - 1) - x2
    mx2 = (img_w - 1) - x1
    return mx1, y1, mx2, y2

def mirror_x(x, img_w):
    return (img_w - 1) - x

def get_class_name(class_names, cls_id):
    if cls_id is None:
        return "Object"
    if isinstance(class_names, dict):
        return class_names.get(cls_id, str(cls_id))
    if isinstance(class_names, list):
        if 0 <= cls_id < len(class_names):
            return class_names[cls_id]
    return str(cls_id)

def median_or_none(dq):
    if len(dq) == 0:
        return None
    return float(np.median(np.array(dq, dtype=np.float32)))

def compute_body_roi(cx, cy, x1, y1, x2, y2, img_w, img_h):
    bw = x2 - x1
    bh = y2 - y1

    half_w = max(MIN_BODY_ROI_HALF_W, min(int(round(bw * BODY_ROI_W_FRAC * 0.5)), MAX_BODY_ROI_HALF_W))
    half_h = max(MIN_BODY_ROI_HALF_H, min(int(round(bh * BODY_ROI_H_FRAC * 0.5)), MAX_BODY_ROI_HALF_H))

    rx1 = max(0, cx - half_w)
    rx2 = min(img_w, cx + half_w)
    ry1 = max(0, cy - half_h)
    ry2 = min(img_h, cy + half_h)

    return rx1, ry1, rx2, ry2

# ==========================
# MAIN
# ==========================
def main():
    cv2.setUseOptimized(True)
    cv2.setNumThreads(2)

    if not os.path.exists(CALIB_FILE):
        print(f"ERROR: Calibration file not found: {CALIB_FILE}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: YOLO model not found: {MODEL_PATH}")
        return

    # Load calibration
    data = np.load(CALIB_FILE)
    mapLx, mapLy = data["mapLx"], data["mapLy"]
    mapRx, mapRy = data["mapRx"], data["mapRy"]
    P1 = data["P1"]
    P2 = data["P2"]

    # Full-resolution intrinsics
    fx = float(P1[0, 0])
    fy = float(P1[1, 1])
    cx0 = float(P1[0, 2])
    cy0 = float(P1[1, 2])
    baseline_mm = abs(float(P2[0, 3]) / fx)

    # Scale intrinsics to processing resolution
    sx = PROCESS_W / WIDTH
    sy = PROCESS_H / HEIGHT
    fx_p = fx * sx
    fy_p = fy * sy
    cx0_p = cx0 * sx
    cy0_p = cy0 * sy

    print(f"Using fx(full): {fx:.2f} px")
    print(f"Using fy(full): {fy:.2f} px")
    print(f"Using cx0(full): {cx0:.2f} px")
    print(f"Using cy0(full): {cy0:.2f} px")
    print(f"Using baseline: {baseline_mm:.2f} mm")
    print(f"Processing at: {PROCESS_W}x{PROCESS_H}")

    # Load YOLO model
    model = YOLO(MODEL_PATH)
    class_names = model.names

    # Stereo matcher
    stereo_left = cv2.StereoSGBM_create(
        minDisparity=MIN_DISPARITY,
        numDisparities=NUM_DISPARITIES,
        blockSize=BLOCK_SIZE,
        P1=8 * BLOCK_SIZE * BLOCK_SIZE,
        P2=32 * BLOCK_SIZE * BLOCK_SIZE,
        disp12MaxDiff=1,
        uniquenessRatio=12,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    stereo_right = None
    wls_filter = None
    use_wls = False

    if ENABLE_WLS and hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createRightMatcher"):
        try:
            stereo_right = cv2.ximgproc.createRightMatcher(stereo_left)
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_left)
            wls_filter.setLambda(WLS_LAMBDA)
            wls_filter.setSigmaColor(WLS_SIGMA)
            use_wls = True
            print("Using WLS disparity filtering")
        except Exception as e:
            print(f"WLS unavailable, using raw SGBM disparity: {e}")
            use_wls = False
    else:
        print("WLS not available, using raw SGBM disparity")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if USE_CLAHE else None

    left = LatestFrameGrabber(LEFT_DEV)
    right = LatestFrameGrabber(RIGHT_DEV)

    if not left.open():
        print("ERROR: Left camera failed to open")
        return
    if not right.open():
        print("ERROR: Right camera failed to open")
        return

    left.start()
    right.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_W, DISPLAY_H)

    print("\nControls:")
    print("  q -> quit")
    print("  r -> reopen cameras\n")

    frame_idx = 0
    tracked_box = None
    tracked_conf = None
    tracked_cls = None
    hold_count = 0

    x_hist = deque(maxlen=XYZ_HISTORY_LEN)
    y_hist = deque(maxlen=XYZ_HISTORY_LEN)
    z_hist = deque(maxlen=XYZ_HISTORY_LEN)

    fps_ema = 0.0
    prev_loop_time = time.time()

    while True:
        okL, frameL, tL = left.get()
        okR, frameR, tR = right.get()
        now = time.time()

        # FPS estimate
        dt = max(now - prev_loop_time, 1e-6)
        inst_fps = 1.0 / dt
        fps_ema = inst_fps if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * inst_fps
        prev_loop_time = now

        if okL and (now - tL > STALE_TIMEOUT):
            print("Left camera stream stale -> reopening left camera...")
            left.reopen()
            time.sleep(0.2)

        if okR and (now - tR > STALE_TIMEOUT):
            print("Right camera stream stale -> reopening right camera...")
            right.reopen()
            time.sleep(0.2)

        if not okL or not okR:
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                print("Reopening cameras...")
                left.reopen()
                right.reopen()
            continue

        # Rectify at full res
        rectL_full = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
        rectR_full = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)

        # Downscale for processing
        procL = cv2.resize(rectL_full, (PROCESS_W, PROCESS_H), interpolation=cv2.INTER_AREA)
        procR = cv2.resize(rectR_full, (PROCESS_W, PROCESS_H), interpolation=cv2.INTER_AREA)

        # Disparity on unflipped processing frames
        grayL = cv2.cvtColor(procL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(procR, cv2.COLOR_BGR2GRAY)

        if clahe is not None:
            grayL = clahe.apply(grayL)
            grayR = clahe.apply(grayR)

        dispL_raw = stereo_left.compute(grayL, grayR)

        if use_wls:
            dispR_raw = stereo_right.compute(grayR, grayL)
            disp_filtered = wls_filter.filter(dispL_raw, procL, None, dispR_raw)
            disparity = disp_filtered.astype(np.float32) / 16.0
        else:
            disparity = dispL_raw.astype(np.float32) / 16.0

        disparity[disparity < MIN_VALID_DISP] = 0

        # -------- YOLO DETECTION ON LEFT PROCESSING FRAME --------
        frame_idx += 1

        if frame_idx % DETECT_EVERY_N_FRAMES == 0:
            classes_arg = [TARGET_CLASS_ID] if TARGET_CLASS_ID is not None else None
            results = model.predict(
                source=procL,
                conf=CONF_THRESH,
                imgsz=YOLO_IMGSZ,
                device=0,
                half=USE_HALF,
                classes=classes_arg,
                verbose=False
            )

            if results and len(results) > 0:
                boxes = results[0].boxes
                best_box, best_conf, best_cls = choose_best_detection(boxes, TARGET_CLASS_ID)

                if best_box is not None:
                    tracked_box = smooth_box(tracked_box, best_box, BOX_SMOOTH_ALPHA)
                    tracked_conf = best_conf
                    tracked_cls = best_cls
                    hold_count = BOX_HOLD_FRAMES
                else:
                    if hold_count > 0:
                        hold_count -= 1
                    else:
                        tracked_box = None
                        tracked_conf = None
                        tracked_cls = None
                        x_hist.clear()
                        y_hist.clear()
                        z_hist.clear()
            else:
                if hold_count > 0:
                    hold_count -= 1
                else:
                    tracked_box = None
                    tracked_conf = None
                    tracked_cls = None
                    x_hist.clear()
                    y_hist.clear()
                    z_hist.clear()
        else:
            if tracked_box is not None and hold_count > 0:
                hold_count -= 1

        info_lines = ["No drone detected"]
        xyz_text_terminal = None

        # Disparity visualization
        disp_vis = disparity.copy()
        disp_vis = cv2.normalize(disp_vis, None, 0, 255, cv2.NORM_MINMAX)
        disp_vis = np.uint8(disp_vis)
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

        # Display images
        if MIRROR_DISPLAY:
            viewL = cv2.flip(procL.copy(), 1)
            viewDisp = cv2.flip(disp_vis.copy(), 1)
        else:
            viewL = procL.copy()
            viewDisp = disp_vis.copy()

        if tracked_box is not None:
            x1, y1, x2, y2 = clamp_box(tracked_box, PROCESS_W, PROCESS_H)

            if x2 > x1 and y2 > y1:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Use bbox-relative inner ROI instead of fixed center window
                rx1, ry1, rx2, ry2 = compute_body_roi(cx, cy, x1, y1, x2, y2, disparity.shape[1], disparity.shape[0])

                roi_disp = disparity[ry1:ry2, rx1:rx2]
                valid_disp = roi_disp[roi_disp > MIN_VALID_DISP]

                if MIRROR_DISPLAY:
                    dx1, dy1, dx2, dy2 = mirror_box_coords(x1, y1, x2, y2, PROCESS_W)
                    dcx = mirror_x(cx, PROCESS_W)
                    dcy = cy
                    drx1, dry1, drx2, dry2 = mirror_box_coords(rx1, ry1, rx2, ry2, PROCESS_W)
                else:
                    dx1, dy1, dx2, dy2 = x1, y1, x2, y2
                    dcx, dcy = cx, cy
                    drx1, dry1, drx2, dry2 = rx1, ry1, rx2, ry2

                # Draw overlays on left image
                cv2.rectangle(viewL, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
                cv2.circle(viewL, (dcx, dcy), 5, (0, 0, 255), -1)
                cv2.rectangle(viewL, (drx1, dry1), (drx2, dry2), (0, 255, 255), 2)

                # Also draw depth ROI on disparity panel
                cv2.rectangle(viewDisp, (drx1, dry1), (drx2, dry2), (0, 255, 255), 2)
                cv2.circle(viewDisp, (dcx, dcy), 5, (0, 0, 255), -1)

                cls_name = get_class_name(class_names, tracked_cls)
                if tracked_conf is not None:
                    label = f"{cls_name} conf={tracked_conf:.2f}"
                else:
                    label = cls_name

                cv2.putText(
                    viewL,
                    label,
                    (dx1, max(20, dy1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                if valid_disp.size >= MIN_VALID_DISP_PIXELS:
                    # Trim disparity outliers before taking the median
                    p_low, p_high = np.percentile(valid_disp, [TRIM_PERCENT_LOW, TRIM_PERCENT_HIGH])
                    trimmed_disp = valid_disp[(valid_disp >= p_low) & (valid_disp <= p_high)]

                    if trimmed_disp.size > 0:
                        disp_med = float(np.median(trimmed_disp))
                    else:
                        disp_med = float(np.median(valid_disp))

                    # XYZ in mm (processing-frame coordinates)
                    Z_mm = (fx_p * baseline_mm) / disp_med
                    X_mm = ((cx - cx0_p) * Z_mm) / fx_p
                    Y_mm = ((cy - cy0_p) * Z_mm) / fy_p

                    # Smooth XYZ with short median history
                    x_hist.append(X_mm)
                    y_hist.append(Y_mm)
                    z_hist.append(Z_mm)

                    X_mm_s = median_or_none(x_hist)
                    Y_mm_s = median_or_none(y_hist)
                    Z_mm_s = median_or_none(z_hist)

                    # Convert to inches first
                    X_in = X_mm_s * MM_TO_IN
                    Y_in = Y_mm_s * MM_TO_IN
                    Z_in = Z_mm_s * MM_TO_IN

                    # Round for display/output in 0.5-foot increments
                    X_ft = -round_to_half_foot(X_in)
                    Y_ft = -round_to_half_foot(Y_in)
                    Z_ft = round_to_half_foot(Z_in)

                    info_lines = [
                        f"X = {X_ft:.1f} ft",
                        f"Y = {Y_ft:.1f} ft",
                        f"Z = {Z_ft:.1f} ft",
                        f"FPS = {fps_ema:.1f}"
                    ]

                    if SHOW_DEBUG_DEPTH_TEXT:
                        info_lines.insert(3, f"disp = {disp_med:.2f}")
                        info_lines.insert(4, f"n = {valid_disp.size}")

                    xyz_text_terminal = f"X = {X_ft:.1f} ft | Y = {Y_ft:.1f} ft | Z = {Z_ft:.1f} ft"
                else:
                    info_lines = [
                        "Drone detected",
                        "Depth unavailable",
                        f"FPS = {fps_ema:.1f}"
                    ]
            else:
                x_hist.clear()
                y_hist.clear()
                z_hist.clear()
        else:
            x_hist.clear()
            y_hist.clear()
            z_hist.clear()
            info_lines = [
                "No drone detected",
                f"FPS = {fps_ema:.1f}"
            ]

        # Overlay text on left image
        y_text = 30
        for line in info_lines:
            cv2.putText(
                viewL,
                line,
                (20, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            y_text += 28

        # Resize for display
        left_disp = cv2.resize(viewL, (DISPLAY_W // 2, DISPLAY_H), interpolation=cv2.INTER_LINEAR)
        right_disp = cv2.resize(viewDisp, (DISPLAY_W // 2, DISPLAY_H), interpolation=cv2.INTER_LINEAR)

        combined = cv2.hconcat([left_disp, right_disp])
        cv2.putText(
            combined,
            FOOTER_TEXT,
            (20, DISPLAY_H - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(WINDOW_NAME, combined)

        if xyz_text_terminal is not None:
            print(xyz_text_terminal, end="\r")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            print("\nReopening cameras...")
            left.reopen()
            right.reopen()

    left.stop()
    right.stop()
    cv2.destroyAllWindows()
    print("\nDone.")

if __name__ == "__main__":
    main()


