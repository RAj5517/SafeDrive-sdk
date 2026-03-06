"""
realtime_detector.py  — v2.1
─────────────────────────────
CNN_WEIGHT = 0.5, EAR_WEIGHT = 0.5
Full logging to diagnose CNN domain gap before retraining.
"""

import cv2
import numpy as np
import torch
import threading
import queue
import time
import sys
import os
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))

from landmark_extractor import LandmarkExtractor
from ear_calculator import average_ear
from eye_extractor import extract_both_eyes
from perclos import PERCLOSTracker
from alarm import AlertSystem
from mobilenet_model import load_mobilenet, predict_eye_state

# ── Detection config ───────────────────────────────────────────────────────────
EAR_CLOSED_THRESH    = 0.20
CONSEC_FRAMES_CLOSED = 3
CNN_WEIGHT           = 0.50   # 50/50 split — testing CNN contribution
EAR_WEIGHT           = 0.50


class RealtimeDetector:

    def __init__(self,
                 model_path: str = "models/mobilenet_best.pth",
                 camera_id: int = 0,
                 fps_target: int = 30,
                 device: str = None):

        self.camera_id = camera_id
        self.device    = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("=" * 52)
        print("  SafeDrive — Real-time Drowsiness Detector v2.1")
        print("=" * 52)
        print(f"  Device:       {self.device}")
        print(f"  EAR thresh:   {EAR_CLOSED_THRESH}")
        print(f"  Smoothing:    {CONSEC_FRAMES_CLOSED} consecutive frames")
        print(f"  Weights:      EAR={EAR_WEIGHT}  CNN={CNN_WEIGHT}")

        print("\n  Loading components...")
        self.landmark_ext = LandmarkExtractor()
        self.perclos      = PERCLOSTracker(fps=fps_target)
        self.alert_system = AlertSystem(fps=fps_target)

        print("  Loading MobileNetV3...")
        self.model = load_mobilenet(model_path, self.device)
        print("  Model loaded ✅")

        self.closed_counter = 0
        self.frame_queue    = queue.Queue(maxsize=2)
        self.stop_event     = threading.Event()
        self.fps_buffer     = deque(maxlen=30)
        self.last_time      = time.time()

        print("\n  ✅ Ready — Press Q to quit\n")

    # ── Capture thread ─────────────────────────────────────────────────────────

    def _capture_thread(self, cap):
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    # ── FPS ────────────────────────────────────────────────────────────────────

    def _get_fps(self) -> float:
        now = time.time()
        self.fps_buffer.append(now - self.last_time)
        self.last_time = now
        if len(self.fps_buffer) < 2:
            return 0.0
        return 1.0 / (sum(self.fps_buffer) / len(self.fps_buffer))

    # ── Face box from landmarks ────────────────────────────────────────────────

    def _face_box_from_landmarks(self, data: dict, fh: int, fw: int) -> tuple:
        pts      = np.vstack([data["left_eye"], data["right_eye"]])
        cx       = float(pts[:, 0].mean())
        cy       = float(pts[:, 1].mean())
        eye_span = float(pts[:, 0].max() - pts[:, 0].min())
        face_w   = int(eye_span * 2.8)
        face_h   = int(face_w * 1.3)
        x = max(0, int(cx - face_w // 2))
        y = max(0, int(cy - face_h * 0.35))
        w = min(fw - x, face_w)
        h = min(fh - y, face_h)
        return (x, y, w, h)

    # ── Per-frame processing ───────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        fh, fw = frame.shape[:2]

        # 1. MediaPipe landmarks
        data = self.landmark_ext.extract(frame)
        if data is None:
            self.closed_counter = 0
            cv2.putText(frame, "No face detected",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (100, 100, 100), 2)
            return frame

        # 2. Face box from landmarks
        x, y, bw, bh = self._face_box_from_landmarks(data, fh, fw)
        cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 220, 255), 2)

        # 3. EAR
        ear        = average_ear(data["left_eye"], data["right_eye"])
        ear_closed = ear < EAR_CLOSED_THRESH

        # 4. CNN inference with detailed per-eye logging
        cnn_closed_prob = 0.0
        cnn_label       = "N/A"
        if CNN_WEIGHT > 0:
            eyes       = extract_both_eyes(frame, data["left_eye"], data["right_eye"])
            probs_list = []
            for side, eye_roi in [("L", eyes["left"]), ("R", eyes["right"])]:
                if eye_roi is not None:
                    result = predict_eye_state(self.model, eye_roi, self.device)
                    probs_list.append(result["closed_prob"])
                    probs = result["probabilities"]          # ← fixed
                    print(
                        f"  eye_{side}: "
                        f"open={probs[0]:.2f}  "
                        f"half={probs[1]:.2f}  "
                        f"closed={probs[2]:.2f}  "
                        f"→ {result['class_name']}"
                    )
                else:
                    print(f"  eye_{side}: None ← ROI extraction failed")  # ← new
            if probs_list:
                cnn_closed_prob = max(probs_list)
                cnn_label = "CLOSED" if cnn_closed_prob > 0.5 else "OPEN"

        # 5. Fusion score
        score = EAR_WEIGHT * float(ear_closed) + CNN_WEIGHT * cnn_closed_prob

        # 6. Temporal smoothing
        if score > 0.3:   # either signal suggests closed
            self.closed_counter = min(self.closed_counter + 1,
                                      CONSEC_FRAMES_CLOSED + 10)
        else:
            self.closed_counter = max(0, self.closed_counter - 1)

        is_closed = self.closed_counter >= CONSEC_FRAMES_CLOSED

        # 7. Fusion log — one line per frame
        print(
            f"EAR={ear:.3f}({'C' if ear_closed else 'O'})  "
            f"CNN={cnn_closed_prob:.3f}({cnn_label})  "
            f"score={score:.3f}  "
            f"cnt={self.closed_counter}/{CONSEC_FRAMES_CLOSED}  "
            f"→ {'⚠ DROWSY' if is_closed else 'ok'}"
        )

        # 8. PERCLOS + alert
        self.perclos.update(2 if is_closed else 0)
        self.alert_system.update(is_closed)
        frame = self.alert_system.draw_overlay(frame)

        # 9. Draw overlays
        fps         = self._get_fps()
        perclos_val = self.perclos.get_perclos()
        eye_label   = "CLOSED" if is_closed else "OPEN"
        eye_color   = (0, 0, 255) if is_closed else (0, 255, 0)

        # Eye landmark dots
        for pt in data["left_eye"]:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)
        for pt in data["right_eye"]:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (255, 255, 0), -1)

        # EAR bar (top right)
        bx    = fw - 180
        by    = 30
        e_len = min(int(ear * 500), 150)
        e_col = (0, 255, 0) if not ear_closed else (0, 0, 255)
        cv2.rectangle(frame, (bx, by), (bx + e_len, by + 14), e_col, -1)
        cv2.rectangle(frame, (bx, by), (bx + 150, by + 14), (180, 180, 180), 1)
        cv2.putText(frame, f"EAR {ear:.2f}",
                    (bx - 72, by + 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (220, 220, 220), 1)

        # Counter bar
        c_len = min(int((self.closed_counter / CONSEC_FRAMES_CLOSED) * 150), 150)
        cv2.rectangle(frame, (bx, by + 20), (bx + c_len, by + 34),
                      (0, 140, 255), -1)
        cv2.rectangle(frame, (bx, by + 20), (bx + 150, by + 34),
                      (180, 180, 180), 1)
        cv2.putText(frame, f"CNT {self.closed_counter}/{CONSEC_FRAMES_CLOSED}",
                    (bx - 72, by + 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (220, 220, 220), 1)

        # CNN bar
        cn_len = min(int(cnn_closed_prob * 150), 150)
        cn_col = (0, 0, 255) if cnn_closed_prob > 0.5 else (200, 100, 0)
        cv2.rectangle(frame, (bx, by + 40), (bx + cn_len, by + 54),
                      cn_col, -1)
        cv2.rectangle(frame, (bx, by + 40), (bx + 150, by + 54),
                      (180, 180, 180), 1)
        cv2.putText(frame, f"CNN {cnn_closed_prob:.2f}",
                    (bx - 72, by + 52), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (220, 220, 220), 1)

        # Stats panel (top left)
        stats = [
            (f"Eye:     {eye_label}",                       eye_color),
            (f"EAR:     {ear:.3f}",                         (255, 255, 255)),
            (f"CNN:     {cnn_closed_prob:.2f} {cnn_label}", (200, 200, 255)),
            (f"Score:   {score:.3f}",                       (255, 200, 100)),
            (f"PERCLOS: {perclos_val:.2f}",                 (255, 255, 0)),
            (f"FPS:     {fps:.1f}",                         (180, 180, 180)),
        ]
        for i, (text, color) in enumerate(stats):
            cv2.putText(frame, text,
                        (10, 35 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)

        return frame

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")

        print(f"Camera: {cap.get(cv2.CAP_PROP_FPS):.0f} FPS")

        t = threading.Thread(
            target=self._capture_thread, args=(cap,), daemon=True
        )
        t.start()

        try:
            while True:
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    break
                with torch.no_grad():
                    annotated = self._process_frame(frame)
                cv2.imshow("SafeDrive — Drowsiness Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.stop_event.set()
            cap.release()
            cv2.destroyAllWindows()
            self.landmark_ext.close()
            print("Pipeline stopped.")