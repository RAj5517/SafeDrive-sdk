"""
realtime_detector.py
────────────────────
Main real-time drowsiness detection pipeline.

Architecture:
    Thread 1 (capture_thread):  reads frames from webcam → puts in queue
    Thread 2 (main thread):     processes frames → displays output

Face detection runs every 3rd frame.
Frames 1 & 2 use OpenCV CSRT tracker (faster than re-detecting).

Fusion: combined_score = 0.4 × EAR_closed + 0.6 × CNN_closed_prob
        If combined_score > 0.5 → eyes closed

FPS: rolling average of last 30 frames displayed on screen.
"""

import cv2
import numpy as np
import torch
import threading
import queue
import time
from collections import deque

from face_detector import FaceDetector
from landmark_extractor import LandmarkExtractor
from ear_calculator import average_ear, is_eye_closed
from eye_extractor import extract_both_eyes
from eye_state_model import EyeStateCNN, load_model, predict_eye_state, CLASS_NAMES
from perclos import PERCLOSTracker
from alarm import AlertSystem


# ── Fusion weights ─────────────────────────────────────────────────────────────
EAR_WEIGHT  = 0.4
CNN_WEIGHT  = 0.6
FUSION_THRESH = 0.5

# ── Face detection every N frames ──────────────────────────────────────────────
DETECT_EVERY = 3


class RealtimeDetector:
    """
    Full real-time drowsiness detection pipeline.

    Args:
        model_path:   Path to trained .pth model weights
        camera_id:    Webcam index (default 0)
        fps_target:   Target FPS for display
        device:       torch device ('cuda' or 'cpu')
    """

    def __init__(self,
                 model_path: str,
                 camera_id: int = 0,
                 fps_target: int = 30,
                 device: str = None):

        self.camera_id  = camera_id
        self.fps_target = fps_target
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Device: {self.device}")
        print(f"Loading model from: {model_path}")

        # Components
        self.face_detector   = FaceDetector()
        self.landmark_extractor = LandmarkExtractor()
        self.model           = load_model(model_path, self.device)
        self.perclos         = PERCLOSTracker(fps=fps_target)
        self.alert_system    = AlertSystem(fps=fps_target)

        # CSRT tracker (used between face detection frames)
        self.tracker         = None
        self.tracking_face   = False
        self.frame_idx       = 0

        # Threading
        self.frame_queue  = queue.Queue(maxsize=2)
        self.stop_event   = threading.Event()

        # FPS tracking
        self.fps_buffer   = deque(maxlen=30)
        self.last_time    = time.time()

    def _capture_thread(self, cap):
        """Runs in background — reads frames and puts in queue."""
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass   # drop frame if processing is behind

    def _get_fps(self) -> float:
        now = time.time()
        self.fps_buffer.append(now - self.last_time)
        self.last_time = now
        if len(self.fps_buffer) < 2:
            return 0.0
        return 1.0 / (sum(self.fps_buffer) / len(self.fps_buffer))

    def _detect_or_track(self, frame):
        """
        Returns face bounding box.
        Runs full detection every DETECT_EVERY frames, tracks in between.
        """
        self.frame_idx += 1

        if self.frame_idx % DETECT_EVERY == 0 or not self.tracking_face:
            # Full detection
            box = self.face_detector.detect(frame)
            if box:
                x, y, w, h = box
                # Reinitialize tracker
                self.tracker = cv2.TrackerCSRT_create()
                self.tracker.init(frame, (x, y, w, h))
                self.tracking_face = True
                return box
            else:
                self.tracking_face = False
                return None
        else:
            # Use tracker
            ok, bbox = self.tracker.update(frame)
            if ok:
                return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            else:
                self.tracking_face = False
                return None

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run full pipeline on one frame. Returns annotated frame."""

        # ── 1. Face detection / tracking ──────────────────────────────────────
        box = self._detect_or_track(frame)
        if box is None:
            cv2.putText(frame, "No face detected", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            return frame

        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)

        # ── 2. Landmark extraction ─────────────────────────────────────────────
        data = self.landmark_extractor.extract(frame)
        if data is None:
            return frame

        # ── 3. EAR calculation ─────────────────────────────────────────────────
        ear = average_ear(data["left_eye"], data["right_eye"])
        ear_closed = 1.0 if is_eye_closed(ear) else 0.0

        # ── 4. CNN inference ───────────────────────────────────────────────────
        eyes = extract_both_eyes(frame, data["left_eye"], data["right_eye"])
        cnn_closed_prob = 0.0

        for eye_roi in [eyes["left"], eyes["right"]]:
            if eye_roi is not None:
                result = predict_eye_state(self.model, eye_roi, self.device)
                cnn_closed_prob = max(cnn_closed_prob, result["closed_prob"])

        # ── 5. Fusion ──────────────────────────────────────────────────────────
        combined = EAR_WEIGHT * ear_closed + CNN_WEIGHT * cnn_closed_prob
        is_closed = combined > FUSION_THRESH

        # ── 6. Update PERCLOS + Alert ──────────────────────────────────────────
        self.perclos.update(2 if is_closed else 0)
        self.alert_system.update(is_closed)

        # ── 7. Draw overlays ───────────────────────────────────────────────────
        frame = self.alert_system.draw_overlay(frame)

        # Stats panel
        perclos_val = self.perclos.get_perclos()
        fps = self._get_fps()
        eye_label = "CLOSED" if is_closed else "OPEN"
        eye_color = (0, 0, 255) if is_closed else (0, 255, 0)

        cv2.putText(frame, f"Eye: {eye_label}",     (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
        cv2.putText(frame, f"EAR: {ear:.3f}",       (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"CNN: {cnn_closed_prob:.2f}", (10, 85),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"PERCLOS: {perclos_val:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS: {fps:.1f}",        (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return frame

    def run(self):
        """Start the real-time detection loop."""
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # prevent stale frames

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")

        actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"Camera FPS: {actual_fps}")
        print("Press Q to quit.")

        # Start capture thread
        t = threading.Thread(target=self._capture_thread, args=(cap,), daemon=True)
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
            self.face_detector.close()
            self.landmark_extractor.close()
            print("Detector stopped.")