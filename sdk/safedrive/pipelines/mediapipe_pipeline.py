"""
mediapipe_pipeline.py
─────────────────────
MediaPipe + MobileNetV3 pipeline for SafeDrive SDK.

Models auto-downloaded from HuggingFace on first run:
    raj5517/safedrive-model

Uses:
    MediaPipe Face Landmarker  → 468 face landmarks → eye positions
    EAR formula                → geometric eye openness
    MobileNetV3 (webcam ft)    → CNN eye state classification
    Fusion                     → EAR + CNN combined score
"""

import cv2
import numpy as np
import torch
import time
from collections import deque
from pathlib import Path

from .base_pipeline import BasePipeline

# ── Detection config ──────────────────────────────────────────────────────────
EAR_CLOSED_THRESH = 0.20
CNN_WEIGHT        = 0.50
EAR_WEIGHT        = 0.50


class MediaPipePipeline(BasePipeline):
    """
    MediaPipe + MobileNetV3 drowsiness detection pipeline.
    Models are auto-downloaded from HuggingFace if not cached.
    """

    def __init__(self,
                 model_path:    str   = None,
                 device:        str   = None,
                 ear_weight:    float = EAR_WEIGHT,
                 cnn_weight:    float = CNN_WEIGHT,
                 ear_threshold: float = EAR_CLOSED_THRESH):

        self.model_path    = model_path   # None = auto-download
        self.device        = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.ear_weight    = ear_weight
        self.cnn_weight    = cnn_weight
        self.ear_threshold = ear_threshold

        self._landmark_ext = None
        self._model        = None
        self._perclos      = None
        self._fps_buffer   = deque(maxlen=30)
        self._last_time    = time.time()
        self._ready        = False

    @property
    def name(self) -> str:
        return "mediapipe"

    def start(self) -> None:
        """
        Load all models and initialize MediaPipe.
        Downloads from HuggingFace automatically if not cached.
        """
        import sys

        # ── Resolve src/ path ─────────────────────────────────────────────
        for candidate in ["src", "../src", "../../src"]:
            if Path(candidate).exists():
                sys.path.insert(0, str(Path(candidate).resolve()))
                break

        # ── Import after path setup ───────────────────────────────────────
        from landmark_extractor import LandmarkExtractor
        from safedrive.perclos import PERCLOSTracker
        from mobilenet_model import load_mobilenet
        from safedrive.model_manager import get_best_eye_model, get_model_path

        # ── Face landmarker — auto download if missing ────────────────────
        landmark_path = get_model_path("face_landmarker")

        # ── Eye model — user override OR auto-select best cached ──────────
        model_to_load = self.model_path or get_best_eye_model()

        # ── Init MediaPipe with resolved landmark path ────────────────────
        self._landmark_ext = LandmarkExtractor(task_path=landmark_path)
        self._perclos      = PERCLOSTracker(fps=30)

        # ── Load MobileNetV3 weights ──────────────────────────────────────
        print(f"  [MediaPipe] Loading model: {Path(model_to_load).name}")
        self._model = load_mobilenet(model_to_load, self.device)

        self._ready = True
        print(f"  [MediaPipe] Pipeline ready  |  device={self.device}")

    def stop(self) -> None:
        """Release all resources."""
        if self._landmark_ext:
            self._landmark_ext.close()
        self._ready = False
        print("  [MediaPipe] Pipeline stopped.")

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process one frame through MediaPipe + CNN pipeline.

        Returns standardized result dict:
            ear, eye_state, cnn_prob, score,
            face_found, perclos, landmarks, frame, fps
        """
        if not self._ready:
            raise RuntimeError("Call pipeline.start() first.")

        from ear_calculator import average_ear
        from eye_extractor import extract_both_eyes
        from mobilenet_model import predict_eye_state

        annotated = frame.copy()

        # ── Face landmarks ────────────────────────────────────────────────
        data = self._landmark_ext.extract(frame)
        if data is None:
            return {
                "ear":        0.0,
                "eye_state":  "unknown",
                "cnn_prob":   0.0,
                "score":      0.0,
                "face_found": False,
                "perclos":    self._perclos.get_perclos(),
                "landmarks":  None,
                "frame":      annotated,
                "fps":        self._get_fps(),
            }

        # ── EAR ───────────────────────────────────────────────────────────
        ear        = average_ear(data["left_eye"], data["right_eye"])
        ear_closed = ear < self.ear_threshold

        # ── CNN inference ─────────────────────────────────────────────────
        cnn_closed_prob = 0.0
        if self.cnn_weight > 0 and self._model:
            eyes = extract_both_eyes(
                frame, data["left_eye"], data["right_eye"])
            probs_list = []
            for roi in [eyes["left"], eyes["right"]]:
                if roi is not None:
                    result = predict_eye_state(self._model, roi, self.device)
                    probs_list.append(result["closed_prob"])
            if probs_list:
                cnn_closed_prob = max(probs_list)

        # ── Fusion score ──────────────────────────────────────────────────
        score = (self.ear_weight * float(ear_closed)
                 + self.cnn_weight * cnn_closed_prob)

        # ── Eye state label ───────────────────────────────────────────────
        if ear < self.ear_threshold:
            eye_state = "closed"
        elif ear < self.ear_threshold + 0.05:
            eye_state = "half"
        else:
            eye_state = "open"

        # ── PERCLOS ───────────────────────────────────────────────────────
        self._perclos.update(2 if eye_state == "closed" else 0)

        # ── Annotate frame ────────────────────────────────────────────────
        annotated = self._draw_landmarks(annotated, data)
        annotated = self._draw_bars(
            annotated, ear, score, cnn_closed_prob, frame.shape[1])

        return {
            "ear":        ear,
            "eye_state":  eye_state,
            "cnn_prob":   cnn_closed_prob,
            "score":      score,
            "face_found": True,
            "perclos":    self._perclos.get_perclos(),
            "landmarks":  data,
            "frame":      annotated,
            "fps":        self._get_fps(),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_fps(self) -> float:
        now = time.time()
        self._fps_buffer.append(now - self._last_time)
        self._last_time = now
        if len(self._fps_buffer) < 2:
            return 0.0
        return 1.0 / (sum(self._fps_buffer) / len(self._fps_buffer))

    def _draw_landmarks(self, frame: np.ndarray, data: dict) -> np.ndarray:
        for pt in data["left_eye"]:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)
        for pt in data["right_eye"]:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (255, 255, 0), -1)
        return frame

    def _draw_bars(self, frame, ear, score, cnn_prob, fw) -> np.ndarray:
        font = cv2.FONT_HERSHEY_SIMPLEX
        bx   = fw - 180
        by   = 30

        def bar(y, val, color, label):
            length = min(int(val * 150), 150)
            cv2.rectangle(frame, (bx, y), (bx + length, y + 14), color, -1)
            cv2.rectangle(frame, (bx, y), (bx + 150,    y + 14),
                          (180, 180, 180), 1)
            cv2.putText(frame, label, (bx - 72, y + 12),
                        font, 0.45, (220, 220, 220), 1)

        e_col = (0, 255, 0) if ear >= self.ear_threshold else (0, 0, 255)
        bar(by,      min(ear * 3, 1.0), e_col,       f"EAR {ear:.2f}")
        bar(by + 20, min(score,   1.0), (0, 140, 255), f"SCR {score:.2f}")
        bar(by + 40, cnn_prob,          (200, 100, 0), f"CNN {cnn_prob:.2f}")
        return frame