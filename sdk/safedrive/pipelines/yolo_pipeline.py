"""
yolo_pipeline.py
────────────────
YOLOv8 multi-feature pipeline for SafeDrive SDK v0.2.0.

Detects in a single pass:
    0  eye_open       → drowsiness
    1  eye_half       → drowsiness
    2  eye_closed     → drowsiness
    3  mouth_open     → yawn detection
    4  mouth_closed
    5  phone          → distraction alert
    6  cigarette      → distraction alert
    7  seatbelt_on    → safety monitoring
    8  seatbelt_off   → safety alert

Head pose (nodding/tilting) uses MediaPipe landmarks —
YOLO cannot reliably output angles, this is industry standard approach.
"""

import cv2
import numpy as np
import time
from collections import deque
from pathlib import Path

from .base_pipeline import BasePipeline

# ── Class IDs (must match training) ──────────────────────────────────────────
CLS_EYE_OPEN     = 0
CLS_EYE_HALF     = 1
CLS_EYE_CLOSED   = 2
CLS_MOUTH_OPEN   = 3
CLS_MOUTH_CLOSED = 4
CLS_PHONE        = 5
CLS_CIGARETTE    = 6
CLS_SEATBELT_ON  = 7
CLS_SEATBELT_OFF = 8

CLASS_NAMES = {
    0: "eye_open",   1: "eye_half",    2: "eye_closed",
    3: "mouth_open", 4: "mouth_closed",
    5: "phone",      6: "cigarette",
    7: "seatbelt_on", 8: "seatbelt_off",
}

# ── Detection thresholds ──────────────────────────────────────────────────────
CONF_THRESHOLD   = 0.45    # minimum confidence to count a detection
IOU_THRESHOLD    = 0.45    # NMS IOU threshold
EYE_CLOSED_CONF  = 0.50    # min confidence to count eye as closed

# ── Colors ────────────────────────────────────────────────────────────────────
BOX_COLORS = {
    0: (0, 255,   0),    # eye_open    → green
    1: (0, 140, 255),    # eye_half    → orange
    2: (0,   0, 255),    # eye_closed  → red
    3: (0, 220, 255),    # mouth_open  → yellow
    4: (200,200,200),    # mouth_closed→ gray
    5: (255,   0, 255),  # phone       → magenta
    6: (128,   0, 128),  # cigarette   → purple
    7: (0,   255, 128),  # seatbelt_on → teal
    8: (0,   100, 255),  # seatbelt_off→ orange-red
}


class YoloPipeline(BasePipeline):
    """
    YOLOv8-nano multi-feature drowsiness detection pipeline.
    Single forward pass detects eyes, mouth, phone, seatbelt, cigarette.

    Head pose uses MediaPipe (angles can't be detected by YOLO).
    """

    def __init__(self,
                 model_path:      str   = None,
                 device:          str   = None,
                 conf:            float = CONF_THRESHOLD,
                 iou:             float = IOU_THRESHOLD,
                 detect_phone:    bool  = True,
                 detect_seatbelt: bool  = True,
                 detect_smoking:  bool  = True,
                 detect_yawn:     bool  = True):

        self.model_path      = model_path
        self.device          = device or "cuda"
        self.conf            = conf
        self.iou             = iou
        self.detect_phone    = detect_phone
        self.detect_seatbelt = detect_seatbelt
        self.detect_smoking  = detect_smoking
        self.detect_yawn     = detect_yawn

        self._model       = None
        self._mp_landmarker = None
        self._perclos     = None
        self._fps_buffer  = deque(maxlen=30)
        self._last_time   = time.time()
        self._ready       = False

    @property
    def name(self) -> str:
        return "yolo"

    def start(self) -> None:
        """Load YOLOv8 model and MediaPipe for head pose."""
        import sys
        for candidate in ["src", "../src", "../../src"]:
            if Path(candidate).exists():
                sys.path.insert(0, str(Path(candidate).resolve()))
                break

        from safedrive.model_manager import get_model_path
        from perclos import PERCLOSTracker

        # ── YOLO model ────────────────────────────────────────────────────
        model_to_load = self.model_path or get_model_path("yolo_safedrive")

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics not installed.\n"
                "Run: pip install safedrive-ai[yolo]\n"
                "or:  pip install ultralytics"
            )

        self._model   = YOLO(model_to_load)
        self._perclos = PERCLOSTracker(fps=30)

        # ── MediaPipe for head pose ───────────────────────────────────────
        self._init_mediapipe(get_model_path("face_landmarker"))

        self._ready = True
        print(f"  [YOLO] Pipeline ready  |  device={self.device}")

    def _init_mediapipe(self, task_path: str) -> None:
        """Initialize MediaPipe for head pose estimation."""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=task_path),
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._mp_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            print("  [YOLO] MediaPipe head pose ready ✅")
        except Exception as e:
            print(f"  [YOLO] Head pose unavailable: {e}")
            self._mp_landmarker = None

    def stop(self) -> None:
        if self._mp_landmarker:
            self._mp_landmarker.close()
        self._ready = False
        print("  [YOLO] Pipeline stopped.")

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Single YOLO forward pass → detects all features.

        Returns standardized result dict compatible with AlertSystem.
        Adds extra keys: phone_detected, smoking_detected,
                         seatbelt_present, head_tilt, head_nod,
                         yawn_detected, detections
        """
        if not self._ready:
            raise RuntimeError("Call pipeline.start() first.")

        import mediapipe as mp
        fh, fw    = frame.shape[:2]
        annotated = frame.copy()

        # ── YOLO inference ────────────────────────────────────────────────
        results = self._model.predict(
            source  = frame,
            conf    = self.conf,
            iou     = self.iou,
            device  = self.device,
            verbose = False,
        )

        # ── Parse detections ──────────────────────────────────────────────
        detections  = []
        eye_states  = []   # all detected eye states this frame
        mouth_open  = False
        phone       = False
        phone_conf  = 0.0
        cigarette   = False
        cig_conf    = 0.0
        seatbelt_on = None   # None = not detected

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf   = float(box.conf[0].item())
                xyxy   = box.xyxy[0].cpu().numpy().astype(int)

                detections.append({
                    "class_id":   cls_id,
                    "class_name": CLASS_NAMES.get(cls_id, str(cls_id)),
                    "conf":       conf,
                    "bbox":       xyxy.tolist(),
                })

                if cls_id in (CLS_EYE_OPEN, CLS_EYE_HALF, CLS_EYE_CLOSED):
                    eye_states.append((cls_id, conf))

                elif cls_id == CLS_MOUTH_OPEN and self.detect_yawn:
                    mouth_open = True

                elif cls_id == CLS_PHONE and self.detect_phone:
                    phone      = True
                    phone_conf = max(phone_conf, conf)

                elif cls_id == CLS_CIGARETTE and self.detect_smoking:
                    cigarette = True
                    cig_conf  = max(cig_conf, conf)

                elif cls_id == CLS_SEATBELT_ON and self.detect_seatbelt:
                    seatbelt_on = True

                elif cls_id == CLS_SEATBELT_OFF and self.detect_seatbelt:
                    if seatbelt_on is None:
                        seatbelt_on = False

                # Draw bounding box
                x1, y1, x2, y2 = xyxy
                color = BOX_COLORS.get(cls_id, (255,255,255))
                cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
                label = f"{CLASS_NAMES.get(cls_id,str(cls_id))} {conf:.2f}"
                cv2.putText(annotated, label, (x1, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # ── Derive eye state from detections ──────────────────────────────
        # Take the worst (most closed) eye state detected
        face_found = len(eye_states) > 0

        if not eye_states:
            eye_state    = "unknown"
            cnn_prob     = 0.0
            score        = 0.0
            ear          = 0.0
        else:
            # Worst eye wins
            if any(cid == CLS_EYE_CLOSED and c >= EYE_CLOSED_CONF
                   for cid, c in eye_states):
                eye_state = "closed"
                cnn_prob  = max(c for cid, c in eye_states
                                if cid == CLS_EYE_CLOSED)
            elif any(cid == CLS_EYE_HALF for cid, c in eye_states):
                eye_state = "half"
                cnn_prob  = max(c for cid, c in eye_states
                                if cid == CLS_EYE_HALF) * 0.5
            else:
                eye_state = "open"
                cnn_prob  = 0.0

            score = cnn_prob
            ear   = 0.25 if eye_state == "open" else (
                    0.20 if eye_state == "half" else 0.15)

        # ── Head pose via MediaPipe ───────────────────────────────────────
        head_tilt, head_nod = self._get_head_pose(frame)

        # ── PERCLOS ───────────────────────────────────────────────────────
        self._perclos.update(2 if eye_state == "closed" else 0)

        # ── Draw HUD ──────────────────────────────────────────────────────
        annotated = self._draw_hud(
            annotated, eye_state, score, mouth_open,
            phone, cigarette,
            seatbelt_on, head_tilt, head_nod,
            self._get_fps(), fw, fh
        )

        return {
            # Standard keys (same as MediaPipe pipeline)
            "ear":        ear,
            "eye_state":  eye_state,
            "cnn_prob":   cnn_prob,
            "score":      score,
            "face_found": face_found,
            "perclos":    self._perclos.get_perclos(),
            "landmarks":  None,
            "frame":      annotated,
            "fps":        self._get_fps(),

            # YOLO-specific keys (used by AlertSystem)
            "phone_detected":       phone,
            "phone_confidence":     phone_conf,
            "smoking_detected":     cigarette,
            "smoking_confidence":   cig_conf,
            "seatbelt_present":     seatbelt_on,
            "yawn_detected":        mouth_open,
            "head_tilt":            head_tilt,
            "head_nod":             head_nod,
            "detections":           detections,
        }

    def _get_head_pose(self, frame: np.ndarray) -> tuple[float, float]:
        """Estimate head tilt and nod from MediaPipe landmarks."""
        if self._mp_landmarker is None:
            return 0.0, 0.0

        try:
            import mediapipe as mp
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._mp_landmarker.detect(mp_img)

            if not result.face_landmarks:
                return 0.0, 0.0

            lm = result.face_landmarks[0]
            fh, fw = frame.shape[:2]

            # Use nose tip (1), chin (152), left ear (234), right ear (454)
            nose  = np.array([lm[1].x * fw,   lm[1].y * fh])
            chin  = np.array([lm[152].x * fw, lm[152].y * fh])
            l_ear = np.array([lm[234].x * fw, lm[234].y * fh])
            r_ear = np.array([lm[454].x * fw, lm[454].y * fh])

            # Tilt = angle of ear-to-ear line from horizontal
            ear_vec  = r_ear - l_ear
            tilt_deg = abs(np.degrees(np.arctan2(ear_vec[1], ear_vec[0])))

            # Nod = nose relative to midpoint of ears (vertical drop)
            ear_mid  = (l_ear + r_ear) / 2
            nod_deg  = np.degrees(
                np.arctan2(chin[1] - nose[1], abs(chin[0] - ear_mid[0]) + 1e-6))
            nod_deg  = max(0, nod_deg - 60)   # 60° is neutral, excess = nod

            return float(tilt_deg), float(nod_deg)

        except Exception:
            return 0.0, 0.0

    def _draw_hud(self, frame, eye_state, score, mouth_open,
                  phone, cigarette, seatbelt_on,
                  tilt, nod, fps, fw, fh) -> np.ndarray:
        font = cv2.FONT_HERSHEY_SIMPLEX
        ov   = frame.copy()
        cv2.rectangle(ov, (0, 0), (fw, 160), (0,0,0), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

        W = (255,255,255)
        def feat(enabled, detected, label_on, label_off, col_on, col_off):
            if not enabled:
                return (f"{label_on:<9}: disabled", (80, 80, 80))
            if detected:
                return (f"{label_on:<9}: {label_on.upper()}", col_on)
            return (f"{label_on:<9}: {label_off}", col_off)

        lines = [
            (f"Pipeline : YOLO v0.2.0",                                        (180,180,255)),
            (f"Eye      : {eye_state.upper():<8} score={score:.2f}",            W),
            feat(True,         mouth_open, "Mouth",     "closed",   (0,220,255), W),
            feat(self.detect_phone,    phone,     "Phone",     "clear",    (255,0,255),  W),
            feat(self.detect_smoking,  cigarette, "Cigarette", "clear",    (200,0,200),  W),
            feat(self.detect_seatbelt,
                 seatbelt_on is True, "Seatbelt",
                 "OFF" if seatbelt_on is False else "n/a",
                 (0,255,128), (0,100,255) if seatbelt_on is False else W),
            (f"Head     : tilt={tilt:.1f}°  nod={nod:.1f}°  FPS={fps:.1f}",   W),
        ]
        for i, (text, col) in enumerate(lines):
            cv2.putText(frame, text, (12, 26 + i*21), font, 0.5, col, 1)

        return frame

    def _get_fps(self) -> float:
        now = time.time()
        self._fps_buffer.append(now - self._last_time)
        self._last_time = now
        if len(self._fps_buffer) < 2:
            return 0.0
        return 1.0 / (sum(self._fps_buffer) / len(self._fps_buffer))