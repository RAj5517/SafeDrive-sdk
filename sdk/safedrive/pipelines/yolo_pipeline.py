"""
yolo_pipeline.py
────────────────
YOLOv8 multi-feature pipeline for SafeDrive SDK v0.2.0.

Architecture (final):
    MediaPipe  →  eye state, PERCLOS, head pose, yawn (MAR)
    YOLO       →  phone, seatbelt, cigarette

Why not YOLO for eyes?
    Eye training data was collected as isolated crops, not full-frame scenes.
    The model detects eyes perfectly on training images (eye_open:0.90,
    mouth_open:0.96 confirmed) but cannot generalize to full webcam frames
    where the eye region is a small part of the scene.
    MediaPipe is scale-invariant (landmark math) and has no this limitation.

This architecture is the precursor to the v1.0.0 ensemble pipeline, which
will formally combine both models with a fusion layer.
"""

import cv2
import numpy as np
import time
from collections import deque
from pathlib import Path

from .base_pipeline import BasePipeline

# ── YOLO Class IDs (object detection only) ───────────────────────────────────
CLS_PHONE        = 5
CLS_CIGARETTE    = 6
CLS_SEATBELT_ON  = 7
CLS_SEATBELT_OFF = 8

CLASS_NAMES = {
    0: "eye_open",    1: "eye_half",     2: "eye_closed",
    3: "mouth_open",  4: "mouth_closed",
    5: "phone",       6: "cigarette",
    7: "seatbelt_on", 8: "seatbelt_off",
}

# ── Thresholds ────────────────────────────────────────────────────────────────
CONF_THRESHOLD   = 0.30
IOU_THRESHOLD    = 0.45

# ── EAR thresholds (Soukupova & Cech, 2016) ──────────────────────────────────
EAR_OPEN         = 0.25
EAR_HALF         = 0.20
# below EAR_HALF = closed

# ── MAR threshold for yawn ────────────────────────────────────────────────────
MAR_YAWN         = 0.6

# ── Colors ────────────────────────────────────────────────────────────────────
BOX_COLORS = {
    5: (255,   0, 255),   # phone        magenta
    6: (128,   0, 128),   # cigarette    purple
    7: (0,   255, 128),   # seatbelt_on  teal
    8: (0,   100, 255),   # seatbelt_off orange-red
}


class YoloPipeline(BasePipeline):
    """
    YOLO + MediaPipe hybrid pipeline.

    MediaPipe: eye state, PERCLOS, head pose, yawn
    YOLO:      phone, seatbelt, cigarette
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

        self._model          = None
        self._mp_landmarker  = None
        self._perclos        = None
        self._fps_buffer     = deque(maxlen=30)
        self._last_time      = time.time()
        self._ready          = False

    @property
    def name(self) -> str:
        return "yolo"

    def start(self) -> None:
        from safedrive.model_manager import get_model_path
        from safedrive.perclos import PERCLOSTracker

        model_to_load = self.model_path or get_model_path("yolo_safedrive")

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics not installed.\n"
                "Run: pip install safedrive-ai[yolo]"
            )

        self._model   = YOLO(model_to_load)
        self._perclos = PERCLOSTracker(fps=30)
        self._init_mediapipe(get_model_path("face_landmarker"))

        self._ready = True
        print(f"  [YOLO] Pipeline ready  |  device={self.device}")
        print(f"  [YOLO] Eyes/pose: MediaPipe  |  Objects: YOLO conf={self.conf}")

    def _init_mediapipe(self, task_path: str) -> None:
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
            print("  [YOLO] MediaPipe ready ✅")
        except Exception as e:
            print(f"  [YOLO] MediaPipe unavailable: {e}")
            self._mp_landmarker = None

    def stop(self) -> None:
        if self._mp_landmarker:
            self._mp_landmarker.close()
        self._ready = False
        print("  [YOLO] Pipeline stopped.")

    def process_frame(self, frame: np.ndarray) -> dict:
        if not self._ready:
            raise RuntimeError("Call pipeline.start() first.")

        fh, fw    = frame.shape[:2]
        annotated = frame.copy()

        # ── MediaPipe: eyes + head pose + yawn ───────────────────────────
        mp_result = self._run_mediapipe(frame)
        eye_state  = mp_result["eye_state"]
        ear        = mp_result["ear"]
        mar        = mp_result["mar"]
        head_tilt  = mp_result["head_tilt"]
        head_nod   = mp_result["head_nod"]
        face_found = mp_result["face_found"]
        landmarks  = mp_result["landmarks"]

        # Yawn from MAR
        mouth_open = (mar > MAR_YAWN) if self.detect_yawn else False

        # CNN prob approximation from EAR
        if eye_state == "closed":
            cnn_prob = 0.85
        elif eye_state == "half":
            cnn_prob = 0.45
        else:
            cnn_prob = 0.05
        score = cnn_prob if face_found else 0.0

        # PERCLOS
        if face_found:
            self._perclos.update(2 if eye_state == "closed" else 0)

        # ── YOLO: phone + seatbelt + cigarette ───────────────────────────
        phone        = False
        phone_conf   = 0.0
        cigarette    = False
        cig_conf     = 0.0
        seatbelt_on  = None
        detections   = []

        r = self._model.predict(
            source=frame, conf=self.conf, iou=self.iou,
            device=self.device, verbose=False)

        if r and r[0].boxes is not None:
            for box in r[0].boxes:
                cls_id = int(box.cls[0].item())
                conf_v = float(box.conf[0].item())
                xyxy   = box.xyxy[0].cpu().numpy().astype(int)

                if cls_id == CLS_PHONE and self.detect_phone:
                    phone = True; phone_conf = max(phone_conf, conf_v)
                elif cls_id == CLS_CIGARETTE and self.detect_smoking:
                    cigarette = True; cig_conf = max(cig_conf, conf_v)
                elif cls_id == CLS_SEATBELT_ON and self.detect_seatbelt:
                    seatbelt_on = True
                elif cls_id == CLS_SEATBELT_OFF and self.detect_seatbelt:
                    if seatbelt_on is None:
                        seatbelt_on = False

                if cls_id in BOX_COLORS:
                    detections.append({
                        "class_id":   cls_id,
                        "class_name": CLASS_NAMES.get(cls_id),
                        "conf":       conf_v,
                        "bbox":       xyxy.tolist(),
                    })
                    x1, y1, x2, y2 = xyxy
                    color = BOX_COLORS[cls_id]
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(annotated,
                                f"{CLASS_NAMES.get(cls_id)} {conf_v:.2f}",
                                (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 1)

        # ── HUD ───────────────────────────────────────────────────────────
        annotated = self._draw_hud(
            annotated, eye_state, score, mouth_open, mar,
            phone, cigarette, seatbelt_on,
            head_tilt, head_nod, self._get_fps(), fw, fh)

        return {
            "ear":        ear,
            "eye_state":  eye_state,
            "cnn_prob":   cnn_prob,
            "score":      score,
            "face_found": face_found,
            "perclos":    self._perclos.get_perclos(),
            "landmarks":  landmarks,
            "frame":      annotated,
            "fps":        self._get_fps(),
            "phone_detected":     phone,
            "phone_confidence":   phone_conf,
            "smoking_detected":   cigarette,
            "smoking_confidence": cig_conf,
            "seatbelt_present":   seatbelt_on,
            "yawn_detected":      mouth_open,
            "head_tilt":          head_tilt,
            "head_nod":           head_nod,
            "detections":         detections,
        }

    def _run_mediapipe(self, frame: np.ndarray) -> dict:
        """
        Run MediaPipe face landmarker.
        Returns eye_state, EAR, MAR, head_tilt, head_nod, face_found.
        """
        default = {
            "eye_state": "unknown", "ear": 0.0, "mar": 0.0,
            "head_tilt": 0.0, "head_nod": 0.0,
            "face_found": False, "landmarks": None,
        }

        if self._mp_landmarker is None:
            return default

        try:
            import mediapipe as mp
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._mp_landmarker.detect(mp_img)

            if not result.face_landmarks:
                return default

            lm = result.face_landmarks[0]
            fh, fw = frame.shape[:2]

            def pt(idx):
                return np.array([lm[idx].x * fw, lm[idx].y * fh])

            # ── EAR (both eyes averaged) ──────────────────────────────────
            # Right eye landmarks: 33,160,158,133,153,144
            # Left eye landmarks:  362,385,387,263,373,380
            def ear(p1,p2,p3,p4,p5,p6):
                return (np.linalg.norm(pt(p2)-pt(p6)) +
                        np.linalg.norm(pt(p3)-pt(p5))) / \
                       (2.0 * np.linalg.norm(pt(p1)-pt(p4)) + 1e-6)

            ear_right = ear(33,  160, 158, 133, 153, 144)
            ear_left  = ear(362, 385, 387, 263, 373, 380)
            ear_avg   = (ear_right + ear_left) / 2.0

            if ear_avg >= EAR_OPEN:
                eye_state = "open"
            elif ear_avg >= EAR_HALF:
                eye_state = "half"
            else:
                eye_state = "closed"

            # ── MAR (mouth aspect ratio) ──────────────────────────────────
            # Top:13 Bottom:14 Left:78 Right:308
            mar = (np.linalg.norm(pt(13) - pt(14))) / \
                  (np.linalg.norm(pt(78) - pt(308)) + 1e-6)

            # ── Head pose ─────────────────────────────────────────────────
            nose  = pt(1);   chin  = pt(152)
            l_ear = pt(234); r_ear = pt(454)

            ear_vec  = r_ear - l_ear
            tilt_deg = abs(np.degrees(np.arctan2(ear_vec[1], ear_vec[0])))

            ear_mid  = (l_ear + r_ear) / 2
            nod_deg  = np.degrees(
                np.arctan2(chin[1] - nose[1],
                           abs(chin[0] - ear_mid[0]) + 1e-6))
            nod_deg  = max(0.0, nod_deg - 60.0)

            return {
                "eye_state": eye_state,
                "ear":       float(ear_avg),
                "mar":       float(mar),
                "head_tilt": float(tilt_deg),
                "head_nod":  float(nod_deg),
                "face_found":True,
                "landmarks": lm,
            }

        except Exception:
            return default

    def _draw_hud(self, frame, eye_state, score, mouth_open, mar,
                  phone, cigarette, seatbelt_on,
                  tilt, nod, fps, fw, fh) -> np.ndarray:

        font = cv2.FONT_HERSHEY_SIMPLEX
        ov   = frame.copy()
        cv2.rectangle(ov, (0,0), (fw, 175), (0,0,0), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        W = (255, 255, 255)

        def feat(en, det, lbl, off, con, coff):
            if not en:
                return (f"{lbl:<9}: disabled", (80,80,80))
            return ((f"{lbl:<9}: {lbl.upper()}", con) if det
                    else (f"{lbl:<9}: {off}", coff))

        lines = [
            (f"Pipeline : YOLO+MP hybrid", (180,180,255)),
            (f"Eye      : {eye_state.upper():<8} score={score:.2f}", W),
            feat(True, mouth_open, "Mouth",
                 f"closed MAR={mar:.2f}", (0,220,255), W),
            feat(self.detect_phone,    phone,
                 "Phone",     "clear",    (255,0,255), W),
            feat(self.detect_smoking,  cigarette,
                 "Cigarette", "clear",    (200,0,200), W),
            feat(self.detect_seatbelt, seatbelt_on is True,
                 "Seatbelt",
                 "OFF" if seatbelt_on is False else "n/a",
                 (0,255,128),
                 (0,100,255) if seatbelt_on is False else W),
            (f"Head     : tilt={tilt:.1f}  nod={nod:.1f}  FPS={fps:.1f}", W),
        ]
        for i, (text, col) in enumerate(lines):
            cv2.putText(frame, text, (12, 26+i*21), font, 0.5, col, 1)
        return frame

    def _get_fps(self) -> float:
        now = time.time()
        self._fps_buffer.append(now - self._last_time)
        self._last_time = now
        if len(self._fps_buffer) < 2:
            return 0.0
        return 1.0 / (sum(self._fps_buffer) / len(self._fps_buffer))