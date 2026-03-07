"""
landmark_extractor.py
─────────────────────
Eye landmark extraction using MediaPipe Face Landmarker (Tasks API).
Compatible with mediapipe 0.10.x+

Falls back to OpenCV Haar if MediaPipe model file not found.

Eye landmark indices (MediaPipe 468-point Face Mesh):
    Left eye:  [362, 385, 387, 263, 373, 380]
    Right eye: [33,  160, 158, 133, 153, 144]

These 6 points per eye form the EAR formula layout:
        p2──p3
       /      \
    p1          p4
       \      /
        p6──p5
"""

import cv2
import numpy as np
import os

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

# Default path — used when called directly from src/ (not via SDK)
DEFAULT_TASK_PATH = "models/face_landmarker.task"


class LandmarkExtractor:
    """
    Extracts 6-point eye landmarks from a face image.

    Tries MediaPipe Tasks API first.
    Falls back to OpenCV Haar if .task model file not present.

    Usage (direct):
        extractor = LandmarkExtractor()

    Usage (SDK — model auto-downloaded from HuggingFace):
        extractor = LandmarkExtractor(task_path="/path/from/model_manager")

    Returns:
        {"left_eye": np.array(6,2), "right_eye": np.array(6,2)}
        or None if no face detected
    """

    def __init__(self, task_path: str = None):
        """
        Args:
            task_path: path to face_landmarker.task file.
                       If None, tries DEFAULT_TASK_PATH.
                       SDK passes the HuggingFace cached path here.
        """
        self.task_path = task_path or DEFAULT_TASK_PATH
        self.mode      = None
        self._init()

    def _init(self):
        if os.path.exists(self.task_path):
            self._init_mediapipe()
        else:
            print(f"  face_landmarker.task not found at {self.task_path}")
            print(f"  Using OpenCV Haar fallback for landmarks")
            self._init_haar()

    def _init_mediapipe(self):
        try:
            base_options = mp_python.BaseOptions(
                model_asset_path=self.task_path      # ← uses resolved path
            )
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self.mode = "mediapipe"
            print("  MediaPipe Face Landmarker initialized ✅")
        except Exception as e:
            print(f"  MediaPipe init failed: {e}")
            self._init_haar()

    def _init_haar(self):
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.mode = "haar"
        print("  OpenCV Haar landmark extractor ready ✅")

    def extract(self, frame: np.ndarray) -> dict | None:
        """
        Extract eye landmarks from frame.

        Returns:
            {"left_eye": np.array(6,2), "right_eye": np.array(6,2)}
            or None if no face/eyes detected
        """
        if self.mode == "mediapipe":
            return self._extract_mediapipe(frame)
        else:
            return self._extract_haar(frame)

    def _extract_mediapipe(self, frame: np.ndarray) -> dict | None:
        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect(mp_img)
        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]

        def get_pts(indices):
            return np.array(
                [[lm[i].x * w, lm[i].y * h] for i in indices],
                dtype=np.float32
            )

        return {
            "left_eye":  get_pts(LEFT_EYE_IDX),
            "right_eye": get_pts(RIGHT_EYE_IDX),
        }

    def _extract_haar(self, frame: np.ndarray) -> dict | None:
        """
        Approximate 6-point landmarks from Haar eye bounding boxes.
        Less accurate than MediaPipe but works without the .task file.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            return None

        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        face_gray = gray[fy:fy+fh, fx:fx+fw]

        eyes = self.eye_cascade.detectMultiScale(
            face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
        )
        if len(eyes) < 2:
            return None

        eyes = sorted(eyes[:2], key=lambda e: e[0])

        def bbox_to_6pts(ex, ey, ew, eh):
            cx = fx + ex + ew // 2
            cy = fy + ey + eh // 2
            hw = ew // 2
            hh = eh // 2
            return np.array([
                [cx - hw,      cy      ],
                [cx - hw // 2, cy - hh ],
                [cx + hw // 2, cy - hh ],
                [cx + hw,      cy      ],
                [cx + hw // 2, cy + hh ],
                [cx - hw // 2, cy + hh ],
            ], dtype=np.float32)

        return {
            "left_eye":  bbox_to_6pts(*eyes[0]),
            "right_eye": bbox_to_6pts(*eyes[1]),
        }

    def draw_eye_landmarks(self, frame: np.ndarray, data: dict) -> np.ndarray:
        """Draw eye landmark dots on frame for debugging."""
        for pt in data["left_eye"]:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)
        for pt in data["right_eye"]:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (255, 255, 0), -1)
        return frame

    def close(self):
        if self.mode == "mediapipe" and hasattr(self, "landmarker"):
            self.landmarker.close()


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    extractor = LandmarkExtractor()
    cap = cv2.VideoCapture(0)
    print("Landmark extractor test — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        data = extractor.extract(frame)
        if data:
            frame = extractor.draw_eye_landmarks(frame, data)
            cv2.putText(frame, "Landmarks detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

        cv2.imshow("Landmark Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()