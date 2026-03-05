"""
landmark_extractor.py
─────────────────────
Extracts 468 facial landmarks using MediaPipe Face Mesh.
Returns the 6 key points around each eye needed for EAR calculation.

Left eye  indices: [362, 385, 387, 263, 373, 380]
Right eye indices: [33,  160, 158, 133, 153, 144]

Points are ordered: p1=left_corner, p2=upper_outer, p3=upper_inner,
                    p4=right_corner, p5=lower_inner, p6=lower_outer
This order matches the EAR formula exactly.
"""

import cv2
import mediapipe as mp
import numpy as np


# MediaPipe landmark indices for each eye (6 points per eye)
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]


class LandmarkExtractor:
    def __init__(self, refine_landmarks: bool = True):
        """
        Args:
            refine_landmarks: If True, uses iris refinement model for more
                              accurate eyelid landmark positions.
        """
        self.mp_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, frame: np.ndarray) -> dict | None:
        """
        Extract eye landmark points from a frame.

        Args:
            frame: BGR frame from OpenCV (H, W, 3)

        Returns:
            dict with keys:
                'left_eye':  np.array shape (6, 2) — pixel coords
                'right_eye': np.array shape (6, 2) — pixel coords
                'all_landmarks': full 468-point array (normalized 0–1)
            or None if no face detected.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        # Use first (and only) face
        landmarks = results.multi_face_landmarks[0].landmark

        def get_eye_points(indices):
            return np.array([
                [landmarks[i].x * w, landmarks[i].y * h]
                for i in indices
            ], dtype=np.float32)

        return {
            "left_eye":      get_eye_points(LEFT_EYE_IDX),
            "right_eye":     get_eye_points(RIGHT_EYE_IDX),
            "all_landmarks": landmarks,
        }

    def draw_eye_landmarks(self, frame: np.ndarray, eye_points: np.ndarray,
                           color: tuple = (0, 255, 255)) -> np.ndarray:
        """Draw 6 landmark dots around an eye. Useful for debugging."""
        for pt in eye_points:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, color, -1)
        return frame

    def close(self):
        self.face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    extractor = LandmarkExtractor()

    print("Press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        data = extractor.extract(frame)
        if data:
            extractor.draw_eye_landmarks(frame, data["left_eye"],  color=(0, 255, 255))
            extractor.draw_eye_landmarks(frame, data["right_eye"], color=(255, 255, 0))

        cv2.imshow("Landmark Extractor Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()