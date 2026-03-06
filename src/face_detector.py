"""
face_detector.py
────────────────
Face detection using OpenCV Haar Cascade.

Replaces old MediaPipe Face Detection (solutions API removed in 0.10.x).
OpenCV Haar Cascade is built into OpenCV — no downloads, no version issues.

Returns largest detected face as (x, y, w, h) bounding box.
"""

import cv2
import numpy as np


class FaceDetector:
    """
    Detects faces in frames using OpenCV Haar Cascade.

    Args:
        min_face_size:  Minimum face size in pixels (filters out tiny detections)
        scale_factor:   How much image size is reduced at each scale (1.1 = 10%)
        min_neighbors:  How many neighbors each detection needs (higher = stricter)
    """

    def __init__(self,
                 min_face_size: int = 80,
                 scale_factor: float = 1.1,
                 min_neighbors: int = 5):

        self.min_face_size  = min_face_size
        self.scale_factor   = scale_factor
        self.min_neighbors  = min_neighbors

        # Load Haar cascade — built into OpenCV, no download needed
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)

        if self.detector.empty():
            raise RuntimeError(
                "Failed to load haarcascade_frontalface_default.xml. "
                "Check your OpenCV installation."
            )

    def detect(self, frame: np.ndarray) -> tuple | None:
        """
        Detect the largest face in frame.

        Args:
            frame: BGR image (np.ndarray)

        Returns:
            (x, y, w, h) of largest face, or None if no face found
        """
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Equalize histogram for better detection in varying lighting
        gray  = cv2.equalizeHist(gray)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_face_size, self.min_face_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            return None

        # Return largest face by area
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return (int(x), int(y), int(w), int(h))

    def detect_all(self, frame: np.ndarray) -> list:
        """
        Detect all faces in frame.

        Returns:
            List of (x, y, w, h) tuples, sorted by area descending
        """
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_face_size, self.min_face_size),
        )

        if len(faces) == 0:
            return []

        return sorted(
            [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces],
            key=lambda f: f[2] * f[3],
            reverse=True
        )

    def close(self):
        """No resources to release for Haar cascade."""
        pass


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    detector = FaceDetector()
    cap      = cv2.VideoCapture(0)

    print("Face detector test — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        box = detector.detect(frame)

        if box:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face detected",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

        cv2.imshow("Face Detector Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()