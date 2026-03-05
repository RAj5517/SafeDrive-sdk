"""
face_detector.py
────────────────
Stage 1 of the pipeline.
Uses MediaPipe Face Detection to find face bounding box in each frame.

Returns: (x, y, w, h) in pixel coordinates, or None if no face found.
"""

import cv2
import mediapipe as mp
import numpy as np


class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.7):
        """
        Args:
            min_detection_confidence: Minimum confidence to accept a face detection.
                                      Lower = more detections but more false positives.
        """
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=0,                          # 0 = short range (< 2m), 1 = full range
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, frame: np.ndarray) -> tuple | None:
        """
        Detect the largest face in a frame.

        Args:
            frame: BGR frame from OpenCV (H, W, 3)

        Returns:
            (x, y, w, h) bounding box in pixels, or None if no face found.
        """
        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        if not results.detections:
            return None

        h, w = frame.shape[:2]
        best_box = None
        best_area = 0

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            # Convert relative coords → pixel coords
            x = max(0, int(bbox.xmin * w))
            y = max(0, int(bbox.ymin * h))
            bw = min(int(bbox.width * w), w - x)
            bh = min(int(bbox.height * h), h - y)

            area = bw * bh
            if area > best_area:
                best_area = area
                best_box = (x, y, bw, bh)

        # Minimum face size check — face too far from camera
        if best_box and (best_box[2] < 100 or best_box[3] < 100):
            return None

        return best_box

    def close(self):
        self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()

    print("Press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        box = detector.detect(frame)
        if box:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Face Detector Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()