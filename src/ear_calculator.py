"""
ear_calculator.py
─────────────────
Eye Aspect Ratio (EAR) — geometric drowsiness detection.

Formula:
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)

Where p1–p6 are the 6 landmark points around one eye:
    p1 = left corner
    p2 = upper outer
    p3 = upper inner
    p4 = right corner
    p5 = lower inner
    p6 = lower outer

Typical values:
    Eye fully open  → EAR ≈ 0.25–0.35
    Eye half-closed → EAR ≈ 0.15–0.25
    Eye fully closed→ EAR ≈ 0.0–0.15

Default threshold: EAR < 0.25 → closed
"""

import numpy as np
from scipy.spatial import distance


# Default threshold — tune per person if needed
DEFAULT_EAR_THRESHOLD = 0.25


def eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """
    Compute EAR for one eye.

    Args:
        eye_points: np.array of shape (6, 2) — pixel (x, y) coordinates
                    ordered as [p1, p2, p3, p4, p5, p6]

    Returns:
        EAR value as float
    """
    p1, p2, p3, p4, p5, p6 = eye_points

    # Vertical distances (numerator)
    A = distance.euclidean(p2, p6)
    B = distance.euclidean(p3, p5)

    # Horizontal distance (denominator)
    C = distance.euclidean(p1, p4)

    if C == 0:
        return 0.0

    return (A + B) / (2.0 * C)


def average_ear(left_eye: np.ndarray, right_eye: np.ndarray) -> float:
    """
    Compute average EAR across both eyes.

    Args:
        left_eye:  np.array (6, 2)
        right_eye: np.array (6, 2)

    Returns:
        Average EAR float
    """
    left_ear  = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    return (left_ear + right_ear) / 2.0


def is_eye_closed(ear: float, threshold: float = DEFAULT_EAR_THRESHOLD) -> bool:
    """Returns True if EAR is below the closed threshold."""
    return ear < threshold


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import cv2
    from landmark_extractor import LandmarkExtractor

    cap = cv2.VideoCapture(0)
    extractor = LandmarkExtractor()

    print("Press Q to quit. Watch EAR value — blink to see it drop.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        data = extractor.extract(frame)
        if data:
            ear = average_ear(data["left_eye"], data["right_eye"])
            closed = is_eye_closed(ear)

            color = (0, 0, 255) if closed else (0, 255, 0)
            label = f"EAR: {ear:.3f} | {'CLOSED' if closed else 'OPEN'}"
            cv2.putText(frame, label, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("EAR Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()