"""
eye_extractor.py
────────────────
Extracts cropped eye region (ROI) from a frame using facial landmark points.
Output: 64×64 grayscale image, normalized to [0, 1] — ready for CNN input.

Steps:
    1. Compute bounding box from 6 eye landmark points
    2. Add 20% padding around box (context helps CNN)
    3. Clip to image bounds
    4. Crop from frame
    5. Resize to 64×64
    6. Convert to grayscale
    7. Normalize to [0, 1]
"""

import cv2
import numpy as np


TARGET_SIZE = (64, 64)
PADDING_RATIO = 0.20       # 20% padding around eye bounding box
MIN_ROI_SIZE = 10          # Skip if eye ROI smaller than this (face too far)


def extract_eye_roi(frame: np.ndarray, eye_points: np.ndarray) -> np.ndarray | None:
    h, w = frame.shape[:2]

    # Use eye CENTER + fixed crop size
    # Much more reliable than tight bounding box (eye height is only 4-8px)
    cx = float(np.mean(eye_points[:, 0]))
    cy = float(np.mean(eye_points[:, 1]))

    # Eye width from landmarks → scale up for context
    eye_w = float(np.max(eye_points[:, 0]) - np.min(eye_points[:, 0]))

    # If landmarks are garbage (face too far), eye_w will be tiny
    if eye_w < 8:
        return None

    # Crop box = 2.5× eye width, centered on eye
    half = int(eye_w * 1.25)
    half = max(half, 20)   # minimum 40×40 crop regardless

    x1 = max(0, int(cx) - half)
    x2 = min(w, int(cx) + half)
    y1 = max(0, int(cy) - half)
    y2 = min(h, int(cy) + half)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def extract_both_eyes(frame: np.ndarray,
                      left_points: np.ndarray,
                      right_points: np.ndarray) -> dict:
    """
    Extract both eyes at once.

    Returns:
        dict with 'left' and 'right' keys, each holding np.array (64,64) or None
    """
    return {
        "left":  extract_eye_roi(frame, left_points),
        "right": extract_eye_roi(frame, right_points),
    }


def roi_to_tensor(roi: np.ndarray):
    """
    Convert (64, 64) float32 numpy array → PyTorch tensor (1, 64, 64).
    Import torch only when needed to keep this module lightweight.
    """
    import torch
    return torch.from_numpy(roi).unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from landmark_extractor import LandmarkExtractor

    cap = cv2.VideoCapture(0)
    extractor = LandmarkExtractor()

    print("Press Q to quit. Watch cropped eye ROI windows.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        data = extractor.extract(frame)
        if data:
            eyes = extract_both_eyes(frame, data["left_eye"], data["right_eye"])

            if eyes["left"] is not None:
                # Scale back to 0–255 for display
                left_display = (eyes["left"] * 255).astype(np.uint8)
                left_display = cv2.resize(left_display, (128, 128))
                cv2.imshow("Left Eye ROI", left_display)

            if eyes["right"] is not None:
                right_display = (eyes["right"] * 255).astype(np.uint8)
                right_display = cv2.resize(right_display, (128, 128))
                cv2.imshow("Right Eye ROI", right_display)

        cv2.imshow("Eye Extractor Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()