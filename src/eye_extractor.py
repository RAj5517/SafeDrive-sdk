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
    """
    Extract and preprocess a single eye ROI from a frame.

    Args:
        frame:      BGR frame from OpenCV (H, W, 3)
        eye_points: np.array (6, 2) — pixel coordinates of eye landmarks

    Returns:
        np.array (64, 64) grayscale float32 normalized [0, 1]
        or None if ROI is too small / out of bounds
    """
    h, w = frame.shape[:2]

    # Bounding box from landmark min/max
    x_min = int(np.min(eye_points[:, 0]))
    x_max = int(np.max(eye_points[:, 0]))
    y_min = int(np.min(eye_points[:, 1]))
    y_max = int(np.max(eye_points[:, 1]))

    # Sanity check — eye too small (face far from camera)
    if (x_max - x_min) < MIN_ROI_SIZE or (y_max - y_min) < MIN_ROI_SIZE:
        return None

    # Add padding
    eye_w = x_max - x_min
    eye_h = y_max - y_min
    pad_x = int(eye_w * PADDING_RATIO)
    pad_y = int(eye_h * PADDING_RATIO)

    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(w, x_max + pad_x)
    y2 = min(h, y_max + pad_y)

    # Crop
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    # Grayscale → resize → normalize
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0

    return normalized


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