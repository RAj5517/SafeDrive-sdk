"""
collect_yolo_data.py
────────────────────
Records webcam frames and AUTO-LABELS them using MediaPipe.
Outputs full YOLO-format dataset (images + .txt label files).

YOLO Classes:
    0  eye_open
    1  eye_half
    2  eye_closed
    3  mouth_open    (yawning)
    4  mouth_closed

Output structure:
    data/yolo_webcam/
        images/frame_000001.jpg
        labels/frame_000001.txt
        dataset.yaml
        session_log.txt

YOLO label format (per line in .txt):
    class x_center y_center width height  (normalized 0.0-1.0)

Controls:
    SPACE → pause / resume
    Q     → quit and save
"""

import cv2
import numpy as np
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

# ── MediaPipe landmark indices ────────────────────────────────────────────────
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

MOUTH_IDX_LIST = [61, 37, 267, 291, 84, 314]
# layout:  left, top_left, top_right, right, bottom_left, bottom_right

# ── ✅ FIXED THRESHOLDS ───────────────────────────────────────────────────────

# EAR — widened half range significantly
EAR_OPEN_THRESH   = 0.25   # above this = open     (was 0.21 — too close to half)
EAR_CLOSED_THRESH = 0.18   # below this = closed   (was 0.20 — too close to half)
# HALF = 0.18 → 0.25  (wide range, easy to hit by squinting slightly)

# MAR — raised significantly so normal mouth = closed
MAR_OPEN_THRESH   = 0.65   # above this = yawning  (was 0.45 — way too low)
# CLOSED = below 0.65 (normal talking/relaxed mouth stays closed)

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_FRAMES  = 3000
SAVE_EVERY_N   = 3
OUTPUT_DIR     = Path("data/yolo_webcam")
EYE_PAD        = 0.4
MOUTH_PAD      = 0.3

FONT   = cv2.FONT_HERSHEY_SIMPLEX
GREEN  = (0, 255,   0)
RED    = (0,   0, 255)
ORANGE = (0, 140, 255)
YELLOW = (0, 220, 255)
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
GRAY   = (160, 160, 160)

CLASS_NAMES  = {0:"eye_open", 1:"eye_half", 2:"eye_closed",
                3:"mouth_open", 4:"mouth_closed"}
CLASS_COLORS = {0:GREEN, 1:ORANGE, 2:RED, 3:YELLOW, 4:WHITE}


# ── Geometry helpers ──────────────────────────────────────────────────────────

def compute_ear(pts: np.ndarray) -> float:
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def compute_mar(pts: np.ndarray) -> float:
    """Mouth Aspect Ratio — high = open/yawning, low = closed."""
    A = np.linalg.norm(pts[1] - pts[4])   # top_left  - bottom_left
    B = np.linalg.norm(pts[2] - pts[5])   # top_right - bottom_right
    C = np.linalg.norm(pts[0] - pts[3])   # left      - right
    return (A + B) / (2.0 * C + 1e-6)


def to_yolo_bbox(pts: np.ndarray, pad: float, fw: int, fh: int) -> tuple:
    """Landmark points → YOLO normalized bbox (xc, yc, w, h)."""
    x_min, y_min = np.min(pts[:, 0]), np.min(pts[:, 1])
    x_max, y_max = np.max(pts[:, 0]), np.max(pts[:, 1])
    w = x_max - x_min
    h = y_max - y_min
    x_min = max(0, x_min - w * pad)
    x_max = min(fw, x_max + w * pad)
    y_min = max(0, y_min - h * pad)
    y_max = min(fh, y_max + h * pad)
    return ((x_min+x_max)/2/fw, (y_min+y_max)/2/fh,
            (x_max-x_min)/fw,  (y_max-y_min)/fh)


def draw_box(frame, class_id, bbox, fw, fh, val=None):
    xc, yc, bw, bh = bbox
    x1 = int((xc - bw/2) * fw);  y1 = int((yc - bh/2) * fh)
    x2 = int((xc + bw/2) * fw);  y2 = int((yc + bh/2) * fh)
    color = CLASS_COLORS.get(class_id, WHITE)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = CLASS_NAMES.get(class_id, str(class_id))
    if val is not None:
        label += f" {val:.3f}"
    cv2.putText(frame, label, (x1, y1 - 5), FONT, 0.45, color, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    (OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels").mkdir(parents=True, exist_ok=True)

    existing = len(list((OUTPUT_DIR / "images").glob("*.jpg")))
    print(f"Existing frames: {existing}")

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    task_path = "models/face_landmarker.task"
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=task_path),
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    print("MediaPipe initialized ✅")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_idx     = 0
    saved_count   = existing
    paused        = False
    session_start = time.time()
    counts        = {i: 0 for i in range(5)}

    print(f"\nTarget: {TARGET_FRAMES} frames")
    print("Controls: SPACE=pause  Q=quit\n")
    print("TIPS:")
    print("  Eyes: blink slowly, squint slightly for HALF")
    print("  Mouth: open VERY wide / yawn for mouth_open (threshold raised)")
    print("  Move head slowly in all directions\n")

    while saved_count < TARGET_FRAMES + existing:
        ret, frame = cap.read()
        if not ret:
            break

        frame      = cv2.flip(frame, 1)
        frame_idx += 1
        fh, fw     = frame.shape[:2]

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)

        annotations = []
        ear = mar = 0.0

        if result.face_landmarks:
            lm = result.face_landmarks[0]

            def get_pts(indices):
                return np.array(
                    [[lm[i].x * fw, lm[i].y * fh] for i in indices],
                    dtype=np.float32)

            # ── Eyes ──────────────────────────────────────────────────────
            left_eye  = get_pts(LEFT_EYE_IDX)
            right_eye = get_pts(RIGHT_EYE_IDX)
            left_ear  = compute_ear(left_eye)
            right_ear = compute_ear(right_eye)
            ear       = (left_ear + right_ear) / 2.0

            if ear >= EAR_OPEN_THRESH:
                eye_class = 0   # open
            elif ear <= EAR_CLOSED_THRESH:
                eye_class = 2   # closed
            else:
                eye_class = 1   # half  ← now 0.18-0.25, much easier to hit

            annotations.append((eye_class, to_yolo_bbox(left_eye,  EYE_PAD, fw, fh)))
            annotations.append((eye_class, to_yolo_bbox(right_eye, EYE_PAD, fw, fh)))
            draw_box(frame, eye_class, to_yolo_bbox(left_eye,  EYE_PAD, fw, fh), fw, fh, left_ear)
            draw_box(frame, eye_class, to_yolo_bbox(right_eye, EYE_PAD, fw, fh), fw, fh, right_ear)

            # ── Mouth ─────────────────────────────────────────────────────
            mouth_pts   = get_pts(MOUTH_IDX_LIST)
            mar         = compute_mar(mouth_pts)
            mouth_class = 3 if mar >= MAR_OPEN_THRESH else 4  # ← raised to 0.65

            annotations.append((mouth_class, to_yolo_bbox(mouth_pts, MOUTH_PAD, fw, fh)))
            draw_box(frame, mouth_class, to_yolo_bbox(mouth_pts, MOUTH_PAD, fw, fh), fw, fh, mar)

        # ── Save ──────────────────────────────────────────────────────────
        if not paused and annotations and frame_idx % SAVE_EVERY_N == 0:
            name = f"frame_{saved_count:06d}"
            cv2.imwrite(str(OUTPUT_DIR / "images" / f"{name}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            with open(OUTPUT_DIR / "labels" / f"{name}.txt", "w") as f:
                for cid, (xc, yc, bw, bh) in annotations:
                    f.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                    counts[cid] += 1
            saved_count += 1

        # ── HUD ───────────────────────────────────────────────────────────
        progress = (saved_count - existing) / TARGET_FRAMES * 100
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (fw, 140), BLACK, -1)
        cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)

        status = "PAUSED" if paused else "RECORDING"
        col    = GRAY if paused else GREEN
        cv2.putText(frame,
                    f"{status}  {saved_count-existing}/{TARGET_FRAMES}  ({progress:.0f}%)",
                    (15, 32), FONT, 0.85, col, 2)

        bar_w = fw - 30
        cv2.rectangle(frame, (15, 42), (15+bar_w, 58), GRAY, 1)
        fill = int(bar_w * progress / 100)
        if fill > 0:
            cv2.rectangle(frame, (15, 42), (15+fill, 58), GREEN, -1)

        cv2.putText(frame,
                    f"eye_open:{counts[0]}  half:{counts[1]}  closed:{counts[2]}  "
                    f"mouth_open:{counts[3]}  mouth_closed:{counts[4]}",
                    (15, 78), FONT, 0.45, WHITE, 1)

        # Live EAR/MAR with thresholds shown
        ear_col = GREEN if ear >= EAR_OPEN_THRESH else (RED if ear <= EAR_CLOSED_THRESH else ORANGE)
        mar_col = YELLOW if mar >= MAR_OPEN_THRESH else WHITE
        cv2.putText(frame,
                    f"EAR: {ear:.3f}  (open>{EAR_OPEN_THRESH} half={EAR_CLOSED_THRESH}-{EAR_OPEN_THRESH} closed<{EAR_CLOSED_THRESH})",
                    (15, 100), FONT, 0.43, ear_col, 1)
        cv2.putText(frame,
                    f"MAR: {mar:.3f}  (open>{MAR_OPEN_THRESH} — open mouth WIDE to trigger)",
                    (15, 118), FONT, 0.43, mar_col, 1)

        cv2.putText(frame,
                    "SPACE=pause  Q=quit  |  Squint for HALF eyes  |  Open mouth VERY wide for mouth_open",
                    (15, fh - 12), FONT, 0.38, GRAY, 1)

        cv2.imshow("SafeDrive — YOLO Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

    elapsed     = time.time() - session_start
    total_saved = saved_count - existing

    yaml_content = f"""# SafeDrive YOLO Dataset
# Generated: {datetime.now().isoformat()}

path: {OUTPUT_DIR.resolve()}
train: images
val:   images

nc: 5
names:
  0: eye_open
  1: eye_half
  2: eye_closed
  3: mouth_open
  4: mouth_closed
"""
    with open(OUTPUT_DIR / "dataset.yaml", "w") as f:
        f.write(yaml_content)

    summary = f"""
═══════════════════════════════════════════════════
  YOLO DATA COLLECTION SUMMARY
═══════════════════════════════════════════════════
  Session duration : {elapsed/60:.1f} min
  Frames saved     : {total_saved:,}  (total: {saved_count:,})

  Label distribution:
    eye_open     : {counts[0]:,}
    eye_half     : {counts[1]:,}
    eye_closed   : {counts[2]:,}
    mouth_open   : {counts[3]:,}
    mouth_closed : {counts[4]:,}

  Output : {OUTPUT_DIR.resolve()}
═══════════════════════════════════════════════════
  NEXT: python merge_yolo_datasets.py
═══════════════════════════════════════════════════
"""
    print(summary)
    with open(OUTPUT_DIR / "session_log.txt", "a") as f:
        f.write(summary)


if __name__ == "__main__":
    main()