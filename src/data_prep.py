"""
data_prep.py
────────────
Merges all 4 downloaded datasets into data/splits/*.csv
Creates stratified train/val/test splits (70/15/15)

Uses OpenCV Haar Cascade for eye detection from face images.
No MediaPipe dependency — works with all versions.

Run:
    python src/data_prep.py
"""

import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Labels ─────────────────────────────────────────────────────────────────────
LABEL_OPEN      = 0
LABEL_HALF_OPEN = 1
LABEL_CLOSED    = 2
LABEL_NAMES     = {0: "Open", 1: "Half-Open", 2: "Closed"}

# ── Dataset paths ───────────────────────────────────────────────────────────────
DD_OPEN   = "data/drowsiness-detection/open_eye"
DD_CLOSED = "data/drowsiness-detection/closed_eye"

MRL_SMALL_OPEN   = "data/mrl-dataset/train/Open_Eyes"
MRL_SMALL_CLOSED = "data/mrl-dataset/train/Closed_Eyes"

MRL_FULL_TRAIN_AWAKE  = "data/mrl-eye-full/data/train/awake"
MRL_FULL_TRAIN_SLEEPY = "data/mrl-eye-full/data/train/sleepy"
MRL_FULL_VAL_AWAKE    = "data/mrl-eye-full/data/val/awake"
MRL_FULL_VAL_SLEEPY   = "data/mrl-eye-full/data/val/sleepy"
MRL_FULL_TEST_AWAKE   = "data/mrl-eye-full/data/test/awake"
MRL_FULL_TEST_SLEEPY  = "data/mrl-eye-full/data/test/sleepy"

FACE_ACTIVE  = "data/drowsiness-prediction-dataset/0 FaceImages/Active Subjects"
FACE_FATIGUE = "data/drowsiness-prediction-dataset/0 FaceImages/Fatigue Subjects"

EXTRACTED_ACTIVE  = "data/combined/extracted_active"
EXTRACTED_FATIGUE = "data/combined/extracted_fatigue"
SPLITS_DIR        = "data/splits"

VALID_EXTS  = {".jpg", ".jpeg", ".png", ".bmp"}
TARGET_SIZE = (64, 64)


# ── OpenCV Haar Cascade setup ───────────────────────────────────────────────────
# Built into OpenCV — no download needed
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


# ── Helper functions ────────────────────────────────────────────────────────────

def get_all_images(folder: str) -> list:
    paths = []
    if not os.path.exists(folder):
        print(f"  ⚠️  Not found: {folder} — skipping")
        return paths
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in VALID_EXTS:
                paths.append(os.path.join(root, f))
    return paths


def collect_eye_images(folder: str, label: int, tag: str = "") -> list:
    """Collect already-cropped eye images — no processing needed."""
    paths   = get_all_images(folder)
    records = [{"image_path": p, "label": label} for p in paths]
    print(f"  {len(records):>7,}  {LABEL_NAMES[label]:<10}  ← {tag or folder}")
    return records


def extract_eyes_haar(img_path: str) -> list:
    """
    Extract eye ROIs from a face image using OpenCV Haar Cascade.
    Returns list of (64,64) grayscale numpy arrays.

    Strategy:
    1. Detect face in image
    2. Within face region, detect eyes
    3. Crop, resize, return eye ROIs
    Fallback: if no face found, run eye detector on full image
    """
    img = cv2.imread(img_path)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Try to find face first
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    search_regions = []

    if len(faces) > 0:
        # Use the largest face found
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        search_regions.append((gray[y:y+h, x:x+w], x, y))
    else:
        # No face found — search full image
        search_regions.append((gray, 0, 0))

    eyes_found = []

    for region, offset_x, offset_y in search_regions:
        eyes = EYE_CASCADE.detectMultiScale(
            region, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
        )

        for (ex, ey, ew, eh) in eyes[:2]:  # max 2 eyes
            # Add padding
            pad = int(min(ew, eh) * 0.15)
            h_reg, w_reg = region.shape

            x1 = max(0, ex - pad)
            y1 = max(0, ey - pad)
            x2 = min(w_reg, ex + ew + pad)
            y2 = min(h_reg, ey + eh + pad)

            eye_crop = region[y1:y2, x1:x2]
            if eye_crop.size == 0:
                continue

            resized = cv2.resize(eye_crop, TARGET_SIZE)
            eyes_found.append(resized)

    return eyes_found


def collect_face_images_extract_eyes(folder: str,
                                     label: int,
                                     save_dir: str,
                                     tag: str = "") -> list:
    """Extract eye ROIs from face images, save to save_dir."""
    os.makedirs(save_dir, exist_ok=True)
    records = []
    paths   = get_all_images(folder)
    skipped = 0

    for p in tqdm(paths, desc=f"  Extracting {tag}", ncols=70):
        eyes = extract_eyes_haar(p)

        if not eyes:
            skipped += 1
            continue

        base = os.path.splitext(os.path.basename(p))[0]
        for eye_idx, eye_arr in enumerate(eyes):
            save_path = os.path.join(save_dir, f"{base}_eye{eye_idx}.jpg")
            cv2.imwrite(save_path, eye_arr)
            records.append({"image_path": save_path, "label": label})

    print(f"  {len(records):>7,}  {LABEL_NAMES[label]:<10}  ← {tag} ({skipped} skipped)")
    return records


def create_splits(df: pd.DataFrame):
    """Stratified 70/15/15 split → train.csv, val.csv, test.csv"""
    os.makedirs(SPLITS_DIR, exist_ok=True)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
    )

    train_df.to_csv(f"{SPLITS_DIR}/train.csv", index=False)
    val_df.to_csv(f"{SPLITS_DIR}/val.csv",     index=False)
    test_df.to_csv(f"{SPLITS_DIR}/test.csv",   index=False)

    print(f"\n  {'Split':<8} {'Total':>8}  {'Open':>8}  {'Half':>8}  {'Closed':>8}")
    print(f"  {'─'*52}")
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        counts = split_df["label"].value_counts().sort_index()
        o = counts.get(0, 0)
        h = counts.get(1, 0)
        c = counts.get(2, 0)
        print(f"  {name:<8} {len(split_df):>8,}  {o:>8,}  {h:>8,}  {c:>8,}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SafeDrive — Dataset Preparation")
    print("=" * 60)

    all_records = []

    # ── Step 1: Direct eye image datasets ─────────────────────────────────────
    print("\n📂 Step 1 — Collecting eye images directly")

    all_records += collect_eye_images(DD_OPEN,   LABEL_OPEN,   "drowsiness-detection/open_eye")
    all_records += collect_eye_images(DD_CLOSED, LABEL_CLOSED, "drowsiness-detection/closed_eye")
    all_records += collect_eye_images(MRL_SMALL_OPEN,   LABEL_OPEN,   "mrl-small/Open_Eyes")
    all_records += collect_eye_images(MRL_SMALL_CLOSED, LABEL_CLOSED, "mrl-small/Closed_Eyes")
    all_records += collect_eye_images(MRL_FULL_TRAIN_AWAKE,  LABEL_OPEN,   "mrl-full/train/awake")
    all_records += collect_eye_images(MRL_FULL_TRAIN_SLEEPY, LABEL_CLOSED, "mrl-full/train/sleepy")
    all_records += collect_eye_images(MRL_FULL_VAL_AWAKE,    LABEL_OPEN,   "mrl-full/val/awake")
    all_records += collect_eye_images(MRL_FULL_VAL_SLEEPY,   LABEL_CLOSED, "mrl-full/val/sleepy")
    all_records += collect_eye_images(MRL_FULL_TEST_AWAKE,   LABEL_OPEN,   "mrl-full/test/awake")
    all_records += collect_eye_images(MRL_FULL_TEST_SLEEPY,  LABEL_CLOSED, "mrl-full/test/sleepy")

    # ── Step 2: Extract eyes from face images ─────────────────────────────────
    print("\n📂 Step 2 — Extracting eyes from face images (2–4 min)")

    all_records += collect_face_images_extract_eyes(
        FACE_ACTIVE,  LABEL_OPEN,      EXTRACTED_ACTIVE,  "Active Subjects"
    )
    all_records += collect_face_images_extract_eyes(
        FACE_FATIGUE, LABEL_HALF_OPEN, EXTRACTED_FATIGUE, "Fatigue Subjects"
    )

    # ── Step 3: Summary + splits ───────────────────────────────────────────────
    df = pd.DataFrame(all_records)

    print(f"\n{'='*60}")
    print(f"  TOTAL IMAGES: {len(df):,}")
    print(f"\n  Class breakdown:")
    for label, count in df["label"].value_counts().sort_index().items():
        pct = count / len(df) * 100
        print(f"  Label {label} {LABEL_NAMES[label]:<10}: {count:>8,}  ({pct:.1f}%)")

    if len(df) == 0:
        print("\n❌ No images found. Check paths.")
        return

    print(f"\n📂 Step 3 — Creating splits (70/15/15)")
    create_splits(df)

    print(f"\n{'='*60}")
    print(f"  ✅ Done! Next: python src/train_eye_state.py")


if __name__ == "__main__":
    main()