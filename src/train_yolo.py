"""
train_yolo.py
─────────────
Train YOLOv8-nano on the merged SafeDrive dataset.

Classes:
    0  eye_open       5  phone
    1  eye_half       6  cigarette
    2  eye_closed     7  seatbelt_on
    3  mouth_open     8  seatbelt_off
    4  mouth_closed

Usage:
    python train_yolo.py

Output:
    models/yolo_safedrive.pt      ← final model for SDK
    outputs/yolo_train/           ← training curves, confusion matrix
"""

from pathlib import Path
from datetime import datetime
import shutil


# ── Config ────────────────────────────────────────────────────────────────────
DATASET_YAML  = "data/yolo_merged/dataset.yaml"
MODEL_BASE    = "yolov8n.pt"          # nano — fastest, good enough
OUTPUT_NAME   = "yolo_safedrive"
EPOCHS        = 100
IMG_SIZE      = 640
BATCH         = 16
PATIENCE      = 20                    # early stopping
DEVICE        = "0"                   # GPU — change to "cpu" if no GPU
WORKERS       = 4

SAVE_DIR      = Path("outputs/yolo_train")
FINAL_MODEL   = Path("models/yolo_safedrive.pt")


def main():
    print("\n══════════════════════════════════════════")
    print("  SafeDrive YOLOv8 Training")
    print("══════════════════════════════════════════")
    print(f"  Dataset : {DATASET_YAML}")
    print(f"  Model   : {MODEL_BASE}  (YOLOv8-nano)")
    print(f"  Epochs  : {EPOCHS}  (early stop patience={PATIENCE})")
    print(f"  Device  : {DEVICE}")
    print(f"  Batch   : {BATCH}")
    print("══════════════════════════════════════════\n")

    # ── Check dataset exists ──────────────────────────────────────────────
    if not Path(DATASET_YAML).exists():
        print(f"ERROR: {DATASET_YAML} not found.")
        print("Run: python merge_yolo_datasets.py first")
        return

    # ── Install ultralytics if needed ─────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics...")
        import subprocess
        subprocess.run(["pip", "install", "ultralytics"], check=True)
        from ultralytics import YOLO

    # ── Train ─────────────────────────────────────────────────────────────
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    model = YOLO(MODEL_BASE)

    print("Starting training...\n")
    results = model.train(
        data      = DATASET_YAML,
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = BATCH,
        patience  = PATIENCE,
        device    = DEVICE,
        workers   = WORKERS,
        project   = str(SAVE_DIR),
        name      = OUTPUT_NAME,
        exist_ok  = True,

        # Augmentation — important for driver monitoring
        # (varies lighting, angle, blur to improve real-world performance)
        hsv_h     = 0.02,    # hue shift (lighting color changes)
        hsv_s     = 0.5,     # saturation (bright/dim environments)
        hsv_v     = 0.4,     # value/brightness
        degrees   = 10.0,    # rotation (head tilt)
        translate = 0.1,     # position shift
        scale     = 0.3,     # zoom in/out (distance from camera)
        flipud    = 0.0,     # no vertical flip (driver always upright)
        fliplr    = 0.5,     # horizontal flip (mirror image)
        mosaic    = 0.5,     # mosaic augmentation
        perspective = 0.0005,
        # blur      = 0.1,     # slight blur (drowsy = blurry footage sometimes)
    )

    # ── Copy best model to models/ ────────────────────────────────────────
    best_model = SAVE_DIR / OUTPUT_NAME / "weights" / "best.pt"
    if best_model.exists():
        shutil.copy2(best_model, FINAL_MODEL)
        print(f"\n  Best model saved to: {FINAL_MODEL}")
    else:
        print(f"\n  WARNING: best.pt not found at {best_model}")

    # ── Print results ─────────────────────────────────────────────────────
    print("\n══════════════════════════════════════════")
    print("  TRAINING COMPLETE")
    print("══════════════════════════════════════════")
    try:
        metrics = results.results_dict
        print(f"  mAP50     : {metrics.get('metrics/mAP50(B)',    0):.4f}")
        print(f"  mAP50-95  : {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  Precision : {metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall    : {metrics.get('metrics/recall(B)',    0):.4f}")
    except Exception:
        pass
    print(f"\n  Model     : {FINAL_MODEL}")
    print(f"  Curves    : {SAVE_DIR / OUTPUT_NAME}")
    print(f"\n  NEXT: Add model to HuggingFace")
    print(f"        python -c \"from huggingface_hub import HfApi; "
          f"HfApi().upload_file(path_or_fileobj='models/yolo_safedrive.pt', "
          f"path_in_repo='yolo_safedrive.pt', "
          f"repo_id='raj5517/safedrive-model', repo_type='model')\"")
    print("══════════════════════════════════════════\n")


if __name__ == "__main__":
    main()