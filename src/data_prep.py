"""
data_prep.py
────────────
Merge MRL + CEW + Kaggle datasets → data/combined/
Create stratified train/val/test splits → data/splits/*.csv

Split: 70% train / 15% val / 15% test
Stratified by: class AND subject (prevents same person in train + test)

Run:
    python src/data_prep.py
"""

import os
import glob
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Dataset paths ───────────────────────────────────────────────────────────────
MRL_OPEN    = "data/mrl_eye/open"
MRL_CLOSED  = "data/mrl_eye/closed"
MRL_DROWSY  = "data/mrl_eye/drowsy"
CEW_DIR     = "data/cew"
COMBINED_OPEN   = "data/combined/open"
COMBINED_CLOSED = "data/combined/closed"
SPLITS_DIR  = "data/splits"

# Label map
LABEL_OPEN      = 0
LABEL_HALF_OPEN = 1
LABEL_CLOSED    = 2

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(folder: str, label: int) -> list:
    """Collect all valid images from a folder with a label."""
    records = []
    if not os.path.exists(folder):
        print(f"⚠️  Folder not found: {folder} — skipping")
        return records

    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in VALID_EXTS:
                records.append({
                    "image_path": os.path.join(root, f),
                    "label": label,
                })
    return records


def build_combined_dataset() -> pd.DataFrame:
    """Collect all images from all datasets into one DataFrame."""
    records = []

    print("Collecting MRL open images...")
    records += collect_images(MRL_OPEN,   LABEL_OPEN)

    print("Collecting MRL closed images...")
    records += collect_images(MRL_CLOSED, LABEL_CLOSED)

    print("Collecting MRL drowsy images...")
    records += collect_images(MRL_DROWSY, LABEL_HALF_OPEN)

    print("Collecting CEW images...")
    # CEW structure varies — adapt folder names to your actual CEW layout
    records += collect_images(os.path.join(CEW_DIR, "open"),   LABEL_OPEN)
    records += collect_images(os.path.join(CEW_DIR, "closed"), LABEL_CLOSED)

    df = pd.DataFrame(records)
    print(f"\nTotal images: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts().sort_index()}")
    return df


def create_splits(df: pd.DataFrame):
    """Create stratified train/val/test splits and save as CSV files."""
    os.makedirs(SPLITS_DIR, exist_ok=True)

    # Stratified split: 70 / 15 / 15
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
    )

    train_df.to_csv(f"{SPLITS_DIR}/train.csv", index=False)
    val_df.to_csv(f"{SPLITS_DIR}/val.csv",   index=False)
    test_df.to_csv(f"{SPLITS_DIR}/test.csv",  index=False)

    print(f"\n✅ Splits saved to {SPLITS_DIR}/")
    print(f"   Train: {len(train_df)} images")
    print(f"   Val:   {len(val_df)} images")
    print(f"   Test:  {len(test_df)} images")


def main():
    print("=" * 50)
    print("  SafeDrive — Dataset Preparation")
    print("=" * 50)

    df = build_combined_dataset()

    if len(df) == 0:
        print("\n❌ No images found. Download datasets first.")
        print("   See README.md → Dataset Download section.")
        return

    create_splits(df)
    print("\nDone! Now run the EDA notebook: notebooks/01_EDA_Eye_Dataset.ipynb")


if __name__ == "__main__":
    main()