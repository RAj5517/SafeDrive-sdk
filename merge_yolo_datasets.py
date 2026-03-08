"""
merge_yolo_datasets.py
──────────────────────
Merges all YOLO datasets into one unified training set.

Sources:
    1. data/yolo_webcam/          ← your webcam data (eyes + mouth)
    2. data/roboflow/phone/       ← Roboflow phone dataset
    3. data/roboflow/seatbelt/    ← Roboflow seatbelt dataset
    4. data/roboflow/cigarette/   ← Roboflow cigarette dataset

Output:
    data/yolo_merged/
        images/train/
        images/val/
        labels/train/
        labels/val/
        dataset.yaml              ← final config for YOLOv8 training

Final class mapping:
    0  eye_open
    1  eye_half
    2  eye_closed
    3  mouth_open
    4  mouth_closed
    5  phone
    6  cigarette
    7  seatbelt_on
    8  seatbelt_off

Usage:
    python merge_yolo_datasets.py

NOTE: Edit ROBOFLOW_CLASS_MAPS below to match the class names
in YOUR downloaded Roboflow datasets — they vary by dataset.
"""

import os
import shutil
import random
from pathlib import Path
import yaml


# ── Output ────────────────────────────────────────────────────────────────────
MERGED_DIR = Path("data/yolo_merged")
VAL_SPLIT  = 0.15    # 15% validation

# ── Final unified class IDs ───────────────────────────────────────────────────
FINAL_CLASSES = {
    "eye_open":      0,
    "eye_half":      1,
    "eye_closed":    2,
    "mouth_open":    3,
    "mouth_closed":  4,
    "phone":         5,
    "cigarette":     6,
    "seatbelt_on":   7,
    "seatbelt_off":  8,
}

# ── Roboflow class name → our unified class name ──────────────────────────────
# Edit these to match the class names in your downloaded datasets.
# Open each data.yaml from Roboflow and check the 'names' list.

ROBOFLOW_CLASS_MAPS = {

    "phone": {
        # Common Roboflow phone dataset class names → our class
        "phone":       "phone",
        "cell-phone":  "phone",
        "cellphone":   "phone",
        "mobile":      "phone",
        "smartphone":  "phone",
        "phone": "phone",
        "object": "phone",
        "phone - v3 2024-02-16 5-38am": "phone",
    },

    "seatbelt": {
        # Common Roboflow seatbelt dataset class names → our class
        "seatbelt":    "seatbelt_on",
        "seatbelt_on": "seatbelt_on",
        "belt":        "seatbelt_on",
        "no-seatbelt": "seatbelt_off",
        "no_seatbelt": "seatbelt_off",
        "no-belt":     "seatbelt_off",
        "without":     "seatbelt_off",
        "seat-belt detection - v5 2023-10-12 10-52pm": "seatbelt_on",
        "seatbelt": "seatbelt_on",
        "belt": "seatbelt_on",
    },

    "cigarette": {
        # Common Roboflow cigarette/smoking dataset class names → our class
        "cigarette":   "cigarette",
        "smoking":     "cigarette",
        "smoke":       "cigarette",
        "cigar":       "cigarette",
        "vape":        "cigarette",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_roboflow_classes(yaml_path: Path) -> dict:
    """Load class id → name mapping from Roboflow data.yaml."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        return names
    return {}


def remap_label(label_path: Path,
                src_id_to_name: dict,
                class_map: dict,
                out_path: Path) -> int:
    """
    Read a YOLO .txt label file, remap class IDs to unified IDs.
    Returns number of valid annotations written.
    """
    lines_out = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            src_id = int(parts[0])
            src_name = src_id_to_name.get(src_id, "").lower()

            # Find our unified class name
            unified_name = class_map.get(src_name)
            if unified_name is None:
                continue   # skip unknown classes

            unified_id = FINAL_CLASSES.get(unified_name)
            if unified_id is None:
                continue

            lines_out.append(
                f"{unified_id} {' '.join(parts[1:])}\n"
            )

    if lines_out:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.writelines(lines_out)
        return len(lines_out)
    return 0


def collect_pairs(images_dir: Path,
                  labels_dir: Path) -> list[tuple[Path, Path]]:
    """Return (image_path, label_path) pairs where both exist."""
    pairs = []
    for img in images_dir.glob("*.[jJ][pP][gG]"):
        lbl = labels_dir / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))
    for img in images_dir.glob("*.png"):
        lbl = labels_dir / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Prepare output dirs ───────────────────────────────────────────────
    for split in ["train", "val"]:
        (MERGED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_pairs  = []   # (img_src, lbl_src, already_remapped)
    stats      = {}

    print("\n══════════════════════════════════════════")
    print("  SafeDrive YOLO Dataset Merger")
    print("══════════════════════════════════════════\n")

    # ── 1. Webcam data (eyes + mouth) ─────────────────────────────────────
    webcam_dir = Path("data/yolo_webcam")
    if webcam_dir.exists():
        pairs = collect_pairs(
            webcam_dir / "images",
            webcam_dir / "labels"
        )
        all_pairs.extend([(img, lbl, True) for img, lbl in pairs])
        print(f"  Webcam data  : {len(pairs):,} frames (eyes + mouth)")
        stats["webcam"] = len(pairs)
    else:
        print("  Webcam data  : NOT FOUND — run collect_yolo_data.py first")

    # ── 2. Roboflow datasets ──────────────────────────────────────────────
    roboflow_base = Path("data/roboflow")

    for dataset_name, class_map in ROBOFLOW_CLASS_MAPS.items():
        dataset_dir = roboflow_base / dataset_name
        if not dataset_dir.exists():
            print(f"  {dataset_name:12s} : NOT FOUND at {dataset_dir}")
            print(f"               → Download from Roboflow and place in {dataset_dir}")
            stats[dataset_name] = 0
            continue

        # Load class names from Roboflow yaml
        yaml_files = list(dataset_dir.glob("*.yaml")) + \
                     list(dataset_dir.glob("**/*.yaml"))
        if not yaml_files:
            print(f"  {dataset_name:12s} : data.yaml not found in {dataset_dir}")
            continue

        src_id_to_name = load_roboflow_classes(yaml_files[0])
        print(f"  {dataset_name:12s} : classes = {src_id_to_name}")

        count = 0
        # Check train/ and valid/ subdirs
        for split_name in ["train", "valid", "val", ""]:
            img_dir = dataset_dir / split_name / "images" if split_name else dataset_dir / "images"
            lbl_dir = dataset_dir / split_name / "labels" if split_name else dataset_dir / "labels"
            if not img_dir.exists():
                continue
            pairs = collect_pairs(img_dir, lbl_dir)
            for img_path, lbl_path in pairs:
                # Remap to temp location
                temp_lbl = MERGED_DIR / f"_temp_{dataset_name}_{img_path.stem}.txt"
                n = remap_label(lbl_path, src_id_to_name, class_map, temp_lbl)
                if n > 0:
                    all_pairs.append((img_path, temp_lbl, True))
                    count += 1

        print(f"  {dataset_name:12s} : {count:,} images loaded")
        stats[dataset_name] = count

    # ── Split into train / val ────────────────────────────────────────────
    random.shuffle(all_pairs)
    n_val   = int(len(all_pairs) * VAL_SPLIT)
    val_set = all_pairs[:n_val]
    trn_set = all_pairs[n_val:]

    print(f"\n  Total pairs  : {len(all_pairs):,}")
    print(f"  Train        : {len(trn_set):,}")
    print(f"  Val          : {len(val_set):,}")

    # ── Copy to merged dirs ───────────────────────────────────────────────
    counts = {"train": 0, "val": 0}

    def copy_pair(img_src, lbl_src, split, idx):
        ext     = img_src.suffix
        name    = f"{split}_{idx:06d}"
        img_dst = MERGED_DIR / "images" / split / f"{name}{ext}"
        lbl_dst = MERGED_DIR / "labels" / split / f"{name}.txt"
        shutil.copy2(img_src, img_dst)
        shutil.copy2(lbl_src, lbl_dst)
        # Clean up temp files
        if "_temp_" in str(lbl_src):
            lbl_src.unlink(missing_ok=True)

    for i, (img, lbl, _) in enumerate(trn_set):
        copy_pair(img, lbl, "train", i)
        counts["train"] += 1

    for i, (img, lbl, _) in enumerate(val_set):
        copy_pair(img, lbl, "val", i)
        counts["val"] += 1

    # ── Write final dataset.yaml ──────────────────────────────────────────
    yaml_data = {
        "path":  str(MERGED_DIR.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(FINAL_CLASSES),
        "names": {v: k for k, v in sorted(FINAL_CLASSES.items(), key=lambda x: x[1])},
    }
    with open(MERGED_DIR / "dataset.yaml", "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    print(f"\n══════════════════════════════════════════")
    print(f"  MERGE COMPLETE")
    print(f"══════════════════════════════════════════")
    print(f"  Train: {counts['train']:,} images")
    print(f"  Val  : {counts['val']:,} images")
    print(f"  YAML : {MERGED_DIR / 'dataset.yaml'}")
    print(f"\n  Classes:")
    for name, cid in sorted(FINAL_CLASSES.items(), key=lambda x: x[1]):
        print(f"    {cid}  {name}")
    print(f"\n  NEXT: python train_yolo.py")
    print(f"══════════════════════════════════════════\n")


if __name__ == "__main__":
    main()