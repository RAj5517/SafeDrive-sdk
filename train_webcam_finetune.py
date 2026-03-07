"""
train_webcam_finetune.py
────────────────────────
Domain adaptation — fine-tune models/mobilenet_best.pth on YOUR webcam data.

WHY this is separate from train_mobilenet.py:
    train_mobilenet.py       → trains from ImageNet pretrained weights
                               uses 152,208 lab dataset images (3 classes)
                               builds the base model from scratch

    train_webcam_finetune.py → starts from YOUR best saved model weights
                               uses 15,000 webcam images (your face, your camera)
                               fixes domain gap between lab data and reality

STRATEGY — 2-phase fine-tuning:
    Phase 1 (epochs 1-5):   freeze backbone → head only        LR=1e-3
    Phase 2 (epochs 6-20):  unfreeze features[7,8,9] + head    LR=5e-5

    Lower LR than original training → avoid overwriting lab knowledge
    Fewer epochs → prevent catastrophic forgetting

Label mapping:
    data/webcam/open/   → label 0  (open)
    data/webcam/half/   → label 1  (half)
    data/webcam/closed/ → label 2  (closed)
    Matches model output exactly — no remapping needed.

Usage:
    python train_webcam_finetune.py

Saves:
    models/mobilenet_webcam.pth          <- use this in realtime_detector.py
    outputs/webcam_finetune_curves.png
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from mobilenet_model import build_mobilenet, CLASS_NAMES

torch.set_float32_matmul_precision("high")

# ── Config ────────────────────────────────────────────────────────────────────
WEBCAM_DIR   = Path("data/webcam")
BASE_MODEL   = "models/mobilenet_best.pth"    # YOUR best weights (97.99%)
SAVE_MODEL   = "models/mobilenet_webcam.pth"  # domain-adapted output
CURVES_SAVE  = "outputs/webcam_finetune_curves.png"

PHASE1_END   = 5       # epochs 1-5:  head only
TOTAL_EPOCHS = 20      # epochs 6-20: last 3 blocks + head

LR_PHASE1    = 1e-3    # head-only — higher LR is fine
LR_PHASE2    = 5e-5    # backbone — very low to preserve lab knowledge

BATCH_SIZE   = 32
NUM_WORKERS  = 0       # 0 = safest on Windows
VAL_SPLIT    = 0.15    # 15% validation
EARLY_STOP   = 6
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_FOLDERS = {0: "open", 1: "half", 2: "closed"}


# ── Dataset ───────────────────────────────────────────────────────────────────

def get_transforms(is_train: bool):
    if is_train:
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.35, contrast_limit=0.35, p=0.8),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.GaussNoise(p=0.3),
            A.Rotate(limit=20, p=0.4),
            A.RandomShadow(p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class WebcamEyeDataset(Dataset):
    """
    Loads 64x64 grayscale PNGs from data/webcam/{open,half,closed}/
    Converts to RGB 224x224 for MobileNetV3 input.
    """
    def __init__(self, samples: list, is_train: bool = False):
        self.samples   = samples
        self.transform = get_transforms(is_train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((64, 64), dtype=np.uint8)

        # Grayscale -> fake RGB (stack same channel 3x)
        rgb = np.stack([img, img, img], axis=-1)
        aug = self.transform(image=rgb)
        tensor = torch.from_numpy(
            aug["image"].transpose(2, 0, 1)   # (3, 224, 224)
        ).float()
        return tensor, label


def load_samples() -> tuple:
    """Scans webcam folders, returns (train_samples, val_samples)."""
    all_samples = []
    for label, folder_name in CLASS_FOLDERS.items():
        folder = WEBCAM_DIR / folder_name
        if not folder.exists():
            print(f"  WARNING: Missing folder: {folder}")
            continue
        images = sorted(folder.glob("*.png"))
        print(f"  {folder_name:8s}  (label {label})  ->  {len(images):,} images")
        all_samples.extend([(p, label) for p in images])

    if not all_samples:
        raise RuntimeError(f"No images found in {WEBCAM_DIR}")

    random.seed(42)
    random.shuffle(all_samples)
    n_val = int(len(all_samples) * VAL_SPLIT)
    return all_samples[n_val:], all_samples[:n_val]


# ── Freeze helpers ────────────────────────────────────────────────────────────

def phase1_setup(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\n  PHASE 1  --  Head only (backbone frozen)")
    print(f"  Trainable: {trainable:,} / {total:,}  ({trainable/total*100:.1f}%)")
    print(f"  Goal: teach head to read webcam feature distribution")


def phase2_setup(model):
    for p in model.parameters():
        p.requires_grad = False
    for layer in [model.features[7],
                  model.features[8],
                  model.features[9],
                  model.classifier]:
        for p in layer.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\n  PHASE 2  --  Last 3 blocks + head")
    print(f"  Trainable: {trainable:,} / {total:,}  ({trainable/total*100:.1f}%)")
    print(f"  LR={LR_PHASE2}  (low -- nudge, don't overwrite lab knowledge)")


# ── Train / Eval ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = correct = total = 0

    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    per_correct = {0: 0, 1: 0, 2: 0}
    per_total   = {0: 0, 1: 0, 2: 0}

    for imgs, labels in tqdm(loader, desc="  Eval ", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out   = model(imgs)
        loss  = criterion(out, labels)
        preds = out.argmax(1)

        total_loss += loss.item() * imgs.size(0)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)

        for cls in [0, 1, 2]:
            mask = labels == cls
            if mask.any():
                per_correct[cls] += (preds[mask] == cls).sum().item()
                per_total[cls]   += mask.sum().item()

    per_class_acc = {
        CLASS_FOLDERS[cls]: (per_correct[cls] / per_total[cls]
                             if per_total[cls] > 0 else 0.0)
        for cls in [0, 1, 2]
    }
    return total_loss / total, correct / total, per_class_acc


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train", color="steelblue")
    axes[0].plot(history["val_loss"],   label="Val",   color="coral")
    axes[0].axvline(x=PHASE1_END, color="gray",
                    linestyle="--", alpha=0.7, label="Phase 1->2")
    axes[0].set_title("Loss -- Webcam Fine-tuning")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_acc"],  label="Train Overall",
                 color="steelblue", lw=2)
    axes[1].plot(history["val_acc"],    label="Val Overall",
                 color="coral",     lw=2)
    axes[1].plot(history["open_acc"],   label="Val Open",
                 color="green",     linestyle="--")
    axes[1].plot(history["half_acc"],   label="Val Half",
                 color="orange",    linestyle="--")
    axes[1].plot(history["closed_acc"], label="Val Closed",
                 color="red",       linestyle="--")
    axes[1].axvline(x=PHASE1_END, color="gray",
                    linestyle="--", alpha=0.7, label="Phase 1->2")
    axes[1].axhline(y=0.95, color="gold", linestyle=":",
                    alpha=0.8, label="95% target")
    axes[1].set_title("Accuracy -- Webcam Fine-tuning")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.3, 1.05)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("MobileNetV3  --  Lab to Webcam Domain Adaptation",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(CURVES_SAVE, dpi=150)
    plt.close()
    print(f"  Curves saved -> {CURVES_SAVE}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  MobileNetV3  --  Webcam Domain Adaptation")
    print(f"  Base weights : {BASE_MODEL}")
    print(f"  Device       : {DEVICE}")
    print(f"  Batch        : {BATCH_SIZE}   Epochs: {TOTAL_EPOCHS}")
    print("=" * 60)

    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    if not os.path.exists(BASE_MODEL):
        print(f"\n  ERROR: Base model not found: {BASE_MODEL}")
        print("  Run train_mobilenet.py first.")
        sys.exit(1)

    # ── Data ──────────────────────────────────────────────────────────────
    print(f"\n  Loading webcam data from {WEBCAM_DIR} ...")
    train_samples, val_samples = load_samples()
    print(f"\n  Train: {len(train_samples):,}   Val: {len(val_samples):,}")

    train_loader = DataLoader(
        WebcamEyeDataset(train_samples, is_train=True),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        WebcamEyeDataset(val_samples, is_train=False),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
    )

    # ── Load YOUR best saved weights ──────────────────────────────────────
    print(f"\n  Loading {BASE_MODEL} ...")
    model = build_mobilenet(pretrained=False).to(DEVICE)
    state = torch.load(BASE_MODEL, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    print(f"  Loaded  (lab accuracy: 97.99%)")
    print(f"  Goal: fix closed-eye recognition for YOUR webcam")

    # Weight closed class 2x higher -- that's our domain gap target
    class_weights = torch.tensor([1.0, 1.0, 2.0], device=DEVICE)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    # ── Training ──────────────────────────────────────────────────────────
    history = {k: [] for k in [
        "train_loss", "val_loss",
        "train_acc",  "val_acc",
        "open_acc",   "half_acc", "closed_acc",
    ]}
    best_val_acc    = 0.0
    best_closed_acc = 0.0
    no_improve      = 0
    current_phase   = 0

    for epoch in range(1, TOTAL_EPOCHS + 1):

        if epoch == 1:
            phase1_setup(model)
            optimizer     = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR_PHASE1)
            current_phase = 1

        elif epoch == PHASE1_END + 1:
            phase2_setup(model)
            optimizer     = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR_PHASE2)
            no_improve    = 0
            current_phase = 2

        print(f"\nEpoch {epoch}/{TOTAL_EPOCHS}  [Phase {current_phase}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion)
        val_loss, val_acc, per_cls = evaluate(
            model, val_loader, criterion)

        open_acc   = per_cls["open"]
        half_acc   = per_cls["half"]
        closed_acc = per_cls["closed"]

        for k, v in [
            ("train_loss", train_loss), ("val_loss",   val_loss),
            ("train_acc",  train_acc),  ("val_acc",    val_acc),
            ("open_acc",   open_acc),   ("half_acc",   half_acc),
            ("closed_acc", closed_acc),
        ]:
            history[k].append(v)

        print(f"  Train  loss: {train_loss:.4f}   acc: {train_acc:.4f}")
        print(f"  Val    loss: {val_loss:.4f}   acc: {val_acc:.4f}")
        print(f"  Val    open: {open_acc:.4f}   "
              f"half: {half_acc:.4f}   closed: {closed_acc:.4f}")

        if closed_acc > best_closed_acc:
            best_closed_acc = closed_acc

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_MODEL)
            print(f"  SAVED -> {SAVE_MODEL}"
                  f"   val={val_acc:.4f}  closed={closed_acc:.4f}")
            no_improve = 0
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{EARLY_STOP})")

        if epoch > PHASE1_END and no_improve >= EARLY_STOP:
            print(f"\n  Early stopping at epoch {epoch}.")
            break

    # ── Summary ───────────────────────────────────────────────────────────
    plot_curves(history)

    print(f"\n{'='*60}")
    print(f"  WEBCAM FINE-TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best val accuracy    : {best_val_acc*100:.2f}%")
    print(f"  Best closed accuracy : {best_closed_acc*100:.2f}%")
    print(f"\n  Domain gap before:  closed ~0%   (misclassified as half)")
    print(f"  Domain gap after :  closed ~{best_closed_acc*100:.0f}%  (webcam-adapted)")
    print(f"\n  Model saved -> {SAVE_MODEL}")
    print(f"\n  To use in detector:")
    print(f"    python app/run_detector.py --model models/mobilenet_webcam.pth")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()