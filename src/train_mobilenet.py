"""
train_mobilenet.py
──────────────────
2-Phase fine-tuning of MobileNetV3-Small for eye state classification.

WHY 2 phases not 3:
    Early backbone layers (features[0-6]) learned universal features
    from ImageNet — edge detection, gradient patterns, textures.
    These are directly useful for eye images too.
    Unfreezing them risks catastrophic forgetting with minimal gain.

Phase 1 (epochs 1–3):    freeze ALL backbone → train head only    LR=3e-4
Phase 2 (epochs 4–30):   unfreeze last 3 blocks only              LR=1e-4

Frozen forever:  features[0] to features[6]  (early+mid layers)
Trainable Ph2:   features[7], [8], [9] + classifier head

Usage:
    python src/train_mobilenet.py

Saves:
    models/mobilenet_best.pth
    outputs/training_curves_mobilenet.png
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import albumentations as A

sys.path.insert(0, os.path.dirname(__file__))
from mobilenet_model import build_mobilenet, CLASS_NAMES

torch.set_float32_matmul_precision('high')

# ── Config ──────────────────────────────────────────────────────────────────────
SPLITS_DIR   = "data/splits"
MODEL_SAVE   = "models/mobilenet_best.pth"
CURVES_SAVE  = "outputs/training_curves_mobilenet.png"

PHASE1_END   = 3      # epochs 1–3:  head only
TOTAL_EPOCHS = 30     # epochs 4–30: last 3 blocks + head

LR_PHASE1    = 3e-4
LR_PHASE2    = 1e-4

BATCH_SIZE   = 32     # 224×224 inputs — keep batch small for 4GB VRAM
NUM_WORKERS  = 4
EARLY_STOP   = 8
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CUSTOM_CNN_BASELINE = 0.9674   # our custom CNN test accuracy to beat


# ── Dataset ──────────────────────────────────────────────────────────────────────

def get_transforms(is_train: bool):
    if is_train:
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.7),
            A.GaussianBlur(blur_limit=(3, 3), p=0.2),
            A.GaussNoise(p=0.2),
            A.Rotate(limit=15, p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class EyeDatasetMobileNet(Dataset):
    """
    Loads grayscale eye images → RGB 224×224 with ImageNet normalization.
    Grayscale channel is replicated 3 times to make RGB.
    This is correct — MobileNetV3 processes each channel independently
    so identical channels just means it sees the same grayscale info
    through all 3 color pathways.
    """
    def __init__(self, csv_path: str, is_train: bool = False):
        self.df        = pd.read_csv(csv_path)
        self.transform = get_transforms(is_train)
        print(f"  Loaded {len(self.df):,} from {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row["label"])

        img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((64, 64), dtype=np.uint8)

        # Grayscale → fake RGB (stack same channel 3 times)
        rgb = np.stack([img, img, img], axis=-1)

        aug    = self.transform(image=rgb)
        img_t  = aug["image"]                           # (224,224,3)
        tensor = torch.from_numpy(
            img_t.transpose(2, 0, 1)                    # (3,224,224)
        ).float()

        return tensor, label


def get_class_weights() -> torch.Tensor:
    df      = pd.read_csv(f"{SPLITS_DIR}/train.csv")
    counts  = df["label"].value_counts().sort_index()
    weights = len(df) / (len(counts) * counts.values)
    return torch.tensor(weights, dtype=torch.float32)


# ── Freeze / Unfreeze ────────────────────────────────────────────────────────────

def phase1_setup(model):
    """Freeze everything except classifier head."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\n  PHASE 1 — Head only")
    print(f"  Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    print(f"  Frozen: features[0-9] backbone")
    print(f"  Training: classifier head (our Linear 1024→3)")


def phase2_setup(model):
    """
    Freeze early+mid backbone (features[0-6]).
    Unfreeze last 3 blocks (features[7,8,9]) + classifier.

    Why features[7,8,9]:
        These are the high-level feature extractors.
        They need to learn eye-specific patterns.
        Earlier layers (edges/textures) are universal — keep frozen.
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only last 3 blocks + classifier
    for layer in [model.features[7],
                  model.features[8],
                  model.features[9],
                  model.classifier]:
        for param in layer.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\n  PHASE 2 — Last 3 blocks + head")
    print(f"  Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    print(f"  Frozen:   features[0-6] (edges, textures — universal, keep as-is)")
    print(f"  Training: features[7,8,9] + classifier (eye-specific features)")
    print(f"  NOTE: Phase 3 (unfreeze all) SKIPPED")
    print(f"        Early layers already optimal for eye edge/texture detection")


# ── Train / Eval ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

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
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="  Eval ", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out  = model(imgs)
        loss = criterion(out, labels)

        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


# ── Plot ─────────────────────────────────────────────────────────────────────────

def plot_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train", color="steelblue")
    axes[0].plot(history["val_loss"],   label="Val",   color="coral")
    axes[0].axvline(x=PHASE1_END, color="gray",
                    linestyle="--", alpha=0.6, label="Phase 1→2")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train", color="steelblue")
    axes[1].plot(history["val_acc"],   label="Val",   color="coral")
    axes[1].axhline(y=CUSTOM_CNN_BASELINE, color="green",
                    linestyle="--", alpha=0.8,
                    label=f"Custom CNN baseline ({CUSTOM_CNN_BASELINE:.2%})")
    axes[1].axvline(x=PHASE1_END, color="gray",
                    linestyle="--", alpha=0.6, label="Phase 1→2")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("MobileNetV3-Small  |  2-Phase Fine-tuning", fontsize=13)
    plt.tight_layout()
    plt.savefig(CURVES_SAVE, dpi=150)
    plt.close()
    print(f"  Saved → {CURVES_SAVE}")


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  MobileNetV3-Small — 2-Phase Fine-tuning")
    print(f"  Device: {DEVICE} | Batch: {BATCH_SIZE} | Epochs: {TOTAL_EPOCHS}")
    print("=" * 60)

    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Data
    print("\n📂 Loading data...")
    train_loader = DataLoader(
        EyeDatasetMobileNet(f"{SPLITS_DIR}/train.csv", is_train=True),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        EyeDatasetMobileNet(f"{SPLITS_DIR}/val.csv", is_train=False),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        EyeDatasetMobileNet(f"{SPLITS_DIR}/test.csv", is_train=False),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # Model + loss
    print("\n🧠 Loading pretrained MobileNetV3-Small...")
    model     = build_mobilenet(pretrained=True).to(DEVICE)
    weights   = get_class_weights().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Training state
    history      = {"train_loss": [], "val_loss": [],
                    "train_acc":  [], "val_acc":  []}
    best_val_acc = 0.0
    no_improve   = 0
    current_phase = 0

    for epoch in range(1, TOTAL_EPOCHS + 1):

        # Phase transitions
        if epoch == 1:
            phase1_setup(model)
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR_PHASE1
            )
            current_phase = 1

        elif epoch == PHASE1_END + 1:
            phase2_setup(model)
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR_PHASE2
            )
            no_improve = 0   # reset patience at phase transition
            current_phase = 2

        print(f"\nEpoch {epoch}/{TOTAL_EPOCHS}  [Phase {current_phase}]")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Train → loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        print(f"  Val   → loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f"  ✅ Best saved → {MODEL_SAVE}  (val_acc={val_acc:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{EARLY_STOP})")

        # Checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "best_val_acc": best_val_acc, "history": history,
                "phase": current_phase,
            }, f"models/mobilenet_ckpt_ep{epoch}.pth")
            print(f"  📌 Checkpoint saved (epoch {epoch})")

        # Early stopping — only after phase 2 starts
        if epoch > PHASE1_END and no_improve >= EARLY_STOP:
            print(f"\n  Early stopping at epoch {epoch}.")
            break

    # Final test evaluation
    print(f"\n{'='*60}")
    print(f"  Training complete. Best val acc: {best_val_acc:.4f}")
    plot_curves(history)

    print("\n📊 Test set evaluation...")
    model.load_state_dict(
        torch.load(MODEL_SAVE, map_location=DEVICE, weights_only=True)
    )
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    print(f"\n{'='*60}")
    print(f"  ── FINAL COMPARISON ──")
    print(f"  Custom CNN   →  96.74%  (2MB,  30+ FPS)")
    print(f"  MobileNetV3  →  {test_acc*100:.2f}%  (10MB, ~22 FPS)")
    print(f"{'─'*60}")
    if test_acc > CUSTOM_CNN_BASELINE:
        diff = (test_acc - CUSTOM_CNN_BASELINE) * 100
        print(f"  🏆 MobileNetV3 wins by +{diff:.2f}%")
        print(f"  → Use mobilenet_best.pth for main pipeline")
    else:
        diff = (CUSTOM_CNN_BASELINE - test_acc) * 100
        print(f"  🏆 Custom CNN wins by +{diff:.2f}%")
        print(f"  → Keep drowsiness_cnn_best.pth for main pipeline")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()