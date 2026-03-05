"""
train_eye_state.py
──────────────────
Full training loop for EyeStateCNN.

Usage:
    python src/train_eye_state.py

Saves:
    models/drowsiness_cnn_best.pth   — best model by val accuracy
    outputs/training_curves.png      — loss + accuracy plots
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from eye_state_model import EyeStateCNN
from eye_state_dataset import get_dataloaders, get_class_weights


# ── Config ─────────────────────────────────────────────────────────────────────
SPLITS_DIR    = "data/splits"
MODEL_SAVE    = "models/drowsiness_cnn_best.pth"
CURVES_SAVE   = "outputs/training_curves.png"

EPOCHS        = 30
BATCH_SIZE    = 64
LR            = 3e-4
EARLY_STOP    = 10       # patience epochs
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


# ── Training functions ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="  Val  ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


def plot_curves(history: dict, save_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"],   label="Val Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history["train_acc"], label="Train Acc")
    ax2.plot(history["val_acc"],   label="Val Acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training curves → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Data
    loaders = get_dataloaders(SPLITS_DIR, batch_size=BATCH_SIZE)

    # Model
    model = EyeStateCNN().to(DEVICE)

    # Compile for speed (PyTorch 2.0+)
    try:
        model = torch.compile(model)
        print("torch.compile() enabled — ~10–30% speed boost")
    except Exception:
        print("torch.compile() not available — continuing without it")

    # Loss with class weighting
    weights   = get_class_weights(f"{SPLITS_DIR}/train.csv").to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Optimizer + scheduler
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5, verbose=True
    )

    # Training loop
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc  = 0.0
    no_improve    = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, loaders["train"], optimizer, criterion, DEVICE)
        val_loss,   val_acc   = evaluate(model, loaders["val"], criterion, DEVICE)

        scheduler.step(val_acc)

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
            print(f"  ✅ Best model saved → {MODEL_SAVE} (val_acc={val_acc:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{EARLY_STOP})")

        # Early stopping
        if no_improve >= EARLY_STOP:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\n✅ Training complete. Best val accuracy: {best_val_acc:.4f}")
    plot_curves(history, CURVES_SAVE)

    # Final test set evaluation
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(MODEL_SAVE))
    test_loss, test_acc = evaluate(model, loaders["test"], criterion, DEVICE)
    print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()