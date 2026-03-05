"""
eye_state_dataset.py
────────────────────
PyTorch Dataset for eye state classification.

Reads from data/splits/train.csv, val.csv, test.csv.
Each CSV has columns: [image_path, label]
Labels: 0=Open, 1=Half-Open, 2=Closed

Applies Albumentations augmentations during training.
"""

import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A


TARGET_SIZE = 64

# ── Augmentation pipelines ─────────────────────────────────────────────────────

def get_train_transforms():
    """Light augmentations — eye images are small, heavy distortion destroys signal."""
    return A.Compose([
        A.Resize(TARGET_SIZE, TARGET_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.GaussianBlur(blur_limit=(3, 3), p=0.2),
        A.GaussNoise(p=0.2),
        A.Rotate(limit=15, p=0.3),
    ])

def get_val_transforms():
    """No augmentation for val/test — just resize."""
    return A.Compose([
        A.Resize(TARGET_SIZE, TARGET_SIZE),
    ])


# ── Dataset class ──────────────────────────────────────────────────────────────

class EyeStateDataset(Dataset):
    """
    Args:
        csv_path:   Path to split CSV file (image_path, label columns)
        transform:  Albumentations transform pipeline
    """

    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        assert "image_path" in self.df.columns, "CSV must have 'image_path' column"
        assert "label"      in self.df.columns, "CSV must have 'label' column"

        print(f"Loaded {len(self.df)} samples from {csv_path}")
        print(f"Class distribution:\n{self.df['label'].value_counts().sort_index()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = row["image_path"]
        label = int(row["label"])

        # Load image — grayscale
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Return a blank image if file missing (fail-safe)
            img = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)

        # Albumentations expects HxWxC for most transforms
        img = np.stack([img, img, img], axis=-1)   # fake RGB for compatibility

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        # Back to single channel grayscale
        img = img[:, :, 0]

        # Normalize to [0, 1] and add channel dim
        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)  # (1, 64, 64)

        return tensor, label


def get_class_weights(csv_path: str) -> torch.Tensor:
    """
    Compute inverse frequency class weights for CrossEntropyLoss.
    Handles class imbalance in dataset.
    """
    df = pd.read_csv(csv_path)
    counts = df["label"].value_counts().sort_index()
    total  = len(df)
    weights = total / (len(counts) * counts.values)
    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders(splits_dir: str,
                    batch_size: int = 64,
                    num_workers: int = 4) -> dict:
    """
    Build DataLoaders for train / val / test.

    Args:
        splits_dir:  Path to data/splits/ folder containing train/val/test CSV files
        batch_size:  Training batch size
        num_workers: DataLoader parallel workers

    Returns:
        dict with 'train', 'val', 'test' DataLoaders
    """
    datasets = {
        "train": EyeStateDataset(
            os.path.join(splits_dir, "train.csv"),
            transform=get_train_transforms()
        ),
        "val": EyeStateDataset(
            os.path.join(splits_dir, "val.csv"),
            transform=get_val_transforms()
        ),
        "test": EyeStateDataset(
            os.path.join(splits_dir, "test.csv"),
            transform=get_val_transforms()
        ),
    }

    loaders = {}
    for split, ds in datasets.items():
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )

    return loaders


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    splits_dir = sys.argv[1] if len(sys.argv) > 1 else "data/splits"

    loaders = get_dataloaders(splits_dir, batch_size=8, num_workers=0)

    batch_imgs, batch_labels = next(iter(loaders["train"]))
    print(f"Batch images shape: {batch_imgs.shape}")
    print(f"Batch labels:       {batch_labels}")
    print(f"Pixel value range:  [{batch_imgs.min():.3f}, {batch_imgs.max():.3f}]")