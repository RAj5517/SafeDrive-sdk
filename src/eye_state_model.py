"""
eye_state_model.py
──────────────────
EyeStateCNN — Custom CNN for eye state classification.

Input:  (batch, 1, 64, 64) grayscale normalized eye image
Output: (batch, 3) logits for [Open, Half-Open, Closed]

Architecture:
    Block 1: Conv(1→32)  → BN → ReLU → MaxPool(2)   → 32×32
    Block 2: Conv(32→64) → BN → ReLU → MaxPool(2)   → 16×16
    Block 3: Conv(64→128)→ BN → ReLU → MaxPool(2)   →  8×8
    Flatten: 128 × 8 × 8 = 8192
    FC1:     8192 → 256 → ReLU → Dropout(0.5)
    FC2:     256  → 3

Labels:
    0 = Open
    1 = Half-Open  (early warning)
    2 = Closed
"""

import torch
import torch.nn as nn


CLASS_NAMES = ["Open", "Half-Open", "Closed"]
NUM_CLASSES = len(CLASS_NAMES)


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool"""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class EyeStateCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1,  32),   # (1,  64, 64) → (32, 32, 32)
            ConvBlock(32, 64),   # (32, 32, 32) → (64, 16, 16)
            ConvBlock(64, 128),  # (64, 16, 16) → (128, 8,  8)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                       # 128 × 8 × 8 = 8192
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_model(weights_path: str, device: str = "cpu") -> EyeStateCNN:
    """
    Load a trained EyeStateCNN from a .pth file.

    Args:
        weights_path: Path to .pth file
        device:       'cpu', 'cuda', or 'mps'

    Returns:
        Model in eval mode, moved to device
    """
    model = EyeStateCNN()
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_eye_state(model: EyeStateCNN,
                      roi: "np.ndarray",
                      device: str = "cpu") -> dict:
    """
    Run inference on a single eye ROI.

    Args:
        model:  Trained EyeStateCNN in eval mode
        roi:    np.array (64, 64) float32 normalized [0, 1]
        device: torch device string

    Returns:
        dict with:
            'class_id':    int (0=Open, 1=Half-Open, 2=Closed)
            'class_name':  str
            'probabilities': list of 3 floats
            'closed_prob': float (probability of Closed class — for fusion)
    """
    import numpy as np
    import torch

    tensor = torch.from_numpy(roi).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,64,64)

    with torch.no_grad():
        logits = model(tensor)                        # (1, 3)
        probs  = torch.softmax(logits, dim=1)[0]     # (3,)

    class_id = int(probs.argmax().item())

    return {
        "class_id":      class_id,
        "class_name":    CLASS_NAMES[class_id],
        "probabilities": probs.cpu().numpy().tolist(),
        "closed_prob":   float(probs[2].item()),
    }


# ── Quick test — model forward pass ───────────────────────────────────────────
if __name__ == "__main__":
    model = EyeStateCNN()
    dummy = torch.randn(4, 1, 64, 64)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
    print(f"Approx model size: {total * 4 / 1024 / 1024:.2f} MB")