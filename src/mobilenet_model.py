"""
mobilenet_model.py
──────────────────
MobileNetV3-Small fine-tuned for eye state classification.

3-Phase fine-tuning strategy:
    Phase 1 (epochs 1–3):   freeze ALL backbone, train classifier head only
    Phase 2 (epochs 4–13):  unfreeze last 3 blocks, train with low LR
    Phase 3 (epochs 14+):   unfreeze everything, very low LR final polish

Input:  (batch, 3, 224, 224) RGB — MobileNetV3 expects RGB not grayscale
Output: (batch, 3) logits → [Open, Half-Open, Closed]

Why MobileNetV3-Small:
    - Pretrained on ImageNet (1.2M images) — already knows edges/textures/shapes
    - Small enough for 20+ FPS real-time inference
    - Only 2.5M parameters vs our custom CNN's 500K
    - Expected accuracy: 97–98% vs 96.74% custom CNN
"""

import torch
import torch.nn as nn
import torchvision.models as models


CLASS_NAMES = ["Open", "Half-Open", "Closed"]
NUM_CLASSES = 3


def build_mobilenet(num_classes: int = NUM_CLASSES,
                    pretrained: bool = True) -> nn.Module:
    """
    Build MobileNetV3-Small with custom classifier head.

    Original classifier head (1000 classes) is replaced with:
        Linear → Hardswish → Dropout → Linear(3)

    Args:
        num_classes: Number of output classes (3 for us)
        pretrained:  Load ImageNet pretrained weights

    Returns:
        MobileNetV3-Small model with custom head
    """
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.mobilenet_v3_small(weights=weights)

    # Replace classifier head
    # Original: Linear(576→1024) → Hardswish → Dropout → Linear(1024→1000)
    # Ours:     Linear(576→1024) → Hardswish → Dropout → Linear(1024→3)
    in_features = model.classifier[3].in_features   # 1024
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model


def freeze_backbone(model: nn.Module):
    """
    Phase 1 — Freeze ALL layers except classifier head.
    Only the new Linear(1024→3) layer trains.
    """
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only classifier head
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Phase 1 — Frozen backbone")
    print(f"  Trainable: {trainable:,} / {total:,} params ({trainable/total*100:.1f}%)")


def unfreeze_top_blocks(model: nn.Module):
    """
    Phase 2 — Unfreeze last 3 blocks of MobileNetV3 backbone.
    MobileNetV3-Small has 9 InvertedResidual blocks (features[1] to features[9]).
    We unfreeze the last 3: features[7], features[8], features[9] + classifier.
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 3 blocks + pooling + classifier
    for layer in [model.features[7],
                  model.features[8],
                  model.features[9],
                  model.classifier]:
        for param in layer.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Phase 2 — Last 3 blocks + classifier unfrozen")
    print(f"  Trainable: {trainable:,} / {total:,} params ({trainable/total*100:.1f}%)")


def unfreeze_all(model: nn.Module):
    """
    Phase 3 — Unfreeze everything for final fine-tuning.
    Use very low LR (1e-5) to avoid destroying pretrained weights.
    """
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters())
    print(f"  Phase 3 — All layers unfrozen")
    print(f"  Trainable: {trainable:,} / {trainable:,} params (100%)")


def load_mobilenet(weights_path: str, device: str = "cpu") -> nn.Module:
    """
    Load a trained MobileNetV3 from a .pth file.

    Args:
        weights_path: Path to .pth file
        device:       'cpu' or 'cuda'

    Returns:
        Model in eval mode
    """
    model = build_mobilenet(pretrained=False)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_eye_state(model: nn.Module,
                      roi: "np.ndarray",
                      device: str = "cpu") -> dict:
    """
    Run inference on a single eye ROI.

    Args:
        model:  Trained MobileNetV3 in eval mode
        roi:    np.array (64, 64) float32 grayscale [0,1]
                Will be converted to RGB 224×224 internally
        device: torch device string

    Returns:
        dict with class_id, class_name, probabilities, closed_prob
    """
    import numpy as np
    import torch
    import cv2

    # MobileNetV3 needs RGB 224×224
    # Convert grayscale 64×64 → RGB 224×224
    resized = cv2.resize((roi * 255).astype("uint8"), (224, 224))
    rgb     = np.stack([resized, resized, resized], axis=0)  # (3, 224, 224)

    # Normalize with ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    normalized = (rgb.astype(np.float32) / 255.0 - mean) / std

    tensor = torch.from_numpy(normalized).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    class_id = int(probs.argmax().item())

    return {
        "class_id":      class_id,
        "class_name":    CLASS_NAMES[class_id],
        "probabilities": probs.cpu().numpy().tolist(),
        "closed_prob":   float(probs[2].item()),
    }


# ── Quick architecture check ───────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_mobilenet(pretrained=False)

    total     = sum(p.numel() for p in model.parameters())
    print(f"MobileNetV3-Small parameters: {total:,}")
    print(f"Approx size: {total * 4 / 1024 / 1024:.1f} MB")

    # Test forward pass
    dummy = torch.randn(4, 3, 224, 224)
    out   = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")

    # Show layer names for reference
    print("\nBackbone blocks:")
    for i, layer in enumerate(model.features):
        print(f"  features[{i}]: {layer.__class__.__name__}")