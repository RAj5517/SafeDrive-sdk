"""
model_manager.py
────────────────
Handles model weight downloading from HuggingFace Hub.
Models cached locally at ~/.cache/safedrive/models/

HuggingFace repo: huggingface.co/raj5517/safedrive-model

First run:  downloads from HuggingFace automatically
Next runs:  loads from local cache (no internet needed)
"""

from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────
HF_REPO_ID = "raj5517/safedrive-model"

MODELS = {
    "mobilenet_webcam": {
        "filename":    "mobilenet_webcam.pth",
        "description": "MobileNetV3 fine-tuned on webcam data (primary model)",
        "required":    True,
    },
    "mobilenet_base": {
        "filename":    "mobilenet_best.pth",
        "description": "MobileNetV3 trained on 152k lab images (fallback)",
        "required":    False,
    },
    "cnn_base": {
        "filename":    "drowsiness_cnn_best.pth",
        "description": "Custom CNN 96.74% accuracy (last resort fallback)",
        "required":    False,
    },
    "face_landmarker": {
        "filename":    "face_landmarker.task",
        "description": "MediaPipe face landmark model (required for MediaPipe pipeline)",
        "required":    True,
    },
    # v0.2.0 — uncomment when YOLO is ready
    # "yolo_safedrive": {
    #     "filename":    "yolo_safedrive.pt",
    #     "description": "YOLOv8-nano multi-feature detector",
    #     "required":    True,
    # },
}

# Local cache directory
CACHE_DIR = Path.home() / ".cache" / "safedrive" / "models"

# Priority order for eye model selection
MODEL_PRIORITY = ["mobilenet_webcam", "mobilenet_base", "cnn_base"]


def get_model_path(model_key: str = "mobilenet_webcam") -> str:
    """
    Get local path to model, downloading from HuggingFace if not cached.

    Args:
        model_key: "mobilenet_webcam" | "mobilenet_base" |
                   "cnn_base" | "face_landmarker"

    Returns:
        str: absolute local path to model file
    """
    if model_key not in MODELS:
        raise ValueError(
            f"Unknown model: '{model_key}'. "
            f"Available: {list(MODELS.keys())}"
        )

    info       = MODELS[model_key]
    filename   = info["filename"]
    local_path = CACHE_DIR / filename

    if local_path.exists():
        print(f"  [SafeDrive] {filename} loaded from cache")
        return str(local_path)

    print(f"  [SafeDrive] Downloading {filename} from HuggingFace...")
    print(f"  [SafeDrive] {info['description']}")
    print(f"  [SafeDrive] Saving to: {CACHE_DIR}  (one-time download)")

    return _download_from_hf(filename, local_path)


def get_best_eye_model() -> str:
    """
    Returns path to the best available eye classification model.
    Priority: mobilenet_webcam > mobilenet_base > cnn_base
    Downloads primary if none cached.
    """
    for key in MODEL_PRIORITY:
        local_path = CACHE_DIR / MODELS[key]["filename"]
        if local_path.exists():
            print(f"  [SafeDrive] Using: {MODELS[key]['filename']}")
            return str(local_path)

    print("  [SafeDrive] No model cached. Downloading primary model...")
    return get_model_path("mobilenet_webcam")


def ensure_all_required() -> dict:
    """
    Download all required models upfront.
    Useful for offline environments after first setup.

    Returns:
        dict: {model_key: local_path}
    """
    paths = {}
    for key, info in MODELS.items():
        if info["required"]:
            paths[key] = get_model_path(key)
    return paths


def _download_from_hf(filename: str, local_path: Path) -> str:
    """Download a single model file from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "\nhuggingface_hub not installed.\n"
            "Fix: pip install huggingface_hub\n\n"
            "Or manually download from:\n"
            f"  https://huggingface.co/{HF_REPO_ID}\n"
            "and pass model_path= to DrowsinessDetector"
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        downloaded = hf_hub_download(
            repo_id   = HF_REPO_ID,
            filename  = filename,
            local_dir = str(CACHE_DIR),
        )
        print(f"  [SafeDrive] Saved to: {downloaded}")
        return downloaded

    except Exception as e:
        raise RuntimeError(
            f"\nFailed to download {filename}.\n"
            f"Error: {e}\n\n"
            f"Manual fix:\n"
            f"  1. Visit https://huggingface.co/{HF_REPO_ID}\n"
            f"  2. Download {filename}\n"
            f"  3. Place in: {CACHE_DIR}\n"
            f"  OR pass path directly:\n"
            f"     DrowsinessDetector(model_path='your/path/{filename}')"
        )


def list_cached() -> None:
    """Print all locally cached models and their sizes."""
    print(f"\n  SafeDrive model cache: {CACHE_DIR}")
    print("  " + "─" * 55)
    if not CACHE_DIR.exists():
        print("  No models cached yet. Run ensure_all_required().")
        return

    for key, info in MODELS.items():
        path = CACHE_DIR / info["filename"]
        if path.exists():
            mb = path.stat().st_size / 1024 / 1024
            print(f"  OK  {key:20s}  {mb:6.1f} MB  {info['filename']}")
        else:
            print(f"  --  {key:20s}  not cached  {info['filename']}")
    print()


def clear_cache() -> None:
    """Delete all cached model files."""
    import shutil
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        print(f"  [SafeDrive] Cache cleared: {CACHE_DIR}")
    else:
        print("  [SafeDrive] Cache already empty.")


def set_hf_repo(repo_id: str) -> None:
    """
    Override the HuggingFace repo ID.
    Use if you host your own fine-tuned models.

    Args:
        repo_id: e.g. "myusername/my-safedrive-models"
    """
    global HF_REPO_ID
    HF_REPO_ID = repo_id
    print(f"  [SafeDrive] HF repo set to: {repo_id}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"

    if cmd == "list":
        list_cached()
    elif cmd == "download":
        ensure_all_required()
    elif cmd == "clear":
        clear_cache()
    else:
        print("Usage: python model_manager.py [list|download|clear]")