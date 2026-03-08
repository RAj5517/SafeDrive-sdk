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


HF_REPO_ID = "raj5517/safedrive-model"

MODELS = {
    "mobilenet_webcam": {
        "filename":    "mobilenet_webcam.pth",
        "description": "MobileNetV3 fine-tuned on webcam data (primary model)",
        "required_by": ["mediapipe"],
    },
    "mobilenet_base": {
        "filename":    "mobilenet_best.pth",
        "description": "MobileNetV3 trained on 152k lab images (fallback)",
        "required_by": ["mediapipe"],
    },
    "cnn_base": {
        "filename":    "drowsiness_cnn_best.pth",
        "description": "Custom CNN 96.74% (last resort fallback)",
        "required_by": ["mediapipe"],
    },
    "face_landmarker": {
        "filename":    "face_landmarker.task",
        "description": "MediaPipe face landmark model (required by both pipelines)",
        "required_by": ["mediapipe", "yolo"],
    },
    "yolo_safedrive": {
        "filename":    "yolo_safedrive.pt",
        "description": "YOLOv8-nano multi-feature detector (eyes, mouth, phone, seatbelt, cigarette)",
        "required_by": ["yolo"],
    },
}

CACHE_DIR      = Path.home() / ".cache" / "safedrive" / "models"
MODEL_PRIORITY = ["mobilenet_webcam", "mobilenet_base", "cnn_base"]


def get_model_path(model_key: str = "mobilenet_webcam") -> str:
    if model_key not in MODELS:
        raise ValueError(
            f"Unknown model: '{model_key}'. "
            f"Available: {list(MODELS.keys())}"
        )
    info       = MODELS[model_key]
    local_path = CACHE_DIR / info["filename"]
    if local_path.exists():
        print(f"  [SafeDrive] {info['filename']} loaded from cache")
        return str(local_path)
    print(f"  [SafeDrive] Downloading {info['filename']} from HuggingFace...")
    print(f"  [SafeDrive] {info['description']}")
    print(f"  [SafeDrive] Saving to: {CACHE_DIR}  (one-time download)")
    return _download_from_hf(info["filename"], local_path)


def get_best_eye_model() -> str:
    for key in MODEL_PRIORITY:
        local_path = CACHE_DIR / MODELS[key]["filename"]
        if local_path.exists():
            print(f"  [SafeDrive] Using: {MODELS[key]['filename']}")
            return str(local_path)
    print("  [SafeDrive] No eye model cached. Downloading...")
    return get_model_path("mobilenet_webcam")


def ensure_pipeline_models(pipeline: str) -> dict:
    """Download all models required by a pipeline. pipeline = 'mediapipe' or 'yolo'"""
    paths = {}
    for key, info in MODELS.items():
        if pipeline in info.get("required_by", []):
            paths[key] = get_model_path(key)
    return paths


def _download_from_hf(filename: str, local_path: Path) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "\nhuggingface_hub not installed.\n"
            "Fix: pip install huggingface_hub\n\n"
            f"Or manually download from:\n"
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
            f"  OR: DrowsinessDetector(model_path='your/path/{filename}')"
        )


def list_cached() -> None:
    print(f"\n  SafeDrive model cache: {CACHE_DIR}")
    print("  " + "─" * 60)
    if not CACHE_DIR.exists():
        print("  No models cached yet.")
        return
    for key, info in MODELS.items():
        path = CACHE_DIR / info["filename"]
        if path.exists():
            mb        = path.stat().st_size / 1024 / 1024
            pipelines = "/".join(info.get("required_by", []))
            print(f"  OK  {key:20s}  {mb:6.1f} MB  [{pipelines}]")
        else:
            print(f"  --  {key:20s}  not cached")
    print()


def clear_cache() -> None:
    import shutil
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        print(f"  [SafeDrive] Cache cleared: {CACHE_DIR}")
    else:
        print("  [SafeDrive] Cache already empty.")


def set_hf_repo(repo_id: str) -> None:
    global HF_REPO_ID
    HF_REPO_ID = repo_id
    print(f"  [SafeDrive] HF repo set to: {repo_id}")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        list_cached()
    elif cmd == "download":
        pipeline = sys.argv[2] if len(sys.argv) > 2 else "mediapipe"
        ensure_pipeline_models(pipeline)
    elif cmd == "clear":
        clear_cache()
    else:
        print("Usage: python model_manager.py [list | download mediapipe | download yolo | clear]")