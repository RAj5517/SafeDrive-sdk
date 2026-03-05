"""
hf_demo.py
──────────
HuggingFace Spaces demo — video file upload version.

User uploads a dashcam video clip → model runs detection on every frame
→ annotated video returned showing drowsiness alerts.

Deploy:
    1. Create HF Space (Gradio SDK)
    2. Push this file as app.py to HF Space repo
    3. Add requirements.txt to HF Space repo

Model is loaded from HuggingFace Hub (not uploaded to Spaces directly).
"""

import cv2
import numpy as np
import torch
import tempfile
import os
import sys

import gradio as gr
from huggingface_hub import hf_hub_download

# ── Load model from HuggingFace Hub ────────────────────────────────────────────
# Replace with your actual HF username and repo name
HF_REPO_ID  = "YOUR_HF_USERNAME/safedrive-model"
MODEL_FILE  = "drowsiness_cnn_best.pth"
DEVICE      = "cpu"   # HF Spaces free tier = CPU only

sys.path.insert(0, "src")

try:
    from eye_state_model import load_model, predict_eye_state
    from landmark_extractor import LandmarkExtractor
    from ear_calculator import average_ear, is_eye_closed
    from eye_extractor import extract_both_eyes
    from alarm import AlertSystem

    print("Loading model from HuggingFace Hub...")
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE)
    model = load_model(model_path, DEVICE)
    print("✅ Model loaded")

    LANDMARK_EXTRACTOR = LandmarkExtractor()
    MODEL_LOADED = True

except Exception as e:
    print(f"⚠️  Model load failed: {e}")
    MODEL_LOADED = False


EAR_WEIGHT = 0.4
CNN_WEIGHT = 0.6
FUSION_THRESH = 0.5


def process_video(video_path: str) -> str:
    """
    Run drowsiness detection on an uploaded video.

    Args:
        video_path: Path to uploaded video file

    Returns:
        Path to annotated output video
    """
    if not MODEL_LOADED:
        return None

    cap = cv2.VideoCapture(video_path)
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output file
    out_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    alert_system = AlertSystem(fps=fps, alarm_path="")   # no audio on HF

    frame_count = 0
    MAX_FRAMES  = fps * 60   # process max 60 seconds

    while cap.isOpened() and frame_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Landmark extraction
        data = LANDMARK_EXTRACTOR.extract(frame)
        if data:
            ear = average_ear(data["left_eye"], data["right_eye"])
            ear_closed = 1.0 if is_eye_closed(ear) else 0.0

            eyes = extract_both_eyes(frame, data["left_eye"], data["right_eye"])
            cnn_closed = 0.0

            for roi in [eyes["left"], eyes["right"]]:
                if roi is not None:
                    with torch.no_grad():
                        result = predict_eye_state(model, roi, DEVICE)
                    cnn_closed = max(cnn_closed, result["closed_prob"])

            combined  = EAR_WEIGHT * ear_closed + CNN_WEIGHT * cnn_closed
            is_closed = combined > FUSION_THRESH

            alert_system.update(is_closed)
            frame = alert_system.draw_overlay(frame)

            eye_label = "CLOSED" if is_closed else "OPEN"
            eye_color = (0, 0, 255) if is_closed else (0, 255, 0)
            cv2.putText(frame, f"Eye: {eye_label} | EAR: {ear:.3f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)

        writer.write(frame)

    cap.release()
    writer.release()
    return out_path


# ── Gradio Interface ────────────────────────────────────────────────────────────

DESCRIPTION = """
# 🚗 SafeDrive — Driver Drowsiness Detection Demo

Upload a dashcam video clip (max 60 seconds).  
The model will analyze eye state frame-by-frame and overlay drowsiness alerts.

**How it works:**
- MediaPipe detects facial landmarks
- Eye Aspect Ratio (EAR) + CNN classify eye state
- Alerts triggered if eyes closed > 2 seconds

**Alert Levels:**
- 🟡 Level 1 — Warning (1 second)
- 🔴 Level 2 — Alert (2 seconds)
- 🚨 Level 3 — Critical (3+ seconds)

> **Note:** This demo runs on CPU — processing may take 1–2× the video duration.
> For real-time detection (25+ FPS), run locally with GPU.
"""

demo = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload dashcam video (max 60 sec)"),
    outputs=gr.Video(label="Analyzed output with drowsiness overlay"),
    title="SafeDrive Drowsiness Detection",
    description=DESCRIPTION,
    examples=[],   # add sample video paths here if you have them
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()