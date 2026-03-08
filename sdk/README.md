# SafeDrive AI

Real-time driver monitoring SDK — drowsiness, phone, seatbelt, and smoking detection via webcam. No special hardware required.

[![PyPI version](https://badge.fury.io/py/safedrive-ai.svg)](https://pypi.org/project/safedrive-ai/)
[![HuggingFace](https://img.shields.io/badge/🤗-Models-yellow)](https://huggingface.co/raj5517/safedrive-model)
[![GitHub](https://img.shields.io/badge/GitHub-SafeDrive--sdk-black)](https://github.com/RAj5517/SafeDrive-sdk)

---

## Install

```bash
pip install safedrive-ai           # MediaPipe pipeline
pip install safedrive-ai ultralytics  # + YOLO pipeline
```

---

## Quick Start

```python
from safedrive import DrowsinessDetector

# MediaPipe + MobileNet pipeline (CPU friendly)
detector = DrowsinessDetector(pipeline="mediapipe")

# YOLOv8 pipeline (all features: phone, seatbelt, cigarette)
detector = DrowsinessDetector(pipeline="yolo")

@detector.on_drowsy
def handle(event):
    print(f"Level {event.level}: {event.message}")

detector.run(camera=0)
```

Models auto-download from [HuggingFace](https://huggingface.co/raj5517/safedrive-model) on first run.

---

## Pipelines

### `pipeline="mediapipe"` — MediaPipe + MobileNetV3

Two-stage pipeline. MediaPipe extracts 468 face landmarks → EAR geometry formula → eye crop → MobileNetV3 classifier (97.99% accuracy). Best for CPU-only.

### `pipeline="yolo"` — YOLOv8-nano

Single forward pass detects all 9 classes simultaneously. Custom-trained on 28,593 images.

| Metric | Value |
|--------|-------|
| mAP50 | 0.940 |
| mAP50-95 | 0.793 |
| Inference | 3.7ms/frame |
| Model size | 6.3MB |

---

## Features

| Feature | mediapipe | yolo |
|---------|:---------:|:----:|
| Eye state (3-class) | ✅ | ✅ |
| Yawn detection | ✅ | ✅ |
| Head pose | ✅ | ✅ |
| PERCLOS tracking | ✅ | ✅ |
| Phone detection | ❌ | ✅ |
| Seatbelt monitoring | ❌ | ✅ |
| Cigarette detection | ❌ | ✅ |

---

## Alert Levels

```
Level 1  WARNING   eyes half 3s / yawn / head tilt > 15°
Level 2  ALERT     eyes closed 2s / head nod > 25°
Level 3  CRITICAL  eyes closed 4s / face gone 2s

Phone / cigarette → instant DISTRACTION alert
Seatbelt removed  → continuous SAFETY alert
```

---

## Usage Examples

### All event types

```python
detector = DrowsinessDetector(pipeline="yolo")

@detector.on_drowsy
def drowsy(event):
    print(f"Level {event.level}: {event.message}")

@detector.on_distraction
def distraction(event):
    print(f"{event.type} detected")   # "phone" or "smoking"

@detector.on_safety
def safety(event):
    print(event.message)              # seatbelt events

detector.run(camera=0)
```

### Disable features

```python
# Testing without seatbelt or cigarette
detector = DrowsinessDetector(
    pipeline        = "yolo",
    detect_seatbelt = False,
    detect_smoking  = False,
)
```

### Custom thresholds

```python
detector = DrowsinessDetector(
    pipeline           = "yolo",
    eye_close_seconds  = 1.5,    # alert faster (default 2.0)
    head_tilt_degrees  = 20.0,   # more lenient (default 15.0)
)
```

### Headless

```python
detector = DrowsinessDetector(pipeline="yolo", show_window=False)

@detector.on_frame
def process(frame, stats):
    # stats.eye_state, stats.fps, stats.alert_level, stats.perclos
    pass

detector.run()
```

---

## Changelog

### v0.2.1
- Feature disable flags (`detect_phone`, `detect_seatbelt`, `detect_smoking`, `detect_yawn`)

### v0.2.0
- YOLOv8-nano pipeline — phone, seatbelt, cigarette detection
- mAP50 = 0.940 on 9-class custom dataset

### v0.1.1
- HuggingFace auto-download, local model cache

### v0.1.0
- Initial release — MediaPipe + MobileNetV3 pipeline

---

**Links:** [GitHub](https://github.com/RAj5517/SafeDrive-sdk) · [HuggingFace](https://huggingface.co/raj5517/safedrive-model) · [PyPI](https://pypi.org/project/safedrive-ai/)