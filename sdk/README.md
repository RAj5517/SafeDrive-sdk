# SafeDrive AI

Real-time driver monitoring SDK — drowsiness, phone, seatbelt, and smoking detection via webcam. No special hardware required.

[![PyPI version](https://badge.fury.io/py/safedrive-ai.svg)](https://pypi.org/project/safedrive-ai/)
[![HuggingFace](https://img.shields.io/badge/🤗-Models-yellow)](https://huggingface.co/raj5517/safedrive-model)
[![GitHub](https://img.shields.io/badge/GitHub-SafeDrive--sdk-black)](https://github.com/RAj5517/SafeDrive-sdk)

---

## Install

```bash
pip install safedrive-ai pygame              # MediaPipe pipeline + audio
pip install safedrive-ai ultralytics pygame  # + YOLO pipeline
```

---

## Quick Start

```python
from safedrive import DrowsinessDetector

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

MediaPipe 468-point face mesh → EAR geometry → 64×64 eye crop → MobileNetV3 (97.99% acc). Best for CPU-only.

### `pipeline="yolo"` — YOLO + MediaPipe Hybrid

**MediaPipe** handles: eye state, PERCLOS, yawn (MAR), head pose — scale-invariant, works at any distance.  
**YOLO** handles: phone, seatbelt, cigarette — single GPU forward pass.

| Metric | Value |
|--------|-------|
| mAP50 | 0.940 |
| mAP50-95 | 0.793 |
| Avg latency | 19.3ms |
| Model size | 6.3MB |

---

## Features

| Feature | mediapipe | yolo |
|---------|:---------:|:----:|
| Eye state (3-class) | ✅ | ✅ |
| Yawn detection (MAR) | ✅ | ✅ |
| Head pose | ✅ | ✅ |
| PERCLOS tracking | ✅ | ✅ |
| Audio alerts | ✅ | ✅ |
| Phone detection | ❌ | ✅ |
| Seatbelt monitoring | ❌ | ✅ |
| Cigarette detection | ❌ | ✅ |
| CPU-only support | ✅ | ⚠️ slow |

---

## Alert System

```
DROWSINESS — 3 levels:
  Level 1  WARNING   Eyes half-open 3s / yawn / head tilt > 15°
  Level 2  ALERT     Eyes closed 2s / head nod > 25°
  Level 3  CRITICAL  Eyes closed 4s / face out of frame 2s

DISTRACTION — instant, independent:
  Phone / cigarette detected → alert + single audio beep

SAFETY — continuous:
  Seatbelt absent → alert + audio beep

AUDIO:
  Eyes closing → continuous alarm, volume ramps 0.2 → 1.0 over 5s
  Eyes open    → alarm stops instantly
  Blinks < 500ms → ignored completely
```

---

## Usage Examples

### All callbacks

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
    print(event.message)

detector.run(camera=0)
```

### Disable features

```python
detector = DrowsinessDetector(
    pipeline        = "yolo",
    detect_seatbelt = False,
    detect_smoking  = False,
)
```

### Custom thresholds

```python
detector = DrowsinessDetector(
    pipeline          = "yolo",
    eye_close_seconds = 1.5,    # default 2.0
    head_tilt_degrees = 20.0,   # default 15.0
)
```

### Headless

```python
detector = DrowsinessDetector(pipeline="yolo", show_window=False)

@detector.on_frame
def process(frame, stats):
    # stats: eye_state, fps, alert_level, perclos, ear, score
    pass

detector.run()
```

---

## Benchmark (1000 frames, RTX 3050)

| Metric | MediaPipe | YOLO Hybrid |
|--------|-----------|-------------|
| Avg FPS | 39.8 | **51.7** |
| Avg latency | 25.1ms | **19.3ms** |
| Face detection | 100% | 100% |
| Phone detection | ❌ | ✅ |
| GPU memory | 14MB | 20MB |

---

## Changelog

### v0.2.3
- YOLO pipeline now hybrid: MediaPipe for eyes/yawn/head pose, YOLO for phone/seatbelt/cigarette
- Continuous audio alerts via pygame — volume ramps with eye-closure duration
- Blink ignore: closures < 500ms treated as natural blinks, no alarm
- Fixed MAR landmark indices (outer lip) for accurate yawn detection
- Fixed PERCLOS module import path in SDK package

### v0.2.1
- Feature disable flags: `detect_phone`, `detect_seatbelt`, `detect_smoking`, `detect_yawn`

### v0.2.0
- YOLOv8-nano pipeline: single-pass 9-class detection
- Phone, seatbelt, cigarette detection and alerts
- mAP50 = 0.940 on custom 28,593-image dataset

### v0.1.1
- HuggingFace auto-download, local cache at `~/.cache/safedrive/models/`

### v0.1.0
- Initial release: MediaPipe + MobileNetV3 pipeline, PERCLOS, 3-level alerts

---

**Links:** [GitHub](https://github.com/RAj5517/SafeDrive-sdk) · [HuggingFace](https://huggingface.co/raj5517/safedrive-model) · [PyPI](https://pypi.org/project/safedrive-ai/)