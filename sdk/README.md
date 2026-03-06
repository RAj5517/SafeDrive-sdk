# SafeDrive AI — Driver Drowsiness Detection SDK

Real-time driver drowsiness and distraction detection.  
Built on MediaPipe + MobileNetV3 (v0.1.0) with YOLOv8 multi-feature detection coming in v0.2.0.

---

## Install

```bash
pip install safedrive-ai
```

---

## Quick Start

```python
from safedrive import DrowsinessDetector

detector = DrowsinessDetector(pipeline="mediapipe")

@detector.on_drowsy
def handle(event):
    print(f"Level {event.level}: {event.message}")

detector.run(camera=0)
```

---

## Alert Levels

| Level | State    | Trigger                              | Action              |
|-------|----------|--------------------------------------|---------------------|
| 1     | WARNING  | Eyes half-open 3s / yawn / head tilt | Soft beep           |
| 2     | ALERT    | Eyes closed 2s / head nod            | Loud beep           |
| 3     | CRITICAL | Face gone 2s / eyes closed 4s        | Continuous alarm    |

**Separate alerts** (independent of drowsiness level):
- **Phone detected** → instant distraction alert
- **Seatbelt removed** → continuous safety alert
- **Smoking detected** → instant distraction alert

---

## Full API

```python
detector = DrowsinessDetector(
    pipeline          = "mediapipe",  # "yolo" coming in v0.2.0
    eye_close_seconds = 2.0,
    face_gone_seconds = 2.0,
    head_tilt_degrees = 15.0,
    detect_phone      = True,
    detect_seatbelt   = True,
    detect_yawn       = True,
    detect_smoking    = True,
    show_window       = True,
    ear_weight        = 0.5,
    cnn_weight        = 0.5,
)

@detector.on_drowsy      # DrowsyEvent
@detector.on_distraction # DistractionEvent
@detector.on_safety      # SafetyEvent
@detector.on_frame       # called every frame with annotated frame + FrameStats

detector.run(camera=0)
detector.stop()
detector.reset_alert()   # manual reset after CRITICAL
```

---

## Roadmap

| Version | Features                                          |
|---------|---------------------------------------------------|
| v0.1.0  | MediaPipe + MobileNetV3 drowsiness detection ✅   |
| v0.2.0  | YOLOv8 pipeline + full multi-feature detection    |
| v1.0.0  | Ensemble mode + benchmark tool + full docs        |

---

## Pipeline Comparison

| Feature          | MediaPipe (v0.1.0) | YOLO (v0.2.0)  |
|------------------|--------------------|----------------|
| Eyes open/closed | ✅                 | ✅             |
| Yawn detection   | ❌                 | ✅             |
| Phone detection  | ❌                 | ✅             |
| Seatbelt         | ❌                 | ✅             |
| Smoking          | ❌                 | ✅             |
| Head pose        | ✅ (both use MediaPipe landmarks) | ✅ |
| Speed            | ~28 FPS            | ~24 FPS        |

---

## License

MIT