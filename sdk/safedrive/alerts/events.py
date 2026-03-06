"""
events.py
─────────
Standardized event objects passed to all user callbacks.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class DrowsyEvent:
    """Fired when drowsiness level changes."""
    level:      int         # 1=warning, 2=alert, 3=critical
    label:      str         # "WARNING" / "ALERT" / "CRITICAL"
    ear:        float       # current EAR value
    score:      float       # fusion score 0.0-1.0
    duration:   float       # seconds in current state
    timestamp:  datetime    = field(default_factory=datetime.now)

    @property
    def message(self) -> str:
        return {
            1: "Stay Alert",
            2: "Wake Up!",
            3: "DANGER — Pull Over",
        }.get(self.level, "")


@dataclass
class DistractionEvent:
    """Fired instantly when distraction detected (phone, smoking)."""
    type:       str         # "phone" / "smoking"
    confidence: float       # detection confidence 0.0-1.0
    timestamp:  datetime    = field(default_factory=datetime.now)

    @property
    def message(self) -> str:
        return {
            "phone":   "Put Down Your Phone",
            "smoking": "No Smoking While Driving",
        }.get(self.type, "Distraction Detected")


@dataclass
class SafetyEvent:
    """Fired when seatbelt state changes."""
    type:       str         # "seatbelt"
    state:      str         # "removed" / "absent"
    timestamp:  datetime    = field(default_factory=datetime.now)

    @property
    def message(self) -> str:
        return "Fasten Your Seatbelt"


@dataclass
class FrameStats:
    """Passed to on_frame callback every frame."""
    ear:          float
    eye_state:    str        # "open" / "half" / "closed"
    cnn_prob:     float
    score:        float
    perclos:      float
    fps:          float
    face_found:   bool
    alert_level:  int        # 0=awake, 1=warning, 2=alert, 3=critical
    pipeline:     str        # "mediapipe" / "yolo"