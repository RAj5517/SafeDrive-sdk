"""
safedrive
─────────
SafeDrive AI — Real-time driver drowsiness detection SDK.

Quick start:
    from safedrive import DrowsinessDetector

    detector = DrowsinessDetector(pipeline="mediapipe")

    @detector.on_drowsy
    def handle(event):
        print(f"Level {event.level}: {event.message}")

    detector.run(camera=0)
"""

from .detector import DrowsinessDetector
from .alerts.events import DrowsyEvent, DistractionEvent, SafetyEvent, FrameStats
from .alerts.alert_system import AlertSystem

__version__ = "0.1.0"
__author__  = "SafeDrive AI"
__all__     = [
    "DrowsinessDetector",
    "DrowsyEvent",
    "DistractionEvent",
    "SafetyEvent",
    "FrameStats",
    "AlertSystem",
]