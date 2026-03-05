"""
sdk/detector.py
───────────────
Public-facing SDK wrapper around the full pipeline.
This is what trucking companies / customers actually use.

pip install safedrive-sdk
→ from safedrive import DrowsinessDetector
"""

import sys
import os

# Point to src/ modules
SDK_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(SDK_DIR, "..", "src")
sys.path.insert(0, SRC_DIR)


class DrowsinessDetector:
    """
    SafeDrive Drowsiness Detector — main SDK class.

    Args:
        model_path:   Path to trained .pth weights file
        camera_id:    Webcam/dashcam device index (default: 0)
        alarm:        Enable audio alarm (default: True)
        device:       'cuda' or 'cpu' (auto-detect if None)
        on_alert:     Optional callback function — called when alert triggers.
                      Signature: on_alert(level: int, timestamp: str)
        webhook_url:  Optional URL — POST alert JSON to this endpoint

    Example:
        detector = DrowsinessDetector(
            model_path="models/drowsiness_cnn_best.pth",
            camera_id=0,
            alarm=True,
            on_alert=lambda level, ts: print(f"ALERT level {level} at {ts}")
        )
        detector.start()
    """

    def __init__(self,
                 model_path: str,
                 camera_id: int = 0,
                 alarm: bool = True,
                 device: str = None,
                 on_alert=None,
                 webhook_url: str = None):

        self.model_path  = model_path
        self.camera_id   = camera_id
        self.alarm       = alarm
        self.device      = device
        self.on_alert    = on_alert
        self.webhook_url = webhook_url
        self._detector   = None

    def start(self):
        """
        Start real-time detection loop.
        Blocks until user presses Q or Ctrl+C.
        """
        from realtime_detector import RealtimeDetector

        self._detector = RealtimeDetector(
            model_path=self.model_path,
            camera_id=self.camera_id,
            device=self.device,
        )
        self._detector.run()

    def stop(self):
        """Stop detection loop programmatically."""
        if self._detector:
            self._detector.stop_event.set()

    @property
    def version(self):
        from . import __version__
        return __version__