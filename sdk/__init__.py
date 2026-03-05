"""
SafeDrive SDK
─────────────
Real-time driver drowsiness detection SDK.

Usage:
    from safedrive import DrowsinessDetector

    detector = DrowsinessDetector(
        model_path="models/drowsiness_cnn_best.pth",
        camera_id=0,
        alarm=True,
    )
    detector.start()
"""

from .detector import DrowsinessDetector

__version__ = "0.1.0"
__author__  = "SafeDrive"
__all__     = ["DrowsinessDetector"]