"""
base_pipeline.py
────────────────
Abstract base class all pipelines must implement.
Guarantees mediapipe and yolo pipelines have identical interface.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BasePipeline(ABC):
    """
    Every pipeline (mediapipe, yolo) must implement these methods.
    The detector.py routes calls here without knowing which pipeline
    is running underneath.
    """

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single frame and return standardized result dict.

        Returns:
            {
                "ear":         float,        # Eye Aspect Ratio (0.0-1.0)
                "eye_state":   str,          # "open" / "half" / "closed"
                "cnn_prob":    float,        # CNN closed probability (0.0-1.0)
                "score":       float,        # fusion score (0.0-1.0)
                "face_found":  bool,         # was face detected this frame
                "perclos":     float,        # PERCLOS value (0.0-1.0)
                "landmarks":   dict | None,  # raw landmark data if available
                "frame":       np.ndarray,   # annotated frame for display
            }
        """
        pass

    @abstractmethod
    def start(self) -> None:
        """Initialize pipeline resources (models, camera etc)."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Release pipeline resources cleanly."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Pipeline name: 'mediapipe' or 'yolo'."""
        pass