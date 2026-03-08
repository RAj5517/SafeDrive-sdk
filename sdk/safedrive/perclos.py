"""
perclos.py
──────────
PERCLOS — Percentage of Eye Closure over a sliding time window.

Industry standard fatigue metric used in professional driver monitoring.
Definition: % of frames where eyes are ≥ 70% closed in last N seconds.

Alert thresholds:
    PERCLOS > 0.20 → Level 2 alert
    PERCLOS > 0.35 → Level 3 critical alert
"""

from collections import deque


# Eye state scores
SCORE_OPEN      = 0.0
SCORE_HALF_OPEN = 0.5
SCORE_CLOSED    = 1.0

# Alert thresholds
PERCLOS_WARN     = 0.20
PERCLOS_CRITICAL = 0.35


class PERCLOSTracker:
    """
    Tracks eye closure percentage over a sliding window.

    Args:
        fps:         Camera frames per second (used to set window size)
        window_secs: Window duration in seconds (default: 60)
    """

    def __init__(self, fps: int = 30, window_secs: int = 60):
        self.window_size = fps * window_secs
        self.buffer = deque(maxlen=self.window_size)
        self.fps = fps
        self.window_secs = window_secs

    def update(self, class_id: int):
        """
        Add a new frame's eye state to the window.

        Args:
            class_id: 0=Open, 1=Half-Open, 2=Closed
        """
        if class_id == 0:
            score = SCORE_OPEN
        elif class_id == 1:
            score = SCORE_HALF_OPEN
        else:
            score = SCORE_CLOSED

        self.buffer.append(score)

    def get_perclos(self) -> float:
        """
        Compute current PERCLOS value.

        Returns:
            Float 0.0–1.0. Returns 0.0 if buffer is empty.
        """
        if not self.buffer:
            return 0.0
        return sum(self.buffer) / len(self.buffer)

    def get_alert_level(self) -> int:
        """
        Returns alert level based on PERCLOS:
            0 = Normal
            2 = Warning  (PERCLOS > 0.20)
            3 = Critical (PERCLOS > 0.35)
        """
        p = self.get_perclos()
        if p > PERCLOS_CRITICAL:
            return 3
        elif p > PERCLOS_WARN:
            return 2
        return 0

    def reset(self):
        """Clear the buffer (call when driver takes a break)."""
        self.buffer.clear()

    def __repr__(self):
        return (f"PERCLOSTracker(window={self.window_secs}s, "
                f"frames={len(self.buffer)}/{self.window_size}, "
                f"perclos={self.get_perclos():.3f})")