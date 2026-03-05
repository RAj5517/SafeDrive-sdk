"""
alarm.py
────────
Three-level alert system for drowsiness detection.

Level 1 — Warning  (eyes closed 1s):   yellow border + short beep
Level 2 — Alert    (eyes closed 2s):   red border + repeating alarm + text
Level 3 — Critical (eyes closed 3s+):  flashing red screen + continuous alarm + log

Usage:
    alarm = AlertSystem(fps=30)
    alarm.update(is_closed=True, frame)   # call every frame
    annotated_frame = alarm.get_frame()   # get frame with overlays drawn
"""

import cv2
import numpy as np
import time
import os
import logging

# Try to import pygame — graceful fallback if not installed
try:
    import pygame
    pygame.mixer.init()
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False
    print("⚠️  pygame not available — audio alerts disabled")


# ── Logging setup ───────────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
logging.basicConfig(
    filename="outputs/drowsiness_events.log",
    level=logging.INFO,
    format="%(asctime)s — %(message)s",
)


class AlertSystem:
    """
    Manages drowsiness alert state and visual/audio output.

    Args:
        fps:          Frames per second (to convert seconds → frames)
        alarm_path:   Path to alarm .wav file
    """

    LEVEL_1_FRAMES = None   # Set from fps in __init__
    LEVEL_2_FRAMES = None
    LEVEL_3_FRAMES = None
    RESET_FRAMES   = None

    def __init__(self, fps: int = 30, alarm_path: str = "sounds/alarm.wav"):
        self.fps = fps
        self.alarm_path = alarm_path

        # Frame thresholds
        self.LEVEL_1_FRAMES = int(1.0 * fps)   # 1 second
        self.LEVEL_2_FRAMES = int(2.0 * fps)   # 2 seconds
        self.LEVEL_3_FRAMES = int(3.0 * fps)   # 3 seconds
        self.RESET_FRAMES   = int(0.5 * fps)   # 0.5 seconds open → reset

        # State
        self.closed_frames = 0      # consecutive frames eyes closed
        self.open_frames   = 0      # consecutive frames eyes open (for reset)
        self.alert_level   = 0      # current level (0, 1, 2, 3)
        self.flash_state   = False  # for Level 3 screen flash
        self.frame_count   = 0      # total frames processed

        # Audio
        self._load_audio()

    def _load_audio(self):
        self.sound = None
        if not AUDIO_AVAILABLE:
            return
        if os.path.exists(self.alarm_path):
            try:
                self.sound = pygame.mixer.Sound(self.alarm_path)
            except Exception as e:
                print(f"⚠️  Could not load alarm sound: {e}")
        else:
            print(f"⚠️  Alarm file not found: {self.alarm_path} — audio disabled")

    def update(self, is_closed: bool):
        """
        Update alert state based on whether eyes are currently closed.

        Args:
            is_closed: True if fusion score indicates eyes closed
        """
        self.frame_count += 1

        if is_closed:
            self.closed_frames += 1
            self.open_frames = 0
        else:
            self.open_frames += 1
            # Reset once eyes open long enough
            if self.open_frames >= self.RESET_FRAMES:
                self._reset()
                return

        # Determine alert level from consecutive closed frames
        if self.closed_frames >= self.LEVEL_3_FRAMES:
            self._set_level(3)
        elif self.closed_frames >= self.LEVEL_2_FRAMES:
            self._set_level(2)
        elif self.closed_frames >= self.LEVEL_1_FRAMES:
            self._set_level(1)

    def _set_level(self, level: int):
        if level != self.alert_level:
            self.alert_level = level
            if level >= 2:
                logging.info(f"DROWSINESS LEVEL {level} — closed_frames={self.closed_frames}")
            self._trigger_audio(level)

    def _trigger_audio(self, level: int):
        if not AUDIO_AVAILABLE or self.sound is None:
            return
        pygame.mixer.stop()
        if level == 1:
            self.sound.set_volume(0.4)
            self.sound.play()
        elif level == 2:
            self.sound.set_volume(0.8)
            self.sound.play(loops=-1)       # loop
        elif level == 3:
            self.sound.set_volume(1.0)
            self.sound.play(loops=-1)

    def _reset(self):
        if self.alert_level > 0:
            logging.info(f"Alert cleared — was level {self.alert_level}")
        self.closed_frames = 0
        self.alert_level   = 0
        if AUDIO_AVAILABLE:
            pygame.mixer.stop()

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw alert overlays on a frame.

        Args:
            frame: BGR frame from OpenCV

        Returns:
            Annotated frame
        """
        h, w = frame.shape[:2]

        if self.alert_level == 0:
            return frame

        # ── Level 1 — Yellow border ────────────────────────────────────────────
        if self.alert_level == 1:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 215, 255), 8)
            cv2.putText(frame, "⚠ DROWSINESS WARNING",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2)

        # ── Level 2 — Red border + text ────────────────────────────────────────
        elif self.alert_level == 2:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
            cv2.putText(frame, "⛔ DROWSINESS DETECTED",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(frame, "Please rest immediately",
                        (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ── Level 3 — Flashing red screen ─────────────────────────────────────
        elif self.alert_level == 3:
            self.flash_state = not self.flash_state   # toggle every frame
            if self.flash_state:
                red_overlay = np.zeros_like(frame)
                red_overlay[:, :, 2] = 180            # red channel
                frame = cv2.addWeighted(frame, 0.5, red_overlay, 0.5, 0)

            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 16)
            cv2.putText(frame, "🚨 CRITICAL — STOP DRIVING",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(frame, "PULL OVER NOW",
                        (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        return frame

    @property
    def is_alerting(self) -> bool:
        return self.alert_level > 0