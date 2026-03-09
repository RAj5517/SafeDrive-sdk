"""
alert_system.py
───────────────
Multi-level alert state machine for SafeDrive SDK.

Levels:
    0 → AWAKE      (normal)
    1 → WARNING    (early drowsy)
    2 → ALERT      (moderate drowsy)
    3 → CRITICAL   (dangerous / microsleep)

Separate tracks:
    DISTRACTION → phone / smoking (instant, independent)
    SAFETY      → seatbelt (continuous, independent)
"""

import time
import threading
from pathlib import Path
from typing import Callable, Optional
from .events import DrowsyEvent, DistractionEvent, SafetyEvent


# ── Thresholds ────────────────────────────────────────────────────────────────

DEFAULTS = {
    # Drowsiness
    "eye_close_seconds":  2.0,
    "face_gone_seconds":  2.0,
    "head_tilt_degrees":  15.0,
    "head_nod_degrees":   25.0,
    "half_eye_seconds":   3.0,
    "half_eye_seconds_2": 5.0,

    # Score thresholds
    "score_warning":  0.35,
    "score_alert":    0.60,
    "score_critical": 0.85,

    # Recovery — seconds of eyes-open required to step down each level
    "recovery_warning":  4.0,    # L1 → L0
    "recovery_alert":    8.0,    # L2 → L1
    "recovery_critical": 12.0,   # L3 → L2  (CAN recover, needs 12s eyes open)

    # YOLO distraction minimum confidence (below = ignored)
    "phone_min_conf":   0.55,
    "smoking_min_conf": 0.55,

    # Audio
    "audio_enabled": True,
}

LEVEL_LABELS = {0: "AWAKE", 1: "WARNING", 2: "ALERT", 3: "CRITICAL"}
LEVEL_COLORS = {
    0: (0,   255,   0),
    1: (0,   220, 255),
    2: (0,   140, 255),
    3: (0,     0, 255),
}

# Blink ignore threshold — closures shorter than this are normal blinks
BLINK_IGNORE_SECONDS = 0.5

# Volume ramp: (seconds_closed → volume)
# Eyes closed 0.5s → 0.2, ramps to 1.0 at 5s+
VOLUME_RAMP = [
    (0.5, 0.20),
    (1.5, 0.40),
    (2.5, 0.60),
    (3.5, 0.80),
    (5.0, 1.00),
]

# Interval between beeps while alarm is active (seconds)
BEEP_INTERVAL = 1.2

_WAV_PATH = str(Path(__file__).resolve().parents[3] / "sounds" / "alarm.wav")

# Init pygame mixer once at import time
_sound = None
try:
    import pygame
    pygame.mixer.init()
    _sound = pygame.mixer.Sound(_WAV_PATH)
    print("  [SafeDrive] Audio ready ✅")
except Exception as e:
    print(f"  [SafeDrive] Audio unavailable: {e}")


def _volume_for_duration(seconds: float) -> float:
    """Map how long eyes have been closed → volume 0.0–1.0."""
    vol = 0.0
    for threshold, v in VOLUME_RAMP:
        if seconds >= threshold:
            vol = v
    return vol


class _AlarmLoop:
    """
    Continuous alarm using pygame's built-in infinite loop.
    sound.play(loops=-1) plays with zero gap between repeats.
    Volume updated in real-time each frame — no restart, no break.
    """
    def __init__(self):
        self._active = False

    def start(self):
        if self._active or _sound is None:
            return
        self._active = True
        _sound.play(loops=-1)

    def stop(self):
        if not self._active or _sound is None:
            return
        self._active = False
        _sound.stop()

    def set_volume(self, vol: float):
        if _sound is not None:
            _sound.set_volume(max(0.0, min(1.0, vol)))


def _beep_once(volume: float) -> None:
    """Single one-shot beep for distraction/yawn events."""
    if _sound is None:
        return
    def _play():
        try:
            _sound.set_volume(volume)
            _sound.play()
        except Exception as e:
            print(f"  [SafeDrive] Audio error: {e}")
    threading.Thread(target=_play, daemon=True).start()


class AlertSystem:
    """
    Stateful alert system. Feed frame results, get callbacks + audio.

    Usage:
        system = AlertSystem(config)
        system.on_drowsy(callback_fn)
        system.update(frame_result)  ← called every frame
    """

    def __init__(self, config: dict = None):
        self.cfg = {**DEFAULTS, **(config or {})}

        # State
        self._level            = 0
        self._prev_level       = 0
        self._level_since      = time.time()
        self._face_lost_since:  Optional[float] = None
        self._eye_closed_since: Optional[float] = None
        self._half_eye_since:   Optional[float] = None
        self._eyes_open_since:  Optional[float] = None

        # Callbacks
        self._on_drowsy_cbs:      list[Callable] = []
        self._on_distraction_cbs: list[Callable] = []
        self._on_safety_cbs:      list[Callable] = []

        # Distraction cooldown (avoid spam)
        self._last_distraction: dict[str, float] = {}
        self._distraction_cooldown = 3.0

        # Continuous alarm loop — starts when eyes close, stops when open
        self._alarm = _AlarmLoop()

        self._lock = threading.Lock()

    # ── Callback registration ─────────────────────────────────────────────────

    def on_drowsy(self, fn: Callable) -> Callable:
        self._on_drowsy_cbs.append(fn)
        return fn

    def on_distraction(self, fn: Callable) -> Callable:
        self._on_distraction_cbs.append(fn)
        return fn

    def on_safety(self, fn: Callable) -> Callable:
        self._on_safety_cbs.append(fn)
        return fn

    # ── Main update ───────────────────────────────────────────────────────────

    def update(self, result: dict) -> int:
        with self._lock:
            now = time.time()

            face_found = result.get("face_found",  True)
            eye_state  = result.get("eye_state",   "open")
            score      = result.get("score",        0.0)
            ear        = result.get("ear",          0.3)

            # ── Face lost ────────────────────────────────────────────────────
            if not face_found:
                if self._face_lost_since is None:
                    self._face_lost_since = now
                face_gone = now - self._face_lost_since
            else:
                self._face_lost_since = None
                face_gone = 0.0

            # ── Eye closed ───────────────────────────────────────────────────
            if eye_state == "closed":
                if self._eye_closed_since is None:
                    self._eye_closed_since = now
                eye_closed_dur = now - self._eye_closed_since
            else:
                self._eye_closed_since = None
                eye_closed_dur = 0.0

            # ── Half eye ─────────────────────────────────────────────────────
            if eye_state == "half":
                if self._half_eye_since is None:
                    self._half_eye_since = now
                half_dur = now - self._half_eye_since
            else:
                self._half_eye_since = None
                half_dur = 0.0

            # ── Eyes open tracking (for level 3 recovery) ────────────────────
            if eye_state == "open" and face_found:
                if self._eyes_open_since is None:
                    self._eyes_open_since = now
                eyes_open_dur = now - self._eyes_open_since
            else:
                self._eyes_open_since = None
                eyes_open_dur = 0.0

            # ── Determine target level ────────────────────────────────────────
            target_level = 0

            if (half_dur >= self.cfg["half_eye_seconds"]
                    or score >= self.cfg["score_warning"]):
                target_level = max(target_level, 1)

            if (eye_closed_dur >= self.cfg["eye_close_seconds"]
                    or score >= self.cfg["score_alert"]
                    or half_dur >= self.cfg["half_eye_seconds_2"]):
                target_level = max(target_level, 2)

            if (face_gone >= self.cfg["face_gone_seconds"]
                    or eye_closed_dur >= self.cfg["eye_close_seconds"] * 2
                    or score >= self.cfg["score_critical"]):
                target_level = max(target_level, 3)

            # Level rises immediately, falls only after sustained eyes-open recovery
            if target_level > self._level:
                self._level       = target_level
                self._level_since = now
            elif target_level < self._level:
                recovery_map = {
                    1: self.cfg["recovery_warning"],
                    2: self.cfg["recovery_alert"],
                    3: self.cfg["recovery_critical"],
                }
                recovery_needed = recovery_map.get(self._level, 8.0)
                if eyes_open_dur >= recovery_needed:
                    self._level = max(0, self._level - 1)
                    self._level_since = now
                    self._eyes_open_since = None  # reset after step-down

            # ── Fire drowsy callback on level change ──────────────────────────
            if self._level != self._prev_level and self._level > 0:
                event = DrowsyEvent(
                    level    = self._level,
                    label    = LEVEL_LABELS[self._level],
                    ear      = ear,
                    score    = score,
                    duration = now - self._level_since,
                )
                self._fire(self._on_drowsy_cbs, event)
            self._prev_level = self._level

            # ── Continuous alarm: runs while eyes closed > blink threshold ────
            # Ignores blinks (< BLINK_IGNORE_SECONDS).
            # Volume ramps up the longer eyes stay closed.
            # Stops immediately when eyes open.
            if self.cfg.get("audio_enabled", True):
                alarm_dur = eye_closed_dur  # 0 if eyes open
                if alarm_dur >= BLINK_IGNORE_SECONDS:
                    vol = _volume_for_duration(alarm_dur)
                    self._alarm.set_volume(vol)
                    self._alarm.start()
                else:
                    self._alarm.stop()

            # ── Distraction alerts ────────────────────────────────────────────
            for dtype in ("phone", "smoking"):
                if result.get(f"{dtype}_detected", False):
                    conf = result.get(f"{dtype}_confidence", 1.0)
                    if conf < self.cfg.get(f"{dtype}_min_conf", 0.55):
                        continue
                    last = self._last_distraction.get(dtype, 0)
                    if now - last >= self._distraction_cooldown:
                        self._last_distraction[dtype] = now
                        self._fire(self._on_distraction_cbs,
                                   DistractionEvent(type=dtype, confidence=conf))
                        if self.cfg.get("audio_enabled", True):
                            _beep_once(0.5)

            # ── Yawn audio ────────────────────────────────────────────────────
            if result.get("yawn_detected", False):
                last = self._last_distraction.get("yawn", 0)
                if now - last >= 4.0:
                    self._last_distraction["yawn"] = now
                    if self.cfg.get("audio_enabled", True):
                        _beep_once(0.35)

            # ── Seatbelt monitoring ───────────────────────────────────────────
            if "seatbelt_present" in result and result["seatbelt_present"] is False:
                self._fire(self._on_safety_cbs,
                           SafetyEvent(type="seatbelt", state="absent"))
                if self.cfg.get("audio_enabled", True):
                    _beep_once(0.4)

            return self._level

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fire(self, callbacks: list, event) -> None:
        for cb in callbacks:
            try:
                cb(event)
            except Exception as e:
                print(f"  [SafeDrive] Callback error: {e}")

    def reset(self) -> None:
        with self._lock:
            self._alarm.stop()
            self._level            = 0
            self._prev_level       = 0
            self._level_since      = time.time()
            self._face_lost_since  = None
            self._eye_closed_since = None
            self._half_eye_since   = None
            self._eyes_open_since  = None

    @property
    def level(self) -> int:
        return self._level

    @property
    def level_label(self) -> str:
        return LEVEL_LABELS[self._level]

    @property
    def level_color(self) -> tuple:
        return LEVEL_COLORS[self._level]