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
from typing import Callable, Optional
from .events import DrowsyEvent, DistractionEvent, SafetyEvent


# ── Thresholds ────────────────────────────────────────────────────────────────

DEFAULTS = {
    # Drowsiness
    "eye_close_seconds":  2.0,   # closed eyes → Level 2
    "face_gone_seconds":  2.0,   # face out of frame → Level 3
    "head_tilt_degrees":  15.0,  # tilt → Level 1
    "head_nod_degrees":   25.0,  # nod → Level 2
    "half_eye_seconds":   3.0,   # half-open sustained → Level 1
    "half_eye_seconds_2": 5.0,   # half-open sustained → Level 2

    # Score thresholds
    "score_warning":  0.35,
    "score_alert":    0.60,
    "score_critical": 0.85,

    # Recovery
    "recovery_warning":  5.0,   # seconds alert before auto-recovery
    "recovery_alert":    8.0,
}

LEVEL_LABELS = {0: "AWAKE", 1: "WARNING", 2: "ALERT", 3: "CRITICAL"}
LEVEL_COLORS = {
    0: (0,   255,   0),    # green
    1: (0,   220, 255),    # yellow
    2: (0,   140, 255),    # orange
    3: (0,     0, 255),    # red
}


class AlertSystem:
    """
    Stateful alert system. Feed frame results, get callbacks.

    Usage:
        system = AlertSystem(config)
        system.on_drowsy(callback_fn)
        system.update(frame_result)  ← called every frame
    """

    def __init__(self, config: dict = None):
        self.cfg = {**DEFAULTS, **(config or {})}

        # State
        self._level           = 0
        self._prev_level      = 0
        self._level_since     = time.time()
        self._face_lost_since: Optional[float] = None
        self._eye_closed_since: Optional[float] = None
        self._half_eye_since:   Optional[float] = None

        # Callbacks
        self._on_drowsy_cbs:      list[Callable] = []
        self._on_distraction_cbs: list[Callable] = []
        self._on_safety_cbs:      list[Callable] = []

        # Distraction cooldown (avoid spam)
        self._last_distraction: dict[str, float] = {}
        self._distraction_cooldown = 3.0   # seconds

        self._lock = threading.Lock()

    # ── Callback registration ─────────────────────────────────────────────────

    def on_drowsy(self, fn: Callable) -> Callable:
        """Register callback for drowsiness level changes."""
        self._on_drowsy_cbs.append(fn)
        return fn

    def on_distraction(self, fn: Callable) -> Callable:
        """Register callback for phone/smoking detection."""
        self._on_distraction_cbs.append(fn)
        return fn

    def on_safety(self, fn: Callable) -> Callable:
        """Register callback for seatbelt events."""
        self._on_safety_cbs.append(fn)
        return fn

    # ── Main update ───────────────────────────────────────────────────────────

    def update(self, result: dict) -> int:
        """
        Feed a frame result dict, returns current alert level.

        result keys: face_found, eye_state, score, ear,
                     phone_detected, smoking_detected,
                     seatbelt_present, head_tilt, head_nod
        """
        with self._lock:
            now = time.time()

            face_found  = result.get("face_found",  True)
            eye_state   = result.get("eye_state",   "open")
            score       = result.get("score",        0.0)
            ear         = result.get("ear",          0.3)

            # ── Face lost tracking ────────────────────────────────────────────
            if not face_found:
                if self._face_lost_since is None:
                    self._face_lost_since = now
                face_gone = now - self._face_lost_since
            else:
                self._face_lost_since = None
                face_gone = 0.0

            # ── Eye closed tracking ───────────────────────────────────────────
            if eye_state == "closed":
                if self._eye_closed_since is None:
                    self._eye_closed_since = now
                eye_closed_dur = now - self._eye_closed_since
            else:
                self._eye_closed_since = None
                eye_closed_dur = 0.0

            # ── Half-eye tracking ─────────────────────────────────────────────
            if eye_state == "half":
                if self._half_eye_since is None:
                    self._half_eye_since = now
                half_dur = now - self._half_eye_since
            else:
                self._half_eye_since = None
                half_dur = 0.0

            # ── Determine target level ────────────────────────────────────────
            target_level = 0

            # Level 1 triggers
            if (half_dur >= self.cfg["half_eye_seconds"]
                    or score >= self.cfg["score_warning"]):
                target_level = max(target_level, 1)

            # Level 2 triggers
            if (eye_closed_dur >= self.cfg["eye_close_seconds"]
                    or score >= self.cfg["score_alert"]
                    or half_dur >= self.cfg["half_eye_seconds_2"]):
                target_level = max(target_level, 2)

            # Level 3 triggers
            if (face_gone >= self.cfg["face_gone_seconds"]
                    or eye_closed_dur >= self.cfg["eye_close_seconds"] * 2
                    or score >= self.cfg["score_critical"]):
                target_level = max(target_level, 3)

            # Level can only go DOWN through recovery, not jump down instantly
            if target_level > self._level:
                self._level      = target_level
                self._level_since = now
            elif target_level < self._level:
                # Check recovery time
                time_in_level = now - self._level_since
                recovery_needed = (
                    self.cfg["recovery_warning"]
                    if self._level == 1 else
                    self.cfg["recovery_alert"]
                )
                if self._level < 3 and time_in_level >= recovery_needed:
                    self._level = max(0, self._level - 1)
                    self._level_since = now
                # Level 3 never auto-recovers

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

            # ── Distraction alerts (instant, independent) ─────────────────────
            for dtype in ("phone", "smoking"):
                key = f"{dtype}_detected"
                if result.get(key, False):
                    last = self._last_distraction.get(dtype, 0)
                    if now - last >= self._distraction_cooldown:
                        self._last_distraction[dtype] = now
                        ev = DistractionEvent(
                            type=dtype,
                            confidence=result.get(f"{dtype}_confidence", 1.0),
                        )
                        self._fire(self._on_distraction_cbs, ev)

            # ── Seatbelt monitoring (continuous) ──────────────────────────────
            if "seatbelt_present" in result and not result["seatbelt_present"]:
                ev = SafetyEvent(type="seatbelt", state="absent")
                self._fire(self._on_safety_cbs, ev)

            return self._level

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fire(self, callbacks: list, event) -> None:
        for cb in callbacks:
            try:
                cb(event)
            except Exception as e:
                print(f"  [SafeDrive] Callback error: {e}")

    def reset(self) -> None:
        """Manual reset — use after CRITICAL to clear state."""
        with self._lock:
            self._level           = 0
            self._prev_level      = 0
            self._level_since     = time.time()
            self._face_lost_since = None
            self._eye_closed_since = None
            self._half_eye_since  = None

    @property
    def level(self) -> int:
        return self._level

    @property
    def level_label(self) -> str:
        return LEVEL_LABELS[self._level]

    @property
    def level_color(self) -> tuple:
        return LEVEL_COLORS[self._level]