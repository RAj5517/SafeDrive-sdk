"""
detector.py
───────────
Unified DrowsinessDetector interface for SafeDrive SDK.
User-facing class — routes to correct pipeline internally.

Usage:
    from safedrive import DrowsinessDetector

    detector = DrowsinessDetector(pipeline="mediapipe")

    @detector.on_drowsy
    def handle(event):
        print(f"Level {event.level}: {event.message}")

    detector.run(camera=0)
"""

import cv2
import queue
import threading
import time
from typing import Callable, Optional

from .pipelines.mediapipe_pipeline import MediaPipePipeline
from .alerts.alert_system import AlertSystem, LEVEL_COLORS, LEVEL_LABELS
from .alerts.events import FrameStats


class DrowsinessDetector:
    """
    SafeDrive unified drowsiness detector.

    Args:
        pipeline:          "mediapipe" (default) or "yolo" (v0.2.0)
        model_path:        path to .pth model weights
        device:            "cuda" or "cpu" (auto-detected if None)
        alert_mode:        "context_aware" (default) or "strict"
        eye_close_seconds: seconds of closed eyes → Level 2 alert (default 2.0)
        face_gone_seconds: seconds face out of frame → Level 3 (default 2.0)
        head_tilt_degrees: degrees of head tilt → Level 1 (default 15.0)
        detect_phone:      enable phone distraction alert (default True)
        detect_seatbelt:   enable seatbelt monitoring (default True)
        detect_yawn:       enable yawn detection (default True, v0.2.0)
        detect_smoking:    enable smoking detection (default True, v0.2.0)
        show_window:       show OpenCV window (default True)
        ear_weight:        EAR contribution to fusion score (default 0.5)
        cnn_weight:        CNN contribution to fusion score (default 0.5)
    """

    def __init__(self,
                 pipeline:          str   = "mediapipe",
                 model_path:        str   = None,
                 device:            str   = None,
                 alert_mode:        str   = "context_aware",
                 eye_close_seconds: float = 2.0,
                 face_gone_seconds: float = 2.0,
                 head_tilt_degrees: float = 15.0,
                 detect_phone:      bool  = True,
                 detect_seatbelt:   bool  = True,
                 detect_yawn:       bool  = True,
                 detect_smoking:    bool  = True,
                 show_window:       bool  = True,
                 ear_weight:        float = 0.5,
                 cnn_weight:        float = 0.5):

        self._pipeline_name = pipeline
        self._show_window   = show_window
        self._running       = False
        self._frame_queue   = queue.Queue(maxsize=2)
        self._stop_event    = threading.Event()

        # ── Init pipeline ─────────────────────────────────────────────────────
        if pipeline == "mediapipe":
            self._pipeline = MediaPipePipeline(
                model_path  = model_path,
                device      = device,
                ear_weight  = ear_weight,
                cnn_weight  = cnn_weight,
            )
        elif pipeline == "yolo":
            from .pipelines.yolo_pipeline import YoloPipeline
            self._pipeline = YoloPipeline(
                model_path      = model_path,
                device          = device or "cuda",
                detect_phone    = detect_phone,
                detect_seatbelt = detect_seatbelt,
                detect_smoking  = detect_smoking,
                detect_yawn     = detect_yawn,
            )
        else:
            raise ValueError(f"Unknown pipeline: '{pipeline}'. "
                             f"Options: 'mediapipe', 'yolo'")

        # ── Init alert system ─────────────────────────────────────────────────
        self._alerts = AlertSystem(config={
            "eye_close_seconds": eye_close_seconds,
            "face_gone_seconds": face_gone_seconds,
            "head_tilt_degrees": head_tilt_degrees,
            "detect_phone":      detect_phone,
            "detect_seatbelt":   detect_seatbelt,
            "detect_smoking":    detect_smoking,
            "detect_yawn":       detect_yawn,
        })

        # ── Feature flags (for HUD display) ──────────────────────────────────
        self._detect_phone    = detect_phone
        self._detect_seatbelt = detect_seatbelt
        self._detect_smoking  = detect_smoking
        self._detect_yawn     = detect_yawn

        # ── Frame callback ────────────────────────────────────────────────────
        self._on_frame_cbs: list[Callable] = []

        print(f"SafeDrive v0.2.2 | pipeline={pipeline} | "
              f"phone={detect_phone} seatbelt={detect_seatbelt} "
              f"smoking={detect_smoking} yawn={detect_yawn}")

    # ── Callback decorators ───────────────────────────────────────────────────

    def on_drowsy(self, fn: Callable) -> Callable:
        """
        Register callback for drowsiness level changes.

        @detector.on_drowsy
        def handle(event: DrowsyEvent):
            print(event.level, event.message)
        """
        self._alerts.on_drowsy(fn)
        return fn

    def on_distraction(self, fn: Callable) -> Callable:
        """
        Register callback for phone / smoking detection.

        @detector.on_distraction
        def handle(event: DistractionEvent):
            print(event.type, event.message)
        """
        self._alerts.on_distraction(fn)
        return fn

    def on_safety(self, fn: Callable) -> Callable:
        """
        Register callback for seatbelt events.

        @detector.on_safety
        def handle(event: SafetyEvent):
            print(event.message)
        """
        self._alerts.on_safety(fn)
        return fn

    def on_frame(self, fn: Callable) -> Callable:
        """
        Register callback called every frame with FrameStats.

        @detector.on_frame
        def handle(frame, stats: FrameStats):
            cv2.imshow("custom", frame)
        """
        self._on_frame_cbs.append(fn)
        return fn

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self, camera: int = 0, fps_target: int = 30) -> None:
        """
        Start the detection loop (blocking).
        Press Q in the window to quit.

        Args:
            camera:     webcam device index (default 0)
            fps_target: target FPS (default 30)
        """
        self._pipeline.start()

        cap = cv2.VideoCapture(camera)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera}")

        print(f"Camera opened. Press Q to quit.")

        # Capture thread
        t = threading.Thread(
            target=self._capture_thread,
            args=(cap,), daemon=True
        )
        t.start()

        import torch
        try:
            while not self._stop_event.is_set():
                try:
                    frame = self._frame_queue.get(timeout=1.0)
                except queue.Empty:
                    break

                with torch.no_grad():
                    result = self._pipeline.process_frame(frame)

                # Update alert system
                alert_level = self._alerts.update(result)

                # Build FrameStats
                stats = FrameStats(
                    ear         = result["ear"],
                    eye_state   = result["eye_state"],
                    cnn_prob    = result["cnn_prob"],
                    score       = result["score"],
                    perclos     = result["perclos"],
                    fps         = result["fps"],
                    face_found  = result["face_found"],
                    alert_level = alert_level,
                    pipeline    = self._pipeline_name,
                )

                annotated = result["frame"]
                annotated = self._draw_alert_overlay(annotated, alert_level, stats)

                # Fire frame callbacks
                for cb in self._on_frame_cbs:
                    try:
                        cb(annotated, stats)
                    except Exception as e:
                        print(f"  [SafeDrive] Frame callback error: {e}")

                if self._show_window:
                    cv2.imshow("SafeDrive", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        finally:
            self._stop_event.set()
            cap.release()
            cv2.destroyAllWindows()
            self._pipeline.stop()
            print("SafeDrive stopped.")

    def stop(self) -> None:
        """Stop the detection loop programmatically."""
        self._stop_event.set()

    def reset_alert(self) -> None:
        """Manually reset CRITICAL alert state."""
        self._alerts.reset()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _capture_thread(self, cap) -> None:
        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def _draw_alert_overlay(self, frame, level: int, stats: FrameStats):
        """Draw alert level overlay on frame."""
        import cv2
        font  = cv2.FONT_HERSHEY_SIMPLEX
        h, w  = frame.shape[:2]
        color = LEVEL_COLORS[level]
        label = LEVEL_LABELS[level]

        # Stats panel top left
        lines = [
            (f"Eye:     {stats.eye_state.upper()}",   color if stats.eye_state == "closed" else (255,255,255)),
            (f"EAR:     {stats.ear:.3f}",              (255,255,255)),
            (f"CNN:     {stats.cnn_prob:.2f}",         (200,200,255)),
            (f"Score:   {stats.score:.3f}",            (255,200,100)),
            (f"PERCLOS: {stats.perclos:.2f}",          (255,255,0)),
            (f"FPS:     {stats.fps:.1f}",              (180,180,180)),
            (f"Pipeline:{stats.pipeline}",             (180,180,180)),
        ]
        for i, (text, col) in enumerate(lines):
            cv2.putText(frame, text, (10, 35 + i*28), font, 0.58, col, 2)

        # Alert banner at bottom
        if level > 0:
            messages = {
                1: "Stay Alert",
                2: "Wake Up!",
                3: "DANGER — Pull Over Now",
            }
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h-55), (w, h), color, -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cv2.putText(frame, f"[{label}]  {messages[level]}",
                        (w//2 - 180, h-20), font, 0.85, (255,255,255), 2)

        return frame