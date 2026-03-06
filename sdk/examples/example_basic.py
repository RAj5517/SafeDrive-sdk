"""
example_basic.py
────────────────
Basic SafeDrive SDK usage example.

Run:
    python examples/example_basic.py
"""

from safedrive import DrowsinessDetector, DrowsyEvent, DistractionEvent

# ── Create detector ───────────────────────────────────────────────────────────
detector = DrowsinessDetector(
    pipeline          = "mediapipe",
    eye_close_seconds = 2.0,    # seconds closed eyes → Level 2
    face_gone_seconds = 2.0,    # seconds face gone   → Level 3
    show_window       = True,
)

# ── Register callbacks ────────────────────────────────────────────────────────

@detector.on_drowsy
def on_drowsy(event: DrowsyEvent):
    print(f"\n[DROWSY] Level {event.level} — {event.message}")
    print(f"         EAR={event.ear:.3f}  score={event.score:.3f}")

    if event.level == 3:
        # Critical — could trigger external alarm, SMS, fleet alert etc
        print("[CRITICAL] Triggering emergency protocol!")

@detector.on_distraction
def on_distraction(event: DistractionEvent):
    # Fires instantly when phone or smoking detected
    print(f"\n[DISTRACTION] {event.type.upper()} — {event.message}")

@detector.on_safety
def on_safety(event):
    print(f"\n[SAFETY] {event.message}")

# ── Start ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    detector.run(camera=0)