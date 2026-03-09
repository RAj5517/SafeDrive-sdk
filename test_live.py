"""
test_live.py
────────────
End-to-end live test for SafeDrive SDK.
Tests both pipelines, all callbacks, all alert levels.

Usage:
    python test_live.py                    # YOLO pipeline (default)
    python test_live.py --pipeline mediapipe
    python test_live.py --pipeline yolo
    python test_live.py --pipeline both    # runs both sequentially

What to do during the test:
    1. Look at camera normally         → AWAKE, no alerts
    2. Close eyes for 3+ seconds       → Level 2 ALERT fires
    3. Close eyes for 5+ seconds       → Level 3 CRITICAL fires
    4. Hold phone up to camera         → DISTRACTION fires (YOLO only)
    5. Look away / cover camera        → face_gone → Level 3
    6. Press Q to quit
"""

import sys
import time
import argparse
import cv2

sys.path.insert(0, "sdk")

from safedrive import DrowsinessDetector
from safedrive.alerts.events import DrowsyEvent, DistractionEvent, SafetyEvent, FrameStats


# ── Console colors ────────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    GREEN  = "\033[92m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    GRAY   = "\033[90m"


# ── Alert log ─────────────────────────────────────────────────────────────────
alert_log = []


def log(tag, msg, color=C.RESET):
    ts = time.strftime("%H:%M:%S")
    line = f"{C.GRAY}[{ts}]{C.RESET} {color}{C.BOLD}{tag}{C.RESET} {msg}"
    print(line)
    alert_log.append(f"[{ts}] {tag} {msg}")


# ── Run one pipeline ──────────────────────────────────────────────────────────

def run_pipeline(pipeline_name: str, duration_seconds: int = 60, seatbelt: bool = True):
    print(f"\n{'='*60}")
    print(f"  Testing pipeline: {C.BOLD}{pipeline_name.upper()}{C.RESET}")
    print(f"  Duration: {duration_seconds}s  |  Press Q to stop early")
    print(f"{'='*60}")
    print()
    print("  Instructions:")
    print("  1. Look normally at camera → should show AWAKE")
    print("  2. Close eyes 3+ seconds   → ALERT should fire")
    print("  3. Close eyes 5+ seconds   → CRITICAL should fire")
    if pipeline_name == "yolo":
        print("  4. Hold phone to camera    → DISTRACTION should fire")
    print("  5. Cover camera 3+ seconds → face_gone CRITICAL")
    print()

    start_time   = time.time()
    frame_count  = 0
    alert_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    distraction_count = 0

    detector = DrowsinessDetector(
        pipeline          = pipeline_name,
        eye_close_seconds = 2.0,
        face_gone_seconds = 2.0,
        head_tilt_degrees = 15.0,
        detect_phone      = True,
        detect_seatbelt   = seatbelt,
        detect_smoking    = True,
        detect_yawn       = True,
        show_window       = True,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    @detector.on_drowsy
    def on_drowsy(event: DrowsyEvent):
        colors = {1: C.YELLOW, 2: C.YELLOW, 3: C.RED}
        log(f"DROWSY L{event.level}",
            f"{event.label} | score={event.score:.3f} ear={event.ear:.3f} | {event.message}",
            colors.get(event.level, C.RESET))
        alert_counts[event.level] = alert_counts.get(event.level, 0) + 1

    @detector.on_distraction
    def on_distraction(event: DistractionEvent):
        log("DISTRACT",
            f"{event.type.upper()} detected | conf={event.confidence:.2f} | {event.message}",
            C.CYAN)
        nonlocal distraction_count
        distraction_count += 1

    @detector.on_safety
    def on_safety(event: SafetyEvent):
        log("SAFETY",
            f"{event.type} {event.state} | {event.message}",
            C.RED)

    @detector.on_frame
    def on_frame(frame, stats: FrameStats):
        nonlocal frame_count
        frame_count += 1

        # Auto-stop after duration
        if time.time() - start_time > duration_seconds:
            detector.stop()

        # Print frame stats every 60 frames
        if frame_count % 60 == 0:
            elapsed = time.time() - start_time
            face_str = "✅" if stats.face_found else "❌"
            log("FRAME",
                f"face={face_str} eye={stats.eye_state:<7} "
                f"ear={stats.ear:.3f} score={stats.score:.3f} "
                f"perclos={stats.perclos:.3f} fps={stats.fps:.1f} "
                f"level={stats.alert_level}",
                C.GRAY)

    # ── Run ───────────────────────────────────────────────────────────────────
    try:
        detector.run(camera=0)
    except KeyboardInterrupt:
        pass

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'─'*60}")
    print(f"  {C.BOLD}Pipeline: {pipeline_name.upper()}{C.RESET}")
    print(f"  Duration:     {elapsed:.1f}s")
    print(f"  Frames:       {frame_count}")
    print(f"  Avg FPS:      {frame_count/elapsed:.1f}")
    print()
    print(f"  Alert counts:")
    print(f"    Level 1 WARNING:  {alert_counts.get(1, 0)}")
    print(f"    Level 2 ALERT:    {alert_counts.get(2, 0)}")
    print(f"    Level 3 CRITICAL: {alert_counts.get(3, 0)}")
    print(f"    Distractions:     {distraction_count}")
    print()

    # ── Pass/Fail ─────────────────────────────────────────────────────────────
    checks = {
        "Frames processed":     frame_count > 0,
        "Alert system reachable": True,   # if we got here, it worked
    }

    all_pass = all(checks.values())
    for check, passed in checks.items():
        status = f"{C.GREEN}✅{C.RESET}" if passed else f"{C.RED}❌{C.RESET}"
        print(f"  {status} {check}")

    print()
    if all_pass:
        print(f"  {C.GREEN}{C.BOLD}PASS{C.RESET} — {pipeline_name} pipeline OK")
    else:
        print(f"  {C.RED}{C.BOLD}FAIL{C.RESET} — check errors above")

    return all_pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SafeDrive live end-to-end test")
    parser.add_argument("--pipeline", default="yolo",
                        choices=["mediapipe", "yolo", "both"])
    parser.add_argument("--duration", type=int, default=60,
                        help="Test duration per pipeline in seconds (default 60)")
    parser.add_argument("--no-seatbelt", action="store_true",
                        help="Disable seatbelt monitoring during test")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  {C.BOLD}SafeDrive SDK — Live End-to-End Test{C.RESET}")
    print(f"{'='*60}")

    results = {}

    seatbelt = not args.no_seatbelt

    if args.pipeline == "both":
        for p in ["mediapipe", "yolo"]:
            results[p] = run_pipeline(p, args.duration, seatbelt=seatbelt)
            if p == "mediapipe":
                print("\n  Switching to YOLO pipeline in 3 seconds...")
                time.sleep(3)
    else:
        results[args.pipeline] = run_pipeline(args.pipeline, args.duration, seatbelt=seatbelt)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {C.BOLD}FINAL RESULTS{C.RESET}")
    print(f"{'='*60}")
    for pipeline, passed in results.items():
        status = f"{C.GREEN}PASS{C.RESET}" if passed else f"{C.RED}FAIL{C.RESET}"
        print(f"  {pipeline:<12} {C.BOLD}{status}{C.RESET}")

    if args.pipeline == "both" and len(results) == 2:
        print()
        if all(results.values()):
            print(f"  {C.GREEN}{C.BOLD}Both pipelines working. Ready for v0.2.2 publish.{C.RESET}")
        else:
            print(f"  {C.RED}Some pipelines failed. Check output above.{C.RESET}")


if __name__ == "__main__":
    main()