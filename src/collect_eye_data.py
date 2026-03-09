"""
collect_eye_data.py
───────────────────
Collects 3-class labeled eye crop dataset from your webcam.
Uses existing MediaPipe pipeline to extract eye ROIs.

Classes:
    open   → eyes fully open (natural driving state)
    half   → eyes half open  (drowsy, squinting, slow blink)
    closed → eyes fully closed (microsleep)

Usage:
    python collect_eye_data.py

Output structure:
    data/webcam/
        open/     → frame_000001_L.png, frame_000001_R.png ...
        half/     → frame_000001_L.png ...
        closed/   → frame_000001_L.png ...
        session_log.txt

Controls:
    SPACE  → cycle: standby → open → half → closed → done
    Q      → quit and save summary
    P      → pause / resume

HOW TO RECORD HALF-OPEN EYES:
    - Look slightly downward like you're nodding off
    - Let eyelids drop halfway — not fully closed, not fully open
    - Squint like you're tired or in bright light
    - Slow blink and hold at the midpoint
    - EAR target: 0.12 – 0.22  (shown live on screen)
"""

import cv2
import numpy as np
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# ── Add project src to path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))
from landmark_extractor import LandmarkExtractor
from eye_extractor import extract_both_eyes

# ── Config ───────────────────────────────────────────────────────────────────
TARGET_PER_CLASS  = 5000
SAVE_EVERY_N      = 2

# EAR quality gates per class — only save frames in this EAR range
EAR_GATES = {
    "open":   (0.21, 1.00),   # fully open
    "half":   (0.12, 0.22),   # drowsy / half-open  ← overlap intentional
    "closed": (0.00, 0.18),   # fully closed
}

CLASS_ORDER = ["open", "half", "closed"]

OUTPUT_DIR = Path("data/webcam")
FONT       = cv2.FONT_HERSHEY_SIMPLEX

# Colors
GREEN  = (0,  255,   0)
RED    = (0,    0, 255)
YELLOW = (0,  220, 255)
ORANGE = (0,  140, 255)
WHITE  = (255, 255, 255)
BLACK  = (0,    0,   0)
GRAY   = (160, 160, 160)

CLASS_COLORS = {"open": GREEN, "half": ORANGE, "closed": RED}

CLASS_INSTRUCTIONS = {
    "open":   "Eyes FULLY OPEN. Look naturally at screen.",
    "half":   "Eyes HALF OPEN. Squint / look drowsy / look slightly down.",
    "closed": "Eyes FULLY CLOSED. Stay still.",
}


def compute_ear(eye_points: np.ndarray) -> float:
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C + 1e-6)


def ear_in_gate(ear: float, state: str) -> bool:
    lo, hi = EAR_GATES[state]
    return lo <= ear <= hi


def draw_panel(frame, state, counts, ear, paused):
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 135), BLACK, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Main state label
    if paused:
        cv2.putText(frame, "PAUSED", (15, 35), FONT, 0.85, GRAY, 2)
    elif state == "standby":
        cv2.putText(frame, "STANDBY  —  Press SPACE to begin",
                    (15, 35), FONT, 0.85, YELLOW, 2)
    elif state == "done":
        cv2.putText(frame, "ALL DONE!  Press Q to exit",
                    (15, 35), FONT, 0.85, GREEN, 2)
    else:
        color = CLASS_COLORS[state]
        cv2.putText(frame, f"RECORDING  —  {state.upper()} eyes",
                    (15, 35), FONT, 0.85, color, 2)
        cv2.putText(frame, CLASS_INSTRUCTIONS[state],
                    (15, 58), FONT, 0.50, WHITE, 1)

    # EAR indicator (top right)
    if state in EAR_GATES:
        gate_ok  = ear_in_gate(ear, state)
        gate_col = GREEN if gate_ok else RED
        lo, hi   = EAR_GATES[state]
        status   = "IN RANGE" if gate_ok else f"NEED {lo:.2f}-{hi:.2f}"
        cv2.putText(frame, f"EAR {ear:.3f}  {status}",
                    (w - 290, 35), FONT, 0.55, gate_col, 1)
    else:
        cv2.putText(frame, f"EAR {ear:.3f}",
                    (w - 180, 35), FONT, 0.55, WHITE, 1)

    # Per-class progress bars
    col_w = w // 3
    for i, cls in enumerate(CLASS_ORDER):
        pct    = min(counts[cls] / TARGET_PER_CLASS * 100, 100)
        col    = CLASS_COLORS[cls]
        bar_x  = i * col_w + 15
        bar_bw = col_w - 25

        # Highlight active class
        if cls == state:
            cv2.rectangle(frame, (i * col_w, 68),
                          ((i + 1) * col_w - 2, 130), col, 1)

        cv2.putText(frame,
                    f"{cls.upper()}: {counts[cls]:,}/{TARGET_PER_CLASS}",
                    (bar_x, 84), FONT, 0.48, col, 1)

        # Progress bar
        cv2.rectangle(frame, (bar_x, 92), (bar_x + bar_bw, 108), GRAY, 1)
        fill = int(bar_bw * pct / 100)
        if fill > 0:
            cv2.rectangle(frame, (bar_x, 92),
                          (bar_x + fill, 108), col, -1)
        cv2.putText(frame, f"{pct:.0f}%",
                    (bar_x, 122), FONT, 0.40, col, 1)

    # Controls hint
    cv2.putText(frame,
                "SPACE = next class   P = pause/resume   Q = quit",
                (15, h - 12), FONT, 0.42, GRAY, 1)

    return frame


def show_intro(cap):
    lines = [
        ("SAFE DRIVE  —  3-Class Eye Data Collector", WHITE),
        ("", WHITE),
        ("3 recording sessions:", YELLOW),
        ("  1. OPEN   eyes  (~3 min)  —  look naturally at screen", GREEN),
        ("  2. HALF   eyes  (~3 min)  —  squint / drowsy expression", ORANGE),
        ("  3. CLOSED eyes  (~3 min)  —  close eyes fully", RED),
        ("", WHITE),
        ("Half-open tips:", YELLOW),
        ("  Look slightly downward like nodding off", WHITE),
        ("  Let eyelids droop halfway", WHITE),
        ("  Squint like tired or bright light", WHITE),
        ("  EAR 0.12-0.22 shown live on screen", WHITE),
        ("", WHITE),
        ("EAR gate filters bad frames automatically.", GRAY),
        ("No need to worry — just act natural for each state.", GRAY),
        ("", WHITE),
        ("Press ENTER to begin  /  Q to quit", GREEN),
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame   = cv2.flip(frame, 1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0),
                      (frame.shape[1], frame.shape[0]), BLACK, -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        y = 40
        for msg, color in lines:
            cv2.putText(frame, msg, (30, y), FONT, 0.58, color, 1)
            y += 30

        cv2.imshow("Safe Drive — Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            return True
        if key == ord("q"):
            return False
    return False


def main():
    # ── Setup dirs ────────────────────────────────────────────────────────
    for cls in CLASS_ORDER:
        (OUTPUT_DIR / cls).mkdir(parents=True, exist_ok=True)

    # ── Count existing files (resume support) ─────────────────────────────
    counts = {
        cls: len(list((OUTPUT_DIR / cls).glob("*.png")))
        for cls in CLASS_ORDER
    }
    print("Existing data:")
    for cls in CLASS_ORDER:
        print(f"  {cls:8s}: {counts[cls]:,} / {TARGET_PER_CLASS}")

    # Determine where to resume
    class_idx = 0
    for i, cls in enumerate(CLASS_ORDER):
        if counts[cls] >= TARGET_PER_CLASS:
            class_idx = i + 1
        else:
            break

    if class_idx >= len(CLASS_ORDER):
        print("\nAll classes already complete! Delete data/webcam/ to re-record.")
        return

    # ── Init ──────────────────────────────────────────────────────────────
    extractor = LandmarkExtractor()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    if not show_intro(cap):
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        return

    # ── State ─────────────────────────────────────────────────────────────
    state              = "standby"
    paused             = False
    frame_idx          = 0
    last_key_time      = 0
    session_start      = time.time()
    saved_session      = {cls: 0 for cls in CLASS_ORDER}
    skipped_ear        = {cls: 0 for cls in CLASS_ORDER}
    log_lines          = [f"Session: {datetime.now().isoformat()}"]

    print(f"\nResuming from class index {class_idx}: {CLASS_ORDER[class_idx]}")
    print("Press SPACE to start recording")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame     = cv2.flip(frame, 1)
        frame_idx += 1

        data = extractor.extract(frame)
        ear  = 0.0

        if data:
            ear = (compute_ear(data["left_eye"]) +
                   compute_ear(data["right_eye"])) / 2.0

            # ── Save ──────────────────────────────────────────────────────
            if (not paused
                    and state in CLASS_ORDER
                    and frame_idx % SAVE_EVERY_N == 0
                    and counts[state] < TARGET_PER_CLASS):

                if ear_in_gate(ear, state):
                    eyes = extract_both_eyes(
                        frame, data["left_eye"], data["right_eye"])

                    for side, roi in [("L", eyes["left"]),
                                      ("R", eyes["right"])]:
                        if roi is not None and counts[state] < TARGET_PER_CLASS:
                            img_u8 = (roi * 255).astype(np.uint8)
                            fname  = (OUTPUT_DIR / state /
                                      f"frame_{counts[state]:06d}_{side}.png")
                            cv2.imwrite(str(fname), img_u8)
                            counts[state]         += 1
                            saved_session[state]  += 1
                else:
                    if state in CLASS_ORDER:
                        skipped_ear[state] += 1

        # ── Auto-advance when class complete ──────────────────────────────
        if state in CLASS_ORDER and counts[state] >= TARGET_PER_CLASS:
            log_lines.append(f"{state} complete: {counts[state]}")
            class_idx += 1

            if class_idx < len(CLASS_ORDER):
                state     = "standby"
                next_cls  = CLASS_ORDER[class_idx]
                print(f"\n✅ {CLASS_ORDER[class_idx-1].upper()} complete!")
                print(f"Press SPACE to start {next_cls.upper()} recording")
            else:
                state = "done"
                print("\n🎉 ALL 3 CLASSES COMPLETE! Press Q to exit.")

        # ── HUD ───────────────────────────────────────────────────────────
        frame = draw_panel(frame, state, counts, ear, paused)
        if state == "done":
            cv2.putText(frame, "DATASET COMPLETE!  Press Q",
                        (frame.shape[1]//2 - 200, frame.shape[0]//2),
                        FONT, 1.1, GREEN, 3)

        cv2.imshow("Safe Drive — Data Collector", frame)

        # ── Keys ──────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        now = time.time()

        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
            print("PAUSED" if paused else "RESUMED")
        elif key == ord(" ") and (now - last_key_time) > 0.5:
            last_key_time = now
            if state == "done":
                break
            elif state == "standby":
                if class_idx < len(CLASS_ORDER):
                    state  = CLASS_ORDER[class_idx]
                    needed = TARGET_PER_CLASS - counts[state]
                    print(f"\n🔴 Recording {state.upper()} "
                          f"(need {needed:,} more)")
            elif state in CLASS_ORDER:
                state = "standby"
                print(f"⏸  Paused. {state}: {counts.get(state,0):,} saved.")

    # ── Cleanup + summary ─────────────────────────────────────────────────
    elapsed = time.time() - session_start
    cap.release()
    cv2.destroyAllWindows()
    extractor.close()

    total   = sum(counts.values())
    summary = f"""
═══════════════════════════════════════════════════
  DATA COLLECTION SUMMARY
═══════════════════════════════════════════════════
  Session duration  : {elapsed/60:.1f} min

  OPEN   saved : {counts['open']:,}   (+{saved_session['open']:,} this session)
  HALF   saved : {counts['half']:,}   (+{saved_session['half']:,} this session)
  CLOSED saved : {counts['closed']:,}   (+{saved_session['closed']:,} this session)
  Total frames : {total:,}

  EAR-filtered : open={skipped_ear['open']:,}  half={skipped_ear['half']:,}  closed={skipped_ear['closed']:,}
  (filtered frames = good quality gate working)

  Output: {OUTPUT_DIR.resolve()}
═══════════════════════════════════════════════════
  NEXT STEP: python train_webcam_finetune.py
═══════════════════════════════════════════════════
"""
    print(summary)
    log_lines.append(f"Ended: {datetime.now().isoformat()}")
    log_lines.append(summary)

    with open(OUTPUT_DIR / "session_log.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()