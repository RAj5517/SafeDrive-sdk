"""
benchmark.py
────────────
SafeDrive Pipeline Benchmark Tool

Runs both MediaPipe and YOLO pipelines on the same source
and produces a side-by-side comparison report.

Usage:
    python benchmark.py                        # webcam, 500 frames
    python benchmark.py --source webcam        # explicit webcam
    python benchmark.py --source video.mp4     # video file
    python benchmark.py --frames 1000          # more frames
    python benchmark.py --save report.txt      # save report to file
    python benchmark.py --pipeline mediapipe   # single pipeline only
    python benchmark.py --pipeline yolo        # single pipeline only

Output:
    Live side-by-side OpenCV window (optional)
    Console report table
    Optional .txt report file

Requirements:
    pip install safedrive-ai ultralytics
    (YOLO pipeline needs ultralytics)
"""

import cv2
import time
import argparse
import sys
import threading
import queue
import platform
from pathlib import Path
from collections import deque
from datetime import datetime


# ── Try imports ───────────────────────────────────────────────────────────────

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── Benchmark Config ──────────────────────────────────────────────────────────

DEFAULT_FRAMES   = 500
WARMUP_FRAMES    = 30     # discard first N frames (model warm-up)
DISPLAY_EVERY    = 5      # update console progress every N frames
WINDOW_WIDTH     = 1280   # side-by-side window width


# ── Metrics Tracker ───────────────────────────────────────────────────────────

class PipelineMetrics:
    """Collects per-frame metrics for one pipeline."""

    def __init__(self, name: str):
        self.name          = name
        self.latencies     = []
        self.fps_samples   = []
        self.eye_states    = []
        self.scores        = []
        self.perclos       = []
        self.face_found    = []
        self.alerts        = {0: 0, 1: 0, 2: 0, 3: 0}
        self.cpu_samples   = []
        self.gpu_samples   = []
        self.error_count   = 0
        self._last_time    = None

    def record(self, result: dict, latency_ms: float):
        self.latencies.append(latency_ms)
        self.eye_states.append(result.get("eye_state", "unknown"))
        self.scores.append(result.get("score", 0.0))
        self.perclos.append(result.get("perclos", 0.0))
        self.face_found.append(result.get("face_found", False))

        # Track YOLO-specific feature detections
        if not hasattr(self, '_phone'):   self._phone   = []
        if not hasattr(self, '_yawn'):    self._yawn    = []
        if not hasattr(self, '_smoking'): self._smoking = []
        self._phone.append(1 if result.get("phone_detected", False) else 0)
        self._yawn.append(1 if result.get("yawn_detected", False) else 0)
        self._smoking.append(1 if result.get("smoking_detected", False) else 0)

        if HAS_PSUTIL:
            self.cpu_samples.append(psutil.cpu_percent())
        if HAS_TORCH and torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1024 / 1024
            self.gpu_samples.append(mem)

    def avg(self, lst: list) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    def summary(self) -> dict:
        n = len(self.latencies)
        if n == 0:
            return {}

        avg_lat  = self.avg(self.latencies)
        avg_fps  = 1000.0 / avg_lat if avg_lat > 0 else 0.0
        p95_lat  = sorted(self.latencies)[int(n * 0.95)] if n > 5 else avg_lat

        eye_counts = {}
        for s in self.eye_states:
            eye_counts[s] = eye_counts.get(s, 0) + 1

        face_pct = (sum(self.face_found) / n * 100) if n > 0 else 0

        phone   = getattr(self, '_phone',   [])
        yawn    = getattr(self, '_yawn',    [])
        smoking = getattr(self, '_smoking', [])

        return {
            "frames":         n,
            "avg_latency_ms": avg_lat,
            "p95_latency_ms": p95_lat,
            "avg_fps":        avg_fps,
            "avg_score":      self.avg(self.scores),
            "avg_perclos":    self.avg(self.perclos),
            "face_detect_pct":face_pct,
            "eye_open_pct":   eye_counts.get("open",   0) / n * 100,
            "eye_half_pct":   eye_counts.get("half",   0) / n * 100,
            "eye_closed_pct": eye_counts.get("closed", 0) / n * 100,
            "avg_cpu_pct":    self.avg(self.cpu_samples)   if self.cpu_samples else None,
            "avg_gpu_mb":     self.avg(self.gpu_samples)   if self.gpu_samples else None,
            "phone_pct":      sum(phone)   / len(phone)   * 100 if phone   else 0,
            "yawn_pct":       sum(yawn)    / len(yawn)    * 100 if yawn    else 0,
            "smoking_pct":    sum(smoking) / len(smoking) * 100 if smoking else 0,
            "errors":         self.error_count,
        }


# ── Single Pipeline Runner ────────────────────────────────────────────────────

def normalize_result(result: dict, pipeline_name: str) -> dict:
    """
    Normalize pipeline output to common keys so benchmark
    works regardless of what each pipeline returns internally.

    MediaPipe returns:  eye_state, score, perclos, face_found
    YOLO returns:       detections (list), alert_level, eye_class, etc.

    We map both to the same schema:
        eye_state   → "open" | "half" | "closed" | "unknown"
        score       → float 0.0-1.0
        perclos     → float 0.0-1.0
        face_found  → bool
    """
    if pipeline_name == "mediapipe":
        # MediaPipe pipeline already uses standard keys
        return result

    # ── YOLO normalization ────────────────────────────────────────────────
    normalized = dict(result)   # keep original keys too

    # eye_state: YOLO may return eye_class, eye_state, or detections list
    if "eye_state" not in normalized:
        # Try to derive from detections
        detections = result.get("detections", [])
        eye_classes = {"eye_open": "open", "eye_half": "half", "eye_closed": "closed"}
        eye_state = "unknown"

        if isinstance(detections, list):
            for det in detections:
                cls = det.get("class", det.get("name", ""))
                if cls in eye_classes:
                    eye_state = eye_classes[cls]
                    break
        elif isinstance(detections, dict):
            for cls, mapped in eye_classes.items():
                if cls in detections:
                    eye_state = mapped
                    break

        # Also check direct keys the pipeline might set
        for key in ("eye_class", "eye_label", "eye"):
            if key in result:
                val = str(result[key]).lower()
                if "open" in val:   eye_state = "open"
                elif "half" in val: eye_state = "half"
                elif "close" in val: eye_state = "closed"

        normalized["eye_state"] = eye_state

    # face_found: YOLO detects face if any detection exists
    if "face_found" not in normalized:
        detections = result.get("detections", [])
        if isinstance(detections, list):
            normalized["face_found"] = len(detections) > 0
        elif isinstance(detections, dict):
            normalized["face_found"] = len(detections) > 0
        else:
            # If alert_level exists and is not None, face was found
            normalized["face_found"] = result.get("alert_level", None) is not None

    # score: drowsiness confidence 0.0-1.0
    if "score" not in normalized:
        for key in ("drowsy_score", "drowsiness_score", "confidence", "alert_level"):
            if key in result and result[key] is not None:
                val = float(result[key])
                # alert_level is 0-3, normalize to 0-1
                if key == "alert_level":
                    val = val / 3.0
                normalized["score"] = min(1.0, max(0.0, val))
                break
        else:
            normalized["score"] = 0.0

    # perclos: percentage eye closure over time
    if "perclos" not in normalized:
        normalized["perclos"] = result.get("perclos_score", result.get("eye_closure_pct", 0.0))

    return normalized


def run_pipeline(pipeline_name: str,
                 frames: list,
                 metrics: PipelineMetrics,
                 warmup: int = WARMUP_FRAMES) -> list:
    """
    Run one pipeline over a list of frames.
    Returns list of annotated frames.
    """
    import sys
    for path in [".", "src", "../src"]:
        if path not in sys.path:
            sys.path.insert(0, path)

    print(f"\n  [{pipeline_name.upper()}] Starting pipeline...")

    # Add sdk/ to sys.path so benchmark works from project root
    import pathlib as _pl
    _sdk = str(_pl.Path(__file__).parent / "sdk")
    if _sdk not in sys.path:
        sys.path.insert(0, _sdk)

    try:
        if pipeline_name == "mediapipe":
            from safedrive.pipelines.mediapipe_pipeline import MediaPipePipeline
            pipeline = MediaPipePipeline()
        elif pipeline_name == "yolo":
            from safedrive.pipelines.yolo_pipeline import YoloPipeline
            pipeline = YoloPipeline(
                conf            = 0.25,   # lower for benchmark (varied lighting)
                detect_phone    = True,
                detect_seatbelt = True,
                detect_smoking  = True,
                detect_yawn     = True,
            )
        else:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")

        pipeline.start()

    except Exception as e:
        print(f"  [{pipeline_name.upper()}] Failed to start: {e}")
        import traceback; traceback.print_exc()
        metrics.error_count += 1
        return []

    annotated_frames = []
    total = len(frames)

    for i, frame in enumerate(frames):
        try:
            t0 = time.perf_counter()
            result = pipeline.process_frame(frame.copy())
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000

            # Debug: print raw result dict for first 3 post-warmup frames
            import os as _os
            if i >= warmup and i < warmup + 3 and _os.environ.get("SD_DEBUG"):
                keys = list(result.keys())
                sample = {k: v for k, v in result.items() if k != "frame"}
                print("  [" + pipeline_name.upper() + "] raw keys: " + str(keys))
                print("  [" + pipeline_name.upper() + "] sample:   " + str(sample))

            if i >= warmup:
                metrics.record(normalize_result(result, pipeline_name), latency_ms)

            annotated_frames.append(result.get("frame", frame))

            # Progress
            if i % DISPLAY_EVERY == 0:
                fps = 1000.0 / latency_ms if latency_ms > 0 else 0
                bar = "█" * int((i / total) * 20) + "░" * (20 - int((i / total) * 20))
                print(f"\r  [{pipeline_name.upper()}] |{bar}| {i}/{total}  "
                      f"{latency_ms:.1f}ms  {fps:.1f}fps", end="", flush=True)

        except Exception as e:
            metrics.error_count += 1
            annotated_frames.append(frame)

    print(f"\r  [{pipeline_name.upper()}] |████████████████████| {total}/{total}  DONE        ")

    try:
        pipeline.stop()
    except Exception:
        pass

    return annotated_frames


# ── Report Generator ──────────────────────────────────────────────────────────

def generate_report(metrics_mp: PipelineMetrics,
                    metrics_yolo: PipelineMetrics,
                    source: str,
                    args) -> str:
    """Generate the benchmark comparison report string."""

    mp   = metrics_mp.summary()
    yolo = metrics_yolo.summary()
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def fmt(val, unit="", decimals=1, bold_lower=False, bold_higher=False):
        """Format a value, marking which pipeline wins."""
        if val is None:
            return "N/A"
        if isinstance(val, float):
            return f"{val:.{decimals}f}{unit}"
        return f"{val}{unit}"

    def winner(mp_val, yolo_val, lower_is_better=False):
        """Return which pipeline wins for a metric."""
        if mp_val is None or yolo_val is None:
            return "", ""
        if lower_is_better:
            mp_win   = "✅" if mp_val   <= yolo_val else "  "
            yolo_win = "✅" if yolo_val <= mp_val   else "  "
        else:
            mp_win   = "✅" if mp_val   >= yolo_val else "  "
            yolo_win = "✅" if yolo_val >= mp_val   else "  "
        return mp_win, yolo_win

    lines = []
    W = 64

    def div(char="─"):
        lines.append("  " + char * W)

    def row(label, mp_val, yolo_val, mp_w="", yolo_w=""):
        label_col = f"{label:<24}"
        mp_col    = f"{str(mp_val):<16}"
        yolo_col  = f"{str(yolo_val):<16}"
        lines.append(f"  {label_col}  {mp_w}{mp_col}  {yolo_w}{yolo_col}")

    def header(text):
        lines.append(f"\n  ── {text} {'─' * (W - len(text) - 4)}")

    lines.append("")
    div("═")
    lines.append(f"  {'SafeDrive AI — Pipeline Benchmark':^{W}}")
    div("═")
    lines.append(f"  Generated  : {now}")
    lines.append(f"  Source     : {source}")
    actual_frames = mp.get('frames', 0) or yolo.get('frames', 0)
    lines.append(f"  Frames     : {actual_frames} (after {WARMUP_FRAMES}-frame warmup)")
    lines.append(f"  System     : {platform.system()} {platform.machine()}")
    if HAS_TORCH and torch.cuda.is_available():
        lines.append(f"  GPU        : {torch.cuda.get_device_name(0)}")
    else:
        lines.append(f"  GPU        : None (CPU only)")
    div()

    # Column headers
    lines.append(f"  {'Metric':<24}  {'MediaPipe':<18}{'YOLO':<18}")
    div()

    # ── Speed ─────────────────────────────────────────────────────────────
    header("SPEED")

    mp_fps   = mp.get("avg_fps", 0)
    yolo_fps = yolo.get("avg_fps", 0)
    mw, yw   = winner(mp_fps, yolo_fps, lower_is_better=False)
    row("Avg FPS",
        f"{mp_fps:.1f} fps", f"{yolo_fps:.1f} fps", mw, yw)

    mp_lat   = mp.get("avg_latency_ms", 0)
    yolo_lat = yolo.get("avg_latency_ms", 0)
    mw, yw   = winner(mp_lat, yolo_lat, lower_is_better=True)
    row("Avg latency",
        f"{mp_lat:.1f} ms", f"{yolo_lat:.1f} ms", mw, yw)

    mp_p95   = mp.get("p95_latency_ms", 0)
    yolo_p95 = yolo.get("p95_latency_ms", 0)
    mw, yw   = winner(mp_p95, yolo_p95, lower_is_better=True)
    row("P95 latency",
        f"{mp_p95:.1f} ms", f"{yolo_p95:.1f} ms", mw, yw)

    # ── Detection ─────────────────────────────────────────────────────────
    header("DETECTION")

    mp_face   = mp.get("face_detect_pct", 0)
    yolo_face = yolo.get("face_detect_pct", 0)
    mw, yw    = winner(mp_face, yolo_face)
    row("Face detect rate",
        f"{mp_face:.1f}%", f"{yolo_face:.1f}%", mw, yw)

    row("Eye: open",
        f"{mp.get('eye_open_pct',   0):.1f}%",
        f"{yolo.get('eye_open_pct', 0):.1f}%")
    row("Eye: half",
        f"{mp.get('eye_half_pct',   0):.1f}%",
        f"{yolo.get('eye_half_pct', 0):.1f}%")
    row("Eye: closed",
        f"{mp.get('eye_closed_pct', 0):.1f}%",
        f"{yolo.get('eye_closed_pct',0):.1f}%")

    mp_score   = mp.get("avg_score", 0)
    yolo_score = yolo.get("avg_score", 0)
    row("Avg drowsy score",
        f"{mp_score:.3f}", f"{yolo_score:.3f}")

    mp_pc   = mp.get("avg_perclos", 0)
    yolo_pc = yolo.get("avg_perclos", 0)
    # YOLO-specific detection flags (from extra result keys)
    if yolo.get('frames', 0) > 0:
        row("Phone detected",    "N/A", f"{yolo.get('phone_pct', 0):.1f}%")
        row("Yawn detected",     f"{mp.get('yawn_pct',0):.1f}%",  f"{yolo.get('yawn_pct', 0):.1f}%")

    row("Avg PERCLOS",
        f"{mp_pc:.3f}", f"{yolo_pc:.3f}")

    # ── Resources ─────────────────────────────────────────────────────────
    header("RESOURCES")

    mp_cpu   = mp.get("avg_cpu_pct")
    yolo_cpu = yolo.get("avg_cpu_pct")
    if mp_cpu is not None and yolo_cpu is not None:
        mw, yw = winner(mp_cpu, yolo_cpu, lower_is_better=True)
        row("Avg CPU usage",
            f"{mp_cpu:.1f}%", f"{yolo_cpu:.1f}%", mw, yw)

    mp_gpu   = mp.get("avg_gpu_mb")
    yolo_gpu = yolo.get("avg_gpu_mb")
    if mp_gpu is not None and yolo_gpu is not None:
        mw, yw = winner(mp_gpu, yolo_gpu, lower_is_better=True)
        row("Avg GPU memory",
            f"{mp_gpu:.0f} MB", f"{yolo_gpu:.0f} MB", mw, yw)

    row("Errors", mp.get("errors", 0), yolo.get("errors", 0))

    # ── Features ──────────────────────────────────────────────────────────
    header("FEATURES")
    row("Eye state (3-class)", "✅", "✅")
    row("Yawn detection",      "✅ MAR", "✅ YOLO")
    row("Head pose",           "✅", "✅")
    row("PERCLOS tracking",    "✅", "✅")
    row("Phone detection",     "❌", "✅")
    row("Seatbelt monitoring", "❌", "✅")
    row("Cigarette detection", "❌", "✅")
    row("CPU-only support",    "✅", "⚠️ slow")

    # ── Recommendation ────────────────────────────────────────────────────
    div()
    lines.append("")

    yolo_faster = yolo_fps > mp_fps if yolo_fps and mp_fps else False
    if yolo_faster:
        lines.append("  RECOMMENDATION:")
        lines.append("    Use YOLO  → GPU available, all features needed")
        lines.append("    Use MediaPipe → CPU only, highest eye accuracy priority")
    else:
        lines.append("  RECOMMENDATION:")
        lines.append("    Use MediaPipe → both pipelines similar speed, CPU friendly")
        lines.append("    Use YOLO      → need phone/seatbelt/cigarette detection")

    lines.append("")
    div("═")
    lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SafeDrive AI — Pipeline Benchmark Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py                        webcam, 500 frames, both pipelines
  python benchmark.py --frames 1000          1000 frames
  python benchmark.py --source video.mp4     benchmark on video file
  python benchmark.py --pipeline mediapipe   single pipeline only
  python benchmark.py --save report.txt      save report to file
  python benchmark.py --no-display           headless, no OpenCV window
        """
    )
    parser.add_argument("--source",   default="webcam",
                        help="'webcam' or path to video file")
    parser.add_argument("--frames",   type=int, default=DEFAULT_FRAMES,
                        help=f"Number of frames to benchmark (default {DEFAULT_FRAMES})")
    parser.add_argument("--pipeline", default="both",
                        choices=["both", "mediapipe", "yolo"],
                        help="Which pipeline(s) to benchmark")
    parser.add_argument("--save",     default=None,
                        help="Save report to this .txt file")
    parser.add_argument("--no-display", action="store_true",
                        help="Don't show OpenCV window")
    parser.add_argument("--camera",   type=int, default=0,
                        help="Camera index (default 0)")
    parser.add_argument("--debug",    action="store_true",
                        help="Print raw result dict from each pipeline (first 3 frames)")
    args = parser.parse_args()

    print("\n  ══════════════════════════════════════════")
    print("    SafeDrive AI — Pipeline Benchmark")
    print("  ══════════════════════════════════════════")
    print(f"  Source   : {args.source}")
    print(f"  Frames   : {args.frames} (+{WARMUP_FRAMES} warmup)")
    print(f"  Pipelines: {args.pipeline}")
    if HAS_PSUTIL:
        print(f"  psutil   : ✅ (CPU tracking enabled)")
    else:
        print(f"  psutil   : ❌ (pip install psutil for CPU tracking)")
    if HAS_TORCH and torch.cuda.is_available():
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
    else:
        print(f"  GPU      : None")
    print()

    # ── Load frames ───────────────────────────────────────────────────────
    print(f"  Loading {args.frames + WARMUP_FRAMES} frames from {args.source}...")

    frames = []
    total_needed = args.frames + WARMUP_FRAMES

    if args.source == "webcam":
        cap = cv2.VideoCapture(args.camera)
        source_label = f"webcam (camera {args.camera})"
    else:
        if not Path(args.source).exists():
            print(f"  ERROR: File not found: {args.source}")
            sys.exit(1)
        cap = cv2.VideoCapture(args.source)
        source_label = args.source

    if not cap.isOpened():
        print(f"  ERROR: Cannot open source: {args.source}")
        sys.exit(1)

    while len(frames) < total_needed:
        ret, frame = cap.read()
        if not ret:
            if args.source != "webcam":
                # Loop video if not enough frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                break
        frames.append(frame)

        if len(frames) % 100 == 0:
            print(f"  Captured {len(frames)}/{total_needed} frames...", end="\r")

    cap.release()
    print(f"  Captured {len(frames)} frames.           ")

    if len(frames) < 50:
        print("  ERROR: Not enough frames captured. Exiting.")
        sys.exit(1)

    # ── Run pipelines ─────────────────────────────────────────────────────
    metrics_mp   = PipelineMetrics("mediapipe")
    metrics_yolo = PipelineMetrics("yolo")
    frames_mp    = []
    frames_yolo  = []

    if args.pipeline in ("both", "mediapipe"):
        frames_mp = run_pipeline("mediapipe", frames, metrics_mp, WARMUP_FRAMES)

    if args.pipeline in ("both", "yolo"):
        frames_yolo = run_pipeline("yolo", frames, metrics_yolo, WARMUP_FRAMES)

    # ── Generate report ───────────────────────────────────────────────────
    report = generate_report(metrics_mp, metrics_yolo, source_label, args)
    print(report)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  Report saved to: {args.save}")

    # ── Side-by-side display ──────────────────────────────────────────────
    if not args.no_display and frames_mp and frames_yolo:
        print("\n  Showing side-by-side comparison. Press Q to quit.\n")
        n = min(len(frames_mp), len(frames_yolo))
        i = 0
        while True:
            left  = frames_mp[i % n]
            right = frames_yolo[i % n]

            # Resize both to same height
            h     = 480
            left  = cv2.resize(left,  (int(left.shape[1]  * h / left.shape[0]),  h))
            right = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))

            # Add pipeline labels
            cv2.rectangle(left,  (0,0), (200, 36), (0,0,0), -1)
            cv2.rectangle(right, (0,0), (180, 36), (0,0,0), -1)
            cv2.putText(left,  "MEDIAPIPE", (10,26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100,220,255), 2)
            cv2.putText(right, "YOLO",      (10,26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100,255,100), 2)

            # Add latency
            mp_lat   = metrics_mp.latencies[min(i, len(metrics_mp.latencies)-1)] \
                       if metrics_mp.latencies else 0
            yolo_lat = metrics_yolo.latencies[min(i, len(metrics_yolo.latencies)-1)] \
                       if metrics_yolo.latencies else 0

            cv2.putText(left,  f"{mp_lat:.1f}ms",   (10, h-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
            cv2.putText(right, f"{yolo_lat:.1f}ms", (10, h-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

            # Divider
            divider = cv2.copyMakeBorder(
                left, 0, 0, 0, 4,
                cv2.BORDER_CONSTANT, value=(60,60,60)
            )
            combined = cv2.hconcat([divider, right])
            cv2.imshow("SafeDrive Benchmark — MediaPipe vs YOLO (Q to quit)", combined)

            key = cv2.waitKey(33) & 0xFF  # ~30fps playback
            if key == ord("q") or key == 27:
                break
            i += 1

        cv2.destroyAllWindows()

    print("  Benchmark complete.\n")


if __name__ == "__main__":
    main()