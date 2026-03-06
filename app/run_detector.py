"""
run_detector.py
───────────────
Entry point — starts SafeDrive real-time drowsiness detection.

Default model: MobileNetV3 (97.99% accuracy)
Fallback:      Custom CNN  (96.74% accuracy)

Usage:
    python app/run_detector.py
    python app/run_detector.py --camera 1
    python app/run_detector.py --model models/drowsiness_cnn_best.pth
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from realtime_detector import RealtimeDetector


def parse_args():
    parser = argparse.ArgumentParser(
        description="SafeDrive — Real-Time Drowsiness Detector"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Webcam device index (default: 0)"
    )
    parser.add_argument(
        "--model", type=str,
        default="models/mobilenet_best.pth",   # MobileNetV3 by default
        help="Path to model weights (.pth)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="'cuda' or 'cpu' (auto-detect if not set)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Check model exists
    if not os.path.exists(args.model):
        print(f"\n❌ Model not found: {args.model}")

        # Try fallback to custom CNN
        fallback = "models/drowsiness_cnn_best.pth"
        if os.path.exists(fallback):
            print(f"   Falling back to: {fallback}")
            args.model = fallback
        else:
            print("   Train a model first:")
            print("   python src/train_mobilenet.py")
            sys.exit(1)

    detector = RealtimeDetector(
        model_path=args.model,
        camera_id=args.camera,
        device=args.device,
    )
    detector.run()


if __name__ == "__main__":
    main()