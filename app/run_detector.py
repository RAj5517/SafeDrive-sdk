"""
run_detector.py
───────────────
Entry point — starts the SafeDrive real-time drowsiness detector.

Usage:
    python app/run_detector.py
    python app/run_detector.py --camera 0 --model models/drowsiness_cnn_best.pth
    python app/run_detector.py --debug
"""

import argparse
import sys
import os

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from realtime_detector import RealtimeDetector


def parse_args():
    parser = argparse.ArgumentParser(description="SafeDrive Drowsiness Detector")

    parser.add_argument("--camera",  type=int,   default=0,
                        help="Webcam device index (default: 0)")

    parser.add_argument("--model",   type=str,
                        default="models/drowsiness_cnn_best.pth",
                        help="Path to trained model weights")

    parser.add_argument("--device",  type=str,   default=None,
                        help="Torch device: 'cuda' or 'cpu' (auto-detect if not set)")

    parser.add_argument("--debug",   action="store_true",
                        help="Show extra debug info")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print("  SafeDrive — Real-Time Drowsiness Detection")
    print("=" * 50)
    print(f"  Camera:  {args.camera}")
    print(f"  Model:   {args.model}")
    print(f"  Device:  {args.device or 'auto-detect'}")
    print("=" * 50)

    if not os.path.exists(args.model):
        print(f"\n❌ Model not found: {args.model}")
        print("   Train the model first: python src/train_eye_state.py")
        sys.exit(1)

    detector = RealtimeDetector(
        model_path=args.model,
        camera_id=args.camera,
        device=args.device,
    )

    detector.run()


if __name__ == "__main__":
    main()