"""
debug_model.py
Run from project root:
    python debug_model.py
"""
import cv2
import glob
import time
from ultralytics import YOLO

NAMES = {
    0: "eye_open",  1: "eye_half",   2: "eye_closed",
    3: "mouth_open",4: "mouth_closed",5: "phone",
    6: "cigarette", 7: "seatbelt_on", 8: "seatbelt_off",
}

LOCAL_PATH = r"runs\detect\outputs\yolo_train\yolo_safedrive\weights\best.pt"
HF_PATH    = r"C:\Users\sayan\.cache\safedrive\models\yolo_safedrive.pt"

def test_model(name, model, frame):
    print(f"\n=== {name} ===")
    r = model.predict(frame, conf=0.1, verbose=False)
    boxes = r[0].boxes
    print(f"Detections: {len(boxes)}")
    for box in boxes:
        cls = int(box.cls[0])
        cf  = float(box.conf[0])
        print(f"  {NAMES.get(cls, cls):<15} conf={cf:.3f}")
    return len(boxes)

# ── Grab webcam frame ─────────────────────────────────────────────────────────
print("Opening webcam (1 second)...")
cap = cv2.VideoCapture(0)
time.sleep(1)
ret, frame = cap.read()
cap.release()
cv2.imwrite("debug_webcam.jpg", frame)
print(f"Frame: {frame.shape}  saved as debug_webcam.jpg")

# ── Load models ───────────────────────────────────────────────────────────────
print("\nLoading models...")

try:
    local_model = YOLO(LOCAL_PATH)
    print(f"Local:  {LOCAL_PATH}  OK")
except Exception as e:
    local_model = None
    print(f"Local:  FAILED — {e}")

try:
    hf_model = YOLO(HF_PATH)
    print(f"HF:     {HF_PATH}  OK")
except Exception as e:
    hf_model = None
    print(f"HF:     FAILED — {e}")

# ── Test on webcam frame ──────────────────────────────────────────────────────
print("\n--- WEBCAM FRAME ---")
if local_model: test_model("LOCAL best.pt",        local_model, frame)
if hf_model:    test_model("HuggingFace cached",   hf_model,    frame)

# ── Test on training images ───────────────────────────────────────────────────
print("\n--- TRAINING IMAGES ---")
train_imgs = (
    glob.glob(r"data\yolo_merged\images\train\*.jpg")[:5] +
    glob.glob(r"data\yolo_webcam\images\train\*.jpg")[:5]
)

if not train_imgs:
    print("No training images found — check data/ folder")
else:
    model = local_model or hf_model
    for img_path in train_imgs:
        img = cv2.imread(img_path)
        if img is None:
            continue
        r = model.predict(img, conf=0.1, verbose=False)
        dets = [(NAMES.get(int(b.cls[0]), int(b.cls[0])), float(b.conf[0]))
                for b in r[0].boxes]
        print(f"  {img_path.split(chr(92))[-1]:<40} {len(dets)} det  "
              + "  ".join(f"{n}:{c:.2f}" for n,c in dets[:3]))

# ── File size check ───────────────────────────────────────────────────────────
import os
print("\n--- FILE SIZES ---")
for path in [LOCAL_PATH, HF_PATH]:
    if os.path.exists(path):
        mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {path.split(chr(92))[-1]:<30} {mb:.2f} MB")
    else:
        print(f"  {path.split(chr(92))[-1]:<30} NOT FOUND")