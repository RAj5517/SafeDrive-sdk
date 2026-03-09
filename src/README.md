# SafeDrive — Training Scripts

Scripts used to build the models behind the SafeDrive SDK.

## Eye Classifier (MobileNetV3)
- collect_eye_data.py       — webcam data collection with auto-labeling
- train_webcam_finetune.py  — domain adaptation fine-tuning on webcam frames

## YOLO Detector (YOLOv8-nano)
- collect_yolo_data.py      — collect eye/mouth training images via webcam
- merge_yolo_datasets.py    — merge self-collected + Roboflow datasets
- train_yolo.py             — YOLOv8-nano training (28,593 images, 9 classes)

## Utilities
- landmark_extractor.py     — MediaPipe face landmark wrapper
- ear_calculator.py         — EAR formula implementation
- perclos.py                — PERCLOS real-time tracker
- mobilenet_model.py        — MobileNetV3 model definition
- debug_model.py            — model diagnostic tool

Trained models: https://huggingface.co/raj5517/safedrive-model
