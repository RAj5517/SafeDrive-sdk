"""
setup.py
────────
PyPI package config for safedrive-ai

Install:
    pip install safedrive-ai
    pip install safedrive-ai[yolo]

Publish:
    pip install build twine
    python -m build
    twine upload dist/*
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name             = "safedrive-ai",
    version          = "0.2.2",
    author           = "Sayan Raj",
    author_email     = "",
    description      = "Real-time driver monitoring SDK — drowsiness, phone, seatbelt and smoking detection",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url              = "https://github.com/RAj5517/SafeDrive-sdk",

    project_urls = {
        "Bug Tracker": "https://github.com/RAj5517/SafeDrive-sdk/issues",
        "Models":      "https://huggingface.co/raj5517/safedrive-model",
        "PyPI":        "https://pypi.org/project/safedrive-ai/",
    },

    packages        = find_packages(),
    python_requires = ">=3.10",

    install_requires = [
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "huggingface_hub>=0.20.0",
    ],

    extras_require = {
        "yolo": [
            "ultralytics>=8.0.0",
        ],
        "dev": [
            "pytest",
            "twine",
            "build",
        ],
    },

    license = "MIT",

    classifiers = [
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],

    keywords = [
        "drowsiness detection", "driver monitoring",
        "eye tracking", "computer vision", "mediapipe",
        "yolo", "yolov8", "deep learning", "safety",
        "automotive", "PERCLOS", "fatigue detection",
    ],
)