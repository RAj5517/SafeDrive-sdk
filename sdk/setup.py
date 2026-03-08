"""
setup.py
────────
PyPI package config for safedrive-ai v0.1.0

Install:
    pip install safedrive-ai

Publish:
    pip install build twine
    python -m build
    twine upload dist/*
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name            = "safedrive-ai",
    version         = "0.2.0",
    author          = "SafeDrive AI",
    description     = "Driver monitoring SDK - drowsiness, phone, seatbelt detection",
    long_description= long_description,
    long_description_content_type = "text/markdown",
    url             = "https://github.com/yourusername/safedrive-ai",

    packages        = find_packages(),
    python_requires = ">=3.10",

    install_requires = [
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "albumentations>=1.3.0",
    ],

    extras_require = {
        "yolo": [
            "ultralytics>=8.0.0",   # YOLOv8 — added in v0.2.0
        ],
        "dev": [
            "pytest",
            "twine",
            "build",
        ],
    },

    classifiers = [
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Developers",
    ],

    keywords = [
        "drowsiness detection", "driver monitoring",
        "eye tracking", "computer vision", "mediapipe",
        "yolo", "deep learning", "safety", "automotive"
    ],
)