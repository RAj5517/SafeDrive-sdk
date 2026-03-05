"""
setup.py
────────
PyPI packaging config for safedrive-sdk.

Publish to PyPI:
    python setup.py sdist bdist_wheel
    pip install twine
    twine upload dist/*
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="safedrive-sdk",
    version="0.1.0",
    author="Your Name",
    author_email="your@email.com",
    description="Real-time driver drowsiness detection SDK",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/safedrive-sdk",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.7",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pygame>=2.5.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="drowsiness detection driver safety computer vision eye tracking",
)