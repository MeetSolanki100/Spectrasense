#!/bin/bash
#
# This script installs the required dependencies for running the face recognition 
# application on a NVIDIA Jetson device.
#
# It installs PyTorch, TorchVision, and other libraries that require specific builds
# for the Jetson's ARM architecture.

set -e

# 1. Install system-level dependencies
echo "[INFO] Installing system-level dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential cmake libopenblas-dev libjpeg-dev libpython3-dev python3-pip

# 2. Install PyTorch and TorchVision
echo "[INFO] Installing PyTorch and TorchVision for Jetson..."
# These URLs are for JetPack 4.x. You may need to find different wheels for other versions.
# See: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
wget https://nvidia.box.com/shared/static/3dhrj9yulw06k23d8m62i8a5s7i5xht8.whl -O torchvision-0.9.0-cp36-cp36m-linux_aarch64.whl

sudo apt-get install -y libopenblas-base libopenmpi-dev
pip3 install Cython
pip3 install numpy torch-1.8.0-cp36-cp36m-linux_aarch64.whl
pip3 install torchvision-0.9.0-cp36-cp36m-linux_aarch64.whl

# Clean up downloaded wheel files
rm torch-1.8.0-cp36-cp36m-linux_aarch64.whl torchvision-0.9.0-cp36-cp36m-linux_aarch64.whl

# 3. Install dlib and face_recognition
echo "[INFO] Installing dlib and face_recognition..."
# This may take a very long time as dlib is built from source.
pip3 install dlib
pip3 install face_recognition

# 4. Install OpenCV
echo "[INFO] Installing OpenCV..."
pip3 install opencv-python

# 5. Install remaining packages from requirements.txt
echo "[INFO] Installing remaining Python packages..."
pip3 install -r requirements.txt

echo "[SUCCESS] Setup complete. You can now run the application."
