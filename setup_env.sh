#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Install additional required packages for Jetson Nano
# The following command is for Jetson with JetPack 4.6.x. Adjust if your version is different.
# You may need to find the correct wheel for your JetPack version from NVIDIA's developer forums.
wget https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# Install torchvision (must be built from source on Jetson)
sudo apt-get install -y libjpeg-dev zlib1g-dev
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
python3 setup.py install
cd ..
pip install transformers accelerate

echo "Setup complete! Activate the virtual environment with: source venv/bin/activate"
