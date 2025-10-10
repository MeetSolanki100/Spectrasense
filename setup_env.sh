#!/bin/bash

# Jetson Nano Setup Script for Vision Encoder
echo "Setting up environment for Jetson Nano..."

# Check if we're on Jetson Nano
if [[ $(uname -m) != "aarch64" ]]; then
    echo "Warning: This script is optimized for Jetson Nano (aarch64). Current architecture: $(uname -m)"
fi

# Update system packages
echo "Updating system packages..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev python3-venv
sudo apt-get install -y libjpeg-dev zlib1g-dev libpng-dev
sudo apt-get install -y libopencv-dev python3-opencv
sudo apt-get install -y build-essential cmake
sudo apt-get install -y git wget

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch for Jetson Nano
echo "Installing PyTorch for Jetson Nano..."
# For JetPack 5.x (Ubuntu 20.04)
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "Detected CUDA version: $NVCC_VERSION"
    
    if [[ $NVCC_VERSION == 11.4* ]]; then
        echo "Installing PyTorch for CUDA 11.4 (JetPack 5.0/5.1)..."
        pip install torch==1.12.0 torchvision==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu113
    elif [[ $NVCC_VERSION == 10.2* ]]; then
        echo "Installing PyTorch for CUDA 10.2 (JetPack 4.6)..."
        wget https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.10.0-cp36-cp36m-linux_aarch64.whl
        pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
        # Install torchvision from source for JetPack 4.6
        git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
        cd torchvision
        python3 setup.py install
        cd ..
        rm -rf torchvision
    else
        echo "Unsupported CUDA version. Installing CPU-only PyTorch..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
else
    echo "CUDA not found. Installing CPU-only PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "Installing other requirements..."
pip install -r requirements.txt

# Install Jetson-specific packages (only on Jetson devices)
if [[ $(uname -m) == "aarch64" ]]; then
    echo "Installing Jetson-specific packages..."
    pip install jetson-stats pynvml || echo "Warning: Failed to install Jetson-specific packages. This is normal on non-Jetson systems."
else
    echo "Skipping Jetson-specific packages (not on Jetson device)"
fi

# Set up CUDA environment variables
echo "Setting up CUDA environment variables..."
echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Create a startup script for optimal performance
echo "Creating performance optimization script..."
cat > jetson_optimize.sh << 'EOF'
#!/bin/bash
# Jetson Nano Performance Optimization

# Set maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Set GPU memory allocation
export CUDA_VISIBLE_DEVICES=0

# Optimize for inference
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

echo "Jetson Nano optimized for maximum performance"
EOF

chmod +x jetson_optimize.sh

echo "Setup complete!"
echo "To activate the virtual environment: source venv/bin/activate"
echo "To optimize performance: ./jetson_optimize.sh"
echo "To run the application: python main.py"
