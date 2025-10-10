#!/bin/bash

# CUDA 12.6 Specific Setup Script for Jetson Nano
echo "Setting up Vision Encoder for CUDA 12.6 (JetPack 6.x)..."

# Check if we're on Jetson Nano
if [[ $(uname -m) != "aarch64" ]]; then
    echo "Error: This script is for Jetson Nano (aarch64) only."
    echo "Current architecture: $(uname -m)"
    exit 1
fi

# Check CUDA version
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "Detected CUDA version: $NVCC_VERSION"
    
    if [[ ! $NVCC_VERSION == 12.* ]]; then
        echo "Warning: This script is optimized for CUDA 12.x, but detected CUDA $NVCC_VERSION"
        echo "Consider using setup_env.sh instead for automatic version detection"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo "Error: CUDA not found. Please ensure JetPack 6.x is properly installed."
    exit 1
fi

# Update system packages
echo "Updating system packages..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev python3-venv
sudo apt-get install -y libjpeg-dev zlib1g-dev libpng-dev
sudo apt-get install -y libopencv-dev python3-opencv
sudo apt-get install -y build-essential cmake git wget

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

# Install PyTorch for CUDA 12.x
echo "Installing PyTorch for CUDA 12.x..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch installation
echo "Verifying PyTorch installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"

# Install other requirements
echo "Installing other requirements..."
pip install -r requirements.txt

# Install Jetson-specific packages
echo "Installing Jetson-specific packages..."
pip install jetson-stats pynvml

# Set up CUDA environment variables
echo "Setting up CUDA environment variables..."
echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Create performance optimization script
echo "Creating performance optimization script..."
cat > jetson_optimize.sh << 'EOF'
#!/bin/bash
# Jetson Nano Performance Optimization for CUDA 12.6

# Set maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Set GPU memory allocation
export CUDA_VISIBLE_DEVICES=0

# Optimize for inference
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# CUDA 12.x specific optimizations
export CUDA_CACHE_MAXSIZE=2147483648  # 2GB cache
export CUDA_CACHE_DISABLE=0

echo "Jetson Nano optimized for maximum performance with CUDA 12.x"
EOF

chmod +x jetson_optimize.sh

echo ""
echo "✅ Setup complete for CUDA 12.6!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Optimize performance: ./jetson_optimize.sh"
echo "3. Run the application: python main.py"
echo ""
echo "The application will automatically detect CUDA 12.6 and use optimized settings."
