#!/bin/bash

# Jetson Nano Deployment Script
# This script helps deploy the vision encoder to a Jetson Nano

echo "🚀 Jetson Nano Vision Encoder Deployment Script"
echo "=============================================="

# Check if we're on Jetson Nano
if [[ $(uname -m) != "aarch64" ]]; then
    echo "❌ This script must be run on a Jetson Nano (aarch64 architecture)"
    echo "   Current architecture: $(uname -m)"
    exit 1
fi

# Check for Jetson-specific files
if [[ ! -f "/etc/nv_tegra_release" ]]; then
    echo "❌ Jetson system not detected. Please ensure you're running on a Jetson Nano."
    exit 1
fi

echo "✅ Jetson Nano detected!"

# Display system info
echo ""
echo "📋 System Information:"
echo "Architecture: $(uname -m)"
echo "JetPack Version: $(cat /etc/nv_tegra_release 2>/dev/null | head -1)"
echo "CUDA Version: $(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/' || echo "Not detected")"

# Check GPU
echo ""
echo "🔍 GPU Information:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not available, checking via tegrastats..."
    sudo tegrastats --interval 1000 --logfile /tmp/tegrastats.log &
    TEGRA_PID=$!
    sleep 2
    kill $TEGRA_PID 2>/dev/null
    if [[ -f /tmp/tegrastats.log ]]; then
        head -5 /tmp/tegrastats.log
        rm -f /tmp/tegrastats.log
    fi
fi

echo ""
echo "🔧 Setting up Vision Encoder..."

# Run the main setup script
if [[ -f "setup_env.sh" ]]; then
    echo "Running setup_env.sh..."
    chmod +x setup_env.sh
    ./setup_env.sh
else
    echo "❌ setup_env.sh not found. Please ensure you're in the correct directory."
    exit 1
fi

# Optimize Jetson performance
echo ""
echo "⚡ Optimizing Jetson Nano performance..."
if [[ -f "jetson_optimize.sh" ]]; then
    chmod +x jetson_optimize.sh
    ./jetson_optimize.sh
else
    echo "Creating jetson_optimize.sh..."
    cat > jetson_optimize.sh << 'EOF'
#!/bin/bash
# Jetson Nano Performance Optimization

echo "Setting maximum performance mode..."
sudo nvpmodel -m 0
sudo jetson_clocks

echo "Setting GPU memory allocation..."
export CUDA_VISIBLE_DEVICES=0

echo "Optimizing for inference..."
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

echo "✅ Jetson Nano optimized for maximum performance"
EOF
    chmod +x jetson_optimize.sh
    ./jetson_optimize.sh
fi

# Test CUDA availability
echo ""
echo "🧪 Testing CUDA setup..."
source venv/bin/activate
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('❌ CUDA not available - check your PyTorch installation')
"

# Test the application
echo ""
echo "🎯 Testing Vision Encoder application..."
python3 -c "
from main import get_optimal_device, initialize_models
print('Testing device detection...')
device = get_optimal_device()
print('Testing model initialization...')
try:
    yolo_model, blip_processor, blip_model, device = initialize_models()
    print('✅ Application test successful!')
except Exception as e:
    print(f'❌ Application test failed: {e}')
"

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "📖 Next steps:"
echo "1. Access the web interface at: http://$(hostname -I | awk '{print $1}'):5002"
echo "2. Or run: python3 main.py"
echo "3. Monitor performance with: sudo tegrastats"
echo ""
echo "🔧 Performance monitoring commands:"
echo "- GPU usage: sudo tegrastats"
echo "- Memory usage: free -h"
echo "- Temperature: cat /sys/devices/virtual/thermal/thermal_zone*/temp"
echo "- Power mode: sudo nvpmodel -q"
