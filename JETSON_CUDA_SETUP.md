# Jetson Nano CUDA Setup Guide

This guide specifically covers setting up CUDA acceleration for the Vision Encoder on Jetson Nano with its Maxwell GPU (128 CUDA cores).

## Jetson Nano GPU Specifications

- **GPU**: NVIDIA Maxwell GPU
- **CUDA Cores**: 128
- **GPU Memory**: 4GB (shared with system RAM)
- **CUDA Compute Capability**: 5.3
- **Memory Bandwidth**: 25.6 GB/s

## Prerequisites

### 1. JetPack Installation
Ensure you have JetPack 4.6+ or 5.x installed:
```bash
# Check JetPack version
cat /etc/nv_tegra_release

# Check CUDA version
nvcc --version
```

### 2. System Requirements
- At least 32GB SD card (64GB recommended)
- Active cooling (fan or heatsink)
- Stable power supply (5V/4A recommended)

## Step-by-Step CUDA Setup

### 1. Clone and Prepare
```bash
# Clone your repository
git clone <your-repo-url>
cd vision_encoder

# Make scripts executable
chmod +x *.sh
```

### 2. Run the Deployment Script
```bash
# This will handle everything automatically
./jetson_deploy.sh
```

### 3. Manual Setup (if needed)

#### Install PyTorch with CUDA Support
```bash
# For JetPack 5.x (Ubuntu 20.04)
pip3 install torch==1.12.0 torchvision==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu113

# For JetPack 4.6 (Ubuntu 18.04)
wget https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# Install torchvision from source
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
python3 setup.py install
cd ..
```

#### Install Jetson-Specific Packages
```bash
# Install monitoring tools
pip3 install jetson-stats pynvml

# Install other requirements
pip3 install -r requirements.txt
```

### 4. Optimize for CUDA Performance

#### Set Maximum Performance Mode
```bash
# Set to maximum performance mode (mode 0)
sudo nvpmodel -m 0

# Set maximum clock speeds
sudo jetson_clocks

# Verify power mode
sudo nvpmodel -q
```

#### Configure CUDA Environment
```bash
# Add to ~/.bashrc
echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc

# Apply changes
source ~/.bashrc
```

## CUDA Testing and Verification

### 1. Test CUDA Availability
```bash
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
"
```

### 2. Test CUDA Performance
```bash
python3 -c "
import torch
import time

# Test CUDA tensor operations
if torch.cuda.is_available():
    device = torch.device('cuda')
    
    # Create large tensors
    size = 1000
    a = torch.randn(size, size).to(device)
    b = torch.randn(size, size).to(device)
    
    # Time matrix multiplication
    start_time = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f'CUDA matrix multiplication time: {end_time - start_time:.4f} seconds')
    print(f'GPU memory used: {torch.cuda.memory_allocated() / 1024**2:.1f} MB')
    print(f'GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB')
else:
    print('CUDA not available')
"
```

### 3. Test Vision Encoder CUDA Integration
```bash
# Test the application's CUDA detection
python3 -c "
from main import get_optimal_device, initialize_models
device = get_optimal_device()
print(f'Selected device: {device}')
"
```

## Performance Optimization

### 1. Memory Management
The application automatically handles CUDA memory optimization:
- Uses half-precision (FP16) for BLIP model
- Clears CUDA cache between inferences
- Uses smaller YOLO model (YOLOv8s)
- Resizes images for memory efficiency

### 2. Model Optimizations
```python
# The application automatically applies these optimizations:
- YOLO: Uses YOLOv8s instead of YOLOv8x
- BLIP: Loads with torch_dtype=torch.float16
- Inference: Uses torch.no_grad() for memory efficiency
- Generation: Limits max_length and num_beams for speed
```

### 3. Camera Optimization
```python
# Jetson-specific camera settings:
- Resolution: 640x480 (optimal for Jetson)
- Buffer size: 1 (reduces latency)
- Format: MJPEG (better performance)
- CSI camera priority over USB
```

## Monitoring and Troubleshooting

### 1. Real-time Monitoring
```bash
# Monitor GPU usage, temperature, and power
sudo tegrastats

# Monitor memory usage
watch -n 1 free -h

# Check GPU temperature
cat /sys/devices/virtual/thermal/thermal_zone*/temp
```

### 2. Common Issues and Solutions

#### CUDA Out of Memory
```bash
# Reduce batch size or image resolution in code
# The application automatically falls back to CPU if needed
```

#### Poor Performance
```bash
# Ensure maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Check thermal throttling
sudo tegrastats
```

#### Camera Issues
```bash
# List available cameras
ls /dev/video*

# Test camera
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

### 3. Performance Benchmarks

Expected performance on Jetson Nano with CUDA:
- **Object Detection**: 10-15 FPS
- **Scene Description**: 1-2 FPS (every 30 frames)
- **Memory Usage**: 2-3GB RAM, 1-2GB GPU memory
- **Temperature**: 40-60°C (with cooling)

## Running the Application

### 1. Start the Application
```bash
# Activate virtual environment
source venv/bin/activate

# Run with CUDA optimization
python3 main.py
```

### 2. Access Web Interface
```bash
# Find your Jetson's IP address
hostname -I

# Access from browser (replace with your Jetson's IP)
http://<jetson-ip>:5002
```

### 3. Performance Monitoring
```bash
# In another terminal, monitor performance
sudo tegrastats
```

## Advanced CUDA Configuration

### 1. Custom CUDA Memory Allocation
```python
# In main.py, you can adjust these settings:
torch.cuda.empty_cache()  # Clear cache
torch.cuda.set_per_process_memory_fraction(0.8)  # Limit GPU memory usage
```

### 2. Multi-GPU Support (if applicable)
```python
# The application automatically detects and uses the first available GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
```

### 3. CUDA Stream Optimization
```python
# For advanced users, you can add CUDA streams:
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    # Your CUDA operations here
```

## Troubleshooting Checklist

- [ ] JetPack properly installed
- [ ] CUDA toolkit available (`nvcc --version`)
- [ ] PyTorch with CUDA support installed
- [ ] Maximum performance mode enabled (`sudo nvpmodel -m 0`)
- [ ] Adequate cooling (temperature < 70°C)
- [ ] Sufficient power supply (5V/4A)
- [ ] CUDA environment variables set
- [ ] Application detects CUDA (`torch.cuda.is_available()` returns `True`)

## Support and Resources

- [NVIDIA Jetson Developer Forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems)
- [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
- [Jetson Performance Tuning](https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_jetson_xavier.html)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)

Your Jetson Nano's Maxwell GPU with 128 CUDA cores is perfectly capable of running the Vision Encoder with CUDA acceleration! 🚀
