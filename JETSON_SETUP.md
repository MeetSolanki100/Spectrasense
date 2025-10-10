# Jetson Nano Setup Guide

This guide will help you set up and run the Vision Encoder application on a Jetson Nano.

## Prerequisites

- Jetson Nano with JetPack 4.6+ or 5.x installed
- At least 32GB SD card (64GB recommended)
- USB camera or CSI camera
- Internet connection for initial setup

## Quick Setup

### For Jetson Nano:
1. **Clone and navigate to the project directory:**
   ```bash
   git clone <your-repo-url>
   cd vision_encoder
   ```

2. **Run the Jetson setup script:**
   ```bash
   chmod +x setup_env.sh
   ./setup_env.sh
   ```

### For Desktop/Development Systems:
1. **Clone and navigate to the project directory:**
   ```bash
   git clone <your-repo-url>
   cd vision_encoder
   ```

2. **Run the core installation script:**
   ```bash
   chmod +x install_core.sh
   ./install_core.sh
   ```

**Note:** The `jetson-stats` package will fail to install on non-Jetson systems (like macOS, Windows, or x86 Linux). This is expected and normal. Use `install_core.sh` for development on your desktop system.

3. **Optimize Jetson Nano performance:**
   ```bash
   ./jetson_optimize.sh
   ```

4. **Activate virtual environment and run:**
   ```bash
   source venv/bin/activate
   python main.py
   ```

## Manual Setup (if automatic setup fails)

### 1. System Requirements
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev python3-venv
sudo apt-get install -y libjpeg-dev zlib1g-dev libpng-dev
sudo apt-get install -y libopencv-dev python3-opencv
sudo apt-get install -y build-essential cmake git wget
```

### 2. PyTorch Installation

**For JetPack 5.x (Ubuntu 20.04):**
```bash
pip3 install torch==1.12.0 torchvision==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

**For JetPack 4.6 (Ubuntu 18.04):**
```bash
wget https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# Install torchvision from source
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
python3 setup.py install
cd ..
```

### 3. Install Other Dependencies
```bash
pip3 install -r requirements.txt
pip3 install jetson-stats pynvml
```

## Performance Optimization

### 1. Set Maximum Performance Mode
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

### 2. Memory Configuration
The application automatically detects Jetson Nano and applies optimizations:
- Uses YOLOv8s (smaller model) instead of YOLOv8x
- Enables half-precision (FP16) for BLIP model
- Reduces frame analysis frequency
- Optimizes camera buffer settings

### 3. CUDA Memory Management
```bash
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Camera Configuration

### CSI Camera
The application automatically detects and configures CSI cameras (index 0).

### USB Camera
If using a USB camera, it will be detected as index 1 or higher.

### Camera Settings
- Resolution: 640x480 (optimized for Jetson Nano)
- FPS: 30
- Format: MJPEG (for better performance)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory:**
   - The application automatically falls back to CPU if CUDA memory is insufficient
   - Try reducing image resolution in the code

2. **Camera Not Detected:**
   - Check camera permissions: `sudo usermod -a -G video $USER`
   - List available cameras: `ls /dev/video*`
   - Test camera: `v4l2-ctl --list-devices`

3. **Slow Performance:**
   - Ensure Jetson is in maximum performance mode: `sudo nvpmodel -m 0`
   - Check thermal throttling: `sudo tegrastats`
   - Consider using a cooling fan

4. **PyTorch Installation Issues:**
   - Verify JetPack version: `cat /etc/nv_tegra_release`
   - Use the correct PyTorch wheel for your JetPack version

### Performance Monitoring

Monitor system performance:
```bash
# Check GPU usage
sudo tegrastats

# Check memory usage
free -h

# Check CPU temperature
cat /sys/devices/virtual/thermal/thermal_zone*/temp
```

## Application Features

### Automatic Optimizations
- **Device Detection:** Automatically detects Jetson Nano and applies optimizations
- **Memory Management:** Clears CUDA cache between inferences
- **Model Selection:** Uses smaller YOLO model for better performance
- **Precision Optimization:** Uses FP16 for memory efficiency

### Web Interface
- Access at `http://<jetson-ip>:5002`
- Real-time object detection and scene description
- Video upload and analysis
- Camera controls

## Performance Expectations

On Jetson Nano with optimal settings:
- **Object Detection:** ~10-15 FPS
- **Scene Description:** ~1-2 FPS (every 30 frames)
- **Memory Usage:** ~2-3GB RAM, ~1GB GPU memory
- **Temperature:** Monitor with `sudo tegrastats`

## Advanced Configuration

### Custom Model Paths
You can modify the model paths in `main.py`:
```python
# For custom YOLO model
yolo_model = YOLO('path/to/your/model.pt')

# For custom BLIP model
BLIP_MODEL_NAME = "your/custom/blip/model"
```

### Frame Processing Interval
Adjust the analysis frequency in `main.py`:
```python
FRAME_INTERVAL_FOR_ANALYSIS = 30  # Analyze every 30 frames
```

## Support

For issues specific to Jetson Nano:
1. Check NVIDIA Jetson Developer Forums
2. Verify JetPack version compatibility
3. Monitor system resources during execution
4. Consider hardware cooling solutions for sustained performance
