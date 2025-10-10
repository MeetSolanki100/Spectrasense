# Jetson Nano Quick Reference Card

## 🚀 Quick Setup Commands

### 1. Transfer Code to Jetson Nano
```bash
# From your Mac/PC
./transfer_to_jetson.sh <jetson-ip-address>
```

### 2. Deploy on Jetson Nano
```bash
# On Jetson Nano
./jetson_deploy.sh
```

### 3. Run Application
```bash
# Activate environment and run
source venv/bin/activate
python3 main.py
```

## 🔧 Essential Jetson Commands

### Performance Optimization
```bash
# Set maximum performance mode
sudo nvpmodel -m 0

# Set maximum clock speeds
sudo jetson_clocks

# Check current power mode
sudo nvpmodel -q
```

### Monitoring
```bash
# Real-time GPU/system monitoring
sudo tegrastats

# Check GPU status
nvidia-smi

# Memory usage
free -h

# Temperature
cat /sys/devices/virtual/thermal/thermal_zone*/temp
```

### CUDA Testing
```bash
# Test CUDA setup
python3 test_cuda.py

# Quick PyTorch CUDA test
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| **Object Detection** | 10-15 FPS |
| **Scene Description** | 1-2 FPS (every 30 frames) |
| **GPU Memory Usage** | 1-2 GB |
| **System RAM Usage** | 2-3 GB |
| **Temperature** | 40-60°C (with cooling) |

## 🎯 Jetson Nano GPU Specs

- **GPU**: NVIDIA Maxwell
- **CUDA Cores**: 128
- **Memory**: 4GB (shared)
- **Compute Capability**: 5.3
- **Memory Bandwidth**: 25.6 GB/s

## 🌐 Web Interface

Access the application at:
```
http://<jetson-ip>:5002
```

Find your Jetson's IP:
```bash
hostname -I
```

## 🚨 Troubleshooting

### CUDA Not Available
```bash
# Check PyTorch installation
python3 -c "import torch; print(torch.__version__)"

# Reinstall PyTorch for Jetson
pip3 install torch==1.12.0 torchvision==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Poor Performance
```bash
# Enable maximum performance
sudo nvpmodel -m 0
sudo jetson_clocks

# Check for thermal throttling
sudo tegrastats
```

### Camera Issues
```bash
# List cameras
ls /dev/video*

# Test camera
v4l2-ctl --list-devices
```

### Memory Issues
```bash
# Check memory usage
free -h

# Clear swap
sudo swapoff -a && sudo swapon -a
```

## 📁 File Structure

```
vision_encoder/
├── main.py                 # Main application
├── object_detection.py     # Object detection module
├── requirements.txt        # Python dependencies
├── setup_env.sh           # Jetson setup script
├── jetson_deploy.sh       # Deployment script
├── jetson_optimize.sh     # Performance optimization
├── test_cuda.py           # CUDA testing
├── transfer_to_jetson.sh  # Transfer script
└── templates/             # Web interface templates
```

## 🔗 Useful Links

- [NVIDIA Jetson Developer Forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems)
- [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
- [Jetson Performance Tuning](https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/power_management_jetson_xavier.html)

## ⚡ Performance Tips

1. **Always use maximum performance mode** (`sudo nvpmodel -m 0`)
2. **Ensure adequate cooling** (temperature < 70°C)
3. **Use stable power supply** (5V/4A recommended)
4. **Monitor with tegrastats** during development
5. **Clear CUDA cache** between heavy operations

## 🎉 Success Indicators

✅ CUDA available: `torch.cuda.is_available()` returns `True`  
✅ Models load successfully with CUDA  
✅ Web interface accessible at `http://<jetson-ip>:5002`  
✅ Real-time object detection working  
✅ Performance meets expected benchmarks  

Your Jetson Nano's Maxwell GPU with 128 CUDA cores is ready to accelerate your Vision Encoder! 🚀
