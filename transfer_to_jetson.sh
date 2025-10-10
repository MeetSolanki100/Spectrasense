#!/bin/bash

# Script to transfer Vision Encoder to Jetson Nano
# Usage: ./transfer_to_jetson.sh <jetson-ip-address>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <jetson-ip-address>"
    echo "Example: $0 192.168.1.100"
    exit 1
fi

JETSON_IP=$1
LOCAL_DIR=$(pwd)
REMOTE_DIR="~/vision_encoder"

echo "🚀 Transferring Vision Encoder to Jetson Nano"
echo "============================================="
echo "Jetson IP: $JETSON_IP"
echo "Local directory: $LOCAL_DIR"
echo "Remote directory: $REMOTE_DIR"

# Check if SSH key exists
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "🔑 Generating SSH key..."
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
    echo "📋 Copy this public key to your Jetson Nano:"
    cat ~/.ssh/id_rsa.pub
    echo ""
    echo "On your Jetson Nano, run:"
    echo "mkdir -p ~/.ssh"
    echo "echo '$(cat ~/.ssh/id_rsa.pub)' >> ~/.ssh/authorized_keys"
    echo "chmod 600 ~/.ssh/authorized_keys"
    echo "chmod 700 ~/.ssh"
    echo ""
    read -p "Press Enter after setting up SSH key on Jetson..."
fi

echo "📁 Creating remote directory..."
ssh jetson@$JETSON_IP "mkdir -p $REMOTE_DIR"

echo "📤 Transferring files..."
# Exclude virtual environment and large files
rsync -avz --progress \
    --exclude='venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    --exclude='*.log' \
    --exclude='static/uploads/' \
    --exclude='test_images/' \
    --exclude='*.pt' \
    $LOCAL_DIR/ jetson@$JETSON_IP:$REMOTE_DIR/

echo "🔧 Setting up on Jetson Nano..."
ssh jetson@$JETSON_IP "cd $REMOTE_DIR && chmod +x *.sh *.py"

echo "⚡ Running deployment script on Jetson..."
ssh -t jetson@$JETSON_IP "cd $REMOTE_DIR && ./jetson_deploy.sh"

echo ""
echo "🎉 Transfer complete!"
echo ""
echo "📖 Next steps on your Jetson Nano:"
echo "1. SSH into your Jetson: ssh jetson@$JETSON_IP"
echo "2. Navigate to the project: cd $REMOTE_DIR"
echo "3. Run the application: source venv/bin/activate && python3 main.py"
echo "4. Access web interface: http://$JETSON_IP:5002"
echo ""
echo "🔧 Useful commands on Jetson:"
echo "- Monitor performance: sudo tegrastats"
echo "- Test CUDA: python3 test_cuda.py"
echo "- Check GPU status: nvidia-smi"
