#!/bin/bash

# Core installation script for non-Jetson systems
echo "Installing core requirements for Vision Encoder..."

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

# Install core requirements
echo "Installing core requirements..."
pip install -r requirements.txt

echo "Core installation complete!"
echo "To activate the virtual environment: source venv/bin/activate"
echo "To run the application: python main.py"
echo ""
echo "Note: This installation is optimized for desktop systems."
echo "For Jetson Nano, use: ./setup_env.sh"
