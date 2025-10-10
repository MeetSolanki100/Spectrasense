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

# Install additional required packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate

echo "Setup complete! Activate the virtual environment with: source venv/bin/activate"
