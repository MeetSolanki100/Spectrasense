#!/bin/bash
# SpectraSense Audio Server - Quick Setup Script for Linux

echo "=========================================="
echo " SpectraSense Audio Server Setup"
echo "=========================================="
echo

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "❌ This script is for Linux only"
    exit 1
fi

# Check for sudo
if ! command -v sudo &> /dev/null; then
    echo "❌ sudo not found. Please run as root or install sudo."
    exit 1
fi

echo "🔍 Checking system packages..."

# Detect package manager
if command -v apt-get &> /dev/null; then
    PKG_MANAGER="apt-get"
    PKG_INSTALL="sudo apt-get install -y"
    PACKAGES="bluez pulseaudio pulseaudio-module-bluetooth python3 python3-gi"
elif command -v dnf &> /dev/null; then
    PKG_MANAGER="dnf"
    PKG_INSTALL="sudo dnf install -y"
    PACKAGES="bluez pulseaudio pulseaudio-module-bluetooth python3 python3-gobject"
elif command -v pacman &> /dev/null; then
    PKG_MANAGER="pacman"
    PKG_INSTALL="sudo pacman -S --noconfirm"
    PACKAGES="bluez pulseaudio pulseaudio-bluetooth python python-gobject"
else
    echo "❌ No supported package manager found (apt/dnf/pacman)"
    exit 1
fi

echo "✅ Detected package manager: $PKG_MANAGER"
echo

# Check if packages are installed
MISSING_PACKAGES=""
for pkg in bluez pulseaudio python3; do
    if ! command -v $pkg &> /dev/null; then
        MISSING_PACKAGES="$MISSING_PACKAGES $pkg"
    fi
done

if [ -n "$MISSING_PACKAGES" ]; then
    echo "📦 Installing missing packages..."
    $PKG_INSTALL $PACKAGES
    echo
else
    echo "✅ All required packages already installed"
    echo
fi

# Enable and start Bluetooth
echo "🔧 Configuring Bluetooth..."
sudo systemctl enable bluetooth
sudo systemctl start bluetooth
sudo rfkill unblock bluetooth
echo "✅ Bluetooth enabled"
echo

# Start PulseAudio
echo "🔊 Configuring PulseAudio..."
pulseaudio --check || pulseaudio --start
pactl load-module module-bluetooth-discover 2>/dev/null
pactl load-module module-bluetooth-policy 2>/dev/null
pactl load-module module-switch-on-connect 2>/dev/null
echo "✅ PulseAudio configured"
echo

# Make scripts executable
chmod +x bluetooth_audio_advanced.py 2>/dev/null
chmod +x unified_server.py 2>/dev/null

echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo
echo "🚀 To start the audio server:"
echo "   python3 bluetooth_audio_advanced.py"
echo
echo "🚀 To start unified server (commands + audio):"
echo "   python3 unified_server.py"
echo
echo "📱 Then on your Android phone:"
echo "   1. Go to Bluetooth settings"
echo "   2. Scan and connect to 'SpectraSense Audio'"
echo "   3. Enable 'Media audio' in device settings"
echo
echo "📖 For detailed instructions, see:"
echo "   BLUETOOTH_AUDIO_SETUP.md"
echo
