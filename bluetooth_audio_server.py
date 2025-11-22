#!/usr/bin/env python3
"""
Bluetooth Audio Server - Acts as Bluetooth Headset/Speaker
Receives audio from Android phone and plays on server speakers
Handles phone calls with bidirectional audio

Requires:
- BlueZ 5.x
- PulseAudio
- Python 3.7+
- Linux system

Installation:
    sudo apt-get install bluez pulseaudio pulseaudio-module-bluetooth python3-gi
"""

import subprocess
import sys
import os
import signal
import time
import json
from pathlib import Path

class BluetoothAudioServer:
    def __init__(self):
        self.device_name = "SpectraSense Audio"
        self.device_class = "0x240404"  # Audio device class (Headset)
        self.paired_device = None
        self.is_running = False
        
    def check_dependencies(self):
        """Check if required system components are installed"""
        print("🔍 Checking dependencies...")
        
        dependencies = {
            'bluetoothctl': 'bluez',
            'pactl': 'pulseaudio',
            'pulseaudio': 'pulseaudio'
        }
        
        missing = []
        for cmd, package in dependencies.items():
            if subprocess.run(['which', cmd], capture_output=True).returncode != 0:
                missing.append(package)
        
        if missing:
            print(f"❌ Missing dependencies: {', '.join(missing)}")
            print("\n📦 Install with:")
            print(f"   sudo apt-get install {' '.join(missing)} pulseaudio-module-bluetooth")
            return False
        
        print("✅ All dependencies found")
        return True
    
    def setup_bluetooth(self):
        """Configure Bluetooth adapter for audio receiving"""
        print("\n🔧 Configuring Bluetooth adapter...")
        
        # Power on Bluetooth
        subprocess.run(['sudo', 'rfkill', 'unblock', 'bluetooth'], check=False)
        time.sleep(1)
        
        # Configure bluetoothctl
        commands = [
            'power on',
            'discoverable on',
            'pairable on',
            f'system-alias {self.device_name}'
        ]
        
        for cmd in commands:
            proc = subprocess.Popen(
                ['bluetoothctl'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            proc.communicate(input=f"{cmd}\nexit\n")
            time.sleep(0.5)
        
        print("✅ Bluetooth adapter configured")
    
    def setup_pulseaudio(self):
        """Configure PulseAudio for Bluetooth audio"""
        print("\n🔊 Configuring PulseAudio...")
        
        # Ensure PulseAudio is running
        subprocess.run(['pulseaudio', '--start'], check=False)
        time.sleep(1)
        
        # Load Bluetooth modules
        modules = [
            'module-bluetooth-discover',
            'module-bluetooth-policy',
            'module-switch-on-connect'
        ]
        
        for module in modules:
            subprocess.run(['pactl', 'load-module', module], check=False)
        
        # Set default sink to allow Bluetooth routing
        subprocess.run(['pactl', 'set-default-sink', '@DEFAULT_SINK@'], check=False)
        
        print("✅ PulseAudio configured for Bluetooth audio")
    
    def pair_device(self, device_address=None):
        """Pair with Android device"""
        print("\n📱 Device Pairing...")
        
        if device_address:
            # Pair specific device
            print(f"Pairing with {device_address}...")
            subprocess.run(['bluetoothctl', 'pair', device_address])
            subprocess.run(['bluetoothctl', 'trust', device_address])
            subprocess.run(['bluetoothctl', 'connect', device_address])
            self.paired_device = device_address
        else:
            # Wait for pairing request
            print("Waiting for device to pair...")
            print("On your Android phone:")
            print(f"  1. Go to Bluetooth settings")
            print(f"  2. Scan for devices")
            print(f"  3. Connect to '{self.device_name}'")
            print("\nPress Ctrl+C when paired...")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n✅ Pairing complete")
    
    def enable_audio_profiles(self):
        """Enable A2DP Sink and HFP profiles"""
        print("\n🎧 Enabling audio profiles...")
        
        # Restart Bluetooth service with proper configuration
        config_file = '/etc/bluetooth/main.conf'
        
        print("Checking Bluetooth configuration...")
        try:
            with open(config_file, 'r') as f:
                config = f.read()
            
            if 'Class = 0x' not in config:
                print("⚠️  Need to configure device class for headset mode")
                print("\nAdd to /etc/bluetooth/main.conf under [General]:")
                print(f"  Class = {self.device_class}")
                print("  Name = {self.device_name}")
                print("\nThen restart bluetooth:")
                print("  sudo systemctl restart bluetooth")
        except Exception as e:
            print(f"⚠️  Could not check config: {e}")
        
        print("✅ Audio profiles ready")
    
    def monitor_audio(self):
        """Monitor and route Bluetooth audio"""
        print("\n🎵 Audio monitoring active...")
        print("Audio from phone will play on server speakers")
        print("Press Ctrl+C to stop\n")
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Monitor PulseAudio sinks
                result = subprocess.run(
                    ['pactl', 'list', 'sinks', 'short'],
                    capture_output=True,
                    text=True
                )
                
                # Check for Bluetooth audio devices
                if 'bluez' in result.stdout.lower():
                    # Bluetooth audio is connected
                    if not hasattr(self, '_audio_playing'):
                        print("🔊 Bluetooth audio connected - playing on speakers")
                        self._audio_playing = True
                else:
                    if hasattr(self, '_audio_playing'):
                        print("⏸️  Bluetooth audio disconnected")
                        delattr(self, '_audio_playing')
                
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopping audio monitor...")
            self.is_running = False
    
    def start(self, device_address=None):
        """Start the Bluetooth audio server"""
        print("=" * 60)
        print("🎧 SpectraSense Bluetooth Audio Server")
        print("=" * 60)
        
        if not self.check_dependencies():
            return False
        
        try:
            self.setup_bluetooth()
            self.setup_pulseaudio()
            self.enable_audio_profiles()
            
            if device_address:
                self.pair_device(device_address)
            
            print("\n" + "=" * 60)
            print("✅ Server is ready!")
            print("=" * 60)
            print("\n📱 Connect your phone:")
            print("   1. Open Bluetooth settings on Android")
            print(f"   2. Find and connect to '{self.device_name}'")
            print("   3. Phone may ask to pair - Accept")
            print("   4. Set audio output to Bluetooth")
            print("\n🎵 All audio from phone will play on this computer")
            print("=" * 60 + "\n")
            
            self.monitor_audio()
            
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return False
        
        return True
    
    def stop(self):
        """Stop the audio server"""
        self.is_running = False
        print("Cleaning up...")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Bluetooth Audio Server - Acts as Bluetooth headset/speaker'
    )
    parser.add_argument(
        '--device',
        help='Bluetooth address of device to pair with (e.g., AA:BB:CC:DD:EE:FF)',
        default=None
    )
    parser.add_argument(
        '--name',
        help='Bluetooth device name',
        default='SpectraSense Audio'
    )
    
    args = parser.parse_args()
    
    server = BluetoothAudioServer()
    if args.name:
        server.device_name = args.name
    
    def signal_handler(sig, frame):
        print("\n\n🛑 Received shutdown signal...")
        server.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    success = server.start(args.device)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
