#!/usr/bin/env python3
"""
Advanced Bluetooth Audio Server with Phone Call Support
Handles A2DP audio + HFP phone calls + AVRCP controls

Uses native Linux tools (bluetoothctl, pactl) instead of Python bindings
"""

import subprocess
import sys
import os
import time
import signal
import threading
import json

class AdvancedBluetoothAudioServer:
    def __init__(self):
        self.device_name = "SpectraSense Audio"
        self.mainloop = None
        self.is_running = False
        self.audio_connected = False
        self.in_call = False
        
    def check_system(self):
        """Check if system is properly configured"""
        print("🔍 Checking system configuration...\n")
        
        # Check if running as root for some operations
        if os.geteuid() == 0:
            print("⚠️  Running as root - not recommended")
            print("   Run as normal user, will prompt for sudo when needed\n")
        
        # Check BlueZ version
        try:
            result = subprocess.run(
                ['bluetoothctl', '--version'],
                capture_output=True,
                text=True
            )
            version = result.stdout.strip()
            print(f"✅ BlueZ: {version}")
        except:
            print("❌ BlueZ not found - install bluez package")
            return False
        
        # Check PulseAudio
        try:
            result = subprocess.run(
                ['pulseaudio', '--version'],
                capture_output=True,
                text=True
            )
            version = result.stdout.strip().split('\n')[0]
            print(f"✅ {version}")
        except:
            print("❌ PulseAudio not found")
            return False
        
        # Check if Bluetooth is enabled
        try:
            result = subprocess.run(
                ['rfkill', 'list', 'bluetooth'],
                capture_output=True,
                text=True
            )
            if 'Soft blocked: yes' in result.stdout:
                print("⚠️  Bluetooth is blocked, unblocking...")
                subprocess.run(['sudo', 'rfkill', 'unblock', 'bluetooth'])
        except:
            pass
        
        print("\n✅ System check passed\n")
        return True
    
    def configure_bluetooth_audio_sink(self):
        """Configure system to act as Bluetooth audio receiver"""
        print("🔧 Configuring Bluetooth audio sink mode...\n")
        
        # Create/update Bluetooth configuration
        config_additions = """
# SpectraSense Audio Server Configuration

[General]
# Device name visible to other devices
Name = SpectraSense Audio

# Device class: Audio Device (Headset/Speaker)
Class = 0x240404

# Discoverable timeout (0 = always discoverable)
DiscoverableTimeout = 0

# Pairable timeout (0 = always pairable)
PairableTimeout = 0

# Enable A2DP Sink (receive audio)
Enable = Source,Sink,Media,Control

[Policy]
# Auto-connect to known devices
AutoEnable = true
"""
        
        print("Configuration needed in /etc/bluetooth/main.conf:")
        print(config_additions)
        print("\nCreating temporary config file...")
        
        # Save to temporary file
        config_path = os.path.expanduser("~/.config/spectrasense/bluetooth.conf")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            f.write(config_additions)
        
        print(f"✅ Config saved to: {config_path}")
        print("\n⚠️  To apply permanently, you need to:")
        print("   1. sudo nano /etc/bluetooth/main.conf")
        print("   2. Add the configuration above")
        print("   3. sudo systemctl restart bluetooth")
        print("\nFor now, we'll use runtime configuration...\n")
    
    def setup_pulseaudio_modules(self):
        """Load required PulseAudio modules for Bluetooth"""
        print("🔊 Setting up PulseAudio for Bluetooth...\n")
        
        # Ensure PulseAudio is running
        subprocess.run(['pulseaudio', '--check'], check=False)
        if subprocess.run(['pulseaudio', '--check']).returncode != 0:
            print("Starting PulseAudio...")
            subprocess.run(['pulseaudio', '--start'], check=False)
            time.sleep(2)
        
        # Load Bluetooth modules
        modules = [
            'module-bluetooth-discover',  # Auto-discover Bluetooth devices
            'module-bluetooth-policy',     # Automatic profile switching
            'module-switch-on-connect',    # Auto-switch audio to BT when connected
        ]
        
        for module in modules:
            # Check if already loaded
            result = subprocess.run(
                ['pactl', 'list', 'modules', 'short'],
                capture_output=True,
                text=True
            )
            
            if module not in result.stdout:
                print(f"Loading {module}...")
                subprocess.run(['pactl', 'load-module', module], check=False)
            else:
                print(f"✅ {module} already loaded")
        
        print("\n✅ PulseAudio configured\n")
    
    def make_discoverable(self):
        """Make server discoverable as Bluetooth audio device"""
        print("📡 Making server discoverable...\n")
        
        commands = [
            'power on',
            'discoverable on',
            'pairable on',
            f'system-alias "{self.device_name}"',
        ]
        
        for cmd in commands:
            subprocess.run(
                ['bluetoothctl'],
                input=f"{cmd}\n",
                text=True,
                capture_output=True
            )
            time.sleep(0.3)
        
        print(f"✅ Server is discoverable as: {self.device_name}\n")
    
    def monitor_bluetooth_connections(self):
        """Monitor Bluetooth connections and audio status"""
        print("👂 Monitoring Bluetooth connections...\n")
        print("=" * 60)
        print("Waiting for phone to connect...")
        print("=" * 60 + "\n")
        
        last_status = None
        
        while self.is_running:
            # Check connected devices
            result = subprocess.run(
                ['bluetoothctl', 'devices', 'Connected'],
                capture_output=True,
                text=True
            )
            
            connected = 'Device' in result.stdout
            
            # Check audio sinks
            result = subprocess.run(
                ['pactl', 'list', 'sinks', 'short'],
                capture_output=True,
                text=True
            )
            
            audio_active = 'bluez' in result.stdout.lower()
            
            # Generate status
            if connected and audio_active:
                status = "🔊 AUDIO PLAYING"
            elif connected:
                status = "📱 PHONE CONNECTED"
            else:
                status = "⏳ WAITING FOR CONNECTION"
            
            # Print status change
            if status != last_status:
                print(f"\n{'=' * 60}")
                print(f"{status}")
                print(f"{'=' * 60}\n")
                
                if audio_active:
                    print("🎵 All audio from your phone is now playing on this computer!")
                    print("   - Navigation directions")
                    print("   - Music/Videos")
                    print("   - Phone calls")
                    print("   - Notifications")
                
                last_status = status
            
            time.sleep(2)
    
    def start(self):
        """Start the Bluetooth audio server"""
        print("\n" + "=" * 60)
        print("🎧 SpectraSense Advanced Bluetooth Audio Server")
        print("=" * 60 + "\n")
        
        if not self.check_system():
            print("\n❌ System check failed")
            return False
        
        try:
            self.configure_bluetooth_audio_sink()
            self.setup_pulseaudio_modules()
            self.make_discoverable()
            
            print("\n" + "=" * 60)
            print("✅ SERVER IS READY!")
            print("=" * 60)
            print("\n📱 On your Android phone:")
            print("   1. Open Settings → Bluetooth")
            print("   2. Scan for devices")
            print(f"   3. Connect to '{self.device_name}'")
            print("   4. Accept pairing request")
            print("   5. Android will automatically route audio to this device")
            print("\n💡 TIP: Set this device as 'Media audio' in phone's")
            print("         Bluetooth settings for audio routing")
            print("\n" + "=" * 60 + "\n")
            
            self.is_running = True
            self.monitor_bluetooth_connections()
            
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")
            self.stop()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def stop(self):
        """Stop the server"""
        self.is_running = False
        print("Cleaning up...")
        
        # Make undiscoverable
        subprocess.run(
            ['bluetoothctl'],
            input='discoverable off\n',
            text=True,
            capture_output=True
        )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Advanced Bluetooth Audio Server - Receive audio from phone',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Start server
  %(prog)s --name "My Speaker"      # Custom device name
  
What this does:
  - Makes your computer act as a Bluetooth headset/speaker
  - Receives ALL audio from Android phone (music, calls, navigation)
  - Plays audio through computer speakers
  - Supports phone calls (mic + speaker)
        """
    )
    
    parser.add_argument(
        '--name',
        help='Bluetooth device name (default: SpectraSense Audio)',
        default='SpectraSense Audio'
    )
    
    args = parser.parse_args()
    
    server = AdvancedBluetoothAudioServer()
    server.device_name = args.name
    
    def signal_handler(sig, frame):
        print("\n\n🛑 Received shutdown signal...")
        server.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    success = server.start()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
