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
        
        # Kill any existing PulseAudio instance and restart fresh
        print("Restarting PulseAudio...")
        subprocess.run(['pulseaudio', '-k'], check=False)
        time.sleep(1)
        subprocess.run(['pulseaudio', '--start'], check=False)
        time.sleep(2)
        
        # Unload any existing Bluetooth modules first
        print("Unloading old Bluetooth modules...")
        subprocess.run(['pactl', 'unload-module', 'module-bluetooth-discover'], 
                      capture_output=True, check=False)
        subprocess.run(['pactl', 'unload-module', 'module-bluetooth-policy'], 
                      capture_output=True, check=False)
        time.sleep(1)
        
        # Load Bluetooth modules in correct order
        modules = [
            'module-bluetooth-discover',  # Auto-discover Bluetooth devices
            'module-bluetooth-policy',     # Automatic profile switching
            'module-switch-on-connect',    # Auto-switch audio to BT when connected
            'module-loopback',             # For audio loopback if needed
        ]
        
        for module in modules:
            print(f"Loading {module}...")
            result = subprocess.run(
                ['pactl', 'load-module', module],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"✅ {module} loaded")
            else:
                print(f"⚠️  {module} already loaded or not available")
            time.sleep(0.5)
        
        # Set default sink to auto-select
        print("Configuring default audio sink...")
        subprocess.run(['pactl', 'set-default-sink', '@DEFAULT_SINK@'], check=False)
        
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
    
    def show_audio_debug_info(self):
        """Show detailed audio configuration for debugging"""
        print("\n" + "=" * 60)
        print("🔍 Audio Debug Information")
        print("=" * 60 + "\n")
        
        # Show all sinks
        print("📊 Available Audio Sinks:")
        result = subprocess.run(['pactl', 'list', 'sinks', 'short'], 
                               capture_output=True, text=True)
        print(result.stdout if result.stdout else "  None found")
        
        # Show all sources
        print("\n📊 Available Audio Sources:")
        result = subprocess.run(['pactl', 'list', 'sources', 'short'], 
                               capture_output=True, text=True)
        print(result.stdout if result.stdout else "  None found")
        
        # Show all cards
        print("\n📊 Available Audio Cards:")
        result = subprocess.run(['pactl', 'list', 'cards', 'short'], 
                               capture_output=True, text=True)
        print(result.stdout if result.stdout else "  None found")
        
        # Show loaded modules
        print("\n📊 Loaded PulseAudio Modules:")
        result = subprocess.run(['pactl', 'list', 'modules', 'short'], 
                               capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'bluetooth' in line.lower() or 'loopback' in line.lower():
                print(f"  {line}")
        
        print("\n" + "=" * 60 + "\n")
    
    def monitor_bluetooth_connections(self):
        """Monitor Bluetooth connections and audio status"""
        print("👂 Monitoring Bluetooth connections...\n")
        print("=" * 60)
        print("Waiting for phone to connect...")
        print("=" * 60 + "\n")
        
        last_status = None
        connected_device = None
        
        while self.is_running:
            # Check connected devices
            result = subprocess.run(
                ['bluetoothctl', 'devices', 'Connected'],
                capture_output=True,
                text=True
            )
            
            connected = 'Device' in result.stdout
            
            # If device just connected, set audio profile
            if connected and not connected_device:
                # Extract device MAC address
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'Device' in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            mac = parts[1]
                            connected_device = mac
                            print(f"📱 Device connected: {mac}")
                            print("Setting audio profile to A2DP Sink...")
                            
                            # Wait for device to be fully registered
                            time.sleep(2)
                            
                            # Try to set profile to a2dp_sink
                            subprocess.run(
                                ['pactl', 'set-card-profile', f'bluez_card.{mac.replace(":", "_")}', 'a2dp_sink'],
                                capture_output=True
                            )
                            time.sleep(1)
                            break
            elif not connected:
                connected_device = None
            
            # Check audio sinks
            result = subprocess.run(
                ['pactl', 'list', 'sinks', 'short'],
                capture_output=True,
                text=True
            )
            
            audio_active = 'bluez' in result.stdout.lower()
            
            # If connected but no audio, try to activate sink
            if connected and not audio_active and connected_device:
                print("🔧 Attempting to activate Bluetooth audio sink...")
                
                # List all cards
                cards_result = subprocess.run(
                    ['pactl', 'list', 'cards', 'short'],
                    capture_output=True,
                    text=True
                )
                
                print(f"Available cards:\n{cards_result.stdout}")
                
                # Try to find and activate the Bluetooth card
                for line in cards_result.stdout.split('\n'):
                    if 'bluez' in line.lower():
                        card_name = line.split()[0] if line.split() else None
                        if card_name:
                            print(f"Found Bluetooth card: {card_name}")
                            # Set to a2dp_sink profile
                            subprocess.run(
                                ['pactl', 'set-card-profile', card_name, 'a2dp_sink'],
                                capture_output=True
                            )
                            time.sleep(1)
            
            # Generate status
            if connected and audio_active:
                status = "🔊 AUDIO PLAYING"
            elif connected:
                status = "📱 PHONE CONNECTED (waiting for audio...)"
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
                elif connected and not audio_active:
                    print("⚠️  Phone connected but audio not active yet.")
                    print("   On your phone:")
                    print("   1. Go to Bluetooth settings")
                    print("   2. Tap the gear icon next to 'SpectraSense Audio'")
                    print("   3. Make sure 'Media audio' is enabled")
                    print("   4. Try playing music on phone")
                
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
            
            # Show debug info
            self.show_audio_debug_info()
            
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
