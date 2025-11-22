#!/usr/bin/env python3
"""
SpectraSense Unified Server
Combines:
1. Command Server (RFCOMM) - Send commands to phone
2. Audio Server (A2DP Sink) - Receive audio from phone

Both run simultaneously on same machine
"""

import subprocess
import sys
import os
import time
import threading
import signal
from pathlib import Path

# Import command server
sys.path.insert(0, str(Path(__file__).parent))
try:
    from main_fixed import BluetoothServer as CommandServer
except:
    print("⚠️  main_fixed.py not found, command server disabled")
    CommandServer = None

class UnifiedServer:
    def __init__(self):
        self.device_name = "SpectraSense"
        self.command_server = None
        self.audio_thread = None
        self.is_running = False
        
    def setup_system(self):
        """Setup both audio and command capabilities"""
        print("=" * 70)
        print(" SpectraSense Unified Server")
        print(" Commands + Audio")
        print("=" * 70)
        print()
        
        # Check dependencies
        print("🔍 Checking system...")
        
        missing = []
        for cmd in ['bluetoothctl', 'pactl', 'pulseaudio']:
            if subprocess.run(['which', cmd], capture_output=True).returncode != 0:
                missing.append(cmd)
        
        if missing:
            print(f"❌ Missing: {', '.join(missing)}")
            print("\n📦 Install with:")
            print("   sudo apt-get install bluez pulseaudio pulseaudio-module-bluetooth")
            return False
        
        print("✅ All dependencies found\n")
        
        # Setup PulseAudio for audio
        print("🔊 Configuring audio receiver...")
        subprocess.run(['pulseaudio', '--check'], check=False)
        if subprocess.run(['pulseaudio', '--check']).returncode != 0:
            subprocess.run(['pulseaudio', '--start'], check=False)
            time.sleep(1)
        
        modules = ['module-bluetooth-discover', 'module-bluetooth-policy', 'module-switch-on-connect']
        for module in modules:
            subprocess.run(['pactl', 'load-module', module], check=False)
        
        print("✅ Audio receiver ready\n")
        
        # Setup Bluetooth for both command and audio
        print("📡 Configuring Bluetooth...")
        subprocess.run(['sudo', 'rfkill', 'unblock', 'bluetooth'], check=False)
        time.sleep(0.5)
        
        commands = [
            'power on',
            'discoverable on',
            'pairable on',
            f'system-alias {self.device_name}'
        ]
        
        for cmd in commands:
            subprocess.run(
                ['bluetoothctl'],
                input=f"{cmd}\n",
                text=True,
                capture_output=True
            )
            time.sleep(0.2)
        
        print("✅ Bluetooth ready\n")
        return True
    
    def start_command_server(self):
        """Start RFCOMM command server in background"""
        if CommandServer is None:
            print("⚠️  Command server not available\n")
            return
        
        print("🎮 Starting command server...")
        
        def run_command_server():
            try:
                self.command_server = CommandServer()
                self.command_server.start()
            except Exception as e:
                print(f"Command server error: {e}")
        
        thread = threading.Thread(target=run_command_server, daemon=True)
        thread.start()
        time.sleep(2)  # Let it initialize
        
        print("✅ Command server running (RFCOMM channel 4)\n")
    
    def monitor_audio(self):
        """Monitor audio connections"""
        print("🎧 Audio monitor active...\n")
        
        last_status = None
        
        while self.is_running:
            # Check for Bluetooth audio
            result = subprocess.run(
                ['pactl', 'list', 'sinks', 'short'],
                capture_output=True,
                text=True
            )
            
            audio_active = 'bluez' in result.stdout.lower()
            
            if audio_active and not last_status:
                print("=" * 70)
                print("🔊 AUDIO CONNECTED - Playing on speakers")
                print("=" * 70)
                print()
                last_status = True
            elif not audio_active and last_status:
                print("⏸️  Audio disconnected\n")
                last_status = False
            
            time.sleep(2)
    
    def start(self):
        """Start unified server"""
        if not self.setup_system():
            return False
        
        try:
            # Start command server
            self.start_command_server()
            
            print("=" * 70)
            print("✅ SERVER IS READY!")
            print("=" * 70)
            print()
            print("📱 On your Android phone:")
            print("   1. Open Bluetooth settings")
            print(f"   2. Find and connect to '{self.device_name}'")
            print("   3. Accept pairing")
            print()
            print("🎮 COMMAND MODE:")
            print("   • Type commands in this terminal")
            print("   • Commands execute on phone")
            print()
            print("🔊 AUDIO MODE:")
            print("   • Enable 'Media audio' in phone's Bluetooth settings")
            print("   • All phone audio plays on computer speakers")
            print("   • Works for: music, navigation, calls, etc.")
            print()
            print("=" * 70)
            print()
            
            self.is_running = True
            self.monitor_audio()
            
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
        """Stop server"""
        self.is_running = False
        if self.command_server:
            self.command_server.is_running = False
        print("Cleaning up...")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SpectraSense Unified Server - Commands + Audio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Features:
  🎮 Send commands to phone via RFCOMM
  🔊 Receive all audio from phone (music, calls, navigation)
  
Usage:
  python3 %(prog)s
  
Then on Android:
  1. Connect to "SpectraSense" in Bluetooth
  2. Enable "Media audio" in device settings
  3. Use command server OR audio will just work
        """
    )
    
    parser.add_argument(
        '--name',
        help='Bluetooth device name',
        default='SpectraSense'
    )
    
    args = parser.parse_args()
    
    server = UnifiedServer()
    server.device_name = args.name
    
    def signal_handler(sig, frame):
        print("\n\n🛑 Shutting down...")
        server.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    success = server.start()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
