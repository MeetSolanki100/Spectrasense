#!/usr/bin/env python3
"""
SpectraSense Unified Bluetooth Server
Combines Command Server (RFCOMM) + Audio Server (A2DP)

Features:
- Receives commands from app via RFCOMM (like main.py)
- Receives audio from phone via A2DP (music, calls, navigation)
- Single connection point through app
- Automatic audio routing to computer speakers

Connection Flow:
1. App connects to server via RFCOMM (channel 4)
2. Server accepts connection and performs handshake
3. Server automatically configures phone for audio streaming
4. Both commands and audio work simultaneously
"""

import socket
import struct
import threading
import time
import sys
import subprocess
import select

# Protocol Constants
TYPE_TEXT = 1
TYPE_COMMAND = 2
SECRET_HANDSHAKE = "SMART_DEVICE_KEY_2024"
RFCOMM_CHANNEL = 4

class UnifiedBluetoothServer:
    def __init__(self):
        self.server_socket = None
        self.client_socket = None
        self.client_address = None
        self.is_running = False
        self.receive_thread = None
        self.audio_configured = False
        
    def pack_message(self, message_type, payload):
        """Pack message with protocol: [1 byte type][4 bytes length][payload]"""
        payload_bytes = payload.encode('utf-8') if isinstance(payload, str) else payload
        length = len(payload_bytes)
        header = struct.pack('!BI', message_type, length)
        return header + payload_bytes
    
    def unpack_message(self, data):
        """Unpack received message"""
        if len(data) < 5:
            return None, None
        
        message_type = data[0]
        length = struct.unpack('!I', data[1:5])[0]
        
        if len(data) < 5 + length:
            return None, None
        
        payload = data[5:5+length].decode('utf-8', errors='ignore')
        return message_type, payload
    
    def setup_audio_server(self):
        """Configure PulseAudio for Bluetooth audio receiving"""
        print("\n🔊 Configuring audio server...")
        
        try:
            # Restart PulseAudio
            subprocess.run(['pulseaudio', '-k'], check=False, 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1)
            subprocess.run(['pulseaudio', '--start'], check=False,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1)
            
            # Remove any loopback modules
            result = subprocess.run(['pactl', 'list', 'modules', 'short'],
                                  capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'module-loopback' in line:
                    module_id = line.split()[0]
                    subprocess.run(['pactl', 'unload-module', module_id], 
                                 check=False, stdout=subprocess.DEVNULL)
            
            # Load Bluetooth modules
            subprocess.run(['pactl', 'unload-module', 'module-bluetooth-discover'],
                         capture_output=True, check=False)
            subprocess.run(['pactl', 'unload-module', 'module-bluetooth-policy'],
                         capture_output=True, check=False)
            time.sleep(0.5)
            
            modules = ['module-bluetooth-discover', 'module-bluetooth-policy', 
                      'module-switch-on-connect']
            for module in modules:
                subprocess.run(['pactl', 'load-module', module],
                             capture_output=True, check=False)
            
            # Mute microphone monitoring to prevent feedback
            result = subprocess.run(['pactl', 'list', 'sources', 'short'],
                                  capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if line.strip() and 'monitor' not in line.lower():
                    source_id = line.split()[0] if line.split() else None
                    if source_id:
                        subprocess.run(['pactl', 'set-source-mute', source_id, '1'],
                                     capture_output=True)
            
            print("✅ Audio server configured")
            
        except Exception as e:
            print(f"⚠️  Audio configuration warning: {e}")
    
    def configure_phone_audio(self, phone_address):
        """Configure connected phone for audio streaming"""
        if self.audio_configured:
            return
        
        print(f"\n🎧 Configuring audio for {phone_address}...")
        
        try:
            # Wait for Bluetooth to register device
            time.sleep(2)
            
            # Get card name
            card_name = f'bluez_card.{phone_address.replace(":", "_")}'
            
            # Set profile to a2dp_sink (receive audio from phone)
            result = subprocess.run(
                ['pactl', 'set-card-profile', card_name, 'a2dp_sink'],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # Try alternative name
                subprocess.run(
                    ['pactl', 'set-card-profile', card_name, 'a2dp-sink'],
                    capture_output=True
                )
            
            time.sleep(1)
            
            # Ensure local speakers are default (phone audio plays here)
            subprocess.run(['pactl', 'set-default-sink', '0'],
                         capture_output=True)
            
            self.audio_configured = True
            print("✅ Phone audio configured - audio will play on computer speakers")
            
        except Exception as e:
            print(f"⚠️  Audio setup warning: {e}")
    
    def send_to_app(self, message_type, message):
        """Send message to app"""
        if not self.client_socket:
            return False
        
        try:
            data = self.pack_message(message_type, message)
            self.client_socket.sendall(data)
            return True
        except Exception as e:
            print(f"Send error: {e}")
            return False
    
    def receive_from_app(self):
        """Background thread to receive status messages from app"""
        print("📱 Listening for app messages...")
        
        while self.is_running and self.client_socket:
            try:
                # Use select to check if data is available (non-blocking with timeout)
                ready = select.select([self.client_socket], [], [], 1.0)
                
                if not ready[0]:
                    continue
                
                data = self.client_socket.recv(4096)
                
                if not data:
                    print("\n📱 App disconnected")
                    self.is_running = False
                    break
                
                message_type, payload = self.unpack_message(data)
                
                if message_type == TYPE_TEXT:
                    print(f"\n📱 Status from app: {payload}")
                
            except Exception as e:
                if self.is_running:
                    print(f"\nReceive error: {e}")
                break
    
    def send_command(self, command):
        """Send command to phone app"""
        print(f"📤 Sending command: {command}")
        
        if self.send_to_app(TYPE_COMMAND, command):
            print("✓ Command sent")
        else:
            print("✗ Failed to send command")
    
    def command_input_loop(self):
        """Interactive command input loop"""
        print("\n" + "=" * 60)
        print("📝 COMMAND MODE")
        print("=" * 60)
        print("Type commands to send to phone:")
        print("  Examples: 'call mom', 'play music', 'navigate home'")
        print("  Type 'exit' to disconnect")
        print("=" * 60 + "\n")
        
        while self.is_running:
            try:
                command = input("> ").strip()
                
                if not command:
                    continue
                
                if command.lower() == 'exit':
                    print("Disconnecting...")
                    self.is_running = False
                    break
                
                self.send_command(command)
                
            except (EOFError, KeyboardInterrupt):
                print("\n\nDisconnecting...")
                self.is_running = False
                break
            except Exception as e:
                print(f"Input error: {e}")
    
    def start(self):
        """Start the unified server"""
        print("=" * 60)
        print("🎧 SpectraSense Unified Bluetooth Server")
        print("   Commands + Audio")
        print("=" * 60)
        
        # Setup audio
        self.setup_audio_server()
        
        try:
            # Create RFCOMM socket
            self.server_socket = socket.socket(
                socket.AF_BLUETOOTH,
                socket.SOCK_STREAM,
                socket.BTPROTO_RFCOMM
            )
            
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind to any address, channel 4
            self.server_socket.bind(("", RFCOMM_CHANNEL))
            self.server_socket.listen(1)
            
            print(f"\n✅ Server bound to RFCOMM channel {RFCOMM_CHANNEL}")
            print("\n" + "=" * 60)
            print("⏳ Waiting for app connection...")
            print("=" * 60)
            print("\n📱 On your phone:")
            print("   1. Open SpectraSense app")
            print("   2. Go to Server tab")
            print("   3. Tap 'Connect to Server'")
            print("   4. Select this computer from the list")
            print("\n" + "=" * 60 + "\n")
            
            # Accept connection
            self.client_socket, self.client_address = self.server_socket.accept()
            print(f"\n✅ App connected from: {self.client_address}")
            
            # Perform handshake
            print("🔐 Waiting for handshake...")
            self.client_socket.settimeout(5.0)
            
            handshake_data = self.client_socket.recv(1024).decode('utf-8', errors='ignore')
            
            if handshake_data.strip() == SECRET_HANDSHAKE:
                print("✓ Handshake verified!")
                
                # Send access granted
                self.client_socket.sendall("ACCESS_GRANTED".encode('utf-8'))
                self.client_socket.settimeout(None)
                
                # Configure phone audio
                self.configure_phone_audio(self.client_address)
                
                print("\n" + "=" * 60)
                print("✅ CONNECTION ESTABLISHED!")
                print("=" * 60)
                print("\n📋 Active Features:")
                print("   🎮 Commands: Type below to send to phone")
                print("   🔊 Audio: Phone audio plays on computer speakers")
                print("   📞 Calls: Phone calls route to computer")
                print("   🗺️  Navigation: Directions play on computer")
                print("=" * 60 + "\n")
                
                self.is_running = True
                
                # Start receive thread
                self.receive_thread = threading.Thread(
                    target=self.receive_from_app,
                    daemon=True
                )
                self.receive_thread.start()
                
                # Start command input loop
                self.command_input_loop()
                
            else:
                print(f"✗ Invalid handshake: {handshake_data}")
                self.client_socket.close()
                
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")
        except Exception as e:
            print(f"\n❌ Server error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up...")
        
        self.is_running = False
        
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        print("✅ Server stopped\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SpectraSense Unified Server - Commands + Audio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Features:
  🎮 Send commands to phone via app
  🔊 Receive all audio from phone (music, calls, navigation)
  📱 Single connection through SpectraSense app
  
Usage:
  python3 %(prog)s
  
Then in the app:
  - Open SpectraSense app on Android
  - Go to Server tab
  - Connect to this computer
  - Start sending commands OR play audio on phone
        """
    )
    
    server = UnifiedBluetoothServer()
    server.start()


if __name__ == '__main__':
    main()
