"""
SpectraSense Bluetooth Command Server
Production-ready server for sending voice commands to Android app via Bluetooth RFCOMM

Requirements:
- Windows 10/11 with Bluetooth
- Phone paired with PC in Windows Bluetooth settings
- Python 3.7+

Usage:
1. Pair phone with PC in Windows Bluetooth settings
2. Run: python main_fixed.py
3. Open SpectraSense app → Server tab → Connect
4. Type commands in this terminal
"""

import socket
import threading
import struct
import time

# Configuration
SECRET_HANDSHAKE = "SMART_DEVICE_KEY_2024"
TYPE_TEXT = 1
TYPE_COMMAND = 2

def recv_all(sock, n):
    """Receive exactly n bytes"""
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def send_command(client_sock, command):
    """Send command to app"""
    try:
        cmd_bytes = command.encode('utf-8')
        header = struct.pack('>BI', TYPE_COMMAND, len(cmd_bytes))
        client_sock.send(header + cmd_bytes)
        print(f"✓ Sent: {command}")
        return True
    except Exception as e:
        print(f"✗ Send failed: {e}")
        return False

def receive_from_app(client_sock):
    """Listen for app status messages"""
    try:
        while True:
            header = recv_all(client_sock, 5)
            if not header:
                break
            
            msg_type, msg_len = struct.unpack('>BI', header)
            payload = recv_all(client_sock, msg_len)
            if not payload:
                break

            if msg_type == TYPE_TEXT:
                status = payload.decode('utf-8')
                print(f"\n[App]: {status}\n> ", end='', flush=True)
    except:
        pass

def handle_client(client_sock):
    """Handle connected client"""
    print("\n" + "="*60)
    print("✅ CONNECTED! App authenticated successfully!")
    print("="*60)
    print("\n📱 You can now send commands to your phone.")
    print("💡 Type a command and press Enter.")
    print("💡 Type 'exit' to disconnect.\n")
    
    # Start receiver thread
    receiver = threading.Thread(target=receive_from_app, args=(client_sock,), daemon=True)
    receiver.start()
    
    # Command loop
    try:
        while True:
            command = input("> ").strip()
            
            if not command:
                continue
                
            if command.lower() in ['exit', 'quit', 'q']:
                print("Disconnecting...")
                break
            
            if not send_command(client_sock, command):
                break
                
    except (KeyboardInterrupt, EOFError):
        print("\nDisconnecting...")
    finally:
        client_sock.close()
        print("Connection closed.\n")

def main():
    print("\n" + "="*70)
    print("  SpectraSense Bluetooth Command Server")
    print("="*70)
    
    # Check Bluetooth support
    if not hasattr(socket, 'AF_BLUETOOTH'):
        print("❌ ERROR: Bluetooth not supported on this system")
        print("💡 Requires Windows 10/11 or Linux with Bluetooth")
        return 1
    
    try:
        # Create Bluetooth RFCOMM socket
        server_sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Bind to first available port (Windows assigns automatically)
        # We try multiple ports until one works
        port = None
        last_error = None
        for try_port in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
            try:
                # Try binding with explicit local address
                server_sock.bind((socket.BDADDR_ANY, try_port))
                port = try_port
                print(f"✅ Bound to RFCOMM channel {port}")
                break
            except OSError as e:
                last_error = e
                continue
        
        if port is None:
            print("❌ ERROR: Could not bind to any Bluetooth port")
            print(f"   Last error: {last_error}")
            print("\n💡 SOLUTION:")
            print("   Windows requires Bluetooth server apps to run with proper permissions")
            print("   Try these steps:")
            print("   1. Close this program")
            print("   2. Right-click PowerShell/Terminal")
            print("   3. Select 'Run as Administrator'")
            print("   4. Run the server again")
            print("\n   OR use the phone as server (app → server mode)")
            return 1
        
        server_sock.listen(1)
        
        print(f"✅ Server started successfully!")
        print(f"🔐 Security: Handshake authentication enabled")
        print(f"\n📱 CONNECT FROM YOUR PHONE:")
        print(f"   1. Make sure phone is paired with this PC")
        print(f"   2. Open SpectraSense app")
        print(f"   3. Tap 'Server' tab at bottom")
        print(f"   4. Tap 'Connect to Server'")
        print(f"   5. Select this PC from list")
        print("="*70)
        print(f"\n⏳ Waiting for connection...\n")
        
        while True:
            # Accept connection
            client_sock, address = server_sock.accept()
            print(f"[{time.strftime('%H:%M:%S')}] 📞 Incoming connection from {address}")
            
            # Authenticate with handshake
            try:
                print(f"⏳ Waiting up to 5 seconds for handshake...")
                client_sock.settimeout(5.0)
                
                received_data = client_sock.recv(1024)
                print(f"📦 Received {len(received_data)} bytes: {received_data}")
                
                received_key = received_data.decode('utf-8').strip()
                print(f"🔑 Decoded key: '{received_key}' (length: {len(received_key)})")
                print(f"🔑 Expected key: '{SECRET_HANDSHAKE}' (length: {len(SECRET_HANDSHAKE)})")
                
                if received_key == SECRET_HANDSHAKE:
                    client_sock.settimeout(None)
                    client_sock.send("ACCESS_GRANTED".encode('utf-8'))
                    handle_client(client_sock)
                    print("\n⏳ Ready for next connection...\n")
                else:
                    print(f"❌ Wrong handshake: '{received_key}' - Connection rejected")
                    client_sock.send("ACCESS_DENIED".encode('utf-8'))
                    client_sock.close()
                    
            except socket.timeout:
                print("❌ Timeout - No handshake received")
                client_sock.close()
            except Exception as e:
                print(f"❌ Auth error: {e}")
                client_sock.close()
                
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        return 1
    finally:
        try:
            server_sock.close()
        except:
            pass
        print("Server closed.\n")
    
    return 0

if __name__ == "__main__":
    exit(main())
