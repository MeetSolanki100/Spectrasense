import socket
import threading
import struct
import sys
import time
import platform

# --- CONFIGURATION ---
SECRET_HANDSHAKE = "SMART_DEVICE_KEY_2024"  # Only the app knows this

# Standard Serial Port Profile (SPP) UUID - universally recognized
SERVICE_UUID = "00001101-0000-1000-8000-00805F9B34FB"
SERVICE_NAME = "SpectraSense Command Server"
# ---------------------

TYPE_TEXT = 1
TYPE_COMMAND = 2  # Commands to send to app

def recv_all(sock, n):
    """Receive exactly n bytes from socket"""
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet: return None
        data += packet
    return data

def send_command(client_sock, command):
    """Send a text command to the app"""
    try:
        cmd_bytes = command.encode('utf-8')
        header = struct.pack('>BI', TYPE_COMMAND, len(cmd_bytes))
        client_sock.send(header + cmd_bytes)
        print(f"✓ Sent command: {command}")
        return True
    except Exception as e:
        print(f"✗ Failed to send: {e}")
        return False

def receive_from_app(client_sock):
    """Listen for status messages from app"""
    try:
        while True:
            header = recv_all(client_sock, 5)
            if not header: break
            
            msg_type, msg_len = struct.unpack('>BI', header)
            payload = recv_all(client_sock, msg_len)
            if not payload: break

            if msg_type == TYPE_TEXT:
                print(f"\n[App Status]: {payload.decode('utf-8')}\n> ", end='', flush=True)
    except Exception as e:
        print(f"\n[Receiver Error]: {e}")

def client_handler(client_sock):
    """ Handles the connection AFTER the handshake is verified. """
    print("\n" + "="*60)
    print("✓ Handshake Verified. App Connected Successfully!")
    print("="*60)
    print("\nYou can now send voice commands to the app.")
    print("Type your command and press Enter. Type 'exit' to disconnect.\n")
    
    # Start receiver thread for app status messages
    receiver = threading.Thread(target=receive_from_app, args=(client_sock,), daemon=True)
    receiver.start()
    
    try:
        # Command input loop
        while True:
            try:
                command = input("> ").strip()
                
                if not command:
                    continue
                    
                if command.lower() in ['exit', 'quit', 'disconnect']:
                    print("Disconnecting...")
                    break
                
                if not send_command(client_sock, command):
                    break
                    
            except EOFError:
                break
                
    except KeyboardInterrupt:
        print("\nDisconnecting...")
    finally:
        print("Connection closed.")
        client_sock.close()

def main():
    if not hasattr(socket, 'AF_BLUETOOTH'):
        print("="*60)
        print("ERROR: Bluetooth not supported on this platform")
        print("Requirements: Windows 10/11 or Linux with Bluetooth")
        print("="*60)
        return

    print("\n" + "="*60)
    print("    SpectraSense Command Server")
    print("="*60)
    print(f"Platform: {platform.system()}")
    print(f"Service UUID: {SERVICE_UUID}")
    print(f"Service Name: {SERVICE_NAME}")
    
    try:
        # Create Bluetooth RFCOMM socket
        server_sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Use a fixed RFCOMM channel for reliable connection
        # Channel 4 is commonly used for custom SPP services
        RFCOMM_CHANNEL = 4
        
        try:
            # Bind to the fixed RFCOMM channel
            server_sock.bind((socket.BDADDR_ANY, RFCOMM_CHANNEL))
            server_sock.listen(1)
            
            print(f"✓ Server started on Bluetooth RFCOMM Channel {RFCOMM_CHANNEL}")
            print(f"✓ Using standard Serial Port Profile (SPP)")
        except OSError as e:
            print(f"✗ ERROR binding to Bluetooth: {e}")
            print(f"💡 Make sure:")
            print(f"   1. Bluetooth is enabled on your PC")
            print(f"   2. No other Bluetooth servers are running")
            print(f"   3. You have Bluetooth permissions")
            server_sock.close()
            return
            
        print(f"✓ Secret handshake: {SECRET_HANDSHAKE}")
        print("\nWaiting for SpectraSense app to connect...")
        print("(Only devices with the secret key can connect)")
        print("\n💡 On your phone:")
        print("   1. Open SpectraSense app")
        print("   2. Go to 'Server' tab")
        print("   3. Tap 'Connect to Server'")
        print("   4. Select this PC from the list")
        print("="*60 + "\n")

        while True:
            # 1. Accept Physical Connection
            client_sock, address = server_sock.accept()
            print(f"\n[{time.strftime('%H:%M:%S')}] Connection attempt from {address}...")

            # 2. SECURITY CHECK (The Handshake)
            try:
                # Wait 2 seconds for the key
                client_sock.settimeout(2.0) 
                
                # Read the key sent by App
                received_key = client_sock.recv(1024).decode('utf-8').strip()
                
                if received_key == SECRET_HANDSHAKE:
                    # SUCCESS: Reset timeout to None (blocking) and start handling
                    client_sock.settimeout(None)
                    client_sock.send("ACCESS_GRANTED".encode('utf-8'))
                    
                    # Give app time to start its reader thread
                    print("[Debug] Waiting 0.5s for app to start reader thread...")
                    time.sleep(0.5)
                    
                    # Handle this client in main thread (blocking)
                    client_handler(client_sock)
                    
                    # After client disconnects, resume listening
                    print("\nReady for next connection...\n")
                else:
                    print(f"✗ WRONG KEY: '{received_key}'. Rejected.")
                    client_sock.send("ACCESS_DENIED".encode('utf-8'))
                    client_sock.close()
                    
            except socket.timeout:
                print("✗ Timeout: No key received. Rejected.")
                client_sock.close()
            except Exception as e:
                print(f"✗ Handshake Error: {e}")
                client_sock.close()

    except KeyboardInterrupt:
        print("\n\nShutting down server...")
    except Exception as e:
        print(f"\n❌ Server Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            server_sock.close()
        except:
            pass
        print("Server closed.\n")

if __name__ == "__main__":
    main()