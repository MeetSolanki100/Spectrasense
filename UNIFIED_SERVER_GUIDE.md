# SpectraSense Unified Server - Quick Start Guide

## What This Is

A single unified Bluetooth server that provides:
- **Command Server**: Send commands from server to phone via app
- **Audio Server**: Receive ALL audio from phone (music, calls, navigation) on computer speakers

Both features work simultaneously with a single connection through the SpectraSense app.

## Quick Start

### 1. Install Dependencies (One-time setup)

```bash
# Ubuntu/Debian
sudo apt-get install bluez pulseaudio pulseaudio-module-bluetooth python3

# Fedora
sudo dnf install bluez pulseaudio pulseaudio-module-bluetooth python3

# Arch
sudo pacman -S bluez pulseaudio pulseaudio-bluetooth python3
```

### 2. Start the Server

```bash
cd server
python3 unified_bluetooth_server.py
```

You should see:
```
🎧 SpectraSense Unified Bluetooth Server
   Commands + Audio
============================================================
✅ Server bound to RFCOMM channel 4
⏳ Waiting for app connection...
```

### 3. Connect from App

1. Open **SpectraSense app** on Android
2. Go to **Server tab**
3. Tap **"Connect to Server"**
4. Select your computer from the list
5. Wait for connection...

You should see:
```
✅ App connected from: XX:XX:XX:XX:XX:XX
🔐 Waiting for handshake...
✓ Handshake verified!
🎧 Configuring audio for XX:XX:XX:XX:XX:XX...
✅ Phone audio configured - audio will play on computer speakers

============================================================
✅ CONNECTION ESTABLISHED!
============================================================

📋 Active Features:
   🎮 Commands: Type below to send to phone
   🔊 Audio: Phone audio plays on computer speakers
   📞 Calls: Phone calls route to computer
   🗺️  Navigation: Directions play on computer
============================================================
```

### 4. Use It!

**Send Commands:**
```
> call mom
📤 Sending command: call mom
✓ Command sent
```

**Play Audio:**
- Play music on phone → Hear on computer speakers
- Start navigation → Directions on computer
- Receive call → Call audio on computer

## Features

### ✅ Command Mode
- Type commands in terminal
- App executes them on phone
- See execution status

### ✅ Audio Mode
- All phone audio routes to computer automatically
- Music (Spotify, YouTube Music, etc.)
- Navigation (Google Maps directions)
- Phone calls (both incoming and outgoing)
- Notifications and system sounds
- Videos and games

## Connection Flow

```
┌─────────────────────┐
│  SpectraSense App   │
│    (Android)        │
└──────────┬──────────┘
           │
           │ RFCOMM Connection
           │ (Commands)
           │
           │ A2DP Connection
           │ (Audio Stream)
           │
           ↓
┌─────────────────────┐
│   Unified Server    │
│    (Linux)          │
└──────────┬──────────┘
           │
           ├──→ Terminal (Commands)
           └──→ Speakers (Audio)
```

## Differences from Old Servers

### Old Way (Separate Servers)
- `main.py` - Command server only
- `bluetooth_audio_advanced.py` - Audio only, direct Bluetooth pairing
- Two separate connections
- Manual pairing required

### New Way (Unified Server)
- `unified_bluetooth_server.py` - Both in one
- Single connection through app
- Automatic audio configuration
- No manual pairing needed

## Troubleshooting

### "Permission denied" or "Address already in use"
```bash
# Kill any existing Bluetooth processes
sudo systemctl restart bluetooth

# Restart PulseAudio
pulseaudio -k
pulseaudio --start
```

### No Audio Playing
1. Check if phone is playing audio
2. On Android:
   - Go to Settings → Bluetooth
   - Tap gear icon next to connected device
   - Ensure "Media audio" is enabled

3. On Linux, check audio:
```bash
pactl list sinks short
# Should show bluez_sink device when connected
```

### Commands Not Working
- Ensure app is on "Server" tab
- Check server terminal for connection status
- Try reconnecting from app

### Audio Feedback (Hearing Yourself)
Server automatically mutes microphone monitoring. If you still hear feedback:
```bash
# Manual fix
pactl set-source-mute @DEFAULT_SOURCE@ 1
```

## Advanced Usage

### Check Audio Status
```bash
# Show Bluetooth audio devices
pactl list cards short

# Show active sinks
pactl list sinks short

# Show audio sources
pactl list sources short
```

### Run in Background
```bash
nohup python3 unified_bluetooth_server.py > server.log 2>&1 &
```

### Stop Server
- Press `Ctrl+C` in terminal
- Or type `exit` in command prompt
- Or disconnect from app

## What's Happening Behind the Scenes

1. **App connects** via RFCOMM to channel 4
2. **Handshake** with secret key
3. **Server configures PulseAudio** for Bluetooth audio
4. **Phone automatically routes audio** when it detects A2DP Sink
5. **Commands sent/received** via binary protocol
6. **Audio streams** via A2DP profile

## Requirements

- Python 3.7+
- Linux with BlueZ 5.x
- PulseAudio
- Bluetooth adapter
- SpectraSense app on Android

## Comparison Table

| Feature | Old main.py | Old audio server | New Unified |
|---------|-------------|------------------|-------------|
| Commands | ✅ | ❌ | ✅ |
| Audio | ❌ | ✅ | ✅ |
| App Connection | ✅ | ❌ | ✅ |
| Single Server | ❌ | ❌ | ✅ |
| Auto Audio Config | ❌ | Manual | ✅ |

## Next Steps

- Use unified server for all features
- Old servers (`main.py`, `bluetooth_audio_advanced.py`) are still available for specific use cases
- Unified server is recommended for most users

## Support

If issues persist:
1. Check Bluetooth is enabled: `bluetoothctl power on`
2. Check PulseAudio is running: `pulseaudio --check`
3. View detailed logs in terminal
4. Restart both server and app
