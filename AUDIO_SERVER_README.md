# Bluetooth Audio Server - README

## What This Does

Turns your Linux computer into a **Bluetooth audio receiver** - just like wireless headphones or a car's Bluetooth system. All audio from your Android phone will play through your computer's speakers.

## Use Cases

✅ **Navigation Audio**: Google Maps directions on computer speakers  
✅ **Music Streaming**: Spotify, YouTube Music on computer  
✅ **Phone Calls**: Answer calls with computer mic/speakers  
✅ **Video Audio**: Watch YouTube videos, audio plays on computer  
✅ **Notifications**: Hear all phone sounds on computer  

## Quick Start (Linux)

### 1. Run Setup Script
```bash
cd server
chmod +x setup_audio_linux.sh
./setup_audio_linux.sh
```

### 2. Start Audio Server
```bash
python3 bluetooth_audio_advanced.py
```

### 3. Connect Phone
- Open Bluetooth settings on Android
- Find "SpectraSense Audio"
- Tap to connect
- Enable "Media audio" in device settings

### 4. Test
Play music on phone → Hear it on computer! 🎵

## Server Options

### Option 1: Audio Only
```bash
python3 bluetooth_audio_advanced.py
```
- Receives audio from phone
- Plays on computer speakers

### Option 2: Commands + Audio (Unified)
```bash
python3 unified_server.py
```
- Sends commands to phone (RFCOMM)
- Receives audio from phone (A2DP)
- Both work simultaneously

### Option 3: Commands Only
```bash
python3 main_fixed.py
```
- Original command server
- No audio receiving

## Requirements

### System
- Linux (Ubuntu/Debian/Fedora/Arch)
- Bluetooth adapter
- Python 3.7+

### Packages
```bash
# Ubuntu/Debian
sudo apt-get install bluez pulseaudio pulseaudio-module-bluetooth

# Fedora
sudo dnf install bluez pulseaudio pulseaudio-module-bluetooth

# Arch
sudo pacman -S bluez pulseaudio pulseaudio-bluetooth
```

## How It Works

```
┌─────────────────┐
│  Android Phone  │
│                 │
│  🎵 Music       │
│  🗺️  Navigation │
│  📞 Calls       │
└────────┬────────┘
         │ Bluetooth A2DP
         │ (Audio Stream)
         ↓
┌─────────────────┐
│ Linux Computer  │
│                 │
│  🔊 Speakers    │
│  🎤 Microphone  │
└─────────────────┘
```

## Configuration

### Make Device Always Discoverable

Edit `/etc/bluetooth/main.conf`:
```ini
[General]
Name = SpectraSense Audio
Class = 0x240404
DiscoverableTimeout = 0
```

Restart Bluetooth:
```bash
sudo systemctl restart bluetooth
```

### Auto-load PulseAudio Modules

Add to `~/.config/pulse/default.pa`:
```
load-module module-bluetooth-discover
load-module module-bluetooth-policy
load-module module-switch-on-connect
```

Restart PulseAudio:
```bash
pulseaudio -k
pulseaudio --start
```

## Troubleshooting

### No Audio Playing

**Check Bluetooth Connection:**
```bash
bluetoothctl devices Connected
```

**Check Audio Sinks:**
```bash
pactl list sinks short
```
Should show a `bluez` device when phone is connected.

**On Android:**
- Go to Bluetooth settings
- Tap gear icon next to device
- Ensure "Media audio" is enabled

### Connection Fails

**Unblock Bluetooth:**
```bash
sudo rfkill unblock bluetooth
```

**Remove Old Pairing:**
```bash
bluetoothctl
remove XX:XX:XX:XX:XX:XX  # Phone's address
exit
```

**Restart Services:**
```bash
sudo systemctl restart bluetooth
pulseaudio -k
pulseaudio --start
```

### Bad Bluetooth Address Error

This error means Bluetooth address format is wrong. **You don't need to provide addresses** - the phone connects to the server automatically.

### Audio Stuttering

**Increase Buffer Size:**
```bash
pactl set-sink-latency bluez_sink.* 50000
```

**Check CPU Usage:**
```bash
top
```
If PulseAudio uses >10% CPU, restart it:
```bash
pulseaudio -k
pulseaudio --start
```

## Advanced Features

### Run on System Boot

Create systemd service:
```bash
sudo nano /etc/systemd/system/spectrasense-audio.service
```

Add:
```ini
[Unit]
Description=SpectraSense Bluetooth Audio Server
After=bluetooth.target pulseaudio.service

[Service]
Type=simple
User=YOUR_USERNAME
ExecStart=/usr/bin/python3 /full/path/to/bluetooth_audio_advanced.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable spectrasense-audio
sudo systemctl start spectrasense-audio
```

### Multiple Devices

Server handles one phone at a time. To switch:
1. Disconnect current phone
2. Connect new phone

### Check Logs

**Bluetooth Logs:**
```bash
journalctl -f -u bluetooth
```

**PulseAudio Logs:**
```bash
journalctl -f --user -u pulseaudio
```

## Audio Quality

- **Codec**: SBC (default) or AAC (if supported)
- **Bitrate**: 328 kbps (SBC) / 256 kbps (AAC)
- **Latency**: 100-200ms (music), 50-100ms (calls)
- **Range**: ~10 meters / 33 feet

## Security

- ✅ Pairing required (prevents unauthorized access)
- ✅ Audio encrypted over Bluetooth
- ✅ No internet connection needed
- ✅ Works completely offline

## Performance

- **CPU**: ~1-2% during playback
- **RAM**: ~50MB
- **Bluetooth**: Uses standard profiles (A2DP, HFP)

## What Doesn't Work

❌ Sending audio FROM computer to phone (use command server instead)  
❌ File transfer (use different Bluetooth profile)  
❌ Screen mirroring (use different technology)  

## Compatibility

### Tested On
- Ubuntu 20.04, 22.04
- Debian 11, 12
- Fedora 38, 39
- Arch Linux (latest)

### Android Versions
- Android 8.0+ (all versions supported)

## Files

- `bluetooth_audio_advanced.py` - Main audio server
- `unified_server.py` - Commands + Audio combined
- `setup_audio_linux.sh` - Setup script
- `BLUETOOTH_AUDIO_SETUP.md` - Detailed guide
- `requirements_linux.txt` - Dependencies list

## Support

For issues, check:
1. `BLUETOOTH_AUDIO_SETUP.md` - Detailed troubleshooting
2. System logs: `journalctl -f -u bluetooth`
3. PulseAudio logs: `pactl list`

## License

MIT License - Free to use and modify

## Credits

Uses standard Linux Bluetooth stack (BlueZ) and PulseAudio
