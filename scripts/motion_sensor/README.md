# WT901BLECL Motion Sensor Setup Guide

## Radio Telescope Dish Position Monitoring System

This guide will help you set up the WT901BLECL Bluetooth 5.0 motion sensor with your Raspberry Pi 4 to monitor your radio telescope dish position and detect unwanted movement from wind.

---

## Table of Contents
1. [Hardware Requirements](#hardware-requirements)
2. [What You'll Monitor](#what-youll-monitor)
3. [Initial Setup](#initial-setup)
4. [Python Dependencies](#python-dependencies)
5. [Script Installation](#script-installation)
6. [Running the Scripts](#running-the-scripts)
7. [Troubleshooting](#troubleshooting)
8. [Technical Details](#technical-details)

---

## Hardware Requirements

- **WT901BLECL BLE 5.0 Motion Sensor**
  - 9-axis AHRS IMU sensor
  - Measures: 3-axis acceleration, gyroscope, angle, magnetic field
  - Bluetooth 5.0 connectivity
  - Built-in rechargeable battery (~10 hours)
  
- **Raspberry Pi 4** (or newer)
  - Has built-in Bluetooth
  - Running Raspberry Pi OS (Debian Bookworm or newer)

- **Type-C USB Cable** (included with sensor)
  - For charging and wired connection

---

## What You'll Monitor

The sensor provides the following data for your radio telescope:

- **Elevation Angle** (Pitch) - Vertical pointing direction of the dish
- **Azimuth Angle** (Yaw) - Horizontal pointing direction of the dish
- **Roll Angle** - Tilt of the mount (should be ~0° for level installation)
- **Movement Detection** - Detects if wind or other forces are moving the dish
- **Rotation Speed** - Angular velocity in degrees per second

---

## Initial Setup

### 1. Prepare the Sensor

1. **Charge the sensor** using the included Type-C cable
2. **Turn on the sensor** using the power switch
3. **Verify it's working**: The blue LED should flash quickly when searching for connection

### 2. Update Your Raspberry Pi

```bash
sudo apt-get update
sudo apt-get upgrade
```

---

## Python Dependencies

### Install Required System Packages

```bash
sudo apt-get install bluetooth bluez python3-full python3-venv
```

### Install Python Bluetooth Libraries

Raspberry Pi OS uses externally managed Python environments, so install the system packages:

```bash
sudo apt-get install python3-bleak python3-serial
```

**Note:** If you prefer using a virtual environment instead:

```bash
# Install python3-full if not already installed
sudo apt-get install python3-full python3-venv

# Create virtual environment
python3 -m venv ~/wt901_env

# Activate it
source ~/wt901_env/bin/activate

# Install packages in venv
pip install bleak pyserial

# To use this environment in the future, always run:
source ~/wt901_env/bin/activate
```

---

## Script Installation

### 1. Create Scripts Directory

```bash
mkdir -p ~/scripts/motion_sensor
cd ~/scripts/motion_sensor
```

### 2. Copy the Scripts

You should have received two Python scripts:

- **`tilt_detector.py`** - Simplified script for radio telescope monitoring
- **`wti_motion_bluetooth.py`** - Full-featured script with all sensor data

Place these files in `~/scripts/motion_sensor/`

**From your local machine (Mac/PC):**

```bash
# Replace with your Raspberry Pi's IP address
scp tilt_detector.py benschwem@192.168.1.35:/home/benschwem/scripts/motion_sensor/
scp wti_motion_bluetooth.py benschwem@192.168.1.35:/home/benschwem/scripts/motion_sensor/
```

### 3. Make Scripts Executable

```bash
chmod +x ~/scripts/motion_sensor/tilt_detector.py
chmod +x ~/scripts/motion_sensor/wti_motion_bluetooth.py
```

### 4. Update the MAC Address

Each WT901BLE sensor has a unique MAC address. You need to find yours and update the scripts.

**Find your sensor's MAC address:**

```bash
sudo bluetoothctl
power on
scan on
```

Look for a device named **"WT901BLE"** followed by numbers (e.g., `WT901BLE68`). The MAC address will look like `E0:D6:FC:57:08:EF`.

Press `Ctrl+C` to stop scanning, then type `exit` to quit bluetoothctl.

**Update the scripts:**

Edit both Python files and change the `SENSOR_MAC` variable:

```python
SENSOR_MAC = "E0:D6:FC:57:08:EF"  # Replace with YOUR sensor's MAC address
```

---

## Running the Scripts

### Option 1: Radio Telescope Monitor (Recommended)

This is the simplified script designed specifically for dish monitoring:

```bash
cd ~/scripts/motion_sensor
python3 tilt_detector.py
```

**What you'll see:**
```
✓ Stable | Elevation:  45.23° | Azimuth: 180.45° | Roll:  0.12° | Rotation:   0.3°/s
⚠️  MOVING! | Elevation:  45.67° | Azimuth: 180.89° | Roll:  0.15° | Rotation:  12.5°/s
```

**Key Information:**
- **Elevation** = Vertical pointing angle (pitch)
- **Azimuth** = Horizontal pointing angle (yaw)
- **Roll** = Mount tilt (should be ~0° for level mount)
- **Rotation** = Movement speed (alerts if > 5°/s)

Press `Ctrl+C` to stop.

### Option 2: Full Sensor Data

For complete sensor output including all axes:

```bash
cd ~/scripts/motion_sensor
python3 wti_motion_bluetooth.py
```

This displays all acceleration, gyroscope, and angle data.

---

## Troubleshooting

### Sensor Not Found

**Problem:** Script can't find the sensor

**Solutions:**
1. Make sure the sensor is turned ON (blue LED flashing)
2. Verify Bluetooth is enabled: `sudo systemctl status bluetooth`
3. Try moving closer to the sensor (within 30 feet)
4. Re-scan for the MAC address and update your script

### Connection Failed

**Problem:** `Failed to connect` or `Connection timeout`

**Solutions:**
1. Turn the sensor OFF and back ON
2. Restart Bluetooth: `sudo systemctl restart bluetooth`
3. Remove old Bluetooth pairings: 
   ```bash
   sudo bluetoothctl
   devices
   remove E0:D6:FC:57:08:EF  # Use your MAC address
   exit
   ```

### No Data Appearing

**Problem:** Script connects but no data is displayed

**Solutions:**
1. Move the sensor to trigger data transmission
2. Check the sensor battery (charge if LED is red)
3. Try running `wti_motion_bluetooth.py` which shows raw data

### "Externally Managed Environment" Error

**Problem:** Can't install Python packages with pip

**Solution:**
Use the system packages as shown in the [Python Dependencies](#python-dependencies) section, or create a virtual environment.

---

## Technical Details

### Sensor Specifications

- **Model:** WT901BLECL BLE 5.0
- **Chip:** MPU9250
- **Accuracy:** 
  - Angle: ±0.2° (X, Y axes)
  - Gyroscope: ±2000°/s
  - Accelerometer: ±16g
- **Update Rate:** Configurable 0.1Hz - 200Hz (default: 1Hz in tilt_detector.py)
- **Communication:** Bluetooth 5.0 Low Energy
- **Range:** Up to 90 meters (line of sight)
- **Battery:** ~10 hours continuous use
- **Dimensions:** 32.5mm × 23.5mm × 11.4mm

### Data Format

The sensor sends 20-byte packets containing:
- **Bytes 0-1:** Header (0x55, 0x61)
- **Bytes 2-7:** Acceleration (X, Y, Z)
- **Bytes 8-13:** Angular velocity (X, Y, Z)
- **Bytes 14-19:** Angles (Roll, Pitch, Yaw)

### Bluetooth Characteristics

- **Service UUID:** `0000ffe5-0000-1000-8000-00805f9a34fb`
- **Notify Characteristic:** `0000ffe4-0000-1000-8000-00805f9a34fb`
- **Write Characteristic:** `0000ffe9-0000-1000-8000-00805f9a34fb`

### Configuration Commands

The sensor accepts commands via the write characteristic:

- **Unlock:** `FF AA 69 88 B5`
- **Set Rate (1Hz):** `FF AA 03 03 00`
- **Set Rate (10Hz):** `FF AA 03 06 00`
- **Save Settings:** `FF AA 00 00 00`

### Movement Threshold

The default movement detection threshold is **5°/s**. You can adjust this in `tilt_detector.py`:

```python
MOVEMENT_THRESHOLD = 5.0  # Change this value (degrees per second)
```

For a dish, you might want to set it lower (2-3°/s) to detect smaller disturbances.

---

## Running at Startup (Optional)

To have the monitor start automatically when your Raspberry Pi boots:

1. Create a systemd service file:

```bash
sudo nano /etc/systemd/system/dish-monitor.service
```

2. Add the following content:

```ini
[Unit]
Description=Radio Telescope Dish Monitor
After=bluetooth.target

[Service]
Type=simple
User=benschwem
WorkingDirectory=/home/benschwem/scripts/motion_sensor
ExecStart=/usr/bin/python3 /home/benschwem/scripts/motion_sensor/tilt_detector.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

3. Enable and start the service:

```bash
sudo systemctl enable dish-monitor.service
sudo systemctl start dish-monitor.service
```

4. Check status:

```bash
sudo systemctl status dish-monitor.service
```

---

## Additional Resources

- **Manufacturer Documentation:** [WitMotion Wiki](http://wiki.wit-motion.com/english)
- **Product Page:** Search "WT901BLECL" on Amazon or WitMotion store
- **Android App:** "WITMOTION" on Google Play Store
- **iOS App:** "WITMOTION" on Apple App Store

---

## Support

For questions about:
- **Hardware/Sensor:** Contact WitMotion support at support@wit-motion.com
- **Scripts:** Check the Python script comments for inline documentation

---

**Last Updated:** January 2026
**Version:** 1.0