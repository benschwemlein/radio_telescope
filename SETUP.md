# Hardware & Software Setup

## Raspberry Pi Access

| Method | Details |
|--------|---------|
| IP address | `192.168.1.35` |
| SSH | `ssh benschwem@192.168.1.35` |
| VNC | Run `sudo raspi-config` → Interface Options → VNC → Enabled, then connect with RealVNC Viewer on Windows |

---

## RTL-SDR Blog V4 Driver Installation

The default Raspberry Pi OS package is outdated and does **not** fully support the V4 dongle. Install the official driver from source.

```bash
# Update package lists
sudo apt update

# Install build dependencies
sudo apt install -y git cmake build-essential libusb-1.0-0-dev pkg-config

# Remove the outdated built-in package
sudo apt purge -y rtl-sdr

# Blacklist the kernel's built-in TV tuner driver
# (without this the kernel grabs the dongle and blocks rtl_test / rtl_fm)
echo "blacklist dvb_usb_rtl28xxu" | sudo tee /etc/modprobe.d/blacklist-rtl.conf

# Reboot so the blacklist takes effect
sudo reboot

# After reboot — confirm the Pi sees the dongle
lsusb | grep -i Realtek
```

### Build and install the V4 driver

```bash
cd ~
rm -rf rtl-sdr-blog
git clone https://github.com/rtlsdrblog/rtl-sdr-blog
cd rtl-sdr-blog
mkdir build && cd build
cmake .. -DDETACH_KERNEL_DRIVER=ON
make
sudo make install
sudo ldconfig
```

### Verify the install

```bash
rtl_test -t
```

Expected output:

```
Found 1 device(s):
  0:  RTLSDRBlog, Blog V4, SN: 00000001

Using device 0: Generic RTL2832U OEM
Found Rafael Micro R828D tuner
RTL-SDR Blog V4 Detected
Supported gain values (29): 0.0 0.9 1.4 2.7 ...
Sampling at 2048000 S/s.
No E4000 tuner found, aborting.
```

---

## Hardware Chain

```
NooElec dish feed
      │
      ▼
SAWbird+ H1  (RF IN from dish, RF OUT to dongle)
      │
      ▼  SMA cable
RTL-SDR Blog V4 dongle
      │
      ▼  USB
Raspberry Pi
```

**Powering the SAWbird+ H1 LNA**

The RTL-SDR Blog V4 provides bias-tee power internally — do **not** plug anything into the SAWbird USB port. Just enable the bias tee in software:

```bash
rtl_biast -b 1
```

> **You do not need a separate USB power cable for the SAWbird.**

---

## Quick Function Tests

### Tune into WNCI (FM radio — basic dongle test)

```bash
rtl_fm -f 97.9M -M wbfm -s 200000 -r 48000 -g 30 | aplay -r 48000 -f S16_LE -c 1
```

### Hydrogen-line sweep (indoor bench test)

```bash
# 1. Enable bias tee to power the SAWbird H1 LNA
rtl_biast -b 1

# 2. Sweep 1415–1425 MHz in 1 kHz steps, gain 40 dB, write to test.csv
rtl_power -f 1415M:1425M:1k -g 40 test.csv
# Press Ctrl+C after 20–30 seconds

# 3. Confirm the file was created
ls -lh test.csv
```

Expected `rtl_power` output:

```
Number of frequency hops: 4
Dongle bandwidth: 2500000 Hz
Logged FFT bins: 16384
Reporting every 10 seconds
Tuner gain set to 40.20 dB.
Exact sample rate is ~2500000 Hz
```

### Wideband functional scan (1–1.8 GHz)

```bash
rtl_power -f 1000M:1800M:2M -g 40 wide_scan.csv
```

> PLL-not-locked warnings above ~1.75 GHz are normal and not a failure.

---

## Plotting Results

### On the Raspberry Pi

```bash
# Install matplotlib
sudo apt update
sudo apt install -y python3-matplotlib
```

### Copy your plot script from Windows (PowerShell)

```powershell
scp plot_hydrogen.py benschwem@192.168.1.35:/home/benschwem/scripts/
```

### Run the plot (inside a VNC session — required for the window to display)

```bash
chmod +x ~/scripts/plot_hydrogen.py
python3 ~/scripts/plot_hydrogen.py test.csv
```

### Download a data file from the Pi to Windows

```powershell
scp benschwem@192.168.1.35:/home/benschwem/no_sun_scan_122425_1.csv ~/Downloads/
```

---

## Sun Test

Use two runs to measure signal vs. baseline:

1. **Run 1** — dish pointed at the Sun
2. **Run 2** — dish pointed away (baseline)

```bash
# Enable bias tee
rtl_biast -b 1

# Record narrow scan around 1420 MHz for several minutes
rtl_power -f 1415M:1425M:1k -g 40 sun_scan.csv
# Press Ctrl+C when done

# Plot (run inside VNC)
python3 plot_hydrogen.py sun_scan.csv
```

> `rtl_power` steps the tuner across frequencies, performs short FFTs at each step, and averages the power per bin. Repeated averaging reduces random noise while preserving stable spectral features such as the hydrogen line.

---

## Milky Way Drift Scan

```bash
# Enable bias tee
rtl_biast -b 1

# Long, narrow scan around 1420 MHz hydrogen line — run for 30 min (1800 s)
rtl_power_fftw -f 1419.0M:1421.8M -g 40 -i 5 -e 1800 milkyway_hi.csv
```

### `rtl_power` vs `rtl_power_fftw`

| Tool | Behaviour |
|------|-----------|
| `rtl_power` | Steps the tuner across a frequency range; short FFTs at each step; averages per step |
| `rtl_power_fftw` | Stays on a fixed frequency window; continuously streams samples; uses FFTW for efficient FFTs; averages power in stable frequency bins |

Use **`rtl_power_fftw`** for hydrogen-line observations — it gives finer frequency resolution and better sensitivity on a fixed narrow band.

---

## Using the Celestial App for Scan Planning

See [README.md](README.md) for installation and usage of the Python planning application.

Key workflow:

1. Run `python app.py` from `scripts/celestial_app/`
2. Set your latitude, longitude, and date/time
3. Go to **Radio Telescope → Suggest Optimal Scan…**
4. Set your preferred observation window (e.g., 07:00–23:00)
5. Pick a suggestion and click **Use This Scan →**
6. Adjust parameters if needed and click **Save Scan**
7. The scan band appears on the 3D globe — confirm it crosses the Milky Way band
8. At the scheduled start time, run `rtl_power_fftw` with the matching duration on the Pi
