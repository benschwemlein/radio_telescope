#!/usr/bin/env python3

"""
Plot averaged rtl_power spectrum and crop off the low-frequency spike.

Assumes rtl_power CSV format:
date, time, start_Hz, end_Hz, step_Hz, samples, dB_bin_0, dB_bin_1, ...
"""

import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Locate CSV file
#   Default: ../data/sun.csv relative to this script
#   Or override with: python plot_sun_cropped.py path/to/file.csv
# -------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = os.path.join(script_dir, "..", "data", "sun.csv")

filename = os.path.normpath(filename)
print("Reading:", filename)

freqs_rows = []
powers_rows = []

with open(filename, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        # Expect at least: date, time, start, end, step, samples, bin0...
        if len(row) < 7:
            continue

        try:
            start_freq = float(row[2])   # Hz
            end_freq   = float(row[3])   # Hz
        except ValueError:
            # Skip header or malformed lines
            continue

        try:
            # Power bins start at column index 6 and are already in dB
            bins = np.array(row[6:], dtype=float)
        except ValueError:
            continue

        if bins.size == 0:
            continue

        # Frequency axis for this sweep
        n_bins = bins.size
        row_freqs = np.linspace(start_freq, end_freq, n_bins, endpoint=False)

        freqs_rows.append(row_freqs)
        powers_rows.append(bins)

if not freqs_rows:
    print("No valid data found in CSV")
    sys.exit(1)

freqs = np.array(freqs_rows)
powers = np.array(powers_rows)

# Average over all sweeps
avg_freq_hz = freqs[0]              # Hz
avg_power_db = np.mean(powers, 0)   # dB (rtl_power already outputs dB)

avg_freq_mhz = avg_freq_hz / 1e6

# -------------------------------------------------------------------
# Crop off the spike region
#   Adjust these limits based on your band and where the spur is
# -------------------------------------------------------------------
MIN_FREQ_MHZ = 1418.2
MAX_FREQ_MHZ = 1423.0

mask = (avg_freq_mhz >= MIN_FREQ_MHZ) & (avg_freq_mhz <= MAX_FREQ_MHZ)

freq_plot = avg_freq_mhz[mask]
power_plot_db = avg_power_db[mask]

if freq_plot.size == 0:
    print("Crop removed all data. Check MIN_FREQ_MHZ / MAX_FREQ_MHZ.")
    sys.exit(1)

print("Freq range (MHz):", freq_plot.min(), "to", freq_plot.max())
print("Power range (dB):", power_plot_db.min(), "to", power_plot_db.max())

# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(freq_plot, power_plot_db)
plt.xlabel("Frequency in MHz")
plt.ylabel("Power (dB, rtl_power units)")
plt.title("Hydrogen Band Scan (rtl_power average, cropped)")
plt.grid(True)
plt.tight_layout()
plt.show()
