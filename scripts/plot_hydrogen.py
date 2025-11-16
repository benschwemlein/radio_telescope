#!/usr/bin/env python3

# This script reads an rtl_power CSV file and plots the averaged spectrum.
# It handles the rtl_power format: timestamp, start_freq, end_freq, step, FFT bins...

import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

# Change this to your filename, or pass as argument
filename = "test.csv"
if len(sys.argv) > 1:
    filename = sys.argv[1]

freqs = []
powers = []

with open(filename, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        # Skip empty or malformed rows
        if len(row) < 5:
            continue

        try:
            start_freq = float(row[2])     # col 2 = start frequency
            bin_hz = float(row[3]) - float(row[2])
            step = float(row[3]) - float(row[2]) 
        except ValueError:
            continue

        # FFT bins begin at column 4
        bins = np.array(row[4:], dtype=float)

        # Generate frequency scale for this row
        row_freqs = np.linspace(float(row[2]), float(row[3]), len(bins))

        freqs.append(row_freqs)
        powers.append(bins)

# Convert lists to arrays
freqs = np.array(freqs)
powers = np.array(powers)

# Average the power across all sweeps
avg_freq = freqs[0]
avg_power = np.mean(powers, axis=0)

# Plot
plt.figure(figsize=(12,6))
plt.plot(avg_freq / 1e6, avg_power)
plt.xlabel("Frequency in MHz")
plt.ylabel("Power")
plt.title("Hydrogen Band Scan (rtl_power average)")
plt.grid(True)
plt.show()
