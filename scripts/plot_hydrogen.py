#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

filename = sys.argv[1] if len(sys.argv) > 1 else "test.csv"

acc = {}

DROP_BINS = 5   # drop first 5 bins of each hop (increase to 10 if needed)

with open(filename, "r", newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 10:
            continue

        try:
            start_hz = float(row[2])
            end_hz   = float(row[3])
            bin_hz   = float(row[4])
            bins     = np.array(row[6:], dtype=float)
        except ValueError:
            continue

        bins = bins[DROP_BINS:]  # kill hop spike
        freqs = start_hz + (np.arange(len(bins)) + DROP_BINS) * bin_hz

        for f_hz, p in zip(freqs, bins):
            k = int(round(f_hz))
            acc.setdefault(k, []).append(p)

# average
freqs = np.array(sorted(acc.keys()), dtype=float)
power = np.array([np.mean(acc[int(k)]) for k in freqs])

freq_mhz = freqs / 1e6

# focus hydrogen
center = 1420.4058
span = 5.0
mask = (freq_mhz > center - span) & (freq_mhz < center + span)

x = freq_mhz[mask]
y = power[mask]

# robust vertical scale
lo, hi = np.percentile(y, [10, 90])

plt.figure(figsize=(12,6))
plt.plot(x, y)
plt.xlabel("Frequency (MHz)")
plt.ylabel("Power (dB)")
plt.title("Hydrogen scan (hop-edge removed)")
plt.ylim(lo, hi)
plt.grid(True)
plt.tight_layout()
plt.show()
