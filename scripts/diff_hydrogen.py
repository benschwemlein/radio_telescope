#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

if len(sys.argv) != 3:
    raise SystemExit("Usage: python diff_hydrogen.py sun.csv no_sun.csv")

DROP_BINS = 12  # increase to 20 if hop notches still dominate
SMOOTH_WIN = 81 # odd number, increase to 151 for more smoothing

def load_stitched(path):
    acc = {}
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if len(row) < 10:
                continue
            try:
                start_hz = float(row[2])
                bin_hz = float(row[4])
                bins = np.array(row[6:], dtype=float)
            except ValueError:
                continue

            if len(bins) <= DROP_BINS:
                continue

            bins = bins[DROP_BINS:]
            freqs = start_hz + (np.arange(len(bins)) + DROP_BINS) * bin_hz

            for fhz, p in zip(freqs, bins):
                k = int(round(fhz))
                acc.setdefault(k, []).append(float(p))

    if not acc:
        raise SystemExit(f"No usable rows found in {path}")

    fkeys = np.array(sorted(acc.keys()), dtype=int)
    avgp = np.array([np.mean(acc[k]) for k in fkeys], dtype=float)
    return fkeys, avgp

sun_path = sys.argv[1]
nosun_path = sys.argv[2]

print("First file treated as SUN:", sun_path)
print("Second file treated as NO SUN:", nosun_path)

fs, ps = load_stitched(sun_path)
fn, pn = load_stitched(nosun_path)

common = np.intersect1d(fs, fn)
if len(common) < 500:
    raise SystemExit("Not enough overlapping frequencies between files")

sun_map = dict(zip(fs, ps))
nosun_map = dict(zip(fn, pn))

freqs = common.astype(float)
diff = np.array([sun_map[int(f)] - nosun_map[int(f)] for f in common], dtype=float)
diff = diff - np.median(diff)

freq_mhz = freqs / 1e6

center = 1420.4058
span = 2.0
m = (freq_mhz >= center - span) & (freq_mhz <= center + span)

x = freq_mhz[m]
y = diff[m]

if len(y) >= SMOOTH_WIN:
    k = np.ones(SMOOTH_WIN) / SMOOTH_WIN
    y_s = np.convolve(y, k, mode="same")
else:
    y_s = y

plt.figure(figsize=(12, 6))
plt.plot(x, y, alpha=0.25, label="Sun minus No Sun raw")
plt.plot(x, y_s, label="Sun minus No Sun smoothed")
plt.axvline(center, linestyle="--", linewidth=1)
plt.xlabel("Frequency (MHz)")
plt.ylabel("Delta power (dB)")
plt.title("Hydrogen difference spectrum")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
