#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

def load_stitched_average(path, drop_bins=12):
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

            if len(bins) <= drop_bins:
                continue

            freqs = start_hz + np.arange(len(bins)) * bin_hz

            bins = bins[drop_bins:]
            freqs = freqs[drop_bins:]

            for fhz, p in zip(freqs, bins):
                k = int(round(fhz))
                acc.setdefault(k, []).append(float(p))

    if not acc:
        raise SystemExit(f"No usable rtl_power rows found in {path}")

    fkeys = np.array(sorted(acc.keys()), dtype=float)
    avgp = np.array([np.mean(acc[int(k)]) for k in fkeys], dtype=float)
    return fkeys, avgp

def main():
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage:\n"
            "  python plot_overlay.py sun.csv no_sun.csv\n"
            "Optional:\n"
            "  python plot_overlay.py sun.csv no_sun.csv 1419.9 1420.9\n"
        )

    sun_path = sys.argv[1]
    nosun_path = sys.argv[2]

    f_sun, p_sun = load_stitched_average(sun_path, drop_bins=12)
    f_nosun, p_nosun = load_stitched_average(nosun_path, drop_bins=12)

    mhz_sun = f_sun / 1e6
    mhz_nosun = f_nosun / 1e6

    if len(sys.argv) == 5:
        lo = float(sys.argv[3])
        hi = float(sys.argv[4])
    else:
        lo = max(mhz_sun.min(), mhz_nosun.min())
        hi = min(mhz_sun.max(), mhz_nosun.max())

    m_sun = (mhz_sun >= lo) & (mhz_sun <= hi)
    m_nosun = (mhz_nosun >= lo) & (mhz_nosun <= hi)

    plt.figure(figsize=(12, 6))
    plt.plot(mhz_sun[m_sun], p_sun[m_sun], label="Sun raw average")
    plt.plot(mhz_nosun[m_nosun], p_nosun[m_nosun], label="No Sun raw average")
    plt.axvline(1420.4058, linestyle="--", linewidth=1)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power (dB)")
    plt.title("Raw power overlay")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
