#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
from datetime import datetime

# Usage matches your other script
if len(sys.argv) != 3:
    raise SystemExit("Usage: python drift_hi_power.py file1.csv file2.csv")

# Tunables
CENTER_HZ = 1420.4058e6

# Frequency windows in Hz
# Line window is where we measure HI power
LINE_HALF_HZ = 75_000      # +/- 75 kHz around 1420.4058 MHz

# Baseline windows are used to estimate noise floor near the line
BASE_INNER_HZ = 150_000    # start baseline outside the line window
BASE_OUTER_HZ = 350_000    # end baseline window

# Hop notch cleanup like your other script
DROP_BINS = 12

# Light smoothing in time samples, odd number
SMOOTH_WIN = 21

def parse_time(row0, row1):
    """
    rtl_power usually writes date and time in the first two columns.
    Examples seen in the wild:
      2025-12-24, 15:54:01
      2025-12-24, 15:54:01.123
    If parse fails, return None.
    """
    s = f"{row0} {row1}".strip()
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None

def row_hi_metric(row):
    """
    Returns (t, hi_metric) for one rtl_power row, or (None, None) if unusable.
    hi_metric is line power minus nearby baseline power, in the same units as input.
    """
    if len(row) < 10:
        return None, None

    # Attempt to read timestamp
    t = parse_time(row[0], row[1])

    try:
        start_hz = float(row[2])
        bin_hz = float(row[4])
        bins = np.array(row[6:], dtype=float)
    except ValueError:
        return None, None

    if len(bins) <= DROP_BINS:
        return None, None

    bins = bins[DROP_BINS:]
    freqs = start_hz + (np.arange(len(bins)) + DROP_BINS) * bin_hz

    # Build masks
    df = freqs - CENTER_HZ
    line_mask = np.abs(df) <= LINE_HALF_HZ

    left_base_mask = (df <= -BASE_INNER_HZ) & (df >= -BASE_OUTER_HZ)
    right_base_mask = (df >= BASE_INNER_HZ) & (df <= BASE_OUTER_HZ)
    base_mask = left_base_mask | right_base_mask

    if not np.any(line_mask) or not np.any(base_mask):
        return None, None

    line_val = float(np.mean(bins[line_mask]))
    base_val = float(np.mean(bins[base_mask]))

    hi_metric = line_val - base_val
    return t, hi_metric

def load_series(path):
    times = []
    vals = []

    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            t, v = row_hi_metric(row)
            if v is None:
                continue
            times.append(t)
            vals.append(v)

    if not vals:
        raise SystemExit(f"No usable rows found in {path}")

    vals = np.array(vals, dtype=float)

    # If timestamps are missing, use sample index as time axis
    if all(t is None for t in times):
        x = np.arange(len(vals), dtype=float)
        x_label = "Sample index (time order)"
    else:
        # Replace any None timestamps with sequential placeholders to preserve order
        first_valid = next((t for t in times if t is not None), None)
        if first_valid is None:
            x = np.arange(len(vals), dtype=float)
            x_label = "Sample index (time order)"
        else:
            # Convert to seconds from first valid
            secs = []
            last = first_valid
            for t in times:
                if t is None:
                    secs.append(None)
                else:
                    last = t
                    secs.append((t - first_valid).total_seconds())
            # Fill missing with monotonic steps
            x = np.array([s if s is not None else np.nan for s in secs], dtype=float)
            if np.isnan(x).any():
                # forward fill then interpolate
                mask = ~np.isnan(x)
                x[~mask] = np.interp(np.flatnonzero(~mask), np.flatnonzero(mask), x[mask])
            x_label = "Time (seconds from start)"

    return x, vals, x_label

def smooth(y, win):
    if win <= 1 or len(y) < win or win % 2 == 0:
        return y
    k = np.ones(win) / win
    return np.convolve(y, k, mode="same")

file1 = sys.argv[1]
file2 = sys.argv[2]

print("First file:", file1)
print("Second file:", file2)
print("HI metric: mean(power in +/- 75 kHz around 1420.4058 MHz) minus mean(power in baseline bands 150 to 350 kHz away)")

x1, y1, xlab1 = load_series(file1)
x2, y2, xlab2 = load_series(file2)

# Normalize each series to remove arbitrary offsets so bumps are easier to see
y1n = y1 - np.median(y1)
y2n = y2 - np.median(y2)

y1s = smooth(y1n, SMOOTH_WIN)
y2s = smooth(y2n, SMOOTH_WIN)

plt.figure(figsize=(12, 6))
plt.plot(x1, y1n, alpha=0.25, label="File 1 raw")
plt.plot(x1, y1s, linewidth=2, label="File 1 smoothed")

plt.plot(x2, y2n, alpha=0.25, label="File 2 raw")
plt.plot(x2, y2s, linewidth=2, label="File 2 smoothed")

plt.xlabel(xlab1 if xlab1 == xlab2 else "Time")
plt.ylabel("HI line metric (relative units)")
plt.title("HI line metric vs time order")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
