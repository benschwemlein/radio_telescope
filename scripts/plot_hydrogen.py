import numpy as np
import matplotlib.pyplot as plt
import csv

# Load RTL_POWER CSV
filename = "test.csv"

freqs = []
powers = []

with open(filename, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0].startswith("#"):
            continue
        start_freq = float(row[0])
        step = float(row[2])
        bins = row[6:]
        for i, b in enumerate(bins):
            freqs.append(start_freq + i * step)
            powers.append(float(b))

freqs = np.array(freqs) / 1e6  # Convert to MHz
powers = np.array(powers)

plt.figure(figsize=(12,6))
plt.plot(freqs, powers)
plt.xlabel("Frequency MHz")
plt.ylabel("Power dB")
plt.title("Hydrogen Band Spectrum 1415 to 1425 MHz")
plt.grid(True)
plt.show()
