import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import lightkurve as lk
from exoechopy.visualize import plot_flare_array
from exoechopy.disks.qtrfind_cm import find_all_quarters, plot_all_quarters
from exoechopy.disks import star_info
from scipy.signal import find_peaks
import os
import sys
from astropy.table import Table, Column
import pandas as pd
from exoechopy.analyze.autocorrelation import autocorrelate_array

# Save command line input
star = sys.argv[1]

# Grab all quarters, store in short cadence, long cadence, and full lists
longcad, shortcad, full = find_all_quarters(star)

# Grab object name, teff, radius
file0 = fits.open(str(full[0]))
obj_name = file0[1].header["OBJECT"]
# teff = file0[1].header["TEFF"]
# radius = file0[1].header["RADIUS"]

# Create a new directory to store all of this information
newdir = "allqtr_star_info/{}".format(obj_name)

if not os.path.exists(newdir):
    os.makedirs(newdir)

os.chdir(newdir)

# Save plots of short and long cadence data
plot_all_quarters(star)

# Count the number of quarters available
num_sc_quarters = len(shortcad)
num_lc_quarters = len(longcad)

# Grab flare information from each quarter
num_sc_flares = []
median_sc_flare_intensity = []
sc_flares_one_percent = []
sc_flares_four_percent = []
sc_flares_six_sigma = []
flare_heights = []

for filename in shortcad:
    if len(shortcad) == 0:
        print("No short cadence data.")
    else:
        lc_raw = fits.open(str(filename))
        raw_flux = lc_raw[1].data["PDCSAP_FLUX"]
        time = lc_raw[1].data["TIME"]

        lc = lk.LightCurve(time=time, flux=raw_flux)
        lc = lc.remove_nans().flatten()

        x = lc.flux
        median = np.median(x)
        sigma = np.std(x)
        flare_threshold = median + (3 * sigma)
        peaks, peak_val = find_peaks(x, height=flare_threshold, distance=30)
        num_sc_flares.append(len(peaks))

        # Get median flare intensity
        for val in peak_val.values():
            for num in val:
                flare_heights.append(num)

        # Calculate the percentage above background flux of each flare
        peaks_one = []
        peaks_four = []
        for flare in flare_heights:
            raw_val = 100 * (flare - 1)
            if raw_val > 1:
                peaks_one.append(raw_val)

            if raw_val > 4:
                peaks_four.append(raw_val)

        sc_flares_one_percent.append(len(peaks_one))
        sc_flares_four_percent.append(len(peaks_four))

        median_sc_flare_intensity.append(np.median(flare_heights))

        # Get amount of flares above six sigma
        flare_threshold_six_sigma = median + (6 * sigma)
        peaks_six, peak_val_six = find_peaks(x, height=flare_threshold_six_sigma, distance=30)
        sc_flares_six_sigma.append(len(peaks_six))

# Long Cadence
num_lc_flares = []
median_lc_flare_intensity = []
lc_flares_one_percent = []
lc_flares_four_percent = []
lc_flares_six_sigma = []
flare_heights_lc = []

for filename in longcad:
    if len(longcad) == 0:
        print("No long cadence data.")
    else:
        lc_raw = fits.open(str(filename))
        raw_flux = lc_raw[1].data["PDCSAP_FLUX"]
        time = lc_raw[1].data["TIME"]

        lc = lk.LightCurve(time=time, flux=raw_flux)
        lc_detr = lc.remove_nans().flatten()

        x = lc_detr.flux
        median = np.median(x)
        sigma = np.std(x)
        flare_threshold = median + (3 * sigma)
        peaks, peak_val = find_peaks(x, height=flare_threshold, distance=4)
        num_lc_flares.append(len(peaks))

        # Get median flare intensity
        for val in peak_val.values():
            for num in val:
                flare_heights_lc.append(num)

        # Calculate the percentage above background flux of each flare
        peaks_one = []
        peaks_four = []
        for flare in flare_heights_lc:
            raw_val = 100 * (flare - 1)
            if raw_val > 1:
                peaks_one.append(raw_val)

            if raw_val > 4:
                peaks_four.append(raw_val)

        lc_flares_one_percent.append(len(peaks_one))
        lc_flares_four_percent.append(len(peaks_four))

        median_lc_flare_intensity.append(np.median(flare_heights_lc))

        # Get amount of flares above six sigma
        flare_threshold_six_sigma = median + (6 * sigma)
        peaks_six, peak_val_six = find_peaks(x, height=flare_threshold_six_sigma, distance=30)
        lc_flares_six_sigma.append(len(peaks_six))


total_flares = sum(num_sc_flares) + sum(num_lc_flares)

data = [obj_name, num_sc_quarters, num_lc_quarters, sum(num_sc_flares), sum(num_lc_flares), total_flares,
        median_sc_flare_intensity, median_lc_flare_intensity, sc_flares_six_sigma, lc_flares_six_sigma,
        sc_flares_one_percent, sc_flares_four_percent, lc_flares_one_percent, lc_flares_four_percent]

names = ["Object", "# SC Quarters", "# LC Quarters", "# SC Flares", "# LC Flares", "Total Flares",
         "Median SC Flare Intensity", "Median LC Flare Intensity", "SC Flares Above 6 Sigma", "LC Flares Above 6 Sigma",
         "SC Flares Above 1%", "SC Flares Above 4%", "LC Flares Above 1%", "LC Flares Above 4%"]

df = pd.DataFrame([data], columns=names)
t = Table.from_pandas(df)
t.write("{}_full_info.html".format(star), format="ascii.html", overwrite=True)

# Generate Histogram of Flare Intensities
plt.figure(figsize=(12, 6))
plt.hist(flare_heights, bins=100, alpha=0.6, label="Short Cadence Flares")
plt.hist(flare_heights_lc, bins=100, alpha=0.6, label="Long Cadence Flares")
plt.xlabel("Intensity")
plt.ylabel("Count")
plt.title("Full Flare Histogram")
plt.legend()
plt.savefig("Full Flare Histogram.png")

