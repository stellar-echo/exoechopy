import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import lightkurve as lk
from exoechopy.disks.qtrfind_cm import find_all_quarters, plot_all_quarters
from scipy.signal import find_peaks
import os
import sys
from astropy.table import Table, Column
import pandas as pd
from exoechopy.analyze.autocorrelation import autocorrelate_array
from exoechopy.disks import star_info
from astropy.timeseries import LombScargle


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

if num_sc_quarters == 0:
    print("No short cadence data available.")
else:
    for filename in shortcad:
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

if num_lc_quarters == 0:
    print("No long cadence data available.")
else:
    for filename in longcad:
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

# Make full info table
data = [obj_name, num_sc_quarters, num_lc_quarters, sum(num_sc_flares), sum(num_lc_flares), total_flares,
        np.median(median_sc_flare_intensity), np.median(median_lc_flare_intensity), sum(sc_flares_six_sigma),
        sum(lc_flares_six_sigma), sum(sc_flares_one_percent), sum(sc_flares_four_percent), sum(lc_flares_one_percent),
        sum(lc_flares_four_percent)]

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
plt.xlabel("Flare Intensity")
plt.ylabel("Count")
plt.title("Full Flare Histogram")
plt.legend()
plt.savefig("Full Flare Histogram.png")

# Generate Periodogram and Autocorrelation
full_sc_flux = []
for f in shortcad:
    print(f)
    ff = fits.open(f)
    flux = ff[1].data["PDCSAP_FLUX"]
    myflux = flux/np.nanmedian(flux)
    for num in myflux:
        full_sc_flux.append(num)

full_lc_flux = []
for g in longcad:
    print(g)
    gg = fits.open(g)
    flux = gg[1].data["PDCSAP_FLUX"]
    myflux = flux/np.nanmedian(flux)
    for num in myflux:
        full_lc_flux.append(num)

# Full Naive Periodogram
full_sc_time = []
for t in shortcad:
    tt = fits.open(t)
    time = tt[1].data["TIME"]
    for num in time:
        full_sc_time.append(num)

full_lc_time = []
for t in longcad:
    tt = fits.open(t)
    time = tt[1].data["TIME"]
    for num in time:
        full_lc_time.append(num)
        
quicklc = lk.LightCurve(time = full_sc_time, flux = full_sc_flux)
quicklc_long = lk.LightCurve(time = full_lc_time, flux = full_lc_flux)

# Save short cadence flare indices
f_sc = quicklc.flux
median_sc = np.nanmedian(f_sc)
sigma_sc = np.nanstd(f_sc)
flare_threshold_sc = median_sc + (3 * sigma_sc)
peaks_sc, peak_val_sc = find_peaks(f_sc, height=flare_threshold_sc, distance=30)

if num_sc_quarters != 0:
    np.save("short_cadence_flare_indices.npy", peaks_sc)
    
# Save long cadence flare indices
f_lc = quicklc_long.flux
median_lc = np.nanmedian(f_lc)
sigma_lc = np.nanstd(f_lc)
flare_threshold_lc = median_lc + (1.5 * sigma_lc)
peaks_lc, peak_val_lc = find_peaks(f_lc, height=flare_threshold_lc, distance=4)

print(f_lc)
print()
print(peaks_lc)

if num_lc_quarters != 0:
    np.save("long_cadence_flare_indices.npy", peaks_lc)

# If there's any available quarters, save the full flux and time arrays as numpy arrays
if num_sc_quarters != 0:
    np.save("full_sc_flux.npy", full_sc_flux)
    np.save("full_sc_time.npy", full_sc_time)
    
if num_lc_quarters != 0:
    np.save("full_long_cadence_flux.npy", full_lc_flux)
    np.save("full_long_cadence_time.npy", full_lc_time)
    
quicklc = quicklc.remove_nans()
quicklc_long = quicklc_long.remove_nans()   
    
# freq_sc, power_sc = LombScargle(quicklc.time, quicklc.flux).autopower()
# freq_lc, power_lc = LombScargle(quicklc_long.time, quicklc_long.flux).autopower()
"""
plt.figure(figsize=(12, 6))
plt.plot(freq_sc, power_sc, c="b", drawstyle="steps-post")
plt.title("Short Cadence Periodogram")
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.legend()
plt.savefig("short-cadence-periodogram.png")

plt.figure(figsize=(12, 6))
plt.plot(freq_lc, power_lc, c="k", drawstyle="steps-post")
plt.title("Long Cadence Periodogram")
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.legend()
plt.savefig("long-cadence-periodogram.png")
"""
# Full Simple Autocorelation

plt.figure(figsize=(12, 6))

if num_sc_quarters != 0:
    sc_ac = autocorrelate_array(quicklc.flux, max_lag=100)
    plt.plot(sc_ac, c="b", drawstyle="steps-post", label="Short Cadence")

if num_lc_quarters != 0:
    lc_ac = autocorrelate_array(quicklc_long.flux, max_lag=100)
    plt.plot(lc_ac, c="k", drawstyle="steps-post", label="Long Cadence")

plt.legend()
plt.xlabel("Lag Index")
plt.ylabel("Correlation")
plt.title("Full Quarter Autocorrelation")
plt.savefig("full_autocorr.png")

# Make a table with all the quarters, sort of like a breakdown, to find the most active quarters
keys = ["OBSMODE", "QUARTER", "TEFF", "RADIUS"]
hdu = 0

values = []
files = []
raw_variability = []
flares = []
flares_one = []
flares_four = []
median_flare_int = []
flares_above_6_sigma = []

for file_ in full:

    # Get header contents
    header = fits.getheader(file_, hdu)
    values.append([header.get(key) for key in keys])
    files.append(file_)

    # Get Raw Target Variability
    header2 = fits.getheader(file_, 1)
    raw_variability.append(header2.get("PDCVAR"))

    # Get B-V Color estimate
    # gr = header.get("GRCOLOR")
    # if type(gr) != "NoneType":
    #    bv = 0.98 * gr + 0.22

    # bv_color.append(bv)

    # Get number of flares and flare times
    lc_raw = fits.open(str(file_))
    raw_flux = lc_raw[1].data["PDCSAP_FLUX"]
    time = lc_raw[1].data["TIME"]

    lc = lk.LightCurve(time=time, flux=raw_flux)
    lc = lc.remove_nans().flatten()

    # Get raw flux mean, detrended variance and standard deviation, along with 1/sigma sn quality score
    # raw_mean = np.nanmean(raw_flux)
    # raw_means.append(raw_mean)

    # variance = np.var(lc.flux)
    # det_vars.append(variance)

    # sig = np.std(lc.flux)
    # det_sig.append(sig)

    # sn_qual = 1 / sig
    # sn_quality.append(sn_qual)

    # Just for fun, include Raw mean/ Raw sig as well
    # raw_sig = np.nanstd(raw_flux)
    # raw_sn_qual = raw_mean / raw_sig

    # Different cadences require different flare detection windows
    cadence = header.get("OBSMODE")
    if cadence == "short cadence":
        x = lc.flux
        median = np.median(x)
        sigma = np.std(x)
        flare_threshold = median + (3 * sigma)
        peaks, peak_val = find_peaks(x, height=flare_threshold, distance=30)
        flares.append(len(peaks))

        # Get median flare intensity
        flare_heights = []
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

        flares_one.append(len(peaks_one))
        flares_four.append(len(peaks_four))

        median_flare_int.append(np.median(flare_heights))

        # Get amount of flares above six sigma
        flare_threshold_six_sigma = median + (6 * sigma)
        peaks_six, peak_val_six = find_peaks(x, height=flare_threshold_six_sigma, distance=30)
        flares_above_6_sigma.append(len(peaks_six))

        # Convert kep mag to U-band magnitude

    else:
        y = lc.flux
        median = np.median(y)
        sigma = np.std(y)
        flare_threshold = median + (6 * sigma)
        peaks, peak_val = find_peaks(y, height=flare_threshold, distance=4)
        flares.append(len(peaks))

        # Get median flare intensity
        flare_heights = []
        for val in peak_val.values():
            for num in val:
                flare_heights.append(num)

        # Percentage
        peaks_one = []
        peaks_four = []
        for flare in flare_heights:
            raw_val = 100 * (flare - 1)
            if raw_val > 1:
                peaks_one.append(raw_val)

            if raw_val > 4:
                peaks_four.append(raw_val)

        flares_one.append(len(peaks_one))
        flares_four.append(len(peaks_four))

        median_flare_int.append(np.median(flare_heights))

        # Get amount of flares above six sigma
        flare_threshold_six_sigma = median + (6 * sigma)
        peaks_six, peak_val_six = find_peaks(y, height=flare_threshold_six_sigma, distance=4)
        flares_above_6_sigma.append(len(peaks_six))

row0 = [dict(zip(keys, values[0]))]
t2 = Table(row0, names=keys)

for i in range(1, len(values)):
    t2.add_row(values[i])

var = Column(name="Raw Variability", data=raw_variability)
t2.add_column(var)

med = Column(name="Median Flare Intensity", data=median_flare_int)
t2.add_column(med)

fls = Column(name="Number of Flares", data=flares)
t2.add_column(fls)

fls6 = Column(name="Number of Flares Above 6 Sigma", data=flares_above_6_sigma)
t2.add_column(fls6)

fls1 = Column(name="Number of Flares Abouve 1%", data=flares_one)
t2.add_column(fls1)

fls4 = Column(name="Number of Flares Above 4%", data=flares_four)
t2.add_column(fls4)

t2.write("{}_qtr_breakdown.html".format(obj_name), format="ascii.html", overwrite=True)

# Make a breakdown folder for every quarter
for starfile in full:
    star_info.star_info(starfile, lombscarg=False,plot_ind_flares=False)
