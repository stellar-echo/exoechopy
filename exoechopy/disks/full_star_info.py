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


# Save command line input
star = sys.argv[1]

# Grab all quarters, store in short cadence, long cadence, and full lists
longcad, shortcad, full = find_all_quarters(star)

# Grab object name, teff, radius
file0 = fits.open(str(full[0]))
obj_name = file0[1].header["OBJECT"]
#teff = file0[1].header["TEFF"]
#radius = file0[1].header["RADIUS"]

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

        lc_flares_one_percent.append(len(peaks_one))
        lc_flares_four_percent.append(len(peaks_four))

        median_lc_flare_intensity.append(np.median(flare_heights))

        # Get amount of flares above six sigma
        flare_threshold_six_sigma = median + (6 * sigma)
        peaks_six, peak_val_six = find_peaks(x, height=flare_threshold_six_sigma, distance=30)
        lc_flares_six_sigma.append(len(peaks_six))

        

t = Table(data=[obj_name, num_sc_quarters, num_lc_quarters, num_sc_flares, num_lc_flares])

#obj = Column(name="Object", data=obj_name)
#t.add_column(obj)

#scq = Column(name="Number of Short Cadence Quarters", data=num_sc_quarters)
#t.add_column(scq)

#lcq = Column(name="Number of Long Cadence Quarters", data=num_lc_quarters)
#t.add_column(lcq)

#scf = Column(name="Short Cadence Flares", data=num_sc_flares)
#t.add_column(scf)

#lcf = Column(name="Long Cadence Flares", data=num_lc_flares)
#t.add_column(lcf)


t.write("{}_full_info.html".format(star), format="ascii.html", overwrite=True)
