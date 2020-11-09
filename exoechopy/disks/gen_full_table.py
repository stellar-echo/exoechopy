import numpy as np
from astropy.io import fits
from exoechopy.disks.qtrfind_cm import find_all_quarters
import lightkurve as lk
from scipy.signal import find_peaks
import os
from astropy.table import Table, Column


path = "/home/echo/hdd6tb/02_kepler_time_series_scripts/21_Kepler_Q14/"

all_stars_all_quarters = []
all_stars_long_cadence = []
all_stars_short_cadence = []

# Grab all quarters of data from every star in the directory
for star in os.listdir(path):
    lc, sc, full = find_all_quarters(star)
    all_stars_all_quarters.append(full)
    all_stars_long_cadence.append(lc)
    all_stars_short_cadence.append(sc)

# Now, loop through the data and grab all relevant information
sc_qtr_counts = []
lc_qtr_counts = []

for lc_quarter_list in all_stars_long_cadence:
    lc_qtr_counts.append(len(lc_quarter_list))

for sc_quarter_list in all_stars_short_cadence:
    sc_qtr_counts.append(len(sc_quarter_list))


keys = ["OBJECT", "TEFF", "RADIUS"]
hdu = 0

values = []
total_flares_short_cadence = []
total_flares_long_cadence = []

# Loop through each individual quarter of data and extract flare information
for star_quarter_list in all_stars_all_quarters:
    for ind, quarter_file in enumerate(star_quarter_list):

        # Get header contents from first file in list
        header = fits.getheader(quarter_file[0], hdu)
        values.append([header.get(key) for key in keys])

        # Get number of flares and flare times
        lc_raw = fits.open(str(quarter_file))
        raw_flux = lc_raw[1].data["PDCSAP_FLUX"]
        time = lc_raw[1].data["TIME"]

        lc = lk.LightCurve(time=time, flux=raw_flux)
        lc = lc.remove_nans().flatten()

        # Different cadences require different flare detection windows
        if "slc" in quarter_file:
            x = lc.flux
            median = np.median(x)
            sigma = np.std(x)
            flare_threshold = median + (3 * sigma)
            peaks, peak_val = find_peaks(x, height=flare_threshold, distance=30)
            total_flares_short_cadence.append(len(peaks))

            """
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

            flares_one_percent.append(len(peaks_one))
            flares_four_percent.append(len(peaks_four))

            median_flare_int.append(np.median(flare_heights))

            # Get amount of flares above six sigma
            flare_threshold_six_sigma = median + (6 * sigma)
            peaks_six, peak_val_six = find_peaks(x, height=flare_threshold_six_sigma, distance=30)
            flares_above_6_sigma.append(len(peaks_six))
            """

        else:
            y = lc.flux
            median = np.median(y)
            sigma = np.std(y)
            flare_threshold = median + (6 * sigma)
            peaks, peak_val = find_peaks(y, height=flare_threshold, distance=4)
            total_flares_long_cadence.append(len(peaks))

            """
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

            flares_one_percent.append(len(peaks_one))
            flares_four_percent.append(len(peaks_four))

            median_flare_int.append(np.median(flare_heights))

            # Get amount of flares above six sigma
            flare_threshold_six_sigma = median + (6 * sigma)
            peaks_six, peak_val_six = find_peaks(y, height=flare_threshold_six_sigma, distance=4)
            flares_above_6_sigma.append(len(peaks_six))
            """
        print("Finished", ind, "of", len(star_quarter_list))

# Construct Table
row0 = [dict(zip(keys, values[0]))]
t = Table(row0, names=keys)

for i in range(1, len(values)):
    t.add_row(values[i])

sc_quarters = Column(name="Number of Short Cadence Quarters", data=sc_qtr_counts)
t.add_column(sc_quarters)

lc_quarters = Column(name="Number of Long Cadence Quarters", data=lc_qtr_counts)
t.add_column(lc_quarters)

fls_sc = Column(name="Total Short Cadence Flares", data=total_flares_short_cadence)
t.add_column(fls_sc)

fls_lc = Column(name="Total Long Cadence Flares", data=total_flares_long_cadence)
t.add_column(fls_lc)
