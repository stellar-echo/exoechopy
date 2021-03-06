import numpy as np
from astropy.io import fits
# from exoechopy.disks.qtrfind_cm import find_all_quarters
import lightkurve as lk
from scipy.signal import find_peaks
import os
import sys
from pathlib import Path
from astropy.table import Table, Column
import re

def find_all_quarters(star):
    """
    Given a star's kplr number, returns a list of all quarters of data for that star. Separates data by cadence.

    :param file: fits file from the kepler server
    :return: list of paths to all quarters of data for that star
    """
   
    # Extract the kplr number
    # star = sys.argv[1]
    
    kpath = Path('/home/echo/hdd6tb/02_kepler_time_series_scripts/')

    sc_qtr_list = []
    lc_qtr_list = []
    full_qtr_list = []

    all_folders = os.listdir(kpath)
    quarter_folders = [x for x in all_folders if '_Q' in x]
    # print(quarter_folders)
    # print()

    for qd in quarter_folders:
        for f in os.listdir(kpath/qd):
            if star in f:
                if 'llc' in f:
                    lc_qtr_list.append(kpath/qd/f)
                elif 'slc' in f:
                    sc_qtr_list.append(kpath/qd/f)
                full_qtr_list.append(kpath/qd/f)

    return lc_qtr_list, sc_qtr_list, full_qtr_list


# path = "/home/echo/hdd6tb/02_kepler_time_series_scripts/21_Kepler_Q14/"

# These should be lists of lists of paths
all_stars_all_quarters = []
all_stars_long_cadence = []
all_stars_short_cadence = []

print("Grabbing all quarter files for every star....")


top3k = []
with open("top3k.txt","r") as f:
    for line in f:
        top3k.append(line.strip())
        
top_kplr = []
for file in top3k:
    s = re.search("kplr(.+?)-", file)
    top_kplr.append(s.group(1))
        
print(top_kplr[0:10])

# Grab all quarters of data from every star in the directory
for ind, kplr in enumerate(top_kplr):
    lc, sc, full = find_all_quarters(kplr)
    all_stars_all_quarters.append(full)
    all_stars_long_cadence.append(lc)
    all_stars_short_cadence.append(sc)
    print(ind)
    print("Grabbed all quarter data for", kplr, "({} of {})".format(ind, len(top_kplr)))

print()
print("Grabbed all quarter files for every star.")
print()
print("Now looping through the quarter files and grabbing flare info...")
    
# Now, loop through the data and grab all relevant information
sc_qtr_counts = []
lc_qtr_counts = []

# Count the number of quarters for each star
for ind, lc_quarter_list in enumerate(all_stars_long_cadence):
    lc_qtr_counts.append(len(lc_quarter_list))
    print("Counted", ind, "of", len(all_stars_long_cadence), "long cadence stars")
    print("{} LC Quarters Available".format(len(lc_quarter_list)))

for ind, sc_quarter_list in enumerate(all_stars_short_cadence):
    sc_qtr_counts.append(len(sc_quarter_list))
    print("Counted", ind, "of", len(all_stars_short_cadence), "short cadence stars")
    print("{} SC Quarters Available".format(len(sc_quarter_list)))

keys = ["OBJECT", "TEFF", "RADIUS"]
hdu = 0

values = []
total_flares_short_cadence = []
total_flares_long_cadence = []

print(all_stars_all_quarters[0:10])
print(sc_qtr_counts[0])
print(lc_qtr_counts[0])
print(all_stars_all_quarters[0][0])

first_stars = []

# Loop through each individual quarter of data and extract flare information
for sql in all_stars_all_quarters:
    first_stars.append(sql[0])

for initial_star in first_stars:
    # Get header contents from first file in list
    header = fits.getheader(initial_star, hdu)
    values.append([header.get(key) for key in keys])
    
for index, star_quarter_list in enumerate(all_stars_all_quarters):
    for ind, quarter_file in enumerate(star_quarter_list):
        # Get number of flares and flare times
        lc_raw = fits.open(str(quarter_file))
        hdr = fits.getheader(quarter_file, hdu)
        raw_flux = lc_raw[1].data["PDCSAP_FLUX"]
        time = lc_raw[1].data["TIME"]

        lc = lk.LightCurve(time=time, flux=raw_flux)
        lc = lc.remove_nans().flatten()

        # Different cadences require different flare detection windows
        if header.get("OBSMODE") == "short cadence":
            # Store the amount of flares from each quarter in a list for every star - this way we can easily get the sum for each star
            sc_flare_list = []
            
            x = lc.flux
            median = np.median(x)
            sigma = np.std(x)
            flare_threshold = median + (3 * sigma)
            peaks, peak_val = find_peaks(x, height=flare_threshold, distance=30)
            sc_flare_list.append(len(peaks))

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
            lc_flare_list = []
            
            y = lc.flux
            median = np.median(y)
            sigma = np.std(y)
            flare_threshold = median + (6 * sigma)
            peaks, peak_val = find_peaks(y, height=flare_threshold, distance=4)
            lc_flare_list.append(len(peaks))

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
    print("Finished", index, "of", len(all_stars_all_quarters))
    total_flares_short_cadence.append(sc_flare_list)
    total_flares_long_cadence.append(lc_flare_list)

flare_totals_sc_final = []
for starlist in total_flares_short_cadence:
    flare_totals_sc_final.append(sum(starlist))

flare_totals_lc_final = []
for starlist in total_flares_long_cadence:
    flare_totals_lc_final.append(sum(starlist))
    
# active_sc_quarters = []
# for starlist in total_flares_short_cadence:
#     active = 0
#    for qtr in starlist:
#        if qtr > 10:
#            active += 1
#    active_sc_quarters.append(active)

#active_lc_quarters = []
#for starlist in total_flares_long_cadence:
#    active = 0
    
# Construct Table
row0 = [dict(zip(keys, values[0]))]
t = Table(row0, names=keys)

for i in range(1, len(values)):
    t.add_row(values[i])

sc_quarters = Column(name="Number of Short Cadence Quarters", data=sc_qtr_counts)
t.add_column(sc_quarters)

lc_quarters = Column(name="Number of Long Cadence Quarters", data=lc_qtr_counts)
t.add_column(lc_quarters)

fls_sc = Column(name="Total Short Cadence Flares", data=flare_totals_sc_final)
t.add_column(fls_sc)

fls_lc = Column(name="Total Long Cadence Flares", data=flare_totals_lc_final)
t.add_column(fls_lc)
  
print(total_flares_short_cadence[0:10])
print(total_flares_long_cadence[0:10])
t.write("full_table_test.html", format="ascii.html", overwrite=True)
