import numpy as np
import os
import sys
from pathlib import Path
from lowest_detectable_echo import test_detecting_synthetic_echoes, find_lowest_echo


# I think I have to put this in here manually for it to work?
def find_all_quarters(star):
    """
    Given a star's kplr number, returns a list of all quarters of data for that star. Separates data by cadence.
    :param star: kplr number of a star
    :return: list of paths to all quarters of data for that star
    """

    kpath = Path('/home/echo/hdd6tb/02_kepler_time_series_scripts/')

    sc_qtr_list = []
    lc_qtr_list = []
    full_qtr_list = []

    all_folders = os.listdir(kpath)
    quarter_folders = [x for x in all_folders if '_Q' in x]

    for qd in quarter_folders:
        for f in os.listdir(kpath/qd):
            if star in f:
                if 'llc' in f:
                    lc_qtr_list.append(kpath/qd/f)
                elif 'slc' in f:
                    sc_qtr_list.append(kpath/qd/f)
                full_qtr_list.append(kpath/qd/f)

    return lc_qtr_list, sc_qtr_list, full_qtr_list


# Load long cadence data
longcadence = []
with open("longCadence_in_paper.txt", "r") as f:
    for line in f:
        longcadence.append(line.strip())

# Load short cadence data
shortcadence = []
with open("shortCadence_in_paper.txt", "r") as g:
    for line in g:
        shortcadence.append(line.strip())

# Adjust formatting -- remove leading "kplr"
long_cadence_condensed = []
for star in longcadence:
    long_cadence_condensed.append(star[4:])

short_cadence_condensed = []
for star in shortcadence:
    short_cadence_condensed.append(star[4:])

# Now perform the lowest detectable echo search, writing results to files when done.
print("Finding lowest detectable echo for long cadence stars.......")

lc_results = []
for star in long_cadence_condensed:
    star_name, lde, dme = find_lowest_echo(star, 3, "long")
    lc_results.append((star_name, lde, dme))

print("Found all lowest detectable echo strengths for long cadence stars.")
print("Writing to file......")

with open('lc_results.txt', 'w') as f:
    for item in lc_results:
        f.write("%s\n" % item)
    f.close()

print("Wrote long cadence results to file.")

print("----------------------------------------")

print("Finding lowest detectable echo for short cadence stars........")

sc_results = []
for star in short_cadence_condensed:
    star_name, lde, dme = find_lowest_echo(star, 3, "short")
    sc_results.append((star_name, lde, dme))

print("Done with short cadence. Writing to file......")

with open("sc_results.txt", "w") as f:
    for item in sc_results:
        f.write("%s\n" % item)
    f.close()

print("Wrote short cadence results to file.")

print("========= done! :) ==========")

