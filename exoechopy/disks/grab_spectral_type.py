from astropy.io import fits
from astropy.table import Table
from pathlib import Path
import os
import sys

# Looks like I have to put this in here manually for it to work ¯\_(ツ)_/¯
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


# Load file with short-cadence stars
stars = []
with open("shortCadence_in_paper.txt", "r") as f:
    for line in f:
        stars.append(line.strip())
  
print('loaded file')

# Adjust formatting -- remove leading kplr
short_cadence_stars_condensed = []
for star in stars:
    short_cadence_stars_condensed.append(star[4:])

    
print('adjusted formatting')
    
# Grab quarter data
short = []
for i, star in enumerate(stars):
    long, shortcad, full = find_all_quarters(star)
    short.append(shortcad)
    print("grabbed quarter data for ", i, "of", len(stars))
    
    
# Grab temp for spectral type estimation
keys = ["OBJECT", "TEFF"]
values = []
hdu = 0

# Need just first star from each, this info is constant for all quarters
first_stars = []
for qtr in short:
    first_stars.append(qtr[0])

for i, star in enumerate(first_stars):
    header = fits.getheader(star, hdu)
    values.append([header.get(key) for key in keys])
    print("finished", i, "of", len(first_stars))
    
# Construct and save table
row0 = [dict(zip(keys, values[0]))]
t = Table(row0, names=keys)

for i in range(1, len(values)):
    t.add_row(values[i])

t.write("shortCadence_in_paper_temp_table.html", format="ascii.html", overwrite=True)

