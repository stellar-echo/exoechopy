from exoechopy.disks.qtrfind_cm import find_all_quarters
from astropy.io import fits
from astropy.table import Table

# Load file with short-cadence stars
stars = []
with open("shortCadence_in_paper.txt", "r") as f:
    for line in f:
        stars.append(line.strip())

# Adjust formatting -- remove leading kplr
short_cadence_stars_condensed = []
for star in stars:
    short_cadence_stars_condensed.append(star[4:])

# Grab quarter data
short = []
for star in stars:
    long, shortcad, full = find_all_quarters(star)
    short.append(shortcad)

# Grab temp for spectral type estimation
keys = ["OBJECT", "TEFF"]
values = []
hdu = 0

# Need just first star from each, this info is constant for all quarters
first_stars = []
for qtr in short:
    first_stars.append(qtr[0])

for star in first_stars:
    header = fits.getheader(star, hdu)
    values.append([header.get(key) for key in keys])

# Construct and save table
row0 = [dict(zip(keys, values[0]))]
t = Table(row0, names=keys)

for i in range(1, len(values)):
    t.add_row(values[i])

t.write("shortCadence_in_paper_temp_table.html", format="ascii.html", overwrite=True)

