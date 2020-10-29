from astropy.table import Table, vstack
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from exoechopy.disks import star_info

# star_info.star_info("kepler_test/kplr009726699-2010203174610_slc.fits", hist=True, plot_ac=True, plot_ind_flares=True,
#                   lombscarg=True)


# Read in table
tbl0 = Table.read("kepler_q0.html")

# Make new spectral type column, based on temperature
# Source: https://sites.uni.edu/morgans/astro/course/Notes/section2/spectraltemps.html


def spectral_type(row):
    if row["TEFF"] > 37800:
        return "O"
    elif 11400 <= row["TEFF"] < 37800:
        return "B"
    elif 7920 <= row["TEFF"] < 11400:
        return "A"
    elif 6300 <= row["TEFF"] < 7920:
        return "F"
    elif 5440 <= row["TEFF"] < 6300:
        return "G"
    elif 4000 <= row["TEFF"] < 5440:
        return "K"
    elif 2700 <= row["TEFF"] < 4000:
        return "M"
    elif row["TEFF"] < 2700:
        return "L"
    else:
        return "?"


# Convert astropy table to pandas df
df = tbl0.to_pandas()
df = df.dropna()

# Test spectral type function
df["Spectral Type Estimate"] = df.apply(lambda row: spectral_type(row), axis=1)

tbl0 = Table.from_pandas(df)

print(tbl0.show_in_browser(jsviewer=True))

test_df = tbl0.to_pandas()
test_df["sqrt_flares"] = np.sqrt(test_df["Total Flares"])
test_df["inverse_temp"] = 1/test_df["TEFF"]
test_df["inv_radius"] = 1/test_df["RADIUS"]

condensed_df = test_df[["Median Flare Intensity", "inverse_temp", "inv_radius", "sqrt_flares",
                        "B-V Color Estimate", "Flares Above Six Sigma", "Flares above 1%", "Flares above 4%"]]

weights = [5, 14, 10, 7, 5, 5, 8, 10]

df["weighted avg"] = np.average(condensed_df, weights=weights, axis=1)

tbl = Table.from_pandas(df)
print(tbl.show_in_browser(jsviewer=True))
