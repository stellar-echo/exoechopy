import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import lightkurve as lk
# from exoechopy.disks.qtrfind_cm import find_all_quarters, plot_all_quarters
from scipy.signal import find_peaks
import os
import sys
from astropy.table import Table, Column
import pandas as pd
from exoechopy.analyze.autocorrelation import autocorrelate_array
from exoechopy.disks import star_info
from astropy.timeseries import LombScargle
import re
from pathlib import Path


def find_all_quarters(star):
    """
    Given a star's kplr number, returns a list of all quarters of data for that star. Separates data by cadence.

    :param star: kplr number of a star
    :return: list of paths to all quarters of data for that star
    """

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


top3k = []
with open("top3k.txt", "r") as g:
    for line in g:
        top3k.append(line)

top_kplr = []
for file in top3k:
    s = re.search("kplr(.+?)-", file)
    top_kplr.append(s.group(1))


def combine_fluxes(quarterlist):

    if len(quarterlist) == 0:
        return []

    else:
        combined_flux = []
        for quarter in quarterlist:
            fl = fits.open(quarter)
            flux = fl[1].data["PDCSAP_FLUX"]
            myflux = flux/np.nanmedian(flux)
            for value in quarter:
                combined_flux.append(myflux)
        print(combined_flux)
        return np.asarray(combined_flux)


objects = []
sc_quarters = []
lc_quarters = []
sc_flares = []
lc_flares = []

for ind, star in enumerate(top_kplr):

    longcad, shortcad, full = find_all_quarters(star)

    file0 = fits.open(str(full[0]))
    obj_name = file0[1].header["OBJECT"]
    objects.append(obj_name)

    num_sc_quarters = len(shortcad)
    num_lc_quarters = len(longcad)

    sc_quarters.append(num_sc_quarters)
    lc_quarters.append(num_lc_quarters)

    if num_lc_quarters == 0:
        print("No long cadence data.")
        lc_flares.append(0)

    if num_sc_quarters == 0:
        print("No short cadence data.")
        sc_flares.append(0)

    else:
        longflux = combine_fluxes(longcad)
        shortflux = combine_fluxes(shortcad)

        lc_median = np.nanmedian(longflux)
        lc_std = np.nanstd(longflux)
        flare_threshold_lc = lc_median + (3 * lc_std)
        peaks_lc, peak_val_lc = find_peaks(longflux, height=flare_threshold_lc, distance=4)
        lc_flares.append(len(peaks_lc))

        sc_median = np.nanmedian(shortflux)
        sc_std = np.nanstd(shortflux)
        flare_threshold_sc = sc_median + (3 * sc_std)
        peaks_sc, peak_val_sc = find_peaks(shortflux, height=flare_threshold_sc, distance=30)
        sc_flares.append(len(peaks_sc))

    print("Finished star", ind, "of 3000")

data = [objects, sc_quarters, lc_quarters, sc_flares, lc_flares]
names = ["Object", "Short Cadence Quarters", "Long Cadence Quarters", "Short cadence Flares", "Long Cadence Flares"]

df = pd.DataFrame([data], columns=names)
t = Table.from_pandas(df)
t.write("full_table.html", format="ascii.html", overwrite=True)
