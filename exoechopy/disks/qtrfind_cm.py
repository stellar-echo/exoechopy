import os
import sys
from pathlib import Path
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# Check for correct number of command line arguments
if len(sys.argv) != 2:
    print(f'usage {argv[0]} star_id')
    quit()


def find_all_quarters(star):
    """
    Given a star's kplr number, returns a list of all quarters of data for that star. Separates data by cadence.

    :param star: kplr number of a star
    :return: list of paths to all quarters of data for that star
    """

    star = sys.argv[1]

    kpath = Path('/home/echo/hdd6tb/02_kepler_time_series_scripts/')

    sc_qtr_list = []
    lc_qtr_list = []
    full_qtr_list = []

    all_folders = os.listdir(kpath)
    quarter_folders = [x for x in all_folders if '_Q' in x]
    print(quarter_folders)
    print()

    for qd in quarter_folders:
        for f in os.listdir(kpath/qd):
            if star in f:
                if 'llc' in f:
                    lc_qtr_list.append(kpath/qd/f)
                elif 'slc' in f:
                    sc_qtr_list.append(kpath/qd/f)
                full_qtr_list.append(kpath/qd/f)

    print("Long cadence: ")
    print(lc_qtr_list)
    print()
    print("Short cadence: ")
    print(sc_qtr_list)
                
    return lc_qtr_list, sc_qtr_list, full_qtr_list


# Adding plots for short and long cadence - important to separate for stars that have both


def plot_all_quarters(star):
    """
    Given a star's kplr number, saves plots of the long cadence and short cadence data for all quarters of the star
    """

    plt.figure(figsize=(12, 6))
    # SC
    for filename in sc_qtr_list:
        print(filename)
        file = fits.open(filename)
        time = file[1].data["TIME"]
        flux = file[1].data["PDCSAP_FLUX"]
        rawflux = file[1].data["SAP_FLUX"]
        myflux = flux/np.nanmedian(flux)
        plt.plot(time, myflux, c="b")

    plt.xlabel("Time (Days)")
    plt.ylabel("Flux (e-/s)")
    plt.title("{} Short Cadence".format(file[1].header["OBJECT"]))
    plt.show()

    f = "{}_allqtr_short_cadence.png".format(file[1].header["OBJECT"])
    plt.savefig(f)

    plt.figure(figsize=(12,6))
    # LC
    for filename in lc_qtr_list:
        print(filename)
        file = fits.open(filename)
        time = file[1].data["TIME"]
        flux = file[1].data["PDCSAP_FLUX"]
        rawflux = file[1].data["SAP_FLUX"]
        myflux = flux/np.nanmedian(flux)
        plt.plot(time, myflux, c="k")

    plt.xlabel("Time (Days)")
    plt.ylabel("Flux (e-/s)")
    plt.title("{} Long Cadence".format(file[1].header["OBJECT"]))
    plt.show()

    g = "{}_allqtr_long_cadence.png".format(file[1].header["OBJECT"])
    plt.savefig(g)


# Run from command line
if __name__ == "__main__":
    find_all_quarters(sys.argv[1])
    plot_all_quarters(sys.argv[1])
