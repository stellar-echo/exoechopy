import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import lightkurve as lk
from scipy.signal import find_peaks
# from exoechopy.visualize import plot_flare_array
from astropy.timeseries import LombScargle
import os

# Workaround until I sort out importing from echopy on the server
def row_col_grid(num_pts: int) -> (int, int):
    """Generate a number of rows and columns for displaying a grid of num_pts objects
    Parameters
    ----------
    num_pts
        Number of values to break into rows and columns
    Returns
    -------
    tuple
        rows, columns
    """
    return int(np.sqrt(num_pts)), int(np.ceil(num_pts/int(np.sqrt(num_pts))))

# Workaround pt 2
def plot_flare_array(lightcurve: np.ndarray,
                     flare_indices: np.ndarray,
                     back_pad: int,
                     forward_pad: int,
                     savefile: str=None,
                     display_index: bool=False,
                     display_flare_loc: bool=True,
                     title: str=None):
    """Plot an array of flares extracted from a lightcurve
    Parameters
    ----------
    lightcurve
        Raw data to extract flare curve profiles from
    flare_indices
        Indices to plot around
    back_pad
        Indices before each flare index to include in plot
    forward_pad
        Indices after each flare index to include in plot
    savefile
        If None, runs plt.show()
        If filepath str, saves to savefile location (must include file extension)
    display_index
        If True, display the flare index number from the lightcurve on each flare
    display_flare_loc
        If True, plots a red dot on top of the perceived flare peak
    title
        If provided, overrides the default title
    """

    num_flares = len(flare_indices)

    num_row, num_col = row_col_grid(num_flares)
    fig, all_axes = plt.subplots(num_row, num_col, figsize=(10, 6))
    for f_i, flare_index in enumerate(flare_indices):
        c_i = f_i // num_row
        r_i = f_i - num_row * c_i
        all_axes[r_i, c_i].plot(
            lightcurve[flare_index - back_pad:flare_index + forward_pad],
            color='k', lw=1, drawstyle='steps-post')
        if display_index:
            all_axes[r_i, c_i].text(.95, .95, "i="+str(flare_index),
                                    transform=all_axes[r_i, c_i].transAxes,
                                    verticalalignment='top', horizontalalignment='right',
                                    color='b')
        if display_flare_loc:
            all_axes[r_i, c_i].scatter(back_pad, lightcurve[flare_index], color='r')
    for r_i in range(num_row):
        for c_i in range(num_col):
            all_axes[r_i, c_i].set_xticklabels([])
            all_axes[r_i, c_i].set_yticklabels([])
    fig.subplots_adjust(hspace=0, wspace=0)
    if title is None:
        plt.suptitle(str(num_flares)+" normalized flare examples from lightcurve")
    else:
        plt.suptitle(title)
        plt.show()

    if savefile is None:
        plt.show()
    else:
        plt.savefig(savefile)
        plt.close()

# Workaround pt 3
def autocorrelate_array(data_array,
                        max_lag: int,
                        min_lag: int=0) -> np.ndarray:
    """Computes the unnormalized autocorrelation at multiple lags for a dataset

    Parameters
    ----------
    data_array
        Preprocessed data array for analysis
    max_lag
        Largest lag value to compute
    min_lag
        Smallest lag value to compute

    Returns
    -------
    np.ndarray
        Array of autocorrelation values for lags in [min_lag, max_lag]

    """
    data_array = data_array - np.mean(data_array)
    corr_vals = np.correlate(data_array, data_array, mode='same')
    # Need to center the data before returning:
    return corr_vals[len(corr_vals)//2+min_lag:len(corr_vals)//2+max_lag+1]/corr_vals[len(corr_vals)//2]

# ------------------------------------------------------------------------------------------------------- # 

# My code
def star_info(path, hist=True, plot_ac=True, plot_ind_flares=True, lombscarg=True):
    """
    Plots a de-trended Kepler light curve, with peaks marked at different thresholds according to the
    find_peaks() function. Optional parameters provide more detailed star information.

    :param path: Path to the desired file. Must be a fits file
    :param hist: if True, plots and saves a histogram of flare magnitude
    :param plot_ac: if True, plots full lightcurve autocorrelation
    :param plot_ind_flares: if True, plots an array of individual flares (thanks, EchoPy!)
    :param lombscarg: if True, generates a Lomb-Scargle periodogram of the flare times
    """

    hdu = 0
    header = fits.getheader(path, hdu)
    obj = header.get("OBJECT")
    q = header.get("QUARTER")

    # Create a new directory to store images of the star
    if not os.path.exists("detailed_star_info/{}/Q{}".format(obj, q)):
        os.makedirs("detailed_star_info/{}/Q{}".format(obj, q))

    # Store flares based on sigma thresholds
    total_flares = []
    flares_six_sigma = []

    # Get number of flares and flare times
    lc_raw = fits.open(str(path))
    raw_flux = lc_raw[1].data["PDCSAP_FLUX"]

    time = lc_raw[1].data["TIME"]
    
    # Lightkurve makes things easier
    lc = lk.LightCurve(time=time, flux=raw_flux)
    lc_detr = lc.remove_nans().flatten()

    # Different cadences require different flare detection windows
    cadence = header.get("OBSMODE")
    if cadence == "short cadence":
        x = lc_detr.flux
        median = np.median(x)
        sigma = np.std(x)

        three_sig = median + (3*sigma)
        peaks, peak_val = find_peaks(x, height=three_sig, distance=30)
        total_flares.append(len(peaks))

        six_sig = median + (6*sigma)
        peaks_six, peak_val_six = find_peaks(x, height=six_sig, distance=30)
        flares_six_sigma.append(len(peaks_six))

    else:
        y = lc_detr.flux
        median = np.median(y)
        sigma = np.std(y)

        three_sig = median + (3*sigma)
        peaks, peak_val = find_peaks(y, height=three_sig, distance=4)
        total_flares.append(len(peaks))

        six_sig = median + (6*sigma)
        peaks_six, peak_val_six = find_peaks(y, height=six_sig, distance=4)
        flares_six_sigma.append(len(peaks_six))

    lc_raw.close()

    # Get flare heights
    flare_heights = []
    for val in peak_val.values():
        for num in val:
            flare_heights.append(num)

    flare_heights_six = []
    for val in peak_val_six.values():
        for num in val:
            flare_heights_six.append(num)

    # This probably isn't necessary
    peaks_six_overlap = list(set(peaks) & set(peaks_six))

    # Visualize
    fig, ax = plt.subplots(2, figsize=(12, 6))
    plt.suptitle("{} Quarter {}- {} Total Flares Detected, {} above 6sig".format(obj, q, total_flares, flares_six_sigma))
    ax[0].plot(lc.flux, label="Raw Flux", drawstyle="steps-post")
    ax[1].plot(lc_detr.flux, label="Detrended Flux", drawstyle="steps-post")
    ax[1].plot(peaks, flare_heights, "x", label="Detected Flares at 3 Sigma")
    ax[1].plot(peaks_six, flare_heights_six, "o", c="r", label="Detected Flares at 6 Sigma")
    ax[1].axhline(3*np.std(lc.flux) + np.median(lc.flux), c="k", alpha=0.5, linestyle="dotted", label="3 sigma threshold")
    ax[1].axhline(6*np.std(lc.flux) + np.median(lc.flux), c="k", alpha=0.5, linestyle="dashed", label="6 sigma threshold")
    ax[1].axhline(1.01, c="k", alpha=0.5, linestyle="dashdot")
    ax[1].axhline(1.04, c="k", alpha=0.4)
    plt.savefig("detailed_star_info/{}/Q{}/lc_flares_marked.png".format(obj, q))

    # Histogram
    if hist:
        plt.figure(figsize=(12, 6))
        plt.hist(flare_heights, bins=50, color="k")
        plt.title("Flare Magnitude Histogram for {} Quarter {}".format(obj, q))
        plt.xlabel("Detrended Flare Magnitude")
        plt.ylabel("Count")
        plt.savefig("detailed_star_info/{}/Q{}/flare magnitude hist.png".format(obj, q))
    
    # Autocorrelation plot
    if plot_ac:

        lc = lk.LightCurve(time=time, flux=raw_flux)
        lc = lc.remove_nans()
        lc_detr = lc.flatten()

        plt.figure(figsize=(12, 6))
        plt.plot(autocorrelate_array(lc.flux, max_lag=100), drawstyle="steps-post", c="k", label="Raw Flux", lw=1)
        plt.plot(autocorrelate_array(lc_detr.flux, max_lag=100), drawstyle="steps-post", c="b", label="Detrended", lw=1)
        plt.legend()
        plt.title("{} Autocorrelation".format(header.get("OBJECT")))
        plt.xlabel("Lag Index")
        plt.ylabel("Correlation")
        plt.savefig("detailed_star_info/{}/Q{}/autocorr.png".format(header.get("OBJECT"), header.get("QUARTER")))

    # Flare Array
    if plot_ind_flares:
        if header.get("OBSMODE") == "short cadence":
            plot_flare_array(lc.flux, peaks, 10, 30, display_index=False,
                             savefile="detailed_star_info/{}/Q{}/flare_arr.png".format(obj, q))
        else:
            plot_flare_array(lc.flux, peaks, 5, 10, display_index=True,
                             savefile="detailed_star_info/{}/Q{}/flare_arr.png".format(obj, q))
    
    # Naive periodogram
    if lombscarg:
        freq, power = LombScargle(peaks, flare_heights).autopower()

        plt.figure(figsize=(12, 6))
        plt.plot(freq, power, c="k", drawstyle="steps-post")
        plt.xlabel("Frequency")
        plt.ylabel("Power")
        plt.title("Lomb-Scargle of Flare Times for {} Q{}".format(obj, q))
        plt.savefig("detailed_star_info/{}/Q{}/LombScarg.png".format(obj, q))

# Run from command line
if __name__ == "__main__":
    import sys
    star_info(str(sys.argv[1]))
