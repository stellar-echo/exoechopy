import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import lightkurve as lk
from scipy.signal import find_peaks
from exoechopy.visualize import plot_flare_array
from astropy.timeseries import LombScargle
import os


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


def view_flares(path, hist=False, plot_ac=True, plot_ind_flares=True, lombscarg=True):
    """
    Plots a de-trended Kepler light curve, with peaks marked at different thresholds according to the
    find_peaks() function.

    :param path: Path to the desired file. Must be a fits file
    :param hist: if True, plots and saves a histogram of flare magnitude
    :param plot_ac: if True, plots full lightcurve autocorrelation
    :param plot_ind_flares: if True, plots an array of individual flares (thanks, EchoPy!)
    :param lombscarg: if True, generates a Lomb-Scargle periodogram of the flare times
    """

    hdu = 0
    header = fits.getheader(path, hdu)

    # Create a new directory to store images of the star
    if not os.path.exists("detailed_star_info/{}/Q{}".format(header.get("OBJECT"), header.get("QUARTER"))):
        os.makedirs("detailed_star_info/{}/Q{}".format(header.get("OBJECT"), header.get("QUARTER")))

    total_flares = []
    flares_six_sigma = []

    # Get number of flares and flare times
    lc_raw = fits.open(str(path))
    raw_flux = lc_raw[1].data["PDCSAP_FLUX"]

    time = lc_raw[1].data["TIME"]

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

        print(peaks)
        print(type(peaks))

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
    plt.suptitle("{} Quarter {}- {} Total Flares Detected, {} above 6sig".format(header.get("OBJECT"),
                                                                                 header.get("QUARTER"), total_flares,
                                                                                 flares_six_sigma))
    ax[0].plot(lc.flux, label="Raw Flux", drawstyle="steps-post")
    ax[1].plot(lc_detr.flux, label="Detrended Flux", drawstyle="steps-post")
    ax[1].plot(peaks, flare_heights, "x", label="Detected Flares at 3 Sigma")
    ax[1].plot(peaks_six, flare_heights_six, "o", c="r", label="Detected Flares at 6 Sigma")
    ax[1].axhline(3*np.std(lc.flux) + np.median(lc.flux), c="k", alpha=0.5, linestyle="dotted", label="3 sigma threshold")
    ax[1].axhline(6*np.std(lc.flux) + np.median(lc.flux), c="k", alpha=0.5, linestyle="dashed", label="6 sigma threshold")
    ax[1].axhline(1.01, c="k", alpha=0.5, linestyle="dashdot")
    ax[1].axhline(1.04, c="k", alpha=0.4)
    # plt.xlabel("Index")
    # plt.ylabel("Detrended Flux")
    # plt.title("{} Quarter {} - {} Total Flares Detected, {} above 6sig".format(header.get("OBJECT"),
    #                                                                           header.get("QUARTER"),
    #                                                                           total_flares, flares_six_sigma))
    # plt.legend()
    plt.savefig("detailed_star_info/{}/Q{}/lc_flares_marked.png".format(header.get("OBJECT"), header.get("QUARTER")))

    if hist:
        plt.figure(figsize=(12, 6))
        plt.hist(flare_heights, bins=50, density=True, color="g")
        plt.title("Flare Magnitude Histogram for {} Quarter {}".format(header.get("OBJECT"), header.get("QUARTER")))
        plt.xlabel("Detrended Flare Magnitude")
        plt.ylabel("Count")
        plt.savefig("detailed_star_info/{}/Q{}/flare magnitude hist.png".format(header.get("OBJECT"),
                                                                               header.get("QUARTER")))

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

    if plot_ind_flares:
        if header.get("OBSMODE") == "short cadence":
            plot_flare_array(lc.flux, peaks, 10, 30, display_index=True,
                             savefile="detailed_star_info/{}/Q{}/flare_arr.png".format(header.get("OBJECT"),
                                                                                      header.get("QUARTER")))
        else:
            plot_flare_array(lc.flux, peaks, 5, 10, display_index=True,
                             savefile="detailed_star_info/{}/Q{}/flare_arr.png".format(header.get("OBJECT"),
                                                                                   header.get("QUARTER")))

    if lombscarg:
        freq, power = LombScargle(lc.time, lc.flux).autopower()

        plt.figure(figsize=(12, 6))
        plt.plot(freq, power, c="k", drawstyle="steps-post")
        plt.xlabel("Frequency")
        plt.ylabel("Power")
        plt.title("Lomb-Scargle for {} Q{}".format(header.get("OBJECT"), header.get("QUARTER")))
        plt.savefig("detailed_star_info/{}/Q{}/LombScarg.png".format(header.get("OBJECT"), header.get("QUARTER")))


if __name__ == "__main__":
    import sys
    star_info(str(sys.argv[1]))
