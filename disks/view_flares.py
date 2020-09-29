import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import lightkurve as lk
from scipy.signal import find_peaks


def view_flares(path):
    """
    Plots a de-trended Kepler light curve, with peaks marked at different thresholds according to the
    find_peaks() function.

    :param path: Path to the desired file. Must be a fits file

    """

    hdu = 0
    header = fits.getheader(path, hdu)

    total_flares = []
    flares_six_sigma = []

    # Get number of flares and flare times
    lc_raw = fits.open(str(path))
    raw_flux = lc_raw[1].data["PDCSAP_FLUX"]
    time = lc_raw[1].data["TIME"]

    lc = lk.LightCurve(time=time, flux=raw_flux)
    lc = lc.remove_nans().flatten()

    # Different cadences require different flare detection windows
    cadence = header.get("OBSMODE")
    if cadence == "short cadence":
        x = lc.flux
        median = np.median(x)
        sigma = np.std(x)

        three_sig = median + (3*sigma)
        peaks, peak_val = find_peaks(x, height=three_sig, distance=30)
        total_flares.append(len(peaks))

        six_sig = median + (6*sigma)
        peaks_six, peak_val_six = find_peaks(x, height=six_sig, distance=30)
        flares_six_sigma.append(len(peaks_six))

    else:
        y = lc.flux
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
    plt.figure(figsize=(12, 6))
    plt.plot(lc.flux, label="Detrended Flux")
    plt.plot(peaks, flare_heights, "x", label="Detected Flares at 3 Sigma")
    plt.plot(peaks_six, flare_heights_six, "o", c="r", label="Detected Flares at 6 Sigma")
    plt.axhline(3*np.std(lc.flux) + np.median(lc.flux), c="k", alpha=0.5, linestyle="dotted", label="3 sigma threshold")
    plt.axhline(6*np.std(lc.flux) + np.median(lc.flux), c="k", alpha=0.5, linestyle="dashed", label="6 sigma threshold")
    plt.xlabel("Index")
    plt.ylabel("Detrended Flux")
    plt.title("{} - {} Total Flares Detected, {} above 6sig".format(header.get("OBJECT"), total_flares, flares_six_sigma))
    plt.legend()
    plt.savefig("{}.png".format(header.get("OBJECT")))


if __name__ == "__main__":
    import sys
    view_flares(str(sys.argv[1]))
