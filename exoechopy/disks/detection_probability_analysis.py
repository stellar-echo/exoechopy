# Contains various functions for the purpose of determining the lowest detectable echo strength in a given light curve
# at a given confidence interval.
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import lightkurve as lk
from exoechopy.disks.qtrfind_cm import find_all_quarters
import sys
import os
from astropy.io import fits
from scipy.stats import norm


def test_detecting_synthetic_echoes(star, echo_strength, sigma):
    """
    Searches the full light curve of a star and determines the feasibility of detecting an echo at the given echo
    strength, assuming an echo is present at all in the data. Echoes are artificially injected using a convolution
    kernel, and detected by comparing the index-wise mean to the index-wise standard deviation. The sigma threshold for
    detection can be changed with the sigma parameter. For now, assumes all flares are unresolved delta flares that last
    one time bin, and that echoes are perfect copies of this flare shape that also last one time bin.
    This function is meant only as a tool to gauge the feasibility of detecting echoes with other methods,
    and results are not definitive.

    :param star: star's kplr number
    :param echo_strength: echo strength, as a percentage of flare strength
    :param sigma: sigma-based confidence interval used to verify detection
    :return: probability of detecting the echo, and probability of false positive detection
    """
    
    sigma = float(sigma)
    
    # Grab all quarters of the star
    long_cadence, short_cadence, full = find_all_quarters(star)

    # We only care about the long cadence flux for now
    # Stitch together all quarters, normalized by the median:
    full_lc_flux = []
    for qtrfile in long_cadence:
        fl = fits.open(qtrfile)
        flux = fl[1].data["PDCSAP_FLUX"]
        myflux = flux / np.nanmedian(flux)
        for num in flux:
            full_lc_flux.append(num)

    # Do the same for the time:
    full_lc_time = []
    for t in long_cadence:
        tt = fits.open(t)
        time = tt[1].data["TIME"]
        for num in time:
            full_lc_time.append(num)

    # Remove nans with Lightkurve
    lc = lk.LightCurve(full_lc_time, full_lc_flux)
    lc = lc.remove_nans()

    # Detect flares at 3 sigma
    flare_threshold = np.nanmedian(lc.flux) + (6*np.nanstd(lc.flux))
    peaks, peak_vals = find_peaks(lc.flux, height=flare_threshold, distance=5)

    # Chop out flares
    advanced_flare_indices = [slice(i-2, i+14) for i in peaks[0:len(peaks)-1]]
    flares = [lc.flux[advanced_flare_indices[i]] for i in range(len(advanced_flare_indices))]

    # Compute the index-wise mean and std dev
    no_echo_array = np.array(flares)
    no_echo_mean = np.nanmean(no_echo_array, axis=0)
    no_echo_std = np.nanstd(no_echo_array, axis=0)

    # ------- Repeat process with echoes added via convolution/direct injection ------- #

    # Testing direct injection
    echo_array = np.array(flares) - 1
    echo_array[:, 7] = (echo_array[:, 2]*echo_strength) + echo_array[:, 7] 
    echo_mean = np.nanmean(echo_array, axis=0)
    echo_std = np.nanstd(echo_array, axis=0)
    
    """
    # Generate simple convolution kernel
    to_convolve = list(np.zeros(20))
    to_convolve[5] = 1
    to_convolve[10] = echo_strength
    
    # Normalize the kernel
    to_convolve /= np.sum(to_convolve)

    # Generate new LC
    convolved_lc = np.convolve(lc.flux - 1, to_convolve)

    # Find new flares
    peaks, peak_val = find_peaks(convolved_lc, height=np.nanmedian(convolved_lc) + (3 * np.nanstd(convolved_lc)),
                                 distance=4)
    conv_flare_indices = peaks

    # Chop flares w/echoes
    adv_conv_fl_ind = [list(range(i - 2, i + 16)) for i in conv_flare_indices]
    conv_flares = [convolved_lc[adv_conv_fl_ind[i]] for i in range(len(adv_conv_fl_ind))]

    # Compute index-wise mean
    echo_array = np.array(conv_flares)
    echo_mean = np.nanmean(echo_array, axis=0)
    echo_std = np.nanstd(echo_array, axis=0)
    """   
      
    # Detection: If mean - sigma*std > 0 at the echo index, count it as "detected" above the confidence interval.
    if echo_mean[7] - sigma*echo_std[7] > 0:
        print("Potential Echo Detected: {}% echo strength, {} sigma confidence".format(echo_strength*100, sigma))
        return 1

    else:
        print("Echo at {}% strength did not survive {} sigma confidence interval".format(echo_strength*100, sigma))
        return 0


def find_lowest_echo(star, sigma):
    """
    Searches the full light curve of a star and determines the dimmest detectable echo strength, assuming an echo is
    present at all in the data. Echoes are artificially injected using a convolution kernel, and detected by comparing
    the index-wise mean to the standard deviation of the flare tail. For now, assumes all flares are unresolved delta
    flares that last one time bin, and that echoes are perfect copies of this flare shape that also last one time bin.
    This function is meant only as a tool to gauge the feasibility of detecting echoes with other methods, and results
    are not definitive.

    :param star: star's kplr number
    :param sigma: sigma-based confidence interval to gauge reliability of echo detection
    :return: the lowest detectable echo (probability of detection > 0 as a percentage of flare strength, and its
    probability of detection
    """

    # Test echo strengths from 0 - 30
    results = []

    for i in range(0, 51):
        result = test_detecting_synthetic_echoes(star, i/100, sigma)
        results.append(result)

    if sum(results) > 0:
        print("Echoes potentially detected!")
        print("Lowest Detectable Echo Strength at {} Sigma Confidence:".format(sigma), np.where(np.array(results) == 1)[0][0], "%")
      
    else:
        print("No echoes detected above {} sigma confidence interval.".format(sigma))
  

# Run from command line
if __name__ == "__main__":
    find_lowest_echo(sys.argv[1], sys.argv[2])
