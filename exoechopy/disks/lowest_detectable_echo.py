# Contains various functions for the purpose of determining the lowest detectable echo strength in a given light curve
# at a given confidence interval.
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import lightkurve as lk
# from exoechopy.disks.qtrfind_cm import find_all_quarters
import sys
import os
from astropy.io import fits
from scipy.stats import norm
from pathlib import Path


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


def test_detecting_synthetic_echoes(star, echo_strength, sigma, cadence):
    
    """
    Searches the full light curve of a star and determines the feasibility of detecting an echo at a given echo
    strength. Echoes are artificially directly injected into the light curve, and detected by comparing the index-wise mean
    to the index-wise standard error of the mean. The sigma threshold for detection can be changed with the sigma parameter. 
    For now, assumes all flares are unresolved delta flares that last one time bin, and that echoes are perfect copies of this
    flare shape that also last one time bin. This function is meant only as a tool to gauge the feasibility of detecting echoes 
    with other methods, and results are not definitive.

    Parameters
    -----------
    star: star's kplr number
    echo_strength: echo strength, as a percentage of flare strength
    sigma: sigma-based confidence interval used to verify detection
    cadence: "long" for long cadence, "short" for short cadence
    
    Returns
    -----------
    1 if echo detected above sigma confidence threshold, 0 otherwise
    """
    
    sigma = float(sigma)
    
    # Grab all quarters of the star
    long_cadence, short_cadence, full = find_all_quarters(star)
    
    # Stitch together all quarters, normalized by the median:
    if cadence=="long":
        full_lc_flux = []
        for qtrfile in long_cadence:
            fl = fits.open(qtrfile)
            flux = fl[1].data["PDCSAP_FLUX"]
            myflux = flux / np.nanmedian(flux)
            for num in myflux:
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
        lc = lc.remove_nans().flatten()

    else:
        full_sc_flux = []
        for qtrfile in short_cadence:
            fl = fits.open(qtrfile)
            flux = fl[1].data["PDCSAP_FLUX"]
            myflux = flux / np.nanmedian(flux)
            for num in myflux:
                full_sc_flux.append(num)
                
        full_sc_time = []
        for t in short_cadence:
            tt = fits.open(t)
            time = tt[1].data["TIME"]
            for num in time:
                full_sc_time.append(num)
                
        lc = lk.LightCurve(full_sc_time, full_sc_flux)
        lc = lc.remove_nans().flatten()
                
    # Variables
    pre_flare = 2
    post_flare = 14
    sigma_thresh = 6
    echo_index = 10
    
    # Detect flares at 6 sigma -- not the same sigma as provided in func arguments
    flare_threshold = np.nanmedian(lc.flux) + (sigma_thresh*np.nanstd(lc.flux))
    peaks, peak_vals = find_peaks(lc.flux, height=flare_threshold, distance=5)
    
    if len(peaks) > 3:

        # Chop out flares
        advanced_flare_indices = [slice(i-pre_flare, i+post_flare) for i in peaks[0:len(peaks)-1]]
        flares = [lc.flux[advanced_flare_indices[i]] for i in range(len(advanced_flare_indices))]
    
        # Add in normalization by the peak flare value
        normed_flares = [(x-np.nanmedian(lc.flux))/(np.nanmax(flares)-np.nanmedian(lc.flux)) for x in flares]
        normed_flares = np.array(normed_flares)
        
        # Add in echoes via direct injection
        normed_echo_array = np.array(normed_flares)
    
        normed_echo_array[:, echo_index] = normed_echo_array[:, echo_index] + (normed_echo_array[:, 2]*echo_strength)
        normed_echo_mean = np.nanmean(normed_echo_array, axis=0)
        normed_echo_mean = normed_echo_mean - np.nanmedian(normed_echo_mean, axis=0)
    
        # For standard deviation, use standard error of the mean
        normed_echo_std = np.nanstd(normed_echo_array, axis=0) / np.sqrt(len(peaks))
     
        print(normed_echo_mean[echo_index] - sigma*normed_echo_std[echo_index])
    
        # Detection: If mean - sigma*std > 0 at the echo index, count it as "detected" above the confidence interval.
        if normed_echo_mean[echo_index] - sigma*normed_echo_std[echo_index] > 0:
            print("========================================================")
            print()
            print("Potential Echo Detected: {}% echo strength, {} sigma confidence".format(echo_strength*100, sigma))
            print()
            print("========================================================")
            return 1

        else:
            print("========================================================")
            print()
            print("Echo at {}% strength did not survive {} sigma confidence interval".format(echo_strength*100, sigma))
            print()
            print("========================================================")
            return 0
    else:
        print("Not enough flares, skipping star")
        return 0


def find_lowest_echo(star, sigma, cadence):
    
    """
    Searches the full light curve of a star and determines the dimmest detectable echo strength
    using the test_detecting_synthetic_echoes() function. Tests every echo strength from 0% to 30%. As soon as 
    an echo is detected, returns the strength at which it was detected.
    
    Additionally, if an echo is detected, uses the lowest detectable echo strength to give an estimate of the disk mass.
    This estimate uses Eq. (16) from our (currently unfinished) paper, and is based on a number of assumptions, including:
    - A face-on (0 degrees) inclination of the hypothetical disk
    - Isotropic scattering
    - The disk is optically thin
    - The disk is a Kuiper Belt Analog, with an inner radius of 30 AU, outer radius of 50 AU, minimum dust grain radius of 1 micron,
    a monodisperse distribution of particles (X = 1), and the density of dust particles is 2 g/cm^3.

    Parameters
    ----------
    star: star's kplr number
    sigma: sigma-based confidence interval to gauge reliability of echo detection
    cadence: long or short cadence
    
    Returns
    ---------
    Returns the lowest detectable echo (mean at echo index - std. error of mean at echo index > 0) and a hypothetical disk mass estimate
    using the above assumptions.
    """

    # Test echo strengths from 0 - 30
    results = []
    
    while sum(results) == 0:
        
        for i in range(0, 101):
            
            result = test_detecting_synthetic_echoes(star, i/100, sigma, cadence)
            results.append(result)

            if sum(results) > 0:
                
                # print("Echo potentially detected!")
            
                lowest_echo_strength = (np.where(np.array(results) == 1)[0][0])/100
            
                # print("Lowest Detectable Echo Strength at {} Sigma Confidence:".format(sigma), lowest_echo_strength*100, "%")
            
                disk_mass_estimate = lowest_echo_strength*1.125e27
                disk_mass_mEarth = disk_mass_estimate/6e27
            
                # print("Estimate of Disk Mass:", disk_mass_mEarth, "MEarth")
                
                
                return star, lowest_echo_strength, disk_mass_mEarth
                break
            
            elif len(results) == 100 and sum(results) < 1:   
                return star, "null", "null"
                break
            # else:
            #    return star, "null", "null"
                # print("No echoes detected above {} sigma confidence interval.".format(sigma))
  
# Temporarily disable this
# Run from command line
# if __name__ == "__main__":
#    find_lowest_echo(sys.argv[1], sys.argv[2], sys.argv[3])
