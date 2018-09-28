
"""
This module provides different algorithms for identifying and extracting flares from a lightcurve segment.
"""

import numpy as np
from scipy import signal

__all__ = ['find_peaks_stddev_thresh']


def find_peaks_stddev_thresh(lightcurve_data: np.ndarray,
                             std_dev_threshold: float=None,
                             smoothing_radius: float=None,
                             min_index_gap: int=None,
                             extra_search_pad: int=None,
                             num_smoothing_rad: int=None) -> np.ndarray:
    """Relatively low-overhead peak finder for an array based on deviation above a threshold.

    Has some filtering functionality to prevent detection of multiple peaks within the same flare event.
    Works best if slowly varying background has already been subtracted.
    
    Parameters
    ----------
    lightcurve_data
        Input data with low or subtracted low-frequency variability
    std_dev_threshold
        Threshold multiples of sigma above background to consider for peak extraction
        Defaults to sigma=1
    smoothing_radius
        Optional Gaussian smoothing filter to run prior to peak finding, helps eliminate salt & pepper noise
    min_index_gap
        Require each peak be at least this many indices from another peak
    extra_search_pad
        Add an optional extra search on either side of a peak, useful if Gaussian is aggressive
    num_smoothing_rad
        How many points to include in filter window, based on smoothing_radius

    Returns
    -------
    numpy.ndarray(dtype=int)
        Array containing the indices at which peaks were detected in signal_data

    """

    # Using median instead of mean, assumes activity is primarily excess photon events (like flares)
    # Mean would be biased by strong peak events, median less-so
    filtered_lightcurve = lightcurve_data-np.median(lightcurve_data)

    if std_dev_threshold is None:
        std_dev_threshold = 1.

    if smoothing_radius is not None:
        if num_smoothing_rad is None:
            num_smoothing_rad = 5  # Number of radii to include in Gaussian
        _num_window_points = max(smoothing_radius * num_smoothing_rad, 5)  # Ensure a minimum of 5 points in window
        _window = signal.gaussian(_num_window_points, smoothing_radius)
        _window /= np.sum(_window)
        filtered_lightcurve = signal.fftconvolve(filtered_lightcurve, _window, 'same')

    if extra_search_pad is None:
        extra_search_pad = 0

    filtered_lightcurve /= np.std(filtered_lightcurve)

    excess_region_mask = np.where(filtered_lightcurve > std_dev_threshold)[0]  # Find all points that exceed threshold
    delta_mask = excess_region_mask[1:] - excess_region_mask[:-1]  # Computes gap in indices between peaky events
    distinct_regions = np.where(delta_mask > min_index_gap)[0]+1  # array is 1 shorter, need +1 to align indices

    # Initialize the return array, now that we know how many peaks we found:
    return_array = np.zeros(len(distinct_regions)+1, dtype=int)

    # Uses the actual max from the original unfiltered lightcurve, so the maxima can be influenced by noise.
    # Goes through each region that exceeded the threshold and picks the highest point, probably a better way to do this
    min_search_ind = excess_region_mask[0]
    for region_i in range(len(distinct_regions)):
        max_search_ind = excess_region_mask[distinct_regions[region_i]-1]
        return_array[region_i] = np.argmax(
            lightcurve_data[min_search_ind-extra_search_pad:max_search_ind+extra_search_pad]) + \
                                 min_search_ind - extra_search_pad
        min_search_ind = excess_region_mask[distinct_regions[region_i]]
    last_region = excess_region_mask[-1]
    return_array[region_i+1] = np.argmax(
        lightcurve_data[min_search_ind-extra_search_pad:last_region+extra_search_pad]) + \
                               min_search_ind - extra_search_pad
    return return_array


