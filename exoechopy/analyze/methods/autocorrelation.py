
"""
This module provides a variety of methods for performing autocorrelation analysis on lightcurves and flares.

Several autocorrelation estimators are provided, each may be relevant to different analysis techniques.

"""

import numpy as np
from astropy import units as u
from scipy import signal


__all__ = ['autocorrelate_array', 'autocorrelation_overlapping_windows', 'period_folded_autocorrelation']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def autocorrelate_array(data_array: (u.Quantity, np.ndarray),
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
    if isinstance(data_array, u.Quantity):
        data_array = data_array.value
    data_array -= np.mean(data_array)
    corr_vals = np.correlate(data_array, data_array, mode='same')
    # Need to center the data before returning:
    return corr_vals[len(corr_vals)//2+min_lag:len(corr_vals)//2+max_lag+1]/corr_vals[len(corr_vals)//2]


#  =============================================================  #


def autocorrelation_overlapping_windows(data_array: (u.Quantity, np.ndarray),
                                        window_length: int,
                                        max_lag: int,
                                        min_lag: int=0) -> np.ndarray:
    """Generates a 2D array, effectively autocorrelation as a function of time

    Uses a triangular window that overlaps with 50% of the windows on either side
    to create a time-dependent autocorrelation

    Parameters
    ----------
    data_array
        Preprocessed data array for analysis
    window_length
        Odd number, number of datapoints in a window (returns len(data_array)//window_length windows)
        If an odd number is not given, decreases the window size by 1]
        If data_array does not divide evenly into window_length, then ignores the last remainder of data
    max_lag
        Largest lag value to compute
    min_lag
        Smallest lag value to compute

    Returns
    -------
    np.array
        2D array of time bins and lag bins
    """
    window_length -= window_length%2
    num_windows = 2*(len(data_array)//window_length)-1
    window = signal.windows.triang(window_length)
    # Normalize the window:
    window /= np.dot(window, window)

    return_array = np.zeros((num_windows, max_lag-min_lag+1))
    for w_i in range(num_windows):
        init_offset = (w_i//2)*window_length + (w_i%2)*(window_length//2)
        end_offset = (w_i//2+1)*window_length + (w_i%2)*(window_length//2)
        return_array[w_i] = autocorrelate_array(window * data_array[init_offset:end_offset], max_lag=max_lag)

    return return_array

#  =============================================================  #

def period_folded_autocorrelation(data_array: (u.Quantity, np.ndarray),
                                  period_indices: int,
                                  num_windows: int,
                                  max_lag: int,
                                  min_lag: int=0) -> np.ndarray:
    """Folds the data into a period, then computes the time-dependent autocorrelation

    Parameters
    ----------
    data_array
        Preprocessed data array for analysis
    period_indices
        Period to fold at, in indices.  Convert time to index numbers before passing
    num_windows
        Number of windows per period
    max_lag
        Largest lag value to compute
    min_lag

    Returns
    -------
    np.ndarray
        2D array of time bins and lag bins

    """
    window_width = period_indices//num_windows
    num_periods = len(data_array)//period_indices
    init_data = autocorrelation_overlapping_windows(data_array[:period_indices], window_width, max_lag, min_lag)
    for p_i in range(1, num_periods):
        init_data += autocorrelation_overlapping_windows(data_array[p_i*period_indices: (p_i+1)*period_indices],
                                                         window_width, max_lag, min_lag)
    init_data/= num_periods
    return init_data
