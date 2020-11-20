
"""
This module provides a variety of methods for performing autocorrelation analysis on lightcurves and flares.

Several autocorrelation estimators are provided, each may be relevant to different analysis techniques.

"""

import numpy as np
from astropy import units as u
from scipy import signal
from exoechopy.utils.constants import *


__all__ = ['autocorrelate_array', 'autocorrelation_overlapping_windows', 'period_folded_autocorrelation',
           'calculate_bromley_correlator']

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
    data_array = data_array - np.nanmean(data_array)
    corr_vals = np.correlate(data_array, data_array, mode='same')
    # Need to center the data before returning:
    return corr_vals[len(corr_vals)//2+min_lag:len(corr_vals)//2+max_lag+1]/corr_vals[len(corr_vals)//2]

#  =============================================================  #

#
# def autocorrelate_array_type_2(data_array: (u.Quantity, np.ndarray),
#                                max_lag: int,
#                                min_lag: int=0) -> np.ndarray:
#     """Computes the unnormalized autocorrelation at multiple lags for a dataset
#
#     Parameters
#     ----------
#     data_array
#         Preprocessed data array for analysis
#     max_lag
#         Largest lag value to compute
#     min_lag
#         Smallest lag value to compute
#
#     Returns
#     -------
#     np.ndarray
#         Array of autocorrelation values for lags in [min_lag, max_lag]
#
#     """
#     if isinstance(data_array, u.Quantity):
#         data_array = data_array.value
#     data_array = data_array - np.mean(data_array)
#     autocorr = np.array([autocorr_lw(data_array, l_i) for l_i in range(min_lag, max_lag)])
#     return autocorr/autocorr_lw(data_array)
#
#
# def autocorr_lw(data_array, lag=0):
#     """Compute the autocorrelation at a single lag
#
#     Parameters
#     ----------
#     data_array
#         Preprocessed data array for analysis
#     lag
#         Time to compute the value at
#
#     Returns
#     -------
#     float
#         The autocorrelation at that lag
#     """
#     if lag == 0:
#         return np.dot(data_array, data_array) / len(data_array[lag:])
#     else:
#         return np.dot(data_array[lag:], data_array[:-lag]) / len(data_array[lag:])


#  =============================================================  #


def calculate_bromley_correlator(data_array: (u.Quantity, np.ndarray),
                                 peak_width: int,
                                 max_lag: int,
                                 min_lag: int=0,
                                 prefilter_function: FunctionType=None,
                                 **prefilter_function_args) -> np.ndarray:
    """Computes the correlator for a flare described in 'A framework for planet detection with faint lightcurve echoes'

    https://arxiv.org/abs/1808.07029

    Parameters
    ----------
    data_array
        Data array around a flare
    peak_width
        How many indices around the peak to pick for the correlator
    max_lag
        Largest lag value to compute
    min_lag
        Smallest lag value to compute
    prefilter_function
        Filter function to subtract prior to computing correlator
        Must be of the form f(x, args)
    prefilter_function_args
        Arguments to pass on to the prefilter_function

    Returns
    -------
    np.ndarray
        Returns the correlator as a function of lag for the data_array
    """
    if isinstance(data_array, u.Quantity):
        data_array = data_array.value
    data_array = data_array - np.mean(data_array)
    corr_vals = np.correlate(data_array, data_array, mode='same')
    mid_ind = len(corr_vals)//2
    corr_vals /= corr_vals[mid_ind]
    central_peak = corr_vals[mid_ind-peak_width//2:mid_ind+peak_width//2+1].copy()
    central_peak -= np.mean(central_peak)
    if prefilter_function is not None:
        corr_vals -= prefilter_function(corr_vals, **prefilter_function_args)
    correlators = np.convolve(corr_vals, central_peak, mode='same')
    return correlators[mid_ind+min_lag: mid_ind+max_lag+1]


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
