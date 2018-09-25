
"""
This module provides functions that are useful in computing values with vectors.
Will be broken into separate modules once I know all of the math that is needed.
"""

import decimal
import numpy as np
from scipy import stats
from astropy import units as u
from astropy import constants
from scipy.signal import savgol_filter

__all__ = ['angle_between_vectors', 'vect_from_spherical_coords', 'compute_lag',
           'SphericalLatitudeGen', 'stochastic_flare_process',
           'window_range',
           'take_noisy_derivative', 'take_noisy_2nd_derivative',
           'round_dec', 'row_col_grid']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Vector math

def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Smallest angle between two unnormalized vectors

    Parameters
    ----------
    v1
        Vector 1
    v2
        Vector 2

    Returns
    -------
    float
        Angle in radians as a float
    """
    return 2*np.arctan2(np.linalg.norm(np.linalg.norm(v2)*v1-np.linalg.norm(v1)*v2),
                        np.linalg.norm(np.linalg.norm(v2)*v1+np.linalg.norm(v1)*v2))


def vect_from_spherical_coords(longitude, latitude) -> np.ndarray:
    vect = np.array((np.sin(latitude) * np.cos(longitude),
                     np.sin(latitude) * np.sin(longitude),
                     np.cos(latitude) * np.ones(np.shape(longitude))))
    return np.transpose(vect)


def compute_lag(start_vect: u.Quantity,
                echo_vect: u.Quantity,
                detect_vect: np.ndarray,
                return_v2_norm: bool=False) -> u.Quantity:
    """Calculate the lag from a flare echo

    Parameters
    ----------
    start_vect
        Starting point of the flare
    echo_vect
        Location of the echo-ing object
    detect_vect
        *Normalized* vector in the direction of the detection (Earth)
    return_v2_norm
        Whether or not to return the norm of echo_vect - start_vect
        Since this value is often used in other calculations, can reduce the number of times it's calculated.

    Returns
    -------
    u.Quantity
        Echo lag quantity

    """
    # Vector from the starting point to the echo source:
    v2 = echo_vect-start_vect
    v2_norm = u.Quantity(np.linalg.norm(v2), v2.unit)
    if return_v2_norm:
        return (v2_norm-u.Quantity(np.dot(v2, detect_vect), v2.unit))/constants.c, v2_norm
    else:
        return (v2_norm-u.Quantity(np.dot(v2, detect_vect), v2.unit))/constants.c


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Stats math


class SphericalLatitudeGen(stats.rv_continuous):
    def __init__(self, a=0, b=np.pi, min_val=0, max_val=np.pi, **kwargs):
        super().__init__(a=a, b=b, **kwargs)
        self._minval = min_val
        self._maxval = max_val

    def _pdf(self, x):
        return np.sin(x)/(np.cos(self.a) - np.cos(self.b))


def stochastic_flare_process(stop_value,
                             distribution: stats.rv_continuous,
                             start_value=0,
                             max_iter=1000, *dist_args):
    # TODO implement a faster version, such as generating many values at once and identifying where it exceeds stop_val
    ct = 0
    all_values = []
    total_displacement = start_value
    while ct < max_iter:
        next_val = distribution.rvs(*dist_args)
        test_val = total_displacement + next_val
        if test_val < stop_value:
            all_values.append(test_val)
            total_displacement += next_val
        else:
            break
    return all_values


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Plot math

def window_range(ind: int, max_ind: int, width: int) -> [int, int]:
    """Constrain a sub-window from exceeding a range while maintaing a fixed width

    Parameters
    ----------
    ind
        "central" index
    max_ind
        Largest index value allowed-- typically len(whatever)-1
    width
        fixed window width

    Returns
    -------
    [int, int]
        Lowest index, highest index

    """
    w1 = width//2 + width%2
    w2 = width//2
    min1 = ind - w1
    max2 = ind + w2
    if min1 < 0:
        return [0, width-1]
    elif max2 >= max_ind:
        return [max_ind - width, max_ind-1]
    else:
        return [ind - w1, ind + w2 - 1]


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Signal processing math

def take_noisy_derivative(data_array: np.ndarray,
                          window_size: int=19,
                          sample_cadence: float=0.2) -> np.ndarray:
    """Applies Savitzky-Golay filter to estimate the first derivative, which is robust to noise

    Essentially an easy-to-remember wrapper for the scipy savgol_filter

    Parameters
    ----------
    data_array
        Data for analysis
    window_size
        Size of window for polynomial fit
    sample_cadence
        Scale factor for derivative

    Returns
    -------
    np.ndarray
        Returns the Savitzky-Golay estimate of the first derivative
    """
    return savgol_filter(data_array, window_length=window_size, polyorder=2, deriv=1, delta=sample_cadence,
                         mode='mirror')


def take_noisy_2nd_derivative(data_array: np.ndarray,
                              window_size: int=19,
                              sample_cadence: float=0.2) -> np.ndarray:
    """Applies Savitzky-Golay filter to estimate the second derivative, which is robust to noise

    Essentially an easy-to-remember wrapper for the scipy savgol_filter

    Parameters
    ----------
    data_array
        Data for analysis
    window_size
        Size of window for polynomial fit
    sample_cadence
        Scale factor for derivative

    Returns
    -------
    np.ndarray
        Returns the Savitzky-Golay estimate of the second derivative
    """
    return savgol_filter(data_array, window_length=window_size, polyorder=2, deriv=2, delta=sample_cadence,
                         mode='mirror')


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Indexing and integer math

def round_dec(val: float) -> int:
    """Round a number to an integer in the way that rounding is expected to behave

    In python 3, round(2.5) = 2, round(3.5) = 4.
    This can really screw up indexing that relies on decimal rounding.
    This function rounds the way I expect a number that is entered like a decimal to be rounded.

    Parameters
    ----------
    val
        Value to be rounded

    Returns
    -------
    int
        The round-half-up version of val

    """
    return int(decimal.Decimal(val).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


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
