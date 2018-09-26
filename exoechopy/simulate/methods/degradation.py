
"""
This module provides routines for degrading the simulated data to provide more realistic noise conditions.
"""

import warnings
from astropy.utils.exceptions import AstropyUserWarning

import numpy as np
from astropy import units as u

__all__ = ['add_poisson_noise']

# TODO: Add detector noise
# TODO: Add readout or other platform noise
# TODO: Chop up a light curve to imitate constrained observation times
# TODO: Adjust data to match observations made from orbital platform
# TODO: Adjust data to match specific observation sites, times, seasons, etc.


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def add_poisson_noise(data_array: np.ndarray,
                      rnd_seed: int=None) -> np.ndarray:
    """Adds counting noise to a data array

    Sets negative values to zero and raises a warning

    Parameters
    ----------
    data_array
        Data for analysis
    rnd_seed
        Optionally change the random seed within this function
        This can streamline generating an ensemble of light curves

    Returns
    -------
    np.ndarray
        An array converted from ints back to floats
    """
    if rnd_seed is not None:
        np.random.seed(rnd_seed)
    if isinstance(data_array, u.Quantity):
        return_array = data_array.value
    else:
        return_array = data_array.copy()
    test_for_negatives = return_array[return_array < 0]
    if len(test_for_negatives) > 0:
        warnings.warn("Found " + str(len(test_for_negatives)) + " values below zero, setting to zero.", AstropyUserWarning)
        return_array[return_array < 0] = 0
    return 1. * np.random.poisson(return_array)

