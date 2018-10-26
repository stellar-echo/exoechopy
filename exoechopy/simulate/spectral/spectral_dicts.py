
"""
This module provides specific spectral band properties.
"""

import numpy as np
from astropy import units as u

johnson_band_center_dict = {"U": 360 * u.nm,
                            "B": 440 * u.nm,
                            "V": 550 * u.nm,
                            "R": 640 * u.nm,
                            "I": 790 * u.nm,
                            "J": 1260 * u.nm,
                            "H": 1600 * u.nm,
                            "K": 2220 * u.nm,
                            "g": 520 * u.nm,
                            "r": 670 * u.nm,
                            "i": 790 * u.nm,
                            "z": 910 * u.nm
                            }

johnson_bandwidth_dict = {"U": 54 * u.nm,
                          "B": 97 * u.nm,
                          "V": 88 * u.nm,
                          "R": 147 * u.nm,
                          "I": 150 * u.nm,
                          "J": 202 * u.nm,
                          "H": 368 * u.nm,
                          "K": 511 * u.nm,
                          "g": 73 * u.nm,
                          "r": 94 * u.nm,
                          "i": 126 * u.nm,
                          "z": 118 * u.nm
                          }

johnson_band_flux_dict = {"U": 1810 * u.Jy,
                          "B": 4260 * u.Jy,
                          "V": 3640 * u.Jy,
                          "R": 3080 * u.Jy,
                          "I": 2550 * u.Jy,
                          "J": 1600 * u.Jy,
                          "H": 1080 * u.Jy,
                          "K": 670 * u.Jy,
                          "g": 3730 * u.Jy,
                          "r": 4490 * u.Jy,
                          "i": 4760 * u.Jy,
                          "z": 4810 * u.Jy
                          }


# Based on the models from BATMAN
# See https://www.cfa.harvard.edu/~lkreidberg/batman/tutorial.html#limb-darkening-options

def uniform(mu, *args):
    try:
        return np.ones(len(mu))
    except TypeError:
        return 1.


def linear(mu, *args):
    return 1. - args[0]*(1-mu)


def quadratic(mu, *args):
    return 1. - args[0]*(1-mu) - args[1]*(1-mu)**2


def square_root(mu, *args):
    return 1. - args[0]*(1-mu) - args[1]*(1-mu**.5)


def logarithmic(mu, *args):
    return 1. - args[0]*(1-mu) - args[1]*mu*np.log(mu)


def exponential(mu, *args):
    return 1. - args[0]*(1-mu) - args[1]/(1-np.exp(mu))


def power2(mu, *args):
    return 1. - args[0]*(1-mu**args[1])


def nonlinear(mu, *args):
    return 1. - args[0]*(1-mu**.5) - args[1]*(1-mu) - args[2]*(1-mu**1.5) - args[3]*(1-mu**2)


limb_dict = {'uniform': uniform,
             'linear': linear,
             'quadratic': quadratic,
             'square-root': square_root,
             'logarithmic': logarithmic,
             'exponential': exponential,
             'power2': power2,
             'nonlinear': nonlinear}

