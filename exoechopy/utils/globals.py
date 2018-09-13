
"""
Basically just pi.
"""

import numpy as np
from astropy import units as u
from matplotlib import colors as mcolors
from scipy import stats

__all__ = ['pi_u',
           'lw_distance_unit', 'lw_time_unit', 'lw_freq_unit',
           'CountType', 'PDFType', 'IntFloat',
           'mpl_colors']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


# An astropy unit-version of pi radians, for simplicity in a number of instances
pi_u = np.pi * u.rad


# Base units, often used in lightweight (lw) calculations
lw_distance_unit = u.m
lw_time_unit = u.s
lw_freq_unit = u.uHz  # Frequency of occurrences, such as flares


# Specialty types (currently using tuples, had compatibility issues with typing module)
CountType = (int, float, u.Quantity)
PDFType = (*CountType, list, tuple, stats._distn_infrastructure.rv_frozen)
IntFloat = (int, float)


# All basic colors in matplotlib:
mpl_colors = [ki for ki in dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys()]


