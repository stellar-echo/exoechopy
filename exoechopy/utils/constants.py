
"""
Basically just pi.
"""

import numpy as np
import types
from astropy import units as u
from matplotlib import colors as mcolors
from scipy import stats

__all__ = ['pi_u',
           'lw_distance_unit', 'lw_time_unit', 'lw_freq_unit', 'lw_mass_unit',
           'CountType', 'PDFType', 'IntFloat', 'RVFrozen', 'FunctionType',
           'mpl_colors']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


# An astropy unit-version of pi radians, for simplicity in a number of instances
pi_u = np.pi * u.rad


# Base units, often used in lightweight (lw) calculations
lw_distance_unit = u.m
lw_time_unit = u.s
lw_freq_unit = u.uHz  # Frequency of occurrences, such as flares
lw_mass_unit = u.kg


# Specialty types (currently using tuples, had compatibility issues with typing module)
CountType = (int, float, u.Quantity)
PDFType = (*CountType, list, tuple, stats._distn_infrastructure.rv_frozen)
IntFloat = (int, float)
RVFrozen = stats._distn_infrastructure.rv_frozen
FunctionType = types.FunctionType

# All basic colors in matplotlib:
mpl_colors = [ki for ki in dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys()]


