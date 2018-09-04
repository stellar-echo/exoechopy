
"""
Basically just pi.
"""

import numpy as np
from astropy import units as u
from matplotlib import colors as mcolors

__all__ = ['pi_u', 'mpl_colors']

# A unit-version of pi radians, for simplicity in a number of instances
pi_u = np.pi * u.rad

# All basic colors in matplotlib:
mpl_colors = [ki for ki in dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys()]
