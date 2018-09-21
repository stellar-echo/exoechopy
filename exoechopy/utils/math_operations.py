
"""
This module provides functions that are useful in computing values with vectors.
Will be broken into separate modules once I know all of the math that is needed.
"""

import numpy as np
from scipy import stats

__all__ = ['angle_between_vectors', 'vect_from_spherical_coords',
           'SphericalLatitudeGen', 'stochastic_flare_process',
           'window_range']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Vector math


def angle_between_vectors(v1, v2):
    return 2*np.arctan2(np.linalg.norm(np.linalg.norm(v2)*v1-np.linalg.norm(v1)*v2),
                        np.linalg.norm(np.linalg.norm(v2)*v1+np.linalg.norm(v1)*v2))


def vect_from_spherical_coords(longitude, latitude) -> np.ndarray:
    vect = np.array((np.sin(latitude) * np.cos(longitude),
                     np.sin(latitude) * np.sin(longitude),
                     np.cos(latitude) * np.ones(np.shape(longitude))))
    return np.transpose(vect)

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


