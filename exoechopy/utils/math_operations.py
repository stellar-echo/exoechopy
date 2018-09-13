
"""
This module provides functions that are useful in computing values with vectors.
Will be broken into separate modules once I know all of the math that is needed.
"""

import numpy as np
from scipy import stats

__all__ = ['angle_between_vectors', 'vect_from_spherical_coords', 'SphericalLatitudeGen']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Vector math


def angle_between_vectors(v1, v2):
    return 2*np.arctan2(np.linalg.norm(np.linalg.norm(v2)*v1-np.linalg.norm(v1)*v2),
                        np.linalg.norm(np.linalg.norm(v2)*v1+np.linalg.norm(v1)*v2))


def vect_from_spherical_coords(longitude, latitude):
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
        return np.sin(x)/2


# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from exoechopy.utils import PointCloud
    from exoechopy.visualize import scatter_plot_3d

    LongitudeGenerator = stats.uniform(loc=0, scale=np.pi*2)
    LatitudeGenerator = SphericalLatitudeGen()(loc=np.pi/4, scale=1/4)

    num_points = 1000
    theta_points = LongitudeGenerator.rvs(size=num_points)
    phi_points = LatitudeGenerator.rvs(size=num_points)

    points = vect_from_spherical_coords(theta_points, phi_points)
    MyPointCloud = PointCloud(points, point_color="k", display_marker='.', name="Uniformly distributed points?")

    scatter_plot_3d(MyPointCloud, savefile='hold')

    plt.legend()
    plt.show()
