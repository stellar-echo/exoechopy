
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
        return np.sin(x)/(np.cos(self.a) - np.cos(self.b))


# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from exoechopy.utils import PointCloud
    from exoechopy.visualize import *

    min_long = np.pi/12
    max_long = 2*np.pi - np.pi/12
    LongitudeGenerator = stats.uniform(loc=min_long, scale=max_long-min_long)
    # Generate, then freeze the distribution:
    min_lat = np.pi/6
    max_lat = np.pi / 4
    LatitudeGenerator = SphericalLatitudeGen(a=min_lat, b=max_lat)()

    num_points = 1000
    theta_points = LongitudeGenerator.rvs(size=num_points)
    phi_points = LatitudeGenerator.rvs(size=num_points)

    points = vect_from_spherical_coords(theta_points, phi_points)
    MyPointCloud = PointCloud(points, point_color="k", display_marker='.',
                              point_size=4, linewidth=0, name="Uniformly distributed points")

    ax_dic = scatter_plot_3d(MyPointCloud, savefile='hold')

    set_3d_axes_equal(ax_dic['ax'])
    plt.legend()

    plt.show()
