

"""Shows how the math_operations module functions through examples."""


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from exoechopy.utils import PointCloud
from exoechopy.visualize import *
from exoechopy.utils import math_operations


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def run():
    min_long = np.pi/12
    max_long = 2*np.pi - np.pi/12
    LongitudeGenerator = stats.uniform(loc=min_long, scale=max_long-min_long)
    # Generate, then freeze the distribution:
    min_lat = np.pi/6
    max_lat = np.pi / 4
    LatitudeGenerator = math_operations.SphericalLatitudeGen(a=min_lat, b=max_lat)()

    n_points = 1000
    theta_points = LongitudeGenerator.rvs(size=n_points)
    phi_points = LatitudeGenerator.rvs(size=n_points)

    points = math_operations.vect_from_spherical_coords(theta_points, phi_points)
    MyPointCloud = PointCloud(points, point_color="k", display_marker='.',
                              point_size=4, linewidth=0, name="Uniformly distributed points")

    ax_dic = scatter_plot_3d(MyPointCloud, savefile='hold')

    set_3d_axes_equal(ax_dic['ax'])
    plt.legend()

    plt.show()

    #  =============================================================  #

    max_val = 100
    rv_gen_1 = stats.uniform(0, 10)
    random_walk_1 = math_operations.stochastic_flare_process(stop_value=max_val,
                                                             distribution=rv_gen_1)
    plt.plot(random_walk_1, color='r', label="Uniform")

    rv_gen_2 = stats.maxwell(0, 3)
    random_walk_2 = math_operations.stochastic_flare_process(stop_value=max_val,
                                                             distribution=rv_gen_2)
    plt.plot(random_walk_2, color='k', label='Maxwell')

    plt.legend()
    plt.show()

# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #


if __name__ == "__main__":

    run()
