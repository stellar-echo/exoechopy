

"""Shows how the visualization modules function through examples."""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from exoechopy.utils import vect_from_spherical_coords
from exoechopy.utils.plottables import PointCloud
from exoechopy.visualize.standard_3d_plots import scatter_plot_3d


def run():
    n_points = 2000
    theta_points_1 = np.random.uniform(0, np.pi, size=n_points)
    theta_points_2 = np.random.uniform(np.pi, np.pi * 2, size=n_points)
    phi_dist = np.random.uniform(0, 1, size=n_points)
    phi_points_wrong = np.pi*phi_dist
    phi_points = np.arccos(2*phi_dist-1)

    points = vect_from_spherical_coords(theta_points_1, phi_points)
    points_wrong = vect_from_spherical_coords(theta_points_2, phi_points_wrong)

    MyPointCloud = PointCloud(points, point_color="k", display_marker='.', point_size=2, linewidth=0,
                              name="Uniformly distributed points")
    MyPointCloud2 = PointCloud(points_wrong, point_color="r", display_marker='^', point_size=2, linewidth=0,
                               name="Points bunched at poles")

    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')

    scatter_plot_3d(MyPointCloud, ax_object=ax, savefile='hold')
    scatter_plot_3d(MyPointCloud2, ax_object=ax, savefile='hold')

    ax.legend()

    plt.show()


# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #


if __name__ == "__main__":
    run()
