
"""
This module generates generic 3D plots for diagnostic and visualization purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from exoechopy.utils import PointCloud

__all__ = ['scatter_plot_3d']


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def scatter_plot_3d(point_cloud: PointCloud,
                    axes_object=None,
                    dict_key='scatter_plot_3d',
                    savefile=None) -> dict:
    """
    Accepts as PointCloud and generates a plot
    :param point_cloud:
    :param axes_object:
    :param dict_key:
    :param str savefile: If None: Show()
                         If savefile=='hold', return
                         If filepath str, saves to savefile location (must include file extension)
    :return dict: If successful, returns a dictionary with the various plots
    """
    if not isinstance(point_cloud, PointCloud):
        # Then duck type for now
        pass

    if isinstance(axes_object, plt.Axes):  # Is being called from something else, typically
        ax = axes_object
    else:  # Is being used as a stand-alone function, typically
        fig = plt.figure(figsize=plt.figaspect(1))
        ax = fig.gca(projection='3d')
        ax.set_aspect('equal')

    if isinstance(dict_key, (str, int)):
        _dict_key = dict_key
    else:
        raise TypeError("Unsupported dict_key type in scatter_plot_3d")

    all_plots_dict = dict()

    all_plots_dict[_dict_key] = ax.scatter(point_cloud[0], point_cloud[1], point_cloud[2],
                                           c=point_cloud.point_color,
                                           s=point_cloud.point_size,
                                           zorder=10,
                                           label=point_cloud.name,
                                           marker=point_cloud.display_marker)
    if savefile is None:
        plt.show()
    elif savefile == 'hold':
        return all_plots_dict
    else:
        plt.savefig(savefile)
        plt.close()


# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #


if __name__ == "__main__":
    from exoechopy.utils import vect_from_spherical_coords

    N_points = 500
    theta_points = np.random.uniform(0, np.pi*2, size=N_points)
    phi_dist = np.random.uniform(0, 1, size=N_points)
    phi_points_wrong = np.pi*phi_dist
    phi_points = np.arccos(2*phi_dist-1)

    points = vect_from_spherical_coords(theta_points, phi_points)
    points_wrong = vect_from_spherical_coords(theta_points, phi_points_wrong)

    MyPointCloud = PointCloud(points, point_color="k", display_marker='.', name="Uniformly distributed points")
    MyPointCloud2 = PointCloud(points_wrong, point_color="r", display_marker='^', name="Points bunched at poles")

    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')

    scatter_plot_3d(MyPointCloud, axes_object=ax, savefile='hold')
    scatter_plot_3d(MyPointCloud2, axes_object=ax, savefile='hold')

    ax.legend()

    plt.show()
