
"""
This module generates generic 3D plots for diagnostic and visualization purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ..utils import PointCloud


__all__ = ['scatter_plot_3d', 'set_3d_axes_equal', 'plot_sphere']


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def scatter_plot_3d(point_cloud: PointCloud,
                    ax_object: plt.Axes=None,
                    dict_key: str='scatter_plot_3d',
                    savefile=None) -> dict:
    """Accepts as PointCloud and generates a plot

    Parameters
    ----------
    point_cloud
    ax_object
    dict_key
    savefile
        If None: Show()
        If savefile=='hold', return
        If savefile is a filepath str, saves to savefile location (must include file extension)

    Returns
    -------
    dict
        If successful, returns a dictionary with the various plots

    """

    if not isinstance(point_cloud, PointCloud):
        # Then duck type for now
        pass

    if isinstance(ax_object, plt.Axes):  # Is being called from something else, typically
        ax = ax_object
        all_plots_dict = dict()
    else:  # Is being used as a stand-alone function, typically
        fig = plt.figure(figsize=plt.figaspect(1))
        ax = fig.gca(projection='3d')
        ax.set_aspect('equal', 'box')
        all_plots_dict = {'ax': ax}

    if isinstance(dict_key, (str, int)):
        _dict_key = dict_key
    else:
        raise TypeError("Unsupported dict_key type in scatter_plot_3d")

    all_plots_dict[_dict_key] = ax.scatter(point_cloud[0], point_cloud[1], point_cloud[2],
                                           c=point_cloud.point_color,
                                           s=point_cloud.point_size,
                                           zorder=10,
                                           label=point_cloud.name,
                                           linewidths=point_cloud.linewidth)
    if savefile is None:
        plt.show()
    elif savefile == 'hold':
        return all_plots_dict
    else:
        plt.savefig(savefile)
        plt.close()


#  =============================================================  #


def set_3d_axes_equal(ax: plt.Axes):
    """Squares axes of a 3D plot

    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    From: https://stackoverflow.com/a/31364297

    Parameters
    ----------
    ax

    Returns
    -------

    """
    """

    :param ax: object, a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


#  =============================================================  #


def plot_sphere(ax_object=None, rad=None, mesh_res=20, sphere_color='k', alpha=.5, linewidth=.5):
    if isinstance(ax_object, plt.Axes):  # Is being called from something else, typically
        ax = ax_object
    else:  # Is being used as a stand-alone function, typically
        fig = plt.figure(figsize=plt.figaspect(1))
        ax = fig.gca(projection='3d')
        ax.set_aspect('equal', 'box')

    if rad is None:
        rad = 1

    _u = np.linspace(0, 2 * np.pi, 50)
    _v = np.linspace(0, np.pi, 40)

    _x = rad * np.outer(np.cos(_u), np.sin(_v))
    _y = rad * np.outer(np.sin(_u), np.sin(_v))
    _z = rad * np.outer(np.ones(np.size(_u)), np.cos(_v))
    ax.plot_surface(_x, _y, _z, rcount=mesh_res, ccount=mesh_res,
                    color=sphere_color, linewidth=linewidth, alpha=alpha)
    return ax



