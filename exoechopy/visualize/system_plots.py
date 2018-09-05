
"""
This module generates plots of planetary systems for diagnostic and visualization purposes.
"""

import numpy as np
from astropy import units as u
from ..simulate.models import *
from ..utils.orbital_physics import *
import matplotlib.pyplot as plt
from astropy.visualization import quantity_support
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation


__all__ = ['render_3d_planetary_system', 'plot_3d_keplerian_orbit']


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def plot_3d_keplerian_orbit(keplerian_orbit,
                            axes_object=None):
    if not isinstance(keplerian_orbit, KeplerianOrbit):
        raise TypeError("Must provide a KeplerianOrbit class for plotting.")
    if isinstance(axes_object, plt.Axes):  # Is being called from something else, typically
        ax = axes_object
    else:  # Is being used as a stand-alone function, typically
        fig = plt.figure(figsize=plt.figaspect(1))
        ax = fig.gca(projection='3d')
        ax.set_aspect('equal')
    positions = np.array(keplerian_orbit.generate_orbital_positions_by_angle(100)).transpose()

    x0, y0, z0 = keplerian_orbit.calc_xyz_at_angle_au(0 * u.deg)
    ax.scatter(x0, y0, z0,
               c=keplerian_orbit.point_color,
               s=keplerian_orbit.point_size,
               zorder=10,
               label=keplerian_orbit.name,
               marker=keplerian_orbit.display_marker)

    ax.plot(positions[0], positions[1], positions[2],
            color=keplerian_orbit.path_color,
            lw=keplerian_orbit.linewidth, zorder=1)

    ascending_node_xyz = keplerian_orbit.calc_xyz_ascending_node_au()
    ax.plot([0, ascending_node_xyz[0]],
            [0, ascending_node_xyz[1]],
            [0, ascending_node_xyz[2]],
            lw=keplerian_orbit.linewidth/2,
            linestyle='dashed',
            c=keplerian_orbit.point_color)


def render_3d_planetary_system(star_system,
                               savefile=None):
    """
    Accepts a Star class object and generates a visualization of the planets orbiting and observation angles.
    Plots in u.au units

    :param Star star_system:
    :param str savefile: If None, Show(), if filepath str, saves to savefile location (must include file extension)
    :return bool: If successful, returns True
    """
    if not isinstance(star_system, Star):
        raise TypeError("star_system must be an instance of Star")

    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    # ax = fig.add_subplot(111, projection='3d')

    with quantity_support():
        orbiting_bodies = star_system.get_exoplanets()
        if len(orbiting_bodies) > 0:
            for body in orbiting_bodies:
                plot_3d_keplerian_orbit(body, ax)
            representative_distance = (orbiting_bodies[-1].semimajor_axis.to(u.au)).value
        else:
            representative_distance = 1.

        # Show which direction Earth is located at:
        earth_direction_vector = star_system.earth_direction_vector
        ax.quiver(0, 0, 0,
                  representative_distance * earth_direction_vector[0],
                  representative_distance * earth_direction_vector[1],
                  representative_distance * earth_direction_vector[2],
                  color='r', arrow_length_ratio=.1)

        _u = np.linspace(0, 2 * np.pi, 50)
        _v = np.linspace(0, np.pi, 40)

        _x = star_system.radius.to(u.au) * np.outer(np.cos(_u), np.sin(_v))
        _y = star_system.radius.to(u.au) * np.outer(np.sin(_u), np.sin(_v))
        _z = star_system.radius.to(u.au) * np.outer(np.ones(np.size(_u)), np.cos(_v))
        ax.plot_surface(_x, _y, _z, rcount=20, ccount=20, color=star_system.point_color, linewidth=0, alpha=0.5)

        ax.set_xbound(-representative_distance, representative_distance)
        ax.set_ybound(-representative_distance, representative_distance)
        ax.set_zbound(-representative_distance, representative_distance)

        ax.legend()
        ax.view_init(elev=45, azim=45)

        if savefile is None:
            plt.show()
        else:
            plt.savefig(savefile)
            plt.close()


# def animate_3d_planetary_system(star_system,
#                                 savefile=None):
#     """
#     Accepts a Star class object and generates a visualization of the planets orbiting and observation angles.
#     Plots in u.au units
#
#     :param Star star_system:
#     :param str savefile: If None, Show(), if filepath str, saves to savefile location (must include file extension)
#     :return bool: If successful, returns True
#     """
#     if not isinstance(star_system, Star):
#         raise TypeError("star_system must be an instance of Star")
#
#     fig = plt.figure(figsize=plt.figaspect(1))
#     ax = fig.gca(projection='3d')
#     ax.set_aspect('equal')
#     # ax = fig.add_subplot(111, projection='3d')
#
#     with quantity_support():
#         orbiting_bodies = star_system.get_exoplanets()
#         if len(orbiting_bodies) > 0:
#             for body in orbiting_bodies:
#                 plot_3d_keplerian_orbit(body, ax)
#             representative_distance = (orbiting_bodies[-1].semimajor_axis.to(u.au)).value
#         else:
#             representative_distance = 1.
#
#         # Show which direction Earth is located at:
#         earth_direction_vector = star_system.earth_direction_vector
#         ax.quiver(0, 0, 0,
#                   representative_distance * earth_direction_vector[0],
#                   representative_distance * earth_direction_vector[1],
#                   representative_distance * earth_direction_vector[2],
#                   color='r', arrow_length_ratio=.1)
#
#         _u = np.linspace(0, 2 * np.pi, 50)
#         _v = np.linspace(0, np.pi, 40)
#
#         _x = star_system.radius.to(u.au) * np.outer(np.cos(_u), np.sin(_v))
#         _y = star_system.radius.to(u.au) * np.outer(np.sin(_u), np.sin(_v))
#         _z = star_system.radius.to(u.au) * np.outer(np.ones(np.size(_u)), np.cos(_v))
#         ax.plot_surface(_x, _y, _z, rcount=20, ccount=20, color=star_system.point_color, linewidth=0, alpha=0.5)
#
#         ax.set_xbound(-representative_distance, representative_distance)
#         ax.set_ybound(-representative_distance, representative_distance)
#         ax.set_zbound(-representative_distance, representative_distance)
#
#         ax.legend()
#         ax.view_init(elev=45, azim=45)
#
#         system_animation = animation.FuncAnimation(fig, )
#
#         if savefile is None:
#             plt.show()
#         else:
#             plt.savefig(savefile)
#             plt.close()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
