
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


__all__ = ['render_3d_planetary_system', 'plot_3d_keplerian_orbit']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def plot_3d_keplerian_orbit(keplerian_orbit,
                            axes_object=None):
    if not isinstance(keplerian_orbit, KeplerianOrbit):
        raise TypeError("Must provide a KeplerianOrbit class for plotting.")
    if isinstance(axes_object, plt.Axes):  # Is being called from something else, typically
        ax = axes_object
    else:  # Is being used as a stand-alone function, typically
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    positions = np.array(keplerian_orbit.generate_orbital_positions(100)).transpose()

    x0, y0, z0 = keplerian_orbit.calc_xyz_position_in_au(0*u.deg)
    ax.scatter(x0, y0, z0,
               c=keplerian_orbit.point_color,
               s=keplerian_orbit.point_size,
               zorder=10)

    ax.plot(positions[0], positions[1], positions[2],
            color=keplerian_orbit.path_color,
            lw=keplerian_orbit.linewidth, zorder=1
            )



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
        ax.plot([0, representative_distance * earth_direction_vector[0]],
                [0, representative_distance * earth_direction_vector[1]],
                [0, representative_distance * earth_direction_vector[2]], color='r')

        ax.set_xbound(-representative_distance, representative_distance)
        ax.set_ybound(-representative_distance, representative_distance)
        ax.set_zbound(-representative_distance, representative_distance)

        if savefile is None:
            plt.show()
        else:
            plt.savefig(savefile)
            plt.close()



