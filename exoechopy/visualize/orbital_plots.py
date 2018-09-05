
"""
This module generates plots of individual orbits for diagnostic and visualization purposes.
"""

import numpy as np
from astropy import units as u
from ..simulate.models import *
from ..utils.orbital_physics import *
import matplotlib.pyplot as plt
from astropy.visualization import quantity_support


__all__ = ['orbit_plot']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def orbit_plot(keplerian_orbit, num_points=100, savefile=None):
    if not isinstance(keplerian_orbit, KeplerianOrbit):
        raise TypeError("Must provide a KeplerianOrbit class for plotting.")

    angle_color = 'darkblue'
    time_color = 'darkorange'

    positions_angle = np.array(keplerian_orbit.generate_orbital_positions_by_angle(num_points)).transpose()
    positions_time = np.array(keplerian_orbit.generate_orbital_positions_by_time(num_points)).transpose()

    #  ---------------------------------------------------------  #
    fig, (ax1a, ax2a, ax3a) = plt.subplots(1, 3)
    ax1a.plot(np.linspace(0, 2, num_points), positions_angle[0], color=angle_color)
    ax1a.set_xlabel('Angle (pi)', color=angle_color)
    ax1a.tick_params('x', color=angle_color)
    ax1a.set_ylabel('x position (AU)')

    ax1b = ax1a.twiny()
    ax1b.plot(np.linspace(0, 1, num_points), positions_time[0], color=time_color)
    ax1b.set_xlabel('Time (t/T)', color=time_color)

    #  ---------------------------------------------------------  #
    ax2a.plot(np.linspace(0, 2, num_points), positions_angle[1], color=angle_color)
    ax2a.set_xlabel('Angle (pi)', color=angle_color)
    ax2a.set_ylabel('y position (AU)')

    ax2b = ax2a.twiny()
    ax2b.plot(np.linspace(0, 1, num_points), positions_time[1], color=time_color)
    ax2b.set_xlabel('Time (t/T)', color=time_color)

    #  ---------------------------------------------------------  #
    ax3a.plot(np.linspace(0, 2, num_points), positions_angle[2], color=angle_color)
    ax3a.set_xlabel('Angle (pi)', color=angle_color)
    ax3a.set_ylabel('y position (AU)')

    ax3b = ax3a.twiny()
    ax3b.plot(np.linspace(0, 1, num_points), positions_time[2], color=time_color)
    ax3b.set_xlabel('Time (t/T)', color=time_color)

    #  ---------------------------------------------------------  #
    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile)
        plt.close()
    else:
        plt.show()
