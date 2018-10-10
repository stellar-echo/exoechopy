
"""
This module generates plots of individual orbits for diagnostic and visualization purposes.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants
from astropy import units as u

from ..simulate.orbital_physics import *

__all__ = ['orbit_plot']


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def orbit_plot(keplerian_orbit: KeplerianOrbit,
               num_points: int=100,
               savefile: str=None):
    """Plots a few useful things for a KeplerianOrbit instance

    Parameters
    ----------
    keplerian_orbit
        Instance of a KeplerianOrbit for plotting
    num_points
        Number of points to plot
    savefile
        If None, plt.show()
        Else, saves to savefile = location+filename

    Returns
    -------

    """
    if not isinstance(keplerian_orbit, KeplerianOrbit):
        raise TypeError("Must provide a KeplerianOrbit class for plotting.")

    angle_color = 'darkblue'
    time_color = 'darkorange'

    positions_angle = np.array(keplerian_orbit.generate_orbital_positions_by_angle(num_points)).transpose()
    positions_time = np.array(keplerian_orbit.generate_orbital_positions_by_time(num_points)).transpose()

    #  ---------------------------------------------------------  #
    fig, (ax1a, ax2a, ax3a, ax4a, ax5) = plt.subplots(1, 5, figsize=(14, 4))
    ax1a.plot(np.linspace(0, 2, num_points), positions_angle[0], color=angle_color)
    ax1a.set_xlabel('Angle (pi)', color=angle_color)
    ax1a.tick_params('x', color=angle_color, labelcolor=angle_color)
    ax1a.set_ylabel('x position (AU)')

    ax1b = ax1a.twiny()
    ax1b.plot(np.linspace(0, 1, num_points), positions_time[0], color=time_color)
    ax1b.set_xlabel('Time (t/T)', color=time_color)
    ax1b.tick_params('x', color=time_color, labelcolor=time_color)

    #  ---------------------------------------------------------  #
    ax2a.plot(np.linspace(0, 2, num_points), positions_angle[1], color=angle_color)
    ax2a.set_xlabel('Angle (pi)', color=angle_color)
    ax2a.tick_params('x', color=angle_color, labelcolor=angle_color)
    ax2a.set_ylabel('y position (AU)')

    ax2b = ax2a.twiny()
    ax2b.plot(np.linspace(0, 1, num_points), positions_time[1], color=time_color)
    ax2b.set_xlabel('Time (t/T)', color=time_color)
    ax2b.tick_params('x', color=time_color, labelcolor=time_color)

    #  ---------------------------------------------------------  #
    ax3a.plot(np.linspace(0, 2, num_points), positions_angle[2], color=angle_color)
    ax3a.set_xlabel('Angle (pi)', color=angle_color)
    ax3a.tick_params('x', color=angle_color, labelcolor=angle_color)
    ax3a.set_ylabel('z position (AU)')

    ax3b = ax3a.twiny()
    ax3b.plot(np.linspace(0, 1, num_points), positions_time[2], color=time_color)
    ax3b.set_xlabel('Time (t/T)', color=time_color)
    ax3b.tick_params('x', color=time_color, labelcolor=time_color)

    #  ---------------------------------------------------------  #
    r_pos_angle = np.linalg.norm(positions_angle, axis=0)
    r_pos_time = np.linalg.norm(positions_time, axis=0)
    ax4a.plot(np.linspace(0, 2, num_points), r_pos_angle, color=angle_color)
    ax4a.set_xlabel('Angle (pi)', color=angle_color)
    ax4a.tick_params('x', color=angle_color, labelcolor=angle_color)
    ax4a.set_ylabel('Distance (AU)')

    ax4b = ax4a.twiny()
    ax4b.plot(np.linspace(0, 1, num_points), r_pos_time, color=time_color)
    ax4b.set_xlabel('Time (t/T)', color=time_color)
    ax4b.tick_params('x', color=time_color, labelcolor=time_color)

    #  ---------------------------------------------------------  #
    # Pad the positions to support differentiation:
    # print("positions_time[:, 0]: ", positions_time[:, 0], "\tpositions_time[:, -1]", positions_time[:, -1])
    # r_pos_time_2 = np.hstack((positions_time, positions_time[:, 0, np.newaxis]))*u.au
    dt = keplerian_orbit.orbital_period/num_points
    v_time = (positions_time[:, 2:]-positions_time[:, :-2])/(2*dt)
    kinetic_energy = .5*np.linalg.norm(v_time, axis=0)**2*u.kg*u.au**2/u.s**2
    ax5.plot(np.linspace(0, 1, len(kinetic_energy)), kinetic_energy.to(u.erg), color=angle_color, label='KE')

    potential_energy = -keplerian_orbit.parent_mass * constants.G / (r_pos_time * u.au) * u.kg
    ax5.plot(np.linspace(0, 1, num_points), potential_energy.to(u.erg), color=time_color, label='PE')

    ax5.plot(np.linspace(0, 1, len(kinetic_energy)), (potential_energy[1:-1]+kinetic_energy).to(u.erg), color='k', label='PE+KE')

    ax5.set_xlabel('Time (t/T)')
    ax5.set_ylabel('Energy (erg)')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    #  ---------------------------------------------------------  #
    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile)
        plt.close()
    else:
        plt.show()
