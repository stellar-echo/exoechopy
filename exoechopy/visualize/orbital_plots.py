
"""
This module generates plots of orbital properties for diagnostic and visualization purposes.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as u

from ..simulate.orbital_physics import *
from ..utils import *

__all__ = ['keplerian_orbit_plot', 'precomputed_orbit_plot', 'precomputed_system_plot']


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def keplerian_orbit_plot(keplerian_orbit: KeplerianOrbit,
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

    potential_energy = -keplerian_orbit.parent_mass * const.G / (r_pos_time * u.au) * u.kg
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


#  =============================================================  #

def precomputed_orbit_plot(orbital_object: MassiveObject,
                           savefile: str=None):
    """Plots a few useful things for a precomputed MassiveObject instance

    Parameters
    ----------
    orbital_object
        Instance of a MassiveObject for plotting
    savefile
        If None, plt.show()
        Else, saves to savefile = location+filename

    """
    if not isinstance(orbital_object, MassiveObject):
        raise TypeError("Must provide a MassiveObject class for plotting.")

    color_1 = 'darkorange'
    color_2 = 'mediumblue'

    positions_time = orbital_object._all_positions
    num_points = len(positions_time)

    #  ---------------------------------------------------------  #
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(14, 4))
    ax1.plot(np.linspace(0, 1, num_points), positions_time[:, 0].to(u.au), color=color_1)
    ax1.set_xlabel('Time (t/T_max)', color=color_1)
    ax1.tick_params('x', color=color_1, labelcolor=color_1)
    ax1.set_ylabel('x position (AU)')

    #  ---------------------------------------------------------  #
    ax2.plot(np.linspace(0, 1, num_points), positions_time[:, 1].to(u.au), color=color_1)
    ax2.set_xlabel('Time (t/T_max)', color=color_1)
    ax2.tick_params('x', color=color_1, labelcolor=color_1)
    ax2.set_ylabel('y position (AU)')

    #  ---------------------------------------------------------  #
    ax3.plot(np.linspace(0, 1, num_points), positions_time[:, 2], color=color_1)
    ax3.set_xlabel('Time (t/T_max)', color=color_1)
    ax3.tick_params('x', color=color_1, labelcolor=color_1)
    ax3.set_ylabel('z position (AU)')

    #  ---------------------------------------------------------  #
    r_pos_time = np.linalg.norm(positions_time, axis=1)
    ax4.plot(np.linspace(0, 2, num_points), r_pos_time, color=color_1)
    ax4.set_xlabel('Time (t/T_max)', color=color_1)
    ax4.tick_params('x', color=color_1, labelcolor=color_1)
    ax4.set_ylabel('Distance to origin (AU)')

    #  ---------------------------------------------------------  #
    v_time = np.transpose(orbital_object._all_velocities)
    kinetic_energy = .5*np.linalg.norm(v_time, axis=0)**2*u.kg*u.au**2/u.s**2
    ax5.plot(np.linspace(0, 1, len(kinetic_energy)), kinetic_energy.to(u.erg), color=color_1, label='KE')

    potential_energy = -orbital_object.parent_mass * const.G / (r_pos_time * u.au) * u.kg
    ax5.plot(np.linspace(0, 1, num_points), potential_energy.to(u.erg), color=color_2, label='PE')

    ax5.plot(np.linspace(0, 1, len(kinetic_energy)), (potential_energy+kinetic_energy).to(u.erg), color='k', label='PE+KE')

    ax5.set_xlabel('Time (t/T_max)')
    ax5.set_ylabel('Energy (erg)')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    #  ---------------------------------------------------------  #
    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile)
        plt.close()
    else:
        plt.show()


#  =============================================================  #

def precomputed_system_plot(*orbital_objects: MassiveObject,
                            savefile: str=None):
    """Plots system energy and momentum for a precomputed MassiveObject system instance

    Parameters
    ----------
    orbital_objects
        Instances of MassiveObjects for plotting
    savefile
        If None, plt.show()
        Else, saves to savefile = location+filename

    """
    all_objects = orbital_objects
    for obj in all_objects:
        if not isinstance(obj, MassiveObject):
            raise TypeError("Must provide a MassiveObject class for plotting.")

    color_1 = 'darkorange'
    color_2 = 'mediumblue'

    num_points = len(all_objects[0]._all_velocities)

    #  ---------------------------------------------------------  #
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(14, 4))

    total_momentum = all_objects[0]._all_velocities*(0*u.kg)
    total_potential_energy = u.Quantity(np.zeros(len(all_objects[0]._time_domain)), u.erg)
    total_kinetic_energy = u.Quantity(np.zeros(len(all_objects[0]._time_domain)), u.erg)
    for obj in all_objects:
        obj_momentum = obj._all_velocities*obj.mass
        total_momentum += obj_momentum
        total_kinetic_energy += .5*obj.mass*np.sum(obj._all_velocities*obj._all_velocities, axis=1)
    for o_i in range(len(all_objects)):
        positions_i = all_objects[o_i]._all_positions
        for o_j in range(o_i+1, len(all_objects)):
            positions_j = all_objects[o_j]._all_positions
            distance_array = u.Quantity(np.linalg.norm(positions_j-positions_i, axis=1), positions_i.unit)
            total_potential_energy -= const.G*all_objects[o_i].mass*all_objects[o_j].mass/distance_array

    total_momentum = np.transpose(total_momentum)

    #  ---------------------------------------------------------  #
    ax1.plot(np.linspace(0, 1, num_points), total_momentum[0], color=color_1)
    ax1.set_xlabel('Time (t/T_max)', color=color_1)
    ax1.tick_params('x', color=color_1, labelcolor=color_1)
    ax1.set_ylabel('Total system x momentum (' + u_labelstr(total_momentum) + ")")

    #  ---------------------------------------------------------  #
    ax2.plot(np.linspace(0, 1, num_points), total_momentum[1], color=color_1)
    ax2.set_xlabel('Time (t/T_max)', color=color_1)
    ax2.tick_params('x', color=color_1, labelcolor=color_1)
    ax2.set_ylabel('Total system y momentum (' + u_labelstr(total_momentum) + ")")

    #  ---------------------------------------------------------  #
    ax3.plot(np.linspace(0, 1, num_points), total_momentum[2], color=color_1)
    ax3.set_xlabel('Time (t/T_max)', color=color_1)
    ax3.tick_params('x', color=color_1, labelcolor=color_1)
    ax3.set_ylabel('Total system z momentum (' + u_labelstr(total_momentum) + ")")

    #  ---------------------------------------------------------  #

    ax4.plot(np.linspace(0, 1, num_points), total_kinetic_energy.to(u.erg),
             color=color_1, label='KE')
    ax4.plot(np.linspace(0, 1, num_points), total_potential_energy.to(u.erg),
             color=color_2, label='PE')

    ax4.set_xlabel('Time (t/T_max)')
    ax4.set_ylabel('System energy (erg)')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    #  ---------------------------------------------------------  #
    total_energy = total_kinetic_energy+total_potential_energy
    ax5.plot(np.linspace(0, 1, num_points),
             100*(total_energy-total_energy[0])/total_energy[0],
             color='k', label='PE+KE', lw=1)
    ax5.set_xlabel('Time (t/T_max)')
    ax5.set_ylabel('System energy deviation from t_0 (%)')

    #  ---------------------------------------------------------  #
    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile)
        plt.close()
    else:
        plt.show()

