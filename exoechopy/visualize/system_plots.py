
"""
This module generates plots of planetary systems for diagnostic and visualization purposes.
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.visualization import quantity_support
from astropy.utils.exceptions import AstropyUserWarning
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

from ..simulate import *
from .standard_3d_plots import *
from ..utils import *


__all__ = ['render_3d_planetary_system', 'plot_3d_keplerian_orbit', 'animate_3d_planetary_system']


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def plot_3d_keplerian_orbit(keplerian_orbit: KeplerianExoplanet,
                            time: u.Quantity=None,
                            axes_object: plt.Axes=None) -> dict:
    """Tool to plot positions along a Keplerian orbit

    Parameters
    ----------
    keplerian_orbit
        Class instance to be plotted
    time
        Optional argument to provide a specific time for the plot, otherwise plots at 0-deg
    axes_object
        Optional axes object to plot keplerian_orbit onto

    Returns
    -------
    dict
        Returns a dictionary of axes objects that provide access to the various orbital plots.

    """
    all_plots_dict = {}
    if not isinstance(keplerian_orbit, KeplerianOrbit):
        raise TypeError("Must provide a KeplerianOrbit class for plotting.")
    if isinstance(axes_object, plt.Axes):  # Is being called from something else, typically
        ax = axes_object
    else:  # Is being used as a stand-alone function, typically
        fig = plt.figure(figsize=plt.figaspect(1))
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')
    positions = np.array(keplerian_orbit.generate_orbital_positions_by_angle(100)).transpose()

    if time is None:
        x0, y0, z0 = keplerian_orbit.calc_xyz_at_angle_au(keplerian_orbit.initial_anomaly)
    else:
        x0, y0, z0 = keplerian_orbit.calc_xyz_at_time_au_lw(time)

    all_plots_dict['planet_pos'] = ax.scatter(x0, y0, z0,
                                              c=keplerian_orbit.point_color,
                                              s=keplerian_orbit.point_size,
                                              label=keplerian_orbit.name,
                                              marker=keplerian_orbit.display_marker)  #


    all_plots_dict['planet_orbit'] = ax.plot(positions[0], positions[1], positions[2],
                                             color=keplerian_orbit.path_color,
                                             lw=keplerian_orbit.linewidth, zorder=1)

    ascending_node_xyz = keplerian_orbit.calc_xyz_ascending_node_au()
    all_plots_dict['planet_asc_node'] = ax.plot([0, ascending_node_xyz[0]],
                                                [0, ascending_node_xyz[1]],
                                                [0, ascending_node_xyz[2]],
                                                lw=keplerian_orbit.linewidth/2,
                                                linestyle='dashed',
                                                c=keplerian_orbit.point_color)
    return all_plots_dict


#  ------------------------------------------------  #

def plot_3d_precomputed_orbit(object_with_computed_orbit: MassiveObject,
                              time: u.Quantity=None,
                              axes_object: plt.Axes=None) -> dict:
    """Tool to plot positions along a Keplerian orbit

    Parameters
    ----------
    object_with_computed_orbit
        Class instance to be plotted
    time
        Optional argument to provide a specific time for the plot, otherwise plots at 0 sec
    axes_object
        Optional axes object to plot object_with_computed_orbit onto

    Returns
    -------
    dict
        Returns a dictionary of axes objects that provide access to the various orbital plots.

    """
    all_plots_dict = {}
    if not isinstance(object_with_computed_orbit, MassiveObject):
        raise TypeError("Must provide a MassiveObject class for plotting.")
    if isinstance(axes_object, plt.Axes):  # Is being called from something else, typically
        ax = axes_object
    else:  # Is being used as a stand-alone function, typically
        fig = plt.figure(figsize=plt.figaspect(1))
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')

    positions = object_with_computed_orbit._all_positions
    time_domain = object_with_computed_orbit._time_domain

    if time is None:
        x0, y0, z0 = positions[0]
    elif time <= time_domain[-1]:
        x0 = np.interp(time, time_domain, positions[:, 0])
        y0 = np.interp(time, time_domain, positions[:, 1])
        z0 = np.interp(time, time_domain, positions[:, 2])
    else:
        time %= time_domain[-1]

    all_plots_dict['planet_pos'] = ax.scatter(x0, y0, z0,
                                              c=object_with_computed_orbit.point_color,
                                              s=object_with_computed_orbit.point_size,
                                              label=object_with_computed_orbit.name,
                                              marker=object_with_computed_orbit.display_marker)  #

    all_plots_dict['planet_orbit'] = ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                                             color=object_with_computed_orbit.path_color,
                                             lw=object_with_computed_orbit.linewidth, zorder=1)

    return all_plots_dict


#  ------------------------------------------------  #

def render_3d_planetary_system(star_system: DeltaStar,
                               savefile: str=None,
                               show_earth_vector: bool=True) -> dict:
    """Generates a visualization of a Star system

    Accepts a Star class object and generates a visualization of the planets orbiting and observation angles.
    Plots in u.au units

    Parameters
    ----------
    star_system
    savefile
        If None, runs plt.show()
        If filepath str, saves to savefile location (must include file extension)
        If 'return_axes', returns a dictionary with the various axes
    show_earth_vector
        Whether or not to draw the vector that points towards Earth

    Returns
    -------
    dict
        If savefile=='return_axes', returns a dictionary with the various axes for external control

    """

    if not isinstance(star_system, MassiveObject):
        raise TypeError("star_system must be an instance of MassiveObject")

    fig = plt.figure(figsize=plt.figaspect(1))
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    # ax = fig.add_subplot(111, projection='3d')

    system_plot_dict = {'system_plot': fig}

    with quantity_support():
        orbiting_bodies = star_system.get_all_orbiting_objects()
        if len(orbiting_bodies) > 0:
            representative_distance = 0
            for i, body in enumerate(orbiting_bodies):
                if body._all_positions is not None:
                    system_plot_dict[i] = plot_3d_precomputed_orbit(body, axes_object=ax)
                    representative_distance = max(representative_distance,
                                                  np.max(np.abs(body._all_positions.to(u.au).value)))
                elif isinstance(body, KeplerianOrbit):
                    system_plot_dict[i] = plot_3d_keplerian_orbit(body, axes_object=ax)
                    representative_distance = max(representative_distance, body.semimajor_axis.to(u.au).value)
        else:
            try:
                representative_distance = star_system.radius.to(u.au).value*3
            except AttributeError:
                representative_distance = .3

        if show_earth_vector:
            # Show which direction Earth is located at:
            try:
                earth_direction_vector = star_system.earth_direction_vector
                system_plot_dict['earth_pointer'] = ax.quiver(0, 0, 0,
                          representative_distance * earth_direction_vector[0],
                          representative_distance * earth_direction_vector[1],
                          representative_distance * earth_direction_vector[2],
                          color='r', arrow_length_ratio=.1)
            except AttributeError:
                warnings.warn("No Earth direction vector specified for an object, probably not a problem...",
                              AstropyUserWarning)
        try:
            star_radius = star_system.radius.to(u.au)
        except AttributeError:
            star_radius = (.5*u.R_sun).to(u.au)

        # Negative radii are used to block plotting of the star surface
        if star_radius >= 0:
            system_plot_dict['star_surface'] = plot_sphere(ax_object=ax, rad=star_radius,
                                                           mesh_res=20, sphere_color=star_system.point_color, alpha=.5)

        ax.set_xbound(-representative_distance, representative_distance)
        ax.set_ybound(-representative_distance, representative_distance)
        ax.set_zbound(-representative_distance, representative_distance)

        ax.set_xlabel("(au)")
        ax.set_ylabel("(au)")
        ax.set_zlabel("(au)")

        ax.legend()
        ax.view_init(elev=45, azim=45)

        if savefile is None:
            plt.show()
        elif savefile == "return_axes":
            return system_plot_dict
        else:
            plt.savefig(savefile)
            plt.close()


#  ------------------------------------------------  #

def animate_3d_planetary_system(star_system: Star,
                                savefile: str=None,
                                show_earth_vector: bool=True,
                                num_frames: int=500):
    """
    Accepts a Star class object and generates a visualization of the planets orbiting and observation angles.
    Plots in u.au units

    Parameters
    ----------
    star_system
    savefile
        If None, runs plt.show()
        If filepath str, saves to savefile location (must include file extension)
    show_earth_vector
        Whether or not to draw the vector that points towards Earth
    num_frames
        Number of frames to include in animation

    Returns
    -------

    """
    if not isinstance(star_system, MassiveObject):
        raise TypeError("star_system must be an instance of MassiveObject")

    all_plots_dict = render_3d_planetary_system(star_system,
                                                savefile='return_axes', show_earth_vector=show_earth_vector)
    fig = all_plots_dict['system_plot']
    all_satellites = star_system.get_all_orbiting_objects()

    all_orbital_periods = []
    for body in all_satellites:
        # First see if it has a pre-computed path -- this would override Keplerian parameters
        try:
            all_orbital_periods.append(body._time_domain[-1])
        # If not, then see if it has a Keplerian orbital period:
        except TypeError:
            all_orbital_periods.append(body.orbital_period)
    # print('all_orbital_periods: ', all_orbital_periods)
    max_time = max(all_orbital_periods)

    display_times = np.linspace(0, max_time, num_frames)
    # print('display_times: ', display_times)

    system_animation = animation.FuncAnimation(fig, update_orbits,
                                               fargs=(all_plots_dict, all_satellites),
                                               frames=display_times,
                                               interval=10,
                                               repeat=True, blit=False)

    if savefile is None:
        plt.show()
    else:
        plt.savefig(savefile)
        plt.close()


#  ------------------------------------------------  #

def update_orbits(frame: float, plot_dict: dict, orbiting_bodies_list: list):
    """Used in animations to move planets within an exoplanet system.

    Parameters
    ----------
    frame
        A value passed to update the planet positions
    plot_dict
        A dictionary of axes with a key of 'planet_pos' that contains their 3d scatter plots
    orbiting_bodies_list
        The list of bodies that are to be plotted

    Returns
    -------

    """
    for key, body in enumerate(orbiting_bodies_list):
        try:
            _x = np.interp(frame, body._time_domain, body._all_positions[:, 0])
            _y = np.interp(frame, body._time_domain, body._all_positions[:, 1])
            _z = np.interp(frame, body._time_domain, body._all_positions[:, 2])
            new_position = [_x, _y, _z]
        except TypeError:
            new_position = body.calc_xyz_at_time_au_lw(frame)
        plot_dict[key]['planet_pos']._offsets3d = ([new_position[0]], [new_position[1]], [new_position[2]])


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
