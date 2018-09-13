
"""
This module generates plots of planetary systems for diagnostic and visualization purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.visualization import quantity_support
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from exoechopy.simulate.models.orbital_physics import *
from ..simulate.models import *
from exoechopy.visualize.standard_3d_plots import *

__all__ = ['render_3d_planetary_system', 'plot_3d_keplerian_orbit', 'animate_3d_planetary_system']


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def plot_3d_keplerian_orbit(keplerian_orbit,
                            time=None,
                            axes_object=None):
    all_plots_dict = {}
    if not isinstance(keplerian_orbit, KeplerianOrbit):
        raise TypeError("Must provide a KeplerianOrbit class for plotting.")
    if isinstance(axes_object, plt.Axes):  # Is being called from something else, typically
        ax = axes_object
    else:  # Is being used as a stand-alone function, typically
        fig = plt.figure(figsize=plt.figaspect(1))
        ax = fig.gca(projection='3d')
        ax.set_aspect('equal')
    positions = np.array(keplerian_orbit.generate_orbital_positions_by_angle(100)).transpose()

    if time is None:
        x0, y0, z0 = keplerian_orbit.calc_xyz_at_angle_au(0 * u.deg)
    else:
        x0, y0, z0 = keplerian_orbit.calc_xyz_at_time_au(time)

    all_plots_dict['planet_pos'] = ax.scatter(x0, y0, z0,
                                              c=keplerian_orbit.point_color,
                                              s=keplerian_orbit.point_size,
                                              zorder=10,
                                              label=keplerian_orbit.name,
                                              marker=keplerian_orbit.display_marker)

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

def render_3d_planetary_system(star_system,
                               savefile=None,
                               show_earth_vector=True):
    """
    Accepts a Star class object and generates a visualization of the planets orbiting and observation angles.
    Plots in u.au units

    :param Star star_system:
    :param bool show_earth_vector: Whether or not to draw the vector that points towards Earth
    :param str savefile: If None, Show(), if filepath str, saves to savefile location (must include file extension)
    :return dict: If successful, returns a dictionary with the various plots
    """
    if not isinstance(star_system, Star):
        raise TypeError("star_system must be an instance of Star")

    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    # ax = fig.add_subplot(111, projection='3d')

    system_plot_dict = {'system_plot':fig}

    with quantity_support():
        orbiting_bodies = star_system.get_exoplanets()
        if len(orbiting_bodies) > 0:
            for i, body in enumerate(orbiting_bodies):
                system_plot_dict[i] = plot_3d_keplerian_orbit(body, axes_object=ax)
            representative_distance = (orbiting_bodies[-1].semimajor_axis.to(u.au)).value
        else:
            representative_distance = 1.

        if show_earth_vector:
            # Show which direction Earth is located at:
            earth_direction_vector = star_system.earth_direction_vector
            system_plot_dict['earth_pointer'] = ax.quiver(0, 0, 0,
                      representative_distance * earth_direction_vector[0],
                      representative_distance * earth_direction_vector[1],
                      representative_distance * earth_direction_vector[2],
                      color='r', arrow_length_ratio=.1)

        system_plot_dict['star_surface'] = plot_sphere(ax_object=ax, rad=star_system.radius.to(u.au),
                                                       mesh_res=20, sphere_color=star_system.point_color, alpha=.5)

        ax.set_xbound(-representative_distance, representative_distance)
        ax.set_ybound(-representative_distance, representative_distance)
        ax.set_zbound(-representative_distance, representative_distance)

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

def animate_3d_planetary_system(star_system,
                                savefile=None,
                                show_earth_vector=True,
                                num_frames=500):
    """
    Accepts a Star class object and generates a visualization of the planets orbiting and observation angles.
    Plots in u.au units

    :param num_frames: number of frames to include in animation
    :param Star star_system:
    :param str savefile: If None, Show(), if filepath str, saves to savefile location (must include file extension)
    :param bool show_earth_vector: Whether or not to draw the vector that points towards Earth
    """
    if not isinstance(star_system, Star):
        raise TypeError("star_system must be an instance of Star")

    all_plots_dict = render_3d_planetary_system(star_system,
                                                savefile='return_axes', show_earth_vector=show_earth_vector)
    fig = all_plots_dict['system_plot']
    all_exos = star_system.get_exoplanets()

    all_orbital_periods = [body.orbital_period for body in all_exos]
    # print('all_orbital_periods: ', all_orbital_periods)
    max_time = max(all_orbital_periods)

    display_times = np.linspace(0, max_time, num_frames)
    # print('display_times: ', display_times)

    system_animation = animation.FuncAnimation(fig, update_orbits,
                                               fargs=(all_plots_dict, all_exos),
                                               frames=display_times,
                                               interval=10,
                                               repeat=True, blit=False)

    if savefile is None:
        plt.show()
    else:
        plt.savefig(savefile)
        plt.close()


#  ------------------------------------------------  #

def update_orbits(frame, plot_dict, exoplanet_list):
    for key, exoplanet in enumerate(exoplanet_list):
        new_position = exoplanet.calc_xyz_at_time_au(frame)
        plot_dict[key]['planet_pos']._offsets3d = ([new_position[0]], [new_position[1]], [new_position[2]])


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
