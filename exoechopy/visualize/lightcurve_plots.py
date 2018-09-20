
"""
This module generates plots of lightcurves for diagnostic and visualization purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from ..utils import *
from ..simulate.models import *

__all__ = ['render_telescope_lightcurve']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

# TODO generalize render_telescope_lightcurve, make it work outside the demo
# TODO make a simpler plotter for showing flares and the light curve


def render_telescope_lightcurve(telescope: Telescope,
                                max_plot_points: (int, list, tuple) = None,
                                flare_color: (str, dict) = None,
                                savefile: str = None
                                ):
    """Generates a visualization of
    
    Parameters
    ----------
    telescope
        Telescope object to grab data from
    max_plot_points
        Limit the number of points to prevent excessive plots.
        Either a single max index or a tuple/list of (min_index, max_index)
    flare_color
        string or mapping of colors onto flares
    savefile
        If None, runs plt.show()
        If filepath str, saves to savefile location (must include file extension)
    """

    if flare_color is None:
        flare_color = 'orange'
    elif isinstance(flare_color, dict):
        flare_color = [flare_color[f.label] if f.label in flare_color else 'k'
                       for f in telescope._all_flares.all_flares]

    if max_plot_points is None:
        min_plot_points = 0
        max_plot_points = len(telescope._time_domain)
    elif isinstance(max_plot_points, (list, tuple)):
        min_plot_points = max_plot_points[0]
        max_plot_points = max_plot_points[1]

    time_domain = telescope._time_domain[min_plot_points:max_plot_points]
    dt = time_domain[1]-time_domain[0]
    #  Convert from counts to ct/s and crop out excess points:
    lightcurve = telescope._pure_lightcurve[min_plot_points:max_plot_points]/dt

    min_time = time_domain[0]
    max_time = time_domain[-1]

    flare_times = telescope._all_flares.all_flare_times
    time_mask = np.where((min_time <= flare_times) & (flare_times <= max_time))
    flare_times = flare_times[time_mask]
    if isinstance(flare_color, list):
        flare_color = [flare_color[ind] for ind in time_mask[0]]

    # =============================================================  #

    fig, ax_array = plt.subplots(nrows=3, figsize=(12, 6))
    ax_array[0].plot(time_domain, lightcurve,
                     color='k', lw=1., drawstyle='steps-post', label="Lightcurve")

    ax_array[0].scatter(flare_times, np.ones(len(flare_times))*np.median(lightcurve),
                        marker="^", color=flare_color)
    ax_array[0].set_xlim(time_domain[0].value, time_domain[-1].value)
    ax_array[0].set_ylim(np.min(lightcurve.value)*.95,
                         np.max(lightcurve.value)*1.1)
    ax_array[0].set_ylabel(u_labelstr(lightcurve))
    ax_array[0].text(.025, .95,
                     "Telescope generates the flares (each marker represents a flare), "
                     "even if they aren't visible from Earth",
                     transform=ax_array[0].transAxes, verticalalignment='top', horizontalalignment='left')

    #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  #
    ax_array[1].scatter(flare_times, telescope._all_earth_flare_angles[time_mask],
                        color=flare_color, marker='^', s=12, label="earth_flare_angles")
    ax_array[1].set_xlim(time_domain[0].value, time_domain[-1].value)
    ax_array[1].set_ylim(0, np.pi*1.2)
    ax_array[1].set_ylabel("Angle ("+u_labelstr(telescope._all_earth_flare_angles)+")")
    ax_array[1].text(.025, .95,
                     "Flare angle relative to Earth depends on star rotation and where it spawns",
                     transform=ax_array[1].transAxes, verticalalignment='top', horizontalalignment='left')

    #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  #
    ax_array[2].scatter(flare_times, telescope._earth_flare_visibility[time_mask],
                        color=flare_color, marker='^', s=12, label="earth_flare_visibility")
    ax_array[2].set_xlim(time_domain[0].value, time_domain[-1].value)
    ax_array[2].set_ylim(-.05, 1.2)
    ax_array[2].set_ylabel("Flare visibility")
    ax_array[2].text(.025, .95,
                     "Flare visibility from Earth "
                     "(as the star rotates, the active region emerges then disappears behind), determined by limb model",
                     transform=ax_array[2].transAxes, verticalalignment='top', horizontalalignment='left')

    #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  #
    if len(telescope.observation_target._active_region_list) == 1:
        ar = telescope.observation_target._active_region_list[0]
        long_, lat_ = ar.center_of_region

        angle_array = np.abs(long_-telescope.observation_target.get_rotation(time_domain))
        ax_array[1].plot(time_domain, angle_array,
                         color='wheat', zorder=0, ls='--', lw=1)
        limb_exact = [telescope.observation_target._limb_func(ai)
                      for ai in angle_array]
        ax_array[2].plot(time_domain, limb_exact,
                         color='wheat', zorder=0, ls='--', lw=1)

    fig.subplots_adjust(hspace=0)

    if savefile is None:
        plt.show()
    else:
        plt.savefig(savefile)
        plt.close()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

