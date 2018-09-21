
"""
This module generates plots of lightcurves for diagnostic and visualization purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from astropy import units as u

from ..utils import *
from ..simulate import *

__all__ = ['interactive_lightcurve', 'render_telescope_lightcurve']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

# TODO generalize render_telescope_lightcurve, make it work outside the demo
# TODO add flare times to interactive_lightcurve


def interactive_lightcurve(time_domain: np.ndarray,
                           lightcurve: np.ndarray,
                           flare_times: np.ndarray=None,
                           max_plot_points: int=5000):
    """Provides an interactive plot of a lightcurve

    Parameters
    ----------
    time_domain
        x-values for the plot
    lightcurve
        y values for the plot
    flare_times
        times that flares occur, for optional visualization
    max_plot_points
        default maximum number of points to plot

    """
    if len(time_domain) < max_plot_points:
        max_plot_points = len(time_domain)

    if isinstance(time_domain, u.Quantity):
        x_label = "Time ("+u_labelstr(time_domain)+")"
        time_domain = time_domain.value
    else:
        x_label = "Time"

    if isinstance(lightcurve, u.Quantity):
        y_label = "Intensity ("+u_labelstr(lightcurve)+")"
        lightcurve = lightcurve.value
    else:
        y_label = "Intensity (arb)"

    fig, ax = plt.subplots(figsize=(14, 6))
    plt.subplots_adjust(bottom=0.25)

    line1, = ax.plot(time_domain[:max_plot_points], lightcurve[:max_plot_points],
                     lw=1, drawstyle='steps-post', color='k')
    ax.set_xlim(time_domain[0], time_domain[max_plot_points])
    y0 = min(lightcurve[:max_plot_points])
    y1 = max(lightcurve[:max_plot_points])
    dy = max(1, (y1 - y0) * .025)
    ax.set_ylim(y0 - dy, y1 + dy)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    slider_bg_color = 'whitesmoke'
    pos_ax = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor=slider_bg_color)
    scale_ax = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor=slider_bg_color)

    pos_slider = Slider(pos_ax, "Time", 0, len(time_domain), valinit=max_plot_points//2, valstep=1)
    scale_slider = Slider(scale_ax, "Window", 100, max_plot_points, valinit=max_plot_points, valstep=1)

    reset_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    reset_button = Button(reset_ax, 'Reset', color=slider_bg_color, hovercolor='0.975')

    def update_lightcurve_inner(val):
        ind = pos_slider.val
        width = scale_slider.val
        low_ind, high_ind = window_range(int(ind), len(time_domain)-1, int(width))
        line1.set_data(time_domain[low_ind:high_ind], lightcurve[low_ind:high_ind])
        ax.set_xlim(time_domain[low_ind], time_domain[high_ind])
        y0 = min(lightcurve[low_ind:high_ind])
        y1 = max(lightcurve[low_ind:high_ind])
        dy = max(1, (y1-y0)*.025)
        ax.set_ylim(y0-dy, y1+dy)
        # Future: add np.where( ) to capture correct scatter plot values
        # I'm hesitating because the number of points will change...
        # Also iterate through all axes
        fig.canvas.draw_idle()

    pos_slider.on_changed(update_lightcurve_inner)
    scale_slider.on_changed(update_lightcurve_inner)

    def reset(event):
        pos_slider.reset()
        scale_slider.reset()

    reset_button.on_clicked(reset)

    plt.show()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


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

