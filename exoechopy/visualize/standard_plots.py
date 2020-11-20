"""
A handful of useful plot types.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
from matplotlib.collections import PolyCollection

__all__ = ['plot_simple',
           'plot_lines_and_points', 'plot_time_series_w_fft', 'plot_two_signals_different_y_axis',
           'plot_signal_w_uncertainty', 'plot_signal_w_threshold', 'pseudo_waterfall',
           'step_plot_w_confidence_intervals', 'step_plot_with_zoom_subplot']


# TODO: use the style sheets instead of manually defining everything

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def plot_simple(x_vals, y_vals,
                save=None, x_label="", y_label="", plt_title="",
                x_min=None, x_max=None):
    plt.plot(x_vals, y_vals, color='k', zorder=0, drawstyle='steps-post', lw=1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plt_title, y=1.1)
    if x_min is not None and x_max is not None:
        plt.xlim(left=x_min, right=x_max)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()


def plot_lines_and_points(line_x, line_y, pts_x, pts_y,
                          save=None, line_label="", pts_label="",
                          x_label="", y_label="", plt_title="",
                          x_min=None, x_max=None, plt_type='scatter'):
    plt.plot(line_x, line_y, color='k', label=line_label, zorder=0)
    if plt_type == 'stem':
        plt.stem(pts_x, pts_y, linefmt='r--', markerfmt='ro', label=pts_label)
    else:
        plt.scatter(pts_x, pts_y, marker='.', color='.5', label=pts_label, zorder=1)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plt_title, y=1.1)
    if x_min is not None and x_max is not None:
        plt.xlim(left=x_min, right=x_max)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def plot_time_series_w_fft(x_data, y_data, cadence=1, save=None,
                           x_label="", y_label="", plt_title="",
                           x_min=None, x_max=None):
    """
    Generate a time series plot and its FFT
    :param x_data: time domain in the units you want
    :param y_data: the data
    :param cadence: used for getting the right frequencies
    :param save: if None, shows the result, if a string, saves to that location
    :param x_label:
    :param y_label:
    :param plt_title:
    :param x_min: optionally clip the time domain, for very long samples typically
    :param x_max:
    :return: nothing
    """
    plt.subplot(2, 1, 1)
    plt.plot(x_data, y_data, color='k')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plt_title)
    if x_min is not None and x_max is not None:
        plt.xlim(left=x_min, right=x_max)
    plt.subplot(2, 1, 2)
    fft = np.fft.rfft((y_data - np.mean(y_data)) * np.blackman(len(y_data)))
    fft_freq = np.fft.rfftfreq(len(x_data), cadence)
    plt.plot(fft_freq, np.abs(fft), color='b')
    plt.ylabel("FFT magnitude")
    plt.xlabel("Frequency (Hz)")
    plt.title(plt_title + " FFT (Blackman window)")
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def plot_two_signals_different_y_axis(x_data, y_data1, y_data2, save=None,
                                      x_label="", y_label_1="", y_label_2="", plt_title="",
                                      color_1='r', color_2='b'):
    fig, ax1 = plt.subplots()
    ax1.plot(x_data, y_data1, label=y_label_1, color=color_1)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label_1, color=color_1)
    ax1.tick_params('y', colors=color_1)

    ax2 = ax1.twinx()
    ax2.plot(x_data, y_data2, label=y_label_2, color=color_2)
    ax2.set_ylabel(y_label_2, color=color_2)
    ax2.tick_params('y', colors=color_2)

    plt.title(plt_title, y=1.1)
    fig.tight_layout()
    if save is not None:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def plot_signal_w_uncertainty(x_data: np.ndarray, y_data: np.ndarray, y_plus_interval: np.ndarray,
                              y_minus_interval: np.ndarray = None, y_data_2: np.ndarray = None,
                              save: str = None, x_axis_label: str = "", y_axis_label: str = "",
                              y_label_1: str = "", y_label_2: str = "", plt_title: str = "",
                              color_1: str = 'k', color_2: str = 'r',
                              uncertainty_color: (str, list) = '.85', uncertainty_label: (str, list) = "",
                              axes_object: plt.Axes = None):
    """Generates a standardized plot for a signal and its associated errors

    Parameters
    ----------
    x_data
        Values along x-axis
    y_data
        Primary signal values along y-axis
    y_plus_interval
        Uncertainty in signal on the high side
        Note: this plots as y_plus_interval, not as y_data + y_plus_interval
    y_minus_interval
        Uncertainty in signal on the low side
        Note: this plots as y_minus_interval, not as y_data + y_minus_interval
    y_data_2
        Optional second line to plot, such as the exact solution
    save
        If None, runs plt.show()
        If a filepath is provided, saves file
        If 'ax', returns the axes for further plotting (also works with 'hold')
    x_axis_label
        Plot label for x axis
    y_axis_label
        Plot label for y axis
    y_label_1
        Data 1 label, used in generating legend
    y_label_2
        Data 2 label, used in generating legend
    plt_title
        Plot title
    color_1
        Color for data 1
    color_2
        Color for data 2
    uncertainty_color
        Base color(s) for uncertainty
    uncertainty_label
        Label for uncertainty interval(s), used in generating legend
        May be input as a list, such as ['r', 'b'] if multiple uncertainty intervals are provided
    axes_object
        Optional existing axes object to plot onto


    Returns
    -------
    plt.Axes
        Only returns if save is 'ax' or 'hold'
    """

    if isinstance(axes_object, plt.Axes):  # Is being called from something else, typically
        ax = axes_object
    else:  # Is being used as a stand-alone function, typically
        fig, ax = plt.subplots()

    if y_minus_interval is None:
        y_minus_interval = y_plus_interval

    max_z = 20

    ax.plot(x_data, y_data, color=color_1, label=y_label_1, zorder=max_z, drawstyle='steps-post')
    if y_data_2 is not None:
        max_z -= 1
        ax.plot(x_data, y_data_2,
                color=color_2, linestyle='--', drawstyle='steps-post',
                lw=1, label=y_label_2, zorder=max_z)

    # See if we have a single interval or multiple, handle separately
    if len(np.shape(y_plus_interval)) == 2:
        if isinstance(uncertainty_color, (str, int, float)):
            uncertainty_color = [uncertainty_color for c_i in range(len(y_plus_interval))]
        if isinstance(uncertainty_label, str):
            uncertainty_label = [uncertainty_label for c_i in range(len(y_plus_interval))]
        for y_plus, y_minus, unc_col, unc_label in zip(y_plus_interval, y_minus_interval,
                                                       uncertainty_color, uncertainty_label):
            fill_col = [(1 - (1 - c_i) * .5) for c_i in mpl_colors.to_rgb(unc_col)]
            max_z -= 1
            ax.plot(x_data, y_plus, color=unc_col, lw=1, zorder=max_z, drawstyle='steps-post')
            ax.plot(x_data, y_minus, color=unc_col, lw=1, zorder=max_z, drawstyle='steps-post')
            max_z -= 1
            ax.fill_between(x_data, y_plus, y_minus, facecolor=fill_col, label=unc_label, zorder=max_z, step='post')
    else:
        fill_col = [(1 - (1 - c_i) * .5) for c_i in mpl_colors.to_rgba(uncertainty_color)]
        max_z -= 1
        ax.plot(x_data, y_plus_interval, color=uncertainty_color, lw=1, zorder=max_z, drawstyle='steps-post')
        ax.plot(x_data, y_minus_interval, color=uncertainty_color, lw=1, zorder=max_z, drawstyle='steps-post')
        max_z -= 1
        ax.fill_between(x_data, y_plus_interval, y_minus_interval, facecolor=fill_col,
                        label=uncertainty_label, zorder=max_z, step='post')

    ax.set_title(plt_title, y=1.1)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    if save is None:
        ax.legend(loc='best')
        plt.tight_layout()
        plt.show(block=True)
    elif save == 'ax' or save == 'hold':
        return ax
    else:
        ax.legend(loc='best')
        plt.tight_layout()
        plt.savefig(save)
        plt.close()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def plot_signal_w_threshold(x_data, y_data, y_threshold,
                            save=None, x_label="", y_label="", plt_title="",
                            color_1="k", color_threshold="r", threshold_label=""):
    plt.plot(x_data, y_data, color=color_1, label=y_label, zorder=0, linewidth=1)
    plt.scatter(x_data[np.where(y_data > y_threshold)], y_data[np.where(y_data > y_threshold)],
                color=color_threshold, zorder=1, label=threshold_label)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode="expand", borderaxespad=0.)
    plt.title(plt_title, y=1.1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    if save is None:
        plt.show()
    else:
        plt.savefig(save)
        plt.close()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def pseudo_waterfall(domain, depth, data_array,
                     save=None, x_label="", y_label="", z_label="", plt_title="",
                     color_line='k', color_face='gray', plot_style='step', log_depth=False):
    verts = []
    if plot_style == 'step':
        step_indexer = [int(x) for x in np.floor(np.arange(len(domain), step=.5))]
        dt = (domain[2] - domain[1]) / 2
        step_domain_offset = dt * (2 * (np.arange(2 * len(domain)) % 2) - 1)
        xvals = domain[step_indexer] + step_domain_offset
        xvals = np.pad(xvals, 1, mode='edge')
        for i_z in range(len(depth)):
            yvals = data_array[i_z][step_indexer]
            yvals = np.pad(yvals, 1, mode='constant', constant_values=np.min(yvals))
            verts.append(list(zip(xvals, yvals)))
        # Need to tweak axes

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    poly = PolyCollection(verts, facecolors=color_face, edgecolors=color_line)
    if log_depth:
        depth_ticks = [str(x) for x in depth]
        depth = np.log(depth)
        ax.set_yticklabels(depth_ticks)
        ax.set_yticks(depth)
    ax.add_collection3d(poly, zs=depth, zdir='y')
    ax.set_xlabel(x_label)
    ax.set_xlim3d(min(domain), max(domain))
    ax.set_ylabel(y_label)
    ax.set_ylim3d(min(depth), max(depth))
    ax.set_zlabel(z_label)
    ax.set_zlim3d(np.min(data_array), np.max(data_array))
    plt.title(plt_title)

    if save is None:
        plt.show()
    else:
        plt.savefig(save)
        plt.close()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def step_plot_w_confidence_intervals(x_data, y_data_1, y_data_2=None, y_data_3=None,
                                     confidence_interval_list=None, special_points_xy=None,
                                     y_data_1_error_max_list=None, y_data_1_error_min_list=None,
                                     y_data_2_error_max_list=None, y_data_2_error_min_list=None,
                                     x_label="", y_label="", plt_title="", special_points_label="",
                                     y_label_1="", y_label_2="", y_label_3="",
                                     save=None, figure_size=None, y_range=None, ticks_left=True):
    if figure_size is not None:
        f = plt.figure(figsize=figure_size)
        ax = f.add_subplot(111)
    else:
        f = plt.figure()
        ax = f.add_subplot(111)
    if y_data_2 is None or y_data_3 is not None:
        dt = (x_data[1] - x_data[0]) / 2
        segments = np.linspace(x_data[0] - dt, x_data[-1] + dt, len(x_data) + 1)
        delta = dt / 10
        old_seg = segments[0] + delta
        for si, seg in enumerate(segments[1:]):
            if y_data_3 is None:
                yval = y_data_1[si]
                line_label = y_label_1
            else:
                yval = y_data_3[si]
                line_label = y_label_3
            if si == 0:
                plt.plot([old_seg, seg - delta], [yval, yval],
                         color="k", zorder=10, label=line_label)
            else:
                plt.plot([old_seg, seg], [yval, yval],
                         color="k", zorder=10)
            old_seg = seg + delta
    if y_data_2 is not None:
        plt.scatter(x_data, y_data_2, color="orangered", marker='^',
                    label=y_label_2, zorder=7)
        if y_data_3 is not None:
            plt.scatter(x_data, y_data_1, color="orange", marker='v',
                        label=y_label_1, zorder=6)
    if confidence_interval_list is not None:
        for ci, confidence in enumerate(confidence_interval_list):
            plt.step(x_data, y_data_1_error_max_list[ci], color="orangered",
                     label=str(confidence) + "% confidence", zorder=3, where='mid',
                     linewidth=0.5)
            plt.step(x_data, y_data_1_error_min_list[ci], color="orangered", where='mid', zorder=3,
                     linewidth=0.5)
            plt.fill_between(x_data, y_data_1_error_max_list[ci],
                             y_data_1_error_min_list[ci], step='mid',
                             color='orangered', alpha=.2, zorder=1)
            if y_data_2_error_max_list is not None:
                plt.step(x_data, y_data_2_error_max_list[ci], color="orange", where='mid', zorder=5,
                         linewidth=0.5)
                plt.step(x_data, y_data_2_error_min_list[ci], color="orange", where='mid', zorder=5,
                         linewidth=0.5)
                plt.fill_between(x_data, y_data_2_error_max_list[ci],
                                 y_data_2_error_min_list[ci], step='mid',
                                 color='bisque', zorder=4)
    if special_points_xy is not None:
        plt.autoscale(False)
        plt.scatter(special_points_xy[0], special_points_xy[1], color='k', marker='+', zorder=12,
                    label=special_points_label)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.title(plt_title, y=1.35)
    if not ticks_left:
        ax.yaxis.tick_right()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if y_range is not None:
        plt.ylim(y_range)
    plt.tight_layout()
    if save is None:
        plt.show()
    else:
        plt.savefig(save)
        plt.close()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def step_plot_with_zoom_subplot(x_data, y_data, ylabel="", xlabel="", data_label="", plt_title="",
                                save=None, figsize=(6, 4.5), subplot_rect=[0.4, 0.3, 0.5, 0.5]):
    # Plot the autocorrelation function in a helpful way:

    fig, ax1 = plt.subplots(figsize=figsize)
    left, bottom, width, height = subplot_rect

    ax1.step(x_data, y_data, color='gray', label=data_label, where='mid')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plt_title, y=1.1)
    plt.tight_layout()

    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.step(x_data, y_data, color='gray', zorder=1, where='mid')
    maxval_array = y_data
    maxval_array[0] = maxval_array[-1]
    minval = np.min(y_data)
    maxval = np.max(maxval_array)
    d_v = (maxval - minval)
    ax2.set_ylim(bottom=minval - .05 * d_v, top=maxval + .2 * d_v)
    if save is None:
        plt.show()
    else:
        plt.savefig(save)
        plt.close()
