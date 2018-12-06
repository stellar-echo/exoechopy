
"""
This module generates standard autocorrelation plots for diagnostic and visualization purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from ..utils import *

__all__ = ['plot_autocorr_with_derivatives', 'display_1d_orbit_bigauss_search']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def plot_autocorr_with_derivatives(raw_autocorr: np.ndarray,
                                   autocorr_time_domain: np.ndarray,
                                   lag_index_spotlight: int=None,
                                   lag_index_spotlight_width: int=None,
                                   savefile: str=None,
                                   deriv_window: int=7,
                                   title: str=None):
    """Shows a three-plot inspection of an autocorrelation function (or related curve)

    Parameters
    ----------
    raw_autocorr
        Function to plot and process
    autocorr_time_domain
        Time domain for autocorrelation
    lag_index_spotlight
        Optional index to produce an inset plot to spotlight part of the curve
    lag_index_spotlight_width
        Width around lag_index_spotlight to show
    savefile
        If None, plt.show()
        Else, saves to savefile = location+filename
    deriv_window
        Derivative window width
    title
        Optional plot title

    """
    fig, (ax_raw, ax_deriv, ax_2nd_deriv) = plt.subplots(1, 3, figsize=(12, 4))

    dt = autocorr_time_domain[1]-autocorr_time_domain[0]
    if isinstance(dt, u.Quantity):
        dt = dt.value

    if lag_index_spotlight is not None:
        if lag_index_spotlight_width is None:
            lag_index_spotlight_width = 10
        ind_min = lag_index_spotlight - lag_index_spotlight_width//2
        ind_max = lag_index_spotlight + lag_index_spotlight_width//2

    ax_raw.plot(autocorr_time_domain, raw_autocorr,
                color='k', lw=1, drawstyle='steps-post')
    ax_raw.set_title("Autocorrelation of signal")
    ax_raw.set_xlabel("Time (s)")
    if lag_index_spotlight is not None:
        inset_ax = ax_raw.inset_axes([0.45, 0.45, 0.45, 0.45])
        inset_ax.plot(autocorr_time_domain[ind_min:ind_max], raw_autocorr[ind_min:ind_max],
                      color='k', lw=1, drawstyle='steps-post')

    #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  #
    deriv_corr = take_noisy_derivative(raw_autocorr, window_size=deriv_window, sample_cadence=dt)
    ax_deriv.plot(autocorr_time_domain, deriv_corr,
                  color='k', lw=1, drawstyle='steps-post')
    ax_deriv.set_title("1st derivative of autocorrelation")
    ax_deriv.set_xlabel("Lag (s)")

    if lag_index_spotlight is not None:
        inset_ax = ax_deriv.inset_axes([0.45, 0.2, 0.45, 0.45])
        inset_ax.plot(autocorr_time_domain[ind_min:ind_max], deriv_corr[ind_min:ind_max],
                      color='k', lw=1, drawstyle='steps-post')

    #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  #
    deriv_corr_2 = -take_noisy_2nd_derivative(raw_autocorr, window_size=deriv_window, sample_cadence=dt)
    ax_2nd_deriv.plot(autocorr_time_domain,
                      deriv_corr_2,
                      color='k', lw=1, drawstyle='steps-post')
    ax_2nd_deriv.set_title("-2nd derivative of autocorrelation")
    ax_2nd_deriv.set_xlabel("Lag (s)")

    if lag_index_spotlight is not None:
        inset_ax = ax_2nd_deriv.inset_axes([0.5, 0.5, 0.45, 0.45])
        inset_ax.plot(autocorr_time_domain[ind_min:ind_max], deriv_corr_2[ind_min:ind_max],
                      color='k', lw=1, drawstyle='steps-post')

    if title is not None:
        fig.suptitle(title, y=1.01)

    fig.tight_layout()

    if savefile is not None:
        plt.savefig(savefile)
        plt.close()
    else:
        plt.show()


def display_1d_orbit_bigauss_search(packed_results,
                                    domain: np.ndarray,
                                    mask: np.ndarray = None,
                                    baseline_result: np.ndarray = None,
                                    interval: float = None,
                                    domain_label: str = None,
                                    save: str = None,
                                    title_str: str = None):
    """Special function to help display the results from the bigauss_kde_orbit_search function

    Note, there's an error with the matplotlib fill_between plots that prevents the last step of a stepwise
    plot from filling.  display_1d_orbit_bigauss_search() compensates for that, but if two discontinuous
    plot regions are separated by a single datapoint, it will fill across the gap!

    Parameters
    ----------
    packed_results
    domain
    mask
    baseline_result
    interval
    domain_label
    save
        If None, runs plt.show()
        If a filepath is provided, saves file
        If 'ax', returns the axes for further plotting (also works with 'hold')
    title_str

    Returns
    -------

    """

    means, lower_means, upper_means, lower_min_conf, lower_max_conf, upper_min_conf, upper_max_conf = packed_results

    if mask is not None:
        # To co-plot the old results and new results that were only computed for select data points,
        # we use np.nans to mask off the other points
        lower_means = discontinuous_plot_data(lower_means, mask)
        upper_means = discontinuous_plot_data(upper_means, mask)
        # TODO: At discontinuities, add a point after in each of the conf's to help the fill_between plot work right
        lower_min_conf = discontinuous_plot_data(lower_min_conf, mask, fill_between_correction=True)
        lower_max_conf = discontinuous_plot_data(lower_max_conf, mask, fill_between_correction=True)
        upper_min_conf = discontinuous_plot_data(upper_min_conf, mask, fill_between_correction=True)
        upper_max_conf = discontinuous_plot_data(upper_max_conf, mask, fill_between_correction=True)

    if baseline_result is None:
        if mask is None:
            baseline_result = means
        else:
            baseline_result = discontinuous_plot_data(means, mask)

    if interval is None:
        interval_str = ""
    else:
        interval_str = " " + str(interval) + "%"

    if domain_label is None:
        domain_label = "Search domain"

    if title_str is None:
        title_str = "Resampled data to identify bimodal distributions"

    plt.figure(figsize=(12.5, 6))
    plt.plot(domain, baseline_result, color='k', drawstyle='steps-post', label="Raw autocorrelation result",
             zorder=20)
    plt.plot(domain, lower_means, color='darkblue', drawstyle='steps-post', ls=':', label="Low estimate",
             zorder=19)
    plt.plot(domain, upper_means, color='darkred', drawstyle='steps-post', ls='--', label="High estimate",
             zorder=18)

    plt.fill_between(domain, lower_min_conf, lower_max_conf,
                     facecolor='skyblue', step='post',
                     label="Low estimate" + interval_str + " conf interval", zorder=5)
    plt.fill_between(domain, upper_min_conf, upper_max_conf,
                     facecolor='salmon', step='post',
                     label="High estimate" + interval_str + " conf interval", zorder=4)
    plt.xlabel(domain_label)
    plt.ylabel("Correlation signal")
    plt.title(title_str)

    plt.legend(bbox_to_anchor=(1.05, .5), loc=2, borderaxespad=0.)
    plt.subplots_adjust(left=.1, right=.7, top=.9)

    if save is None:
        plt.tight_layout()
        plt.show()
    elif save == 'ax' or save == 'hold':
        return plt.gca()
    else:
        plt.tight_layout()
        plt.savefig(save)
        plt.close()

