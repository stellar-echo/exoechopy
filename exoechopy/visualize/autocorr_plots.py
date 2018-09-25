
"""
This module generates standard autocorrelation plots for diagnostic and visualization purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from ..utils import *

__all__ = ['plot_autocorr_with_derivatives']

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

