
"""
This module provides examples of autocorrelation and deconvolution methods useful for detecting echoes
"""

import numpy as np
import matplotlib.pyplot as plt
from exoechopy.analyze.methods import *
from exoechopy.visualize.lightcurve_plots import *
from exoechopy.utils.astropyio import *
from astropy import units as u

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def run():
    num_points = 2500
    cadence = .5*u.s
    lag_index = 25
    echo_mag = .2
    np.random.seed(101)
    # pad by lag_index to get final dataset length of num_points after adding echo:
    some_random_noise = np.random.rand(num_points+lag_index)
    my_data = u.Quantity(some_random_noise[:-lag_index] + echo_mag*some_random_noise[lag_index:], u.ct)
    time_domain = np.arange(0, num_points*cadence.value, cadence.value)*cadence.unit

    print("We've buried an echo in the noise")
    interactive_lightcurve(time_domain=time_domain, lightcurve=my_data)

    #  =============================================================  #
    print("""
    The stock numpy autocorrelation function returns an unaligned array, this call aligns and normalizes it
    """)

    max_lag = lag_index*2
    lag_domain = np.arange(0, (max_lag+1)*cadence.value, cadence.value)*cadence.unit
    my_data_short = my_data[:len(my_data)//10]
    col_1 = 'k'
    col_2 = 'orangered'

    autocorr = autocorrelate_array(my_data, max_lag=max_lag)
    autocorr_short = autocorrelate_array(my_data_short, max_lag=max_lag)
    print(len(autocorr))
    print(len(autocorr_short))
    plt.plot(lag_domain, autocorr,
             color=col_1, lw=1, drawstyle='steps-post', zorder=5, label=str(num_points)+" pts")
    plt.plot(lag_domain, autocorr_short,
             color=col_2, lw=1, drawstyle='steps-post', zorder=3, label=str(num_points//10)+" pts")
    plt.title("Autocorrelation")
    plt.ylabel("Correlation ("+u_labelstr(autocorr_short)+")")
    plt.xlabel("Lag (time)")

    plt.legend()

    plt.tight_layout()
    plt.show()


    #  =============================================================  #
    print("""
    In some cases, it's advantageous to look at a time-dependent autocorrelation.  
    The autocorrelation_overlapping_windows function is a good first choice.
    """)
    lag_amplitude = 5
    lag_offset = 20
    num_points = 20000
    echo_mag = .5
    raw_data = np.random.rand(num_points)
    offset_list = [(lag_offset+ii+int(round(x)))%num_points for ii, x in zip(range(num_points),
                                            lag_amplitude*np.sin(np.linspace(0, 2*np.pi, num_points)))]
    raw_data += echo_mag*raw_data[offset_list]

    max_lag = lag_offset*2

    time_dependent_autocorr = autocorrelation_overlapping_windows(raw_data,
                                                                  window_length=num_points//10,
                                                                  max_lag=max_lag)

    plt.plot(autocorrelate_array(raw_data, max_lag=max_lag), color=col_1, lw=1, drawstyle='steps-post', zorder=5)
    plt.title("Raw autocorrelation")
    plt.show()

    plt.imshow(time_dependent_autocorr[:, 1:].T, origin='lower', cmap='binary')
    plt.title("Time-dependent autocorrelation")
    plt.show()

    #  =============================================================  #
    print("""
    In some cases, it's advantageous to perform period folding of data as a means of searching datasets.
    """)
    lag_amplitude = 5
    lag_offset = 20
    num_points = 20000
    echo_mag = .2
    num_periods = 4
    raw_data = np.random.rand(num_points)
    offset_list = [(lag_offset+ii+int(round(x)))%num_points for ii, x in zip(range(num_points),
                                            lag_amplitude*np.sin(np.linspace(0, num_periods*2*np.pi, num_points)))]
    raw_data += echo_mag*raw_data[offset_list]

    max_lag = lag_offset*2

    time_dependent_autocorr = autocorrelation_overlapping_windows(raw_data,
                                                                  window_length=num_points//10,
                                                                  max_lag=max_lag)

    period_folded = period_folded_autocorrelation(raw_data, int(num_points/num_periods), 10, max_lag)

    plt.plot(autocorrelate_array(raw_data, max_lag=max_lag), color=col_1, lw=1, drawstyle='steps-post', zorder=5)
    plt.title("Raw autocorrelation")
    plt.show()

    plt.imshow(time_dependent_autocorr[:, 1:].T, origin='lower', cmap='binary')
    plt.title("Time-dependent autocorrelation")
    plt.show()

    plt.imshow(period_folded[:, 1:].T, origin='lower', cmap='binary')
    plt.title("Period-folded autocorrelation")
    plt.show()


# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #


if __name__ == "__main__":

    run()
