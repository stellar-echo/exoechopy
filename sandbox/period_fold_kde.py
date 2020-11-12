import numpy as np
from matplotlib import pyplot as plt
import exoechopy as eep


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


if __name__ == '__main__':
    test_period = 10
    num_periods = 20
    num_plot_pts = 250
    num_data_pts = num_periods*5

    # Create some artificial signal and noise:
    background_noise = np.random.random(num_data_pts) ** 2 * test_period*num_periods
    periodic_signal = test_period*.25+test_period*np.arange(num_periods)
    test_data = np.concatenate((periodic_signal, background_noise))
    # So every period there will be one additional datapoint at 25% of the period on top of noise

    # Since this is a demo and we know the exact period, we can simply fold the data at that period:
    domain, values = eep.analyze.periodic_kde(test_data, test_period, num_plot_pts, mode='gaussian')

    # We should also examine the raw data that this came from just to show that periodicity isn't necessarily obvious:
    raw_kde = eep.analyze.GaussianKDE(test_data, bandwidth=test_period/200)
    raw_kde_domain = np.linspace(0, test_period*num_periods, len(test_data)*5)
    raw_kde_values = raw_kde(raw_kde_domain)

    # Plot the data to see that yes, right at 25% of the cycle you can pick out a spike!
    f, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].scatter(test_data, np.zeros_like(test_data), s=2)
    ax[0].plot(raw_kde_domain, raw_kde_values, drawstyle='steps-mid', color='k')
    ax[0].set_title("Raw data")
    ax[1].plot(domain, values, drawstyle='steps-mid', color='k', label='Gaussian')
    domain_th, values_th = eep.analyze.periodic_kde(test_data, test_period, num_plot_pts, mode='tophat')
    ax[1].plot(domain_th, values_th, drawstyle='steps-mid', color='b', label='Tophat', ls='--')
    ax[1].legend()
    ax[1].set_title("Period-folded histogram")
    plt.tight_layout()
    plt.show()
