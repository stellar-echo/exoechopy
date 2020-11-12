import numpy as np
from matplotlib import pyplot as plt
import exoechopy as eep


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


if __name__ == '__main__':
    test_period = 10
    num_plot_pts = 250
    num_data_pts = 100

    test_data = np.random.random(num_data_pts) ** 2 * test_period
    domain, values = eep.analyze.periodic_kde(test_data, test_period, num_plot_pts, mode='gaussian', bandwidth=test_period/100)
    plt.plot(domain, values, drawstyle='steps-mid', color='k', label='Gaussian')
    domain_th, values_th = eep.analyze.periodic_kde(test_data, test_period, num_plot_pts, mode='tophat')
    plt.plot(domain_th, values_th, drawstyle='steps-mid', color='b', label='Tophat', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.show()
