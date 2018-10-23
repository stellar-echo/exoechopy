
"""This module provides examples of different resampling methods provided by the stats package"""

import numpy as np
from exoechopy.analyze.stats import *
import matplotlib.pyplot as plt
from scipy.stats import rayleigh


def run():
    my_distribution = rayleigh()

    np.random.seed(747)

    num_points = 25
    test_data = my_distribution.rvs(size=num_points)
    exact_mean = my_distribution.stats(moments='m')
    mean_val = np.mean(test_data)

    print("Distribution mean: ", exact_mean)
    print("Sample mean: ", mean_val)

    # Add some bad datapoints:
    test_data[0] = max(test_data)*5
    test_data[1] = max(test_data)*3.5
    print("Sample mean after data corruption: ", np.mean(test_data))

    MyAnalysis = ResampleAnalysis(test_data)

    jackknifed_means, distances, conf_min, conf_max = MyAnalysis.jackknife(analysis_func=np.mean, conf_lvl=.95)
    print("np.std(jackknifed_means): ", np.std(jackknifed_means))
    histo = TophatKDE(jackknifed_means)
    spread = max(jackknifed_means)-min(jackknifed_means)
    xvals = np.linspace(min(jackknifed_means)-spread*.1, max(jackknifed_means)+spread*.1, num_points*5)
    plt.plot(xvals, histo(xvals), drawstyle='steps-post', color='b', lw=1)
    plt.scatter(jackknifed_means, np.zeros(len(jackknifed_means)), marker='+', color='k')
    plt.annotate("Found a couple outliers...", xy=(1.7, .33), xytext=(2, 2),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.3"))
    plt.annotate("", xy=(2.9, .33), xytext=(2.5, 1.9),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-.3"))
    plt.title("Distribution of jackknifed samples")
    plt.xlabel("Jackknifed distribution mean value")
    plt.ylabel("Counts")
    plt.show()

    rejected_points = MyAnalysis.remove_outliers_by_jackknife_sigma_testing(np.mean, 1)
    print("Rejected points from analysis based on outlier status: ", rejected_points)

    print("Sample mean after outlier rejection: ", MyAnalysis.apply_function(np.mean))

    num_resamples = 1000
    resampled_means, conf_min, conf_max = MyAnalysis.bootstrap(num_resamples, analysis_func=np.mean, conf_lvl=.95)

    print("Mean of resampled means: ", np.mean(resampled_means))
    print("Minimum estimate of mean within 95% confidence: ", conf_min)
    print("Maximum estimate of mean within 95% confidence: ", conf_max)

    plt.hist(resampled_means)
    plt.title("Distribution of bootstrap resamples after outlier removal")
    plt.xlabel("Resampled mean values")
    plt.ylabel("Counts")
    plt.show()

    #  =============================================================  #

    test_data = my_distribution.rvs(size=num_points)
    xvals = np.linspace(0, max(test_data)*1.25, 500)
    MyKDEAnalysis = ResampleKDEAnalysis(test_data, xvals, kde=GaussianKDE)
    xvals, computed_kde = MyKDEAnalysis.compute_kde()

    resampled_kde, confidence_intervals = MyKDEAnalysis.bootstrap_with_kde(num_resamples,
                                                                           conf_lvl=.9)
    print("resampled_kde.shape: ", resampled_kde.shape)
    print("confidence_intervals.shape: ", confidence_intervals.shape)

    plt.plot(xvals, computed_kde, color='k', zorder=10, label="Kernel density estimate")
    plt.plot(xvals, confidence_intervals[0, 0, :], lw=1, zorder=2, color='r', label="90% confidence interval")
    plt.plot(xvals, confidence_intervals[0, 1, :], lw=1, zorder=2, color='r')
    plt.fill_between(xvals, confidence_intervals[0, 0, :], confidence_intervals[0, 1, :], zorder=0, color='coral')
    plt.title("Computed KDE")
    plt.legend()
    plt.show()


# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #


if __name__ == "__main__":

    run()


