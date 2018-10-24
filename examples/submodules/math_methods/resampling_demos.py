
"""This module provides examples of different resampling methods provided by the stats package"""

import numpy as np
from exoechopy.analyze.stats import *
from exoechopy import visualize
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

    test_data = my_distribution.rvs(size=50)
    xvals = np.linspace(0, max(test_data)*1.25, 500)
    MyKDEAnalysis = ResampleKDEAnalysis(test_data, xvals, kde=GaussianKDE, kde_bandwidth='silverman')
    xvals, computed_kde = MyKDEAnalysis.compute_kde()

    conf_lvls = [.6, .9]

    resampled_kde, confidence_intervals = MyKDEAnalysis.bootstrap_with_kde(num_resamples,
                                                                           conf_lvl=conf_lvls)
    print("resampled_kde.shape: ", resampled_kde.shape)
    print("confidence_intervals.shape: ", confidence_intervals.shape)

    visualize.plot_signal_w_uncertainty(xvals, computed_kde,
                                        confidence_intervals[:, 1, :], confidence_intervals[:, 0, :],
                                        y_data_2=rayleigh.pdf(xvals), y_axis_label="PDF", color_2='darkblue',
                                        y_label_1="KDE", y_label_2="Exact solution",
                                        plt_title="Computed KDE and uncertainty",
                                        uncertainty_color=['cadetblue', 'slategray'],
                                        uncertainty_label=[str(ci*100)+"%" for ci in conf_lvls])


# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #


if __name__ == "__main__":

    run()


