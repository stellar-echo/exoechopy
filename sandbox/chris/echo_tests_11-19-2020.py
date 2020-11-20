import os
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import exoechopy as eep
from astropy.utils import NumpyRNGContext

# How far before the flare to plot when producing the overview data product:
back_pad = 5
# How far after the flare to include in the analysis:
forward_pad = 80
# Threshold for detecting flares:
peak_find_std_dev_thresh = 4.5
# Threshold for rejecting flare based on a jackknife resample:
jk_std_dev_thresh = 4
# Number of bootstrap resamples for developing confidence intervals
num_bootstrap = 10000
# Range to use for confidence intervals
conf_range = 97.7
# Seed to make results reproducible:
random_seed = 0
# Where we are getting raw data from:
file_folder = "11-19-2020 flux"

#  ------------------------------------------  #

np.random.seed(random_seed)

cwd = Path(os.getcwd())
fp = cwd.parent.parent / "disks" / file_folder

all_files = os.listdir(fp)
conf_min = (100 - conf_range) / 2
conf_max = 100 - conf_min

star_data_dict = {}

for f in all_files:
    star_id = f.split('_')[0]
    if 'flux' in f:
        data_type = 'flux'
    elif 'time' in f:
        data_type = 'time'
    else:
        data_type = None
    if star_id not in star_data_dict:
        star_data_dict[star_id] = {}
    star_data_dict[star_id][data_type] = np.load(fp / f)

# Skip the flare itself when just looking at echo:
echo_slice = slice(back_pad + 1, None)


def gaussian(x, mu, sigma):
    denom = np.sqrt(np.pi * 2) * sigma
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / denom


def weighted_avg_and_std(values, weights):
    total_weight = np.sum(weights)
    weights /= total_weight
    normed_mean = np.nansum(values * weights)
    variance = np.nansum((values - normed_mean) ** 2 * weights)
    return (normed_mean, np.sqrt(variance))


def echo_analysis_do_it_all(all_flare_arr: np.ndarray, flare_peak_index=0):
    """

    Parameters
    ----------
    all_flare_arr
        Array of all flares curves for analysis
        Should be the same length
    flare_peak_index
        Which index the flare peaks at


    Weight is calculated by 1/stdDev(post-flare-LC)

    Returns
    -------
    all flares normalized by peak value, their associated weight, and their associated weighted std. dev
    """
    _normed_flares = np.zeros_like(all_flare_arr)
    _normed_flare_weight = np.zeros(len(_normed_flares))

    for f_i, flare in enumerate(all_flare_arr):
        _norm_flare = (flare - 1) / (np.nanmax(flare) - 1)
        _normed_flares[f_i] = _norm_flare
        # Weight is defined here:
        _normed_flare_weight[f_i] = 1 / np.nanstd(flare[flare_peak_index + 1:])

    total_weight = np.sum(_normed_flare_weight)
    _normed_flare_weight /= total_weight
    _normed_mean = np.nansum(_normed_flares * _normed_flare_weight[:, np.newaxis], axis=0)

    # Weighted std deviation of the mean: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Standard_error
    var = np.zeros(len(_normed_flares[0]))
    for f_i, flare in enumerate(_normed_flares.T):
        var[f_i] = np.nansum((flare - _normed_mean[f_i]) ** 2 * _normed_flare_weight ** 2)
    _normed_std_dev = np.sqrt(var)

    return _normed_flares, _normed_flare_weight, _normed_std_dev


def generate_kde_plots(flare_array, weight_array=None, plot_title=None, savefile=None):
    flare_tail_histo = flare_array.flatten()
    flare_tail_weights = weight_array.flatten()
    all_nans = np.isnan(flare_tail_histo)
    flare_tail_histo = flare_tail_histo[~all_nans]
    flare_tail_weights = flare_tail_weights[~all_nans]

    xvals = np.linspace(np.nanmin(flare_tail_histo), np.nanmax(flare_tail_histo), 500)
    dx = xvals[1] - xvals[0]

    final_normed_flare_histo = eep.analyze.TophatKDE(flare_tail_histo,
                                                     bandwidth=5 * dx,
                                                     weights=flare_tail_weights)

    final_normed_flare_histo_gauss = eep.analyze.GaussianKDE(flare_tail_histo,
                                                             bandwidth='scott',
                                                             weights=flare_tail_weights)

    flare_tail_kde = final_normed_flare_histo(xvals)
    flare_tail_kde_gauss = final_normed_flare_histo_gauss(xvals)
    normal_approx = gaussian(xvals, np.nanmean(flare_tail_histo), np.nanstd(flare_tail_histo))

    f, ax = plt.subplots(ncols=2, figsize=(12, 6))
    ax[0].plot(xvals, flare_tail_kde_gauss, color='gray', zorder=0,
               label="Scott's rule Gaussian KDE")
    ax[0].plot(xvals, flare_tail_kde, color='k', zorder=1, label='5-bin Tophat KDE')
    ax[0].plot(xvals, normal_approx, color='b', ls='--', zorder=-1, label='Gaussian fit to distribution')
    ax[0].set_xlabel("Values")
    ax[0].set_ylabel("Relative frequency")
    ax[0].set_title("Distribution of values")
    ax[0].legend()

    cumulative_distribution = np.cumsum(flare_tail_kde)
    cumulative_distribution /= np.max(cumulative_distribution)
    ax[1].plot(xvals, cumulative_distribution, color='k')
    ax[1].set_xlabel("Values")
    ax[1].set_ylabel("Cumulative frequency")
    ax[1].set_title("Cumulative values")
    if plot_title is not None:
        plt.suptitle(plot_title)
    plt.tight_layout()
    if savefile is not None:
        plt.savefig(savefile)
        plt.close()
    else:
        plt.show(block=True)


for star_name in star_data_dict:
    flux = star_data_dict[star_name]['flux']
    time = star_data_dict[star_name]['time']
    f, ax = plt.subplots(figsize=(12, 4))
    plt.plot(time, flux, color='k', drawstyle='steps-mid')
    plt.title("Flux from " + star_name)
    plt.xlabel("Time (d)")
    plt.ylabel("Background-normalized flux")
    plt.show(block=True)

    flare_cat = eep.analyze.LightcurveFlareCatalog(flux,
                                                   extract_range=(back_pad, forward_pad),
                                                   time_domain=time)
    flare_cat.identify_flares_with_protocol(eep.analyze.find_peaks_stddev_thresh,
                                            std_dev_threshold=peak_find_std_dev_thresh,
                                            single_flare_gap=forward_pad)
    print("Number of flares identified: ", flare_cat.num_flares)

    eep.visualize.plot_flare_array(flux, flare_cat.get_flare_indices(),
                                   back_pad=back_pad, forward_pad=forward_pad,
                                   title="Flare catalog from " + star_name + ", "
                                         + str(flare_cat.num_flares) + " flares found for peak>"
                                         + str(peak_find_std_dev_thresh) + "σ")

    all_flares = flare_cat.get_flare_curves()

    time_domain_min = flare_cat.get_relative_indices() * flare_cat.cadence * 24 * 60

    normed_flares, normed_flare_weight, normed_std_dev = echo_analysis_do_it_all(all_flares, back_pad + 1)

    f, ax = plt.subplots(ncols=4, figsize=(16, 4))
    for f_i in range(len(all_flares)):
        ax[0].plot(time_domain_min, all_flares[f_i], drawstyle='steps-mid')
        ax[2].plot(time_domain_min, normed_flares[f_i], drawstyle='steps-mid')

    ax[0].set_title("All flares from " + star_name)
    ax[2].set_title("All peak-normalized flares")

    # Standard deviation of the mean, https://en.wikipedia.org/wiki/Standard_error#Standard_error_of_the_mean
    std_dev_plot = np.nanstd(all_flares, axis=0) / np.sqrt(len(all_flares))
    mean_plot = np.nanmean(all_flares, axis=0)
    normed_mean = np.nansum(normed_flares * normed_flare_weight[:, np.newaxis], axis=0)

    ax[1].plot(time_domain_min, mean_plot + std_dev_plot, drawstyle='steps-mid', color='r')
    ax[1].plot(time_domain_min, mean_plot - std_dev_plot, drawstyle='steps-mid', color='r')
    ax[1].plot(time_domain_min, mean_plot, drawstyle='steps-mid', color='k')
    ax[1].set_title("Mean flare magnitude")

    ax[3].plot(time_domain_min, normed_mean + normed_std_dev, drawstyle='steps-mid', color='r')
    ax[3].plot(time_domain_min, normed_mean - normed_std_dev, drawstyle='steps-mid', color='r')
    ax[3].plot(time_domain_min, normed_mean, drawstyle='steps-mid', color='k')
    ax[3].set_title("Weighted, normalized flare mean")

    plt.show(block=True)
    print("cadence: ", flare_cat.cadence)
    eep.visualize.plot_signal_w_uncertainty(
        time_domain_min[echo_slice],
        normed_mean[echo_slice],
        (normed_mean + 2 * normed_std_dev)[echo_slice],
        (normed_mean - 2 * normed_std_dev)[echo_slice],
        x_axis_label="Time since flare (min)",
        y_axis_label="Relative, weighted normed mean intensity",
        y_label_1="Weighted mean of all flares",
        plt_title="Normed, weighted mean lag intensity for " + star_name,
        uncertainty_label="±2 SE of Mean",
        uncertainty_color='salmon')

    # todo - histogram along lag axis
    # Jackknife analysis:
    all_ind_list = np.arange(len(all_flares))
    all_jk_normed_means = np.zeros((len(all_flares), len(normed_mean)))
    all_jk_normed_stds = np.zeros((len(all_flares), len(normed_std_dev)))
    for ind in all_ind_list:
        jk_flare_inds = np.delete(all_ind_list, ind)
        jk_normed_flares, jk_normed_flare_weight, jk_normed_std_dev = \
            echo_analysis_do_it_all(all_flares[jk_flare_inds], back_pad + 1)
        jk_normed_mean = np.nansum(jk_normed_flares * jk_normed_flare_weight[:, np.newaxis], axis=0)
        all_jk_normed_means[ind] = jk_normed_mean
        all_jk_normed_stds[ind] = jk_normed_std_dev

    jk_mean = np.mean(all_jk_normed_means, axis=0)
    jk_mean_std = np.std(all_jk_normed_means, axis=0)
    mean_offset_jk = all_jk_normed_means - jk_mean
    sigma_dev = np.abs(mean_offset_jk / jk_mean_std)
    outlier_mask = sigma_dev > jk_std_dev_thresh
    flares_with_outliers = all_ind_list[np.any(outlier_mask, axis=1)]
    print("flares_with_outliers: ", flares_with_outliers)
    for outlier_ind in flares_with_outliers:
        plt.plot(time_domain_min, all_flares[outlier_ind], color='k', drawstyle='steps-mid', zorder=1)
        plt.scatter(time_domain_min[outlier_mask[outlier_ind]], all_flares[outlier_ind][outlier_mask[outlier_ind]],
                    zorder=0, color='r', marker='x')
        plt.title("Outlier(s) detected in flare: " + str(outlier_ind))
        plt.show(block=True)

    f, ax = plt.subplots(nrows=3, figsize=(16, 10))
    ax[0].plot(time_domain_min[echo_slice], (normed_mean + normed_std_dev)[echo_slice], drawstyle='steps-mid',
               color='r')
    ax[0].plot(time_domain_min[echo_slice], (normed_mean - normed_std_dev)[echo_slice], drawstyle='steps-mid',
               color='r')
    ax[0].plot(time_domain_min[echo_slice], normed_mean[echo_slice], drawstyle='steps-mid', color='k')
    ax[0].set_title("Weighted, normalized flare mean from " + star_name)

    for ind in all_ind_list:
        ax[1].scatter(time_domain_min[echo_slice], (all_jk_normed_means[ind] + all_jk_normed_stds[ind])[echo_slice],
                      marker='^', color='r')
        ax[1].scatter(time_domain_min[echo_slice], (all_jk_normed_means[ind] - all_jk_normed_stds[ind])[echo_slice],
                      marker='v', color='r')
        ax[1].scatter(time_domain_min[echo_slice], all_jk_normed_means[ind][echo_slice], marker="_", color='k')
    ax[1].set_title("Jackknifed-weighted, normalized flare means")

    ax[2].fill_between(time_domain_min[echo_slice],
                       np.max(all_jk_normed_means + all_jk_normed_stds, axis=0)[echo_slice],
                       np.min(all_jk_normed_means - all_jk_normed_stds, axis=0)[echo_slice],
                       facecolor='lightsalmon', label='Range of Std Devs', zorder=0, step='mid')
    ax[2].plot(time_domain_min[echo_slice], np.max((all_jk_normed_means + all_jk_normed_stds), axis=0)[echo_slice],
               drawstyle='steps-mid', color='salmon', zorder=1)
    ax[2].plot(time_domain_min[echo_slice], np.min((all_jk_normed_means - all_jk_normed_stds), axis=0)[echo_slice],
               drawstyle='steps-mid', color='salmon', zorder=2)

    ax[2].fill_between(time_domain_min[echo_slice],
                       np.max(all_jk_normed_means, axis=0)[echo_slice],
                       np.min(all_jk_normed_means, axis=0)[echo_slice],
                       facecolor='lightgray', label='Range of Std Devs', zorder=3, step='mid')
    ax[2].plot(time_domain_min[echo_slice], np.max(all_jk_normed_means, axis=0)[echo_slice],
               drawstyle='steps-mid', color='k', zorder=5)
    ax[2].plot(time_domain_min[echo_slice], np.min(all_jk_normed_means, axis=0)[echo_slice],
               drawstyle='steps-mid', color='k', zorder=6)
    ax[2].set_title("Max and min ranges from JK resample")
    plt.tight_layout()
    plt.show(block=True)

    outlier_culled_flare_list = np.delete(all_ind_list, flares_with_outliers)
    final_flare_list = all_flares[outlier_culled_flare_list]
    print("Keeping", len(final_flare_list), "of", len(all_flares), "flares after jackknife outlier removal")

    final_normed_flares, final_normed_flare_weight, final_normed_std_dev = \
        echo_analysis_do_it_all(final_flare_list, back_pad + 1)

    final_normed_mean = np.nansum(final_normed_flares * final_normed_flare_weight[:, np.newaxis], axis=0)

    # Build alternative threshold by bootstrap confidence intervals
    all_boot_samples = np.zeros((num_bootstrap, len(final_normed_flares), len(final_normed_flares[0])))
    with NumpyRNGContext(random_seed):
        num_samples = len(final_flare_list)
        for b_i in range(num_bootstrap):
            new_indices = np.random.choice(num_samples, num_samples, replace=True)
            all_boot_samples[b_i] = np.nansum(final_normed_flares[new_indices]
                                              * final_normed_flare_weight[new_indices][:, np.newaxis], axis=0)
    boot_mean = np.mean(all_boot_samples, axis=(0, 1))
    boot_conf = np.percentile(all_boot_samples, q=(conf_min, conf_max), axis=(0, 1))
    print("boot_conf.shape: ", boot_conf.shape)

    f, ax = plt.subplots(figsize=(12, 5))
    eep.visualize.plot_signal_w_uncertainty(
        time_domain_min[echo_slice],
        final_normed_mean[echo_slice],
        (final_normed_mean + 2 * final_normed_std_dev)[echo_slice],
        (final_normed_mean - 2 * final_normed_std_dev)[echo_slice],
        y_data_2=boot_mean[echo_slice],
        x_axis_label="Time since flare (min)",
        y_axis_label="Relative, weighted normed mean intensity",
        y_label_1="Weighted mean of inlier flares",
        y_label_2="Mean of bootstrap-resampled values",
        plt_title="Flare-peak normalized, variance-weighted mean lag intensity w/o outliers for " + star_name,
        uncertainty_label="±2sigma of mean",
        uncertainty_color='salmon',
        save='hold',
        axes_object=ax)
    ax.plot((time_domain_min[back_pad + 1], time_domain_min[-1]), (0, 0),
            color='gray', ls='--', zorder=30)
    ax.plot(time_domain_min[echo_slice], boot_conf[0][echo_slice],
            color='darkred', drawstyle='steps-post', zorder=0, lw=1,
            label='Boostrap ' + str(conf_range) + "% conf. range")
    ax.plot(time_domain_min[echo_slice], boot_conf[1][echo_slice],
            color='darkred', drawstyle='steps-post', zorder=0, lw=1)
    ax.fill_between(time_domain_min[echo_slice],
                    boot_conf[0][echo_slice],
                    boot_conf[1][echo_slice],
                    facecolor='indianred', zorder=-1, step='post')
    ax.legend().set_zorder(100)
    plt.show(block=True)

    # Some summary statistics:
    flare_tail_array = final_normed_flares[:, back_pad + 1:]
    flare_weights_array = np.ones_like(flare_tail_array) * final_normed_flare_weight[:, np.newaxis]
    generate_kde_plots(flare_tail_array, flare_weights_array,
                       "Normalized inlier post-flare lightcurve distribution")
    raw_flare_tail_array = final_flare_list[:, back_pad + 1:]
    generate_kde_plots(raw_flare_tail_array, np.ones_like(raw_flare_tail_array),
                       "Raw (unnormalized) inlier post-flare lightcurve distribution")
