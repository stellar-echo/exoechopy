import os
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import exoechopy as eep

cwd = Path(os.getcwd())
fp = cwd.parent.parent / "disks" / "11-19-2020 flux"

all_files = os.listdir(fp)

star_data_dict = {}

for f in all_files:
    star_name = f.split('_')[0]
    if 'flux' in f:
        data_type = 'flux'
    elif 'time' in f:
        data_type = 'time'
    else:
        data_type = None
    if star_name not in star_data_dict:
        star_data_dict[star_name] = {}
    star_data_dict[star_name][data_type] = np.load(fp / f)

back_pad = 5
forward_pad = 80
std_dev_thresh = 4.5
jk_std_dev_thresh = 4

min_lag = 2
max_lag = 40

# Skip the flare itself when just looking at echo:
echo_slice = slice(back_pad + 1, None)


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


for star_title in star_data_dict:
    flux = star_data_dict[star_title]['flux']
    time = star_data_dict[star_title]['time']
    plt.plot(time, flux)
    plt.show(block=True)

    flare_cat = eep.analyze.LightcurveFlareCatalog(flux,
                                                   extract_range=(back_pad, forward_pad),
                                                   time_domain=time)
    flare_cat.identify_flares_with_protocol(eep.analyze.find_peaks_stddev_thresh,
                                            std_dev_threshold=std_dev_thresh,
                                            single_flare_gap=max_lag)
    print("Number of flares identified: ", flare_cat.num_flares)

    eep.visualize.plot_flare_array(flux, flare_cat.get_flare_indices(),
                                   back_pad=back_pad, forward_pad=forward_pad,
                                   title="Flare catalog from "+star_title)

    flare_cat.generate_correlator_matrix((eep.analyze.autocorrelate_array, {'min_lag': min_lag, 'max_lag': max_lag}))
    all_flares = flare_cat.get_flare_curves()

    time_domain_min = flare_cat.get_relative_indices() * flare_cat.cadence * 24 * 60

    normed_flares, normed_flare_weight, normed_std_dev = echo_analysis_do_it_all(all_flares, back_pad + 1)

    f, ax = plt.subplots(ncols=4, figsize=(16, 4))
    for f_i in range(len(all_flares)):
        ax[0].plot(time_domain_min, all_flares[f_i], drawstyle='steps-mid')
        ax[1].plot(time_domain_min, normed_flares[f_i], drawstyle='steps-mid')

    ax[0].set_title("All flares from " + star_title)
    ax[1].set_title("All peak-normalized flares")

    # Standard deviation of the mean, https://en.wikipedia.org/wiki/Standard_error#Standard_error_of_the_mean
    std_dev_plot = np.nanstd(all_flares, axis=0) / np.sqrt(len(all_flares))
    mean_plot = np.nanmean(all_flares, axis=0)
    normed_mean = np.nansum(normed_flares * normed_flare_weight[:, np.newaxis], axis=0)

    ax[2].plot(time_domain_min, mean_plot + std_dev_plot, drawstyle='steps-mid', color='r')
    ax[2].plot(time_domain_min, mean_plot - std_dev_plot, drawstyle='steps-mid', color='r')
    ax[2].plot(time_domain_min, mean_plot, drawstyle='steps-mid', color='k')

    ax[2].set_title("Mean flare magnitude")
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
        plt_title="Normed, weighted mean lag intensity for " + star_title,
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
    ax[0].set_title("Weighted, normalized flare mean from " + star_title)

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

    f, ax = plt.subplots(figsize=(12, 5))
    eep.visualize.plot_signal_w_uncertainty(
        time_domain_min[echo_slice],
        final_normed_mean[echo_slice],
        (final_normed_mean + 2 * final_normed_std_dev)[echo_slice],
        (final_normed_mean - 2 * final_normed_std_dev)[echo_slice],
        x_axis_label="Time since flare (min)",
        y_axis_label="Relative, weighted normed mean intensity",
        y_label_1="Weighted mean of inlier flares",
        plt_title="Flare-peak normalized, variance-weighted mean lag intensity w/o outliers for " + star_title,
        uncertainty_label="±2 SE of Mean",
        uncertainty_color='salmon',
        save='hold',
        axes_object=ax)
    ax.plot((time_domain_min[back_pad + 1], time_domain_min[-1]), (0, 0), color='gray', ls='--', zorder=30)
    plt.show(block=True)
