import os
import numpy as np
from pathlib import Path
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import exoechopy as eep
from astropy.utils import NumpyRNGContext
from scipy.ndimage import morphology

#  ------------------------------------------------------------------------------------------------------------------  #
#  User inputs:

# Where we are getting raw data from:
file_folder = "11-19-2020 flux"
# How far before the flare to plot when producing the overview data product:
back_pad = 5
# How far after the flare to include in the analysis:
forward_pad = 80
# Threshold for detecting flares:
peak_find_std_dev_thresh = 6
# Threshold for rejecting flare based on a jackknife resample:
jk_std_dev_thresh = 4
# Number of bootstrap resamples for developing confidence intervals
num_bootstrap = 10000
# Range to use for confidence intervals
conf_range = 97.7

# Number of flare re-injection tests to perform to develop false-hit-rate values:
num_false_positive_tests = 5
# Threshold for microflares, used to mask them out for re-injection studies:
microflare_find_std_dev_thresh = 4.5
# pre and post-flare indices for flare injection, only weakly used currently:
pre_flare_brightening = 1
post_flare_decay = 1

# Seed to make results reproducible:
random_seed = 0

#  ------------------------------------------------------------------------------------------------------------------  #
#  Automated section

np.random.seed(random_seed)

# Where we store results:
output_folder = file_folder + "_results"

cwd = Path(os.getcwd())
fp = cwd.parent.parent / "disks" / file_folder
results_fp = cwd.parent.parent / "disks" / output_folder

if not results_fp.exists():
    os.mkdir(results_fp)

all_files = os.listdir(fp)
conf_min = (100 - conf_range) / 2
conf_max = 100 - conf_min

# Skip the flare itself when just looking at echo:
echo_slice = slice(back_pad + 1, None)

# This is just to import the data as provided in the email, this should be rewritten for the server version:
#  +++++++++++++++++++++++++++++++++++++++++++  #
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


#  +++++++++++++++++++++++++++++++++++++++++++  #


#  ------------------------------------------------------------------------------------------------------------------  #
#  These functions should be moved to another utility package

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
    normal_approx = eep.utils.gaussian(xvals, np.nanmean(flare_tail_histo), np.nanstd(flare_tail_histo))

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


def find_nonflaring_regions(lightcurve, known_flares, forwardpad, backpad=2, dilation_iter=1):
    """Identify non-flaring regions for false-alarm testing

    If you plan to search for echoes with 6-sigma flares, say,
     then you may want to use a list of 4-sigma flares to seed this function
     to avoid picking up minor, though still significant, flares

    Parameters
    ----------
    lightcurve
        Flux value array
    known_flares
        Indices of known flares, these will be masked out
    forwardpad
        Indices before known flares to mask out
    backpad
        Indices after known flares to mask out
    dilation_iter
        Number of dilation iterations to perform on the nan's from the base lightcurve

    Returns
    -------
    indices of nonflaring regions
    """
    # Rejected regions are areas where we do not allow a flare to be injected
    # Reject nans:
    rejected_regions = np.isnan(lightcurve)
    rejected_regions = morphology.binary_dilation(rejected_regions, iterations=dilation_iter)
    # Reject regions already covered by our lightcurve:
    for flare in known_flares:
        # Make sure existing flaring regions will not end up in the tails, either:
        start_ind = max(0, flare - backpad - forwardpad)
        end_ind = min(flare + forwardpad + backpad, len(lightcurve))
        rejected_regions[start_ind:end_ind] = True
    return np.nonzero(~rejected_regions)[0]


def select_flare_injection_sites(lightcurve, candidate_indices,
                                 num_flares, forwardpad, backpad=2, num_reselections=1,
                                 trial_timeout=100):
    """

    Parameters
    ----------
    lightcurve
        Original lightcurve
    candidate_indices
        List of available indices to select from
    num_flares
        Number of flares to select
    forwardpad
        Number of indices after a flare to mask out
    backpad
        Number of indices before a flare to mask out
    num_reselections
        Number of collections to produce (does not currently check if collections are different from each other!)
    trial_timeout
        If no collection of flares is found after this many iterations, break
        This can happen if, for example, there are 40 available indices, but they are all adjacent so
        the forwardpad term will mask them all from a single flare selection.
        Typically failure will be more subtle than that, so we just use a timeout.

    TODO - update to allow comparison of nans between original flare collection and remix collection

    Returns
    -------
    List of np.ndarray of selected indices that can be used for flare injection studies
    """
    injection_remixes = []
    if len(candidate_indices) < num_flares:
        raise AttributeError("More flares requested than available indices to select from")
    for _i in range(num_reselections):
        reset_ct = 0
        while reset_ct < trial_timeout:
            available_lc = np.zeros_like(lightcurve, dtype=bool)
            available_lc[candidate_indices] = True
            remaining_candidates = np.copy(candidate_indices)
            index_selections = []
            success = True
            for f_i in range(num_flares):
                # If no remaining locations, try again:
                if not np.any(available_lc):
                    success = False
                    break
                new_index = np.random.choice(remaining_candidates)
                index_selections.append(new_index)
                start_ind = max(0, new_index - backpad)
                end_ind = min(new_index + forwardpad, len(lightcurve))
                available_lc[start_ind:end_ind] = False
                remaining_candidates = np.nonzero(available_lc)[0]
            if not success:
                reset_ct += 1
                continue
            else:
                break
        if reset_ct >= trial_timeout:
            raise ValueError("Unable to identify a solution")
        injection_remixes.append(np.array(index_selections))
    return injection_remixes


#  ------------------------------------------------------------------------------------------------------------------  #
#  Actual calculations


# Loop through all stars:
for star_name in star_data_dict:
    # This is just to import the data as provided in the email, this should be rewritten for the server version:
    #  +++++++++++++++++++++++++++++++++++++++++++  #
    flux = star_data_dict[star_name]['flux']
    time = star_data_dict[star_name]['time']
    #  +++++++++++++++++++++++++++++++++++++++++++  #

    save_fp = results_fp / star_name

    if not save_fp.exists():
        os.mkdir(save_fp)

    f, ax = plt.subplots(figsize=(12, 4))
    plt.plot(time, flux, color='k', drawstyle='steps-mid')
    plt.title("Flux from " + star_name)
    plt.xlabel("Time (d)")
    plt.ylabel("Background-normalized flux")
    plt.savefig((save_fp / "lc_overview.png"))
    plt.close()

    base_flare_cat = eep.analyze.LightcurveFlareCatalog(flux,
                                                        extract_range=(back_pad, forward_pad),
                                                        time_domain=time)
    base_flare_cat.identify_flares_with_protocol(eep.analyze.find_peaks_stddev_thresh,
                                                 std_dev_threshold=peak_find_std_dev_thresh,
                                                 single_flare_gap=forward_pad)
    print("Number of flares identified: ", base_flare_cat.num_flares)
    flare_indices = base_flare_cat.get_flare_indices()
    time_domain_min = base_flare_cat.get_relative_indices() * base_flare_cat.cadence * 24 * 60

    # More inclusive list of flares for later injection tests:
    list_of_microflares = eep.analyze.find_peaks_stddev_thresh(flux, microflare_find_std_dev_thresh,
                                                               single_flare_gap=forward_pad)
    nonflaring_indices = find_nonflaring_regions(flux, list_of_microflares, forwardpad=forward_pad, backpad=back_pad)

    # False-positive test list:
    test_nonflare = select_flare_injection_sites(flux, nonflaring_indices, base_flare_cat.num_flares,
                                                 forwardpad=forward_pad, num_reselections=num_false_positive_tests)

    # for test_region_list in test_nonflare:
    #     eep.visualize.plot_flare_array(flux, test_region_list,
    #                                    back_pad=back_pad, forward_pad=forward_pad,
    #                                    title="Nonflaring test")

    eep.visualize.plot_flare_array(flux, flare_indices,
                                   back_pad=back_pad, forward_pad=forward_pad,
                                   title="Flare catalog from " + star_name + ", "
                                         + str(base_flare_cat.num_flares) + " flares found for peak>"
                                         + str(peak_find_std_dev_thresh) + "σ",
                                   savefile=save_fp / "flare_overview.png")

    all_flares = base_flare_cat.get_flare_curves()

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

    # plt.show(block=True)
    plt.savefig((save_fp / "flare_stats_overview.png"))
    plt.close()

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
        uncertainty_color='salmon',
        save=save_fp / "naive_flare_analysis_overview.png")

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

    rejected_outlier_fp = save_fp / "outliers"
    if not rejected_outlier_fp.exists():
        os.mkdir(rejected_outlier_fp)
    for outlier_ind in flares_with_outliers:
        plt.plot(time_domain_min, all_flares[outlier_ind], color='k', drawstyle='steps-mid', zorder=1)
        plt.scatter(time_domain_min[outlier_mask[outlier_ind]], all_flares[outlier_ind][outlier_mask[outlier_ind]],
                    zorder=0, color='r', marker='x')
        plt.title("Outlier(s) detected in flare " + str(outlier_ind))
        plt.savefig((rejected_outlier_fp / ("outliers_in_flare_" + str(outlier_ind) + ".png")))
        plt.close()

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
    plt.savefig((save_fp / "jk_stats_overview.png"))
    plt.close()

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
    plt.savefig((save_fp / "echo_detection_overview.png"))
    plt.close()

    # Some summary statistics:
    flare_tail_array = final_normed_flares[:, back_pad + 1:]
    flare_weights_array = np.ones_like(flare_tail_array) * final_normed_flare_weight[:, np.newaxis]
    generate_kde_plots(flare_tail_array, flare_weights_array,
                       "Normalized inlier post-flare lightcurve distribution",
                       savefile=save_fp / "post-flare_echo-normed_histogram.png")
    raw_flare_tail_array = final_flare_list[:, back_pad + 1:]
    generate_kde_plots(raw_flare_tail_array, np.ones_like(raw_flare_tail_array),
                       "Raw (unnormalized) inlier post-flare lightcurve distribution",
                       savefile=save_fp / "post-flare_raw_histogram.png")

print("All done!")
