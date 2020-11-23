import os
import numpy as np
from pathlib import Path
import matplotlib
import exoechopy as eep
from astropy.utils import NumpyRNGContext
from scipy.ndimage import morphology
from astropy.stats import jackknife_stats

matplotlib.use('Agg')
from matplotlib import pyplot as plt

#  ------------------------------------------------------------------------------------------------------------------  #
#  User inputs:

# Where we are getting raw data from:
file_folder = "11-19-2020 flux"
# How far before the flare to plot when producing the overview data product:
back_pad = 5
# How far after the flare to include in the analysis:
forward_pad = 80
# Number of indices after a flare to skip for analysis, anticipating systematic decay errors:
post_flare_mask = 1  # Typically 0 or 1 for long cadence data
# Threshold for detecting flares:
peak_find_std_dev_thresh = 5
# Threshold for rejecting flare based on a jackknife resample:
jk_std_dev_thresh = 4
# Number of bootstrap resamples for developing confidence intervals
num_bootstrap = 10000
# Range to use for confidence intervals
conf_range = 97.7

# TODO - start to export measures as a function of input value for sensitivity and false-positive analysis

# Number of flare re-injection tests to perform to develop false-hit-rate values:
num_false_positive_tests = 50
# Threshold for microflares, used to mask them out for re-injection studies:
microflare_find_std_dev_thresh = 4.5
# pre and post-flare indices for flare injection, only weakly used currently:
pre_flare_brightening = 1
post_flare_decay = 1
# How many full datasets to export figures from for the resample analysis
num_reinjection_verbosity = 5

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
    results_fp.mkdir(results_fp)

all_files = os.listdir(fp)
conf_min = (100 - conf_range) / 2
conf_max = 100 - conf_min

# Skip the flare itself when just looking at echo:
echo_slice = slice(back_pad + 1 + post_flare_mask, None)

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


def generate_kde_plots(flare_array, weight_array=None,
                       plot_title=None, savefile=None, num_plot_pts=500,
                       x_axis_loc=None, y_axis_loc=None):
    flare_tail_histo = flare_array.flatten()
    all_nans = np.isnan(flare_tail_histo)
    flare_tail_histo = flare_tail_histo[~all_nans]
    if weight_array is not None:
        flare_tail_weights = weight_array.flatten()
        flare_tail_weights = flare_tail_weights[~all_nans]
    else:
        flare_tail_weights = None

    xvals = np.linspace(np.nanmin(flare_tail_histo), np.nanmax(flare_tail_histo), num_plot_pts)
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
    ax[0].plot(xvals, flare_tail_kde_gauss, color='k', zorder=1,
               label="Scott's rule Gaussian KDE")
    ax[0].plot(xvals, flare_tail_kde, color='gray', zorder=0, label='5-bin Tophat KDE')
    ax[0].plot(xvals, normal_approx, color='b', ls='--', zorder=-1, label='Gaussian fit to distribution')
    ax[0].set_xlabel("Values")
    ax[0].set_ylabel("Density")
    ax[0].set_title("Distribution of values")
    if y_axis_loc is not None:
        ax[0].axvline(x=y_axis_loc, color='k', lw=1)
    ax[0].legend()

    ax[1].plot(xvals, flare_tail_kde_gauss - normal_approx, color='k', zorder=1, label="Scott's rule KDE - normal")
    ax[1].plot(xvals, flare_tail_kde - normal_approx, color='gray', zorder=0, label="5-bin Tophat KDE - normal")
    ax[1].set_xlabel("Values")
    ax[1].set_ylabel("Difference from normal density")
    ax[1].set_title("Difference from normal distribution")
    if y_axis_loc is not None:
        ax[1].axvline(x=y_axis_loc, color='k', lw=1)
    if x_axis_loc is not None:
        ax[1].axhline(y=x_axis_loc, color='k', lw=1)
    ax[1].legend()

    # cumulative_distribution = np.cumsum(flare_tail_kde)
    # cumulative_distribution /= np.max(cumulative_distribution)
    # ax[1].plot(xvals, cumulative_distribution, color='k')
    # ax[1].set_xlabel("Values")
    # ax[1].set_ylabel("Cumulative frequency")
    # ax[1].set_title("Cumulative values")
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
    return np.nonzero(~rejected_regions)[0], ~rejected_regions


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


# TODO - We are routinely seeing echo values below 0 at a significant ratio,
#  rebuild the noise profile to correctly capture (asymmetric?) these values in ±2SE bins.
#  Use the global KDE to resample for the echo significance testing?
#  Use global KDE to build the "missed" distribution, show whether it overlaps and if so, how much?
#  What about lag-resampling?  If we assume an active tail, then lag resampling should flatten that out...

# Loop through all stars:
for star_name in star_data_dict:
    # This is just to import the data as provided in the email, this should be rewritten for the server version:
    #  +++++++++++++++++++++++++++++++++++++++++++  #
    flux = star_data_dict[star_name]['flux']
    time = star_data_dict[star_name]['time']
    #  +++++++++++++++++++++++++++++++++++++++++++  #

    save_fp = results_fp / star_name

    if not save_fp.exists():
        save_fp.mkdir(save_fp)

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
    num_flares = base_flare_cat.num_flares
    print("Number of flares identified: ", num_flares)
    base_flare_indices = base_flare_cat.get_flare_indices()
    time_domain_min = base_flare_cat.get_relative_indices() * base_flare_cat.cadence * 24 * 60

    # More inclusive list of flares for later injection tests:
    list_of_microflares = eep.analyze.find_peaks_stddev_thresh(flux, microflare_find_std_dev_thresh,
                                                               single_flare_gap=forward_pad)
    nonflaring_indices, nonflaring_mask = find_nonflaring_regions(flux, list_of_microflares,
                                                                  forwardpad=forward_pad, backpad=back_pad)

    # Examine the global nonflaring flux distribution
    generate_kde_plots(flux[nonflaring_mask],
                       plot_title="Nonflaring lightcurve distribution",
                       savefile=save_fp / "nonflaring_raw_histogram.png",
                       num_plot_pts=1000, y_axis_loc=1., x_axis_loc=0)

    # False-positive test list:
    test_nonflare = select_flare_injection_sites(flux, nonflaring_indices, num_flares,
                                                 forwardpad=forward_pad, num_reselections=num_false_positive_tests)

    # for test_region_list in test_nonflare:
    #     eep.visualize.plot_flare_array(flux, test_region_list,
    #                                    back_pad=back_pad, forward_pad=forward_pad,
    #                                    title="Nonflaring test")

    flare_ind_lists = [base_flare_indices]
    flare_ind_lists.extend(test_nonflare)

    # TODO - figure out which summary values we want to extract from this effort (frequency of stat sig. values, etc.),
    #  generate the global summary plots and ROC curves.
    #  E.g., are certain bins more likely to generate false values?
    #   (They shouldn't be in this particular analysis, but worth including in case the analysis changes)
    #  Rerun with larger numbers of false flare resamples to come up with good guidance.
    #  Also help build a multi-star composite for Ben.

    # TODO - To build ROC curve, we should to manually inject echoes at a given threshold to the non-flaring regions?

    flares_rejected_by_jk = []
    num_stat_meaningful_indices = []

    for study_i, flare_indices in enumerate(flare_ind_lists):
        # Check if this is ground truth:
        if study_i != 0:

            save_fp = results_fp / star_name / "flare_reinjection" / str(study_i)

            if not save_fp.exists():
                save_fp.mkdir(parents=True, exist_ok=True)
            plot_flare_array_title = "Artificial flare catalog " + str(study_i)
            # Create a flux copy to work from
            fake_flux = np.copy(flux)
            # Inject true flares into fake positions:
            for true_flare_index, new_index in zip(base_flare_indices, flare_indices):
                fake_flux[new_index - pre_flare_brightening:new_index + post_flare_decay + 1] = \
                    flux[true_flare_index - pre_flare_brightening:true_flare_index + post_flare_decay + 1]
            new_flare_catalog = eep.analyze.LightcurveFlareCatalog(fake_flux,
                                                                   extract_range=(back_pad, forward_pad),
                                                                   time_domain=time)
            new_flare_catalog.manually_set_flare_indices(flare_indices)
            # Extract our newly faked lightcurves:
            all_flares = new_flare_catalog.get_flare_curves()
        else:
            plot_flare_array_title = "Flare catalog from " + star_name + ", " \
                                     + str(base_flare_cat.num_flares) + " flares found for peak>" \
                                     + str(peak_find_std_dev_thresh) + "σ"
            all_flares = base_flare_cat.get_flare_curves()

        normed_flares, normed_flare_weight, normed_std_dev = echo_analysis_do_it_all(all_flares, back_pad + 1)

        # Standard deviation of the mean, https://en.wikipedia.org/wiki/Standard_error#Standard_error_of_the_mean
        std_dev_plot = np.nanstd(all_flares, axis=0) / np.sqrt(len(all_flares))
        mean_plot = np.nanmean(all_flares, axis=0)
        normed_mean = np.nansum(normed_flares * normed_flare_weight[:, np.newaxis], axis=0)

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
        flares_rejected_by_jk.append(len(flares_with_outliers))
        print("flares_with_outliers: ", flares_with_outliers)

        outlier_culled_flare_list = np.delete(all_ind_list, flares_with_outliers)
        final_flare_list = all_flares[outlier_culled_flare_list]
        print("Keeping", len(final_flare_list), "of", len(all_flares), "flares after jackknife outlier removal")

        final_normed_flares, final_normed_flare_weight, final_normed_std_dev = \
            echo_analysis_do_it_all(final_flare_list, back_pad + 1)
        final_unweighted_mean = np.nanmean(final_normed_flares, axis=0)

        weighted_final_flares = final_normed_flares * final_normed_flare_weight[:, np.newaxis]
        final_normed_mean = np.nansum(weighted_final_flares, axis=0)

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

        # Build alternative jackknife threshold confidence intervals
        jk_conf_low = []
        jk_conf_high = []
        jk_est = []
        for ind in range(len(final_normed_flares[0])):
            current_col = final_normed_flares[:, ind]
            est, bias, std_err, conf_interval = jackknife_stats(current_col,
                                                                np.nanmean,
                                                                confidence_level=conf_range / 100)
            jk_est.append(est)
            jk_conf_low.append(conf_interval[0])
            jk_conf_high.append(conf_interval[1])
        jk_conf_low = np.array(jk_conf_low)
        jk_conf_high = np.array(jk_conf_high)

        statistically_interesting_indices = \
            np.nonzero((boot_conf[0][echo_slice] > 0)
                       & ((final_normed_mean - 2 * final_normed_std_dev)[
                              echo_slice] > 0)
                       & (jk_conf_low[echo_slice] > 0))[0]
        print("statistically_interesting_indices: ", statistically_interesting_indices)
        num_stat_meaningful_indices.append(len(statistically_interesting_indices))

        # Some summary statistics:
        flare_tail_array = final_normed_flares[:, back_pad + 1:]
        flare_weights_array = np.ones_like(flare_tail_array) * final_normed_flare_weight[:, np.newaxis]
        raw_flare_tail_array = final_flare_list[:, back_pad + 1:]

        #  ----------------------------------------------------------------------------------------------------------  #
        # Plot all of the results:
        if study_i <= num_reinjection_verbosity:

            eep.visualize.plot_flare_array(flux, flare_indices,
                                           back_pad=back_pad, forward_pad=forward_pad,
                                           title=plot_flare_array_title,
                                           savefile=save_fp / "flare_overview.png")

            f, ax = plt.subplots(ncols=4, figsize=(16, 4))
            for f_i in range(len(all_flares)):
                ax[0].plot(time_domain_min, all_flares[f_i], drawstyle='steps-mid')
                ax[2].plot(time_domain_min, normed_flares[f_i], drawstyle='steps-mid')

            ax[0].set_title("All flares from " + star_name)
            ax[2].set_title("All peak-normalized flares")

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

            rejected_outlier_fp = save_fp / "outliers"
            if not rejected_outlier_fp.exists():
                os.mkdir(rejected_outlier_fp)

            for outlier_ind in flares_with_outliers:
                plt.plot(time_domain_min, all_flares[outlier_ind], color='k', drawstyle='steps-mid', zorder=1)
                plt.scatter(time_domain_min[outlier_mask[outlier_ind]],
                            all_flares[outlier_ind][outlier_mask[outlier_ind]],
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
                ax[1].scatter(time_domain_min[echo_slice],
                              (all_jk_normed_means[ind] + all_jk_normed_stds[ind])[echo_slice],
                              marker='^', color='r')
                ax[1].scatter(time_domain_min[echo_slice],
                              (all_jk_normed_means[ind] - all_jk_normed_stds[ind])[echo_slice],
                              marker='v', color='r')
                ax[1].scatter(time_domain_min[echo_slice], all_jk_normed_means[ind][echo_slice], marker="_", color='k')
            ax[1].set_title("Jackknifed-weighted, normalized flare means")

            ax[2].fill_between(time_domain_min[echo_slice],
                               np.max(all_jk_normed_means + all_jk_normed_stds, axis=0)[echo_slice],
                               np.min(all_jk_normed_means - all_jk_normed_stds, axis=0)[echo_slice],
                               facecolor='lightsalmon', label='Range of Std Devs', zorder=0, step='mid')
            ax[2].plot(time_domain_min[echo_slice],
                       np.max((all_jk_normed_means + all_jk_normed_stds), axis=0)[echo_slice],
                       drawstyle='steps-mid', color='salmon', zorder=1)
            ax[2].plot(time_domain_min[echo_slice],
                       np.min((all_jk_normed_means - all_jk_normed_stds), axis=0)[echo_slice],
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

            f, ax = plt.subplots(figsize=(12, 5))
            eep.visualize.plot_signal_w_uncertainty(
                time_domain_min[echo_slice],
                final_normed_mean[echo_slice],
                (final_normed_mean + 2 * final_normed_std_dev)[echo_slice],
                (final_normed_mean - 2 * final_normed_std_dev)[echo_slice],
                y_data_2=final_unweighted_mean[echo_slice],
                x_axis_label="Time since flare (min)",
                y_axis_label="Relative, weighted normed mean intensity",
                y_label_1="Weighted mean of inlier flares",
                y_label_2="Unweighted mean of inlier flares",
                plt_title="Flare-peak normalized, variance-weighted mean lag intensity w/o outliers for " + star_name,
                uncertainty_label="±2sigma of mean",
                uncertainty_color='salmon',
                save='hold',
                axes_object=ax)

            ax.axhline(y=0, color='gray', ls='--', zorder=30)
            ax.plot(time_domain_min[echo_slice], boot_conf[0][echo_slice],
                    color='darkred', drawstyle='steps-post', zorder=0, lw=1,
                    label='Boostrap ' + str(conf_range) + "% conf. range")
            ax.plot(time_domain_min[echo_slice], boot_conf[1][echo_slice],
                    color='darkred', drawstyle='steps-post', zorder=0, lw=1)

            ax.fill_between(time_domain_min[echo_slice],
                            boot_conf[0][echo_slice],
                            boot_conf[1][echo_slice],
                            facecolor='indianred', zorder=-1, step='post')

            # JK thresh test:
            # ax.plot(time_domain_min[echo_slice], jk_est[echo_slice], zorder=50, color='purple', drawstyle='steps-post')
            ax.plot(time_domain_min[echo_slice], jk_conf_low[echo_slice],
                    color='royalblue', drawstyle='steps-post', zorder=-2, lw=1, ls=':',
                    label='Unweighted JK ' + str(conf_range) + "% conf. range")
            ax.plot(time_domain_min[echo_slice], jk_conf_high[echo_slice],
                    color='royalblue', drawstyle='steps-post', zorder=-2, lw=1, ls=':')
            ax.fill_between(time_domain_min[echo_slice],
                            jk_conf_low[echo_slice],
                            jk_conf_high[echo_slice],
                            facecolor='lightsteelblue', zorder=-3, step='post')

            ax.legend().set_zorder(100)
            plt.savefig((save_fp / "echo_detection_overview.png"))
            plt.close()

            generate_kde_plots(flare_tail_array, flare_weights_array,
                               "Normalized inlier post-flare lightcurve distribution",
                               savefile=save_fp / "post-flare_echo-normed_histogram.png",
                               y_axis_loc=0, x_axis_loc=0)
            generate_kde_plots(raw_flare_tail_array, np.ones_like(raw_flare_tail_array),
                               "Raw (unnormalized) inlier post-flare lightcurve distribution",
                               savefile=save_fp / "post-flare_raw_histogram.png",
                               y_axis_loc=1, x_axis_loc=0)

    # Compile composite results

    # Reset filepath:
    save_fp = results_fp / star_name

    baseline_jk_rejection = flares_rejected_by_jk.pop(0)
    baseline_num_meaningful = num_stat_meaningful_indices.pop(0)

    f, ax = plt.subplots(ncols=3, figsize=(15, 5))

    ax[0].hist(flares_rejected_by_jk)
    ax[0].set_xlabel("Number of flares rejected by jackknife outlier")
    ax[0].set_title("Flare outlier rejection rate")
    ax[0].axvline(x=baseline_jk_rejection, color='r', label="Number rejected from measurement")

    ax[1].hist(num_stat_meaningful_indices)
    ax[1].set_xlabel("Number of statistically meaningful indices detected")
    ax[1].set_title("Frequency of statistically meaningful flare detection")
    ax[1].axvline(x=baseline_num_meaningful, color='r', label="Number of meaningful indices detected from measurement")

    ax[2].hist2d(flares_rejected_by_jk, num_stat_meaningful_indices,
                 bins=[max(flares_rejected_by_jk) + 1, max(num_stat_meaningful_indices) + 1],
                 label="Control")
    ax[2].scatter([baseline_jk_rejection], [baseline_num_meaningful], color='r', label="Measurement")
    ax[2].set_xlabel("Number of flares rejected by JK outliers")
    ax[2].set_ylabel("Number of statistically interesting indices")
    ax[2].set_title("Is there a correlation between outlier rejection and echo detection?")

    plt.tight_layout()
    plt.savefig(save_fp / "outliers and echo detection stats.png")
    plt.close()

print("All done!")
