import os
import numpy as np
from pathlib import Path
import matplotlib
import exoechopy as eep
from astropy.utils import NumpyRNGContext
from scipy.ndimage import morphology
from astropy.stats import jackknife_stats
import lightkurve as lk
import pickle

matplotlib.use('Agg')
from matplotlib import pyplot as plt

#  ------------------------------------------------------------------------------------------------------------------  #
#  User inputs:

# TODO - We are routinely seeing echo values below 0 at a significant ratio,
#  rebuild the noise profile to correctly capture (asymmetric?) these values in ±2SE bins.
#  Use the global KDE to resample for the echo significance testing?
#  Use global KDE to build the "missed" distribution, show whether it overlaps and if so, how much?
#  What about lag-resampling?  If we assume an active tail, then lag resampling should flatten that out...

# TODO - figure out which summary values we want to extract from this effort (frequency of stat sig. values, etc.),
#  generate the global summary plots and ROC curves.
#  E.g., are certain bins more likely to generate false values?
#   (They shouldn't be in this particular analysis, but worth including in case the analysis changes)
#  Rerun with larger numbers of false flare resamples to come up with good guidance.
#  Also help build a multi-star composite for Ben.

# Where we are getting raw data from:
cache_folder = 'local_cache_sc'
# What directory are we working from:
file_folder = 'SC_disk_echo_search'

# List of all targets to interrogate:
all_stars_filename = 'top1k_short_cadence.txt'
good_stars_filename = 'top_stars_with_flares.txt'
stat_meaningful_stars_filename = 'stars_with_significant_echoes.txt'
complete_stars_filename = 'completed_stars.txt'

clean_delta_flare_stars_filename = 'clean_delta_flare_stars_with_many_flares.txt'
clean_resolved_flare_stars_filename = 'clean_resolved_flare_stars_with_many_flares.txt'
clean_mixed_flare_stars_filename = 'clean_mixed_flare_stars_with_many_flares.txt'
# Some stars may require special handling:
skip_stars_filename = 'skip_stars.txt'

max_targets = None

# How far before the flare to plot when producing the overview data product:
pre_peak_pad = 5
# How far after the flare to include in the analysis:
post_peak_pad = 70
# Number of indices after a flare to skip for analysis, anticipating systematic decay errors:
post_flare_mask = 1  # Typically 0 or 1 for long cadence data
post_flare_mask_resample = 3  # Special case for the jk/leave-2-out error analysis
# Used to test whether a flare was resolved or not.  If resolved, it will be at least ±flare_width_test bins wide:
# flare_width_test = 2
flare_heuristic_thresh = .5
# Flare decay thresholds for resolved flares:
flare_decay_intensity_thresh_0 = 0.35
flare_decay_intensity_thresh_1 = 0.125
flare_decay_intensity_thresh_2 = 0.045
# Threshold for detecting flares:
peak_find_std_dev_thresh = 7
# Threshold for rejecting flare based on a jackknife resample:
jk_std_dev_thresh = 4.
l2o_std_dev_thresh = 5.
# Threshold for rejecting a flare based on high intensities in the echo domain:
relative_flare_intensity_thresh = 0.4
# Number of bootstrap resamples for developing confidence intervals
num_bootstrap = 10000
# Range to use for confidence intervals
conf_range = 99.7
# Number of flares to warrant further investigation:
min_flares = 10
# Periodic power rejection for high-amplitude variable stars
periodogram_power_thresh = 0.008
# Autocorrelation index to walk past before implementing rejection threshold:
autocorr_ind_thresh = 8
# Autocorrelation rejection threshold:
autocorr_reject_thresh = 0.1

generate_figures = True
skip_error_analysis = True

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

# Whether or not to overwrite previously cached results dictionaries
cache_dict_overwrite = False

#  ------------------------------------------------------------------------------------------------------------------  #
#  Automated section

cwd = Path(os.getcwd())

# Where we store results:
fp = cwd.parent.parent / "disks" / file_folder

if not fp.exists():
    fp.mkdir()

cache_fp = cwd.parent.parent / "disks" / cache_folder

if not cache_fp.exists():
    cache_fp.mkdir()

# Where we should save the results:
summary_filename = "_results_summary.txt"
summary_dict_filename = "_results_summary.dict"
# What should we call the dictionary that locates all the cache files:
cached_results_dict_name = "all_cached_stars.dict"

all_files = os.listdir(fp)
conf_min = (100 - conf_range) / 2
conf_max = 100 - conf_min

# Skip the flare itself when just looking at echo:
echo_slice = slice(pre_peak_pad + 1 + post_flare_mask, None)
echo_mask = np.zeros(pre_peak_pad + post_peak_pad, dtype=bool)
echo_mask[pre_peak_pad + 1 + post_flare_mask + post_flare_mask_resample:] = True
# print("echo_mask: ", echo_mask)

with open(fp / all_stars_filename, 'r') as tnames:
    raw_target_names = tnames.readlines()

if max_targets is None:
    max_targets = len(raw_target_names)
raw_target_names = raw_target_names[:max_targets]

target_names = [star_name.split('-')[0] for star_name in raw_target_names]
print(len(target_names), "total targets identified for analysis")
# # Don't run on entire database yet:
# target_names = target_names[:50]

try:
    with open(fp / good_stars_filename, 'r') as _f:
        pass
except FileNotFoundError:
    with open(fp / good_stars_filename, 'w') as _f:
        pass

try:
    with open(fp / clean_delta_flare_stars_filename, 'r') as _f:
        pass
except FileNotFoundError:
    with open(fp / clean_delta_flare_stars_filename, 'w') as _f:
        pass

try:
    with open(fp / clean_resolved_flare_stars_filename, 'r') as _f:
        pass
except FileNotFoundError:
    with open(fp / clean_resolved_flare_stars_filename, 'w') as _f:
        pass

try:
    with open(fp / clean_mixed_flare_stars_filename, 'r') as _f:
        pass
except FileNotFoundError:
    with open(fp / clean_mixed_flare_stars_filename, 'w') as _f:
        pass

try:
    with open(fp / stat_meaningful_stars_filename, 'r') as _f:
        pass
except FileNotFoundError:
    with open(fp / stat_meaningful_stars_filename, 'w') as _f:
        pass

target_len = len(target_names)
complete_star_list = []
try:
    with open(fp / complete_stars_filename, 'r') as _f:
        complete_star_list = _f.readlines()
except FileNotFoundError:
    with open(fp / complete_stars_filename, 'w') as _f:
        pass

skip_star_list = []
try:
    with open(fp / skip_stars_filename, 'r') as _f:
        skip_star_list = _f.readlines()
except FileNotFoundError:
    with open(fp / skip_stars_filename, 'w') as _f:
        pass

complete_star_list.extend(skip_star_list)

complete_star_list = [star_name.split('\n')[0] for star_name in complete_star_list]

target_names = [star_name for star_name in target_names if star_name not in complete_star_list]
remaining_target_len = len(target_names)
print("Of", target_len, "candidates,", remaining_target_len, "remain for analysis")

#  +++++++++++++++++++++++++++++++++++++++++++  #
try:
    all_cached_stars = pickle.load(open(cache_fp / cached_results_dict_name, 'rb'))
except FileNotFoundError:
    all_cached_stars = {}


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
    rejected_regions[0] = True
    rejected_regions[-1] = True
    # Pad the region around nans by dilation_iter:
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


def dict_key_gen(dict_for_gen: dict):
    """Converts a dictionary into a tuple to serve as a key for another dictionary"""
    all_keys = [k for k in dict_for_gen.keys()]
    all_keys.sort()
    key_gen = [(k, dict_for_gen[k]) for k in all_keys]
    return tuple(key_gen)


#  ------------------------------------------------------------------------------------------------------------------  #
#  Actual calculations

# Loop through all stars:
for star_name in target_names:
    print("\nStarting analysis of", star_name)

    np.random.seed(random_seed)

    # This is just to import the data as provided in the email, this should be rewritten for the server version:
    #  +++++++++++++++++++++++++++++++++++++++++++  #
    load_success = False
    if star_name in all_cached_stars:
        star_dict = all_cached_stars[star_name]
        try:
            time = np.load(star_dict['time_fp'])
            print("star_dict['time_fp']: ", star_dict['time_fp'])
            flux = np.load(star_dict['flux_fp'])
            flux_raw = np.load(star_dict['flux_raw_fp'])
            flux_norm = np.load(star_dict['flux_norm_fp'])
            flux_err = np.load(star_dict['flux_err_fp'])
            load_success = True
            print("Successfully loaded from cache")
        except:
            print("Unable to load from cache, re-downloading:")
    if not load_success:
        search = lk.search_lightcurvefile(star_name, cadence='short', mission='Kepler')
        if len(search) > 0:
            lc_collection = search.download_all()
            try:
                lc = lc_collection.PDCSAP_FLUX.stitch(corrector_func=lambda x: x.flatten())
                lc_norm = lc_collection.PDCSAP_FLUX.stitch(corrector_func=lambda x: x.normalize())
                lc_raw = lc_collection.PDCSAP_FLUX.stitch(corrector_func=None)
            except ValueError as err:
                print("Unable to generate lightcurves for", star_name)
                print("Error: ", err)
                with open(fp / skip_stars_filename, 'a') as _f:
                    _f.write(star_name + "\n")
                continue
            time = lc.time
            flux = lc.flux
            flux_raw = lc_raw.flux
            flux_norm = lc_norm.flux
            flux_err = lc.flux_err
            time_fp = cache_fp / (star_name + "_time.npy")
            flux_fp = cache_fp / (star_name + "_flux.npy")
            flux_raw_fp = cache_fp / (star_name + "_flux_raw.npy")
            flux_norm_fp = cache_fp / (star_name + "_flux_norm.npy")
            flux_err_fp = cache_fp / (star_name + "_flux_err.npy")
            np.save(time_fp, time)
            np.save(flux_fp, flux)
            np.save(flux_raw_fp, flux_raw)
            np.save(flux_norm_fp, flux_norm)
            np.save(flux_err_fp, flux_err)
            star_dict = {'time_fp': time_fp.as_posix(),
                         'flux_fp': flux_fp.as_posix(),
                         'flux_raw_fp': flux_raw_fp.as_posix(),
                         'flux_norm_fp': flux_norm_fp.as_posix(),
                         'flux_err_fp': flux_err_fp.as_posix()}
            all_cached_stars[star_name] = star_dict.copy()
            pickle.dump(all_cached_stars, open(cache_fp / cached_results_dict_name, 'wb'))
            load_success = True
        else:
            print("Unable to find star in lightkurve")

    #  +++++++++++++++++++++++++++++++++++++++++++  #

    if not load_success:
        print(star_name, "failed to load")
        continue
    save_fp = fp / star_name

    if not save_fp.exists():
        save_fp.mkdir()

    autocorr = eep.analyze.autocorrelate_array_w_nans(flux, post_peak_pad)
    lag_min = [(time[1] - time[0]) * 24 * 60 * t_i for t_i in range(len(autocorr))]

    # +++++++++++++++++++++++++++ #
    if generate_figures:
        f, ax = plt.subplots(nrows=3, figsize=(22, 12))
        ax[0].plot(time, flux_raw, color='k', drawstyle='steps-mid')
        ax[0].set_title("Raw PDCSAP flux from " + star_name)
        ax[0].set_xlabel("Time (d)")
        ax[0].set_ylabel("Raw PDCSAP flux")

        ax[1].plot(time, flux_norm, color='k', drawstyle='steps-mid')
        ax[1].set_title("Normalized PDCSAP flux from " + star_name)
        ax[1].set_xlabel("Time (d)")
        ax[1].set_ylabel("Raw PDCSAP flux")

        ax[2].plot(time, flux, color='k', drawstyle='steps-mid')
        ax[2].set_title("Flattened PDCSAP flux from " + star_name)
        ax[2].set_xlabel("Time (d)")
        ax[2].set_ylabel("Background-normalized flux")
        plt.tight_layout()
        plt.savefig((save_fp / "lc_overview.png"))
        plt.close()

        f, ax = plt.subplots(ncols=2, figsize=(10, 4))
        ax[0].plot(lag_min, autocorr, color='k', drawstyle='steps-mid')
        ax[0].set_title("Autocorrelation of entire lightcurve")
        ax[0].set_ylabel("Correlation (normed)")
        ax[0].set_xlabel("Time lag (minutes)")
        ax[1].plot(lag_min[1:], autocorr[1:], color='k', drawstyle='steps-mid')
        ax[1].set_title("Autocorrelation of entire lightcurve (w/o 0)")
        ax[1].set_ylabel("Correlation (normed)")
        ax[1].set_xlabel("Time lag (minutes)")
        plt.tight_layout()
        plt.savefig((save_fp / "lc_autocorr_overview.png"))
        plt.close()

    if np.any(autocorr[autocorr_ind_thresh:] > autocorr_reject_thresh):
        print("Star rejected for high-amplitude autocorrelation values: ", np.max(autocorr[autocorr_ind_thresh:]))
        with open(fp / complete_stars_filename, 'a') as _f:
            _f.write(star_name + "\n")
        continue

    # Test for ultra-large-amplitude variability:
    lc = lk.LightCurve(time, flux)
    lc = lc.remove_nans()
    pgram = lc.to_periodogram()
    print("pgram.max_power: ", pgram.max_power)
    # +++++++++++++++++++++++++++ #
    if generate_figures:
        pgram.plot()
        plt.savefig((save_fp / "ls_periodogram.png"))
        plt.close()
    if pgram.max_power > periodogram_power_thresh:
        print("Star rejected for high-amplitude periodicity: ", pgram.max_power)
        with open(fp / complete_stars_filename, 'a') as _f:
            _f.write(star_name + "\n")
        continue

    # Check for flares:
    base_flare_cat = eep.analyze.LightcurveFlareCatalog(flux,
                                                        extract_range=(pre_peak_pad, post_peak_pad),
                                                        time_domain=time)
    base_flare_cat.identify_flares_with_protocol(eep.analyze.find_peaks_stddev_thresh,
                                                 std_dev_threshold=peak_find_std_dev_thresh,
                                                 single_flare_gap=post_peak_pad)
    num_flares = base_flare_cat.num_flares
    print("Number of flares identified: ", num_flares)

    results_dict = {}

    with open(save_fp / (star_name + summary_filename), 'w') as file:
        input_dict = {}
        file.write("INPUTS\n")
        file.write("back_pad: " + str(pre_peak_pad) + "\n")
        input_dict['back_pad'] = pre_peak_pad
        file.write("forward_pad: " + str(post_peak_pad) + "\n")
        input_dict['forward_pad'] = post_peak_pad
        file.write("post_flare_mask: " + str(post_flare_mask) + "\n")
        input_dict['post_flare_mask'] = post_flare_mask
        file.write("peak_find_std_dev_thresh: " + str(peak_find_std_dev_thresh) + "\n")
        input_dict['peak_find_std_dev_thresh'] = peak_find_std_dev_thresh
        file.write("jk_std_dev_thresh: " + str(jk_std_dev_thresh) + "\n")
        input_dict['jk_std_dev_thresh'] = jk_std_dev_thresh
        file.write("l2o_std_dev_thresh: " + str(l2o_std_dev_thresh) + "\n")
        input_dict['l2o_std_dev_thresh'] = l2o_std_dev_thresh
        file.write("num_bootstrap: " + str(num_bootstrap) + "\n")
        input_dict['num_bootstrap'] = num_bootstrap
        file.write("conf_range: " + str(conf_range) + "\n")
        input_dict['conf_range'] = conf_range
        file.write("num_false_positive_tests: " + str(num_false_positive_tests) + "\n")
        input_dict['num_false_positive_tests'] = num_false_positive_tests
        file.write("microflare_find_std_dev_thresh: " + str(microflare_find_std_dev_thresh) + "\n")
        input_dict['microflare_find_std_dev_thresh'] = microflare_find_std_dev_thresh
        file.write("pre_flare_brightening: " + str(pre_flare_brightening) + "\n")
        input_dict['pre_flare_brightening'] = pre_flare_brightening
        file.write("post_flare_decay: " + str(post_flare_decay) + "\n")
        input_dict['post_flare_decay'] = post_flare_decay
        file.write("random_seed: " + str(random_seed) + "\n")
        input_dict['random_seed'] = random_seed
        results_dict['inputs'] = input_dict

        output_dict = {}
        file.write("\nOUTPUTS\n")
        file.write("Number of flares: " + str(num_flares) + "\n")
        output_dict['num_flares'] = num_flares
        file.write("Flare indices: " + str(base_flare_cat.get_flare_indices()) + "\n")
        output_dict['flare_indices'] = base_flare_cat.get_flare_indices()
        file.write("Periodogram peak power: " + str(pgram.max_power) + "\n")
        output_dict['pgram_power'] = pgram.max_power
        results_dict['outputs'] = output_dict

    base_flare_indices = base_flare_cat.get_flare_indices()
    time_domain_min = base_flare_cat.get_relative_indices() * base_flare_cat.cadence * 24 * 60

    flare_resolved_tests = []
    flare_peak_values = []
    for flare_ind in base_flare_indices:
        # start_ind = max(0, flare_ind - pre_peak_pad)
        # end_ind = min(len(flux), flare_ind + post_peak_pad)
        # sliced_flare = flux[start_ind:end_ind]
        # med_val = np.nanmedian(sliced_flare)
        # peakval = np.nanmax(sliced_flare)
        # neigh_start_ind = max(0, flare_ind - flare_width_test)
        # neigh_end_ind = min(len(flux), flare_ind + flare_width_test + 1)
        # neighbor_vals = np.nansum(flux[neigh_start_ind:neigh_end_ind]) - flux[flare_ind]
        # non_nan = np.sum(~np.isnan(flux[neigh_start_ind:neigh_end_ind]))
        # resolved_test = (neighbor_vals / (non_nan - 1)) / med_val

        f_0 = flux[flare_ind] - 1
        post_1_value = (flux[flare_ind + 1] - 1) / f_0 - flare_decay_intensity_thresh_0
        post_2_value = (flux[flare_ind + 2] - 1) / f_0 - flare_decay_intensity_thresh_1
        post_3_value = (flux[flare_ind + 3] - 1) / f_0 - flare_decay_intensity_thresh_2

        resolved_test = np.nansum([post_1_value, 2 * post_2_value, 3 * post_3_value])

        flare_resolved_tests.append(resolved_test)
        flare_peak_values.append(flux[flare_ind])

    flare_resolved_list = [True if f_h > flare_heuristic_thresh else False for f_h in flare_resolved_tests]

    # TODO - include:
    #  post_1_bool = normed_flares[ind][pre_peak_pad + 1] > flare_decay_intensity_thresh_0
    #  post_2_bool = normed_flares[ind][pre_peak_pad + 2] > flare_decay_intensity_thresh_1

    num_resolved = np.sum(flare_resolved_list)

    with open(save_fp / (star_name + summary_filename), 'a') as file:
        file.write("Number of resolved flares: " + str(num_resolved) + "\n")
        results_dict['outputs']['num_resolved'] = num_resolved
        file.write("Number of unresolved flares: " + str(num_flares - num_resolved) + "\n")
        results_dict['outputs']['num_unresolved'] = num_flares - num_resolved

    # +++++++++++++++++++++++++++ #
    if generate_figures:
        f, ax = plt.subplots(figsize=(5, 5))
        col_array = ['r' if f_h > flare_heuristic_thresh else 'k' for f_h in flare_resolved_tests]
        ax.scatter(flare_peak_values, flare_resolved_tests, c=col_array, marker='.')
        ax.set_xlabel("Flare peak value")
        ax.set_ylabel("Flare width heuristic")
        plt.title("Flare intensity vs width heuristic")
        plt.tight_layout()
        plt.savefig((save_fp / "flare_intensity_width_estimates.png"))
        plt.close()

    # More inclusive list of flares for later injection tests:
    list_of_microflares = eep.analyze.find_peaks_stddev_thresh(flux, microflare_find_std_dev_thresh,
                                                               single_flare_gap=post_peak_pad)
    nonflaring_indices, nonflaring_mask = find_nonflaring_regions(flux, list_of_microflares,
                                                                  forwardpad=post_peak_pad, backpad=pre_peak_pad,
                                                                  dilation_iter=pre_flare_brightening + post_flare_decay + 1)

    # Examine the global nonflaring flux distribution
    # +++++++++++++++++++++++++++ #
    if generate_figures:
        generate_kde_plots(flux[nonflaring_mask],
                           plot_title="Nonflaring lightcurve distribution",
                           savefile=save_fp / "nonflaring_raw_histogram.png",
                           num_plot_pts=1000, y_axis_loc=1., x_axis_loc=0)

    # Non-flaring autocorrelation because why not?
    nonflare_flux = flux.copy()
    nonflare_flux[~nonflaring_mask] = np.nan
    nonflare_autocorr = eep.analyze.autocorrelate_array_w_nans(nonflare_flux, post_peak_pad)
    lag_min = [(time[1] - time[0]) * 24 * 60 * t_i for t_i in range(len(nonflare_autocorr))]

    # +++++++++++++++++++++++++++ #
    if generate_figures:
        f, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
        ax[0, 0].plot(lag_min, nonflare_autocorr, color='k', drawstyle='steps-mid')
        ax[0, 0].set_title("Autocorrelation of non-flaring lightcurve")
        ax[0, 0].set_ylabel("Correlation (normed)")
        ax[0, 0].set_xlabel("Time lag (minutes)")

        ax[0, 1].plot(lag_min[1:], nonflare_autocorr[1:], color='k', drawstyle='steps-mid')
        ax[0, 1].set_title("Autocorrelation of non-flaring lightcurve (w/o 0)")
        ax[0, 1].set_ylabel("Correlation (normed)")
        ax[0, 1].set_xlabel("Time lag (minutes)")

        ax[1, 0].plot(lag_min, autocorr - nonflare_autocorr, color='k', drawstyle='steps-mid')
        ax[1, 0].set_title("Global AC minus non-flaring AC")
        ax[1, 0].set_ylabel("Correlation diff (normed)")
        ax[1, 0].set_xlabel("Time lag (minutes)")

        ax[1, 1].plot(lag_min[1:], nonflare_autocorr[1:], color='r', drawstyle='steps-mid',
                      label='Non-flaring autocorr')
        ax[1, 1].plot(lag_min[1:], autocorr[1:], color='k', drawstyle='steps-mid', label='Global autocorr')
        ax[1, 1].legend()
        ax[1, 1].set_title("Global AC and non-flaring AC (w/o 0)")
        ax[1, 1].set_ylabel("Correlation (normed)")
        ax[1, 1].set_xlabel("Time lag (minutes)")

        plt.tight_layout()
        plt.savefig((save_fp / "nonflare_lc_autocorr_overview.png"))
        plt.close()

    # False-positive test list:
    test_nonflare = select_flare_injection_sites(flux, nonflaring_indices, num_flares,
                                                 forwardpad=post_peak_pad, num_reselections=num_false_positive_tests)

    flare_ind_lists = [base_flare_indices]
    flare_ind_lists.extend(test_nonflare)

    flares_rejected_by_jk = []
    stat_meaningful_indices_list = []

    if num_flares < min_flares:
        print("Insufficient flares detected for analysis, moving to next star")
        with open(fp / complete_stars_filename, 'a') as _f:
            _f.write(star_name + "\n")
        continue
    else:
        with open(fp / good_stars_filename, 'a') as _f:
            _f.write(star_name + "\n")

    # Perform analysis and false-positive tests:
    for study_i, flare_indices in enumerate(flare_ind_lists):
        # Check if this is ground truth:
        if study_i % (len(flare_ind_lists) // 10) == 0:
            print(study_i, "of", len(flare_ind_lists))
        if study_i != 0:
            plot_flare_array_title = "Artificial flare catalog " + str(study_i)
            # Create a flux copy to work from
            fake_flux = np.copy(flux)
            # Inject true flares into fake positions:
            for true_flare_index, new_index in zip(base_flare_indices, flare_indices):
                try:
                    fake_flux[new_index - pre_flare_brightening:new_index + post_flare_decay + 1] = \
                        flux[true_flare_index - pre_flare_brightening:true_flare_index + post_flare_decay + 1]
                except ValueError as err:
                    print("Error: ", err)
                    print("Study: ", study_i)
                    print(new_index, true_flare_index, len(fake_flux), len(flux))
                    print(fake_flux[new_index - pre_flare_brightening:new_index + post_flare_decay + 1])
            new_flare_catalog = eep.analyze.LightcurveFlareCatalog(fake_flux,
                                                                   extract_range=(pre_peak_pad, post_peak_pad),
                                                                   time_domain=time)
            new_flare_catalog.manually_set_flare_indices(flare_indices)
            # Extract our newly faked lightcurves:
            all_flares = new_flare_catalog.get_flare_curves()
        else:
            plot_flare_array_title = "Flare catalog from " + star_name + ", " \
                                     + str(base_flare_cat.num_flares) + " flares found for peak>" \
                                     + str(peak_find_std_dev_thresh) + "σ"
            all_flares = base_flare_cat.get_flare_curves()

        normed_flares, normed_flare_weight, normed_std_dev = echo_analysis_do_it_all(all_flares, pre_peak_pad + 1)

        # Standard deviation of the mean, https://en.wikipedia.org/wiki/Standard_error#Standard_error_of_the_mean
        std_dev_plot = np.nanstd(all_flares, axis=0) / np.sqrt(len(all_flares))
        mean_plot = np.nanmean(all_flares, axis=0)
        normed_mean = np.nansum(normed_flares * normed_flare_weight[:, np.newaxis], axis=0)

        all_ind_list = np.arange(len(all_flares))

        # Intensity outliers:
        intensity_outlier_mask = np.zeros((len(normed_flares), len(normed_flares[0])), dtype=bool)
        for ind in all_ind_list:
            intensity_excursions = np.abs(normed_flares[ind]) > relative_flare_intensity_thresh
            intensity_outlier_mask[ind] = intensity_excursions & echo_mask

        flares_with_intensity_outliers = all_ind_list[np.any(intensity_outlier_mask, axis=1)]

        # Jackknife analysis:
        all_jk_normed_means = np.zeros((len(all_flares), len(normed_mean)))
        all_jk_normed_stds = np.zeros((len(all_flares), len(normed_std_dev)))
        for ind in all_ind_list:
            jk_flare_inds = np.delete(all_ind_list, ind)
            jk_normed_flares, jk_normed_flare_weight, jk_normed_std_dev = \
                echo_analysis_do_it_all(all_flares[jk_flare_inds], pre_peak_pad + 1)
            jk_normed_mean = np.nansum(jk_normed_flares * jk_normed_flare_weight[:, np.newaxis], axis=0)
            all_jk_normed_means[ind] = jk_normed_mean
            all_jk_normed_stds[ind] = jk_normed_std_dev

        jk_mean = np.mean(all_jk_normed_means, axis=0)
        jk_mean_std = np.std(all_jk_normed_means, axis=0)
        mean_offset_jk = all_jk_normed_means - jk_mean
        sigma_dev = np.abs(mean_offset_jk / jk_mean_std)
        jk_outlier_mask = sigma_dev > jk_std_dev_thresh
        flares_with_jk_outliers = all_ind_list[np.any(jk_outlier_mask & echo_mask, axis=1)]
        flares_rejected_by_jk.append(len(flares_with_jk_outliers))

        # Combine the jackknife and intensity outliers, reject them:
        jk_int_outliers = list(flares_with_jk_outliers)
        jk_int_outliers.extend(x for x in flares_with_intensity_outliers if x not in flares_with_jk_outliers)
        jk_int_outliers.sort()

        outlier_culled_flare_list = np.delete(all_ind_list, jk_int_outliers)
        jk_flare_list = all_flares[outlier_culled_flare_list]
        jk_flare_indices = np.array(flare_indices)[outlier_culled_flare_list]

        if len(outlier_culled_flare_list) < min_flares:
            print("Insufficient flares after initial outlier removal to continue analysis:",
                  len(outlier_culled_flare_list))
            if study_i == 0:
                break
            else:
                continue

        # Leave-2-out analysis:
        all_ind_list2 = outlier_culled_flare_list.copy()
        num_l2o_tests = (len(all_ind_list2) * (len(all_ind_list2) - 1))
        all_l2o_normed_means = np.zeros((num_l2o_tests, len(normed_mean)))
        all_l2o_normed_stds = np.zeros((num_l2o_tests, len(normed_std_dev)))
        l2o_lookup = {}
        ct = 0
        # print("all_ind_list: ", all_ind_list)
        for ind in all_ind_list2:
            jk_flare_inds = [x for x in all_ind_list2 if x != ind]
            # print("jk_flare_inds: ", jk_flare_inds)
            for ind2 in jk_flare_inds:
                l2o_flare_inds = np.array([x for x in jk_flare_inds if x != ind2])
                l2o_normed_flares, l2o_normed_flare_weight, l2o_normed_std_dev = \
                    echo_analysis_do_it_all(all_flares[l2o_flare_inds], pre_peak_pad + 1)
                l2o_normed_mean = np.nansum(l2o_normed_flares * l2o_normed_flare_weight[:, np.newaxis], axis=0)
                all_l2o_normed_means[ct] = l2o_normed_mean
                all_l2o_normed_stds[ct] = l2o_normed_std_dev
                l2o_lookup[ct] = (ind, ind2)
                ct += 1

        l2o_mean = np.mean(all_l2o_normed_means, axis=0)
        l2o_mean_std = np.std(all_l2o_normed_means, axis=0)
        mean_offset_l2o = all_l2o_normed_means - l2o_mean
        sigma_dev = np.abs(mean_offset_l2o / l2o_mean_std)
        l2o_outlier_mask = sigma_dev > l2o_std_dev_thresh
        l2o_with_outliers_ct = np.any(l2o_outlier_mask & echo_mask, axis=1)

        l2o_rejected_flares = set(jk_int_outliers)
        for ind, bool_ in enumerate(l2o_with_outliers_ct):
            if bool_:
                f1, f2 = l2o_lookup[ind]
                l2o_rejected_flares.add(f1)
                l2o_rejected_flares.add(f2)

        outlier_list = [x for x in l2o_rejected_flares]
        outlier_list.sort()
        set_array = np.array(outlier_list, dtype=int)
        if len(set_array) > 0:
            outlier_culled_flare_list = np.delete(all_ind_list, set_array)

        final_flare_list = all_flares[outlier_culled_flare_list]
        final_flare_indices = np.array(flare_indices)[outlier_culled_flare_list]
        # print("Keeping", len(final_flare_list), "of", len(all_flares), "flares after jackknife outlier removal")

        if len(final_flare_list) > min_flares & study_i == 0:
            if num_resolved / num_flares < 0.1:
                with open(fp / clean_delta_flare_stars_filename, 'a') as _f:
                    _f.write(star_name + "\n")
            elif num_resolved / num_flares > 0.9:
                with open(fp / clean_resolved_flare_stars_filename, 'a') as _f:
                    _f.write(star_name + "\n")
            else:
                with open(fp / clean_mixed_flare_stars_filename, 'a') as _f:
                    _f.write(star_name + "\n")

        final_normed_flares, final_normed_flare_weight, final_normed_std_dev = \
            echo_analysis_do_it_all(final_flare_list, pre_peak_pad + 1)
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
        # print("statistically_interesting_indices: ", statistically_interesting_indices)
        stat_meaningful_indices_list.append(len(statistically_interesting_indices))

        # Some summary statistics:
        flare_tail_array = final_normed_flares[:, pre_peak_pad + 1:]
        flare_weights_array = np.ones_like(flare_tail_array) * final_normed_flare_weight[:, np.newaxis]
        raw_flare_tail_array = final_flare_list[:, pre_peak_pad + 1:]

        # For the real-flare case:
        if study_i == 0:
            with open(save_fp / (star_name + summary_filename), 'a') as file:
                file.write("Number of outlier-rejected flares: " + str(len(outlier_list)) + "\n")
                results_dict['outputs']['num_outliers'] = len(outlier_list)
                file.write("Outlier-rejected flares: " + str(outlier_list) + "\n")
                results_dict['outputs']['outliers'] = outlier_list
                file.write(
                    "Number of statistically interesting peaks: " + str(len(statistically_interesting_indices)) + "\n")
                results_dict['outputs']['num_stat_meaningful_inds'] = len(statistically_interesting_indices)
                file.write("Statistically interesting indices: " + str(statistically_interesting_indices) + "\n")
                results_dict['outputs']['statistically_interesting_indices'] = statistically_interesting_indices

        #  ----------------------------------------------------------------------------------------------------------  #
        # Plot the results if it's meaningful, such as the first few or any others that detect 'echoes'
        if generate_figures and (study_i < num_reinjection_verbosity or len(statistically_interesting_indices) > 0):
            if study_i != 0:
                save_fp = fp / star_name / "flare_reinjection" / str(study_i)
                if not save_fp.exists():
                    save_fp.mkdir(parents=True, exist_ok=True)

            # +++++++++++++++++++++++++++ #
            eep.visualize.plot_flare_array(flux, flare_indices,
                                           back_pad=pre_peak_pad, forward_pad=post_peak_pad,
                                           title=plot_flare_array_title,
                                           savefile=save_fp / "flare_overview.png")

            # +++++++++++++++++++++++++++ #
            inlier_plot_flare_array_title = "Inlier flare catalog from " + star_name + ", " \
                                            + str(len(final_flare_indices)) + " flares found for peak>" \
                                            + str(peak_find_std_dev_thresh) + "σ"
            eep.visualize.plot_flare_array(flux, final_flare_indices,
                                           back_pad=pre_peak_pad, forward_pad=post_peak_pad,
                                           title=inlier_plot_flare_array_title,
                                           savefile=save_fp / "inlier_flare_overview.png")

            # +++++++++++++++++++++++++++ #
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

            # +++++++++++++++++++++++++++ #
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
                rejected_outlier_fp.mkdir()

            # +++++++++++++++++++++++++++ #
            for outlier_ind in flares_with_jk_outliers:
                plt.plot(time_domain_min, all_flares[outlier_ind], color='k', drawstyle='steps-mid', zorder=1)
                plt.scatter(time_domain_min[jk_outlier_mask[outlier_ind]],
                            all_flares[outlier_ind][jk_outlier_mask[outlier_ind]],
                            zorder=0, color='r', marker='x')
                plt.title("Jackknife outlier(s) detected in flare " + str(outlier_ind))
                plt.savefig((rejected_outlier_fp / ("jk_outliers_in_flare_" + str(outlier_ind) + ".png")))
                plt.close()

            # +++++++++++++++++++++++++++ #
            for outlier_ind in flares_with_intensity_outliers:
                plt.plot(time_domain_min, normed_flares[outlier_ind], color='k', drawstyle='steps-mid', zorder=1)
                plt.scatter(time_domain_min[intensity_outlier_mask[outlier_ind]],
                            normed_flares[outlier_ind][intensity_outlier_mask[outlier_ind]],
                            zorder=0, color='g', marker='x')
                plt.title("Intensity outlier(s) detected in flare " + str(outlier_ind))
                plt.savefig((rejected_outlier_fp / ("intensity_outliers_in_flare_" + str(outlier_ind) + ".png")))
                plt.close()

            # +++++++++++++++++++++++++++ #
            for ct, bool_ in enumerate(l2o_with_outliers_ct):
                if bool_:
                    f1, f2 = l2o_lookup[ct]
                    f, ax = plt.subplots(ncols=2, figsize=(10, 5))
                    ax[0].plot(time_domain_min, all_flares[f1], color='k', drawstyle='steps-mid', zorder=1)
                    ax[0].scatter(time_domain_min[l2o_outlier_mask[ct]],
                                  all_flares[f1][l2o_outlier_mask[ct]],
                                  zorder=0, color='r', marker='x')
                    ax[0].set_title("Flare " + str(f1))
                    ax[1].plot(time_domain_min, all_flares[f2], color='k', drawstyle='steps-mid', zorder=1)
                    ax[1].scatter(time_domain_min[l2o_outlier_mask[ct]],
                                  all_flares[f2][l2o_outlier_mask[ct]],
                                  zorder=0, color='r', marker='x')
                    ax[1].set_title("Flare " + str(f2))
                    plt.suptitle("Outlier(s) detected in flare " + str(f1) + " plus " + str(f2))
                    plt.savefig((rejected_outlier_fp / ("l2o_outliers_in_flares_" + str(f1) + "_" + str(f2) + ".png")))
                    plt.close()

            # +++++++++++++++++++++++++++ #
            f, ax = plt.subplots(figsize=(16, 4))
            for ind in all_ind_list:
                ax.scatter(time_domain_min[echo_slice],
                           (all_l2o_normed_means[ind] + all_l2o_normed_stds[ind])[echo_slice],
                           marker='^', color='r', zorder=0)
                ax.scatter(time_domain_min[echo_slice],
                           (all_l2o_normed_means[ind] - all_l2o_normed_stds[ind])[echo_slice],
                           marker='v', color='r', zorder=0)
                ax.scatter(time_domain_min[echo_slice], all_l2o_normed_means[ind][echo_slice], marker="_", color='k',
                           zorder=100)
            ax.set_title("Leave-2-out, weighted, normalized flare means")
            ax.set_ylabel("Leave-2-out value")
            ax.set_xlabel("Post-flare index")
            plt.tight_layout()
            plt.savefig((save_fp / "l2o_stats_overview.png"))
            plt.close()

            # +++++++++++++++++++++++++++ #
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
                              marker='^', color='r', zorder=0)
                ax[1].scatter(time_domain_min[echo_slice],
                              (all_jk_normed_means[ind] - all_jk_normed_stds[ind])[echo_slice],
                              marker='v', color='r', zorder=0)
                ax[1].scatter(time_domain_min[echo_slice], all_jk_normed_means[ind][echo_slice], marker="_", color='k',
                              zorder=10)
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

            # +++++++++++++++++++++++++++ #
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
                    label='Bootstrap ' + str(conf_range) + "% conf. range")
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

            # +++++++++++++++++++++++++++ #
            generate_kde_plots(flare_tail_array, flare_weights_array,
                               "Normalized inlier post-flare lightcurve distribution",
                               savefile=save_fp / "post-flare_echo-normed_histogram.png",
                               y_axis_loc=0, x_axis_loc=0)

            # +++++++++++++++++++++++++++ #
            generate_kde_plots(raw_flare_tail_array, np.ones_like(raw_flare_tail_array),
                               "Raw (unnormalized) inlier post-flare lightcurve distribution",
                               savefile=save_fp / "post-flare_raw_histogram.png",
                               y_axis_loc=1, x_axis_loc=0)

        # Check and see if we found anything, in which case, we'll do a false-positive test
        if study_i == 0 and len(statistically_interesting_indices) == 0:
            print("No statistically interesting indices, not performing error analysis")
            break
        elif study_i == 0:
            with open(fp / stat_meaningful_stars_filename, 'a') as _f:
                _f.write(star_name + "\n")
            if skip_error_analysis:
                print("skip_error_analysis == True")
                break

    # Reset filepath:
    save_fp = fp / star_name

    # Compile composite results
    if len(stat_meaningful_indices_list) > 1:
        baseline_jk_rejection = flares_rejected_by_jk.pop(0)
        baseline_num_meaningful = stat_meaningful_indices_list.pop(0)

        if generate_figures:
            f, ax = plt.subplots(ncols=3, figsize=(15, 5))

            ax[0].hist(flares_rejected_by_jk)
            ax[0].set_xlabel("Number of flares rejected by jackknife outlier")
            ax[0].set_title("Flare outlier rejection rate")
            ax[0].axvline(x=baseline_jk_rejection, color='r',
                          label="Number rejected from measurement")

            ax[1].hist(stat_meaningful_indices_list)
            ax[1].set_xlabel("Number of statistically meaningful indices detected")
            ax[1].set_title("Frequency of statistically meaningful flare detection")
            ax[1].axvline(x=baseline_num_meaningful, color='r',
                          label="Number of meaningful indices detected from measurement")

            ax[2].hist2d(flares_rejected_by_jk, stat_meaningful_indices_list,
                         bins=[max(flares_rejected_by_jk) + 1, max(stat_meaningful_indices_list) + 1],
                         label="Control")
            ax[2].scatter([baseline_jk_rejection], [baseline_num_meaningful], color='r',
                          label="Measurement")
            ax[2].set_xlabel("Number of flares rejected by JK outliers")
            ax[2].set_ylabel("Number of statistically interesting indices")
            ax[2].set_title("Is there a correlation between outlier rejection and echo detection?")

            plt.tight_layout()
            plt.savefig(save_fp / "outliers and echo detection stats.png")
            plt.close()

        total_false_positives = np.sum(stat_meaningful_indices_list)
        num_at_least_one_false_positive = np.sum(np.array(stat_meaningful_indices_list) > 0)

        with open(save_fp / (star_name + summary_filename), 'a') as file:
            file.write("Total false positives: " + str(total_false_positives) + "\n")
            results_dict['outputs']['total_false_positives'] = total_false_positives
            file.write("Number of tests with false positives: " + str(num_at_least_one_false_positive) + "\n")
            results_dict['outputs']['num_at_least_one_false_positive'] = num_at_least_one_false_positive
            file.write("False positive rate (total): " + str(total_false_positives / num_false_positive_tests) + "\n")
            results_dict['outputs']['false_positive_total_rate'] = total_false_positives / num_false_positive_tests
            file.write(
                "False positive rate (any): " + str(num_at_least_one_false_positive / num_false_positive_tests) + "\n")
            results_dict['outputs'][
                'false_positive_any_rate'] = num_at_least_one_false_positive / num_false_positive_tests

    try:
        full_dict = pickle.load(open(save_fp / (star_name + summary_dict_filename), 'rb'))
        if cache_dict_overwrite:
            full_dict = {}
    except FileNotFoundError:
        full_dict = {}
    dict_key = dict_key_gen(results_dict['inputs'])
    full_dict[dict_key] = results_dict['outputs']
    pickle.dump(full_dict, open(save_fp / (star_name + summary_dict_filename), 'wb'))

    with open(fp / complete_stars_filename, 'a') as _f:
        _f.write(star_name + "\n")

print("All done!")
