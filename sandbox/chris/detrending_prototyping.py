import os
import exoechopy as eep
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
import lightkurve as lk
from scipy import signal

#  ----------------------------------------------------------------------------------------  #

current_path = Path(os.getcwd())
flare_std_dev_thresh = 1.25
pre_mask_dilation = 10
post_mask_dilation = 90
weight_mask_kernel_dim = 15

flare_separation = 180
back_pad = 20
echo_search_range = flare_separation - back_pad

fast_fit_poly_order = 3
fast_fit_std_dev_thresh = 1.25
fast_fit_window = 101
fast_fit_std_dev_clip = 10

poly_fit_order = 12
flatten_window = 2401
residual_clip_val = 4.5

# TODO - Move flattening routines into a class


weight_kernel = signal.windows.blackmanharris(weight_mask_kernel_dim)
weight_kernel /= np.sum(weight_kernel)

print("Flattening window length: ", flatten_window)
print("Weight kernel length: ", len(weight_kernel))


def dilate_mask(original_mask, pre_dilate, post_dilate):
    """For each True in original mask, turn the pre_dilate points before it and post_dilate points after it into True"""
    all_mask_inds = np.nonzero(original_mask)[0]
    new_mask = np.zeros_like(original_mask)
    max_ind = len(original_mask)
    for ind in all_mask_inds:
        start_ind = max(0, ind - pre_dilate)
        end_ind = min(max_ind, ind + post_dilate)
        new_mask[start_ind:end_ind] = True
    return new_mask


def parametric_mask_dilation(original_mask, pre_dilate_array, post_dilate_array):
    """For each True in original mask, turn the pre_dilate points before it and post_dilate points after it into True"""
    all_mask_inds = np.nonzero(original_mask)[0]
    new_mask = np.zeros(len(original_mask), dtype=bool)
    max_ind = len(original_mask)
    for ind in all_mask_inds:
        start_ind = max(0, ind - pre_dilate_array[ind])
        end_ind = min(max_ind, ind + post_dilate_array[ind])
        new_mask[start_ind:end_ind] = True
    return new_mask


def sg_with_weighting_brute_force(raw_data, window_size, poly_order, weight_mask, weight_tolerance=5):
    """Subtracts background by fitting a weighted polynomial at each point in the dataset

    To handle edge condition, it crops the beginning and end data points,
    so the output array will run from [window_size//2 : len(raw_data) - window_size // 2 + 1]

    Parameters
    ----------
    raw_data
        Data to be smoothed
    window_size
        Size of window for polynomial fitting
    poly_order
        Order of polynomial (x**n)
    weight_mask
        Weights of values in raw_data.  Can set nans to 0, for instance
    weight_tolerance
        If a given fit has inadequate weight_tolerance, set to nan.
        For example, if weight_tolerance = 5, then np.sum(weight_mask) must be at least 5 within that window_size region

    Returns
    -------
    (np.array of background-subtracted elements, np.array of interpolated background, slice of valid raw_data)
    TODO - also return any uncertainty in each value
    """
    # TODO - add local weight, as well, so futher points are deweighted (Tukey? Hann?)
    if window_size % 2 == 0:
        window_size += 1
    local_weight_map = signal.windows.tukey(window_size)
    ind_list = np.arange(window_size // 2, len(raw_data) - window_size // 2)
    output_result = np.zeros(len(raw_data) - window_size + 1)
    output_residual = np.zeros_like(output_result)
    x_vals = np.arange(window_size)
    array_slicer = slice(ind_list[0], ind_list[-1] + 1)
    for i_i, ind in enumerate(ind_list):
        data_slice = raw_data[ind - window_size // 2:ind + window_size // 2 + 1]
        weight_slice = weight_mask[ind - window_size // 2:ind + window_size // 2 + 1] * local_weight_map
        # if np.sum(np.isnan(data_slice)):
        # print(i_i, " nan", np.sum(weight_slice))
        # Make sure there are enough points to fit:
        if np.sum(weight_slice) < max(weight_tolerance, poly_order + 1):
            output_result[i_i] = 0
            output_residual[i_i] = np.nan
        else:
            polyfit, diagnostics = np.polynomial.polynomial.Polynomial.fit(x_vals, data_slice, poly_order,
                                                                           w=weight_slice, full=True)
            if len(diagnostics[0]) == 0:
                # print(diagnostics[0])
                pass
            #     output_result[i_i] = 0
            #     output_residual[i_i] = np.nan
            else:
                output_result[i_i] = polyfit(window_size // 2)
                output_residual[i_i] = diagnostics[0]  # residual, rank, sing_vals, cond_number
                output_residual[i_i] **= .5
                output_residual[i_i] /= np.sum(weight_slice)
    return raw_data[array_slicer] - output_result, output_result, array_slicer, output_residual


def linear_interp(val_arr, y0, y1, x0, x1):
    return (y0 * (x1 - val_arr) + y1 * (val_arr - x0)) / (x1 - x0)


def prep_sigma_flare_mask(raw_data: np.array,
                          sigma_min_thresh, sigma_max_thresh,
                          max_pre_pad, max_post_pad,
                          min_pre_pad=7, min_post_pad=10, nan_mask=None):
    sigma_prep_arr = raw_data - np.nanmean(raw_data)
    sigma_dev_arr = (sigma_prep_arr / np.nanstd(sigma_prep_arr))
    sigma_dev_arr[np.isnan(sigma_dev_arr)] = 0

    # If sigmas are below threshold, clip to zero to better ignore:
    sub_thresh_sigma_mask = (sigma_dev_arr < sigma_min_thresh) & (sigma_dev_arr > -sigma_min_thresh)
    sigma_dev_arr[sub_thresh_sigma_mask] = 0
    # If sigmas are above max threshold, clip to max thresh:
    sigma_dev_arr[sigma_dev_arr > sigma_max_thresh] = sigma_max_thresh
    # Set nan's to min threshold (for now)

    if nan_mask is not None:
        sigma_dev_arr[nan_mask] = sigma_min_thresh

    sigma_excursions = sigma_dev_arr

    pre_pad_arr = linear_interp(sigma_dev_arr,
                                min_pre_pad, max_pre_pad,
                                sigma_min_thresh, sigma_max_thresh).astype(int)
    post_pad_arr = linear_interp(sigma_dev_arr,
                                 min_post_pad, max_post_pad,
                                 sigma_min_thresh, sigma_max_thresh).astype(int)
    # Negative excursions are also bad for fitting, but aren't flares, so they don't get a long tail:
    negative_excursions = sigma_dev_arr <= -sigma_min_thresh
    pre_pad_arr[negative_excursions] = min_pre_pad
    post_pad_arr[negative_excursions] = min_pre_pad
    return ~sub_thresh_sigma_mask, pre_pad_arr, post_pad_arr, sigma_excursions


test_data = lk.search_lightcurvefile('AU Mic',
                                     mission='Tess').download()
au_mic_lc = test_data.PDCSAP_FLUX

time_series = au_mic_lc.astropy_time
flux_values = au_mic_lc.flux_quantity

time_series -= time_series[0]
time_series_min = time_series.to('min')


def generate_weights(flattened_lightcurve,
                     std_dev_thresh,
                     std_dev_clip,
                     max_pre_pad,
                     max_post_pad,
                     weight_kernel_dim,
                     nan_mask=None):
    sigma_mask, pre_pad_arr, post_pad_arr, sigma_excursion = prep_sigma_flare_mask(
        flattened_lightcurve,
        std_dev_thresh,
        std_dev_clip,
        max_pre_pad, max_post_pad,
        min_pre_pad=weight_kernel_dim // 2,
        min_post_pad=weight_kernel_dim,
        nan_mask=nan_mask)

    new_flare_mask = parametric_mask_dilation(sigma_mask, pre_pad_arr, post_pad_arr)

    new_flare_mask_weighted = np.ones(len(new_flare_mask))
    new_flare_mask_weighted[new_flare_mask] = 0

    # Soften the edges of the masked regions:
    new_flare_mask_weighted = np.convolve(weight_kernel, new_flare_mask, mode='same')

    new_flare_mask_weighted = 1 - new_flare_mask_weighted
    return new_flare_mask_weighted, sigma_mask, sigma_excursion


def fast_flatten(lightcurve: lk.LightCurve,
                 init_window_fit,
                 init_std_dev_thresh, init_std_dev_clip,
                 max_pre_pad, max_post_pad,
                 weight_kernel_dim,
                 nan_mask=None,
                 poly_order=2):
    # Round 1: Find flare estimates:

    flattened_lightcurve = lightcurve.flatten(window_length=init_window_fit, polyorder=poly_order).flux_quantity
    flattened_lightcurve -= np.nanmean(flattened_lightcurve)

    new_flare_mask_weighted, sigma_mask, sigma_excursion = generate_weights(flattened_lightcurve.value,
                                                                            init_std_dev_thresh,
                                                                            init_std_dev_clip,
                                                                            max_pre_pad, max_post_pad,
                                                                            weight_kernel_dim,
                                                                            nan_mask=nan_mask)

    return flattened_lightcurve, new_flare_mask_weighted, sigma_mask, sigma_excursion


def flatten_lc_with_weighted_sg(flux_values,
                                sg_poly_fit_order,
                                flatten_window_len,
                                data_weights,
                                num_iter=1,
                                iter_sigma_thresh=1,
                                std_dev_clip_thresh=10,
                                max_pre_pad=10,
                                max_post_pad=90,
                                weight_kernel_dim=15,
                                nan_mask=None,
                                residual_clip=5.):
    """Performs a weighted Savitsky Golay detrend, attempting to ignore flares in detrending."""
    assert num_iter >= 1
    _data_weights = data_weights.copy()
    for _iter in range(num_iter):
        print("Iter: ", _iter)
        reflattened_lc, bg_lc, lc_subslice, flattening_residuals = \
            sg_with_weighting_brute_force(flux_values,
                                          window_size=flatten_window_len,
                                          poly_order=sg_poly_fit_order,
                                          weight_mask=_data_weights)

        reflattened_lc -= np.nanmean(reflattened_lc)
        if _iter + 1 < num_iter:
            # std_dev = np.nanstd(reflattened_lc)
            # print(_iter, std_dev)
            # new_sigma_mask = np.abs(reflattened_lc / std_dev) > iter_sigma_thresh
            # _data_weights[lc_subslice][new_sigma_mask] = 0
            _new_data_weights, _, _ = generate_weights(reflattened_lc,
                                                       std_dev_thresh=iter_sigma_thresh,
                                                       std_dev_clip=std_dev_clip_thresh,
                                                       max_pre_pad=max_pre_pad,
                                                       max_post_pad=max_post_pad,
                                                       weight_kernel_dim=weight_kernel_dim,
                                                       nan_mask=nan_mask[lc_subslice])
            # excessive_residual_mask = (flattening_residuals > residual_clip) | (np.isnan(flattening_residuals))
            # _new_data_weights[excessive_residual_mask] = 0
            _data_weights[lc_subslice] = _new_data_weights

    return reflattened_lc, bg_lc, lc_subslice, flattening_residuals, _data_weights


lc_nan_mask = np.isnan(au_mic_lc.flux_quantity)

flattened_lc, weighted_flare_mask, sigma_mask, sigma_excursion = \
    fast_flatten(au_mic_lc,
                 fast_fit_window, fast_fit_std_dev_thresh, fast_fit_std_dev_clip,
                 pre_mask_dilation, post_mask_dilation, weight_mask_kernel_dim,
                 lc_nan_mask, poly_order=fast_fit_poly_order)

# TODO - clip excessive residuals within flattening iterations?

sg_flattened_lc, sg_trend_lc, sg_subslice, sg_residuals, updated_weights = \
    flatten_lc_with_weighted_sg(flux_values.value,
                                poly_fit_order, flatten_window, weighted_flare_mask,
                                num_iter=3,
                                iter_sigma_thresh=1,
                                std_dev_clip_thresh=fast_fit_std_dev_clip,
                                max_pre_pad=pre_mask_dilation,
                                max_post_pad=post_mask_dilation,
                                weight_kernel_dim=weight_mask_kernel_dim,
                                nan_mask=lc_nan_mask)

# plt.plot(time_series_min, flux_values, color='r')
f, ax = plt.subplots(nrows=4, sharex='all')
ax[0].plot(time_series_min[sg_subslice], sg_residuals,
           drawstyle='steps-post', color='k', lw=1, label='Detrending residuals')
ax[0].legend()
ax[1].plot(time_series_min, flux_values,
           drawstyle='steps-post', color='k', lw=1, label='PDCSAP_FLUX')
ax[1].plot(time_series_min[sg_subslice], sg_trend_lc,
           drawstyle='steps-post', color='b', lw=1, label='Detrended flux')
ax[1].legend()
ax[2].plot(time_series_min, sigma_mask,
           drawstyle='steps-post', color='r', label='Sigma-masked points', lw=1)
ax[2].plot(time_series_min, weighted_flare_mask,
           drawstyle='steps-post', color='gray', label='Initial data weights', lw=1)
ax[2].plot(time_series_min, updated_weights,
           drawstyle='steps-post', color='k', label='Data weights', lw=1)
ax[2].scatter(time_series_min, sigma_excursion, color='orange', marker='.', label="Clipped sigma excursion")
ax[2].legend()
ax[3].scatter(time_series_min[sigma_mask], flattened_lc[sigma_mask],
              color='r', marker='.', label='Sigma-masked points', lw=1)
ax[3].plot(time_series_min, flattened_lc,
           drawstyle='steps-post', color='gray', ls='--', label='Blind SG subtraction', lw=1)
ax[3].plot(time_series_min[sg_subslice], sg_flattened_lc,
           drawstyle='steps-post', color='k', label='Masked SG subtraction', lw=1)
ax[1].legend()
plt.show()

excessive_residuals = sg_residuals > residual_clip_val
detrended_lc = sg_flattened_lc.copy()
detrended_lc[excessive_residuals] = 0
updated_weights[sg_subslice][excessive_residuals] = 0

detrended_time_series = time_series_min[sg_subslice]
detrended_weights = updated_weights[sg_subslice]

f, ax = plt.subplots(nrows=2, sharex='all')
ax[0].plot(detrended_time_series, detrended_weights,
           drawstyle='steps-post', color='k', label='Final detrending weights', lw=1)
ax[1].plot(detrended_time_series, detrended_lc,
           drawstyle='steps-post', color='k', label='Final detrended lightcurve', lw=1)
plt.show()

list_of_flares_indices = eep.analyze.find_peaks_stddev_thresh(detrended_lc,
                                                              std_dev_threshold=flare_std_dev_thresh, min_index_gap=10,
                                                              single_flare_gap=None)
print("Number of flares identified: ", len(list_of_flares_indices))

eep.visualize.plot_flare_array(sg_flattened_lc, list_of_flares_indices, back_pad=20, forward_pad=90)
plt.plot(detrended_time_series, detrended_lc,
         drawstyle='steps-post', color='k', label='Masked SG subtraction', lw=1)
plt.scatter(detrended_time_series[list_of_flares_indices], detrended_lc[list_of_flares_indices],
            color='r')
plt.xlabel("Time (min)")
plt.ylabel("Flux (e-)")
plt.tight_layout()
plt.show()

non_overlapping_flares = []
rejected_flares = []
prev_ind = list_of_flares_indices[0]
for flare_ind in list_of_flares_indices[1:]:
    if flare_ind - prev_ind > flare_separation:
        non_overlapping_flares.append(prev_ind)
    else:
        rejected_flares.append(prev_ind)
    prev_ind = flare_ind

print("Number of non-overlapping flares: ", len(non_overlapping_flares))

eep.visualize.plot_flare_array(sg_flattened_lc, non_overlapping_flares,
                               back_pad=20, forward_pad=flare_separation, title="Non-overlapping flare selections")
eep.visualize.plot_flare_array(sg_flattened_lc, rejected_flares,
                               back_pad=20, forward_pad=flare_separation, title="Rejected flares")

extracted_flares = np.zeros((len(non_overlapping_flares), 1 + back_pad + echo_search_range))
for f_i, flare_ind in enumerate(non_overlapping_flares):
    extracted_flares[f_i] = detrended_lc[flare_ind - back_pad:flare_ind + echo_search_range + 1]

plt.plot(np.nanmean(extracted_flares, axis=0),
         drawstyle='steps-post', color='k', lw=1)
plt.title("Mean flare")
plt.show()

normalized_flares = extracted_flares.copy()
for f_i, flare_ind in enumerate(non_overlapping_flares):
    normalized_flares[f_i] /= detrended_lc[flare_ind]

f, ax = plt.subplots(ncols=2)
ax[0].plot(np.nanmean(normalized_flares, axis=0),
           drawstyle='steps-post', color='k', lw=1)
ax[0].set_title("Mean normalized flare")
for f_i in range(len(normalized_flares)):
    ax[1].plot(normalized_flares[f_i],
               drawstyle='steps-post', lw=1)
ax[1].set_title("All normalized flares")
plt.show()

# Window stability testing:
region_of_interest = (1600, 3000)

window_tests = np.arange(flatten_window - 400, flatten_window + 700, 100, dtype=int)
window_results = np.zeros((len(window_tests), region_of_interest[1] - region_of_interest[0]))

for w_i, window_len in enumerate(window_tests):
    sg_flattened_lc, sg_trend_lc, sg_subslice, sg_residuals, new_weights = \
        flatten_lc_with_weighted_sg(flux_values.value,
                                    poly_fit_order, window_len, updated_weights)

    start_ind = region_of_interest[0] - sg_subslice.start
    end_ind = region_of_interest[-1] - sg_subslice.start
    window_results[w_i] = sg_flattened_lc[start_ind:end_ind]
    # plt.plot(time_series_min[sg_subslice], sg_flattened_lc, lw=1, label="Window length: " + str(window_len))
# plt.title("Impact of different SG windows")
# plt.legend()
# plt.show()

window_difference = window_results[2:] + window_results[:-2] - 2 * window_results[1:-1]
sum_diff = np.sum(window_difference ** 2, axis=1)
plt.plot(window_tests[1:-1], sum_diff, color='k')
plt.xlabel("Window size")
plt.ylabel("Window-size difference sum squared")
plt.tight_layout()
plt.show()

poly_order_tests = np.arange(5, 20, dtype=int)
poly_order_results = np.zeros((len(poly_order_tests), region_of_interest[1] - region_of_interest[0]))

for p_i, poly_order in enumerate(poly_order_tests):
    sg_flattened_lc, sg_trend_lc, sg_subslice, sg_residuals, new_weights = \
        flatten_lc_with_weighted_sg(flux_values.value,
                                    poly_order, flatten_window, updated_weights)

    start_ind = region_of_interest[0] - sg_subslice.start
    end_ind = region_of_interest[-1] - sg_subslice.start
    poly_order_results[p_i] = sg_flattened_lc[start_ind:end_ind]
#     plt.plot(time_series_min[sg_subslice], sg_flattened_lc, lw=1, label="Window length: " + str(window_len))
# plt.title("Impact of different SG polynomial orders")
# plt.legend()
# plt.show()

poly_difference = poly_order_results[2:] + poly_order_results[:-2] - 2 * poly_order_results[1:-1]
sum_diff = np.sum(poly_difference ** 2, axis=1)
plt.plot(poly_order_tests[1:-1], sum_diff, color='k')
plt.xlabel("Polynomial order")
plt.ylabel("Poly-order difference sum squared")
plt.tight_layout()
plt.show()

# TODO - use optimally detrended curves to identify flares
