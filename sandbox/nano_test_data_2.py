import os
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.ndimage import morphology
import exoechopy as eep

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Manual inputs section
fp = "C:\\Users\\cmann\\Documents\\Nanohmics\\Stellar Echo\\Disk project\\blind test data\\"
filename = "short_cadence_with_echo_nano.npy"

time_domain_filename = "kepler_sc_time.npy"

pre_flare_pad = 5
max_lag_inds = 25
flare_sigma_test = 1

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Load and pre-process the data
filepath = Path(fp)

output_folder = filepath/(filename.split('.')[0])

if not output_folder.exists():
    os.mkdir(output_folder)

raw_data = np.load(filepath / filename)
time_domain = np.load(filepath / time_domain_filename)

cadence = (time_domain[1] - time_domain[0]) * 24 * 60
speed_of_light = 299792458  # m/s
au_to_m = 1.496e+11
print("Cadence (min): ", cadence)
print("Light travel per cadence (au): ", (speed_of_light * cadence * 60) / au_to_m)


# list of indices:
ind_dat = np.arange(len(raw_data))
# locations of nans, boolean array:
nan_mask = np.isnan(raw_data)

# Note, this is super naive and just happening because this data set is so simple,
# a much more robust solution should happen here:

# First, heal the easy ones (single missing point):
healed_data = raw_data.copy()
for ind in ind_dat[nan_mask]:
    healed_data[ind] = (healed_data[ind - 1] + healed_data[ind + 1]) / 2

nan_mask_2 = np.isnan(healed_data)
healed_data[nan_mask_2] = np.nanmean(healed_data)

mean_subtracted_data = healed_data.copy()
mean_subtracted_data -= np.mean(mean_subtracted_data)

# Remove long-range periodic background with fft filter (better ways exist, this works for now):
fft_data = np.fft.rfft(mean_subtracted_data)
# Picking a cutoff value:
biggest_contribution = np.argmax(np.abs(fft_data))
# Logistic function (not sure if this is a good idea or not, but gets it quickly):
window = 1 / (1 + np.exp(-2.5 * (np.arange(len(fft_data)) - 1.5 * biggest_contribution)))

filtered_data = np.fft.irfft(fft_data * window)

flare_mask = mean_subtracted_data > 1 * np.std(filtered_data)
flare_mask = morphology.binary_dilation(flare_mask, iterations=5)

flare_indices = eep.analyze.find_peaks_stddev_thresh(filtered_data,
                                                     std_dev_threshold=flare_sigma_test,
                                                     min_index_gap=10,
                                                     extra_search_pad=10,
                                                     num_smoothing_rad=5,
                                                     single_flare_gap=10)
print("flare_indices: ", flare_indices)
unreliable_values = morphology.binary_dilation(nan_mask_2, iterations=10)
flare_indices = [x for x in flare_indices if x not in ind_dat[unreliable_values]]

time_since_start = np.arange(0, len(healed_data)) * (time_domain[1] - time_domain[0])

#  =============================================================  #
# Review work so far...
f, ax = plt.subplots(nrows=2, figsize=(18, 6))
ax[0].plot(time_since_start, healed_data, color='k', lw=1, drawstyle='steps-post')
ax[0].scatter(time_since_start[ind_dat[nan_mask]], healed_data[nan_mask], color='b', marker='+', label='healed_nan')
ax[0].scatter(time_since_start[ind_dat[nan_mask_2]], healed_data[nan_mask_2], color='r', marker='x', label='unhealed_nan')
ax[0].set_xlabel("Time (days)")
ax[0].set_ylabel("Counts")
ax[0].legend()
ax[0].set_title("Raw lightcurve (days since obs. start)")

ax[1].plot(filtered_data, color='k', lw=1, drawstyle='steps-post')
ax[1].scatter(flare_indices, filtered_data[flare_indices], color='g', label='flare local')
ax[1].set_xlabel("Time (indices)")
ax[1].set_ylabel("Counts - fft filtered")
ax[1].set_title("Background-subtracted lightcurve")

plt.tight_layout()
plt.savefig(output_folder/"lightcurve_overview.png")
plt.close()

eep.visualize.plot_flare_array(filtered_data, flare_indices,
                               back_pad=pre_flare_pad, forward_pad=max_lag_inds,
                               display_index=True)

full_autocorr = eep.analyze.autocorrelate_array(filtered_data, max_lag=max_lag_inds)

plt.plot(full_autocorr, color='k', lw=1, drawstyle='steps-post')
plt.xlabel("Time lag (indices)")
plt.ylabel("Correlation (normed)")
plt.title("Autocorrelation of entire dataset")
plt.savefig(output_folder/"raw_lightcurve_autocorr.png")
plt.close()

#  =============================================================  #
# More detailed analysis
all_autocorr = []
sum_autocorr = np.zeros(max_lag_inds + 1)
sum_autocorr_weighted = np.zeros(max_lag_inds + 1)
ct = 0
total_weight = 0
for flare_index in flare_indices:
    post_flare_pad = max(max_lag_inds + pre_flare_pad + 1, 2 * max_lag_inds)
    # post_flare_pad = max_lag_inds + pre_flare_pad + 1
    if flare_index - pre_flare_pad > 0 and flare_index + post_flare_pad < len(filtered_data):
        flare_curve = filtered_data[flare_index - pre_flare_pad:flare_index + post_flare_pad]
        this_autocorr = eep.analyze.autocorrelate_array(flare_curve, max_lag=max_lag_inds)

        f, ax = plt.subplots(nrows=2)
        ax[0].plot(np.arange(flare_index - pre_flare_pad, flare_index + post_flare_pad),
                   flare_curve, color='k', lw=1, drawstyle='steps-post')
        ax[0].scatter([flare_index], filtered_data[flare_index], color='r')
        ax[0].set_title("BG-subbed flare at index: " + str(flare_index))
        ax[0].set_ylabel("Intensity - background")
        ax[0].set_xlabel("Time (indices)")
        ax[1].plot(this_autocorr, color='k', lw=1, drawstyle='steps-post')
        ax[1].set_ylabel("Correlation (normed)")
        ax[1].set_xlabel("Lag (indices)")
        ax[1].set_title("Associated autocorrelation")
        plt.tight_layout()
        plt.savefig(output_folder/("flare_"+str(flare_index)+"_correlation.png"))
        plt.close()

        all_autocorr.append(this_autocorr)
        print(flare_index, len(flare_curve), len(this_autocorr), len(sum_autocorr))
        sum_autocorr += this_autocorr
        # Add a magnitude-based weight to the autocorrelation (more weight to brighter flares)
        sum_autocorr_weighted += this_autocorr * np.max(flare_curve)

        total_weight += np.max(flare_curve)
        ct += 1

print(ct)

sum_autocorr /= ct
sum_autocorr_weighted /= total_weight

num_row, num_col = eep.utils.row_col_grid(ct)
fig, all_axes = plt.subplots(num_row, num_col, figsize=(10, 6))
for f_i in range(ct):
    c_i = f_i // num_row
    r_i = f_i - num_row * c_i
    all_axes[r_i, c_i].plot(all_autocorr[f_i], color='k', lw=1, drawstyle='steps-post')
plt.savefig(output_folder/"autocorrelation_overview.png")
plt.close()

f, ax = plt.subplots(nrows=2, figsize=(8, 6))
ax[0].plot(np.arange(4, len(full_autocorr)), full_autocorr[4:], color='k', lw=1, drawstyle='steps-post')
ax[0].set_title("Naive full autocorrelation")

ax[1].plot(np.arange(4, len(full_autocorr)), sum_autocorr[4:], color='k', lw=1, drawstyle='steps-post', label='summed')
ax[1].plot(np.arange(4, len(full_autocorr)), sum_autocorr_weighted[4:], color='b', lw=1, drawstyle='steps-post',
           label='weighted')
ax[1].set_title("Summed extracted flare autocorration")
ax[1].set_xlabel("Lag (ind)")
ax[1].legend()
plt.savefig(output_folder/"Summed extracted flare results.png")
plt.close()

print("Done.")