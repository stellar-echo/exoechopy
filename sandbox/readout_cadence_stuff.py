import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import exoechopy as eep
from astropy import units as u
from pathlib import Path


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# User input section

# Flare parameters:
onset_time = 8
decay_time = 16

# Kepler values:
int_time = 6.02  # s
readout_time = 0.52  # s
cadence = int_time + readout_time

# How many slightly shifted times to study
num_offsets = 10

# How many frames are summed (short cadence, it's always 9)
frames_per_sum = 9

# Duration of the simulation (approximately)
num_sums = 5

# Update this to match your computer if you want to save the results
save_filepath = "C:\\Users\\cmann\\Documents\\Nanohmics\\Stellar Echo\\" \
                "T3 Reports\\Progress Report 10\\Flare shape study\\"

#  =============================================================  #

saveloc = Path(save_filepath)
base_filename = str(int_time) + "_" + str(readout_time) + "_" + str(onset_time) + "_" + str(decay_time) + "_"

num_times = 2*(num_sums+2)*frames_per_sum

start_offsets = np.linspace(0, int_time+readout_time, num_offsets)[:-1]

onset_s = u.Quantity(onset_time, 's')
decay_s = u.Quantity(decay_time, 's')

flare_1 = eep.simulate.ParabolicRiseExponentialDecay(onset_s, decay_s)

exact_times = u.Quantity(np.linspace(-2*cadence*frames_per_sum, cadence*num_sums*frames_per_sum, num_times*20), 's')
exact_flare = flare_1.evaluate_over_array(exact_times)

print("Exact solution | \tcounts:",
      "{:.2f}".format(np.sum(exact_flare)*(exact_times[1]-exact_times[0])))

for offset in start_offsets:
    all_times = [-cadence*frames_per_sum*2+offset]
    mask = []
    for t_i in range(num_times):
        if t_i % 2 == 0:
            all_times.append(all_times[-1] + int_time)
            mask.append(True)
        else:
            all_times.append(all_times[-1] + readout_time)
            mask.append(False)

    all_times_s = u.Quantity(all_times, 's')
    dt_array = all_times_s[1:] - all_times_s[:-1]

    flare_values = flare_1.evaluate_over_array(all_times_s)

    inv_mask = ~np.array(mask)
    masked_flare_values = flare_values[mask]
    inv_masked_flare_values = flare_values[inv_mask]
    masked_times = all_times_s[:-1][mask]

    # plt.plot(all_times_s[:-1], flare_values, lw=1, drawstyle='steps-post', label=str(offset))
    f, ax = plt.subplots(ncols=5, nrows=2, figsize=(20, 8))
    ax[0, 0].plot(exact_times[:-1], exact_flare,
                  color='gray', lw=1.25, drawstyle='steps-post', label="Perfect cadence")
    ax[0, 0].plot(masked_times, masked_flare_values,
                  color='orangered', lw=1.25, drawstyle='steps-post', label="{:.2f}".format(offset))

    total_counts = np.sum(masked_flare_values*dt_array[mask])
    lost_counts = np.sum(inv_masked_flare_values*dt_array[inv_mask])
    verify_counts = total_counts + lost_counts
    print("Offset: ",
          "{:.2f}".format(offset), "| \tcounts:",
          "{:.2f}".format(total_counts),
          "| \tlost photons:",
          "{:.2f}".format(lost_counts), "| \tcombined:",
          "{:.2f}".format(verify_counts), "| \t% lost:",
          "{:.2f}".format(100*lost_counts/verify_counts)+"%")
    ax[0, 0].set_title("Fraction of signal lost: " + "{:.2f}".format(100*lost_counts/verify_counts)+"%")
    ax[0, 0].set_xlabel("Time (s)")
    ax[0, 0].set_ylabel("Flux (ct/m²s)")
    ax[0, 0].legend()

    for frame_offset in range(frames_per_sum):
        times = []
        integrated_lc = []
        for f_i, (time, measurement) in enumerate(zip(masked_times[frame_offset:], masked_flare_values[frame_offset:])):
            if f_i % frames_per_sum == 0:
                times.append(time.value)
                integrated_lc.append(measurement.value)
            else:
                integrated_lc[-1] += measurement.value
        if frame_offset < 4:
            ax_slice = 0, frame_offset + 1
        else:
            ax_slice = 1, frame_offset - 4
        ax[ax_slice].plot(exact_times[:-1], 3*exact_flare,
                          color='gray', lw=1.25, drawstyle='steps-post', label="Exact flare (arb scale)")
        ax[ax_slice].plot(times, integrated_lc, lw=1, drawstyle='steps-post', label=str(frame_offset))
        ax[ax_slice].set_title("Summed frames, frame offset "+str(frame_offset))
        ax[ax_slice].set_xlabel("Time (s)")
        ax[ax_slice].set_ylabel("Counts (/m²)")
    plt.tight_layout()
    plt.savefig(saveloc / (base_filename+"{:.2f}".format(offset) + ".png"))
    plt.close()


# TODO:
# - Make a helper function that accepts cadence, readout time, start time, and end time and generates points
# - Make a function that uses the helper function to discretize the flare, integrate with exoechopy

