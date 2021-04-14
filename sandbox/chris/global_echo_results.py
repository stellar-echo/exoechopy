import os
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import exoechopy as eep

# _fp = "C:\\Users\\cmann\\PycharmProjects\\exoechopy\\disks\\export_deltas_only_snr_101\\"
_fp = "C:\\Users\\cmann\\PycharmProjects\\exoechopy\\disks\\SC_disk_echo_export_deltas_only_snr\\"

fp = Path(_fp)
output_folder = fp/"Processed data results"
if not output_folder.exists():
    os.mkdir(output_folder)

try:
    composite_array = np.load(fp / 'composite_array.npy')
except FileNotFoundError:
    all_folders = [f for f in fp.iterdir() if f.is_dir()]
    composite_data = []

    for folder in all_folders:
        try:
            data_file = [x for x in os.listdir(fp / folder) if x.split('.')[-1] == 'npy'][0]
            if len(data_file) > 0:
                composite_data.append(np.load(fp / folder / data_file))
        except IndexError:
            continue

    composite_array = np.array(composite_data)

    print(composite_array.shape)

    np.save(fp / 'composite_array.npy', composite_array)

    print("Done!")

lag_slice = slice(8, None)

time_domain = composite_array[0][0]
all_echoes = composite_array[:, 1, 8:]
all_std_dev = composite_array[:, 2, 8:]

print(np.mean(all_echoes))
print(np.mean(all_echoes, axis=0))


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


generate_kde_plots(all_echoes, weight_array=all_std_dev, x_axis_loc=0, y_axis_loc=0,
                   savefile=(output_folder/"all_data.png"))

for lag in range(len(all_echoes[0])):
    generate_kde_plots(all_echoes[:, lag], weight_array=all_std_dev[:, lag], plot_title="lag=" + str(lag),
                       x_axis_loc=0, y_axis_loc=0,
                       savefile=(output_folder/("data_at_lag_" + str(lag) + ".png")))

#
# xvals = np.linspace(np.nanmin(flare_tail_flat), np.nanmax(flare_tail_flat), 500)
# dx = xvals[1] - xvals[0]
#
# final_normed_flare_histo = eep.analyze.TophatKDE(flare_tail_flat,
#                                                  bandwidth=5 * dx,
#                                                  weights=std_dev_flat)
# final_normed_flare_histo_gauss = eep.analyze.GaussianKDE(flare_tail_flat,
#                                                          bandwidth='scott',
#                                                          weights=std_dev_flat)
#
# flare_tail_kde_gauss = final_normed_flare_histo_gauss(xvals)
# flare_tail_kde = final_normed_flare_histo(xvals)
# normal_approx = eep.utils.gaussian(xvals, np.nanmean(flare_tail_flat), np.nanstd(flare_tail_flat))
#
# f, ax = plt.subplots(ncols=2, figsize=(12, 6))
# ax[0].plot(xvals, flare_tail_kde_gauss, color='k', zorder=1,
#            label="Scott's rule Gaussian KDE")
# ax[0].plot(xvals, flare_tail_kde, color='gray', zorder=0, label='5-bin Tophat KDE')
# ax[0].plot(xvals, normal_approx, color='b', ls='--', zorder=-1, label='Gaussian fit to distribution')
# ax[0].set_xlabel("Values")
# ax[0].set_ylabel("Density")
# ax[0].set_title("Distribution of values")
# # if y_axis_loc is not None:
# #     ax[0].axvline(x=y_axis_loc, color='k', lw=1)
# ax[0].legend()
#
# ax[1].plot(xvals, flare_tail_kde_gauss - normal_approx, color='k', zorder=1, label="Scott's rule KDE - normal")
# ax[1].plot(xvals, flare_tail_kde - normal_approx, color='gray', zorder=0, label="5-bin Tophat KDE - normal")
# ax[1].set_xlabel("Values")
# ax[1].set_ylabel("Difference from normal density")
# ax[1].set_title("Difference from normal distribution")
# # if y_axis_loc is not None:
# #     ax[1].axvline(x=y_axis_loc, color='k', lw=1)
# # if x_axis_loc is not None:
# #     ax[1].axhline(y=x_axis_loc, color='k', lw=1)
# ax[1].legend()
#
