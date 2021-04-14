import os
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

fp = Path("C:\\Users\\cmann\\PycharmProjects\\exoechopy\\disks\\")
folder_list = ['export_deltas_only_snr_51', 'export_deltas_only_snr_101',
               'export_deltas_only_snr_151']  # , 'export_deltas_only_snr_201'

clip_index = 1

window_list = [int(f_i.split('_')[-1]) for f_i in folder_list]

file_list = os.listdir(fp / folder_list[0])
all_stars = [x for x in file_list if os.path.isdir(fp / folder_list[0] / x)]

test_import_file = [x for x in os.listdir(fp / folder_list[0] / all_stars[0]) if x.split('.')[-1] == 'npy'][0]
test_data = np.load(fp / folder_list[0] / all_stars[0] / test_import_file)

mean_results = np.zeros((len(folder_list), len(test_data[0]) - clip_index))
std_dev_results = np.zeros((len(folder_list), len(test_data[0]) - clip_index))

results_dict = {}

for star in all_stars:
    for f_i, folder in enumerate(folder_list):
        current_dir = fp / folder / star
        try:
            data_file = [x for x in os.listdir(current_dir) if x.split('.')[-1] == 'npy'][0]
        except IndexError:
            print(star)
            print(os.listdir(current_dir))
            print(folder)
            results_dict[star] = "No .npy file"
            break
        star_data = np.load(current_dir / data_file)
        mean_results[f_i] = star_data[1][clip_index:]
        std_dev_results[f_i] = star_data[2][clip_index:]
    f, ax = plt.subplots(ncols=5, figsize=(18, 5))
    for mean, window in zip(mean_results, window_list):
        ax[0].plot(mean, label=window)
    ax[0].legend()
    ax[0].set_title("Mean values")
    for std, window in zip(std_dev_results, window_list):
        ax[1].plot(std, label=window)
    ax[1].legend()
    ax[1].set_title("Std error of means")
    ax[2].plot(np.max(mean_results, axis=0) - np.min(mean_results, axis=0))
    ax[2].set_title("Maximum difference in mean values")
    std_dev_max_diff = np.max(std_dev_results, axis=0) - np.min(std_dev_results, axis=0)
    ax[3].plot(std_dev_max_diff)
    ax[3].set_title("Maximum difference in std dev values")
    ax[4].plot(std_dev_max_diff/std_dev_results[1]*100)
    ax[4].set_title("Max percent difference in std dev from 101")
    plt.suptitle(star)
    plt.tight_layout()
    plt.show(block=True)
