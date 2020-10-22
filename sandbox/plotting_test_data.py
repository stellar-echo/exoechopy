
import os
import numpy as np
import exoechopy as eep
from pathlib import Path


cwd = Path.cwd()
test_data_folder = (cwd.parent/'disks')/'kepler_candidate_test_data'

#   ------------------------------------    #
# This section is specific to how the data was previously named,
# we can/should enforce a specific naming convention moving forward
all_test_data = [x for x in os.listdir(test_data_folder) if x[-3:] == 'npy']

all_flux_data = [x for x in all_test_data if x.split('.')[0].split('_')[-1] == 'flux']
all_time_data = [x for x in all_test_data if x.split('.')[0].split('_')[-1] == 'time']

# Build dict of all candidate data
flux_dict = {}
for flux_filename in all_flux_data:
    filename = flux_filename.split('_')[1]
    flux_data = np.load(test_data_folder / flux_filename)
    time_filename = [x for x in all_time_data if x.split('_')[1] == filename][0]
    time_data = np.load(test_data_folder/time_filename)
    temp_dict = {'flux': flux_data,
                 'time': time_data,
                 'filename': filename}
    flux_dict[filename] = temp_dict

#   --------------------------------------   #
# Visualize:
for filename in flux_dict.keys():
    lightcurve = flux_dict[filename]['flux']
    lc_nans = np.isnan(lightcurve)
    # hard_to_heal = lc_nans[1:] & lc_nans[:-1]
    # ez_heal = lc_nans[1:] & ~hard_to_heal
    #
    eep.visualize.interactive_lightcurve(flux_dict[filename]['time'],
                                         lightcurve)




