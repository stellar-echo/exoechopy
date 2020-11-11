# Creates an astropy table from a directory of fits files. Displays relevant information from headers and light curves. 
# Tables can be converted into pandas DataFrame objects and manipulated in many ways.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
import os
import glob
from astropy.table import Table, Column
import lightkurve as lk
from scipy.signal import find_peaks

keys = ["OBJECT", "OBSMODE", "QUARTER", "TEFF", "RADIUS", "KEPMAG"]
hdu = 0

# Eventually, this will loop through all top-level directories. For now, directory must be manually specified
dir_ = "../../../02_kepler_time_series_scripts/21_Kepler_Q14/"
directories = glob.glob(dir_ + "*_Kepler_Q*/")

# Store the relevant information
values = []
files = []
total_flares = []
flares_above_6_sigma = []
median_flare_int = []
bv_color = []
flares_one_percent = []
flares_four_percent = []
raw_variability = []
raw_means = []
det_vars = []
det_sig = []
sn_quality = []
raw_sigma = []
raw_sn_quality = []

# Loop through each directory and each file in it
# for directory in directories:
for ind,file in enumerate(glob.glob(dir_+"*.fits")):
    # Get header contents
    header = fits.getheader(file, hdu)
    
    print("File number", ind, "has Temp", header.get("TEFF"), type(header.get("TEFF")))
    if header.get("TEFF") is None:
        header["TEFF"] = 1
        
    print("File number", ind, "has Temp", header.get("TEFF"), type(header.get("TEFF")))
    
    values.append([header.get(key) for key in keys])
    files.append(file)   
    
    # Get Raw Target Variability
    header2 = fits.getheader(file, 1)
    raw_variability.append(header2.get("PDCVAR"))
    
    # Convert G-R Color to B-V Color
    # Source: http://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php
    
    print("File number", ind, "has G-R Color", header.get("GRCOLOR"), type(header.get("GRCOLOR")))
    if header.get("GRCOLOR") is None:
        header["GRCOLOR"] = 0.0
    
    print("File number", ind, "has G-R Color", header.get("GRCOLOR"), type(header.get("GRCOLOR")))
    
    gr = header.get("GRCOLOR")
    bv = 0.98*(gr) + 0.22
    
    bv_color.append(bv)
    
    # Get number of flares and flare times
    lc_raw = fits.open(str(file))
    raw_flux = lc_raw[1].data["PDCSAP_FLUX"]
    time = lc_raw[1].data["TIME"]
    
    lc = lk.LightCurve(time = time, flux = raw_flux)
    lc = lc.remove_nans().flatten()
    
    # Get raw flux mean, detrended variance and standard deviation, along with 1/sigma sn quality score
    raw_mean = np.nanmean(raw_flux)
    raw_means.append(raw_mean)

    variance = np.var(lc.flux)
    det_vars.append(variance)

    sig = np.std(lc.flux)
    det_sig.append(sig)

    sn_qual = 1/sig
    sn_quality.append(sn_qual)

    # Just for fun, include Raw mean/ Raw sig as well
    raw_sig = np.nanstd(raw_flux)
    raw_sn_qual = raw_mean/raw_sig
    
    # different cadences require different flare detection windows
    cadence = header.get("OBSMODE")
    if cadence == "short cadence":
        x = lc.flux
        median = np.median(x)
        sigma = np.std(x)
        flare_threshold = median + (3*sigma)
        peaks, peak_val = find_peaks(x, height=flare_threshold, distance=30)
        total_flares.append(len(peaks))
        
        # Get median flare intensity
        flare_heights = []
        for val in peak_val.values():
            for num in val:
                flare_heights.append(num)
                
        # Calculate the percentage above background flux of each flare
        peaks_one = []
        peaks_four = []
        for flare in flare_heights:
            raw_val = 100*(flare - 1)
            if raw_val > 1:
                peaks_one.append(raw_val)

            if raw_val > 4:
                peaks_four.append(raw_val)

        flares_one_percent.append(len(peaks_one))
        flares_four_percent.append(len(peaks_four))
            
        median_flare_int.append(np.median(flare_heights))
            
        flare_threshold_six_sigma = median + (6*sigma)
        peaks_six, peak_val_six = find_peaks(x, height=flare_threshold_six_sigma, distance=30)
        flares_above_6_sigma.append(len(peaks_six))
        
    else:
        y = lc.flux
        median = np.median(y)
        sigma = np.std(y)
        flare_threshold = median + (3*sigma)
        peaks, peak_val = find_peaks(y, height=flare_threshold, distance=4)
        total_flares.append(len(peaks))
        
        # Get median flare intensity
        flare_heights = []
        for val in peak_val.values():
            for num in val:
                flare_heights.append(num)
                
        # Calculate the percentage above background flux of each flare
        peaks_one = []
        peaks_four = []
        for flare in flare_heights:
            raw_val = 100*(flare - 1)
            if raw_val > 1:
                peaks_one.append(raw_val)

            if raw_val > 4:
                peaks_four.append(raw_val)

        flares_one_percent.append(len(peaks_one))
        flares_four_percent.append(len(peaks_four))
                    
        median_flare_int.append(np.median(flare_heights))
        
        flare_threshold_six_sigma = median + (6*sigma)
        peaks_six, peak_val_six = find_peaks(y, height=flare_threshold_six_sigma, distance=4)
        flares_above_6_sigma.append(len(peaks_six))
        
    lc_raw.close()
        
    print("Finished", ind, "of", len(os.listdir(dir_)), "in directory", dir_)
       

# Construct table
row0 = [dict(zip(keys, values[0]))]
t = Table(row0, names=keys, masked = True)

for i in range(1, len(values)):
    t.add_row(values[i])

new_column = Column(name='path', data=files)
t.add_column(new_column, 0)

flares = Column(name = "Total Flares", data = total_flares)
t.add_column(flares)

flares_sixsig = Column(name = "Flares Above Six Sigma", data = flares_above_6_sigma)
t.add_column(flares_sixsig)

med_int = Column(name="Median Flare Intensity", data=median_flare_int)
t.add_column(med_int)

bv_col = Column(name="B-V Color Estimate", data=bv_color)
t.add_column(bv_col)

fls_one = Column(name="Flares above 1% ", data=flares_one_percent)
t.add_column(fls_one)

fls_four = Column(name="Flares above 4%", data=flares_four_percent)
t.add_column(fls_four)

raw_targ_var = Column(name="Raw Target Variability", data=raw_variability)
t.add_column(raw_targ_var)

var_ = Column(name="De-trended Variance", data=det_vars)
t.add_column(var_)

sigs = Column(name="De-trended Sigma", data=det_sig)
t.add_column(sigs)

sn_q = Column(name="SN Quality (1/Sigma)", data=sn_quality)
t.add_column(sn_q)

raw_f_mean = Column(name="Raw Flux Mean", data=raw_means)
t.add_column(raw_f_mean)

# raw_sigs = Column(name="Raw Sigma", data=raw_sigma)
# t.add_column(raw_sigs)

# raw_sn = Column(name="Raw SN Quality (Raw Mean/Raw Sigma)", data=raw_sn_quality)
# t.add_column(raw_sn)

# df = t.to_pandas()

# ax1 = df.plot.scatter(x="Raw Target Variability", y="Raw Flux Mean")
# plt.savefig("var_vs_flux.png", overwrite=True)

# Save table as a file
t.write("kepler_q14.html", format = "ascii.html", overwrite = True)
