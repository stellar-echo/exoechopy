#Input: star name, quarter
#Output: time index for all flares
from lightkurve import search_lightcurvefile
import lightkurve as lk
import warnings
import numpy as np
from scipy.signal import find_peaks

star_name, star_quarter = input("Enter star name and quarter: ").split(', ')

warnings.filterwarnings("ignore", message="Cannot download from an empty search result.")
#ignore multiple file warning
warnings.filterwarnings("ignore", message="Warning: {} files available to download. ")
lcf_short = search_lightcurvefile(str(star_name),quarter=star_quarter, cadence='short').download()
lcf_long = search_lightcurvefile(str(star_name),quarter=star_quarter).download()

if lcf_long is None:
	pass
else:
	lc = lcf_long.PDCSAP_FLUX.remove_nans().flatten()
	y = lc.flux
	median = np.median(y)
	sigma = np.std(y)
	flare_threshold = median+(6*sigma)
	peak_times, peak_val = find_peaks(y, height=flare_threshold,distance=4)
	lc.plot()
	print("Long cadence flare time index values: ", peak_times)
	print(" ")

if lcf_short is None:
	pass
else:
	#download all files for this quarter 
	lcf_short_all = search_lightcurvefile(str(star_name),quarter=star_quarter, cadence='short').download_all()
	total_num = len(lcf_short_all)
	counter = 0
	#Run loop parsing through all files and print info for all of them seperately
	for i in lcf_short_all:
		counter +=1
		lc = i.PDCSAP_FLUX.remove_nans().flatten()
		y = lc.flux
		median = np.median(y)
		sigma = np.std(y)
		flare_threshold = median+(6*sigma)
		peak_times, peak_val = find_peaks(y, height=flare_threshold,distance=30)
		#lc.plot()
		print("Short cadence flare time index values (file ", counter, "/", total_num, "): ", peak_times)
		print(" ")