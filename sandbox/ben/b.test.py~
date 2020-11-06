import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import lightkurve as lk


file = fits.open("../02_kepler_time_series_scripts/01_Kepler_KOI/kplr010514429-2012310112549_slc.fits")

time = file[1].data["TIME"]
flux = file[1].data["PDCSAP_FLUX"]
rawflux = file[1].data["SAP_FLUX"]

plt.figure(figsize=(12,6))
plt.plot(time, flux, c="b", label="PDCSAP_FLUX")
plt.plot(time, rawflux, c="g", label="SAP_FLUX")
plt.xlabel("Time (Days)")
plt.ylabel("Flux (e-/s)")
plt.title("{}".format(file[1].header["OBJECT"]))
plt.legend()
plt.show()


lc = lk.LightCurve(time=time, flux=flux)
lc_det = lc.flatten()
lc_det.plot()
plt.title("De-trended curve")
plt.show()

