import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import lightkurve as lk
import os
import sys

if len(sys.argv) != 2:
   print(f'usage {sys.argv[0]} star_id')
   quit()
star=sys.argv[1]

kpath = '/home/echo/hdd6tb/02_kepler_time_series_scripts/'
qtrlis  = []
for qd in os.listdir(kpath):
   if '_Q' in qd:
      for f in os.listdir(kpath+qd):
         if star in f:#
            qtrlis.append(kpath+qd+'/'+f)
#            print(kpath+qd+'/'+f)

plt.figure(figsize=(12,6))

for fin in qtrlis:
    print(fin)
    file = fits.open(fin)
    time = file[1].data["TIME"]
    flux = file[1].data["PDCSAP_FLUX"]
    rawflux = file[1].data["SAP_FLUX"]
    myflux = flux/np.nanmedian(flux)
    plt.plot(time, myflux, c="b")
    #plt.plot(time, rawflux, c="g")
    print(time[0],time[-1])
#    plt.plot(time, flux, c="b", label="PDCSAP_FLUX")
#    plt.plot(time, rawflux, c="g", label="SAP_FLUX")


plt.xlabel("Time (Days)")
plt.ylabel("Flux (e-/s)")
plt.title("{}".format(file[1].header["OBJECT"]))
plt.legend()
plt.show()

# f = 'b.tmp.png'

# Adding this to help differentiate btwn different stars when playing w/this code
f = "{}_allqtr_lc.png".format(file[1].header["OBJECT"])
plt.savefig(f)




