
import numpy as np
from astropy.io import fits
from astroquery.mast import Catalogs
from astroquery.mast import Observations
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.signal as signal

target_name = "AU Mic"

# not...
## Let's label the axes and define a title for the figure.
#fig.suptitle("WASP-126 b Light Curve - Sector 1")

# fits_file = "https://archive.stsci.edu/missions/tess/tid/s0001/0000/0000/2515/5310/tess2018206045859-s0001-0000000025155310-0120-s_lc.fits"


if False:
    fits_file = "https://archive.stsci.edu/missions/tess/tid/s0001/0000/0004/4142/0236/tess2018206045859-s0001-0000000441420236-0120-s_lc.fits"
    fits.info(fits_file)
    fits.getdata(fits_file, ext=1).columns
    with fits.open(fits_file, mode="readonly") as hdulist:
        tess_bjds = hdulist[1].data['TIME']
        sap_fluxes = hdulist[1].data['SAP_FLUX']
        pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']
    #dataraw = np.array([tess_bjds, sap_fluxes, pdcsap_fluxes])
    #np.save('au_mic.lc.npy',dataraw)


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def do_flares(t,f):
    msk = np.isnan(f)==False
    f = f[msk]
    t = t[msk]
    fbar = np.mean(f)
    f /= fbar
    t -= t[0]
    fn = interp.interp1d(t,f,kind='nearest')
    dtest = t[1]-t[0]

    #t/=t[-1]
    #fcut = 1000.0 # 1.03
    #msk = f<fcut
    #fc,tc = f[msk],t[msk]
    tc, fc = t,f
    fnfi = interp.interp1d(tc,fc,kind='linear')
    ng = len(tc)
    tg  = np.linspace(tc[0],tc[-1],ng)
    fg = fnfi(tg)
    if 1:
        fs = 1/dtest
        fcut = 1/dtest/(30*24*1)
        fgf = butter_highpass_filter(fg,fcut,fs)
    else:
        fgf = fg-signal.savgol_filter(fg,211,3)
    fnfg = interp.interp1d(tg,fgf)
    return tc,fn(tc),fnfg(tc)

    na = int((t[-1]-t[0])/dtest+1.5)
    ta = np.linspace(t[0],t[-1],na)
    goodlis = []
    for j in range(na):
        k = np.argmin(np.abs(t-ta[j]))
        if np.abs(t[k]-ta[j])<0.1*dtest: goodlis.append(k)
    fa = fn(ta)-fnfg(ta)
    fa -= np.mean(fa)
    
def mycorr(tt,df):
    nbins = 30*8
    autocorr = np.zeros(nbins)  # 30 bins per hour, 5 hours
    ncorr = np.zeros(nbins)  # 30 bins per hour, 5 hours
    dt = tt[1]-tt[0]
    tau = np.linspace(0,nbins*dt,nbins)
    nn = len(df)
    corrnorm = np.sum(df*df)/nn
    for j in range(nbins):
        delt = tt[j:]-tt[0:nn-j]
        df0,dfj = df[0:nn-j],df[j:]
        msk = np.abs(delt-dt*j)<0.2*dt
        df0 = df0[msk]
        dfj = dfj[msk]
        dfdf = df0*dfj
        #dfdf = (df0-np.mean(df0))*(dfj-np.mean(dfj))
        nc = np.sum(msk)
        if nc:
            autocorr[j] = np.sum(dfdf)
            autocorr[j] /= nc
            if j==0: corrnorm = autocorr[0]
            autocorr[j] /= corrnorm
    return tau,autocorr


# Plot the timeseries in black circles.
#ax.plot(tess_bjds, pdcsap_fluxes, 'ko')

t,xs,pdc = np.load('au_mic.lc.npy')

#ax.plot(t,pdc,'m-')

tt,ff,ffss = do_flares(t,pdc)
#ax.plot(tt,ff,'ok')

df = ff-ffss
df = ffss
nn = len(df)


fig, axs = plt.subplots(2, 1, constrained_layout=True)

msk = df<0.002

dfrnd = np.copy(df)
np.random.shuffle(dfrnd)
dfrnd2 = np.copy(dfrnd)
msk2 = dfrnd2 < 0.002
dfrnd2[msk2==False]=0
dfrnd2[msk==False]=df[msk==False]

if 1:
    ax = axs[1]
    tau, autocorr = mycorr(tt,df)
    ax.plot(tau,autocorr,'k.',ms=3)

    tau, autocorr = mycorr(tt[msk],df[msk])
    ax.plot(tau,autocorr,'m.',ms=3)
    
#    tau, arnd = mycorr(tt,dfrnd)
#    ax.plot(tau,arnd,'c.',ms=3)

    tau, arnd = mycorr(tt,dfrnd2)
    ax.plot(tau,arnd,'c.',ms=3)

    
    ax.set_xlabel("lag (TBJD)")
    ax.set_ylabel("Correlator")

if 1:
    ax = axs[0]
    ax.plot(tt,ff-np.mean(ff),'.',c='#aaaaaa',ms=1,zorder=0)
    ax.plot(tt,df,'k.',ms=1,zorder=20)
    ax.plot(tt[msk],df[msk],'m.',ms=1,zorder=99)
    ax.plot(tt,dfrnd2,'c.',ms=1,zorder=10)

    ax.set_ylabel("relative flux")
    ax.set_xlabel("time (TBJD)")

fig.suptitle("AU Mic Light Curve - Sector 1")


#ax.set_ylabel("PDCSAP Flux (e-/s)")

# Adjust the left margin so the y-axis label shows up.
#plt.subplots_adjust(left=0.15)


#fig.tight_layout()




plt.savefig("tmp.png")
import os
os.system("convert tmp.png ~/www/tmp.jpg")
