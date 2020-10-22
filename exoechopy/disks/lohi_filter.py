'''
 lohi_filter
 
   routines to do butterworth filtering on flux data with
   nan-masked data. Expect ordered, uniformly spaced elements
   in time arrays T, with corresponding flux F
 
   main software products:
   
   slowfluctuations(T,F, tsmoo): estimates slow, detrended rms fluctuations
   lohi_filter(T, F, tsmoo, flarethresh=1e99, gapfill=False): use this!
         returns lowpass, hipass filtered flux with (limited) flare masking

   internal, utility routines:

   butter_highpass(cutoff, fs, order=orderdef): the filter
   butter_highpass_filter(data, cutoff, fs, order=orderdef): does filtering
   nangapfill(T,F): fills in nan masked regions in time, flux arrays
   hipass_filter_basic(t,f,tcut): does high pass, freq=1/tcut
   lohi_filter_nangap(T,F,tsmoo,gapfill=False): does lo+hi filtering
   passband_filter(T,F, tshort, tlong): does passband...

   globals:

   orderdef: order of the Butterworth filter



 bcb 2020
'''

import numpy as np
import scipy.interpolate as interp
import scipy.signal as signal

orderdef = 4

def butter_highpass(cutoff, fs, order=orderdef):
    global orderdef
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=orderdef):
    global orderdef
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

# build in bg amplitude maybe fitting a sine ?????

def nangapfill(T,F):
    # fills in "gaps" in t, f arrays (t=time grid, f=flux) that 
    # nan values.
    msk = np.isnan(T) # do we have gaps in the time sequence?
    if np.sum(msk):   # if so, fill them in...problems if nans are on ends
        dtest = min((T[1:]-T[:-1]))
        t = np.linspace(0,dtest*(len(T)-1),len(T))
        itmin,tmin = np.nanargmin(T),min(T)
        t = t + tmin - dtest*itmin # guess t start if first few are bad
    else: t = T
    msk = np.isnan(F) # do we have nans in the light curve data
    f = F
    if np.sum(msk)>0: # if yes, interp to nearest good point
        f[msk] = np.interp(np.flatnonzero(msk), np.flatnonzero(~msk), f[~msk])
    return t,f


def hipass_filter_basic(t,f,tsmoo):
    # all is expected to be good here:
    # t: array of sampling times, uniform
    # f: array of samples (same size as t)
    # tsmoo: smoothing time scale
    dtest = t[1]-t[0]
    freqmax = 1/dtest
    freqcut = 1/tsmoo
    ffilter = butter_highpass_filter(f,freqcut,freqmax)
    return ffilter


def lohi_filter_nangap(T,F,tsmoo,gapfill=False):

    # splits signal F(T) into low-pass, high-pass components,
    # 1/tsmoo is the delineating frequency.
    # note: if there are nans in T or F, this code does its best to
    # fill in the gaps.
    #
    # T: array, uniformly sampled times, expect equally spaced.
    # F: array, corresponding flux samples
    # gapfill: if true, return arrays with "nan gap" values filled in
    #
    # returns:  t,flo,fhi
    # t: array with times (including "gaps" if gapfill==True)
    # flo, fhi: arrays with lo, hi-pass filtered data

    dtest = T[1]-T[0]  # we will need delta_T
    t = np.copy(T)
    f = np.copy(F)
    t,f = nangapfill(t,f) # do this no matter what...
    ffilter = hipass_filter_basic(t,f,tsmoo)
    if gapfill == False: # restore the nans
        t[np.isnan(T)] = np.nan
        msk = np.isnan(F)
        ffilter[msk] = np.nan
        f[msk] = np.nan
    return t,f-ffilter,ffilter


def passband_filter(T,F, tshort, tlong):
    # "Nan-gap"-compliant pass-band filter.
    # T,F: arrays, (orderedm uniformly sampled) time and flux.
    # tshort, tlong are times that define the pass band.  freq=1/time,
    td,flod,fhid = lohi_filter_nangap(T,F,tlong)
    ts,flos,fhis = lohi_filter_nangap(T,F,tshort)
    fpass = flod-flos
    return td,fpass

def slowfluctuations(T,F, tsmoo):
    #
    # assessment of the fluctuations in the lightcurve F as function of T
    # in a passband that runs from frequency range  (0.25--1)1*1/tsmoo
    # or 4/(T[-1]-T[0]), whichever is broader.
    # 
    # the idea is to detrend on a large a timescale tdetrend=((T[-1]-T[0])/4)
    # and then delineate slow and fast fluctuations rel to tsmoo.
    # Returns:
    # stddev: std dev = rms of passband-filtered signal (1/tdetrend,1/tsmoo)
    # 
    delt = np.nanmax(T)-np.nanmin(T)
    tdetrend = 0.25*delt
    if tdetrend<=tsmoo:
        pass
        #print(f'can not detrend in {__name__}.lohi()')
        #quit()
    td, dfsmoo = passband_filter(T,F,tsmoo,tdetrend)
    sstd = np.nanstd(dfsmoo)
    return sstd

def lohi_filter(T, F, tsmoo, flarethresh=1e99, gapfill=False):
    #
    # extracts lopass and hipass signals with a crude flare removal.
    # T: array of sample times, uniformly spaced, may contain nans
    # F: array of samples, may contain nans
    # flarethresh: z-score of flux peaks that are nan'd out in
    #     the final lopass calculation
    #     mean and stddev 
    #
    ts,flos,fhis = lohi_filter_nangap(T,F,tsmoo,gapfill=gapfill)
    if (flarethresh < 1e33):
        if (gapfill == False):
            mskgap = np.isnan(F)
        fstd = np.nanstd(fhis)
        fmean = np.nanmean(fhis)
        msk = fhis>flarethresh*fstd
        Fclip = np.copy(F)
        Fclip[msk]=np.nan
        ts,flos,fhis = lohi_filter_nangap(T,Fclip,tsmoo,gapfill=True)
        if (gapfill == False):
            flos[mskgap] = F[mskgap]
            ts = T
    return ts, flos, F-flos

 
if __name__ == "__main__" :

    import matplotlib
    import matplotlib.pyplot as plt
    import os.path
    import scipy.stats
    
    # looking for this...

    file = 'au_mic.lc.npy'
    file = ['kepler_candidate5_time.npy','kepler_candidate5_flux.npy']
    if isinstance(file,list):
        t = np.load(file[0])
        pdc = np.load(file[1])
    elif os.path.isfile(file):
        t, xs, pdc = np.load(file)
    else:
        t = np.arange(0,30,0.01)
        xs = np.sin(t*2*np.pi/3) + 5.0 + np.random.poisson(0.01,len(t))
        pdc = xs[:]

    fmed = np.nanmedian(pdc)
    fbgfluctuate = np.nanstd(pdc)
    delt = t[-1]-t[0]
    
    tsmoo = [0.5,1,2,4]
    fthresh=3.0 # clips flares at this z-score when filtering 
    # fthresh=1e99

    # plot stuff
    fig, ax = plt.subplots(1, 1, constrained_layout=True)  #  was set for 2x1
    col = ['orange','blue','magenta','green','purple']
    plt.xlim(t[0]-0.05*delt,t[-1]+0.1*delt)
    plothipass = True
    #plothipass = False

    ax.set_xlabel("time (TBJD)")
    ax.set_ylabel("flux")
    # ax.plot(tt,ff,'.',c='k',ms=0.25,zorder=999)
    xoff = 0.02*delt # label offsets
    dyoff = 5*fbgfluctuate
    yoff = len(tsmoo) * dyoff # data curve offsets
    y = pdc-fmed if (plothipass) else pdc
    label = 'raw'
    ax.plot(t,y+yoff,'-',c='k',zorder=999,label=label)    
    ax.text(t[-1]+xoff,np.nanmedian(y+yoff),label)
    #plt.xlim(0,10)
    for i,ts in enumerate(tsmoo):
        tt,fflo,ffhi = lohi_filter(t,pdc,ts,flarethresh=fthresh,gapfill=False)
        relslowstd =  slowfluctuations(t,pdc,ts) / np.nanmean(pdc)
        print(f'rel stellar fluctuations (scale {ts} d): {relslowstd}')
        yoff = i * dyoff
        y = ffhi if (plothipass) else fflo
        label = f'{ts} d'
        ax.plot(tt,y + yoff,'-',color=col[i],zorder=99+i,label=label)
        ax.text(t[-1]+xoff,np.nanmedian(y+yoff),label)

    if 'au_mic' in file: fig.suptitle("AU Mic Light Curve - Sector 1")

    plt.savefig("tmp.png")
    import os
    #os.system("cp tmp.pdf ~/www/tmp.pdf")
    os.system("convert tmp.png ~/www/tmp.jpg")

