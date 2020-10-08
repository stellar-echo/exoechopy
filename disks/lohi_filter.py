# lightcurve_filter
import numpy as np
import scipy.interpolate as interp
import scipy.signal as signal

orderdef = 4

def butter_highpass(cutoff, fs, order=orderdef):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=orderdef):
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

def filter_lohi(T,F,tsmoo,gapfill=False):
    # splits signal F(T) into low-pass, high-pass components,
    # 1/tsmoo is the delineating frequency.
    # note: if there nans in T or F, this code does its best to
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
    t,f = T[:],F[:]
    t,f = nangapfill(t,f) # do this no matter what...
    dtest = t[1]-t[0]
    freqmax = 1/dtest
    freqcut = 1/tsmoo
    ffilter = butter_highpass_filter(f,freqcut,freqmax)
    if gapfill == False: # restore the nans
        t[np.isnan(T)] = np.nan
        msk = np.isnan(F)
        ffilter[msk] = np.nan
        f[msk] = np.nan
    return t,f-ffilter,ffilter

def fstd(f,uselower=False):
    if uselower:
        fmed = np.nanmedian(f)
        ffaint = f[f<=fmed]
        std = np.sqrt(np.mean((ffaint-fmed)**2))
    else:
        std = np.nanstd(f)
    return std

def slowfluctuations(T,F, tsmoo, gapfill = False):
    # trotationmin: stellar rotation time scale -- min value
    delt = np.nanmax(T)-np.nanmin(T)
    tdetrend = max(0.25*delt,2*tsmoo)
    if tdetrend<=tsmoo:
        print(f'can not detrend in {__name__}.lohi()')
        quit()
    td,flod,fhid = filter_lohi(T,F,tdetrend,gapfill=gapfill)
    ts,flos,fhis = filter_lohi(T,F,tsmoo,gapfill=gapfill)
    dfsmoo = flod-flos
    smed = np.nanmedian(flos)
    sstd = fstd(dfsmoo)
    return smed,sstd,ts,dfsmoo


if __name__ == "__main__" :

    import matplotlib
    import matplotlib.pyplot as plt
    import os.path
    import scipy.stats
    
    # looking for this...

    file = 'au_mic.lc.npy'
    #file = ['kepler_candidate5_time.npy','kepler_candidate5_flux.npy']
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
    fbgfluctuate = fstd(pdc,uselower=True)
    delt = t[-1]-t[0]
    
    tsmoo = [1,2,4,0.25*delt]

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
        tt,fflo,ffhi = filter_lohi(t,pdc,ts,gapfill=False)
        smed,sstd,_,_ =  slowfluctuations(t,pdc,ts)
        print(f'rel slow fluctuation amp: {sstd/smed}')
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

