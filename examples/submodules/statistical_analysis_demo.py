

"""An example of a statistical method to produce preliminary uncertainty analysis"""

import numpy as np
from astropy import units as u
from astropy import constants
from scipy import stats, signal

import exoechopy as eep
from exoechopy.utils.math_operations import *
from exoechopy.utils.constants import *
import matplotlib.pyplot as plt


def run():

    print("""
    It's one thing to identify an echo, and another to quantify the uncertainty of the detection.
    This demo provides an overview of a histogram-type analysis.
    """)

    telescope_area = np.pi * (4*u.m)**2
    total_efficiency = 0.8
    observation_cadence = 2. * u.s
    observation_duration = 12 * u.hr
    telescope_random_seed = 99
    approximate_num_flares = 200

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    #  Create an observation target:
    spectral_band = eep.simulate.spectral.JohnsonPhotometricBand('U')
    emission_type = eep.simulate.spectral.SpectralEmitter(spectral_band, magnitude=16)

    # Create a star that is effectively a delta-function, but has radius effects like variable flare-planet visibility.
    MyStar = eep.simulate.Star(radius=1*u.m, spectral_type=emission_type, rotation_rate=8*pi_u/u.d,
                               point_color='saddlebrown')

    # Face-on circular
    MyStar.set_view_from_earth(0*u.deg, 0*u.deg)

    #  =============================================================  #
    #  Create a planet to orbit the star
    planet_albedo = eep.simulate.spectral.Albedo(spectral_band, 1.)
    planet_radius = 2*u.R_jup
    e_1 = 0.  # eccentricity
    a_1 = 0.04 * u.au  # semimajor axis
    i_1 = 0 * u.deg  # inclination
    L_1 = 0 * u.deg  # longitude
    w_1 = 0 * u.deg  # arg of periapsis
    m0_1 = 0 * u.deg  # initial anomaly

    approx_index_lag = int((a_1/constants.c)/observation_cadence)
    print("approx_index_lag: ", approx_index_lag)

    planet_name = "HelloExoWorld"
    print("Approximate time delay: ", (a_1/constants.c).to(u.s))

    Planet1 = eep.simulate.KeplerianExoplanet(semimajor_axis=a_1,
                                              eccentricity=e_1,
                                              inclination=i_1,
                                              longitude=L_1,
                                              periapsis_arg=w_1,
                                              initial_anomaly=m0_1,
                                              albedo=planet_albedo,
                                              radius=planet_radius,
                                              point_color='k', path_color='dimgray',
                                              name=planet_name)
    MyStar.add_orbiting_object(Planet1)
    print("Planet1.orbital_period: ", Planet1.orbital_period.to(u.d))

    eep.visualize.render_3d_planetary_system(MyStar)

    #  =============================================================  #

    #  Approx. max flare intensity:
    max_intensity = 10*MyStar.get_flux()
    min_intensity = max_intensity/10

    #  To explicitly specify units, use a tuple of (pdf, unit):
    flare_intensities = ([min_intensity, max_intensity], max_intensity.unit)

    MyFlareActivity = eep.simulate.active_regions.FlareActivity(eep.simulate.flares.DeltaFlare,
                                                                intensity_pdf=flare_intensities)

    # For a first test, we don't want it to get too complicated,
    # so we'll use a probability distribution that prevents overlapping flares
    scale_factor = 2/approximate_num_flares*observation_duration.to(u.s).value
    occurrence_freq_pdf = stats.uniform(loc=20*approx_index_lag/observation_cadence.value,
                                        scale=scale_factor-20*approx_index_lag/observation_cadence.value)

    flare_location = eep.simulate.Region([0, 2*np.pi], np.pi/3)
    Starspot = eep.simulate.ActiveRegion(flare_activity=MyFlareActivity,
                                         occurrence_freq_pdf=occurrence_freq_pdf,
                                         region=flare_location)

    MyStar.add_active_regions(Starspot)

    #  =============================================================  #
    MyTelescope = eep.simulate.Telescope(collection_area=telescope_area,
                                         efficiency=total_efficiency,
                                         cadence=observation_cadence,
                                         observation_target=MyStar,
                                         random_seed=telescope_random_seed,
                                         name="My Telescope")

    MyTelescope.prepare_continuous_observational_run(observation_duration, print_report=True)
    MyTelescope.collect_data(save_diagnostic_data=True)

    print("In this case, half the flares hit, half the flares miss."
          "Running the autocorrelation on a flare that misses will systematically de-correlate the signal."
          )

    flare_times = MyTelescope._all_flares.all_flare_times
    plt.scatter(flare_times, MyTelescope._all_exoplanet_contrasts, marker='.', color='k')
    plt.show()

    print("Exoplanet-star contrast: ", MyTelescope._all_exoplanet_contrasts[0][0])
    print("Exoplanet lag: ", MyTelescope._all_exoplanet_lags[0][0]*u.s)



    lightcurve = MyTelescope._pure_lightcurve.copy()
    time_domain = MyTelescope._time_domain.copy()

    print("""
    Next, we process the data blindly with the autocorrelation algorithm, as usual.
    If you zoom in, you can find the echo peak!        
    The signal without noise still produces an echo, even with the decorrelation.""")

    max_lag = approx_index_lag*3
    autocorr = eep.analyze.autocorrelate_array(lightcurve,
                                               max_lag=max_lag)

    subplot_width = 16

    fig, ax = plt.subplots()
    autocorr_domain = np.arange(0, len(autocorr)*observation_cadence.value, observation_cadence.value)
    ax.plot(autocorr_domain, autocorr,
            color='k', lw=1, drawstyle='steps-post')
    inset_ax = ax.inset_axes([0.5, 0.5, 0.45, 0.45])
    ind_min = approx_index_lag-subplot_width//2
    ind_max = approx_index_lag+subplot_width//2
    inset_ax.plot(autocorr_domain[ind_min:ind_max], autocorr[ind_min:ind_max],
                  color='k', lw=1, drawstyle='steps-post')
    plt.show()

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    print("""
    Now that it works without noise, we'll try again with counting noise in the background.
    """)

    noisy_signal = eep.simulate.methods.add_poisson_noise(lightcurve)

    print("""
    The signal is now decorrelated AND buried in the noise.  
    Much more effort is required to convincingly extract the echo.  
    """)

    autocorr = eep.analyze.autocorrelate_array(noisy_signal,
                                               max_lag=max_lag)

    fig, ax = plt.subplots()
    autocorr_domain = np.arange(0, len(autocorr)*observation_cadence.value, observation_cadence.value)
    ax.plot(autocorr_domain, autocorr,
            color='k', lw=1, drawstyle='steps-post')
    inset_ax = ax.inset_axes([0.5, 0.5, 0.45, 0.45])
    ind_min = approx_index_lag-subplot_width//2
    ind_max = approx_index_lag+subplot_width//2
    inset_ax.plot(autocorr_domain[ind_min:ind_max], autocorr[ind_min:ind_max],
                  color='k', lw=1, drawstyle='steps-post')
    plt.show()

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

    flare_indices = eep.analyze.find_peaks_stddev_thresh(noisy_signal-np.mean(noisy_signal),
                                                         std_dev_threshold=2,
                                                         min_index_gap=approx_index_lag*2,
                                                         extra_search_pad=4)

    print("Next, we pick out each flare and processing it individually to reduce the "
          "signal from the quiescent background.")

    num_flares = len(flare_indices)

    # if num_flares > 0:
    #     eep.visualize.plot_flare_array(noisy_signal, flare_indices,
    #                                    back_pad=subplot_width//2, forward_pad=subplot_width*3,
    #                                    display_index=True)

    back_index = 40
    forward_index = back_index*3
    sum_autocorr_noisy = np.zeros(max_lag+1)
    sum_autocorr_weighted = np.zeros(max_lag+1)
    ct = 0
    total_weight = 0
    for f_i, flare_index in enumerate(flare_indices):
        if flare_index-back_index > 0 and flare_index+forward_index < len(noisy_signal):
            flare_curve = noisy_signal[flare_index - back_index:flare_index + forward_index]
            this_autocorr = eep.analyze.autocorrelate_array(flare_curve, max_lag=max_lag)
            sum_autocorr_noisy += this_autocorr
            # Add a magnitude-based weight to the autocorrelation (more weight to brighter flares)
            sum_autocorr_weighted += this_autocorr*np.max(flare_curve)
            total_weight += np.max(flare_curve)
            ct += 1

    sum_autocorr_noisy /= ct
    sum_autocorr_weighted /= total_weight
    print("Found "+str(ct)+" flares for analysis")

    detrended_correlation = linear_detrend(sum_autocorr_noisy[1:max_lag-1])
    detrended_weighted_correlation = linear_detrend(sum_autocorr_weighted[1:max_lag-1])

    fig, ax = plt.subplots()
    autocorr_domain = np.arange(0, len(sum_autocorr_noisy)*observation_cadence.value, observation_cadence.value)
    ax.plot(autocorr_domain[1:max_lag - 1], detrended_correlation,
            color='gray', lw=1, drawstyle='steps-post', label="Summed autocorrelation")
    ax.plot(autocorr_domain[1:max_lag - 1], detrended_weighted_correlation,
            color='k', lw=1, drawstyle='steps-post', label="Summed weighted autocorrelation")

    plt.legend()
    plt.show()

    print("""
    Here, the decorrelating effect suppresses the signal, unless a VERY large sample is available.
    Instead, it is necessary to subsample the flares.  
    In this analysis, we assume that the flares occurred randomly on the surface, so the probability of hitting
    the planet was ~50%, but no two flares are correlated (even if the flare generator did have correlation)
    
    The key insight is this: at the flare lag, there is is a bimodal distribution of echo intensities.
    At all other lags, there is a unimodal distribution.  The ability to isolate the distribution can be estimated by
    the sensitivity index.  
    
    So instead, we don't just look at the flares autocorrelation signal: 
    it is also necessary to evaluate the distribution of correlation signatures.
    """)

    all_autocorr = np.zeros((num_flares, max_lag))
    weights = np.zeros(num_flares)

    for f_i, flare_index in enumerate(flare_indices):
        if flare_index-back_index > 0 and flare_index+forward_index < len(noisy_signal):
            flare_curve = noisy_signal[flare_index - back_index:flare_index + forward_index]
            this_autocorr = eep.analyze.autocorrelate_array(flare_curve, max_lag=max_lag, min_lag=1)
            this_autocorr = linear_detrend(this_autocorr)
            all_autocorr[f_i] = this_autocorr
            # Add a magnitude-based weight to the autocorrelation (more weight to brighter flares)
            weights[f_i] = np.max(flare_curve)

    weights /= np.sum(weights)

    print("""Excepting exceptional signal-to-noise cases, the histogram of points will not reveal bimodality willingly.
    """)

    plt.hist(all_autocorr[:, approx_index_lag],
             weights=weights, color='gray', bins=50, zorder=0, density=True, label="matplotlib histogram")

    sigma = np.std(all_autocorr[:, approx_index_lag])
    x_vals, y_vals = produce_tophat_kde(x_min=np.min(all_autocorr), x_max=np.max(all_autocorr),
                                        data_for_analysis=all_autocorr[:, approx_index_lag], bandwidth=sigma,
                                        data_weights=weights)
    plt.plot(x_vals, y_vals, color='k', lw=1, label="KDE at echo")

    plt.legend()
    plt.show()

    print("""
    However, we have a hypothesis that we can test: If half the flares hit and half miss, 
    then we technically only need half the data to get the signal out.
    But we don't know which flares are the right ones, and it's dangerous to cherry pick them.
    Instead, a variety of resampling techniques, like bootstrapping and jackknifing, are more appropriate.
    While we haven't yet determined the best approach to performing this analysis, here's one example of how to proceed:
    
    - Subsample the data by only including half of the data points
    - Repeat pseudo-exhaustively, going through a very large number of subsamples (or all, if computationally feasible)
    - For each subsample, repeat the analysis
    - Analyze the distribution of subsample results somehow.  There are a variety of analysis options available.
    
    There will be one perfect subsample that contains all the hits, and one that contains all the misses.
    Everything else will be somewhere in between.  

    To be continued...    
    """)



# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #


if __name__ == "__main__":

    run()
