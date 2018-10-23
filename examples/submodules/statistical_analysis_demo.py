

"""An example of a statistical method to produce preliminary uncertainty analysis"""

import numpy as np
from astropy import units as u
from astropy import constants
from scipy import stats

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
                               point_color='saddlebrown', mass=1*u.M_sun)

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

    approx_index_lag = round_dec(((a_1/constants.c)/observation_cadence).decompose().value)
    approx_time_lag = approx_index_lag*observation_cadence
    print("approx_index_lag: ", approx_index_lag, ", approx_time_lag: ", approx_time_lag)

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

    print("""
    In this case, half the flares hit, half the flares miss.  
    Running the autocorrelation on a flare that misses will systematically de-correlate the signal.
    """)

    flare_times = MyTelescope._all_flares.all_flare_times

    fig, ax_array = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 8))
    #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  #
    scatter_colors = ['k' if x < np.pi/2 else 'r' for x in MyTelescope._all_exoplanet_flare_angles.value]

    ax_array[0].scatter(flare_times, MyTelescope._all_exoplanet_flare_angles.to(u.deg), marker='.', color=scatter_colors)
    ax_array[0].set_ylabel("Exoplanet-flare angle (deg)")
    ax_array[0].set_title("Angles during each flare as a function of time")
    #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  #
    ax_array[1].scatter(flare_times, MyTelescope._all_exoplanet_contrasts, marker='.', color='k')
    ax_array[1].text(0.1, .7, "Half the flares hit the planet, half miss,\n"
                     "hence the two possible values",
                     transform=ax_array[1].transAxes)
    ax_array[1].set_ylabel("Echo magnitude")
    ax_array[1].set_title("Echo contrast during each flare as a function of time")
    ax_array[1].set_ylim(-.001, .002)
    #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  #
    ax_array[2].scatter(flare_times, MyTelescope._all_exoplanet_lags, marker='.', color='k')
    ax_array[2].set_ylabel("Exoplanet time lag")
    ax_array[2].set_title("Exoplanet time lag")
    ax_array[2].text(0.1, .5, "To save a little computation time,\n"
                              "lags are not computed if the exoplanet is not visible to the flare\n"
                              "(hence the zero values)",
                     transform=ax_array[2].transAxes)

    ax_array[2].set_xlabel("Time "+eep.utils.u_labelstr(flare_times, True))
    #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  #
    plt.show()

    lightcurve = MyTelescope._pure_lightcurve.copy()
    time_domain = MyTelescope._time_domain.copy()

    print("""
    Next, we process the data blindly with the autocorrelation algorithm, as usual.
    If you zoom in, you can find the echo peak!        
    The signal without noise still produces an echo, even with the decorrelation.
    """)

    max_lag = round_dec(approx_index_lag*2.5)
    autocorr = eep.analyze.autocorrelate_array(lightcurve,
                                               max_lag=max_lag)

    subplot_width = 16

    fig, ax = plt.subplots()
    autocorr_domain = np.arange(0, len(autocorr)*observation_cadence.value, observation_cadence.value)
    ax.plot(autocorr_domain, autocorr,
            color='k', lw=1, drawstyle='steps-post')
    inset_ax = ax.inset_axes([0.5, 0.5, 0.45, 0.45])
    ind_min = max(1, approx_index_lag-subplot_width//2)
    ind_max = min(len(autocorr_domain)-1, approx_index_lag+subplot_width//2)
    inset_ax.plot(autocorr_domain[ind_min:ind_max], autocorr[ind_min:ind_max],
                  color='k', lw=1, drawstyle='steps-post')
    ax.annotate("Echo is here!", xy=(approx_time_lag.value+observation_cadence.value/2, autocorr[approx_index_lag+1]),
                xytext=(approx_time_lag.value*1.1, .2), arrowprops=dict(arrowstyle="->",
                                                                        connectionstyle="arc3, rad=.3"))
    ax.set_title("Autocorrelation of lightcurve (no noise added)")
    ax.set_xlabel("Time lag "+eep.utils.u_labelstr(observation_cadence))
    ax.set_ylabel("Correlation signal")
    plt.tight_layout()
    plt.show()

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    print("""
    Now that it works without noise, we'll try again with counting noise in the background.""")

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
    ind_min = max(1, approx_index_lag-subplot_width//2)
    ind_max = min(len(autocorr_domain)-1, approx_index_lag+subplot_width//2)
    inset_ax.plot(autocorr_domain[ind_min:ind_max], autocorr[ind_min:ind_max],
                  color='k', lw=1, drawstyle='steps-post')
    ax.annotate("Echo is here!", xy=(approx_time_lag.value+observation_cadence.value/2, autocorr[approx_index_lag+1]),
                xytext=(approx_time_lag.value*1.1, .2), arrowprops=dict(arrowstyle="->",
                                                                        connectionstyle="arc3, rad=.3"))
    ax.set_title("Autocorrelation of lightcurve (Poisson noise added)")
    ax.set_xlabel("Time lag "+eep.utils.u_labelstr(observation_cadence))
    ax.set_ylabel("Correlation signal")
    plt.show()

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

    flare_indices = eep.analyze.find_peaks_stddev_thresh(noisy_signal-np.mean(noisy_signal),
                                                         std_dev_threshold=2,
                                                         min_index_gap=approx_index_lag*2,
                                                         extra_search_pad=4)

    print("""
    Next, we pick out each flare and process it individually to reduce the 
    signal from the quiescent background.""")

    num_flares = len(flare_indices)

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
    ax.annotate("Echo is around here", xy=(autocorr_domain[approx_index_lag]+observation_cadence.value/2,
                                           detrended_weighted_correlation[approx_index_lag-1] + .00005),
                xytext=(autocorr_domain[approx_index_lag]*1.1, max(detrended_correlation[1:max_lag-1])),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.3"))
    ax.set_title("Autocorrelation of noisy lightcurve (effects of data weight)", y=1.09)
    ax.set_xlabel("Time lag "+eep.utils.u_labelstr(observation_cadence))
    ax.set_ylabel("Correlation signal")
    plt.legend(bbox_to_anchor=(0., 1.01), loc=3, ncol=2, borderaxespad=0.)
    plt.subplots_adjust(left=.17, top=.85)
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
    min_lag_offset = 1

    for f_i, flare_index in enumerate(flare_indices):
        if flare_index-back_index > 0 and flare_index+forward_index < len(noisy_signal):
            flare_curve = noisy_signal[flare_index - back_index:flare_index + forward_index]
            this_autocorr = eep.analyze.autocorrelate_array(flare_curve, max_lag=max_lag, min_lag=min_lag_offset)
            this_autocorr = linear_detrend(this_autocorr)
            all_autocorr[f_i] = this_autocorr
            # Add a magnitude-based weight to the autocorrelation (more weight to brighter flares)
            weights[f_i] = np.max(flare_curve)

    weights /= np.sum(weights)

    print("""
    Excepting exceptional signal-to-noise cases, the histogram of points will not reveal bimodality willingly.
    """)

    test_indices = [approx_index_lag-2, approx_index_lag-1, approx_index_lag, approx_index_lag+1, approx_index_lag+2]
    fig, ax_array = plt.subplots(1, len(test_indices), figsize=(12, 5), sharex=True, sharey=True)

    num_bins = 50
    x_vals = np.linspace(np.min(all_autocorr), np.max(all_autocorr), num_bins)
    for c_i, current_index in enumerate(test_indices):
        if current_index == approx_index_lag:
            ax_array[c_i].text(.05, .95, "Echo signal", transform=ax_array[c_i].transAxes)
        ax_array[c_i].hist(all_autocorr[:, current_index-min_lag_offset],
                           weights=weights, color='gray', bins=num_bins//2,
                           zorder=0, density=True, label="matplotlib histogram")
        #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  #
        sigma = np.std(all_autocorr[:, current_index-min_lag_offset])
        tophat_kde = eep.analyze.TophatKDE(dataset=all_autocorr[:, current_index-min_lag_offset],
                                           bandwidth=sigma, weights=weights)
        ax_array[c_i].plot(x_vals, tophat_kde(x_vals),
                           drawstyle='steps-post', color='k', lw=1, label="Tophat kernel density estimate")
        #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  #
        gauss_kde = eep.analyze.GaussianKDE(dataset=all_autocorr[:, current_index-min_lag_offset],
                                            bandwidth='silverman')
        ax_array[c_i].plot(x_vals, gauss_kde(x_vals), color='r', lw=1, label="Gaussian kernel density estimate")
        #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  #
        ax_array[c_i].set_title("Lag: " +
                                str(current_index*observation_cadence.value)+eep.utils.u_labelstr(observation_cadence))
        ax_array[c_i].set_xlabel("Correlation value")

    ax_array[0].set_ylabel("Counts")
    ax_array[0].legend(bbox_to_anchor=(0., 1.1), loc=3, ncol=3, borderaxespad=0.)
    plt.subplots_adjust(left=.1, right=.98, top=.85, wspace=0)
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
