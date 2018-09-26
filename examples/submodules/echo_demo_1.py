

"""Shows how to identify echoes through examples."""

import numpy as np
from astropy import units as u
from astropy import constants
from scipy import stats, signal

import exoechopy as eep
from exoechopy.utils.math_operations import *
from exoechopy.visualize.autocorr_plots import *
import matplotlib.pyplot as plt


def run():

    print("""
    Detecting echoes is hard--we start by showing what results look like using artifically optimistic values.
    First, create a delta-function active star (zero radius), give it really clear delta-function flares,
    and give it a big bright planet.  If you can't detect it here, you can't detect it with more realistic parameters.
    """)

    telescope_area = np.pi * (4*u.m)**2
    total_efficiency = 0.8
    observation_cadence = 1. * u.s
    observation_duration = 4 * u.hr
    telescope_random_seed = 99
    approximate_num_flares = 75

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    #  Create an observation target:
    spectral_band = eep.simulate.spectral.JohnsonPhotometricBand('U')
    emission_type = eep.simulate.spectral.SpectralEmitter(spectral_band, magnitude=16)

    MyStar = eep.simulate.DeltaStar(spectral_type=emission_type, point_color='saddlebrown')

    # Face-on circular
    MyStar.set_view_from_earth(0*u.deg, 0*u.deg)

    #  =============================================================  #
    #  Create a planet to orbit the star
    planet_albedo = eep.simulate.spectral.Albedo(spectral_band, 1.)
    planet_radius = 2*u.R_jup
    e_1 = 0.  # eccentricity
    a_1 = 0.03 * u.au  # semimajor axis
    i_1 = 0 * u.deg  # inclination
    L_1 = 0 * u.deg  # longitude
    w_1 = 0 * u.deg  # arg of periapsis
    m0_1 = 0 * u.deg  # initial anomaly

    approx_index_lag = int((a_1/constants.c)/observation_cadence)

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
    MyStar.add_exoplanet(Planet1)

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
    Starspot = eep.simulate.active_regions.ActiveRegion(flare_activity=MyFlareActivity,
                                                        occurrence_freq_pdf=occurrence_freq_pdf)

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

    print("Exoplanet-star contrast: ", MyTelescope._all_exoplanet_contrasts[0][0])
    print("Exoplanet lag: ", MyTelescope._all_exoplanet_lags[0][0]*u.s)

    print("""
    The echoes are typically extremely small, even in this case where we haven't introduced noise yet.
    First, see if you can find them in the plot--they should occur near 10s after the flare peak.
    In this scenario, with a perfect albedo on a super jupiter that is incredibly close to its star,
    the echo intensity is <0.001 times the flare strength.
    """)

    lightcurve = MyTelescope._pure_lightcurve.copy()
    time_domain = MyTelescope._time_domain.copy()
    eep.visualize.interactive_lightcurve(time_domain, lightcurve)

    print("""
    Next, process the data blindly with the autocorrelation algorithm, because...why not try it.
    If you zoom in, you can find the echo peak!""")

    max_lag = approx_index_lag*3
    autocorr = eep.analyze.autocorrelate_array(lightcurve,
                                               max_lag=max_lag)

    subplot_width = 10

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

    eep.visualize.interactive_lightcurve(time_domain, noisy_signal)

    print("""
    The signal is often buried in the noise (depending on the noise seed).  
    More effort is required to convincingly extract the echo.  
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

    print("By picking out each flare and processing it individually, "
          "the signal from the quiescent background is suppressed.")

    num_flares = len(flare_indices)
    if num_flares > 0:
        eep.visualize.plot_flare_array(noisy_signal, flare_indices,
                                       back_pad=subplot_width//2, forward_pad=subplot_width*3,
                                       display_index=True)

    back_index = 40
    forward_index = back_index*3
    sum_autocorr_noisy = np.zeros(max_lag+1)
    ct = 0
    for flare_index in flare_indices:
        if flare_index-back_index > 0 and flare_index+forward_index < len(noisy_signal):
            this_autocorr = eep.analyze.autocorrelate_array(noisy_signal[flare_index -
                                                                  back_index:flare_index+forward_index],
                                                                  max_lag=max_lag)
            # plt.plot(this_autocorr[1:], color='k', lw=1, drawstyle='steps-post')
            # plt.show()
            sum_autocorr_noisy += this_autocorr
            ct += 1

    sum_autocorr_noisy /= ct

    detrended_correlation = linear_detrend(sum_autocorr_noisy[1:max_lag-1])

    fig, ax = plt.subplots()
    autocorr_domain = np.arange(0, len(sum_autocorr_noisy)*observation_cadence.value, observation_cadence.value)
    ax.plot(autocorr_domain[1:max_lag-1], detrended_correlation,
            color='k', lw=1, drawstyle='steps-post')
    # inset_ax = ax.inset_axes([0.5, 0.5, 0.45, 0.45])
    # ind_min = approx_index_lag-subplot_width//2
    # ind_max = approx_index_lag+subplot_width//2
    # inset_ax.plot(autocorr_domain[ind_min:ind_max], detrended_correlation[ind_min:ind_max],
    #               color='k', lw=1, drawstyle='steps-post')
    plt.show()

    print("""
    Ultimately, the echo starts to emerge even in the noisy data, but only barely.  
    More flares or a bigger telescope are needed to improve the signal-to-noise ratio.
    This is where the art of stellar echo starts: 
     - How can we better identify flare echoes?
     - How can we determine if these small peaks are real?
     - How can we place confidence intervals on the data? 
    The purpose of this library is to open-source our research to help others develop methods 
    for extracting faint echoes from the light curves of active stars.
    """)

# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #


if __name__ == "__main__":

    run()
