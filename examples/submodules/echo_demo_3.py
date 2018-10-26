

"""Explores generation of echoes with non-Keplerian objects."""

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
    This module repeats echo_demo_2, but with the symplectic solver and a star with a radius.
    """)

    telescope_area = np.pi * (4*u.m)**2
    total_efficiency = 0.8
    observation_cadence = .5 * u.s
    observation_duration = 4 * u.hr
    telescope_random_seed = 99
    approximate_num_flares = 35

    # Solver parameters (NOT the cadence, this is for the position integration):
    solver_timestep = 5 * u.s
    # Note, if the timestep is longer than the observation cadence, the telescope will interpolate the position

    spectral_band = eep.simulate.spectral.JohnsonPhotometricBand('U')
    emission_type = eep.simulate.spectral.SpectralEmitter(spectral_band, magnitude=16)
    star_mass = 1 * u.M_sun
    star_radius = 1*u.R_sun

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    #  Create a planet
    planet_albedo = eep.simulate.spectral.Albedo(spectral_band, 1.)
    planet_radius = 2*u.R_jup
    e_1 = 0.  # eccentricity
    a_1 = 0.06 * u.au  # semimajor axis
    i_1 = 0 * u.deg  # inclination
    L_1 = 0 * u.deg  # longitude
    w_1 = 0 * u.deg  # arg of periapsis
    m0_1 = 0 * u.deg  # initial anomaly
    planet_mass = 10. * u.M_jup

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
                                              mass=planet_mass,
                                              point_color='k', path_color='dimgray',
                                              name=planet_name,
                                              star_mass=planet_mass+star_mass)

    #  =============================================================  #
    #  Create a star
    MyStar = eep.simulate.Star(radius=star_radius,
                               spectral_type=emission_type,
                               limb=eep.simulate.Limb('quadratic', coeffs=[.1, .3]),
                               point_color='saddlebrown',
                               mass=star_mass)

    # Pole-on view
    MyStar.set_view_from_earth(0*u.deg, 0*u.deg)

    MyStar.add_orbiting_object(Planet1)

    # Validate that the orbit is what we thought it would be:
    eep.visualize.render_3d_planetary_system(MyStar)

    #  Approx. max flare intensity:
    max_intensity = 10*MyStar.get_flux()
    min_intensity = max_intensity/10

    #  To explicitly specify units, use a tuple of (pdf, unit):
    flare_intensities = ([min_intensity, max_intensity], max_intensity.unit)

    MyExpFlareActivity = eep.simulate.active_regions.FlareActivity(eep.simulate.flares.ExponentialFlare1,
                                                                   intensity_pdf=flare_intensities,
                                                                   onset_pdf=[1, 5] * u.s,
                                                                   decay_pdf=(stats.rayleigh(scale=6), u.s))

    # For a first test, we don't want it to get too complicated,
    # so we'll use a probability distribution that prevents overlapping flares
    scale_factor = 2/approximate_num_flares*observation_duration.to(u.s).value
    occurrence_freq_pdf = stats.uniform(loc=50/observation_cadence.value,
                                        scale=scale_factor-50/observation_cadence.value)
    Starspot = eep.simulate.active_regions.ActiveRegion(flare_activity=MyExpFlareActivity,
                                                        occurrence_freq_pdf=occurrence_freq_pdf,
                                                        region=eep.simulate.Region())

    MyStar.add_active_regions(Starspot)

    #  =============================================================  #
    #  Pass these on to the numerical solver:
    planet_pos, star_pos, planet_vel, star_vel = eep.simulate.reduced_keplerian_to_2body(Planet1,
                                                                                         planet_mass,
                                                                                         star_mass)

    Planet1.position = planet_pos
    Planet1.velocity = planet_vel
    MyStar.position = star_pos
    MyStar.velocity = star_vel
    y6solver = eep.simulate.SymplecticSolver(MyStar, Planet1, dt=solver_timestep, steps_per_save=100)

    print("Planet1.position: ", Planet1.position)
    print("MyStar.position: ", MyStar.position)
    print("Planet1.velocity: ", Planet1.velocity)
    print("MyStar.velocity: ", MyStar.velocity)

    #  =============================================================  #
    #  Initialize the telescope:
    MyTelescope = eep.simulate.Telescope(collection_area=telescope_area,
                                         efficiency=total_efficiency,
                                         cadence=observation_cadence,
                                         observation_target=MyStar,
                                         random_seed=telescope_random_seed,
                                         name="My Telescope")

    MyTelescope.prepare_continuous_observational_run(observation_duration,
                                                     print_report=True,
                                                     solver=y6solver)
    MyTelescope.collect_data(save_diagnostic_data=True)

    print("Exoplanet-star contrast: ", MyTelescope._all_exoplanet_contrasts[0][0])
    print("Exoplanet lag: ", MyTelescope._all_exoplanet_lags[0][0]*u.s)

    print("""
    The echoes are typically extremely small, even in this case where we haven't introduced noise yet.
    First, see if you can find them in the plot--(yeah, they're invisible).
    In this scenario, with a perfect albedo on a super jupiter that is incredibly close to its star,
    the echo intensity is <0.001 times the flare strength.
    """)

    lightcurve = MyTelescope._pure_lightcurve
    time_domain = MyTelescope._time_domain
    eep.visualize.interactive_lightcurve(time_domain, lightcurve)

    print("""
    Next, process the data blindly with the autocorrelation algorithm, because it worked before.""")

    max_lag = approx_index_lag*3
    autocorr = eep.analyze.autocorrelate_array(lightcurve,
                                               max_lag=max_lag)

    subplot_width = 60

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

    print("""
    Anything useful would be hidden on the back of the autocorrelation, so we need to process the autocorrelation.
    Here are a couple filters that are interesting to consider...
    """)

    plot_autocorr_with_derivatives(autocorr, autocorr_domain,
                                   lag_index_spotlight=approx_index_lag, lag_index_spotlight_width=40)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    print("""
    Unfortunately, the signal is quite well hidden.  
    Instead, we can pre-process the lightcurve to accentuate rapid changes, like the peak of the flare.
    """)

    filter_kernel = signal.gaussian(40, std=3)
    filter_kernel /= np.sum(filter_kernel)
    filtered_signal = lightcurve.value \
                      - signal.fftconvolve(lightcurve.copy().value, filter_kernel, mode='same')

    eep.visualize.interactive_lightcurve(time_domain, filtered_signal)

    print("""
    Now, the autocorrelation will be more effective:
    """)

    autocorr = eep.analyze.autocorrelate_array(filtered_signal,
                                               max_lag=max_lag)
    plot_autocorr_with_derivatives(autocorr, autocorr_domain,
                                   lag_index_spotlight=approx_index_lag, lag_index_spotlight_width=40,
                                   deriv_window=7)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    print("""
        Now that it works without noise, we'll try again with counting noise in the background.
        """)

    noisy_signal = eep.simulate.methods.add_poisson_noise(lightcurve)

    eep.visualize.interactive_lightcurve(time_domain, noisy_signal)

    filtered_noisy_signal = noisy_signal - signal.fftconvolve(noisy_signal, filter_kernel, mode='same')

    eep.visualize.interactive_lightcurve(time_domain, filtered_noisy_signal)

    autocorr = eep.analyze.autocorrelate_array(filtered_noisy_signal,
                                               max_lag=max_lag)

    print("""
    Unfortunately, the signal is now buried in the noise.  More effort is required to convincingly extract the echo.  
    """)

    plot_autocorr_with_derivatives(autocorr, autocorr_domain,
                                   lag_index_spotlight=approx_index_lag, lag_index_spotlight_width=subplot_width,
                                   deriv_window=25)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

    flare_indices = eep.analyze.find_peaks_stddev_thresh(noisy_signal-np.mean(noisy_signal),
                                                         std_dev_threshold=3,
                                                         smoothing_radius=3,
                                                         min_index_gap=approx_index_lag,
                                                         extra_search_pad=4)

    print("By picking out each flare and processing it individually, "
          "the signal from the quiescent background is suppressed.")

    num_flares = len(flare_indices)
    if num_flares > 0:
        eep.visualize.plot_flare_array(noisy_signal, flare_indices,
                                       back_pad=subplot_width//2, forward_pad=subplot_width*3,
                                       display_index=True)

    back_index = 200
    forward_index = back_index*4
    sum_autocorr = np.zeros(max_lag+1)
    sum_autocorr_noisy = np.zeros(max_lag+1)
    ct = 0
    for flare_index in flare_indices:
        if flare_index-back_index > 0 and flare_index+forward_index < len(noisy_signal):
            sum_autocorr_noisy += eep.analyze.autocorrelate_array(filtered_noisy_signal[flare_index-
                                                                  back_index:flare_index+forward_index],
                                                                  max_lag=max_lag)
            sum_autocorr += eep.analyze.autocorrelate_array(filtered_signal[flare_index-
                                                            back_index:flare_index+forward_index],
                                                            max_lag=max_lag)
            ct += 1
    sum_autocorr /= ct
    sum_autocorr_noisy /= ct
    plot_autocorr_with_derivatives(sum_autocorr, np.arange(0, observation_cadence.value*len(sum_autocorr),
                                                           observation_cadence.value),
                                   lag_index_spotlight=approx_index_lag,
                                   lag_index_spotlight_width=subplot_width,
                                   deriv_window=7,
                                   title="Summed autocorrelation")

    plot_autocorr_with_derivatives(sum_autocorr_noisy, np.arange(0, observation_cadence.value * len(sum_autocorr),
                                                                 observation_cadence.value),
                                   lag_index_spotlight=approx_index_lag,
                                   lag_index_spotlight_width=subplot_width,
                                   deriv_window=21,
                                   title="Noisy summed autocorrelation")

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
