

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
    First, create a delta-function active star (zero radius), give it really clear flares,
    and give it a big bright planet...
    """)

    telescope_area = np.pi * (2*u.m)**2
    total_efficiency = 0.8
    observation_cadence = .5 * u.s
    observation_duration = 2 * u.hr
    telescope_random_seed = 99
    approximate_num_flares = 12

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
    a_1 = 0.05 * u.au  # semimajor axis
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

    MyExpFlareActivity = eep.simulate.active_regions.FlareActivity(eep.simulate.flares.ExponentialFlare1,
                                                                   intensity_pdf=flare_intensities,
                                                                   onset_pdf=[1, 10] * u.s,
                                                                   decay_pdf=(stats.rayleigh(scale=6), u.s))

    # For a first test, we don't want it to get too complicated,
    # so we'll use a probability distribution that prevents overlapping flares
    scale_factor = 2/approximate_num_flares*observation_duration.to(u.s).value
    print("scale_factor: ", scale_factor)
    occurrence_freq_pdf = stats.uniform(loc=50/observation_cadence.value,
                                        scale=scale_factor-50/observation_cadence.value)
    Starspot = eep.simulate.active_regions.ActiveRegion(flare_activity=MyExpFlareActivity,
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

    lightcurve = MyTelescope._pure_lightcurve
    time_domain = MyTelescope._time_domain
    eep.visualize.interactive_lightcurve(time_domain, lightcurve)

    print("""
    Next, process the data blindly with the autocorrelation algorithm, because...why not try it.""")

    autocorr = eep.analyze.autocorrelate_array(lightcurve,
                                               max_lag=int(40/observation_cadence.to(u.s).value))

    subplot_width = 40

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
                                               max_lag=int(40/observation_cadence.to(u.s).value))
    plot_autocorr_with_derivatives(autocorr, autocorr_domain,
                                   lag_index_spotlight=approx_index_lag, lag_index_spotlight_width=40,
                                   deriv_window=7)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    print("""
        Now that it works without noise, we'll try again with counting noise in the background.
        """)

    noisy_signal = 1.*np.random.poisson(lightcurve.value)
    eep.visualize.interactive_lightcurve(time_domain, noisy_signal)

    filter_kernel = signal.gaussian(40, std=3)
    filter_kernel /= np.sum(filter_kernel)
    filtered_signal = noisy_signal \
                      - signal.fftconvolve(noisy_signal, filter_kernel, mode='same')

    eep.visualize.interactive_lightcurve(time_domain, filtered_signal)

    autocorr = eep.analyze.autocorrelate_array(filtered_signal,
                                               max_lag=int(40 / observation_cadence.to(u.s).value))

    print("""
    Unfortunately, the signal is now buried in the noise.  More effort is required to convincingly extract the echo.  
    For an approach to more challenging data like this, see the next tutorial.
    """)

    plot_autocorr_with_derivatives(autocorr, autocorr_domain,
                                   lag_index_spotlight=approx_index_lag, lag_index_spotlight_width=40,
                                   deriv_window=25)


# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #


if __name__ == "__main__":

    run()
