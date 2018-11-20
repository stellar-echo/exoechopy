
from pathlib import Path
import numpy as np
from scipy import signal
import exoechopy as eep
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import astropy.units as u

import time


def run():
    #   ============================================================================================================   #
    #   Create synthetic data to analyze

    # Load the template star system:
    cwd = Path.cwd()
    observation_campaign = eep.io.generate_system_from_file(cwd/"system_template.exo")

    # Change the planet parameters from the defaults:
    observation_campaign["planet_parameters"]["semimajor_axis"] = (.05, "au")

    # observation_campaign["planet_parameters"]["planet_radius"] = (8, "R_jup")

    # # Tilt the system at an angle relative to Earth:
    observation_campaign["planet_parameters"]["inclination"] = (.5, "rad")

    # Set a uniform flare intensity (defaults to a distribution):
    observation_campaign["star_parameters"]["active_region"]["intensity_pdf"] = 100000.

    # Ensure we watch long enough to get a good sample:
    observation_campaign["telescope_parameters"]["observation_time"] = (12*24., "hr")

    # Set a good ratio of flares/hr based on our observation time:
    # Distribution is uniform between loc and loc+scale:
    observation_campaign["star_parameters"]["active_region"]["occurrence_freq_pdf"] = {"uniform": {"loc": 100,
                                                                                                   "scale": 1500}}

    # Generate the data:
    telescope = observation_campaign.initialize_experiments()[0]
    telescope.collect_data(save_diagnostic_data=True)

    # Add some Poisson noise:
    lightcurve = telescope.get_degraded_lightcurve(eep.simulate.methods.add_poisson_noise)
    time_domain = telescope.get_time_domain()

    print("""
    Despite the flares having identical intensities, if the peak occurs between two measurements, 
    then discretization often causes the peak intensity to be shared between two bins--this results
    in a non-uniform distribution of flares.  This represents a real experimental artifact.
    """)

    # Examine the synthetic lightcurve:
    eep.visualize.interactive_lightcurve(time_domain, lightcurve)

    #   ============================================================================================================   #
    #   Perform analysis on the data:

    # Extract the flares into a flare catalog, after doing some pre-processing:
    flare_catalog = eep.analyze.FlareCatalog(lightcurve - np.median(lightcurve),
                                             time_domain=time_domain)

    flare_catalog.identify_flares_with_protocol(eep.analyze.find_peaks_stddev_thresh,
                                                std_dev_threshold=2,
                                                min_index_gap=15,
                                                extra_search_pad=15,
                                                single_flare_gap=300)
    # At each flare, include this many points before and after the flare peak in the extracted flare:
    look_back = 100
    look_forward = look_back * 3
    min_lag = 0
    max_lag = 45
    lag_domain = np.linspace(min_lag, max_lag, max_lag-min_lag+1)
    flare_catalog.set_extraction_range((look_back, look_forward))

    # Uncomment to see all flares that are identified:
    # eep.visualize.plot_flare_array(lightcurve, flare_catalog.get_flare_indices(),
    #                                back_pad=look_back, forward_pad=look_forward,
    #                                display_index=True)

    #      ---------------      Create the filters used to identify the echo      ---------------      #
    correlation_process = (eep.analyze.autocorrelate_array, {'min_lag': min_lag, 'max_lag': max_lag})

    def subtract_filter(data, func, **kwargs):
        return data - func(data, **kwargs)

    filter_width = 7
    filter_process = (subtract_filter, {'func': signal.savgol_filter,
                                        'window_length': filter_width,
                                        'polyorder': 2})

    def exp_fit(data, amp, decay):
        return amp * np.exp(-decay * data)

    def crop_exp_fit(data, front_bins):
        """Custom data process for this data.
        Crops out the first front_bins, then fits an exponential decay.  Determined empirically."""
        fit_data = data[front_bins:]
        xdata = np.arange(0, len(fit_data))
        popt, pcov = curve_fit(exp_fit,
                               xdata=xdata, ydata=fit_data,
                               p0=[fit_data[0], 1])
        fit_data -= exp_fit(xdata, *popt)
        return fit_data

    crop_fit_process = (crop_exp_fit, {'front_bins': filter_width})

    #      ---------------      Generate lag-correlator matrix      ---------------      #
    flare_catalog.generate_correlator_matrix(correlation_process, filter_process, crop_fit_process)
    correlation_results = flare_catalog.get_correlator_matrix()
    print("Number of flares analyzed: ", len(correlation_results))
    raw_correlation = np.mean(correlation_results, axis=0)

    plt.plot(lag_domain[-len(raw_correlation):], raw_correlation, color='k', lw=1, drawstyle='steps-post')
    plt.xlabel("Lag (index)")
    plt.ylabel("Correlator matrix metric")
    plt.title("Mean of all correlator metrics")
    plt.tight_layout()
    plt.show()

    #      ---------------      Search the matrix for echoes      ---------------      #
    # We are only searching a single orbital element here,
    # so we're going to steal the exact solution for the other elements
    planet = observation_campaign.gen_planets()[0]
    star = observation_campaign.gen_stars()[0]
    earth_vect = star.earth_direction_vector
    cadence = telescope.cadence

    # Initialize our search object:
    dummy_object = eep.simulate.KeplerianOrbit(semimajor_axis=planet.semimajor_axis,
                                               eccentricity=planet.eccentricity,
                                               star_mass=star.mass,
                                               initial_anomaly=planet.initial_anomaly,
                                               inclination=planet.inclination,
                                               longitude=planet.longitude,
                                               periapsis_arg=planet.periapsis_arg)

    # Set the search space:
    min_inclination = 0
    max_inclination = np.pi/2
    num_tests = 50
    inclination_tests = u.Quantity(np.linspace(min_inclination, max_inclination, num_tests), 'rad')
    # Speed up calculation by precalculating the Keplerian orbit on a coarse mesh and using cubic interpolation:
    num_interpolation_points = 50

    orbit_search = eep.analyze.OrbitSearch(flare_catalog, dummy_object, cadence,
                                           lag_offset=-filter_width, clip_range=(0, max_lag-filter_width))

    # Generate the predicted lags associated with each hypothesis orbit:
    print("Searching orbital inclinations")
    results, _ = orbit_search.run(earth_direction_vector=earth_vect,
                                  lag_func=np.mean,
                                  num_interpolation_points=num_interpolation_points,
                                  inclination=inclination_tests)

    plt.plot(inclination_tests, results, color='k', lw=1)
    # plt.annotate("This peak is due to the echo,\n"
    #              "this one is a false positive.", xy=(25, .00004), xytext=(22.5, .0003),
    #              arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.3"), zorder=25)

    plt.xlabel("Orbital inclination (rad)")
    plt.ylabel("Detection metric")
    plt.tight_layout()
    plt.show()

    #      ---------------      Verify the results      ---------------      #
    print("Resampling orbital inclinations along lag domain")
    num_resamples = 1000
    all_resamples = np.random.choice(flare_catalog.num_flares, (num_resamples, flare_catalog.num_flares))
    all_results, _ = orbit_search.run(earth_direction_vector=earth_vect,
                                      lag_func=np.mean,
                                      num_interpolation_points=num_interpolation_points,
                                      resample_order=all_resamples,
                                      inclination=inclination_tests)
    # Reset order, in case we need it again
    flare_catalog.set_sampling_order()

    # Uncertainty interval:
    interval = 95
    lower, upper = np.percentile(all_results, [(100-interval)/2, 50+interval/2], axis=0)
    eep.visualize.plot_signal_w_uncertainty(inclination_tests, results,
                                            y_plus_interval=upper, y_minus_interval=lower,
                                            x_axis_label="Inclination (rad)",
                                            y_axis_label="Detection metric")

    #      ---------------      Run a search along another orbital element      ---------------      #
    min_semimajor_axis = 0.02
    max_semimajor_axis = 0.07
    semimajor_axis_tests = u.Quantity(np.linspace(min_semimajor_axis, max_semimajor_axis, num_tests), "au")

    # Generate the predicted lags associated with each hypothesis orbit:
    print("Searching semimajor axes")
    results, _ = orbit_search.run(earth_direction_vector=earth_vect,
                                  lag_func=np.mean,
                                  num_interpolation_points=num_interpolation_points,
                                  inclination=planet.inclination,  # Reset from previous run
                                  semimajor_axis=semimajor_axis_tests)

    plt.plot(semimajor_axis_tests, results, color='k', lw=1)
    plt.xlabel("Semimajor axis (au)")
    plt.ylabel("Detection metric")
    plt.tight_layout()
    plt.show()

    # Run a 2D search to show how that works:
    a_axis_tests = 20
    min_semimajor_axis = 0.04
    max_semimajor_axis = 0.06
    inc_axis_tests = 20
    min_inclination = .25
    max_inclination = .75

    semimajor_axis_tests = u.Quantity(np.linspace(min_semimajor_axis, max_semimajor_axis, a_axis_tests), "au")
    inclination_tests = u.Quantity(np.linspace(min_inclination, max_inclination, inc_axis_tests), 'rad')

    #      ---------------      Run a search along two orbital elements      ---------------      #
    print("Running 2D orbital search...")
    results_2d, _ = orbit_search.run(earth_direction_vector=earth_vect,
                                     lag_func=np.mean,
                                     num_interpolation_points=num_interpolation_points,
                                     inclination=inclination_tests,
                                     semimajor_axis=semimajor_axis_tests)

    plt.imshow(np.transpose(results_2d),
               cmap='inferno', origin='lower', aspect='auto',
               extent=[min_semimajor_axis, max_semimajor_axis, min_inclination, max_inclination])
    plt.xlabel("Semimajor axis (au)")
    plt.ylabel("Inclination (rad)")
    plt.title("2D orbit search: Semimajor axis v Inclination")
    plt.colorbar()
    plt.tight_layout()
    plt.show()



# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #

if __name__ == "__main__":
    run()

