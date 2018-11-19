
from pathlib import Path
import numpy as np
from scipy import signal
import exoechopy as eep
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import astropy.units as u


def run():
    #  =============================================================  #
    #               Create synthetic data to analyze                  #

    # Load the template star system:
    cwd = Path.cwd()
    observation_campaign = eep.io.generate_system_from_file(cwd/"system_template.exo")

    # Change the planet parameters from the defaults:
    semimajor_axis = .05
    observation_campaign["planet_parameters"]["semimajor_axis"] = (semimajor_axis, "au")

    # observation_campaign["planet_parameters"]["planet_radius"] = (8, "R_jup")

    # # Tilt the system at an angle relative to Earth:
    observation_campaign["planet_parameters"]["inclination"] = (.5, "rad")
    # observation_campaign["planet_parameters"]["planet_albedo"] = 0.
    # Set a uniform flare intensity (defaults to a distribution):
    observation_campaign["star_parameters"]["active_region"]["intensity_pdf"] = 100000.
    # Ensure we watch long enough to get a good sample:
    observation_campaign["telescope_parameters"]["observation_time"] = (6*24., "hr")

    # Generate the data:
    telescope = observation_campaign.initialize_experiments()[0]
    telescope.collect_data()

    # Add some poisson noise:
    lightcurve = telescope.get_degraded_lightcurve(eep.simulate.methods.add_poisson_noise)
    time_domain = telescope.get_time_domain()

    print("""
    Despite the flares having identical intensities, if the peak occurs between two measurements, 
    then discretization often causes the peak intensity to be shared between two bins--this results
    in a non-uniform distribution of flares.  This represents a real experimental artifact.
    """)

    # eep.visualize.interactive_lightcurve(time_domain, lightcurve)

    #  =============================================================  #
    #                 Perform analysis on the data                    #

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
    # for c in correlation_results:
    #     plt.plot(c, color='k', lw=1, drawstyle='steps-post')
    #     plt.show()
    print("Number of flares analyzed: ", len(correlation_results))
    raw_correlation = np.mean(correlation_results, axis=0)
    plt.plot(lag_domain[-len(raw_correlation):], raw_correlation, color='k', lw=1, drawstyle='steps-post')
    plt.show()

    #      ---------------      Search the matrix for echoes      ---------------      #
    # We are only searching a single axis here, so we're going to steal the exact solution for the other elements
    planet = observation_campaign.gen_planets()[0]
    star = observation_campaign.gen_stars()[0]
    earth_vect = star.earth_direction_vector
    cadence = telescope.cadence
    dummy_object = eep.simulate.KeplerianOrbit(semimajor_axis=planet.semimajor_axis,
                                               eccentricity=planet.eccentricity,
                                               star_mass=star.mass,
                                               initial_anomaly=planet.initial_anomaly,
                                               inclination=planet.inclination,
                                               longitude=planet.longitude,
                                               periapsis_arg=planet.periapsis_arg)
    flare_times = flare_catalog.get_flare_times()

    # Set the search space:
    min_inclination = 0
    max_inclination = np.pi/3
    num_tests = 100
    inclination_tests = u.Quantity(np.linspace(min_inclination, max_inclination, num_tests), 'rad')
    results = np.zeros(num_tests)
    for ii, inc in enumerate(inclination_tests):
        dummy_object.inclination = inc
        planet_vectors = [dummy_object.calc_xyz_at_time(f_i) for f_i in flare_times]
        # Subtract the filter_width, since that operation offset the lags, and clip to avoid errors:
        all_lags = [np.clip(-filter_width + eep.utils.round_dec((eep.utils.compute_lag(np.zeros(3),
                                                                                       v_i, earth_vect)/cadence).decompose().value),
                            0, max_lag-filter_width)
                    for v_i in planet_vectors]
        results[ii] = flare_catalog.run_lag_hypothesis(all_lags, func=np.mean)
    plt.plot(inclination_tests, results, color='k', lw=1)
    plt.xlabel("Orbital inclination (rad)")
    plt.ylabel("Detection metric")
    plt.tight_layout()
    plt.show()

    #      ---------------      Verify the results      ---------------      #




# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #


if __name__ == "__main__":
    run()

