
import numpy as np
import exoechopy as eep
import matplotlib.pyplot as plt
import astropy.units as u

from pathlib import Path
from scipy import signal
from scipy.optimize import curve_fit

import time


def run():
    #   ============================================================================================================   #
    #   Create synthetic data to analyze

    print("""
    This demo extends orbit_search_demo_1 by including a distribution of intensities,
    a wider distribution of decay constants, weighting of the autocorrelation signals,
    as well as a star radius effect (some flares hit, some flares miss), but does not
    include a radius effect on the light travel time to the star.
    
    This model generates a continuous light curve, rather than pre-extracted flares, 
    so to avoid memory issues associated with longer light curves, we'll make sure the flares are very frequent.  
    The planet will only undergo a couple orbits in this time, which means periodogram analysis isn't a great approach.
    """)

    # Load the template star system:
    cwd = Path.cwd()
    observation_campaign = eep.io.generate_system_from_file(cwd/"system_template.exo")

    # Change the planet parameters from the defaults:
    semimajor_axis = 0.05
    observation_campaign["planet_parameters"]["semimajor_axis"] = (semimajor_axis, "au")

    # Tilt the system at an angle relative to Earth:
    orbital_inclination = 0.5
    observation_campaign["planet_parameters"]["inclination"] = (orbital_inclination, "rad")

    # # Give the star a radius for hit-miss purposes:
    # observation_campaign["star_parameters"]["star_type"] = "Star"
    # observation_campaign["star_parameters"]["star_radius"] = (1, 'm')

    # Set a uniform flare intensity (defaults to a distribution):
    observation_campaign["star_parameters"]["active_region"]["flare_kwargs"]["decay_pdf"] = \
        {"rayleigh": {"scale": 25}, "unit": "s"}

    # Set a good ratio of flares/hr based on our observation time:
    # Distribution is uniform between loc and loc+scale:
    observation_campaign["star_parameters"]["active_region"]["occurrence_freq_pdf"] = {"uniform": {"loc": 100,
                                                                                                   "scale": 1500}}

    # Ensure we watch long enough to get a good sample:
    observation_campaign["telescope_parameters"]["observation_time"] = (12*24., "hr")

    #      ---------------      Generate the data      ---------------      #
    telescope = observation_campaign.initialize_experiments()[0]
    telescope.collect_data(save_diagnostic_data=True)
    plt.hist(telescope._all_exoplanet_contrasts, bins=100, color="gray")
    plt.annotate("These are all flares that missed the exoplanet",
                 xy=(0, 630),
                 xytext=(1E-5, 500),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.3"), zorder=25)
    plt.annotate("These are the flares that echoed off the exoplanet",
                 xy=(0.00013, 40),
                 xytext=(5E-5, 240),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.1"), zorder=25)
    plt.title("Distribution of exoplanet echo magnitudes")
    plt.xlabel("Echo magnitude")
    plt.ylabel("Number of echoes")
    plt.tight_layout()
    plt.show()

    # Add some Poisson noise:
    lightcurve = telescope.get_degraded_lightcurve(eep.simulate.methods.add_poisson_noise)
    time_domain = telescope.get_time_domain()

    print("""
    Note: Despite the flares having identical intensities, if the peak occurs between two measurements, 
    then discretization often causes the peak intensity to be shared between two bins--this results
    in a non-uniform distribution of flares.  This represents a real experimental artifact.
    """)

    # Examine the synthetic lightcurve:
    eep.visualize.interactive_lightcurve(time_domain, lightcurve)

    #   ============================================================================================================   #
    #   Perform analysis on the data:

    print("""
    To examine the echoes, we first identify and extract the flares from the lightcurve.
    Here, we don't have any background variability, so we can just subtract the quiescent background.
    Then, we'll run a slightly modified standard deviation threshold algorithm to detect flare events.
    The algorithm will ignore flares that occur too close to each other, 
    meaning they would show up in each other's autocorrelation computations.  
    """)

    # Extract the flares into a flare catalog, after doing some pre-processing:
    flare_catalog = eep.analyze.LightcurveFlareCatalog(lightcurve - np.median(lightcurve),
                                                       time_domain=time_domain)

    # At each flare, include this many points before and after the flare peak in the extracted flare:
    min_lag = 0
    max_lag = 80
    look_back = max_lag*2
    look_forward = int(look_back * 3)

    # Build the flare catalog, ignoring flares that would corrupt our autocorrelation analysis
    flare_catalog.identify_flares_with_protocol(eep.analyze.find_peaks_stddev_thresh,
                                                std_dev_threshold=1,
                                                min_index_gap=15,
                                                extra_search_pad=15,
                                                single_flare_gap=look_forward+look_back)

    lag_domain = np.linspace(min_lag, max_lag, max_lag-min_lag+1)
    flare_catalog.set_extraction_range((look_back, look_forward))

    # Display several identified flares at a glance:
    max_displayed_flares = 100
    all_flare_indices = flare_catalog.get_flare_indices()
    eep.visualize.plot_flare_array(lightcurve, all_flare_indices[:min(max_displayed_flares,
                                                                      len(all_flare_indices))],
                                   back_pad=look_back, forward_pad=look_forward)

    #      ---------------      Create the filters used to identify the echo      ---------------      #
    print("""
    To speed up calculations later, we want to pre-calculate all relevant correlators into a matrix.
    
    Echo extraction is still an art, not yet a science.  So there are multiple ways to do it.
    For now, we use a process notation that is a tuple of:
     (the algorithm to run, a dictionary of parameters to pass to the algorithm)
    This process is then run on each extracted flare.
    
    For this example, the processing is as follows:
     1) Run autocorrelation on each flare (correlation_process)
        - The echo is hiding on the back of the correlation decay curve, similar to echo_demo_2.py
        - To detrend the decay curve, we use the following steps:
     2) High-pass filter the autocorrelation (filter_process)
        - This particular process leaves behind a slight residual
     3) Reduce ringing from high-pass, then subtract a residual exponential background trend
        - This process leaves a little ringing behind and is sensitive to initial conditions,
          but is stable in the current configuration.
     
    This process was developed empirically.  To see the impacts of each step, remove the
    individual processes from generate_correlator_matrix(). 
    """)

    correlation_process = (eep.analyze.autocorrelate_array, {'min_lag': min_lag, 'max_lag': max_lag})

    def subtract_filter(data, func, **kwargs):
        return data - func(data, **kwargs)

    filter_width = 9
    filter_process = (subtract_filter, {'func': signal.savgol_filter,
                                        'window_length': filter_width,
                                        'polyorder': 2})

    def exp_fit(data, amp, decay):
        return np.abs(amp) * np.exp(-decay * data)

    def crop_exp_fit(data, front_bins):
        """Custom data process for this data.
        Crops out the first front_bins, then fits an exponential decay.  Determined empirically."""
        fit_data = data[front_bins:]
        xdata = np.arange(0, len(fit_data))
        amp_est = np.max(fit_data[:10])
        decay_est = (amp_est-np.mean(fit_data[5:15]))/5
        try:
            popt, pcov = curve_fit(exp_fit,
                                   xdata=xdata, ydata=fit_data,
                                   p0=[amp_est, decay_est], maxfev=1000)
            fit_data -= exp_fit(xdata, *popt)
        except RuntimeError:
            pass
            # plt.plot(fit_data, color='k')
            # plt.plot(exp_fit(xdata, amp_est, decay_est))
            # plt.show()
        return fit_data

    crop_fit_process = (crop_exp_fit, {'front_bins': filter_width})

    flare_catalog.generate_correlator_matrix(correlation_process, filter_process, crop_fit_process)

    #      ---------------      Generate lag-correlator matrix      ---------------      #
    flare_catalog.generate_correlator_matrix(correlation_process, filter_process, crop_fit_process)

    # Implement a flare weight based on the flare intensity, higher intensity = higher weight:
    def intensity_est(data_array):
        return np.max(data_array)-np.median(data_array)

    # # Implement a flare weight that approximates the flare decay rate
    # def sharpness_est(data_array):
    #     # Determined by how many points are above the half-way point in the light curve
    #     sorted = np.sort(data_array)[::-1]
    #     # Pick a test value, easy choice is half-way between max and min
    #     test_val = (np.max(data_array)+np.min(data_array))/2
    #     est = np.argmax(sorted < test_val)
    #     # Want shorter flares to be weighted more heavily:
    #     return 1/(est+1)
    #
    # # Implement a composite weight of decay constant and flare intensity:
    # def composite_weight(data_array):
    #     return intensity_est(data_array) * sharpness_est(data_array)

    flare_catalog.generate_weights_with_protocol(intensity_est)

    analysis_suite = eep.analyze.EchoEnsembleAnalysis(flare_catalog)

    correlation_results = analysis_suite.correlation_matrix
    weighted_correlation_results = analysis_suite.weighted_correlation_matrix
    print("Number of flares analyzed: ", len(correlation_results))

    raw_correlation = np.mean(correlation_results, axis=0)
    # Note: weighted_correlation requires sum rather than mean to achieve same result, based on how weights are computed
    weighted_correlation = np.sum(weighted_correlation_results, axis=0)

    plt.plot(lag_domain[-len(raw_correlation):], raw_correlation,
             color='darkviolet', lw=1, ls='--', drawstyle='steps-post', label="Unweighted")
    plt.plot(lag_domain[-len(weighted_correlation):], weighted_correlation,
             color='k', lw=1, drawstyle='steps-post', label="Weighted by intensity")
    plt.annotate("Note edge effects due to filters chosen.\n"
                 "These can manifest as ringing when\n"
                 "the filters are aggressive",
                 xy=(10, raw_correlation[0]),
                 xytext=(20, np.min(raw_correlation) * .85),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.3"), zorder=25)
    plt.annotate("",
                 xy=(80, raw_correlation[-1]),
                 xytext=(65, np.min(raw_correlation) * .85),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.3"), zorder=25)
    plt.xlabel("Lag (index)")
    plt.ylabel("Correlator matrix metric")
    plt.title("Mean of all correlator metrics")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    #      ---------------      Search the matrix for echoes      ---------------      #

    print("""
        Now that the lag matrix has been generated, we need to search it for echoes.
        This is a massive search space, including all orbital elements 
        (eccentricity, semimajor axis, star mass, etc).
        To demonstrate how it works, however, we can use the right answer for most elements
        and only search a single unknown at a time.  

        The search will align the computed lag values from the correlator matrix with the 
        predicted lags from each proposed orbit, then generate a value from it.  
        With the processing used here, larger values mean a higher probability of an echo.
        """)

    # We are only searching a single orbital element here,
    # so we're going to steal the exact solution for the other elements
    planet = observation_campaign.gen_planets()[0]
    star = observation_campaign.gen_stars()[0]
    earth_vect = star.earth_direction_vector

    # Set the search space:
    min_inclination = 0
    max_inclination = np.pi / 2
    num_tests = 100
    inclination_tests = u.Quantity(np.linspace(min_inclination, max_inclination, num_tests), 'rad')

    search_params = {'semimajor_axis': planet.semimajor_axis,
                     'eccentricity': planet.eccentricity,
                     'parent_mass': star.mass,
                     'initial_anomaly': planet.initial_anomaly,
                     'inclination': inclination_tests,  # Note: This is an array of values
                     'longitude': planet.longitude,
                     'periapsis_arg': planet.periapsis_arg}

    # Speed up calculation by precalculating the Keplerian orbit on a coarse mesh and using cubic interpolation:
    num_interpolation_points = 50

    # The filter_process (above) clips the correlation function, so to keep lags aligned, we explicitly account for this
    lag_offset = -filter_width

    # Run orbit search:
    print("Searching orbital inclinations...")
    results, _ = analysis_suite.search_orbits(earth_direction_vector=earth_vect,
                                              lag_metric=np.sum,
                                              num_interpolation_points=num_interpolation_points,
                                              lag_offset=lag_offset,
                                              clip_range=(0, max_lag - filter_width),
                                              weighted=True,
                                              **search_params)

    unweighted_results, _ = analysis_suite.search_orbits(earth_direction_vector=earth_vect,
                                                         lag_metric=np.mean,
                                                         num_interpolation_points=num_interpolation_points,
                                                         lag_offset=lag_offset,
                                                         clip_range=(0, max_lag - filter_width),
                                                         weighted=False,
                                                         **search_params)

    # TODO: Add the S/N floor estimator from a 2R_Jup albedo=1 model to the OrbitSearch as an optional param?

    # Generate the predicted lags associated with each hypothesis orbit:

    plt.plot(inclination_tests * 180 / np.pi, results, color='k', lw=1, label="Intensity weighted")
    plt.plot(inclination_tests * 180 / np.pi, unweighted_results, color='darkviolet', lw=1, ls='--', label="Unweighted")
    plt.annotate("This peaky region is likely due to an echo",
                 xy=(orbital_inclination * 180 / np.pi, np.max(results)),
                 xytext=((orbital_inclination + .1) * 180 / np.pi, .85 * np.max(results)),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.3"), zorder=25)

    plt.xlabel("Orbital inclination (deg)")
    plt.ylabel("Detection metric")
    plt.legend(loc="lower right")
    plt.title("Detection metric vs orbital inclination")
    plt.tight_layout()
    plt.show()

    #      ---------------      Characterize the results      ---------------      #
    print("""
        Despite having some peaks in the result, it is difficult to assign meaning to them.
        One means of characterizing the result is to resample the flares with replacement (bootstrapping).
        In this case, we'll reorganize all of the flares, so the times they occur is the same, but which 
        flare occurs when is not the same.
        This has the effect of killing the coherence of the result, with some caveats:
         - Two flares ocurring at different times, but at the same orbital phase (perhaps two orbits later, say),
           would be expected to produce the same echo lag.  Swapping these flares is not expected to change the result.
           A proper resampling would control for these correlation effects.
        While this tool is imperfect, it's a good way to make sure there are enough flares to draw conclusions.
        If your echo signal is coming from a couple of outlier events, the effective sample size may be much smaller
        than expected, and bootstrapping can help reveal those issues.
        """)

    print("Resampling orbital inclinations along lag domain...")

    # TODO: Figure out zero-index wackiness

    est, bias, std_err, conf = analysis_suite.jackknife_orbit_search(np.sum, conf_lvl=0.98, weighted=False)
    print("est.shape: ", est.shape)
    print("bias.shape: ", bias.shape)
    print("std_err.shape: ", std_err.shape)
    print("conf.shape: ", conf.shape)
    plt.plot(inclination_tests * 180 / np.pi, est)
    plt.title("EST")
    plt.show()

    plt.plot(inclination_tests * 180 / np.pi, bias)
    plt.title("BIAS")
    plt.show()

    plt.plot(inclination_tests * 180 / np.pi, std_err)
    plt.title("STD_ERR")
    plt.show()

    plt.plot(inclination_tests * 180 / np.pi, conf[0])
    plt.plot(inclination_tests * 180 / np.pi, conf[1])
    plt.title("CONF")
    plt.show()

    # How many times to resample the lags (with replacement-- i.e., bootstrap)
    # Larger numbers are more accurate, but in this case the result is stable enough by 1000
    num_resamples = 1000

    # Uncertainty interval:
    interval = 98

    resample_result, percentiles = analysis_suite.bootstrap_lag_resample_orbits(stat_func=np.sum,
                                                                                num_resamples=num_resamples,
                                                                                percentile=interval/100,
                                                                                weighted=True)

    meanvals = np.mean(resample_result, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    eep.visualize.plot_signal_w_uncertainty(inclination_tests * 180 / np.pi, results, color_1="midnightblue",
                                            y_label_1="As measured",
                                            y_plus_interval=percentiles[1], y_minus_interval=percentiles[0],
                                            uncertainty_label="Resampled, " + str(interval) + "% confidence",
                                            y_data_2=meanvals, color_2='gray', y_label_2="Resample mean",
                                            x_axis_label="Inclination (deg)",
                                            y_axis_label="Detection metric",
                                            save='hold', axes_object=ax)
    ax.annotate("The peaks no longer protrude above the resampled result",
                xy=(orbital_inclination * 180 / np.pi, np.max(results)),
                xytext=((orbital_inclination + .05) * 180 / np.pi, .85 * np.max(results)),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.3"), zorder=25)
    ax.set_title("Resampled lags to distinguish noise from signal")
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

    # TODO: Add hit-miss histogram analysis

    #      ---------------      Run a search along another orbital element      ---------------      #

    print("""
        Some orbital elements are more sensitive to perturbation than others.
        Semimajor axis is extremely sensitive, so a denser search is necessary.
        This search is at the same density as the orbital inclination search.
        """)

    min_semimajor_axis = 0.02
    max_semimajor_axis = 0.07
    semimajor_axis_tests = u.Quantity(np.linspace(min_semimajor_axis, max_semimajor_axis, num_tests), "au")

    # Only need to update the parameters that have been changed:
    search_params = {'semimajor_axis': semimajor_axis_tests,  # Note: This is an array of values
                     'inclination': planet.inclination}  # Previous search is cached, overwrite with single value

    # Run orbit search:
    # Generate the predicted lags associated with each hypothesis orbit:
    print("Searching semimajor axes...")
    results, _ = analysis_suite.search_orbits(earth_direction_vector=earth_vect,
                                              lag_metric=np.sum,
                                              num_interpolation_points=num_interpolation_points,
                                              lag_offset=lag_offset,
                                              clip_range=(0, max_lag - filter_width),
                                              weighted=True,
                                              **search_params)

    unweighted_results, _ = analysis_suite.search_orbits(earth_direction_vector=earth_vect,
                                                         lag_metric=np.mean,
                                                         num_interpolation_points=num_interpolation_points,
                                                         lag_offset=lag_offset,
                                                         clip_range=(0, max_lag - filter_width),
                                                         weighted=False,
                                                         **search_params)

    plt.plot(semimajor_axis_tests, results, color='k', lw=1, label="Intensity weighted")
    plt.plot(semimajor_axis_tests, unweighted_results,
             color='darkviolet', lw=1, ls="--", label="Unweighted")
    plt.annotate("Peaky region is likely due\n"
                 "to echoes, but is noisy",
                 xy=(semimajor_axis, np.max(results)),
                 xytext=(.02, .75 * np.max(results)),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.3"), zorder=25)
    plt.xlabel("Semimajor axis (au)")
    plt.ylabel("Detection metric")
    plt.title("Detection metric vs semimajor axis")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    #      ---------------      Characterize the results      ---------------      #
    print("Resampling semimajor axes along lag domain...")

    resample_result, percentiles = analysis_suite.bootstrap_lag_resample_orbits(stat_func=np.sum,
                                                                                num_resamples=num_resamples,
                                                                                percentile=interval / 100,
                                                                                weighted=True)

    meanvals = np.mean(resample_result, axis=0)

    ax = eep.visualize.plot_signal_w_uncertainty(semimajor_axis_tests, results, color_1="midnightblue",
                                                 y_label_1="As measured",
                                                 y_plus_interval=percentiles[0], y_minus_interval=percentiles[1],
                                                 uncertainty_label="Resampled, " + str(interval) + "% confidence",
                                                 y_data_2=meanvals, color_2='gray', y_label_2="Resample mean",
                                                 x_axis_label="Semimajor axis (au)",
                                                 y_axis_label="Detection metric",
                                                 save='hold')

    ax.annotate("The echo candidate survived \n"
                "the resample analysis",
                xy=(semimajor_axis, np.max(results)),
                xytext=(.02, .85 * np.max(results)),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.3"), zorder=25)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # TODO: Add hit-miss histogram analysis

    #      ---------------      Run a 2D search to show how that works      ---------------      #

    print("""
        The same notation allows searching along multiple orbital elements (as many as you want).
        Here, we map out orbital inclination vs semimajor axis as a demonstration. 
        The class will use cached values for everything that is not updated in the search. 
        """)

    a_axis_tests = 50
    min_semimajor_axis = 0.035
    max_semimajor_axis = 0.07
    inc_axis_tests = 20
    min_inclination = 0.
    max_inclination = 1.

    semimajor_axis_tests = u.Quantity(np.linspace(min_semimajor_axis, max_semimajor_axis, a_axis_tests), "au")
    inclination_tests = u.Quantity(np.linspace(min_inclination, max_inclination, inc_axis_tests), 'rad')

    print("Running 2D orbital search...")
    results_2d, key_list = analysis_suite.search_orbits(inclination=inclination_tests,
                                                        semimajor_axis=semimajor_axis_tests,
                                                        weighted=True)

    print("key_list: ", key_list)

    plt.imshow(results_2d,
               cmap='inferno', origin='lower', aspect='auto',
               extent=[min_inclination * 180 / np.pi, max_inclination * 180 / np.pi,
                       min_semimajor_axis, max_semimajor_axis])
    plt.colorbar()
    plt.scatter(orbital_inclination * 180 / np.pi, semimajor_axis, marker="+", color='g')
    plt.xlabel("Inclination (deg)")
    plt.ylabel("Semimajor axis (au)")
    plt.title("2D orbit search: Semimajor axis v Inclination")
    plt.tight_layout()
    plt.show()


# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #

if __name__ == "__main__":
    run()

