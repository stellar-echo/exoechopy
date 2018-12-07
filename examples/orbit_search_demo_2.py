
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
    and with the additional complexity, more flares are required to detect a signal.
    To avoid memory issues associated with longer light curves, we'll make sure the flares are very frequent.  
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

    # observation_campaign["planet_parameters"]["planet_radius"] = (4, "R_jup")

    # Give the star a radius for hit-miss purposes:
    observation_campaign["star_parameters"]["star_type"] = "Star"
    observation_campaign["star_parameters"]["star_radius"] = (1, 'm')

    # Set a uniform flare intensity (defaults to a distribution):
    observation_campaign["star_parameters"]["active_region"]["flare_kwargs"]["decay_pdf"] = \
        {"rayleigh": {"scale": 25}, "unit": "s"}

    # Set a good ratio of flares/hr based on our observation time:
    # Distribution is uniform between loc and loc+scale:
    observation_campaign["star_parameters"]["active_region"]["occurrence_freq_pdf"] = {"uniform": {"loc": 200,
                                                                                                   "scale": 1500}}

    # Ensure we watch long enough to get a good sample:
    observation_campaign["telescope_parameters"]["observation_time"] = (36*24., "hr")

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
    max_lag = 100
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
        """Weigh higher intensity flares more heavily than dim flares"""
        # return np.power(np.max(data_array)-np.median(data_array), 2)
        return np.max(data_array) - np.median(data_array)

    flare_catalog.generate_weights_with_protocol(intensity_est)

    analysis_suite = eep.analyze.EchoAnalysisSuite(flare_catalog)

    correlation_results = analysis_suite.correlation_matrix
    weighted_correlation_results = analysis_suite.weighted_correlation_matrix
    print("Number of flares analyzed: ", len(correlation_results))

    raw_correlation = np.mean(correlation_results, axis=0)
    weighted_correlation = np.mean(weighted_correlation_results, axis=0)

    plt.plot(lag_domain[-len(raw_correlation):], raw_correlation,
             color='darkviolet', lw=1, ls='--', drawstyle='steps-post', label="Unweighted")
    plt.plot(lag_domain[-len(weighted_correlation):], weighted_correlation,
             color='k', lw=1, drawstyle='steps-post', label="Weighted by intensity")
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

    phase_law = eep.simulate.lambertian_phase_law
    # phase_law = None

    mask_list = [False, True]
    for mask in mask_list:
        if mask:
            # Initialize a representative search space, so that the outliers impact both searches used in the demo:
            _inclines = u.Quantity(np.linspace(min_inclination, max_inclination, 10), 'rad')
            _semimajors = u.Quantity(np.linspace(0.02, 0.07, 10), "au")
            _ = analysis_suite.search_orbits(inclination=_inclines,
                                             semimajor_axis=_semimajors)

            print("Rerunning search, this time after removing outlier flares...")
            # We use a relative aggressive rejection criteria, partially to speed up the computation
            analysis_suite.set_jackknife_sigma_mask(np.mean,
                                                    sigma=1.,
                                                    weighted=True,
                                                    outlier_fraction=.5)

            print("Number of outlier flares: ", np.sum(analysis_suite.outlier_mask))

        # Run orbit search:
        print("Searching orbital inclinations...")
        search_result = analysis_suite.search_orbits(earth_direction_vector=earth_vect,
                                                     lag_metric=np.mean,
                                                     num_interpolation_points=num_interpolation_points,
                                                     lag_offset=lag_offset,
                                                     clip_range=(0, max_lag - filter_width),
                                                     weighted=False,
                                                     phase_law=None,
                                                     **search_params)
        unweighted_results = search_result.results

        search_result = analysis_suite.search_orbits(earth_direction_vector=earth_vect,
                                                     lag_metric=np.mean,
                                                     num_interpolation_points=num_interpolation_points,
                                                     lag_offset=lag_offset,
                                                     clip_range=(0, max_lag - filter_width),
                                                     weighted=True,
                                                     phase_law=phase_law,
                                                     **search_params)

        results = search_result.results

        # TODO: Add the S/N floor estimator from a 2R_Jup albedo=1 model to the OrbitSearch as an optional param?

        # Generate the predicted lags associated with each hypothesis orbit:

        plt.plot(inclination_tests * 180 / np.pi, results,
                 color='k', lw=1, label="Intensity & phase weighted", drawstyle='steps-post')
        plt.plot(inclination_tests * 180 / np.pi, unweighted_results,
                 color='darkviolet', lw=1, ls='--', label="Unweighted", drawstyle='steps-post')
        plt.annotate("This peaky region is likely due to an echo",
                     xy=(orbital_inclination * 180 / np.pi, np.max(results)),
                     xytext=((orbital_inclination + .1) * 180 / np.pi, .85 * np.max(results)),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.3"), zorder=25)

        plt.xlabel("Orbital inclination (deg)")
        plt.ylabel("Detection metric")
        plt.legend(loc="lower left")
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

    # How many times to resample the lags (with replacement-- i.e., bootstrap)
    # Larger numbers are more accurate, but for demonstration we'll use a smaller value for speed
    num_resamples = 150
    print("num_resamples: ", num_resamples)

    # Uncertainty interval:
    interval = 98

    resample_result, percentiles = analysis_suite.bootstrap_lag_resample_orbits(stat_func=np.mean,
                                                                                num_resamples=num_resamples,
                                                                                percentile=interval/100,
                                                                                weighted=True)

    meanvals = np.mean(resample_result, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    inclination_domain = inclination_tests * 180 / np.pi
    eep.visualize.plot_signal_w_uncertainty(inclination_domain, results, color_1="midnightblue",
                                            y_label_1="As measured",
                                            y_plus_interval=percentiles[1], y_minus_interval=percentiles[0],
                                            uncertainty_label="Resampled, " + str(interval) + "% confidence",
                                            y_data_2=meanvals, color_2='gray', y_label_2="Resample mean",
                                            x_axis_label="Inclination (deg)",
                                            y_axis_label="Detection metric",
                                            save='hold', axes_object=ax)
    ax.annotate("The peaks survive the resample analysis when\n"
                "phase weighting is applied",
                xy=(orbital_inclination * 180 / np.pi, np.max(results)),
                xytext=((orbital_inclination + .15) * 180 / np.pi, .75 * np.max(results)),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.3"), zorder=25)
    ax.set_title("Resampled lags to distinguish noise from signal")
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

    print("""
    Here, we run a search for bigaussian results, which would be expected from a hit-miss situation (most real orbits).
    The calculation is extremely slow, so we pick just our top candidates.
    """)

    # Which results are sufficiently significant to include in search:
    top_percentile_for_analysis = 80

    # When searching for bimodal behavior, pick a sampling resolution
    # Smaller values give inaccurate results, larger values take longer.
    # 600 seems like a good compromise in this model.
    num_kde_bins = 600

    result_threshold = np.percentile(results, top_percentile_for_analysis)
    threshold_mask = results > result_threshold
    print("Number of orbital inclinations to evaluate for bimodality: ", np.sum(threshold_mask))

    # Reinitialize results:
    inclination_tests_subsample = inclination_tests[threshold_mask]
    _ = analysis_suite.search_orbits(inclination=inclination_tests_subsample)

    packed_results = analysis_suite.bigauss_kde_orbit_search(num_resamples,
                                                             kde_bins=num_kde_bins,
                                                             confidence=interval/100)

    print("""
    Note, there's an error with the matplotlib fill_between plots that prevents the last step of a stepwise
    plot from filling.  display_1d_orbit_bigauss_search() compensates for that, but if two discontinuous
    plot regions are separated by a single datapoint, it will fill across the gap!""")

    ax = eep.visualize.display_1d_orbit_bigauss_search(packed_results,
                                                       domain=inclination_domain,
                                                       mask=threshold_mask,
                                                       baseline_result=results,
                                                       interval=interval,
                                                       domain_label="Inclination (deg)",
                                                       save='ax')
    ax.annotate("This gap between the high estimate and the origin\n"
                "indicates that the data is\n"
                "consistent with a bimodal distribution",
                xy=(orbital_inclination*180/np.pi, 3e-6),
                xytext=(40, 1e-5),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.3"), zorder=25)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    #      ---------------      Run a search along another orbital element      ---------------      #

    print("""
        Some orbital elements are more sensitive to perturbation than others.
        Semimajor axis is extremely sensitive, so a denser search is necessary.
        This search is at the same density as the orbital inclination search.
        """)

    min_semimajor_axis = 0.04
    max_semimajor_axis = 0.07
    semimajor_axis_tests = u.Quantity(np.linspace(min_semimajor_axis, max_semimajor_axis, num_tests), "au")

    # Only need to update the parameters that have been changed:
    search_params = {'semimajor_axis': semimajor_axis_tests,  # Note: This is an array of values
                     'inclination': planet.inclination}  # Previous search is cached, overwrite with single value

    # Run orbit search:
    # Generate the predicted lags associated with each hypothesis orbit:
    print("Searching semimajor axes...")
    search_results = analysis_suite.search_orbits(earth_direction_vector=earth_vect,
                                                  lag_metric=np.mean,
                                                  num_interpolation_points=num_interpolation_points,
                                                  lag_offset=lag_offset,
                                                  clip_range=(0, max_lag - filter_width),
                                                  weighted=False,
                                                  **search_params)

    unweighted_results = search_results.results

    search_results = analysis_suite.search_orbits(earth_direction_vector=earth_vect,
                                                  lag_metric=np.mean,
                                                  num_interpolation_points=num_interpolation_points,
                                                  lag_offset=lag_offset,
                                                  clip_range=(0, max_lag - filter_width),
                                                  weighted=True,
                                                  **search_params)

    results = search_results.results

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(semimajor_axis_tests, results, color='k', lw=1, label="Intensity & phase weighted", drawstyle='steps-post')
    ax.plot(semimajor_axis_tests, unweighted_results,
            color='darkviolet', lw=1, ls="--", label="Unweighted", drawstyle='steps-post')
    ax.annotate("This peak is where we expect\n"
                "the signal to be",
                xy=(semimajor_axis, np.max(results)),
                xytext=(.04, .95 * np.max(results)),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.3"), zorder=25)
    plt.xlabel("Semimajor axis (au)")
    plt.ylabel("Detection metric")
    plt.title("Detection metric vs semimajor axis")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    #      ---------------      Characterize the results      ---------------      #
    print("Resampling semimajor axes along lag domain...")

    resample_result, percentiles = analysis_suite.bootstrap_lag_resample_orbits(stat_func=np.mean,
                                                                                num_resamples=num_resamples,
                                                                                percentile=interval / 100,
                                                                                weighted=True)

    meanvals = np.mean(resample_result, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax = eep.visualize.plot_signal_w_uncertainty(semimajor_axis_tests, results, color_1="midnightblue",
                                                 y_label_1="As measured",
                                                 y_plus_interval=percentiles[0], y_minus_interval=percentiles[1],
                                                 uncertainty_label="Resampled, " + str(interval) + "% confidence",
                                                 y_data_2=meanvals, color_2='gray', y_label_2="Resample mean",
                                                 x_axis_label="Semimajor axis (au)",
                                                 y_axis_label="Detection metric",
                                                 save='hold',
                                                 axes_object=ax)

    ax.annotate("The echo candidate narrowly\n"
                "survives the resample analysis",
                xy=(semimajor_axis, np.max(results)),
                xytext=(.04, .75 * np.max(results)),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-.3"), zorder=25)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    print("""
    Here, we run a search for bi-Gaussian results, which would be expected from a hit-miss situation (most real orbits).
    The calculation is extremely slow, so we pick just our top candidates.
    """)

    result_threshold = np.percentile(results, top_percentile_for_analysis)
    threshold_mask = results > result_threshold
    print("Number of points to test: ", np.sum(threshold_mask))

    # Reinitialize results:
    semimajor_axis_tests_subsample = semimajor_axis_tests[threshold_mask]
    _ = analysis_suite.search_orbits(semimajor_axis=semimajor_axis_tests_subsample)

    packed_results = analysis_suite.bigauss_kde_orbit_search(num_resamples,
                                                             kde_bins=num_kde_bins,
                                                             confidence=interval/100)

    ax = eep.visualize.display_1d_orbit_bigauss_search(packed_results,
                                                       domain=semimajor_axis_tests,
                                                       mask=threshold_mask,
                                                       baseline_result=results,
                                                       interval=interval,
                                                       domain_label="Semimajor axis (au)",
                                                       save='ax')

    ax.annotate("This gap between the high\n"
                "estimate and the origin\n"
                "indicates that the data is\n"
                "consistent with a bimodal"
                "\ndistribution",
                xy=(semimajor_axis, 1.5e-6),
                xytext=(min_semimajor_axis, 1e-5),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.3"), zorder=25)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    #      ---------------      Run a 2D search to show how that works      ---------------      #

    print("""
        The same notation allows searching along multiple orbital elements (as many as you want).
        Here, we map out orbital inclination vs semimajor axis as a demonstration. 
        The class will use cached values for everything that is not updated in the search. 
        """)

    a_axis_tests = 50
    min_semimajor_axis = 0.04
    max_semimajor_axis = 0.075
    inc_axis_tests = 21
    min_inclination = 0.
    max_inclination = 1.

    semimajor_axis_tests = u.Quantity(np.linspace(min_semimajor_axis, max_semimajor_axis, a_axis_tests), "au")
    inclination_tests = u.Quantity(np.linspace(min_inclination, max_inclination, inc_axis_tests), 'rad')

    print("Running 2D orbital search...")
    search_results_2d = analysis_suite.search_orbits(inclination=inclination_tests,
                                                     semimajor_axis=semimajor_axis_tests,
                                                     weighted=True)

    results_2d = search_results_2d.results

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

