import numpy as np

from ..utils import *
from ..simulate.orbital_physics import KeplerianOrbit
from ..analyze import *

from astropy import units as u
from scipy.special import erfinv

__all__ = ['OrbitSearch', 'EchoAnalysisSuite']


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class OrbitSearch:
    def __init__(self,
                 flare_catalog: BaseFlareCatalog,
                 dummy_orbiter: 'KeplerianOrbit',
                 lag_offset: int = 0,
                 clip_range: tuple = (0, None)):
        """Class for searching orbits for echoes

        Parameters
        ----------
        flare_catalog
            A pre-processed collection of flares
        dummy_orbiter
            Object to apply orbital tests to
        lag_offset
            Optional lag offset index
            Some flare_catalogs have a fixed index offset (e.g., caused by convolution filters that clip end points)
        clip_range
            Optional lag index clipping to prevent overrun
        """
        self._flare_catalog = flare_catalog
        self._flare_times = flare_catalog.get_flare_times().to(u.s).value

        self._dummy_orbiter = dummy_orbiter

        self._cadence = self._flare_catalog.cadence.to(u.s).value
        self._lag_offset = lag_offset
        self._clip_range = clip_range

    # ------------------------------------------------------------------------------------------------------------ #
    def set_lag_offset(self, lag_offset):
        self._lag_offset = lag_offset

    def set_clip_range(self, clip_range):
        self._clip_range = clip_range

    # ------------------------------------------------------------------------------------------------------------ #
    def run(self,
            earth_direction_vector: np.ndarray = None,
            lag_metric: FunctionType = None,
            num_interpolation_points: int = 50,
            resample_order: np.ndarray = None,
            flare_mask: np.ndarray = None,
            phase_law: FunctionType = None,
            weighted: bool = False,
            **kwargs):
        """
        Search over single or multi-dimensional orbit space for echoes

        Parameters
        ----------
        earth_direction_vector
            Vector or np.array of vectors
            If an array of vectors is provided, will search over the vectors as a parameter
        lag_metric
            Function to apply to the list of aligned lags (e.g., np.mean or np.sum)
            If None, returns lag array
        num_interpolation_points
            Number of points to interpolate for the orbital positions (50 is often plenty)
        resample_order
            Optional array of resampled indices to pass to run_lag_hypothesis
        flare_mask
            Optional mask to apply to the flares, useful for testing subsamples (like outlier removal or jackknifing)
            without changing the actual flare arrays
        phase_law
            Optional phase law that operates on an angle array to weight the computed exoplanet visibility
            at the time of the flares
        weighted
            Optional boolean to include weights in the search, defaults to False
        kwargs
            Dictionary of search parameters
            e.g.,
            {'inclination': u.Quantity(np.linspace(min_inclination, max_inclination, num_tests), 'rad')}

        Returns
        -------
        np.ndarray
        """

        if flare_mask is None:
            num_flares = self.num_flares
        else:
            if np.ndim(flare_mask) == 1:
                num_flares = np.sum(flare_mask)
            else:
                num_flares = np.sum(flare_mask[0])

        if num_interpolation_points is None:
            # Err on safe side:
            num_interpolation_points = 500
        # Verify keys all exist
        # (if there's a typo, would silently create a new variable if we didn't do this):
        for k in kwargs.keys():
            _ = getattr(self._dummy_orbiter, k)

        # Create the search space
        keys = []
        search_parameters = []
        # See if there are multiple earth orientations requested (important to be first in list):
        if earth_direction_vector.ndim > 1:
            keys.append('earth_direction_vector')
            search_parameters.append(earth_direction_vector)
        for k, v in kwargs.items():
            try:
                # Is this a search parameter?
                _ = iter(v)
                keys.append(k)
                search_parameters.append(v)
            except TypeError:
                # Not iterated, just setting a value:
                setattr(self._dummy_orbiter, k, v)

        # Initialize the search, figure out how many dimensions are being considered:
        if resample_order is None:
            if flare_mask is None or np.ndim(flare_mask) == 1:
                dim_list = [len(x) for x in search_parameters]
            else:
                dim_list = [len(flare_mask)]
                dim_list.extend([len(x) for x in search_parameters])
        else:
            if resample_order.ndim > 1:
                dim_list = [len(resample_order)]
                dim_list.extend([len(x) for x in search_parameters])
            else:
                dim_list = [len(x) for x in search_parameters]
        if lag_metric is None:
            dim_list.append(num_flares)
        else:
            # Try the lag_metric on some random data without extracting the real data, should be adequate... right?
            test_result = lag_metric(np.linspace(10, 20, num_flares))
            try:
                _ = iter(test_result)
                if len(test_result) > 1:
                    dim_list.append(len(test_result))
            except TypeError:
                pass
        return_array = np.zeros(dim_list)
        print("dim_list: ", dim_list)

        # Magic happens here:
        def recursive_search(depth, ind, _max_depth):
            if depth < _max_depth:
                _k = keys[depth]
                for _v_i, _val in enumerate(search_parameters[depth]):
                    setattr(self._dummy_orbiter, _k, _val)
                    ind[depth] = _v_i
                    recursive_search(depth + 1, ind, _max_depth)
            else:
                planet_vectors = self._dummy_orbiter.evaluate_positions_at_times_lw(self._flare_times,
                                                                                    num_points=num_interpolation_points)
                # Subtract the filter_width, since that operation offset the lags:
                all_lags = compute_lag_simple(planet_vectors,
                                              earth_direction_vector) / self._cadence + self._lag_offset
                # Clip to avoid running over array size: (note, also should de-weight these points)
                all_lags = np.clip(np.round(all_lags), self._clip_range[0], self._clip_range[1]).astype('int')

                if phase_law is not None:
                    all_angles = u.Quantity(angle_between_vector_array(earth_direction_vector, planet_vectors), 'rad')
                    phase_weights = phase_law(all_angles).value
                else:
                    phase_weights = None

                # TODO: Accept external weight, such as star rotation phase

                if resample_order is None:  # No special shenanigans with flare order, standard case:
                    if flare_mask is None or np.ndim(flare_mask) == 1:
                        return_array[tuple(ind)] = self._flare_catalog.run_lag_hypothesis(all_lags,
                                                                                          func=lag_metric,
                                                                                          phase_weights=phase_weights,
                                                                                          flare_weights=weighted,
                                                                                          flare_mask=flare_mask)
                    else:  # Multidimensional flare_mask:
                        for n_i, new_mask in enumerate(flare_mask):
                            # print("n_i, tuple(ind): ", n_i, tuple(ind))
                            return_array[n_i, tuple(ind)] = self._flare_catalog.run_lag_hypothesis(all_lags,
                                                                                                   func=lag_metric,
                                                                                                   phase_weights=phase_weights,
                                                                                                   flare_weights=weighted,
                                                                                                   flare_mask=new_mask)
                else:
                    # Single resample:
                    if resample_order.ndim == 1:
                        return_array[tuple(ind)] = self._flare_catalog.run_lag_hypothesis(all_lags,
                                                                                          func=lag_metric,
                                                                                          phase_weights=phase_weights,
                                                                                          resample_order=resample_order,
                                                                                          flare_weights=weighted,
                                                                                          flare_mask=flare_mask)
                    # Multiple resamples provided:
                    else:
                        if flare_mask is not None and np.ndim(flare_mask) > 1:
                            for n_i, (new_ordering, new_mask) in enumerate(zip(resample_order, flare_mask)):
                                return_array[n_i, tuple(ind)] = self._flare_catalog.run_lag_hypothesis(all_lags,
                                                                                                       func=lag_metric,
                                                                                                       phase_weights=phase_weights,
                                                                                                       resample_order=new_ordering,
                                                                                                       flare_weights=weighted,
                                                                                                       flare_mask=new_mask)
                        else:
                            for n_i, new_ordering in enumerate(resample_order):
                                return_array[n_i, tuple(ind)] = self._flare_catalog.run_lag_hypothesis(all_lags,
                                                                                                       func=lag_metric,
                                                                                                       phase_weights=phase_weights,
                                                                                                       resample_order=new_ordering,
                                                                                                       flare_weights=weighted,
                                                                                                       flare_mask=flare_mask)

        # Manually iterate through the highest level, since it could be the earth_direction_vector,
        # which requires special treatment:
        k = keys[0]
        max_depth = len(keys)
        index_dim = return_array.ndim
        if resample_order is not None:
            index_dim -= 1
        if lag_metric is None:
            index_dim -= 1
        if np.ndim(flare_mask) > 1:
            index_dim -= 1
        index = np.zeros(index_dim, dtype='int')
        for v_i, val in enumerate(search_parameters[0]):
            if k != 'earth_direction_vector':
                setattr(self._dummy_orbiter, k, val)
            else:
                earth_direction_vector = val
            index[0] = v_i
            recursive_search(1, index, max_depth)

        return return_array, keys

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def num_flares(self):
        return self._flare_catalog.num_flares


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

class EchoAnalysisSuite:
    def __init__(self, flare_catalog: BaseFlareCatalog):
        self._flare_catalog = flare_catalog
        self._calculated_correlations = self._flare_catalog.get_correlator_matrix()

        self._orbit_search = None
        self._earth_direction_vector = None
        self._lag_metric = None
        self._num_interpolation_points = None
        self._lag_offset = 0
        self._clip_range = (0, None)
        self._search_kwargs = None
        self._search_results = None
        self._all_search_results = None
        self._keys = None
        self._outlier_mask = None
        self._weighted = False
        self._phase_law = None

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def correlation_matrix(self):
        if self._outlier_mask is None:
            return self._calculated_correlations
        else:
            return self._calculated_correlations[~self._outlier_mask]

    @property
    def weighted_correlation_matrix(self):
        weights = self._flare_catalog.get_weights()
        if self._outlier_mask is None:
            return weights[:, np.newaxis] * self._calculated_correlations / np.mean(weights)
        else:
            return weights[~self._outlier_mask, np.newaxis] * self.correlation_matrix / np.mean(
                weights[~self._outlier_mask])

    # ------------------------------------------------------------------------------------------------------------ #
    def search_orbits(self,
                      earth_direction_vector: np.ndarray = None,
                      lag_metric: FunctionType = np.mean,
                      num_interpolation_points: int = None,
                      lag_offset: int = None,
                      clip_range: tuple = None,
                      weighted: bool = None,
                      phase_law: FunctionType = False,
                      **kwargs):
        """Perform a search over a variety of orbital parameters using an OrbitalSearch class

        Once run, the parameters are cached.  A new search with different parameters can be performed
        by only providing the new parameters.

        Parameters
        ----------
        earth_direction_vector
            Vector or np.array of vectors
            If an array of vectors is provided, will search over the vectors as a parameter
            e.g., (0, 0, 1) is face-on, (1, 0, 0) is edge-on
        lag_metric
            Function to apply to the list of aligned lags (default to np.mean, np.sum also common)
            If None, returns all correlation values
        num_interpolation_points
            Number of points to interpolate for the orbital positions (50 is often plenty)
        lag_offset
            Optional lag offset index
            Some flare_catalogs have a fixed index offset (e.g., caused by convolution filters that clip end points)
            Otherwise, defaults to 0
        clip_range
            Optional lag index clipping to prevent overrun
            Defaults to (0, None)
        weighted
            Option to include pre-computed weights from the flare_catalog in the orbit search
        phase_law
            Optional phase law to use to weight the echoes
        kwargs
            Dictionary of Keplerian orbit parameters to use and search
            {'semimajor_axis': u.Quantity(0.05, 'au'),
            'inclination': u.Quantity(np.linspace(min_inclination, max_inclination, num_tests), 'rad'), ...}
             - semimajor_axis
             - eccentricity
             - star_mass
             - initial_anomaly
             - inclination
             - longitude
             - periapsis_arg

        Returns
        -------

        """
        if self._flare_catalog.cadence is None:
            raise ValueError("FlareCatalog cadence is not initialized")

        if earth_direction_vector is not None:
            self._earth_direction_vector = earth_direction_vector
        if self._earth_direction_vector is None:
            raise ValueError("An earth_direction_vector must be provided to search_orbits")

        self._lag_metric = lag_metric

        if num_interpolation_points is not None:
            self._num_interpolation_points = num_interpolation_points
        # num_interpolation_points defaults in the OrbitSearch, do not need to raise error

        if lag_offset is not None:
            self._lag_offset = lag_offset

        if clip_range is not None:
            self._clip_range = clip_range

        if self._orbit_search is None:
            dummy_orbiter = KeplerianOrbit()
            self._orbit_search = OrbitSearch(self._flare_catalog, dummy_orbiter, self._lag_offset, self._clip_range)
        else:
            # Just in case they changed:
            self._orbit_search.set_clip_range(self._clip_range)
            self._orbit_search.set_lag_offset(self._lag_offset)

        if self._search_kwargs is None:
            self._search_kwargs = kwargs
        else:
            self._search_kwargs = {**self._search_kwargs, **kwargs}

        if self._outlier_mask is None:
            flare_mask = None
        else:
            flare_mask = ~self._outlier_mask

        if weighted is not None:
            self._weighted = weighted

        # if phase_law is False (default), does not change status
        # This allows phase_law to be reset by setting it to None
        if phase_law is None or phase_law:
            self._phase_law = phase_law

        # =============================================================  #

        search_results, keys = self._orbit_search.run(earth_direction_vector=self._earth_direction_vector,
                                                      lag_metric=None,
                                                      num_interpolation_points=self._num_interpolation_points,
                                                      flare_mask=flare_mask,
                                                      weighted=self._weighted,
                                                      phase_law=self._phase_law,
                                                      **self._search_kwargs)
        if self._lag_metric is None:
            self._all_search_results = search_results
            self._search_results = search_results
        else:
            self._all_search_results = search_results
            self._search_results = np.apply_along_axis(lag_metric, -1, search_results)
        self._keys = keys

        return self._search_results, keys

    # ------------------------------------------------------------------------------------------------------------ #
    def gen_jackknife_sigma_mask(self,
                                 stat_func: FunctionType,
                                 sigma: float = 2.,
                                 weighted: bool = False,
                                 outlier_fraction: float = 0):
        """Return a mask of outlier flares based on jackknife testing

        Parameters
        ----------
        stat_func
            Function to apply to the lag_metric, typically np.mean or np.sum
        sigma
            Sigma threshold for identifying an outlier, typically 2-5
        weighted
            Whether to use flare weights
        outlier_fraction
            In what fraction of searches should a flare be an outlier to be removed
            0 means that it was an outlier at least once in any orbital search configuration
            1 means that it was an outlier in every single orbital search configuration
            0.5 means it's an outlier in 50% of searches

        Returns
        -------
        np.ndarray
            Mask of outlier flares
        """
        if self._orbit_search is None:
            raise ValueError("Orbital search is not initialized, run search_orbits first")

        num_flares = self._orbit_search.num_flares

        # Generate jackknife indices:
        jackknife_resample_mask = np.ones([num_flares, num_flares], dtype=bool)
        np.fill_diagonal(jackknife_resample_mask, False)

        search_results, keys = self._orbit_search.run(earth_direction_vector=self._earth_direction_vector,
                                                      lag_metric=stat_func,
                                                      num_interpolation_points=self._num_interpolation_points,
                                                      weighted=weighted,
                                                      flare_mask=jackknife_resample_mask,
                                                      phase_law=self._phase_law,
                                                      **self._search_kwargs)

        all_sigmas = np.std(search_results, axis=0)
        all_means = np.mean(search_results, axis=0)
        outlier_status = np.abs(search_results - all_means)/all_sigmas
        sigma_mask = outlier_status > sigma
        if outlier_fraction == 0:
            return_mask = np.any(sigma_mask, axis=tuple(np.arange(1, np.ndim(sigma_mask))))
        elif outlier_fraction == 1:
            return_mask = np.all(sigma_mask, axis=tuple(np.arange(1, np.ndim(sigma_mask))))
        else:
            outlier_count = np.sum(sigma_mask, axis=tuple(np.arange(1, np.ndim(sigma_mask))), dtype=np.float64)
            outlier_count /= np.sum(search_results.shape[1:])
            return_mask = outlier_count > outlier_fraction
        return return_mask

    def set_jackknife_sigma_mask(self,
                                 stat_func: FunctionType,
                                 sigma: float = 2.,
                                 weighted: bool = False,
                                 outlier_fraction: float = 0):
        self.set_outlier_mask(self.gen_jackknife_sigma_mask(stat_func, sigma, weighted, outlier_fraction))

    # ------------------------------------------------------------------------------------------------------------ #
    def jackknife_orbit_search(self,
                               stat_func: FunctionType,
                               conf_lvl: float = 0.95,
                               weighted: bool = False):
        """Implementation of astropy's jackknife_stats function along a previously implemented orbital search

        Parameters
        ----------
        stat_func
            Function to apply to the jackknifed dataset, typically np.mean
        conf_lvl
            Confidence level for the confidence interval of the Jackknife estimate.
            Must be a real-valued number in (0,1). Default value is 0.95.
        weighted
            Whether or not to use pre-computed weights in the orbit search

        Returns
        -------
        estimate : numpy.ndarray
            The i-th element is the bias-corrected "jackknifed" estimate.

        bias : numpy.ndarray
            The i-th element is the jackknife bias.

        std_err : numpy.ndarray
            The i-th element is the jackknife standard error.

        conf_interval : numpy.ndarray
            If ``statistic`` is single-valued, the first and second elements are
            the lower and upper bounds, respectively. If ``statistic`` is
            vector-valued, each column corresponds to the confidence interval for
            each component of ``statistic``. The first and second rows contain the
            lower and upper bounds, respectively.

        """
        if self._orbit_search is None:
            raise ValueError("Orbital search is not initialized, run search_orbits first")

        num_flares = self._orbit_search.num_flares

        # Generate jackknife indices:
        jackknife_resample_mask = np.ones([num_flares, num_flares], dtype=bool)
        np.fill_diagonal(jackknife_resample_mask, False)

        search_results, keys = self._orbit_search.run(earth_direction_vector=self._earth_direction_vector,
                                                      lag_metric=stat_func,
                                                      num_interpolation_points=self._num_interpolation_points,
                                                      weighted=weighted,
                                                      flare_mask=jackknife_resample_mask,
                                                      phase_law=self._phase_law,
                                                      **self._search_kwargs)

        mean_jack_stat = np.mean(search_results, axis=0)

        # Data without jackknife:
        stat_data = np.apply_along_axis(stat_func, -1, self._all_search_results)

        # jackknife bias
        bias = (num_flares - 1) * (mean_jack_stat - stat_data)

        # jackknife standard error
        std_err = np.sqrt((num_flares - 1) * np.mean((search_results - mean_jack_stat) * (search_results -
                                                                                          mean_jack_stat), axis=0))

        # bias-corrected "jackknifed estimate"
        estimate = stat_data - bias

        # jackknife confidence interval
        if not (0 < conf_lvl < 1):
            raise ValueError("confidence level must be in (0, 1).")

        z_score = np.sqrt(2.0) * erfinv(conf_lvl)
        conf_interval = estimate + z_score * np.array((-std_err, std_err))

        return estimate, bias, std_err, conf_interval

    # ------------------------------------------------------------------------------------------------------------ #
    def bootstrap_lag_resample_orbits(self,
                                      stat_func: FunctionType,
                                      num_resamples: int,
                                      subsample: int = None,
                                      percentile: float = 0.95,
                                      weighted: bool = False):
        """Resample the flares to have occurred at different times

        Parameters
        ----------
        stat_func
            A function to apply to the bootstrapped result, typically np.mean or np.sum
        num_resamples
            Number of resamples to run
        subsample
            Optional number of subsamples to select
        percentile
            Return the upper and lower values of this result (can also be calculated directly from the return result)
        weighted
            Boolean determining whether or not to use pre-computed weights

        Returns
        -------

        """
        if self._orbit_search is None:
            raise ValueError("Orbital search is not initialized, run search_orbits first")

        num_flares = self._orbit_search.num_flares
        all_flare_indices = np.arange(num_flares)
        if self._outlier_mask is not None:
            all_flare_indices = all_flare_indices[~self._outlier_mask]
            num_flares = np.sum(~self._outlier_mask)

        if subsample is None:
            subsample_size = num_flares
        else:
            subsample_size = subsample

        all_resamples = np.random.choice(all_flare_indices, (num_resamples, subsample_size))

        if subsample is not None:
            subsample_mask = np.ones(all_resamples.shape)
            for ind in range(len(subsample_mask)):
                delete_mask = np.random.choice(np.arange(len(subsample_mask)), num_flares-subsample_size, replace=False)
                subsample_mask[ind][delete_mask] = False
        else:
            if self._outlier_mask is not None:
                subsample_mask = ~self._outlier_mask
            else:
                subsample_mask = None

        search_results, keys = self._orbit_search.run(resample_order=all_resamples,
                                                      flare_mask=subsample_mask,
                                                      earth_direction_vector=self._earth_direction_vector,
                                                      lag_metric=stat_func,
                                                      num_interpolation_points=self._num_interpolation_points,
                                                      weighted=weighted,
                                                      phase_law=self._phase_law,
                                                      **self._search_kwargs)

        conf_interval_array = np.percentile(search_results, [(100 - percentile * 100) / 2, 50 + percentile * 100 / 2],
                                            axis=0)
        return search_results, conf_interval_array

    # ------------------------------------------------------------------------------------------------------------ #
    def bigauss_kde_orbit_search(self,
                                 num_bootstrap_resamples: int,
                                 kde_bins: int = 1000,
                                 kde_func: 'BaseKDE' = GaussianKDE,
                                 kde_bandwidth: (str, float) = 'silverman',
                                 num_cores: int = 1,
                                 chunk_size: int = 1,
                                 confidence: float = .95
                                 ):
        """Analyze an orbital search for bigaussian behavior using a kde bootstrap method

        Runs the analysis on the most recently computed orbit, including all of its parameters (e.g., weights and masks)

        Parameters
        ----------
        num_bootstrap_resamples
            Number of resamples to run on each search
        kde_bins
            Number of bins to use in the KDE analysis, typically need 500-2000 depending on spread of values
        kde_func
            What kernel to use, typically GaussianKDE
        kde_bandwidth
            Bandwidth to pass to KDE, typically 'silverman' or a float
        num_cores
            Number of cores to run the bootstrap on, often not worth the overhead currently
        chunk_size
            Number of tasks to send along to each core, often not worth increasing above 1
        confidence
            Percentile to compute for determining upper and lower bounds of the Gaussian search

        Returns
        -------
        means : np.ndarray
            Mean values of search results, typically corresponds to traditional search metric

        lower_means : np.ndarray
            Estimate of lower Gaussian position

        upper_means : np.ndarray
            Estimate of upper Gaussian position

        lower_min_conf : np.ndarray
            Lower bound of lower Gaussian based on confidence

        lower_max_conf : np.ndarray
            Upper bound of lower Gaussian based on confidence

        upper_min_conf : np.ndarray
            Lower bound of upper Gaussian based on confidence

        upper_max_conf : np.ndarray
            Upper bound of upper Gaussian based on confidence
        """
        if self._orbit_search is None:
            raise ValueError("Orbital search is not initialized, run search_orbits first")

        # mean_results = np.mean(self._all_search_results, axis=tuple(range(1, self._all_search_results.ndim)))
        # search_results = self._all_search_results - mean_results[:, np.newaxis]

        search_results = self._all_search_results

        search_shape = search_results.shape[:-1]

        spread = np.max(search_results) - np.min(search_results)
        overfill_fraction = .2
        x_vals = np.linspace(np.min(search_results) - spread * overfill_fraction,
                             np.max(search_results) + spread * overfill_fraction,
                             kde_bins)

        # Initialize arrays needed to generate plots:
        means = np.zeros(search_shape)
        lower_means = np.zeros(search_shape)
        upper_means = np.zeros(search_shape)
        lower_min_conf = np.zeros(search_shape)
        lower_max_conf = np.zeros(search_shape)
        upper_min_conf = np.zeros(search_shape)
        upper_max_conf = np.zeros(search_shape)

        lower_interval = 100 * ((1 - confidence) / 2)
        upper_interval = 100 * (1 - (1 - confidence) / 2)

        # Analyze data:
        for c_i in np.ndindex(search_shape):
            all_correlators = search_results[c_i]
            means[c_i] = np.mean(all_correlators)
            std_dev = np.std(all_correlators)
            current_analysis = ResampleKDEAnalysis(dataset=all_correlators,
                                                   x_domain=x_vals,
                                                   kde=kde_func,
                                                   kde_bandwidth=kde_bandwidth)

            all_resamples = current_analysis.bootstrap_with_kde(num_bootstrap_resamples,
                                                                num_cores=num_cores,
                                                                chunksize=chunk_size)

            # Provides a list of estimated (mean1, mean2) for a bigaussian
            fit_vals = bigaussian_fit_analysis_1(all_resamples, x_vals,
                                                 fixed_mean=min(means[c_i], 0.),
                                                 init_mean=means[c_i], init_std_dev=std_dev,
                                                 num_cores=num_cores,
                                                 chunksize=chunk_size,
                                                 ratio=.5)

            lower_means[c_i] = np.mean(fit_vals[:, 0])
            upper_means[c_i] = np.mean(fit_vals[:, 1])
            lower_min_conf[c_i], lower_max_conf[c_i] = np.percentile(fit_vals[:, 0], [lower_interval, upper_interval])
            upper_min_conf[c_i], upper_max_conf[c_i] = np.percentile(fit_vals[:, 1], [lower_interval, upper_interval])

        return means, lower_means, upper_means, lower_min_conf, lower_max_conf, upper_min_conf, upper_max_conf

    # ------------------------------------------------------------------------------------------------------------ #
    def get_flare(self, flare_index):
        return self._flare_catalog.get_flare(flare_index)

    def set_outlier_mask(self, flare_mask):
        self._outlier_mask = flare_mask

    @property
    def outlier_mask(self):
        return self._outlier_mask
