"""
This module provides different algorithms for identifying, extracting, and manipulating flares from lightcurve data.
"""

import numpy as np
from scipy import signal
from astropy import units as u

__all__ = ['BaseFlareCatalog', 'LightcurveFlareCatalog', 'find_peaks_stddev_thresh']


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

class BaseFlareCatalog:
    def __init__(self,
                 all_flares: (list, np.ndarray) = None,
                 extract_range: (tuple, list) = None,
                 cadence: u.Quantity = None,
                 flare_weights: np.ndarray = None
                 ):
        # Distance before and after flare peak to extract:
        self._look_back = None
        self._look_forward = None
        if extract_range is not None:
            self.set_extraction_range(extract_range)

        self._slice_indices = None
        self._all_flares = all_flares
        self._correlator_lag_matrix = None
        self._flare_times = None
        self._flare_weights = flare_weights
        self._index_order_tuple = None

        self._cadence = cadence

    # ------------------------------------------------------------------------------------------------------------ #
    def set_extraction_range(self, extract_range):
        self._look_back = extract_range[0]
        self._look_forward = extract_range[1]

    # ------------------------------------------------------------------------------------------------------------ #
    def generate_correlator_matrix(self, *args):
        """Generates a correlator matrix from a set of defined processes.

        Processes should be tuples of a function and the associated kwargs to pass to the function.

        Parameters
        ----------
        args
            Functions or tuples of (func, kwargs) to run on the data
            Example arguments:
            correlation_process = (autocorrelate_array, {'min_lag': min_lag, 'max_lag': max_lag})
            generate_correlator_matrix(correlation_process)

        """
        test_correlation = self._all_flares[0]
        for arg in args:
            try:
                # Test to see if this is a (func, kwarg) pair:
                _ = iter(arg)
                test_correlation = arg[0](test_correlation, **arg[1])
            except TypeError:
                test_correlation = arg(test_correlation)

        self._correlator_lag_matrix = np.zeros((len(self._all_flares), len(test_correlation)))
        self._correlator_lag_matrix[0] = test_correlation
        for ii in range(1, len(self._all_flares)):
            test_correlation = self._all_flares[ii]
            for arg in args:
                try:
                    # Test to see if this is a (func, kwarg) pair:
                    _ = iter(arg)
                    test_correlation = arg[0](test_correlation, **arg[1])
                except TypeError:
                    test_correlation = arg(test_correlation)
            self._correlator_lag_matrix[ii] = test_correlation
        self.set_sampling_order()

    def get_correlator_matrix(self):
        if self._correlator_lag_matrix is None:
            raise ValueError("Correlator matrix is not initialized, run generate_correlator_matrix() first")
        else:
            return self._correlator_lag_matrix

    # ------------------------------------------------------------------------------------------------------------ #
    def set_sampling_order(self, order=None):
        """Allows resampling of lags by changing up their order

        Parameters
        ----------
        order
            New ordering, e.g., [3, 0, 100, 9, ...]
            Can include repeats (e.g., bootstrapping)

        """
        if order is None:
            self._index_order_tuple = tuple(range(len(self._all_flares)))
        else:
            self._index_order_tuple = tuple(order)

    def run_lag_hypothesis(self,
                           lag_array,
                           func=None,
                           resample_order=None,
                           flare_weights=False,
                           flare_mask=None,
                           lag_weights=False,
                           *func_args):
        """Provide a list of lag times to extract correlator values from

        Parameters
        ----------
        lag_array
            Lag indices to test (integers)
        func
            Optional function to run on the resulting lags
            Ex: np.mean
        resample_order
            Optional new order for the flares to have occurred
        flare_weights
            If True, applies weights to the correlator_lag_matrix result
            Implemented as w_i * c_i/np.mean(w)
        flare_mask
            Optional mask to ignore certain flares, good for outlier rejection
        lag_weights
            Placeholder, not yet implemented.  If True, will apply lag-based weights to the correlator_lag_matrix result
            This can be helpful for reducing the weight of short-time lags, which are closer to the peak.
            Will probably allow a unique lag_weight for each flare, since some flares have different FWHM's, which
            would change when to cut-off the short-time lag
        func_args
            Optional args to pass to the process function

        Returns
        -------
        np.ndarray
        """

        if resample_order is None:
            if flare_mask is None:
                self.set_sampling_order()
            else:
                lag_array = lag_array[flare_mask]
                self.set_sampling_order(np.arange(len(self._all_flares))[flare_mask])
        else:
            self.set_sampling_order(resample_order)
            if flare_mask is not None:
                lag_array = lag_array[flare_mask]

        if flare_weights:
            weights = self._flare_weights[np.array(self._index_order_tuple)]
            return_array = weights * self._correlator_lag_matrix[self._index_order_tuple, lag_array] / np.mean(weights)
        else:
            return_array = self._correlator_lag_matrix[self._index_order_tuple, lag_array]

        if func is not None:
            return_array = func(return_array, *func_args)

        # Reset the sampling order if necessary:
        if resample_order is not None:
            self.set_sampling_order()

        return return_array

    # ------------------------------------------------------------------------------------------------------------ #
    def get_flare_curves(self) -> np.ndarray:
        if self._all_flares is not None and self._look_forward is not None:
            return self._all_flares[:, self._look_back:self._look_forward]
        else:
            return self._all_flares.copy()

    def get_flare(self, flare_index):
        return self._all_flares[flare_index].copy()

    def get_flare_times(self):
        return self._flare_times.copy()

    @property
    def cadence(self):
        return self._cadence

    def set_cadence(self, cadence):
        self._cadence = cadence

    @property
    def num_flares(self):
        if self._all_flares is None:
            return 0
        else:
            return len(self._all_flares)

    # ------------------------------------------------------------------------------------------------------------ #
    def generate_weights_with_protocol(self, func, **kwargs):
        """Applies a function to a flare that produces a single-valued output representative of a weight

        The array of weights is stored in a dictionary, keyed by weight_name

        Parameters
        ----------
        func
            A function that provides a flare metric (e.g., np.max)
        kwargs
            Any kwargs to pass on to the func

        """
        if self._all_flares is None:
            raise ValueError("No flares initialized.")
        weights = np.zeros(self.num_flares)
        for f_i, flare in enumerate(self._all_flares):
            weights[f_i] = func(flare, **kwargs)
        self._flare_weights = weights

    def get_weights(self):
        return self._flare_weights.copy()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

class LightcurveFlareCatalog(BaseFlareCatalog):
    def __init__(self,
                 lightcurve: np.ndarray,
                 extract_range: (tuple, list) = None,
                 time_domain: np.ndarray = None,
                 cadence: u.Quantity = None):
        """Extracts flares from a lightcurve and runs simple experiments on the collection of flares.

        This version extracts all flare profiles to be the same number of indices so that they can be stored
        in a single ndarray without masking.

        Parameters
        ----------
        lightcurve
            The original lightcurve to process
        extract_range
            A pair of numbers that defines the range around the peak of a flare: (look_back, look_forward)
            This determines how long before and after the peak of a flare to include in the subsequent processing
        time_domain
            Time domain associated with the lightcurve (since the cadence may not be pre-determined)
        """
        if time_domain is None:
            super().__init__(all_flares=None, extract_range=extract_range, cadence=cadence)
        else:
            super().__init__(all_flares=None, extract_range=extract_range, cadence=time_domain[1] - time_domain[0])
        self._lightcurve = lightcurve
        self._time_domain = time_domain
        self._flare_indices = []

    # ------------------------------------------------------------------------------------------------------------ #
    def set_extraction_range(self, extract_range):
        self._look_back = extract_range[0]
        self._look_forward = extract_range[1]
        self._generate_slices()

    # ------------------------------------------------------------------------------------------------------------ #
    def identify_flares_with_protocol(self, func, **kwargs):
        """Applies the protocol to the lightcurve, generating a list of candidate flares

        Parameters
        ----------
        func
            Function to use to identify flares
            If the function requires args, they should be passed as a tuple with a kwargs dictionary
            Args are applied as:
            flare_indices = func(lightcurve, **kwargs)
        """
        self._flare_indices = func(self._lightcurve, **kwargs)
        self._generate_slices()

    # ------------------------------------------------------------------------------------------------------------ #
    def get_flare_indices(self):
        return self._flare_indices.copy()

    # ------------------------------------------------------------------------------------------------------------ #
    def get_flare_curves(self) -> np.ndarray:
        if self._all_flares is None:
            self._generate_flare_array()
        return self._all_flares.copy()

    # ------------------------------------------------------------------------------------------------------------ #
    def _generate_slices(self):
        if len(self._flare_indices) > 0 and self._look_back is not None and self._look_forward is not None:
            # Collect all flares where the beginning and end slice are not on the edge of the dataset:
            self._flare_indices = [f_i for f_i in self._flare_indices
                                   if f_i - self._look_back >= 0 and f_i + self._look_forward < len(self._lightcurve)]
            self._slice_indices = [(f_i - self._look_back, f_i + self._look_forward) for f_i in self._flare_indices]
            self._generate_flare_array()

    # ------------------------------------------------------------------------------------------------------------ #
    def _generate_flare_array(self):
        if self._slice_indices is not None:
            self._all_flares = np.zeros((len(self._slice_indices), self._look_forward + self._look_back))
            for ii, (i_0, i_N) in enumerate(self._slice_indices):
                self._all_flares[ii] = self._lightcurve[i_0:i_N]
            if self._time_domain is not None:
                self._flare_times = self._time_domain[self._flare_indices]
        else:
            raise ValueError("Extraction range is not initialized, run set_extraction_range() first")

    # ------------------------------------------------------------------------------------------------------------ #
    def generate_correlator_matrix(self, *args):
        """Generates a correlator matrix from a set of defined processes.

        Processes should be tuples of a function and the associated kwargs to pass to the function.

        Parameters
        ----------
        args
            Functions or tuples of (func, kwargs) to run on the data
            Example arguments:
            correlation_process = (autocorrelate_array, {'min_lag': min_lag, 'max_lag': max_lag})
            generate_correlator_matrix(correlation_process)

        """
        if self._all_flares is None:
            self._generate_slices()

        super().generate_correlator_matrix(*args)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def find_peaks_stddev_thresh(lightcurve_data: np.ndarray,
                             std_dev_threshold: float = None,
                             smoothing_radius: float = None,
                             min_index_gap: int = None,
                             extra_search_pad: int = None,
                             num_smoothing_rad: int = None,
                             single_flare_gap: int = None) -> np.ndarray:
    """Relatively low-overhead peak finder for an array based on deviation above a threshold.

    Has some filtering functionality to prevent detection of multiple peaks within the same flare event.
    Works best if slowly varying background has already been subtracted.
    
    Parameters
    ----------
    lightcurve_data
        Input data with low or subtracted low-frequency variability
    std_dev_threshold
        Threshold multiples of sigma above background to consider for peak extraction
        Defaults to sigma=1
    smoothing_radius
        Optional Gaussian smoothing filter to run prior to peak finding, helps eliminate salt & pepper noise
    min_index_gap
        Require each peak be at least this many indices from another peak
        To avoid two peaks in the same window, make this small
        To avoid a noisy event to be registered as multiple events, make this large
    extra_search_pad
        Add an optional extra search on either side of a peak, useful if Gaussian is aggressive
    num_smoothing_rad
        How many points to include in filter window, based on smoothing_radius
    single_flare_gap
        If two flares occur within this gap, remove both from the list

    Returns
    -------
    numpy.ndarray(dtype=int)
        Array containing the indices at which peaks were detected in signal_data

    """

    # Using median instead of mean, assumes activity is primarily excess photon events (like flares)
    # Mean would be biased by strong peak events, median less-so
    filtered_lightcurve = lightcurve_data - np.median(lightcurve_data)

    if std_dev_threshold is None:
        std_dev_threshold = 1.

    if smoothing_radius is not None:
        if num_smoothing_rad is None:
            num_smoothing_rad = 5  # Number of radii to include in Gaussian
        _num_window_points = max(smoothing_radius * num_smoothing_rad, 5)  # Ensure a minimum of 5 points in window
        _window = signal.gaussian(_num_window_points, smoothing_radius)
        _window /= np.sum(_window)
        filtered_lightcurve = signal.fftconvolve(filtered_lightcurve, _window, 'same')

    if extra_search_pad is None:
        extra_search_pad = 0

    filtered_lightcurve /= np.std(filtered_lightcurve)

    excess_region_mask = np.where(filtered_lightcurve > std_dev_threshold)[0]  # Find all points that exceed threshold
    delta_mask = excess_region_mask[1:] - excess_region_mask[:-1]  # Computes gap in indices between peaky events
    distinct_regions = np.where(delta_mask > min_index_gap)[0] + 1  # array is 1 shorter, need +1 to align indices

    # Initialize the return array, now that we know how many peaks we found:
    return_array = np.zeros(len(distinct_regions) + 1, dtype=int)

    # Uses the actual max from the original unfiltered lightcurve, so the maxima can be influenced by noise.
    # Goes through each region that exceeded the threshold and picks the highest point, probably a better way to do this
    min_search_ind = excess_region_mask[0]
    for region_i in range(len(distinct_regions)):
        max_search_ind = excess_region_mask[distinct_regions[region_i] - 1]
        return_array[region_i] = np.argmax(
            lightcurve_data[min_search_ind - extra_search_pad:max_search_ind + extra_search_pad]) + \
                                 min_search_ind - extra_search_pad
        min_search_ind = excess_region_mask[distinct_regions[region_i]]
    last_region = excess_region_mask[-1]
    return_array[region_i + 1] = np.argmax(
        lightcurve_data[min_search_ind - extra_search_pad:last_region + extra_search_pad]) + \
                                 min_search_ind - extra_search_pad

    if single_flare_gap is not None:
        nearby_flares = np.where(np.diff(return_array) < single_flare_gap)[0]
        return_array = np.delete(return_array, np.concatenate((nearby_flares, nearby_flares + 1)))

    return return_array
