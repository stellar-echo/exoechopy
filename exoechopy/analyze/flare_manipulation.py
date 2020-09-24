"""
This module provides different algorithms for identifying, extracting, and manipulating flares from lightcurve data.
"""

import numpy as np
from scipy import signal
from astropy import units as u
from scipy import optimize
from ..utils import find_near_index_floor
from ..simulate import ParabolicRiseExponentialDecay

__all__ = ['BaseFlareCatalog', 'LightcurveFlareCatalog', 'find_peaks_stddev_thresh', 'FitFlareSuperresolutionPRED']


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
                           phase_weights=None,
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
        phase_weights
            Additional weight to apply, typically driven by a planet phase function, and not stored in the FlareCatalog
            If both phase_weights and an internal flare_weight are provided, the total weight is given by:
            phase_weights * flare_weight
        func_args
            Optional args to pass to the process function

        Returns
        -------
        : np.ndarray
        """

        if resample_order is None:
            if flare_mask is None:
                self.set_sampling_order()
            else:
                lag_array = lag_array[flare_mask]
                if phase_weights is not None:
                    phase_weights = phase_weights[flare_mask]
                self.set_sampling_order(np.arange(len(self._all_flares))[flare_mask])
        else:
            self.set_sampling_order(resample_order)
            if flare_mask is not None:
                lag_array = lag_array[flare_mask]
                if phase_weights is not None:
                    phase_weights = phase_weights[flare_mask]

        if flare_weights:
            weights = self._flare_weights[np.array(self._index_order_tuple)]
            if phase_weights is not None:
                weights *= phase_weights
            weights /= np.mean(weights)
            return_array = weights * self._correlator_lag_matrix[self._index_order_tuple, lag_array]
        else:
            if phase_weights is not None:
                weights = phase_weights
                weights /= np.mean(weights)
                return_array = weights * self._correlator_lag_matrix[self._index_order_tuple, lag_array]
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
                             min_index_gap: int = 1,
                             extra_search_pad: int = 5,
                             num_smoothing_rad: int = 5,
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
    filtered_lightcurve = lightcurve_data - np.nanmedian(lightcurve_data)

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

    filtered_lightcurve /= np.nanstd(filtered_lightcurve)

    excess_region_mask = np.nonzero(filtered_lightcurve > std_dev_threshold)[0]  # Find all points that exceed threshold
    # print("excess_region_mask: ", excess_region_mask)
    delta_mask = excess_region_mask[1:] - excess_region_mask[:-1]  # Computes gap in indices between peaky events
    distinct_regions = np.nonzero(delta_mask > min_index_gap)[0] + 1  # array is 1 shorter, need +1 to align indices

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


def sum_frames(frames, frames_per_sum, frame_offset=0):
    all_vals = []
    for f_i, frame in enumerate(frames[frame_offset:]):
        if f_i % frames_per_sum == 0:
            all_vals.append(frame)
        else:
            all_vals[-1] += frame
    return u.Quantity(all_vals, all_vals[0].unit)


def sum_frames_lw(frames, frames_per_sum, frame_offset=0):
    all_vals = []
    for f_i, frame in enumerate(frames[frame_offset:]):
        if f_i % frames_per_sum == 0:
            all_vals.append(frame)
        else:
            all_vals[-1] += frame
    return np.array(all_vals)


class FitFlareSuperresolutionPRED:
    """Class for identifying probable flares signatures based on an observed time-coarsened flare.  In development"""

    def __init__(self,
                 time_domain: u.Quantity,
                 flux_array: u.Quantity,
                 int_time: u.Quantity,
                 read_time: u.Quantity,
                 frame_sum: int):
        self._time_domain_lw = time_domain.to('s').value
        self._time_domain = time_domain.copy()
        self._flux_array = flux_array.copy()
        self._flux_array_lw = self._flux_array.to('ct').value
        self._int_time = int_time.copy()
        self._int_time_lw = int_time.to('s').value
        self._read_time = read_time.copy()
        self._read_time_lw = read_time.to('s').value
        self._frame_cadence = self._int_time + self._read_time
        self._frame_cadence_lw = self._frame_cadence.to('s').value
        self._data_cadence = self._frame_cadence * frame_sum
        self._data_cadence_lw = self._frame_cadence_lw * frame_sum
        self._frame_sum = frame_sum
        self._bounds = [(1, np.inf),  # amplitude, in counts
                        (1E-1, np.inf),  # rise time
                        (1E-2, np.inf),  # decay const
                        (self._time_domain_lw[0], self._time_domain_lw[-1])]  # flare time

        if np.abs((self._time_domain[1] - self._time_domain[0]) - self._data_cadence).to(u.s).value > 1E-6:
            raise AttributeError("Time domain and cadence do not match within tolerance: "
                                 "\nCadence: ", self._data_cadence,
                                 "\ndt", self._time_domain[1] - self._time_domain[0])

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def frame_cadence(self):
        return self._frame_cadence.copy()

    @property
    def data_cadence(self):
        return self._data_cadence.copy()

    @property
    def int_time(self):
        return self._int_time.copy()

    @property
    def read_time(self):
        return self._read_time.copy()

    @property
    def time_domain(self):
        return self._time_domain.copy()

    @property
    def flux_array(self):
        return self._flux_array.copy()

    # ------------------------------------------------------------------------------------------------------------ #
    # TODO - consolidate common sections of exact_flare, integrate_PRED_flare_with_readout?
    def exact_flare(self,
                    amplitude: u.Quantity,
                    peak_time_abs: u.Quantity,
                    rise_time: u.Quantity,
                    decay_const: u.Quantity,
                    max_decay_const=10,
                    resolution=5):
        flare_model = ParabolicRiseExponentialDecay(onset=rise_time,
                                                    decay=decay_const,
                                                    max_decay=max_decay_const)
        local_cadence = self.data_cadence / resolution
        flare_duration = flare_model.flare_duration + local_cadence
        num_frames = int(flare_duration / local_cadence)
        # Relative time local to flare:
        # all_times = u.Quantity(np.arange(0,
        #                                  (local_cadence * num_frames).to('s').value,
        #                                  local_cadence.value), 's')[:num_frames]
        time_domain_HD = u.Quantity(np.linspace(self._time_domain[0].value,
                                                self._time_domain[-1].value,
                                                len(self._time_domain) * resolution), 's')
        # print("Exact flare all_times: ", all_times)
        # print("Exact flare time_domain_HD: ", time_domain_HD)
        flare_start_time_exact = peak_time_abs - rise_time
        print("flare_start_time_exact: ", flare_start_time_exact)
        # TODO - handle case when flare start time is prior to data start
        flare_bin_start_ind = find_near_index_floor(time_domain_HD, flare_start_time_exact)
        print("Exact flare flare_bin_start_ind: ", flare_bin_start_ind)
        flare_start_time_binned = time_domain_HD[flare_bin_start_ind]
        print("exact flare_start_time_binned: ", flare_start_time_binned)
        time_shift = time_domain_HD[0] + flare_start_time_exact - flare_start_time_binned
        print("exact time_shift: ", time_shift)
        # all_times -= local_cadence + time_shift
        # time_domain_HD -= local_cadence + time_shift
        integrated_flare = amplitude * flare_model.evaluate_over_array(time_domain_HD - local_cadence - time_shift)

        max_ind = min(len(time_domain_HD), flare_bin_start_ind + num_frames - 1)
        flare_times = time_domain_HD[flare_bin_start_ind:max_ind]

        return_values = np.zeros(len(time_domain_HD))
        return_values[flare_bin_start_ind:max_ind] = integrated_flare[:len(flare_times)]
        return time_domain_HD, return_values

    def integrate_PRED_flare_with_readout(self,
                                          amplitude: u.Quantity,
                                          peak_time_abs: u.Quantity,
                                          rise_time: u.Quantity,
                                          decay_const: u.Quantity,
                                          max_decay_const=10):
        """
        Integrates a flare, dropping lost photons during readout.


        Parameters
        ----------
        amplitude
            Flare amplitude in counts
        peak_time_abs
            Flare peak time, referenced to class time_domain
        rise_time
            Parabolic rise--peak of flare occurs at rise_time
        decay_const
            Exponential decay
        max_decay_const
            How many decay constants to calculate after the flare to avoid C0 discontinuities

        Returns
        -------
        u.Quantity(flux values) registered with the class's time_domain

        """
        flare_model = ParabolicRiseExponentialDecay(onset=rise_time,
                                                    decay=decay_const,
                                                    max_decay=max_decay_const)
        print("flare_model.flare_duration: ", flare_model.flare_duration)
        flare_duration = flare_model.flare_duration + self._frame_cadence
        num_frames = int(flare_duration / self.frame_cadence)

        # Ensure we have an integer-divisible number of frames:
        if num_frames % self._frame_sum != 0:
            num_frames += self._frame_sum - num_frames % self._frame_sum

        integrated_frames = num_frames // self._frame_sum

        # Relative time local to flare:
        flare_local_time = u.Quantity(np.zeros(num_frames * 2), 's')
        int_times = u.Quantity(np.arange(0,
                                         (self.frame_cadence * num_frames).to('s').value,
                                         self._frame_cadence_lw), 's')[:num_frames]
        read_times = int_times + self.int_time
        flare_local_time[::2] = int_times
        flare_local_time[1::2] = read_times

        flare_start_time_exact = peak_time_abs - rise_time
        flare_bin_start_ind = find_near_index_floor(self._time_domain,
                                                    flare_start_time_exact)
        flare_start_time_binned = self._time_domain[flare_bin_start_ind]

        time_shift = flare_start_time_exact - flare_start_time_binned
        flare_local_time -= time_shift
        integrated_flare = amplitude * flare_model.evaluate_over_array(flare_local_time)
        keep_vals = integrated_flare[::2]

        # TODO - verify max ind is selected correctly
        max_ind = min(len(self._time_domain), flare_bin_start_ind + integrated_frames)
        flare_times = self._time_domain[flare_bin_start_ind:max_ind]

        return_values = np.zeros(len(self._time_domain))
        return_values[flare_bin_start_ind:max_ind] = sum_frames(keep_vals, self._frame_sum)[:len(flare_times)]

        return return_values

    def _integrate_PRED_flare_with_readout_lw(self,
                                              amplitude: float,
                                              peak_time_abs: float,
                                              rise_time: float,
                                              decay_const: float,
                                              max_decay_const=10):
        """
        Integrates a flare, dropping lost photons during readout.


        Parameters
        ----------
        peak_time_abs
            Flare peak time, referenced to class time_domain
        rise_time
            Parabolic rise--peak of flare occurs at rise_time
        decay_const
            Exponential decay
        max_decay_const
            How many decay constants to calculate after the flare to avoid C0 discontinuities

        Returns
        -------
        u.Quantity(flux values) registered with the class's time_domain

        """
        flare_model = ParabolicRiseExponentialDecay(onset=u.Quantity(rise_time, 's'),
                                                    decay=u.Quantity(decay_const, 's'),
                                                    max_decay=max_decay_const)
        print("flare_model.flare_duration: ", flare_model.flare_duration)
        flare_duration = flare_model.flare_duration.to(u.s).value + self._frame_cadence_lw
        num_frames = int(flare_duration / self._frame_cadence_lw)

        # Ensure we have an integer-divisible number of frames:
        if num_frames % self._frame_sum != 0:
            num_frames += self._frame_sum - num_frames % self._frame_sum

        integrated_frames = num_frames // self._frame_sum

        # Relative time local to flare:
        flare_local_time = np.zeros(num_frames * 2)
        int_times = np.arange(0, self._frame_cadence_lw * num_frames, self._frame_cadence_lw)[:num_frames]
        read_times = int_times + self._int_time_lw
        flare_local_time[::2] = int_times
        flare_local_time[1::2] = read_times

        flare_start_time_exact = peak_time_abs - rise_time
        print("peak_time_abs: ", peak_time_abs)
        print("flare_start_time_exact: ", flare_start_time_exact)
        # TODO - handle case when flare start time is prior to data start
        flare_bin_start_ind = find_near_index_floor(self._time_domain,
                                                    u.Quantity(flare_start_time_exact, u.s))
        flare_start_time_binned = self._time_domain_lw[flare_bin_start_ind]

        time_shift = flare_start_time_exact - flare_start_time_binned
        flare_local_time -= time_shift
        integrated_flare = amplitude * flare_model.evaluate_over_array(u.Quantity(flare_local_time, 's')).value
        keep_vals = integrated_flare[::2]

        # TODO - verify max ind is selected correctly
        max_ind = min(len(self._time_domain_lw), flare_bin_start_ind + integrated_frames)
        flare_times = self._time_domain_lw[flare_bin_start_ind:max_ind]

        return_values = np.zeros(len(self._time_domain_lw))
        return_values[flare_bin_start_ind:max_ind] = sum_frames_lw(keep_vals, self._frame_sum)[:len(flare_times)]

        return return_values

    def fitting_cost_function_PRED(self,
                                   opt_params):
        """
        Draft cost function for minimization of difference with real light curve.  Anticipate indexing issues.

        Parameters
        ----------
        opt_params
            Packed optimization parameters:
            rise_time_s
                Rise time, float, specified in seconds.  To be optimized.
            decay_const_s
                Decay constant, float, specified in seconds.  To be optimized.
            seg_peak_time_s
                Time of flare peak in the relative frame of the obs_time_domain, float, specified in seconds.  To be optimized.
        obs_time_domain
            Observation time domain
        obs_lightcurve
            Actual observed lightcurve
        int_time
            Integration time for the detector
        read_time
            Readout time for the detector
        frames_per_sum
            Number of frames summed into the actual data product
        frame_offset
            How many cadences to shift the frame summation by
            Note, this requires compensation of the flare position externally (-cadence * frame_offset)

        Returns
        -------

        """
        amplitude_ct, rise_time_s, decay_const_s, seg_peak_time_s = opt_params
        est_flare = self._integrate_PRED_flare_with_readout_lw(amplitude_ct,
                                                               seg_peak_time_s,
                                                               rise_time_s,
                                                               decay_const_s)

        # Ignore last datapoint just in case the est_flare frame shift artificially reduced the signal in the last bin
        diff = (self._flux_array_lw[:len(est_flare) - 1] - est_flare[:-1])
        cost_func = np.dot(diff, diff)
        return cost_func

    def fit_flare(self,
                  init_guess_amplitude: u.Quantity,
                  init_guess_rise_time: u.Quantity,
                  init_guess_decay_const: u.Quantity,
                  init_guess_flare_time: u.Quantity):
        initial_guesses = np.array((init_guess_amplitude.to(u.ct).value,
                                    init_guess_rise_time.to(u.s).value,
                                    init_guess_decay_const.to(u.s).value,
                                    init_guess_flare_time.to(u.s).value))
        result = optimize.minimize(self.fitting_cost_function_PRED,
                                   x0=initial_guesses,
                                   bounds=self._bounds)
        params = result.x
        amp = u.Quantity(params[0], 'ct')
        rise = u.Quantity(params[1], 's')
        decay = u.Quantity(params[2], 's')
        flare_time = u.Quantity(params[3], 's')
        result_dict = {'amplitude': amp,
                       'rise_time': rise,
                       'decay_const': decay,
                       'flare_time': flare_time,
                       'cost': result.fun}
        return result_dict
