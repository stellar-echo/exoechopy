
"""
This module provides functions that are useful in computing things.
Will likely be broken into separate modules once I know all of the math that is needed.
"""

import decimal
import numpy as np
import multiprocessing as multi
from scipy import stats
from astropy import units as u
from astropy import constants
from scipy import signal
from .constants import FunctionType

__all__ = ['angle_between_vectors', 'vect_from_spherical_coords', 'angle_between_vector_array',
           'compute_lag', 'compute_lag_simple',
           'SphericalLatitudeGen', 'stochastic_flare_process',
           'gaussian', 'bi_erf_model', 'bigaussian_model',
           'window_range',
           'take_noisy_derivative', 'take_noisy_2nd_derivative', 'linear_detrend', 'gaussian_fft_filter',
           'asymmetric_filter_1',
           'round_dec', 'row_col_grid', 'is_nested', 'find_nearest_index', 'find_near_index_floor',
           'PipePool']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Vector math


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Smallest angle between two unnormalized vectors

    Parameters
    ----------
    v1
        Vector 1
    v2
        Vector 2

    Returns
    -------
    float
        Angle in radians as a float
    """
    return 2*np.arctan2(np.linalg.norm(np.linalg.norm(v2)*v1-np.linalg.norm(v1)*v2),
                        np.linalg.norm(np.linalg.norm(v2)*v1+np.linalg.norm(v1)*v2))


def angle_between_vector_array(v1: np.ndarray, vector_array: np.ndarray) -> float:
    """Smallest angle between two unnormalized vectors

    Parameters
    ----------
    v1
        Vector 1
    vector_array
        Array of vectors to test

    Returns
    -------
    float
        Angle in radians as a float
    """
    x1 = np.linalg.norm(np.linalg.norm(vector_array,
                                       axis=1)[:, np.newaxis] * v1 - np.linalg.norm(v1) * vector_array, axis=1)
    x2 = np.linalg.norm(np.linalg.norm(vector_array,
                                       axis=1)[:, np.newaxis] * v1 + np.linalg.norm(v1) * vector_array, axis=1)
    return 2 * np.arctan2(x1, x2)


def vect_from_spherical_coords(longitude, latitude) -> np.ndarray:
    vect = np.array((np.sin(latitude) * np.cos(longitude),
                     np.sin(latitude) * np.sin(longitude),
                     np.cos(latitude) * np.ones(np.shape(longitude))))
    return np.transpose(vect)


def compute_lag(start_vect: u.Quantity,
                echo_vect: u.Quantity,
                detect_vect: np.ndarray,
                return_v2_norm: bool=False) -> u.Quantity:
    """Calculate the lag from a flare echo

    Parameters
    ----------
    start_vect
        Starting point of the flare
    echo_vect
        Location of the echo-ing object
    detect_vect
        *Normalized* vector in the direction of the detection (Earth)
    return_v2_norm
        Whether or not to return the norm of echo_vect - start_vect
        Since this value is often used in other calculations, can reduce the number of times it's calculated.

    Returns
    -------
    u.Quantity
        Echo lag quantity

    """
    # Vector from the starting point to the echo source:
    v2 = echo_vect-start_vect
    v2_norm = u.Quantity(np.linalg.norm(v2), v2.unit)
    if return_v2_norm:
        return (v2_norm-u.Quantity(np.dot(v2, detect_vect), v2.unit))/constants.c, v2_norm
    else:
        return (v2_norm-u.Quantity(np.dot(v2, detect_vect), v2.unit))/constants.c


c_au_s = constants.c.to(u.au/u.s).value


def compute_lag_simple(echo_vect: np.ndarray,
                       detect_vect: np.ndarray) -> np.ndarray:
    """Calculate the lag from a flare echo, assuming source is at origin, and ignoring units

    Assumes vector is in AU and provides lag in seconds

    Parameters
    ----------
    echo_vect
        Location of the echo-ing object
    detect_vect
        *Normalized* vector in the direction of the detection (Earth)

    Returns
    -------
    np.ndarray
        Echo lag quantity

    """
    # Vector from the starting point to the echo source:
    v2_norm = np.linalg.norm(echo_vect, axis=-1)
    return (v2_norm-np.dot(echo_vect, detect_vect, ))/c_au_s


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Stats math


class SphericalLatitudeGen(stats.rv_continuous):
    def __init__(self, a=0, b=np.pi, min_val=0, max_val=np.pi, **kwargs):
        super().__init__(a=a, b=b, **kwargs)
        self._minval = min_val
        self._maxval = max_val

    def _pdf(self, x):
        return np.sin(x)/(np.cos(self.a) - np.cos(self.b))


def stochastic_flare_process(stop_value,
                             distribution: stats.rv_continuous,
                             start_value=0,
                             max_iter=1000, *dist_args):
    # TODO implement a faster version, such as generating many values at once and identifying where it exceeds stop_val
    ct = 0
    all_values = []
    total_displacement = start_value
    while ct < max_iter:
        next_val = distribution.rvs(*dist_args)
        test_val = total_displacement + next_val
        if test_val < stop_value:
            all_values.append(test_val)
            total_displacement += next_val
        else:
            break
    return all_values


def gaussian(x, mu, sigma):
    """Simple Gaussian distribution"""
    denom = np.sqrt(np.pi * 2) * sigma
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / denom


def bigaussian_model(data: np.ndarray,
                     amp: float,
                     mean1: float,
                     mean2: float,
                     sigma: float,
                     ratio: float=.5) -> np.ndarray:
    """Produces a pair of gaussians with identical amplitude and sigma but different means

    Parameters
    ----------
    data
        Data to apply model to
    amp
        Amplitude of the Gaussians
    mean1
        Mean of Gaussian 1
    mean2
        Mean of Gaussian 2
    sigma
        Single-Gaussian standard deviation
    ratio
        Amplitude ratio of Gaussian 1 to Gaussian 2 [0 to 1]

    Returns
    -------
    float or np.ndarray
        Value of the bigaussian at the values specified in data

    """
    denom = 2*sigma**2
    return amp*(ratio*np.exp(-(data-mean1)**2/denom)+(1-ratio)*np.exp(-(data-mean2)**2/denom))


def bi_erf_model(data, mean1, mean2, sigma, ratio: float=.5):
    """Produces a pair of error functions with identical standard deviations (to match bigaussian_model)

    Parameters
    ----------
    data
        Data to apply model to
    mean1
        Mean of Gaussian 1
    mean2
        Mean of Gaussian 2
    sigma
        Standard deviation
    ratio
        Amplitude ratio of Gaussian 1 to Gaussian 2

    Returns
    -------

    """
    return ratio*stats.norm.cdf(data, loc=mean1, scale=sigma)+(1-ratio)*stats.norm.cdf(data, loc=mean2, scale=sigma)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Plot math

def window_range(ind: int, max_ind: int, width: int) -> [int, int]:
    """Constrain a sub-window from exceeding a range while maintaing a fixed width

    Parameters
    ----------
    ind
        "central" index
    max_ind
        Largest index value allowed-- typically len(whatever)-1
    width
        fixed window width

    Returns
    -------
    [int, int]
        Lowest index, highest index

    """
    w1 = width//2 + width%2
    w2 = width//2
    min1 = ind - w1
    max2 = ind + w2
    if min1 < 0:
        return [0, width-1]
    elif max2 >= max_ind:
        return [max_ind - width, max_ind-1]
    else:
        return [ind - w1, ind + w2 - 1]


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Signal processing math

def take_noisy_derivative(data_array: np.ndarray,
                          window_size: int=19,
                          sample_cadence: float=0.2) -> np.ndarray:
    """Applies Savitzky-Golay filter to estimate the first derivative, which is robust to noise

    Essentially an easy-to-remember wrapper for the scipy savgol_filter

    Parameters
    ----------
    data_array
        Data for analysis
    window_size
        Size of window for polynomial fit
    sample_cadence
        Scale factor for derivative

    Returns
    -------
    np.ndarray
        Returns the Savitzky-Golay estimate of the first derivative
    """
    return signal.savgol_filter(data_array,
                                window_length=window_size,
                                polyorder=2,
                                deriv=1,
                                delta=sample_cadence,
                                mode='mirror')


def take_noisy_2nd_derivative(data_array: np.ndarray,
                              window_size: int=19,
                              sample_cadence: float=0.2) -> np.ndarray:
    """Applies Savitzky-Golay filter to estimate the second derivative, which is robust to noise

    Essentially an easy-to-remember wrapper for the scipy savgol_filter

    Parameters
    ----------
    data_array
        Data for analysis
    window_size
        Size of window for polynomial fit
    sample_cadence
        Scale factor for derivative

    Returns
    -------
    np.ndarray
        Returns the Savitzky-Golay estimate of the second derivative
    """
    return signal.savgol_filter(data_array,
                                window_length=window_size,
                                polyorder=2, deriv=2, delta=sample_cadence,
                                mode='mirror')


def linear_detrend(data_array: np.ndarray) -> np.ndarray:
    """Subtract a linear fit from a data set

    Parameters
    ----------
    data_array
        Array to subtract background from

    Returns
    -------
    np.ndarray
        data_array with a linear fit removed
    """
    if isinstance(data_array, u.Quantity):
        unit = data_array.unit
        data_array = data_array.value
    else:
        unit = None
    xvals = range(len(data_array))
    fit = np.poly1d(np.polyfit(xvals, data_array, deg=1))
    if unit is None:
        return data_array-fit(xvals)
    else:
        return u.Quantity(data_array-fit(xvals), unit)


def gaussian_fft_filter(data_array, radius):
    filter_kernel = signal.windows.gaussian(radius*12, std=radius)
    filter_kernel /= np.sum(filter_kernel)
    filtered_signal = data_array - signal.fftconvolve(data_array.copy(), filter_kernel, mode='same')
    return filtered_signal


def asymmetric_filter_1(width: int, skew: float):
    """
    Generate a multi-derivative filter.
    Skew=0 gives 2nd derivative, skew=1 gives 1st derivative, in between interpolates between them.
    Empirically, it seems that the inverse decay constant squared makes for a good skew
    """
    width = max(5, 4*(int(width//4))+1)  # Must satisfy N*4+1, N an integer
    half_index = int(width//2)+1
    quarter_index = int(width//4)+1
    u1 = np.bartlett(half_index)
    deriv_filter = np.zeros(width)
    deriv2_filter = -np.bartlett(width)
    deriv_filter[:half_index] = -u1
    deriv_filter[half_index-1:] = u1
    deriv2_filter[quarter_index-1:-quarter_index+1] += 2*u1

    dot_array = -skew*deriv_filter + (1-skew)*deriv2_filter
    return dot_array


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Indexing and integer math

def round_dec(val: float) -> int:
    """Round a number to an integer in the way that decimal rounding is expected to behave

    In python 3, round(2.5) = 2, round(3.5) = 4.
    This can really screw up indexing that relies on decimal rounding.
    This function rounds the way I expect a number that is entered like a decimal to be rounded.

    Parameters
    ----------
    val
        Value to be rounded

    Returns
    -------
    int
        The round-half-up version of val

    """
    return int(decimal.Decimal(val).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def row_col_grid(num_pts: int) -> (int, int):
    """Generate a number of rows and columns for displaying a grid of num_pts objects

    Parameters
    ----------
    num_pts
        Number of values to break into rows and columns

    Returns
    -------
    tuple
        rows, columns

    """
    return int(np.sqrt(num_pts)), int(np.ceil(num_pts/int(np.sqrt(num_pts))))


def is_nested(iterable) -> bool:
    """Determines if an iterable has lists in it

    Parameters
    ----------
    iterable
        Iterable object, typically a list

    Returns
    -------
    bool
        True if a nested list is found
        False if not
    """
    try:
        _ = iter(iterable)
        return any(isinstance(i, list) for i in iterable)
    except TypeError:
        return False


def find_nearest_index(array: np.ndarray, value: float):
    """Identify the index in the array that best matches the requested value.
    Derived from https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array"""
    idx = (np.abs(array - value)).argmin()
    return idx


def find_near_index_floor(array: u.Quantity, value: u.Quantity):
    """Identify the first index from an array that closely matches the requested value.  Used for finding start times"""
    unit = value.unit
    array = array.to(unit)
    idx = max(0, (array < value).argmin() - 1)
    return idx

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Multiprocessing support

class PipePool:
    """
    Manually implemented asynchronous pipe-based pool-like system.
    No error handling implemented yet.
    Hopefully handles memory better than starmap/map_async/apply_async,
    where all data is saved on memory until pool is complete.
    However, those approaches are viable for a lot of systems.
    """
    def __init__(self,
                 worker_func,
                 output_obj: np.ndarray,
                 iter_kwargs: list,
                 noniter_kwargs: dict=None):
        """Create a pool-like object to perform a process in parallel

        Parameters
        ----------
        worker_func
            Target function, must be structured as f(pipe_access, *args, **kwargs)
            Target function should return (index for output_obj, vals to put into output_obj)
            - pipe_access is used to send data back to the Pool from the thread (see _pipe_wrapped_worker)
            - index is used to determine where in output_obj the data should be placed (see _index_wrapper)
        output_obj
            The indexable object that the PipePool should place results into
        iter_kwargs
            List of kwargs to initialize pipe with
        noniter_kwargs
            Optional list of re-used kwargs to initialize pipe with
            Useful if you need to use the same data over and over and making copies would be memory intensive
        """
        self._worker_func = worker_func
        self._output_obj = output_obj
        self._iterable_kwargs = iter_kwargs
        if noniter_kwargs is None:
            noniter_kwargs = {}
        self._noniterable_kwargs = noniter_kwargs

    def run(self, num_cores: int=None, chunksize: int=1):
        """Implement the PipePool

        Parameters
        ----------
        num_cores
            Number of cores to use in doling out processes
        chunksize
            Approximate number of iterables to pass at a time to limit communication on pipe

        """
        if num_cores is None:
            num_cores = max(multi.cpu_count()-1, 1)
        all_pipes = []
        all_processes = []

        # Add new jobs if available cores / work to perform:
        while len(all_pipes) < min(num_cores, len(self._iterable_kwargs)):
            pipe_local, pipe_remote = multi.Pipe()
            # Initialize the separate process:
            p = multi.Process(target=self._worker_func,
                              args=(pipe_remote,),
                              kwargs=self._noniterable_kwargs)
            all_pipes.append(pipe_local)
            all_processes.append(p)
            p.start()
            num_iters = min(chunksize, len(self._iterable_kwargs))
            pipe_local.send([self._iterable_kwargs.pop() for xi in range(num_iters)])

        while len(self._iterable_kwargs) > 0:
            # See if there are any jobs that have completed:
            for pipe in all_pipes:
                if pipe.poll():
                    results = pipe.recv()
                    for ind, val in results:
                        self._output_obj[ind] = val
                    num_iters = min(chunksize, len(self._iterable_kwargs))
                    pipe.send([self._iterable_kwargs.pop() for xi in range(num_iters)])
                    if len(self._iterable_kwargs) == 0:
                        break

        # Close it out with blocking:
        for pipe in all_pipes:
            results = pipe.recv()
            for ind, val in results:
                self._output_obj[ind] = val
        for proc, pipe in zip(all_processes, all_pipes):
            pipe.send('close')
            proc.terminate()


def _index_wrapper(ind, func, *args, **kwargs):
    """Takes an index as the first arg and returns it as the first value in a tuple"""
    return ind, func(*args, **kwargs)


def _pipe_wrapped_worker(pipe, func, *args, **kwargs):
    """Takes a function and sends its return back through the pipe"""
    result = func(*args, **kwargs)
    pipe.send(result)
    pipe.close()



