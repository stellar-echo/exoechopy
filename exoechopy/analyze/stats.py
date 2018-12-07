
"""
This module provides different algorithms for resampling datasets to produce confidence intervals
"""

import numpy as np

from scipy.stats import gaussian_kde
from ..utils import FunctionType
from typing import Type
from scipy import optimize
from ..utils.math_operations import PipePool
from ..utils.math_operations import bi_erf_model

__all__ = ['GaussianKDE', 'TophatKDE', 'ResampleAnalysis', 'ResampleKDEAnalysis',
           'bigaussian_fit_analysis_1', 'curvefit_passable_wrapper']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class BaseKDE:
    """Kernel density estimator base class"""
    def __init__(self,
                 dataset: np.ndarray,
                 bandwidth: (float, str)):
        """Parent for Kernel Density Estimators, used to combine scipy and locally cooked up methods under one hood

        Parameters
        ----------
        dataset
            Data points for inclusion in dataset
        bandwidth
            String or float.  Typical strings include 'silverman' and 'scott'
        """
        self.dataset = dataset
        self._sigma = np.std(dataset)
        self.set_bandwidth(bandwidth)

    def __call__(self, x_vals: np.ndarray):
        return None

    def set_bandwidth(self, bw_method):
        pass

    def integrate_between(self, low_val, high_val):
        pass


class GaussianKDE(gaussian_kde, BaseKDE):
    """Wrapper for the scipy.stats.gaussian_kde class, only designed for 1D arrays"""
    def __init__(self,
                 dataset: np.ndarray,
                 bandwidth: (float, str)=None):
        """

        Parameters
        ----------
        dataset
            Data points for inclusion in kernel density estimate
        bandwidth
            String or float.  Typical strings include 'silverman' and 'scott'
        """
        super().__init__(dataset=dataset, bw_method=bandwidth)

    def integrate_between(self, low_val: float, high_val: float) -> float:
        """Computes the integral between the two specified bounds

        Parameters
        ----------
        low_val
            Lower bound of integration
        high_val
            Upper bound of integration

        Returns
        -------
        float
            The computed integral

        """
        return self.integrate_box_1d(low_val, high_val)


class TophatKDE(BaseKDE):
    """1D tophat Kernel density estimator"""
    def __init__(self,
                 dataset: np.ndarray,
                 bandwidth: (float, str)=None,
                 weights: np.ndarray=None):
        """

        Parameters
        ----------
        dataset
            Data points for inclusion in kernel density estimate
        bandwidth
            String or float.  Typical strings include 'silverman' and 'scott'
        weights
            Optional array of weights for the values
        """
        self._bandwidth = None
        super(TophatKDE, self).__init__(dataset=dataset, bandwidth=bandwidth)
        if weights is None:
            self._weights = np.ones(dataset.shape)
        else:
            self._weights = np.array(weights)
        self._total_weight = np.sum(self._weights)

    def __call__(self, xvals: np.ndarray) -> np.ndarray:
        """Returns the computed KDE at these points

        Parameters
        ----------
        xvals
            Data points for evaluation, uniformly spaced

        Returns
        -------
        np.ndarray
            Kernel density estimate at the requested points
        """
        if not isinstance(xvals, np.ndarray):
            xvals = np.array(xvals)
        dx = xvals[1] - xvals[0]
        bw2 = self._bandwidth/2.
        lower_range = self.dataset-bw2
        upper_range = self.dataset+bw2
        x_0 = min(xvals)
        lower_bins = np.floor((lower_range-x_0)/dx).astype(int)
        upper_bins = np.ceil((upper_range-x_0)/dx).astype(int)
        bin_width = np.array(upper_bins-lower_bins, dtype=float)
        # If all data fits inside a single bin, make sure it doesn't get reduced to 0-width:
        bin_width[bin_width == 0] = 1.
        data_mask = (upper_bins >= 0) & (lower_bins < len(xvals))
        summation_values = self._weights/(bin_width*dx)
        output_data = np.zeros(len(xvals))
        for l_i, u_i, s_val in zip(lower_bins[data_mask], upper_bins[data_mask], summation_values[data_mask]):
            output_data[l_i:u_i] += s_val
        output_data /= self._total_weight
        return output_data

    def integrate_between(self, low_val: float, high_val: float) -> float:
        """Computes the integral between the two specified bounds

        Parameters
        ----------
        low_val
            Lower bound of integration
        high_val
            Upper bound of integration

        Returns
        -------
        float
            The computed integral
        """
        bw2 = self._bandwidth/2.
        lower_range = self.dataset-bw2
        upper_range = self.dataset+bw2
        multiplier = self._weights.copy()
        # Handle edge cases, crop them proportionately
        edge_cases_lower = (lower_range < low_val) & (upper_range >= low_val)
        edge_cases_upper = (lower_range < high_val) & (upper_range >= high_val)
        multiplier[edge_cases_lower] *= (upper_range[edge_cases_lower]-low_val)/self._bandwidth
        multiplier[edge_cases_upper] *= (high_val-lower_range[edge_cases_upper])/self._bandwidth
        data_mask = (upper_range >= low_val) * (lower_range < high_val)
        return np.sum(multiplier[data_mask])/self._total_weight

    def set_bandwidth(self, bw_method):
        if bw_method is None:
            bw_method = 'scott'
        if bw_method == 'scott':
            self._bandwidth = np.power(len(self.dataset), -1/5.)*self._sigma
        elif bw_method == 'silverman':
            self._bandwidth = np.power(len(self.dataset)*3/4, -1/5.)*self._sigma
        elif callable(bw_method):
            self._bandwidth = bw_method(self.dataset)
        elif not isinstance(bw_method, str):
            self._bandwidth = bw_method
        else:
            msg = "`bw_method` should be 'scott', 'silverman', a scalar, or callable"
            raise ValueError(msg)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class ResampleAnalysis:
    """Class for performing various resampling methods on a dataset"""
    def __init__(self, dataset: np.ndarray):
        self._dataset = dataset
        self._num_pts = len(dataset)

    # ------------------------------------------------------------------------------------------------------------ #
    def jackknife(self,
                  analysis_func: FunctionType = None,
                  conf_lvl: float = None,
                  **analysis_func_kwargs):
        """Performs jackknife analysis on the dataset provided, including some various analysis methods.

        Note: This is not a wrapper for the astropy jackknife_stats method.  Though it's close.

        Parameters
        ----------
        analysis_func
            Function to apply to the subsampled data
        conf_lvl
            If None (default), returns just the resampled dataset
            If an analysis_func is provided, returns the computed values and the lower and upper values of that interval
            The confidence interval is computed using the np.percentile function.
        analysis_func_kwargs
            Arguments to pass on to the analysis_func after passing the dataset under analysis

        Returns
        -------
        np.ndarray or (np.ndarray, np.ndarray, float, float)
            If an analysis_func or confidence intervals are provided, returns 4 terms:
                (jackknifed_data, analysis_func(jackknifed_data), lower interval, upper interval
            These are replaced by None when not called
        """
        subsample_size = self.num_pts-1

        if analysis_func is None:
            jackknifed_data = np.zeros((self.num_pts, subsample_size))
        else:
            try:
                jackknifed_data = np.zeros((self.num_pts,
                                            len(analysis_func(self._dataset[:subsample_size], **analysis_func_kwargs))))
            except TypeError:
                jackknifed_data = np.zeros(self.num_pts)
        for b_i in range(self.num_pts):
            resampled_indices = np.delete(np.arange(0, self.num_pts), b_i)
            if analysis_func is None:
                jackknifed_data[b_i] = self._dataset[resampled_indices]
            else:
                jackknifed_data[b_i] = analysis_func(self._dataset[resampled_indices], **analysis_func_kwargs)

        if analysis_func is not None:
            diff = jackknifed_data - analysis_func(self._dataset)
            distance = np.sqrt(diff**2)
        else:
            distance = None

        if conf_lvl is None:
            if analysis_func is None:
                return jackknifed_data
            else:
                return jackknifed_data, distance, None, None
        else:
            if 0 < conf_lvl < 1:
                upper_interval = 100*(1 - (1 - conf_lvl)/2)
                lower_interval = 100*((1 - conf_lvl)/2)
                return jackknifed_data, distance, \
                       np.percentile(jackknifed_data, lower_interval, axis=-1), \
                       np.percentile(jackknifed_data, upper_interval, axis=-1)
            else:
                raise ValueError("conf_lvl must be in (0, 1)")

    # ------------------------------------------------------------------------------------------------------------ #
    def bootstrap(self,
                  num_resamples: int,
                  subsample_size: int=None,
                  analysis_func: FunctionType = None,
                  conf_lvl: float=None,
                  **analysis_func_kwargs):
        """Perform bootstrap analysis on the dataset provided, including some various analysis methods

        Note: This is not a wrapper for the astropy boostrap method.  Though it's close.

        Parameters
        ----------
        num_resamples
            Number of bootstrap samples to test
        subsample_size
            If None (default), uses the same size as the original dataset.
            If an integer is given, will sample this many subsamples from the original dataset.
        analysis_func
            If None (default), returns the resampled data without analyzing it
            An optional analysis function to perform on the bootstrap resampled data (e.g., np.mean)
        conf_lvl
            If None (default), returns just the resampled dataset
            If an analysis_func is provided, returns the computed values and the lower and upper values of that interval
            The confidence interval is computed using the np.percentile function.
        analysis_func_kwargs
            Arguments to pass on to the analysis_func (dataset is the first argument, kwargs follow)

        Returns
        -------
        np.ndarray or (np.ndarray, float, float)
            If not using confidence intervals, returns the bootstrap resamples
            If using a confidence interval, also returns the lower and upper intervals
        """
        if subsample_size is None:
            subsample_size = self.num_pts

        if analysis_func is None:
            bootstrapped_data = np.zeros((num_resamples, subsample_size))
        else:
            try:
                bootstrapped_data = np.zeros((num_resamples,
                                              len(analysis_func(self._dataset[:subsample_size], **analysis_func_kwargs))))
            except TypeError:
                bootstrapped_data = np.zeros(num_resamples)
        for b_i in range(num_resamples):
            resampled_indices = np.random.choice(self.num_pts, subsample_size, replace=True)
            if analysis_func is None:
                bootstrapped_data[b_i] = self._dataset[resampled_indices]
            else:
                bootstrapped_data[b_i] = analysis_func(self._dataset[resampled_indices], **analysis_func_kwargs)

        if conf_lvl is None:
            return bootstrapped_data
        else:
            if 0 < conf_lvl < 1:
                upper_interval = 100*(1 - (1 - conf_lvl)/2)
                lower_interval = 100*((1 - conf_lvl)/2)
                return bootstrapped_data, \
                       np.percentile(bootstrapped_data, lower_interval, axis=-1), \
                       np.percentile(bootstrapped_data, upper_interval, axis=-1)
            else:
                raise ValueError("conf_lvl must be in (0, 1)")

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def num_pts(self):
        return self._num_pts

    # ------------------------------------------------------------------------------------------------------------ #
    def remove_outliers_by_jackknife_sigma_testing(self,
                                                   analysis_func: FunctionType,
                                                   num_sigma: float = None,
                                                   **analysis_func_kwargs):
        """Removes outliers from the data by performing jackknife testing,
        then removing any data points beyond num_sigma deviations from the analysis_func metric

        Parameters
        ----------
        analysis_func
            Function to apply to the subsampled data
        num_sigma
            Number of deviations a single datapoint must move the distribution to be eliminated
        analysis_func_kwargs
            Arguments to pass on to the analysis_func after passing the dataset under analysis

        Returns
        -------
        np.ndarray
            Values removed from the dataset are returned

        """
        jackknifed_data, deviation, _, _ = self.jackknife(analysis_func, **analysis_func_kwargs)
        jackknifed_data -= np.mean(jackknifed_data)
        jackknifed_data /= np.std(jackknifed_data)
        inlier_mask = np.abs(jackknifed_data) < num_sigma
        outliers = self._dataset[~inlier_mask]
        self._dataset = self._dataset[inlier_mask]
        self._num_pts = len(self._dataset)
        return outliers

    # ------------------------------------------------------------------------------------------------------------ #
    def apply_function(self,
                       analysis_func: FunctionType,
                       **analysis_func_kwargs):
        """Apply a function to a copy of the dataset

        Basically just there to help access the dataset during analysis without exposing it to changes

        Parameters
        ----------
        analysis_func
            Function to be applied
        analysis_func_kwargs
            Optional arguments to pass after passing dataset

        Returns
        -------
            Result of the function
        """
        return analysis_func(self._dataset.copy(), **analysis_func_kwargs)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class ResampleKDEAnalysis(ResampleAnalysis):
    """Class for performing various resampling methods on kernel density estimates of a dataset"""

    def __init__(self,
                 dataset: np.ndarray,
                 x_domain: np.ndarray = None,
                 kde: Type[BaseKDE] = None,
                 kde_bandwidth: (float, str) = None,
                 weights: np.ndarray = None
                 ):
        """Simplifies some resampling analysis when it is necessary to generate a Kernel Density Estimate each resample

        Parameters
        ----------
        dataset
            Data for analysis
        x_domain
            Domain to apply to the KDE function
        kde
            Kernel density estimator class
        kde_bandwidth
            Bandwidth method/value/callable to use for the KDE
        weights
            Optional weights for the dataset values
        """
        super().__init__(dataset)
        self._xdomain = x_domain
        self._kde = kde
        self._kde_bandwidth = kde_bandwidth
        self._weights = weights

    # ------------------------------------------------------------------------------------------------------------ #
    def compute_kde(self,
                    use_weights: bool=False):
        """Computes the current KDE based

        Parameters
        ----------
        use_weights
            Whether or not to apply weights to the calculation (only available if the KDE supports weights)

        Returns
        -------
        np.ndarray, np.ndarray
            Returns the domain and the KDE computed over the domain
        """
        if use_weights:
            if self._weights is None:
                raise TypeError("Weight is None, but compute_kde has been told to use them.")
            return self._xdomain, self._kde(self._dataset,
                                            bandwidth=self._kde_bandwidth,
                                            weights=self._weights)(self._xdomain)
        else:
            return self._xdomain, self._kde(self._dataset, bandwidth=self._kde_bandwidth)(self._xdomain)

    # ------------------------------------------------------------------------------------------------------------ #
    def bootstrap_with_kde(self,
                           num_resamples: int,
                           subsample_size: int=None,
                           analysis_func: FunctionType=None,
                           conf_lvl: float=None,
                           use_weights: bool=False,
                           num_cores: int=1,
                           chunksize: int=1,
                           **analysis_func_kwargs):
        """Resamples a sample with replacement, applies the KDE, then performs analysis and repeats

        Confidence intervals are calculated pointwise

        Parameters
        ----------
        num_resamples
            Number of times to resample the dataset
        subsample_size
            Number of samples in the subsample to use in generating the data
        analysis_func
            Optional function to apply to the KDE-evaluated result
        conf_lvl
            Optional float or list of floats (e.g., [.6, .95, .98] to evaluate three different intervals)
            Each value must be 0 < x < 1
        use_weights
            Boolean, whether or not to use weights, if implemented for the KDE method
        num_cores
            Number of cores to use in running the resampling analysis.
            Recommend using multiprocessing.cpu_count()-1, often
        chunksize
            Number of iterables to pass at a time, if using multiple cores
        analysis_func_kwargs

        Returns
        -------
        np.ndarray or (np.ndarray, np.ndarray)
            If no conf_lvl provided, returns the pointwise resampled kernel density estimate
            If confidence intervals are provided, also provides their upper and lower intervals
        """
        if self._xdomain is None:
            raise ValueError("x_domain is not defined")

        if subsample_size is None:
            subsample_size = self.num_pts

        if conf_lvl is not None:
            try:
                _ = iter(conf_lvl)
            except TypeError:
                conf_lvl = [conf_lvl]

        if use_weights:
            if self._weights is None:
                raise TypeError("Weight is None, but bootstrap_with_kde has been told to use them.")

        # Note, we pre-select these values so that later multiprocessing doesn't accidentally reuse random seeds
        bootstrap_indices = np.random.choice(self.num_pts, (num_resamples, subsample_size), replace=True)

        if analysis_func is None:
            output_data = np.zeros((num_resamples, len(self._xdomain)))
        else:
            try:
                output_data = np.zeros((num_resamples,
                                        len(analysis_func(self._kde(self._dataset[:subsample_size],
                                                                    bandwidth=self._kde_bandwidth)(self._xdomain),
                                                          **analysis_func_kwargs))))
            except TypeError:
                output_data = np.zeros(num_resamples)

        if num_cores == 1:
            for r_i, resample in enumerate(bootstrap_indices):
                if use_weights:
                    new_kde = self._kde(self._dataset[resample],
                                        bandwidth=self._kde_bandwidth,
                                        weights=self._weights[resample])
                else:
                    new_kde = self._kde(self._dataset[resample],
                                        bandwidth=self._kde_bandwidth)

                if analysis_func is None:
                    output_data[r_i] = new_kde(self._xdomain)
                else:
                    output_data[r_i] = analysis_func(new_kde(self._xdomain), **analysis_func_kwargs)
        else:
            assert num_cores > 1
            # Generate the iterables and initialization parameters:
            if use_weights:
                kwargs_list = [{'index': r_i,
                                'dataset': self._dataset[subsample],
                                'weights': self._weights[subsample]}
                               for r_i, subsample in enumerate(bootstrap_indices)]
                non_iter_kwargs = {'xdomain': self._xdomain,
                                   'kde': self._kde,
                                   'kde_band': self._kde_bandwidth,
                                   'use_weights': use_weights,
                                   'analysis_func': analysis_func,
                                   'analysis_func_kwargs': analysis_func_kwargs}
            else:
                kwargs_list = [{'index': r_i,
                                'dataset': self._dataset[subsample]}
                               for r_i, subsample in enumerate(bootstrap_indices)]
                non_iter_kwargs = {'xdomain': self._xdomain,
                                   'kde': self._kde,
                                   'kde_band': self._kde_bandwidth,
                                   'use_weights': use_weights,
                                   'analysis_func': analysis_func,
                                   'analysis_func_kwargs': analysis_func_kwargs}
            # Generate and run the multicore solver:
            resample_pool = PipePool(worker_func=_resample_pipe_func,
                                     output_obj=output_data,
                                     iter_kwargs=kwargs_list,
                                     noniter_kwargs=non_iter_kwargs)
            resample_pool.run(num_cores=num_cores, chunksize=chunksize)

        if conf_lvl is None:
            return output_data
        else:
            confidence_data = np.zeros((len(conf_lvl), 2, len(self._xdomain)))
            for c_i, interval in enumerate(conf_lvl):
                if 0 < interval < 1:
                    lower_interval = 100 * ((1 - interval) / 2)
                    upper_interval = 100 * (1 - (1 - interval) / 2)
                    confidence_data[c_i, 0, :] = np.percentile(output_data, lower_interval, axis=0)
                    confidence_data[c_i, 1, :] = np.percentile(output_data, upper_interval, axis=0)
                else:
                    raise ValueError("conf_lvl must be in (0, 1)")
            return output_data, confidence_data

    # ------------------------------------------------------------------------------------------------------------ #
    def set_x_domain(self, x_domain: np.ndarray):
        """Redefine the domain over which the kernel density estimators are computed

        Parameters
        ----------
        x_domain
            Values for the domain

        """
        self._xdomain = x_domain


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def _resample_pipe_func(pipe_access, xdomain,
                        kde, kde_band,
                        use_weights,
                        analysis_func, analysis_func_kwargs):
    """Function used for communicating with the PipePool and handling resampling analysis

    Parameters
    ----------
    pipe_access
    xdomain
    kde
    kde_band
    use_weights
    analysis_func
    analysis_func_kwargs

    Returns
    -------

    """
    alive = True
    while alive:
        # try:
        if pipe_access.poll():
            list_of_new_args = pipe_access.recv()
            if list_of_new_args == 'close':
                alive = False
                pipe_access.close()
            else:
                result = []
                for new_input in list_of_new_args:
                    if use_weights:
                        new_kde = kde(new_input['dataset'],
                                      bandwidth=kde_band,
                                      weights=new_input['weights'])
                    else:
                        new_kde = kde(new_input['dataset'],
                                      bandwidth=kde_band)

                    if analysis_func is None:
                        output_data = new_kde(xdomain)
                    else:
                        output_data = analysis_func(new_kde(xdomain), **analysis_func_kwargs)
                    result.append((new_input['index'], output_data))
                pipe_access.send(result)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

# Define a custom processing algorithm:
def approximate_cdf(y_data):
    density_cdf = np.cumsum(y_data)
    density_cdf -= np.min(density_cdf)
    density_cdf /= np.max(density_cdf)
    return density_cdf


def bigaussian_fit_analysis_1(all_resamples, x_vals, num_cores=1, chunksize=1,
                              fixed_mean=0., init_mean=0., init_std_dev=None, ratio=.5):
    fit_vals = np.zeros((len(all_resamples), 2))
    if init_std_dev is None:
        init_std_dev = np.std(x_vals)/2
    if num_cores == 1:
        for r_i, resample in enumerate(all_resamples):
            cdf = approximate_cdf(resample)
            # We create a lambda function just to pin one of the two Gaussians to zero, let the other float
            # Could define a specific version of this function, but often you'll want to set a value other than zero
            try:
                popt_bierf, pcov_bierf = optimize.curve_fit(lambda x, m2, s: bi_erf_model(x, fixed_mean, m2, s, ratio),
                                                            x_vals, cdf,
                                                            p0=[init_mean + init_std_dev, init_std_dev * ratio],
                                                            absolute_sigma=True, maxfev=1000)
                popt_bierf = [x for x in popt_bierf]
                popt_bierf.insert(1, fixed_mean)
                fit_vals[r_i, :] = min(popt_bierf[:2]), max(popt_bierf[:2])
            except RuntimeError:
                fit_vals[r_i, :] = np.nan, np.nan
    else:
        assert num_cores > 1
        kwargs_list = [{'index': r_i,
                        'dataset': resample} for r_i, resample in enumerate(all_resamples)]
        non_iter_kwargs = {'xdomain': x_vals,
                           'init_mean': init_mean,
                           'fixed_mean': fixed_mean,
                           'init_std_dev': init_std_dev,
                           'ratio': ratio}

        # Generate and run the multicore solver:
        resample_pool = PipePool(worker_func=_bigaussian_pipe_func,
                                 output_obj=fit_vals,
                                 iter_kwargs=kwargs_list,
                                 noniter_kwargs=non_iter_kwargs)
        resample_pool.run(num_cores=num_cores, chunksize=chunksize)

    return fit_vals


def _bigaussian_pipe_func(pipe_access, xdomain, fixed_mean, init_mean, init_std_dev, ratio):
    alive = True
    while alive:
        # try:
        if pipe_access.poll():
            list_of_new_args = pipe_access.recv()
            if list_of_new_args == 'close':
                alive = False
                pipe_access.close()
            else:
                result = []
                for new_input in list_of_new_args:
                    resample = new_input['dataset']

                    cdf = approximate_cdf(resample)
                    # We create a lambda function just to pin one of the two Gaussians to zero, let the other float
                    # Could define a specific version of this function,
                    # but often you'll want to set a value other than zero
                    popt_bierf, pcov_bierf = optimize.curve_fit(lambda x, m2, s: bi_erf_model(x,
                                                                                              fixed_mean,
                                                                                              m2,
                                                                                              s,
                                                                                              ratio),
                                                                xdomain, cdf,
                                                                p0=[init_mean + init_std_dev, init_std_dev / 2],
                                                                absolute_sigma=True)
                    popt_bierf = [x for x in popt_bierf]
                    popt_bierf.insert(1, fixed_mean)
                    output_data = min(popt_bierf[:2]), max(popt_bierf[:2])

                    result.append((new_input['index'], output_data))
                pipe_access.send(result)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def curvefit_passable_wrapper(datapoints: np.ndarray,
                              fit_func: FunctionType=None,
                              xdata: np.ndarray=None,
                              **curvefit_kwargs):
    """Provides a wrapper around the scipy.optimize.curve_fit function to make it easier to use in bootstrapping

    Parameters
    ----------
    datapoints
        y-values, typically required as first field for passed functions
    fit_func
        Function to use in the curve_fit routine
    xdata
        x-values, provides the domain for analysis
    curvefit_kwargs
        Additional kwargs to pass, such as p0=[...] or absolute_sigma=True

    Returns
    -------

    """
    opt_fit, pcov_fit = optimize.curve_fit(fit_func, xdata=xdata, ydata=datapoints, **curvefit_kwargs)
    return opt_fit

