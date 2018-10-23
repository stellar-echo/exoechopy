
"""
This module provides different algorithms for resampling datasets to produce confidence intervals
"""

import numpy as np
from scipy.stats import gaussian_kde

__all__ = ['GaussianKDE', 'TophatKDE', 'ResampleAnalysis']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class KDE_base:
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

    def set_bandwidth(self, bw_method):
        pass

    def integrate_between(self, low_val, high_val):
        pass


class GaussianKDE(gaussian_kde, KDE_base):
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


class TophatKDE(KDE_base):
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
        elif not isinstance(bw_method, str):
            self._bandwidth = bw_method
        else:
            msg = "`bw_method` should be 'scott', 'silverman', or a scalar "
            raise ValueError(msg)


class ResampleAnalysis:
    """Class for performing various resampling methods on a dataset"""
    def __init__(self,
                 dataset: np.ndarray,
                 x_domain: np.ndarray=None,
                 kde: KDE_base=None
                 ):
        self._dataset = dataset
        self._xdomain = x_domain
        self._kde = kde

