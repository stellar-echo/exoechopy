
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
            Data points for evaluation

        Returns
        -------
        np.ndarray
            Kernel density estimate at the requested points
        """
        if not isinstance(xvals, np.ndarray):
            xvals = np.array(xvals)
        dx = xvals[1] - xvals[0]
        bw2 = self._bandwidth/2
        output_data = np.zeros(xvals.shape)
        lower_range = self.dataset-bw2
        upper_range = self.dataset+bw2
        x_0 = min(xvals)
        lower_bins = np.floor((lower_range-x_0)/dx)
        upper_bins = np.floor((upper_range-x_0)/dx)
        # TODO Handle edge cases!  Then handle easy cases


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
        return asdf

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

