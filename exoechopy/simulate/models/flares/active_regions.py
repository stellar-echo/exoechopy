
"""
This module provides active region classes and methods for stars.
"""

import numpy as np
from scipy import stats
from .flares import *
from astropy import units as u

__all__ = ['FlareCollection', 'ActiveRegion']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class FlareCollection:
    """Data container for a collection of delta-like flares."""
    def __init__(self, number_of_flares=0):
        """
        :param int number_of_flares: How many flares.
        """
        self._N_flares = int(number_of_flares)
        self._flare_times_list = []
        self._flare_intensities_list = []
        self._flare_decay_const_list = []
        self._flare_onset_duration_list = []
        self._flare_vector_list = []  # Array of (Longitude, Latitude)'s

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def num_flares(self):
        return self._N_flares

    @num_flares.setter
    def num_flares(self, num_flares):
        self._N_flares = num_flares

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def all_flare_times(self):
        return self._flare_times_list

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def all_flare_intensities(self):
        return self._flare_intensities_list

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def all_flare_decays(self):
        return self._flare_decay_const_list

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def all_flare_onsets(self):
        return self._flare_onset_duration_list

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def all_flare_locations(self):
        return self._flare_vector_list


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class ActiveRegion:
    """Base active region class."""
    def __init__(self,
                 flare_type=None,
                 intensity_pdf=None,
                 decay_pdf=None,
                 onset_pdf=None,
                 occurrence_freq_pdf=None,
                 longitude_pdf=None,
                 latitude_pdf=None):
        """
        Generates an active region, which produces flares with a variety of tunable statistical properties.

        The distributions are expected to be scipy.stats distributions, but it's not currently required.
        Use other distributions at your own risk.

        :param ProtoFlare flare_type: Class for generating flare shapes
        :param intensity_pdf: Distribution function for generating flare intensities
        :param decay_pdf: Distribution function for generating flare decay constants
        :param onset_pdf: Distribution function for generating flare onset times
        :param occurrence_freq_pdf: Distribution function for determining flaring based on flaring frequency
        :param longitude_pdf: Distribution function for determining flare longitudes
        :param latitude_pdf: Distribution function for determining flare latitudes
        """
        if isinstance(flare_type, ProtoFlare):

        else:
            raise TypeError("flare_type must be a subclass of ProtoFlare")




