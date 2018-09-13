
"""
This module provides active region classes and methods for stars.
"""

import warnings
import numpy as np
from scipy import stats
from .flares import *
from exoechopy.utils import *
from astropy import units as u
from astropy.utils.exceptions import AstropyUserWarning

__all__ = ['FlareCollection', 'FlareActivity', 'Region', 'ActiveRegion']


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class FlareCollection:
    """Data container for a collection of delta-like flares."""
    def __init__(self, number_of_flares=0):
        """
        :param int number_of_flares: How many flares.
        """
        self._N_flares = int(number_of_flares)

        # Flare collection dict can be expanded to hold other lists of args for specialty flares
        self._flare_collection_dict = {'flare_times_list': [],
                                       'flare_intensities_list': []}
        # Common dict add-ons:
        # 'flare_decay_const_list': [],
        # 'flare_onset_duration_list': []

        self._flare_collection_dict_final = {}

        self._flare_vector_list = []  # Array of (Longitude, Latitude)'s
        self._flare_vector_array = None

    # ------------------------------------------------------------------------------------------------------------ #
    def compile_collection(self):
        """
        Initially, lists are used so that it's easier to add and remove flares.
        But before running the simulation, it's faster to have np.array objects, so we switch dicts
        :return:
        """
        all_keys = self._flare_collection_dict.keys()
        for key in all_keys:
            self._flare_collection_dict_final[key] = np.array(self._flare_collection_dict[key])
            if len(self._flare_collection_dict_final[key]) != self.num_flares:
                raise IndexError(key+" is not "+str(self.num_flares)+" iterables long")
        self._flare_vector_array = np.array(self._flare_vector_list)
        if len(self._flare_vector_array) != self.num_flares:
            raise IndexError("_flare_vector_array is not " + str(self.num_flares) + " iterables long")

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
        return self._flare_collection_dict['flare_times_list']

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def all_flare_intensities(self):
        return self._flare_collection_dict['flare_intensities_list']

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def all_flare_locations(self):
        return self._flare_vector_list


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class FlareActivity:
    """Base flare activity class."""
    def __init__(self,
                 flare_type: ProtoFlare=None,
                 intensity_pdf: PDFType=None,
                 decay_pdf: PDFType = None,
                 onset_pdf: PDFType = None):
        """Provides a flare generator for an ActiveRegion."""
        if isinstance(flare_type, ProtoFlare):
            self._flare_type = flare_type
        else:
            raise TypeError("flare_type must be a child class of ProtoFlare")

        if isinstance(intensity_pdf, PDFType):
            if isinstance(intensity_pdf, IntFloat):
                self._intensities = intensity_pdf
        else:
            raise TypeError("intensity_pdf must be a float, list, or frozen stats pdf")


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class Region:
    """Base region class."""
    def __init__(self,
                 longitude_pdf: PDFType=None,
                 latitude_pdf: PDFType=None):
        """Provides a region to determine where activity occurs for an ActiveRegion.
        :param longitude_pdf: If None, defaults to 0-2pi uniform distribution
        :param latitude_pdf: If None or a [min, max], defaults to distribution that is uniform *on a sphere*
        (see http://mathworld.wolfram.com/SpherePointPicking.html)
        If a custom distribution is provided, be aware of the spherical area element!
        """
        # TODO add placeholder for intensity variations, probably spectral objects (maybe as Child)
        self._longitude_pdf = None
        self.longitude_pdf = longitude_pdf

        self._latitude_pdf = None
        self.latitude_pdf = latitude_pdf

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def longitude_pdf(self):
        return self._longitude_pdf

    @longitude_pdf.setter
    def longitude_pdf(self, longitude_pdf):
        if longitude_pdf is None:
            self._longitude_pdf = stats.uniform(loc=0, scale=2*np.pi)
        elif isinstance(longitude_pdf, PDFType):
            if isinstance(longitude_pdf, CountType):
                self._longitude_pdf = longitude_pdf
            elif isinstance(longitude_pdf, (list, tuple)):
                self._longitude_pdf = stats.uniform(loc=longitude_pdf[0], scale=longitude_pdf[1]-longitude_pdf[0])
            elif isinstance(longitude_pdf, stats._distn_infrastructure.rv_frozen):
                self._longitude_pdf = longitude_pdf
            else:
                AstropyUserWarning("longitude_pdf is unanticipated type, proceed with caution")
                self._longitude_pdf = longitude_pdf
        else:
            raise TypeError("longitude_pdf must be None or PDFType")

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def latitude_pdf(self):
        return self._longitude_pdf

    @latitude_pdf.setter
    def latitude_pdf(self, latitude_pdf):
        if latitude_pdf is None:
            self._latitude_pdf = stats.uniform(loc=0, scale=1)
        elif isinstance(latitude_pdf, PDFType):
            if isinstance(latitude_pdf, CountType):
                self._latitude_pdf = latitude_pdf
            elif isinstance(latitude_pdf, (list, tuple)):
                self._latitude_pdf = SphericalLatitudeGen()(loc=latitude_pdf[0],
                                                            scale=latitude_pdf[1] - latitude_pdf[0])
            elif isinstance(latitude_pdf, stats._distn_infrastructure.rv_frozen):
                self._latitude_pdf = latitude_pdf
            else:
                AstropyUserWarning("latitude_pdf is unanticipated type, proceed with caution")
                self._latitude_pdf = latitude_pdf
        else:
            raise TypeError("latitude_pdf must be None or PDFType")

    # ------------------------------------------------------------------------------------------------------------ #
    def get_latitudes(self, num_latitudes: int) -> np.array:
        if isinstance(self.latitude_pdf, CountType):
            return self.latitude_pdf*np.ones(num_latitudes)
        else:
            return self.latitude_pdf.rvs(size=num_latitudes)

    def get_longitudes(self, num_longitudes: int) -> np.array:
        if isinstance(self.longitude_pdf, CountType):
            return self.longitude_pdf * np.ones(num_longitudes)
        else:
            return self.longitude_pdf.rvs(size=num_longitudes)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class ActiveRegion:
    """Base active region class."""
    def __init__(self,
                 flare_activity: FlareActivity=None,
                 num_flares: int=None,
                 occurrence_freq_pdf: PDFType=None,
                 region: Region=None
                 ):
        """
        Generates an active region, which produces flares with a variety of tunable statistical properties.

        The distributions are currently expected to be a single value, a tuple/list of a [min, max], or a
        scipy.stats frozen distribution.  This may be expanded in the future.
        If a single value is provided, ActiveRegion will use that value for all flares.
        If a [min, max] tuple/list is provided, ActiveRegion will use a uniform distribution over that range.
        If a frozen pdf is provided, ActiveRegion will pull values from the pdf via pdf.rvs(num_flares)

        !If num_flares is given, occurrence_freq_pdf is ignored!

        Currently, distributions are not linked (so, e.g., intensity and decay constant are not correlated)
        That should probably be built into a separate class, eventually...

        :param flare_type:
        :param num_flares: Overrides occurrence_freq_pdf if provided
        :param intensity_pdf:
        :param decay_pdf:
        :param onset_pdf:
        :param occurrence_freq_pdf:
        :param longitude_pdf:
        :param latitude_pdf:
        """

        if num_flares is None:
            if occurrence_freq_pdf is None:
                raise ValueError("Must provide num_flares or occurrence_freq_pdf to initialize an ActiveRegion")
            else:
                self._num_flares = None
                self._occurrence_freq_pdf = occurrence_freq_pdf
        else:
            if isinstance(num_flares, int):
                self._num_flares = num_flares
                self._occurrence_freq_pdf = None
            else:
                raise ValueError("num_flares must be an integer or None")



# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from exoechopy.visualize import scatter_plot_3d

    test_region_1 = Region([0, 2*np.pi/3], [np.pi/12, 3*np.pi/4])

    num_flares = 1000
    phi_points = test_region_1.get_latitudes(num_flares)
    theta_points = test_region_1.get_longitudes(num_flares)

    points = vect_from_spherical_coords(theta_points, phi_points)
    MyPointCloud = PointCloud(points, point_color="k", display_marker='.', name="Flare locations")

    scatter_plot_3d(MyPointCloud, savefile='hold')

    plt.legend()
    plt.show()
