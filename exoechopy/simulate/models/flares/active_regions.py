
"""
This module provides active region classes and methods for stars.
"""

import warnings
import numpy as np
from scipy import stats
from exoechopy.simulate.models.flares import *
from exoechopy.utils import *
from astropy import units as u
from astropy.utils.exceptions import AstropyUserWarning

__all__ = ['FlareCollection', 'FlareActivity', 'Region', 'ActiveRegion']


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class FlareCollection:
    """Data container for a collection & subcollections of flares."""

    template_dictionary = {'flare_times': np.array([]),
                           'flare_intensities': np.array([]),
                           'flare_vector_array': np.empty((1,3))}

    def __init__(self):
        """Flare collection is a series of dictionaries stored in a dictionary."""

        #  Flare collection is a dictionary of dictionaries
        self._flare_collection = {}
        self._current_dict = {}

        # TODO Figure out a quick way to assemble metrics on:
        #           n_flares, all_flare_times, all_flare_intensities, all_flare_vectors

        # #  Keep track of a few important variables about the collection for easy outside access:
        # self._collection_totals = {'n_flares': 0,
        #                            'all_flare_times': np.array([]),
        #                            'all_flare_intensities': np.array([]),
        #                            'all_flare_vectors': np.empty((1,3))
        #                            }
        #
        # self._key_equiv_dict = {'flare_times': 'all_flare_times',
        #                         'flare_intensities': 'all_flare_intensities',
        #                         'flare_vector_array': 'all_flare_vectors'}

        # Initialize:
        self.add_subcollection()

        # Common dict add-ons:
        # 'flare_decay_const_list': [],
        # 'flare_onset_duration_list': []
        # 'activity_indices': []

    # ------------------------------------------------------------------------------------------------------------ #
    def __getitem__(self, item):
        return self._flare_collection[item]

    # ------------------------------------------------------------------------------------------------------------ #
    def append(self, another_flare_collection):
        """

        :param FlareCollection another_flare_collection:
        """
        if not isinstance(another_flare_collection, FlareCollection):
            raise TypeError("FlareCollection can only be appended to another FlareCollection")
        # Avoid dupes:
        start_val = max(list(self._flare_collection))+1
        for k in list(another_flare_collection):
            self._current_dict[start_val+k] = another_flare_collection[k]
        # self._N_flares += another_flare_collection.num_flares
        # self._all_flare_times = np.append(self.all_flare_times,
        #                                   another_flare_collection.all_flare_times)
        # self._all_flare_intensities = np.append(self.all_flare_intensities,
        #                                         another_flare_collection.all_flare_intensities)
        # self._all_flare_vectors = np.append(self.all_flare_vectors,
        #                                     another_flare_collection.all_flare_vectors)

    # ------------------------------------------------------------------------------------------------------------ #
    def add_subcollection(self, **kwargs):
        new_key = len(self._flare_collection)
        # Initialize a new subcollection:
        if len(kwargs) > 0:
            self._current_dict = kwargs
        else:
            self._current_dict = self.template_dictionary.copy()

        self._flare_collection[new_key] = self._current_dict

    # ------------------------------------------------------------------------------------------------------------ #
    def assign_property(self, property_key:str, property_val):
        """
        Enables customization of each subcollection by defining special property keys and values
        :param property_key:
        :param property_val:
        :return:
        """
        self._current_dict[property_key] = property_val
        # if property_key in self._key_equiv_dict:
        #     raise NotImplementedError
        #     # Problem: the key totals can now be out of sync.  Would be better to find a way to assemble
        #     self._collection_totals[self._key_equiv_dict[property_key]] = np.append(
        #         self._collection_totals[self._key_equiv_dict[property_key]],  property_val)

    def assign_properties(self, property_dict: dict):
        if property_dict.keys().isdisjoint(self._current_dict):
            self._current_dict.update(property_dict)

    # ------------------------------------------------------------------------------------------------------------ #
    def __call__(self):
        """
        :return dict: full collection of flares
        """
        return self._flare_collection

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
        return self._all_flare_times

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def all_flare_intensities(self):
        return self._all_flare_intensities

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def all_flare_vectors(self):
        return self._all_flare_vectors


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class FlareActivity:
    """Base flare activity class, good for subclassing."""
    def __init__(self,
                 flare_type: ProtoFlare=None,
                 intensity_pdf: PDFType=None,
                 name:str=None,
                 *args):
        """
        Provides a flare generator for an ActiveRegion.
        Produces a single type of flare.

        :param flare_type: The types of flares generated by this FlareActivity
        :param intensity_pdf:
        :param name: Can be used to identify when multiple FlareActivity instances are in use
        :param args: Generators for the arguments that are associated with flare_type
        """
        if isinstance(flare_type, ProtoFlare):
            self._flare_type = flare_type
        else:
            raise TypeError("flare_type must be a child class of ProtoFlare")

        if isinstance(intensity_pdf, PDFType):
            if isinstance(intensity_pdf, IntFloat):
                self._intensities = intensity_pdf
        else:
            raise TypeError("intensity_pdf must be a float, list, or frozen stats pdf")

        self._args = None
        self._arg_types = None
        self.args = args
        self._name = name

    # ------------------------------------------------------------------------------------------------------------ #
    def generate_n_flares(self, n_flares:int) -> FlareCollection:
        flare_collection = FlareCollection()
        for arg, arg_type in zip(self.args, self._arg_types):
            flare_collection.assign_property(property_key=, property_val=)
        return flare_collection

    # ------------------------------------------------------------------------------------------------------------ #
    def _interpret_arg_types(self, args):
        arg_type_list = []
        new_args = []
        for arg in args:
            if isinstance(arg, CountType):
                arg_type_list.append(CountType)
                new_args.append(arg)
            elif isinstance(arg, (list, tuple)):
                new_args.append(stats.uniform(loc=arg[0], scale=arg[1]-arg[0]))
                arg_type_list.append(stats._distn_infrastructure.rv_frozen)
            elif isinstance(arg, stats._distn_infrastructure.rv_frozen):
                arg_type_list.append(stats._distn_infrastructure.rv_frozen)
                new_args.append(arg)
            else:
                arg_type_list.append(type(arg))
                new_args.append(arg)
        return new_args, arg_type_list

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, args):
        self._args, self._arg_types = self._interpret_arg_types(args)
    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def name(self):
        return self._name


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
        return self._latitude_pdf

    @latitude_pdf.setter
    def latitude_pdf(self, latitude_pdf):
        if latitude_pdf is None:
            self._latitude_pdf = stats.uniform(loc=0, scale=1)
        elif isinstance(latitude_pdf, PDFType):
            if isinstance(latitude_pdf, CountType):
                self._latitude_pdf = latitude_pdf
            elif isinstance(latitude_pdf, (list, tuple)):
                # Create, then freeze the distribution:
                self._latitude_pdf = SphericalLatitudeGen(a=latitude_pdf[0], b=latitude_pdf[1], name='lat_gen')
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
                 region: Region=None,
                 flare_activity_ratios: list = None,

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

        :param flare_activity:
        A FlareActivity instance, or list of instances, that determines how the flares in the region behave

        :param num_flares: Overrides occurrence_freq_pdf if provided

        :param occurrence_freq_pdf: Distirbution function governing how frequently a flare occurs

        :param region: Where the flares occur on the star.  Defaults to uniform coverage.

        :param flare_activity_ratios:
        If multiple flare_activity instances are provided, can provide a relative distribution of weights,
        ala np.random.choice p=...
        If relative weights are not provided, all are weighted equally.
        """

        # flare_activity can be a single instance or several different ones provided as a list
        if flare_activity is None or not isinstance(flare_activity, FlareActivity):
            if isinstance(flare_activity, (tuple, list)):
                for fa in flare_activity:
                    if not isinstance(fa, FlareActivity):
                        raise TypeError("All flare_activity items must be FlareActivity instances")
                self._flare_activity = flare_activity
            else:
                raise TypeError("flare_activity must be a FlareActivity instance")
        else:
            self._flare_activity = [flare_activity]

        if num_flares is None:
            if occurrence_freq_pdf is None:
                raise ValueError("Must provide num_flares or occurrence_freq_pdf to initialize an ActiveRegion")
            else:
                self._num_flares = None
                if isinstance(occurrence_freq_pdf, CountType):
                    if isinstance(occurrence_freq_pdf, u.Quantity):
                        mean_time_between_events = occurrence_freq_pdf.to(lw_freq_unit)
                    else:
                        mean_time_between_events = u.Quantity(occurrence_freq_pdf, lw_freq_unit)
                        warnings.warn("Casting occurrence_freq_pdf, input as "
                                      +str(occurrence_freq_pdf)+", to "+str(lw_freq_unit),
                                      AstropyUserWarning)
                    # Default: uniform distribution
                    self._occurrence_freq_pdf = stats.uniform(scale=(2 / mean_time_between_events).to(u.s).value)
                elif isinstance(occurrence_freq_pdf, stats._distn_infrastructure.rv_frozen):
                    self._occurrence_freq_pdf = occurrence_freq_pdf
                else:
                    raise TypeError("Unknown occurrence_freq_pdf type")
        else:
            if isinstance(num_flares, int):
                self._num_flares = num_flares
                self._occurrence_freq_pdf = None
            else:
                raise ValueError("num_flares must be an integer or None")

        if region is None:
            self._region = Region()
        elif isinstance(region, Region):
            self._region = region
        else:
            raise TypeError("region must be a Region instance or None")

        if flare_activity_ratios is None:
            num_activities = len(self._flare_activity)
            self._flare_activity_ratios = [1./num_activities for x in range(num_activities)]
        elif isinstance(flare_activity_ratios, list):
            self._flare_activity_ratios = flare_activity_ratios
        else:
            raise TypeError("Unknown flare_activity_ratios type")


    # ------------------------------------------------------------------------------------------------------------ #
    def __call__(self, duration: CountType=None, max_flares=10000):
        """

        :param duration:
        :param max_flares: Backup value to prevent infinite call
        :return:
        """
        if duration is None and self._num_flares is None:
            raise ValueError("ActiveRegion called with unknown timescale")
        # Used if you just want the flares, don't care about time distribution:
        elif duration is None and self._num_flares:
            return self._generate_n_flares(self._num_flares)
        else:
            if isinstance(duration, u.Quantity):
                duration = duration.to(lw_time_unit).value
            else:
                warnings.warn("Casting duration, input as "
                              + str(duration) + ", to " + str(lw_time_unit),
                              AstropyUserWarning)
            flare_times = stochastic_flare_process(duration, self._occurrence_freq_pdf, max_iter=max_flares)
            return self._generate_flares_at_times(flare_times)


    # ------------------------------------------------------------------------------------------------------------ #
    def _generate_n_flares(self, n_flares:int) -> FlareCollection:
        all_flares = FlareCollection()
        activity_selections = np.random.choice(len(self._flare_activity), size=n_flares, p=self._flare_activity_ratios)
        flare_counts = np.bincount(activity_selections, minlength=len(self._flare_activity))
        for activity, n_counts in zip(self._flare_activity, flare_counts):
            all_flares.append(activity.generate_n_flares(n_counts))
        all_flares.assign_property('activity_selections', activity_selections)
        return all_flares

    # ------------------------------------------------------------------------------------------------------------ #
    def _generate_flares_at_times(self, times:list) -> FlareCollection:
        all_flares = FlareCollection()
        n_flares = len(times)
        activity_selections = np.random.choice(len(self._flare_activity), size=n_flares, p=self._flare_activity_ratios)
        flare_counts = np.bincount(activity_selections, minlength=len(self._flare_activity))
        for activity, n_counts in zip(self._flare_activity, flare_counts):
            _activity = activity.generate_n_flares(n_counts)
            _activity.all_flare_times = times
            all_flares.append(_activity)
        all_flares.assign_property('activity_selections', activity_selections)
        return all_flares


# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from exoechopy.visualize.standard_3d_plots import *

    print("""
    Regions are a Class that supports determining where on a star a flare occurs.
    They can be deterministic, such as a single point, or probability distributions.
    """)

    min_long = np.pi/4
    max_long = 2*np.pi-np.pi/4
    min_lat = np.pi / 4
    max_lat = min_lat + np.pi / 6
    test_region_1 = Region([min_long, max_long], [min_lat, max_lat])

    number_of_flares = 1000
    theta_points = test_region_1.get_longitudes(number_of_flares)
    phi_points = test_region_1.get_latitudes(number_of_flares)

    points = vect_from_spherical_coords(theta_points, phi_points)
    MyPointCloud = PointCloud(points, point_color="k", display_marker='.', point_size=3, linewidth=0,
                              name="Flare locations")

    ax_dic = scatter_plot_3d(MyPointCloud, savefile='hold')

    plot_sphere(ax_dic['ax'], rad=.99, sphere_color='y')

    set_3d_axes_equal(ax_dic['ax'])

    plt.legend()
    plt.show()

    #  =============================================================  #
