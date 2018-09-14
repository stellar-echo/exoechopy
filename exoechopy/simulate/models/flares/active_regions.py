
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


class FlareCollection(dict):
    """Data container for a collection & subcollections of flares."""

    template_dictionary = {'all_flares': [],
                           'flare_times': np.array([]),
                           'flare_intensities': np.array([]),
                           'flare_vector_array': np.array([])}  # shape (n, 3)

    def __init__(self, init_dict=None):
        """Flare collection is a series of dictionaries stored in a dictionary.
        Flare dictionaries are keyed with an integer, other global properties are stored as strings
        """
        if init_dict is None:
            super().__init__({0: self.template_dictionary.copy()})
        else:
            super().__init__(init_dict.copy())

        self._current_dict_key = 0

        # Common dict add-ons:
        # 'flare_decay_const_list': [],
        # 'flare_onset_duration_list': []
        # 'activity_indices': []

    # ------------------------------------------------------------------------------------------------------------ #
    def append(self, another_flare_collection):
        """

        :param FlareCollection another_flare_collection:
        """
        if not isinstance(another_flare_collection, (dict, FlareCollection)):
            raise TypeError("FlareCollection can only be appended to another FlareCollection")
        # Avoid dupes:
        start_val = max(list(self.keys()))+1
        for k in another_flare_collection.keys():
            self[start_val+k] = another_flare_collection[k]
        self.current_dict_key = start_val + k

    # ------------------------------------------------------------------------------------------------------------ #
    def assign_property(self, property_key: str, property_val):
        """
        Enables customization of each subcollection by defining special property keys and values
        :param property_key:
        :param property_val:
        :return:
        """
        self[self.current_dict_key][property_key] = property_val

    def assign_properties(self, property_dict: dict):
        # Currently overwrites existing properties if they have the same key
        # if property_dict.keys().isdisjoint(self._current_dict.keys()):
        for k, v in property_dict.items():
            self.assign_property(k, v)

    def assign_collection_properties(self, property_key: str, property_val):
        """
        Provide collection-level dict entry
        :param property_key:
        :param property_val:
        :return:
        """
        if not isinstance(property_key, int):
            self[property_key] = property_val

    # ------------------------------------------------------------------------------------------------------------ #
    def __call__(self):
        """
        :return dict: full collection of flares
        """
        return self.copy()

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def current_dict_key(self):
        return self._current_dict_key

    @current_dict_key.setter
    def current_dict_key(self, new_dict_key):
        self._current_dict_key = new_dict_key

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def num_flares(self):
        valid_keys = [ki for ki in self.keys() if isinstance(ki, int)]
        return np.sum([len(sub_dict['all_flares']) for sub_dict in [self[ki] for ki in valid_keys]])

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def all_flares(self):
        valid_keys = [ki for ki in self.keys() if isinstance(ki, int)]
        return np.concatenate([sub_dict['all_flares'] for sub_dict in [self[ki] for ki in valid_keys]])

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def all_flare_times(self):
        valid_keys = [ki for ki in self.keys() if isinstance(ki, int)]
        return np.concatenate([sub_dict['flare_times'] for sub_dict in [self[ki] for ki in valid_keys]])

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def all_flare_intensities(self):
        valid_keys = [ki for ki in self.keys() if isinstance(ki, int)]
        return np.concatenate([sub_dict['flare_intensities'] for sub_dict in [self[ki] for ki in valid_keys]])

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def all_flare_vectors(self):
        valid_keys = [ki for ki in self.keys() if isinstance(ki, int)]
        return np.concatenate([sub_dict['flare_vector_array'] for sub_dict in [self[ki] for ki in valid_keys]])


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class FlareActivity:
    """Base flare activity class, good for subclassing."""
    def __init__(self,
                 flare_type: type(ProtoFlare)=None,
                 intensity_pdf: PDFType=None,
                 name: str=None,
                 **kwargs):
        """
        Provides a flare generator for an ActiveRegion.
        Produces a single type of flare.

        :param flare_type: The types of flares generated by this FlareActivity
        :param intensity_pdf:
        :param name: Can be used to identify when multiple FlareActivity instances are in use
        :param kwargs: Generators for the arguments that are associated with flare_type.
        Best practice: append '_pdf' to the flare instance keyword (e.g., onset_pdf=...)
        """
        if issubclass(flare_type, ProtoFlare):
            self._flare_type = flare_type
        else:
            raise TypeError("flare_type must be a child class of ProtoFlare")

        self._intensity_pdf = None
        self.intensity_pdf = intensity_pdf

        print("flare_kwargs: ", kwargs)
        self._flare_kwarg_vals = None  # Holds the distribution functions
        self._flare_kwarg_types = None  # Holds the type of distribution, typically a float or RVFrozen
        self._flare_new_kwargs = None  # Holds the prefix from the kwarg, to be used in the future
        self._kw_units = None
        self.flare_kwargs = kwargs
        print(self._flare_new_kwargs)

        self._name = name

    # ------------------------------------------------------------------------------------------------------------ #
    def generate_n_flares(self, n_flares: int) -> FlareCollection:
        flare_collection = FlareCollection()
        if isinstance(self.intensity_pdf, CountType):
            intensity_list = np.ones(n_flares)*self.intensity_pdf
        elif isinstance(self.intensity_pdf, RVFrozen):
            intensity_list = self.intensity_pdf.rvs(size=n_flares)
        else:
            raise NotImplementedError("self.intensity_pdf does not have a generic handling solution yet")
        flare_collection.assign_property(property_key='flare_intensities', property_val=intensity_list)

        if len(self.flare_kwargs) > 0:
            arg_array = np.zeros((n_flares, len(self.flare_kwargs)))
            for ai, (flare_arg, arg_unit) in enumerate(zip(self.flare_kwargs, self._kw_units)):
                if isinstance(flare_arg, CountType):
                    arg_array[:, ai] = np.ones(n_flares) * flare_arg
                elif isinstance(flare_arg, RVFrozen):
                    arg_array[:, ai] = flare_arg.rvs(size=n_flares)

            # Repack the original units with the computed values using the fancy dict keywords:
            flare_list = [self._flare_type(**dict(zip(self._flare_new_kwargs,
                                                      [ai*ui for ai, ui in zip(arg_i, self._kw_units)])))
                          for arg_i in arg_array]
        else:
            flare_list = [self._flare_type() for ni in range(n_flares)]

        flare_collection.assign_property(property_key='all_flares', property_val=flare_list)
        return flare_collection

    # ------------------------------------------------------------------------------------------------------------ #
    def _interpret_kwarg_types(self, _kwargs):
        arg_type_list = []
        new_args = []
        new_kw = []
        kw_units = []
        # Keep units in registration with different kwarg types such as floats, ints, ranges, and distribution funcs
        for (kw, val) in _kwargs.items():
            if isinstance(val, CountType):
                if isinstance(val, u.Quantity):
                    if isinstance(val.value, np.ndarray):
                        new_args.append(stats.uniform(loc=val[0], scale=val[1] - val[0]))
                        arg_type_list.append(RVFrozen)
                    else:
                        arg_type_list.append(type(val.value))
                        new_args.append(val.value)
                    kw_units.append(val.unit)
                else:
                    arg_type_list.append(type(val))
                    new_args.append(val)
                    kw_units.append(1.)
            elif isinstance(val, (list, tuple)):
                print("val: ", val)
                if isinstance(val[1], u.IrreducibleUnit):
                    new_args.append(val[0])
                    arg_type_list.append(type(val[0]))
                    kw_units.append(val[1])
                else:
                    new_args.append(stats.uniform(loc=val[0], scale=val[1]-val[0]))
                    arg_type_list.append(RVFrozen)
                    kw_units.append(1.)
            elif isinstance(val, RVFrozen):
                arg_type_list.append(RVFrozen)
                new_args.append(val)
                kw_units.append(1.)
            else:
                arg_type_list.append(type(val))
                new_args.append(val)
                kw_units.append(1.)

            if kw[-4:] == '_pdf':
                new_kw.append(kw[:-4])
            else:
                new_kw.append(kw)

        return new_args, arg_type_list, new_kw, kw_units

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def flare_kwargs(self):
        return self._flare_kwarg_vals

    @flare_kwargs.setter
    def flare_kwargs(self, kwargs):
        self._flare_kwarg_vals, self._flare_kwarg_types, self._flare_new_kwargs, self._kw_units \
            = self._interpret_kwarg_types(kwargs)

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def name(self):
        return self._name

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def intensity_pdf(self):
        return self._intensity_pdf

    @intensity_pdf.setter
    def intensity_pdf(self, intensity_pdf):
        if isinstance(intensity_pdf, PDFType):
            if isinstance(intensity_pdf, CountType):
                self._intensity_pdf = intensity_pdf
            elif isinstance(intensity_pdf, (list, tuple)):
                self._intensity_pdf = stats.uniform(loc=intensity_pdf[0], scale=intensity_pdf[1]-intensity_pdf[0])
            elif isinstance(intensity_pdf, RVFrozen):
                self._intensity_pdf = intensity_pdf
            else:
                AstropyUserWarning("intensity_pdf is unanticipated type, proceed with caution")
                self._intensity_pdf = intensity_pdf
        else:
            raise TypeError("intensity_pdf must be None or PDFType")


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
            elif isinstance(longitude_pdf, RVFrozen):
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
            elif isinstance(latitude_pdf, RVFrozen):
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

    def get_vectors(self, n_flares: int) -> np.ndarray:
        longitude_points = self.get_longitudes(n_flares)
        latitude_points = self.get_latitudes(n_flares)
        vectors = vect_from_spherical_coords(longitude_points, latitude_points)
        return vectors


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class ActiveRegion:
    """Base active region class."""
    def __init__(self,
                 flare_activity: (list, FlareActivity)=None,
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
                self._flare_activity_list = flare_activity
            else:
                raise TypeError("flare_activity must be a FlareActivity instance")
        else:
            self._flare_activity_list = [flare_activity]

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
                                      + str(occurrence_freq_pdf)+", to "+str(lw_freq_unit),
                                      AstropyUserWarning)
                    # Default: uniform distribution
                    self._occurrence_freq_pdf = stats.uniform(scale=(2 / mean_time_between_events).to(u.s).value)
                elif isinstance(occurrence_freq_pdf, RVFrozen):
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
            num_activities = len(self._flare_activity_list)
            self._flare_activity_ratios = [1./num_activities for x in range(num_activities)]
        elif isinstance(flare_activity_ratios, list):
            self._flare_activity_ratios = flare_activity_ratios
        else:
            raise TypeError("Unknown flare_activity_ratios type")

        self._all_flares = None

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
            self._generate_n_flares(self._num_flares)
        else:
            if isinstance(duration, u.Quantity):
                duration = duration.to(lw_time_unit).value
            else:
                warnings.warn("Casting duration, input as "
                              + str(duration) + ", to " + str(lw_time_unit),
                              AstropyUserWarning)
            flare_times = stochastic_flare_process(duration, self._occurrence_freq_pdf, max_iter=max_flares)
            self._generate_flares_at_times(flare_times)
        self._generate_flare_locations()
        return self._all_flares

    # ------------------------------------------------------------------------------------------------------------ #
    def _generate_n_flares(self, n_flares: int):
        # all_flares = FlareCollection()
        activity_selections = np.random.choice(len(self._flare_activity_list),
                                               size=n_flares,
                                               p=self._flare_activity_ratios)
        flare_counts = np.bincount(activity_selections, minlength=len(self._flare_activity_list))
        for ii, (activity, n_counts) in enumerate(zip(self._flare_activity_list, flare_counts)):
            # print("activity: ", activity.name, "\tn_counts: ", n_counts)
            if ii == 0:
                all_flares = FlareCollection(activity.generate_n_flares(n_counts))
            else:
                all_flares.append(activity.generate_n_flares(n_counts))
        all_flares.assign_collection_properties('activity_selections', activity_selections)
        self._all_flares = all_flares

    # ------------------------------------------------------------------------------------------------------------ #
    def _generate_flares_at_times(self, list_of_times: list):
        n_flares = len(list_of_times)
        self._generate_n_flares(n_flares)
        list_of_times = np.array(list_of_times)
        activity_selections = self._all_flares['activity_selections']
        for ai, activity in enumerate(self._flare_activity_list):
            self._all_flares.current_dict_key = ai
            self._all_flares.assign_property('flare_times', list_of_times[np.where(activity_selections == ai)])

    # ------------------------------------------------------------------------------------------------------------ #
    def _generate_flare_locations(self):
        for (k, sub_dict) in self._all_flares.items():
            if isinstance(k, int):
                vectors = self._region.get_vectors(len(sub_dict['all_flares']))
                sub_dict['flare_vector_array'] = vectors


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
    MyRegion = Region([min_long, max_long], [min_lat, max_lat])

    number_of_flares = 1000

    points = MyRegion.get_vectors(number_of_flares)

    MyPointCloud = PointCloud(points, point_color="k", display_marker='.', point_size=3, linewidth=0,
                              name="Flare locations")

    ax_dic = scatter_plot_3d(MyPointCloud, savefile='hold')

    plot_sphere(ax_dic['ax'], rad=.99, sphere_color='y')

    set_3d_axes_equal(ax_dic['ax'])

    plt.suptitle("Region demo")
    plt.legend()
    plt.show()

    #  =============================================================  #

    print("""FlareActivity allows modeling multiple flares without requiring an explicit time or regional dependence.
    The FlareActivity class can be passed to the ActiveRegion with a Region to produce a full 3d model of the flares.
    It allows the use of custom Flare instances and can handle different probability generators for each argument.
    """)

    np.random.seed(101)

    num_test_flares = 6
    MyDeltaFlareActivity = FlareActivity(DeltaFlare, intensity_pdf=[1, 10], name='DeltaFlare demo')

    delta_flare_collection = MyDeltaFlareActivity.generate_n_flares(num_test_flares)
    print("Number of flares: ", delta_flare_collection.num_flares)
    print("All flare intensities: ", delta_flare_collection.all_flare_intensities)
    print("All flare times (uninitialized): ", delta_flare_collection.all_flare_times)
    print("All flare vectors (uninitialized): ", delta_flare_collection.all_flare_vectors)

    fig, ax_list = plt.subplots(1, num_test_flares, figsize=(10, 4))
    time_scale = 10*u.s
    num_plot_points = 20
    times = np.linspace(-time_scale.value / 2, time_scale.value / 2, num_plot_points)

    for flare, flare_mag, ax in zip(delta_flare_collection.all_flares,
                                    delta_flare_collection.all_flare_intensities,
                                    ax_list):
        integrated_flare = flare.evaluate_over_array_lw(times)*flare_mag
        ax.plot(times, integrated_flare,
                color='.4', lw=1, drawstyle='steps-post',
                marker='s', markersize=3, markerfacecolor='k', markeredgewidth=0)
        ax.tick_params('x', top=True, direction='in')
        ax.set_ylim(0, 1.1*max(delta_flare_collection.all_flare_intensities))
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Counts')

    fig.suptitle("DeltaFlare FlareActivity demo")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    #  =============================================================  #

    print("""To pass a special set of flare_kwargs, add them when declaring the FlareActivity:  
    For ExponentialFlare1, has keyworks onset= and decay=, so replace with onset_pdf and decay_pdf.
    FlareActivity will recognize '_pdf' as the indicator of a probability density function and will remove '_pdf' 
    from the keyword string before passing to the Flare class.  
    If '_pdf' is not included, it will pass the entire string (so 'onset' and 'onset_pdf' will behave the same), 
    but the best practice is to explicitly remind readers of your code that it's 
    a probability density function generator, not a single value, that is being passed.
    To explicitly include a unit to a frozen pdf, pass a tuple of the pdf with the irreducible unit as the second arg. 
    
    """)

    num_test_flares = 6
    MyExpFlareActivity = FlareActivity(ExponentialFlare1, intensity_pdf=stats.expon(scale=10*u.s),
                                       onset_pdf=[1, 4]*u.s,
                                       decay_pdf=(stats.rayleigh(scale=5), u.s),
                                       name='ExponentialFlare1 demo')

    exp_flare_collection = MyExpFlareActivity.generate_n_flares(num_test_flares)
    print("Number of flares: ", exp_flare_collection.num_flares)
    print("All flare intensities: ", exp_flare_collection.all_flare_intensities)
    print("All flare times (uninitialized): ", exp_flare_collection.all_flare_times)
    print("All flare vectors (uninitialized): ", exp_flare_collection.all_flare_vectors)

    fig, ax_list = plt.subplots(1, num_test_flares, figsize=(10, 4))
    time_scale = 30*u.s
    num_plot_points = 40
    times = np.linspace(-time_scale.value / 6, 5 * time_scale.value / 6, num_plot_points)

    for flare, flare_mag, ax in zip(exp_flare_collection.all_flares,
                                    exp_flare_collection.all_flare_intensities,
                                    ax_list):
        # Normalize to integrated flare peak for display purposes:
        integrated_flare = flare.evaluate_over_array_lw(times)*flare_mag/np.max(flare.evaluate_over_array_lw(times))
        ax.plot(times, integrated_flare,
                color='.4', lw=1, drawstyle='steps-post',
                marker='s', markersize=3, markerfacecolor='k', markeredgewidth=0)
        ax.tick_params('x', top=True, direction='in')
        ax.set_ylim(0, 1.1*max(exp_flare_collection.all_flare_intensities))
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Counts')

    fig.suptitle("ExponentialFlare1 FlareActivity demo")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    #  =============================================================  #

    print("""
    The ActiveRegion is basically a holder to align FlareActivity and Regions.
    Multiple types of FlareActivity can be present, they will be selected based on a probability ratio.
    To get the FlareCollection from the region, call the instance.
    """)

    # Without explicit times:
    num_test_flares = 6
    MyActiveRegion = ActiveRegion(flare_activity=[MyExpFlareActivity, MyDeltaFlareActivity],
                                  num_flares=num_test_flares,
                                  region=MyRegion,
                                  flare_activity_ratios=[.6, .4])
    ar_flare_collection = MyActiveRegion()
    print("Number of flares: ", ar_flare_collection.num_flares)
    print("All flare intensities: ", ar_flare_collection.all_flare_intensities)
    print("All flare times (uninitialized): ", ar_flare_collection.all_flare_times)
    print("All flare vectors: ", ar_flare_collection.all_flare_vectors)

    fig, ax_list = plt.subplots(1, num_test_flares, figsize=(12, 4))
    time_scale = 30*u.s
    num_plot_points = 40
    times = np.linspace(-time_scale.value / 6, 5 * time_scale.value / 6, num_plot_points)

    for flare, flare_mag, ax in zip(ar_flare_collection.all_flares,
                                    ar_flare_collection.all_flare_intensities,
                                    ax_list):
        # Normalize to integrated flare peak for display purposes:
        integrated_flare = flare.evaluate_over_array_lw(times)*flare_mag/np.max(flare.evaluate_over_array_lw(times))
        ax.plot(times, integrated_flare,
                color='.4', lw=1, drawstyle='steps-post',
                marker='s', markersize=3, markerfacecolor='k', markeredgewidth=0)
        ax.tick_params('x', top=True, direction='in')
        ax.set_ylim(0, 1.1*max(ar_flare_collection.all_flare_intensities))
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Counts')

    fig.suptitle("MyActiveRegion ActiveRegion demo")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    MyPointCloud = PointCloud(ar_flare_collection.all_flare_vectors, point_color="k", display_marker='.', point_size=4,
                              linewidth=0,
                              name="Flare locations")

    ax_dic = scatter_plot_3d(MyPointCloud, savefile='hold')

    plot_sphere(ax_dic['ax'], rad=.99, sphere_color='y')

    set_3d_axes_equal(ax_dic['ax'])

    plt.suptitle("MyActiveRegion flare locations")
    plt.legend()
    plt.show()

#  =============================================================  #

    print("""
    Time scales can be explicitly given, as well, which results in an unspecified number of flares that occur 
    according to occurrence_freq_pdf
    """)

    # With explicit times:
    observation_duration = 3*u.hr
    MyActiveRegion = ActiveRegion(flare_activity=[MyExpFlareActivity, MyDeltaFlareActivity],
                                  occurrence_freq_pdf=5/observation_duration,
                                  region=MyRegion,
                                  flare_activity_ratios=[.5, .5])
    ar_flare_collection = MyActiveRegion(observation_duration)
    print("Number of flares: ", ar_flare_collection.num_flares)
    print("All flare intensities: ", ar_flare_collection.all_flare_intensities)
    print("All flare times: ", ar_flare_collection.all_flare_times)
    print("All flare vectors: ", ar_flare_collection.all_flare_vectors)

    num_test_flares = ar_flare_collection.num_flares

    fig, ax_list = plt.subplots(1, num_test_flares, figsize=(12, 4))
    time_scale = 30*u.s
    num_plot_points = 40
    times = np.linspace(-time_scale.value / 6, 5 * time_scale.value / 6, num_plot_points)


    # TODO Turn this into a synthetic light curve instead of an array of plots...


    for flare, flare_mag, flare_time, ax in zip(ar_flare_collection.all_flares,
                                                ar_flare_collection.all_flare_intensities,
                                                ar_flare_collection.all_flare_times,
                                                ax_list):
        # Normalize to integrated flare peak for display purposes:
        integrated_flare = flare.evaluate_over_array_lw(times)*flare_mag/np.max(flare.evaluate_over_array_lw(times))
        ax.plot(times, integrated_flare,
                color='.4', lw=1, drawstyle='steps-post',
                marker='s', markersize=3, markerfacecolor='k', markeredgewidth=0)
        ax.tick_params('x', top=True, direction='in')
        ax.set_ylim(0, 1.1*max(ar_flare_collection.all_flare_intensities))
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Counts')
        ax.set_title("Time: " + '{0:.{1}f}'.format(flare_time, 1))

    fig.suptitle("MyActiveRegion ActiveRegion demo")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


