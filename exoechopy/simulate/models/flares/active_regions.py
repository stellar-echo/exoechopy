
"""
This module provides active region classes and methods for stars.
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from astropy import units as u
from astropy.utils.exceptions import AstropyUserWarning

from .flares import *
from ....utils import *


__all__ = ['FlareCollection', 'FlareActivity', 'Region', 'ActiveRegion']


# TODO Make it possible to use joint probability distributions and include conditional requirements (like decay > onset)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class FlareCollection(dict):
    """Data container for a collection of subcollections of flares."""

    template_dictionary = {'all_flares': [],
                           'flare_times': np.array([]),
                           'flare_intensities': np.array([]),
                           'flare_vector_array': np.array([])}  # shape (n, 3)

    def __init__(self, init_dict: dict=None):
        """Flare collection is a series of dictionaries stored in a dictionary.

        Flare dictionaries are keyed with an integer, other global properties are stored as strings

        Parameters
        ----------
        init_dict
            Optional initialization dictionary

        Notes
        ----------
        The FlareCollection is intended to be flexible, enabling custom flare parameters or properties to be given.

        Common dict add-ons:
        - 'activity_indices': an array of which flare types occur in which order

        Common sub-dictionary add-ons:
        - 'flare_times': an array of times that flare occur
        - 'flare_vector_array': an array of vectors for the positions of the flares

        Methods
        ----------
        append
        assign_property
        assign_collection_properties
        __call__

        Attributes
        ----------
        current_dict_key
        num_flares
        all_flares
        all_flare_times
        all_flare_intensities
        all_flare_vectors

        Examples
        ----------


        """
        if init_dict is None:
            super().__init__({0: self.template_dictionary.copy()})
        else:
            super().__init__(init_dict.copy())

        self._current_dict_key = 0

    # ------------------------------------------------------------------------------------------------------------ #
    def append(self, another_flare_collection: 'FlareCollection'):
        """Appends another FlareCollection and avoids duplicating the same integers as currently in use.

        Parameters
        ----------
        another_flare_collection
            Adds an additional FlareCollection to this collection.

        """
        if not isinstance(another_flare_collection, (dict, FlareCollection)):
            raise TypeError("FlareCollection can only be appended to another FlareCollection")
        # Avoid dupes:
        if len(self.keys()) > 0:
            start_val = max(self.sub_dict_keys)+1
        else:
            start_val = 0
        for k in another_flare_collection.keys():
            if isinstance(k, int):
                self[start_val+k] = another_flare_collection[k]
            else:
                if k not in self.keys():
                    self[k] = another_flare_collection[k]
        self.current_dict_key = max(self.sub_dict_keys)

    # ------------------------------------------------------------------------------------------------------------ #
    def assign_property(self, property_key: str, property_val):
        """Enables customization of each subcollection by defining special property keys and values

        Parameters
        ----------
        property_key
            Key to be added to the current subdictionary
        property_val
            Property to be associated with the new key
        """
        self[self.current_dict_key][property_key] = property_val

    def assign_properties(self, property_dict: dict):
        """Runs assign_property to several keys and values on the current subdictionary

        Currently overwrites existing properties if they have the same key

        Parameters
        ----------
        property_dict
            Dictionary with new keys and values to be added

        """
        for k, v in property_dict.items():
            self.assign_property(k, v)

    def assign_collection_properties(self, property_key: str, property_val):
        """Add collection-level dict entry (above subdictionaries)

        Parameters
        ----------
        property_key
            Key to be added to the highest level dictionary
        property_val
            Value to be associated with the key
        """
        if not isinstance(property_key, int):
            self[property_key] = property_val

    # ------------------------------------------------------------------------------------------------------------ #
    def __call__(self) -> dict:
        """Returns a copy of the collection of flares

        Returns
        -------
        dict
            A copy of the entire dictionary

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
    def sub_dict_keys(self):
        valid_keys = [ki for ki in self.keys() if isinstance(ki, int)]
        return valid_keys

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def num_flares(self):
        valid_keys = self.sub_dict_keys
        return np.sum([len(sub_dict['all_flares']) for sub_dict in [self[ki] for ki in valid_keys]])

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def all_flares(self):
        valid_keys = self.sub_dict_keys
        return np.concatenate([sub_dict['all_flares'] for sub_dict in [self[ki] for ki in valid_keys]])

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def all_flare_times(self):
        valid_keys = self.sub_dict_keys
        # See http://docs.astropy.org/en/latest/known_issues.html#quantities-lose-their-units-with-some-operations
        flare_time_list = [sub_dict['flare_times'] for sub_dict in [self[ki] for ki in valid_keys]]
        # Flatten list, since u.Quantity cannot handle unequal lists:
        flare_time_list = [item for sublist in flare_time_list for item in sublist]
        return u.Quantity(flare_time_list).flatten()

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def all_flare_intensities(self):
        valid_keys = self.sub_dict_keys
        return np.concatenate([sub_dict['flare_intensities'] for sub_dict in [self[ki] for ki in valid_keys]])

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def all_flare_vectors(self):
        valid_keys = self.sub_dict_keys
        return np.concatenate([sub_dict['flare_vector_array'] for sub_dict in [self[ki] for ki in valid_keys]])

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def all_flare_angles(self):
        valid_keys = self.sub_dict_keys
        # See http://docs.astropy.org/en/latest/known_issues.html#quantities-lose-their-units-with-some-operations
        flare_angle_list = [sub_dict['flare_angle_array'] for sub_dict in [self[ki] for ki in valid_keys]]
        # Flatten list, since u.Quantity cannot handle unequal lists:
        flare_angle_list = [item for sublist in flare_angle_list for item in sublist]
        return u.Quantity(flare_angle_list).flatten()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class FlareActivity:
    """Base flare activity class, good for subclassing."""
    def __init__(self,
                 flare_type: type(ProtoFlare)=None,
                 intensity_pdf: PDFType=None,
                 name: str=None,
                 **kwargs):
        """Provides a flare generator for standalone or ActiveRegion applications.

        Creates ensembles for a single type of flare.
        Can support distributions of intensities and passes arguments to flares.

        Parameters
        ----------
        flare_type
            The types of flares generated by this FlareActivity
        intensity_pdf
        name
            Can be used to identify when multiple FlareActivity instances are in use
        kwargs
            Generators for the arguments that are associated with flare_type.
            Best practice: append '_pdf' to the flare instance keyword (e.g., onset_pdf=...)
        """

        if issubclass(flare_type, ProtoFlare):
            self._flare_type = flare_type
        else:
            raise TypeError("flare_type must be a child class of ProtoFlare")

        self._intensity_pdf = None
        self.intensity_pdf = intensity_pdf

        self._flare_kwarg_vals = None  # Holds the distribution functions
        self._flare_kwarg_types = None  # Holds the type of distribution, typically a float or RVFrozen
        self._flare_new_kwargs = None  # Holds the prefix from the kwarg, to be used in the future
        self._kw_units = None
        self.flare_kwargs = kwargs

        # Activity name
        self._name = name

    # ------------------------------------------------------------------------------------------------------------ #
    def generate_n_flares(self, n_flares: int) -> FlareCollection:
        """Generates n flares based on the underlying flare_type

        Parameters
        ----------
        n_flares
            Number of flares to include in the generation

        Returns
        -------
        FlareCollection
            The FlareCollection of the requested number of flares

        """

        flare_collection = FlareCollection()
        if isinstance(self.intensity_pdf, CountType):
            intensity_list = np.ones(n_flares)*self.intensity_pdf
        elif isinstance(self.intensity_pdf, RVFrozen):
            intensity_list = self.intensity_pdf.rvs(size=n_flares)
        else:
            raise NotImplementedError("self.intensity_pdf does not have a generic handling solution yet")
        flare_collection.assign_property(property_key='flare_intensities', property_val=intensity_list)

        if len(self.flare_kwargs) > 0:
            arg_array = np.zeros(n_flares, dtype=self._flare_kwarg_types)
            arg_dataframe = pd.DataFrame(arg_array)
            for flare_arg, arg_label in zip(self.flare_kwargs, arg_dataframe.columns):
                if isinstance(flare_arg, CountType):
                    arg_dataframe[arg_label] = np.ones(n_flares) * flare_arg
                elif isinstance(flare_arg, RVFrozen):
                    arg_dataframe[arg_label] = flare_arg.rvs(size=n_flares)
                else:
                    arg_dataframe[arg_label] = flare_arg

            # Repack the original units with the computed values using the fancy dict keywords:
            flare_list = [self._flare_type(**dict(zip(self._flare_new_kwargs,
                                                      [ai*ui if ui!='str' else ai
                                                       for ai, ui in zip(arg_i, self._kw_units)])))
                          for arg_i in arg_dataframe.values]
        else:
            flare_list = [self._flare_type() for ni in range(n_flares)]

        flare_collection.assign_property(property_key='all_flares', property_val=flare_list)
        return flare_collection

    # ------------------------------------------------------------------------------------------------------------ #
    def _interpret_kwarg_types(self, _kwargs):
        """Interprets custom kwargs to pass flexible options to underlying flares

        Parameters
        ----------
        _kwargs

        Returns
        -------

        """
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
                    else:
                        new_args.append(val.value)
                    kw_units.append(val.unit)
                else:
                    new_args.append(val)
                    kw_units.append(1.)
                arg_type_list.append((str(kw), 'float64'))
            elif isinstance(val, (list, tuple)):
                if isinstance(val[1], u.IrreducibleUnit):
                    new_args.append(val[0])
                    kw_units.append(val[1])
                else:
                    new_args.append(stats.uniform(loc=val[0], scale=val[1]-val[0]))
                    kw_units.append(1.)
                arg_type_list.append((str(kw), 'float64'))
            elif isinstance(val, RVFrozen):
                arg_type_list.append((str(kw), 'float64'))
                new_args.append(val)
                kw_units.append(1.)
            else:
                if isinstance(val, str):
                    string_length = 16
                    # For use in a structured numpy array:
                    arg_type_list.append((str(kw), 'U'+str(string_length)))
                    if len(val) > string_length:
                        AstropyUserWarning("Structured numpy array only configured to accept U"+str(string_length)+" strings, received "
                                           + val +", converting to "+val[:string_length])
                        val = val[:string_length]
                    new_args.append(val)
                    kw_units.append("str")
                else:
                    arg_type_list.append(type(val))
                    new_args.append(val)
                    kw_units.append(1)

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
        """Provides an object that helps determine where activity occurs in an ActiveRegion.

        Parameters
        ----------
        longitude_pdf
            If None, defaults to 0-2pi uniform distribution.
        latitude_pdf
            If None or a [min, max], defaults to distribution that is uniform *on a sphere*
            (see, e.g., http://mathworld.wolfram.com/SpherePointPicking.html)

        Notes
        ----------
        If a custom distribution is provided, be aware of the spherical area element!

        Methods
        ----------
        gen_latitudes
        gen_longitudes
        gen_xyz_vectors

        Attributes
        -----------
        latitude_pdf
        longitude_pdf

        """

        # TODO add placeholder for background intensity variations, probably spectral objects (maybe as Child)

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
    @property
    def center_of_region(self):
        if isinstance(self.latitude_pdf, CountType):
            mean_lat = self.latitude_pdf
        else:
            mean_lat = self.latitude_pdf.expect()
        if isinstance(self.longitude_pdf, CountType):
            mean_long = self.longitude_pdf
        else:
            mean_long = self.longitude_pdf.expect()
        return mean_long*u.rad, mean_lat*u.rad

    # ------------------------------------------------------------------------------------------------------------ #
    def gen_latitudes(self, num_latitudes: int) -> np.ndarray:
        """Generates an array of latitude values based on the latitude_pdf

        Parameters
        ----------
        num_latitudes
            Number of latitudes to generate

        Returns
        -------
        np.ndarray
            Array of latitudes drawn from the latitude_pdf generator
        """
        if isinstance(self.latitude_pdf, CountType):
            return self.latitude_pdf*np.ones(num_latitudes)
        else:
            return self.latitude_pdf.rvs(size=num_latitudes)

    def gen_longitudes(self, num_longitudes: int) -> np.array:
        """Generates an array of longitudes values based on the longitude_pdf

        Parameters
        ----------
        num_longitudes
            Number of longitudes to generate

        Returns
        -------
        np.ndarray
            Array of longitudes drawn from the longitude_pdf generator
        """
        if isinstance(self.longitude_pdf, CountType):
            return self.longitude_pdf * np.ones(num_longitudes)
        else:
            return self.longitude_pdf.rvs(size=num_longitudes)

    def gen_xyz_vectors(self,
                        n_flares: int,
                        return_angles: bool=False) -> (np.ndarray, np.ndarray, np.ndarray):
        """Generate an array of x, y, z points

        Draws longitude and latitude points from the defined _pdf's, then converts to cartesian coordinates.

        Parameters
        ----------
        n_flares
            Number of points to sample.
        return_angles
            Whether or not to explicitly return the angles, or just the vectors

        Returns
        -------
        np.ndarray
            Array of points in xyz
            or
            array of longitude angles, latitude angles, and points in xyz
        """
        longitude_points = self.gen_longitudes(n_flares)
        latitude_points = self.gen_latitudes(n_flares)
        vectors = vect_from_spherical_coords(longitude_points, latitude_points)
        if return_angles:
            return longitude_points, latitude_points, vectors
        else:
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
    def __call__(self, duration: CountType=None, max_flares=10000) -> FlareCollection:
        """Activate the ActiveRegion, causing it to produce flares

        Generates flares based on the underlying Region and FlareActivity

        Parameters
        ----------
        duration
            A timescale of simulation.  Can be ignored if num_flares is defined in the class instance.
        max_flares
            Backup value to prevent infinite call for the duration call

        Returns
        -------
        FlareCollection
            All of the flares generated, as well as their attributes
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
            self._generate_flares_at_times_lw(flare_times)
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
            if ii == 0:
                all_flares = FlareCollection(activity.generate_n_flares(n_counts))
            else:
                all_flares.append(activity.generate_n_flares(n_counts))
        all_flares.assign_collection_properties('activity_selections', activity_selections)
        self._all_flares = all_flares

    # ------------------------------------------------------------------------------------------------------------ #
    def _generate_flares_at_times_lw(self, list_of_times: list):
        n_flares = len(list_of_times)
        self._generate_n_flares(n_flares)
        list_of_times = np.array(list_of_times)
        activity_selections = self._all_flares['activity_selections']
        for ai, activity in enumerate(self._flare_activity_list):
            self._all_flares.current_dict_key = ai
            self._all_flares.assign_property('flare_times',
                                             list_of_times[np.where(activity_selections == ai)]*lw_time_unit)

    # ------------------------------------------------------------------------------------------------------------ #
    def _generate_flare_locations(self):
        for (k, sub_dict) in self._all_flares.items():
            if isinstance(k, int):
                longitudes, latitudes, vectors = self._region.gen_xyz_vectors(len(sub_dict['all_flares']),
                                                                              return_angles=True)
                sub_dict['flare_vector_array'] = vectors
                sub_dict['flare_latitude_array'] = latitudes*u.rad
                sub_dict['flare_longitude_array'] = longitudes*u.rad

    # ------------------------------------------------------------------------------------------------------------ #
    def generate_background(self, time_domain: u.Quantity,
                            observation_vector: np.ndarray) -> (np.ndarray, np.ndarray):
        """Prepare a background intensity level that changes with time and orientation

        Active regions can have brightening or darkening (starspots) effects.
        These effects depend on the star opacity and viewing orientation.
        ActiveRegion is not aware of star opacity effects, so it just provides a relative angle for external use.
        This function takes a time_domain and an observation vector (or array of vectors of len(duration))
        and provides the background intensity variation.

        Parameters
        ----------
        time_domain
            Times to evaluate the background at
        observation_vector
            A single vector pointing in the direction of observation
            or an array of vectors of equal length to time_domain

        Returns
        -------
        (np.ndarray, np.ndarray)
            Array of intensities and array of angles between observation_vector and the star surface normal
        """
        raise NotImplementedError("No means of tracking regional variations implemented yet")

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def center_of_region(self):
        return self._region.center_of_region



