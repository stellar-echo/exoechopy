
"""
This module provides Star classes and methods for simulations and analysis.
"""

import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from astropy.coordinates import Distance
from astropy.utils.exceptions import AstropyUserWarning

from .flares.active_regions import *
from .spectral import *
from ..utils import *
from .planets import *
from .limbs import *

__all__ = ['DeltaStar', 'Star']

# TODO Add binary stars
# TODO Add nonspherical stars (fast rotators)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class DeltaStar(Plottable):
    """Delta-function star, has no size or shape.  Simplest star class."""
    def __init__(self,
                 mass=None,
                 spectral_type=None,
                 earth_longitude=None,
                 earth_latitude=None,
                 dist_to_earth=None,
                 **kwargs):

        self._active_region_list = []

        super().__init__(**kwargs)

        self._position = np.zeros(3)  # placeholder for potential multi-star system considerations
        self._orbiting_bodies_list = []

        #  Initialize mass, then pass to the set function:
        self._mass = None
        self.mass = mass

        if spectral_type is None:
            print("Using default spectral_type: JohnsonPhotometricBand('U'), magnitude=16")
            self._spectral_type = SpectralEmitter(JohnsonPhotometricBand('U'), magnitude=16)
        else:
            if isinstance(spectral_type, SpectralEmitter):
                self._spectral_type = spectral_type
            else:
                raise TypeError("spectral_type is not SpectralEmitter instance.")

        # Initialize angles and vector, then pass to the set/reset function:
        self._earth_longitude = None
        self._earth_latitude = None
        self._earth_direction_vector = None
        self.set_view_from_earth(earth_longitude, earth_latitude)

        if dist_to_earth is None:
            self._dist_to_earth = Distance(10, unit=u.lyr)
        else:
            if isinstance(dist_to_earth, Distance) or isinstance(dist_to_earth, u.Quantity):
                self._dist_to_earth = dist_to_earth.to(u.lyr)
            else:
                self._dist_to_earth = Distance(dist_to_earth, unit=u.lyr)
                warnings.warn("Casting dist_to_earth, input as " + str(dist_to_earth) + ", to LY",
                              AstropyUserWarning)

    # ------------------------------------------------------------------------------------------------------------ #
    def add_exoplanet(self, exoplanet: Exoplanet):
        """Add an Exoplanet object to the star

        Parameters
        ----------
        exoplanet
            Exoplanet object to be added to the star

        """

        if isinstance(exoplanet, Exoplanet):
            exoplanet.star_mass = self._mass
            self._orbiting_bodies_list.append(exoplanet)
        else:
            raise TypeError("exoplanet must be an Exoplanet class instance")

    # ------------------------------------------------------------------------------------------------------------ #
    def get_exoplanets(self) -> [Exoplanet]:
        """Return a *copy* of the list of exoplanets held by the Star

        Returns
        -------
        [*Exoplanet]
            List of exoplanet instances
        """
        return self._orbiting_bodies_list.copy()

    # ------------------------------------------------------------------------------------------------------------ #
    def add_active_regions(self, *active_regions: ActiveRegion):
        """Add active region(s) to star

        Parameters
        ----------
        active_regions
            ActiveRegion(s) to add to the star

        """
        for ar in active_regions:
            if isinstance(ar, ActiveRegion):
                self._active_region_list.append(ar)
            else:
                raise TypeError("Active regions must be ActiveRegion class instance")

    # ------------------------------------------------------------------------------------------------------------ #
    def get_position(self, *args):
        #  *args is placeholder for ability to override with time-dependent position
        return self._position

    # ------------------------------------------------------------------------------------------------------------ #
    def get_flux(self):
        return self._spectral_type.relative_flux()

    # ------------------------------------------------------------------------------------------------------------ #
    def generate_flares_over_time(self, duration):
        if not isinstance(duration, u.Quantity):
            raise ValueError("duration must be u.Quantity")

        duration = duration.to(u.s)
        all_flares = FlareCollection({})
        for active_region in self._active_region_list:
            all_flares.append(active_region(duration))
            # raise NotImplementedError
            # n_flares = active_region.
        return all_flares

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def mass(self):
        """
        Mass of star used for simulations
        :return: star mass
        """
        return self._mass

    @mass.setter
    def mass(self, mass):
        """
        Update the star mass and the associated orbiting body's orbits
        :param u.Quantity mass: New star mass
        """
        if mass is None:
            print("Using default star mass: 1 M_Sol")
            self._mass = u.Quantity(1, u.M_sun)
        else:
            if isinstance(mass, u.Quantity):
                self._mass = mass.to(u.solMass)
            else:
                self._mass = u.Quantity(mass, unit=u.solMass)
                warnings.warn("Casting mass, input as " + str(mass) + ", to solar masses", AstropyUserWarning)

        for orbiting_body in self._orbiting_bodies_list:
            orbiting_body.set_star_mass(self._mass)

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def earth_direction_vector(self):
        return self._earth_direction_vector

    def set_view_from_earth(self,
                            earth_longitude: (u.Quantity, Angle),
                            earth_latitude: (u.Quantity, Angle)) -> np.ndarray:
        """Sets or resets the vector that points from the star to Earth

        Parameters
        ----------
        earth_longitude
            [0, 2 pi) relative to star coordinates
        earth_latitude
            [0, pi) relative to star coordinates

        Returns
        -------
        np.ndarray
            New unit direction vector
        """
        if earth_longitude is None:
            self._earth_longitude = Angle(0, u.rad)
        else:
            if isinstance(earth_longitude, Angle) or isinstance(earth_longitude, u.Quantity):
                self._earth_longitude = earth_longitude.to(u.rad)
            else:
                self._earth_longitude = Angle(earth_longitude, unit=u.rad)
                warnings.warn("Casting earth_longitude, input as " + str(earth_longitude) + ", to radians",
                              AstropyUserWarning)

        if earth_latitude is None:
            self._earth_latitude = Angle(0, u.rad)
        else:
            if isinstance(earth_latitude, Angle) or isinstance(earth_latitude, u.Quantity):
                self._earth_latitude = earth_latitude.to(u.rad)
            else:
                self._earth_latitude = Angle(earth_latitude, unit=u.rad)
                warnings.warn("Casting earth_latitude, input as " + str(earth_latitude) + ", to radians",
                              AstropyUserWarning)

        self._earth_direction_vector = np.array((np.sin(self._earth_latitude) * np.cos(self._earth_longitude),
                                                 np.sin(self._earth_latitude) * np.sin(self._earth_longitude),
                                                 np.cos(self._earth_latitude)))
        return self._earth_direction_vector

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def star_limb(self, *args):
        """Delta star has no limb, but need a placeholder

        Parameters
        ----------
        args

        Returns
        -------

        """
        return lambda x: 1.


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class Star(DeltaStar):
    """Simple Star class, has radius, active regions, and can rotate, does not move."""

    # ------------------------------------------------------------------------------------------------------------ #
    def __init__(self,
                 mass: u.Quantity=None,
                 radius: u.Quantity=None,
                 spectral_type: SpectralEmitter=None,
                 rotation_rate: u.Quantity=None,
                 differential_rotation: float=None,
                 earth_longitude: Angle=None,
                 earth_latitude: Angle=None,
                 dist_to_earth: Distance=None,
                 limb_function: FunctionType=None,
                 limb_args: dict=None,
                 **kwargs
                 ):
        """Defines a simple star with a variety of physical properties.

        Can be given orbital objects that move around it
        Can be given different spectral properties

        Parameters
        ----------
        mass
        radius
        spectral_type
        rotation_rate
            rad/day, typically
        differential_rotation
            Relative differential rotational rate (d_omega/omega)
        earth_longitude
            [0, 2 pi)
        earth_latitude
            [0,  pi)
        dist_to_earth
            Currently not used, may be useful if absolute magnitudes get implemented
        limb_function
            Function to use for calculating limb darkening
        kwargs
        """
        super().__init__(mass=mass,
                         spectral_type=spectral_type,
                         earth_longitude=earth_longitude,
                         earth_latitude=earth_latitude,
                         dist_to_earth=dist_to_earth,
                         **kwargs)

        #  Initialize radius
        if radius is None:
            print("Using default star radius: 1 solar radius")
            self.radius = u.R_sun
        else:
            self._radius = None
            self.radius = radius

        #  Initialize rotation parameters, then pass to the set/reset function:
        self._rotation_rate = None
        self._differential_rotation = None
        self.set_rotation_parameters(rotation_rate, differential_rotation)

        #  Holder for a limb darkening model:
        if limb_function is None:
            self._limb_func = no_limb_darkening
            self._limb_args = {}
        else:
            self._limb_func = limb_function
            if limb_args is None:
                self._limb_args = {}
            else:
                self._limb_args = limb_args

    # ------------------------------------------------------------------------------------------------------------ #
    def star_limb(self,
                  angular_position: Angle,
                  star_radius_over_distance: float=0) -> float:
        """Calculates the effective intensity of a point on a surface relative to an observer position.

        Parameters
        ----------
        angular_position
            Angle between star_origin-->observer and star_origin-->point_on_star
        star_radius_over_distance
            Radius of star / Distance to star

        Returns
        -------
        float
            Value from [0, 1] for visibility of the point on the surface

        """
        return self._limb_func(angular_position, star_radius_over_distance, **self._limb_args)

    # ------------------------------------------------------------------------------------------------------------ #
    def generate_flares_over_time(self, duration: u.Quantity) -> FlareCollection:
        """Given a duration, generate a FlareCollection

        Parameters
        ----------
        duration
            Duration of time to simulate flares over

        Returns
        -------
        FlareCollection
            The dictionary of all flares that occurred and their properties
        """
        if not isinstance(duration, u.Quantity):
            raise ValueError("duration must be u.Quantity")

        duration = duration.to(lw_time_unit)
        all_flares = FlareCollection({})
        for ar_i, active_region in enumerate(self._active_region_list):
            updated_active_region = active_region(duration)
            long_i, lat_i = active_region.center_of_region
            keys = updated_active_region.sub_dict_keys
            for k in keys:
                times = updated_active_region[k]['flare_times']
                d_long_array = self.get_rotation(times, lat=lat_i)
                updated_active_region[k]['flare_longitude_array'] += d_long_array
                updated_active_region[k]['flare_vector_array'] = \
                    self.radius*vect_from_spherical_coords(updated_active_region[k]['flare_longitude_array'],
                                                           updated_active_region[k]['flare_latitude_array'])
                # For potential future applications:
                updated_active_region[k]['region_ID'] = ar_i
            all_flares.append(updated_active_region)
        return all_flares

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, radius):
        if isinstance(radius, u.Quantity):
            self._radius = radius.to(u.R_sun)
        else:
            self._radius = u.Quantity(radius, unit=u.R_sun)
            warnings.warn("Casting radius, input as " + str(radius) + ", to solar radii", AstropyUserWarning)

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def rotation_rate(self):
        """Star rotational rate in rad/day, measured at equator"""
        return self._rotation_rate

    @rotation_rate.setter
    def rotation_rate(self, rotation_rate):
        if rotation_rate is None:
            self._rotation_rate = 0 * u.rad/u.day
        else:
            if isinstance(rotation_rate, u.Quantity):
                self._rotation_rate = rotation_rate.to(u.rad/u.day)
            else:
                self._rotation_rate = u.Quantity(rotation_rate, unit=u.rad/u.day)
                warnings.warn("Casting rotation_rate, input as " + str(rotation_rate) + ", to radians/day",
                              AstropyUserWarning)

    @property
    def differential_rotation(self):
        """Star relative differential rotation rate

        See https://en.wikipedia.org/wiki/Differential_rotation"""
        return self._differential_rotation

    @differential_rotation.setter
    def differential_rotation(self, differential_rotation):
        if differential_rotation is None:
            self._differential_rotation = 0.
        else:
            if isinstance(differential_rotation, float):
                self._differential_rotation = differential_rotation
            else:
                self._differential_rotation = float(differential_rotation)
                warnings.warn("Casting differential_rotation, input as " + str(differential_rotation) +
                              ", to float", AstropyUserWarning)

    def set_rotation_parameters(self, rotation_rate, differential_rotation):
        self.rotation_rate = rotation_rate
        self.differential_rotation = differential_rotation

    def get_rotation(self,
                     time: u.Quantity,
                     lat: Angle=pi_u/2) -> u.Quantity:
        """Determine the angle of a point on the star after a given time

        Parameters
        ----------
        time
            Time since relative epoch
        lat
            Optional latitude of interest, only useful if differential rotation is in play

        Returns
        -------
        u.Quantity
            Longitudinal shift in angle at the given latitude of the star

        """
        if isinstance(time, u.Quantity):
            time = time.to(u.d)
        else:
            raise TypeError("get_rotation requires time be specified as a u.Quantity")
        return time*self.rotation_rate*(1 - self.differential_rotation*np.sin(np.abs(pi_u/2-lat))**2)

