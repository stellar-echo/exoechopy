
"""
This module provides Star classes and methods for simulations and analysis.
"""

import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from astropy.coordinates import Distance
from astropy.utils.exceptions import AstropyUserWarning

from exoechopy.simulate.models.spectral import *
from exoechopy.utils.plottables import *
from .planets import *

__all__ = ['Star']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class Star(Plottable):
    """Simple Star class, does not move."""

    # ------------------------------------------------------------------------------------------------------------ #
    def __init__(self,
                 mass=None,
                 radius=None,
                 spectral_type=None,
                 rotation_rate=None,
                 differential_rotation=None,
                 earth_longitude=None,
                 earth_latitude=None,
                 dist_to_earth=None,
                 **kwargs
                 ):
        """
        Defines a simple star with a variety of properties.
        Can be given orbital objects that move around it and different spectral properties.

        :param u.Quantity mass:
        :param u.Quantity radius:
        :param SpectralEmitter spectral_type:
        :param u.Quantity rotation_rate:
        :param float differential_rotation: relative differential rotational rate (d_omega/omega)
        :param Angle earth_longitude: [0, 2 pi)
        :param Angle earth_latitude: [0, pi)
        :param Distance dist_to_earth: Currently not used, may be useful if absolute magnitudes get implemented
        """

        super().__init__(**kwargs)

        self._position = np.zeros(3)  # placeholder for potential multi-star system considerations
        self._active_region_list = []
        self._orbiting_bodies_list = []

        #  Initialize mass, then pass to the set function:
        self._mass = None
        self.mass = mass

        #  Initialize radius
        if radius is None:
            self.radius = u.R_sun
        else:
            self._radius = None
            self.radius = radius

        if spectral_type is None:
            self._spectral_type = SpectralEmitter(JohnsonPhotometricBand('U'), magnitude=16)
        else:
            if isinstance(spectral_type, SpectralEmitter):
                self._spectral_type = spectral_type
            else:
                raise TypeError("spectral_type is not SpectralEmitter instance.")

        #  Initialize rotation parameters, then pass to the set/reset function:
        self._rotation_rate = None
        self._differential_rotation = None
        self.set_rotation_parameters(rotation_rate, differential_rotation)

        #  Initialize angles and vector, then pass to the set/reset function:
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
                warnings.warn("Casting dist_to_earth, input as " + str(dist_to_earth) + ", to LY", AstropyUserWarning)

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
            self._mass = u.M_sun
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
        """Star relative differential rotation rate, see https://en.wikipedia.org/wiki/Differential_rotation"""
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

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def earth_direction_vector(self):
        return self._earth_direction_vector

    def set_view_from_earth(self, earth_longitude, earth_latitude):
        """
        Sets or resets the vector that points from the star to Earth
        :param Angle earth_longitude: [0, 2 pi)
        :param Angle earth_latitude: [0, pi)
        :return np.array: new unit direction vector
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
    def add_exoplanet(self, exoplanet):
        """
        Add an Exoplanet object to the star
        :param Exoplanet exoplanet:
        """
        if isinstance(exoplanet, Exoplanet):
            exoplanet.star_mass = self._mass
            self._orbiting_bodies_list.append(exoplanet)
        else:
            raise TypeError("exoplanet must be an Exoplanet class")

    # ------------------------------------------------------------------------------------------------------------ #
    def get_exoplanets(self):
        return self._orbiting_bodies_list.copy()

            # if x is None:
        #     self._x = zz
        # else:
        #     if isinstance(x, u.Quantity):
        #         self._x = x.to(zz)
        #     else:
        #         self._x = u.Quantity(x, unit=zz)
        #         warnings.warn("Casting x, input as " + str(x) + ", to zz", AstropyUserWarning)

    # ------------------------------------------------------------------------------------------------------------ #
    def get_position(self, *args):
        return self._position
