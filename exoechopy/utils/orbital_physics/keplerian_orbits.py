
"""
This module provides simple Keplerian orbits for simulations and analysis.
"""

import warnings
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from astropy.coordinates import Distance
from astropy import constants as const
from astropy.utils.exceptions import AstropyUserWarning
from ..plottables import *
from ..astropyio import *
from ..globals import *

__all__ = ['KeplerianOrbit']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class KeplerianOrbit(Plottable):
    """
    Base class for objects that travel on a Keplerian Orbit, solved with Newton's method.
    """
    # ------------------------------------------------------------------------------------------------------------ #
    def __init__(self,
                 semimajor_axis=None,
                 eccentricity=None,
                 star_mass=None,
                 initial_anomaly=None,
                 inclination=None,
                 longitude=None,
                 periapsis_arg=None,
                 **kwargs):
        """
        Provides a base class for objects that obey a Keplerian orbit.
        Provides several methods for retrieving relevant orbital parameters, such as 3D position as a function of time.
        Solves using Newton's method, not recommended for unusual or complex orbits.

        :param Distance semimajor_axis: semimajor axis as an astropy quantity
        Default 1 AU
        :param float eccentricity: orbit eccentricity
        :param u.Quantity star_mass: mass of star as an astropy quantity
        Default 1 M_sun
        :param Angle initial_anomaly: initial anomaly relative to star
        :param Angle inclination: orbital inclination relative to star
        :param Angle longitude: longitude of ascending node
        :param Angle periapsis_arg: argument of the periapsis
        """
        super().__init__(**kwargs)

        if semimajor_axis is None:
            self._semimajor_axis = Distance(1, unit=u.au)
        else:
            if isinstance(semimajor_axis, u.Quantity) or isinstance(semimajor_axis, Distance):
                self._semimajor_axis = semimajor_axis.to(u.au)
            else:
                self._semimajor_axis = Distance(semimajor_axis, unit=u.au)
                warnings.warn("Casting semimajor axis, input as "+str(semimajor_axis)+", to AU", AstropyUserWarning)

        if eccentricity is None:
            self._eccentricity = 0
        else:
            if 0 <= eccentricity < 1:
                self._eccentricity = eccentricity
            else:
                raise ValueError("Eccentricity, input as" + str(eccentricity) + ", must be 0 <= e < 1")
        self._eccentric_factor = np.sqrt((1 + self._eccentricity) / (1 - self._eccentricity))

        #  Initialize star_mass, then call set/reset function:
        self._star_mass = None
        self._grav_param = None
        self._orbital_period = None
        self._orbital_frequency = None
        self.star_mass = star_mass  # Must be called after defining semimajor_axis, enforce later

        if initial_anomaly is None or initial_anomaly == 0.:
            self._initial_anomaly = Angle(0., unit=u.rad)
        else:
            if isinstance(initial_anomaly, Angle) or isinstance(initial_anomaly, u.Quantity):
                self._initial_anomaly = initial_anomaly.to(u.rad)
            else:
                self._initial_anomaly = Angle(initial_anomaly, unit=u.rad)
                warnings.warn("Casting initial_anomaly, input as " + str(initial_anomaly) + ", to radians",
                              AstropyUserWarning)

        if inclination is None or inclination == 0.:
            self._inclination = Angle(0., unit=u.rad)
        else:
            if isinstance(inclination, Angle) or isinstance(inclination, u.Quantity):
                self._inclination = inclination.to(u.rad)
            else:
                self._inclination = Angle(inclination, unit=u.rad)
                warnings.warn("Casting inclination, input as " + str(inclination) + ", to radians",
                              AstropyUserWarning)

        if longitude is None or longitude == 0.:
            self._longitude = Angle(0., unit=u.rad)
        else:
            if isinstance(longitude, Angle) or isinstance(longitude, u.Quantity):
                self._longitude = longitude.to(u.rad)
            else:
                self._longitude = Angle(longitude, unit=u.rad)
                warnings.warn("Casting longitude, input as " + str(longitude) + ", to radians",
                              AstropyUserWarning)

        if periapsis_arg is None or periapsis_arg == 0.:
            self._periapsis_arg = Angle(0., unit=u.rad)
        else:
            if isinstance(periapsis_arg, Angle) or isinstance(periapsis_arg, u.Quantity):
                self._periapsis_arg = periapsis_arg.to(u.rad)
            else:
                self._periapsis_arg = Angle(periapsis_arg, unit=u.rad)
                warnings.warn("Casting periapsis_arg, input as " + str(periapsis_arg) + ", to radians",
                              AstropyUserWarning)

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def star_mass(self):
        return self._star_mass

    @star_mass.setter
    def star_mass(self, star_mass):
        if star_mass is None:
            self._star_mass = u.Quantity(1, u.M_sun)
        else:
            if isinstance(star_mass, u.Quantity):
                self._star_mass = star_mass
            else:
                self._star_mass = u.Quantity(star_mass, u.M_sun)
                warnings.warn("Casting star mass, input as "+str(star_mass)+", to M_sun", AstropyUserWarning)
        self._grav_param = self._star_mass * const.G
        self._orbital_period = (2 * np.pi * self._semimajor_axis**1.5 * np.sqrt(1/self._grav_param)).decompose().to(u.d)
        self._orbital_frequency = np.sqrt(self._grav_param/self._semimajor_axis**3).decompose()

    # ------------------------------------------------------------------------------------------------------------ #
    def about_orbit(self):
        """
        Prints out information about the Keplerian orbit
        :return: None
        """
        print("Star mass: ", u_str(self._star_mass))
        print("Semimajor axis: ", u_str(self._semimajor_axis))
        print("Orbital period: ", u_str(self._orbital_period))
        print("Orbit eccentricity, e: ", self._eccentricity)
        print("Orbit inclination, i: ", self._inclination)
        print("Orbit longitude of ascending node: ", self._longitude)
        print("Orbit periapsis: ", self._periapsis_arg)
        print("Initial anomaly: ", self._initial_anomaly)

    # ------------------------------------------------------------------------------------------------------------ #
    def distance_at_angle(self, theta):
        """
        Provides the distance at an angle, or np.array of angles
        :param Angle theta: Angle, or array of angles, relative to star system to evaluate the distance at
        :return u.Quantity: Returns the distance from the star as a function of angle
        """
        if not (isinstance(theta, Angle) or isinstance(theta, u.Quantity)):
            theta = Angle(theta, unit=u.rad)
            warnings.warn("Casting theta, input as " + str(theta) + ", to radians", AstropyUserWarning)
        return self._semimajor_axis * (1 - self._eccentricity ** 2) \
               / (1 + self._eccentricity * np.cos(theta + self._initial_anomaly))

    # ------------------------------------------------------------------------------------------------------------ #
    def calc_xyz_position_in_au(self, theta):
        """
        Provides the 3D position of the object for a given angle
        :param Angle theta:
        :return np.array:
        """
        r = self.distance_at_angle(theta)
        true_anom = self._periapsis_arg + theta
        x = r*(np.cos(self._longitude)*np.cos(true_anom)
               - np.sin(self._longitude)*np.sin(true_anom)*np.cos(self._inclination))
        y = r*(np.sin(self._longitude)*np.cos(true_anom)
               + np.cos(self._longitude)*np.sin(true_anom)*np.cos(self._inclination))
        z = r*(np.sin(self._inclination)*np.sin(true_anom))
        return np.array((x.to(u.au).value, y.to(u.au).value, z.to(u.au).value))

    # ------------------------------------------------------------------------------------------------------------ #
    def generate_orbital_positions(self, num_points):
        """
        Returns a list of positions along the orbit
        :param int num_points: Number of points to plot
        :return list:
        """
        all_angles = np.linspace(0 * u.rad, 2 * pi_u, num_points)
        orbit_positions = [self.calc_xyz_position_in_au(ang) for ang in all_angles]
        return orbit_positions


def kepler(x, e, M):
    return x - e * np.sin(x) - M


def D_kepler(x, e, M):
    return 1 - e*np.cos(x)


def D2_kepler(x, e, M):
    return e*np.sin(x)
