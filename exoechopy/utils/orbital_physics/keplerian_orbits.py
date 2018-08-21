
"""
This module provides simple Keplerian orbits for analysis
"""

import warnings
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from astropy.coordinates import Distance
from astropy.constants import c, M_sun, G
from astropy.utils.exceptions import AstropyUserWarning

__all__ = ['KeplerianOrbit']

class KeplerianOrbit(object):
    """
    Base class for objects that travel on a Keplerian Orbit, solved with Newton's method.
    """

    def __init__(self,
                 semimajor_axis=None,
                 eccentricity=None,
                 star_mass=None,
                 initial_anomaly=None,
                 inclination=None,
                 longitude=None,
                 periapsis_arg=None):
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

        if semimajor_axis is None:
            self._semimajor_axis = Distance(1, unit=u.au)
        else:
            if isinstance(semimajor_axis, u.Quantity):
                self._semimajor_axis = semimajor_axis
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

        if star_mass is None:
            self._star_mass = u.Quantity(1, M_sun)
        else:
            if isinstance(star_mass, u.Quantity):
                self._star_mass = star_mass
            else:
                self._star_mass = u.Quantity(star_mass, M_sun)
                warnings.warn("Casting star mass, input as "+str(star_mass)+", to M_sun", AstropyUserWarning)

        if initial_anomaly is None or initial_anomaly == 0.:
            self._initial_anomaly = Angle(0., unit=u.rad)
        else:
            if isinstance(initial_anomaly, Angle):
                self._initial_anomaly = initial_anomaly
            else:
                self._initial_anomaly = Angle(initial_anomaly, unit=u.rad)
                warnings.warn("Casting initial_anomaly, input as " + str(initial_anomaly) + ", to radians",
                              AstropyUserWarning)

        if inclination is None or inclination == 0.:
            self._inclination = Angle(0., unit=u.rad)
        else:
            if isinstance(inclination, Angle):
                self._inclination = inclination
            else:
                self._inclination = Angle(inclination, unit=u.rad)
                warnings.warn("Casting inclination, input as " + str(inclination) + ", to radians",
                              AstropyUserWarning)

        if longitude is None or longitude == 0.:
            self._longitude = Angle(0., unit=u.rad)
        else:
            if isinstance(longitude, Angle):
                self._longitude = longitude
            else:
                self._longitude = Angle(longitude, unit=u.rad)
                warnings.warn("Casting longitude, input as " + str(longitude) + ", to radians",
                              AstropyUserWarning)

        if periapsis_arg is None or periapsis_arg == 0.:
            self._periapsis_arg = Angle(0., unit=u.rad)
        else:
            if isinstance(periapsis_arg, Angle):
                self._periapsis_arg = periapsis_arg
            else:
                self._periapsis_arg = Angle(periapsis_arg, unit=u.rad)
                warnings.warn("Casting periapsis_arg, input as " + str(periapsis_arg) + ", to radians",
                              AstropyUserWarning)

        self._grav_param = self._star_mass * G
        self._orbital_period = 2*np.pi * self._semimajor_axis**1.5 * np.sqrt(1/self._grav_param)
        self._orbital_frequency = np.sqrt(self._grav_param/self._semimajor_axis**3)
        self._eccentric_factor = np.sqrt((1+self._eccentricity)/(1-self._eccentricity))

    def about_orbit(self):
        """
        Prints out information about the Keplerian orbit
        :return: None
        """
        print("Star mass: ", self._star_mass.tostring())
        print("Semimajor axis: ", self._semimajor_axis.tostring())
        print("Orbital period: ", self._orbital_period.tostring())
        print("Orbit eccentricity, e: ", self._eccentricity)
        print("Orbit inclination, i: ", self._inclination)
        print("Orbit longitude of ascending node: ", self._longitude)
        print("Orbit periapsis: ", self._periapsis_arg)
        print("Initial anomaly: ", self._initial_anomaly)

    def distance_at_angle(self, theta):
        """
        Provides the distance at an angle, or np.array of angles
        :param Angle theta: Angle, or array of angles, relative to star system to evaluate the distance at
        :return u.Quantity: Returns the distance from the star as a function of angle
        """
        if not isinstance(theta, Angle):
            theta = Angle(theta, unit=u.rad)
            warnings.warn("Casting theta, input as " + str(theta) + ", to radians", AstropyUserWarning)
        return self._semimajor_axis * (1 - self._eccentricity ** 2) / \
               (1 + self._eccentricity * np.cos(theta + self._initial_anomaly))


def kepler(x, e, M):
    return x - e * np.sin(x) - M


def D_kepler(x, e, M):
    return 1 - e*np.cos(x)


def D2_kepler(x, e, M):
    return e*np.sin(x)
