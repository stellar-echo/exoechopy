
"""
This module provides simple Keplerian orbits for simulations and analysis.
"""

import warnings
import numpy as np
from scipy import optimize
from astropy import units as u
from astropy.coordinates import Angle
from astropy.coordinates import Distance
from astropy import constants as const
from astropy.utils.exceptions import AstropyUserWarning
from ....utils.plottables import *
from ....utils.astropyio import *
from ....utils.constants import *

__all__ = ['KeplerianOrbit', 'true_anomaly_from_mean']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

# TODO Fix the problem with initial_anomaly causing garbage results

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

        # Initialize semimajor axis
        self._semimajor_axis = None
        self._semimajor_axis_lw = None
        if semimajor_axis is None:
            self.semimajor_axis = Distance(1, unit=u.au)
        else:
            self.semimajor_axis = semimajor_axis

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

        #  Initialize initial anomaly
        self._initial_anomaly = None
        self._initial_anomaly_lw = None
        if initial_anomaly is None or initial_anomaly == 0.:
            self.initial_anomaly = Angle(0., unit=u.rad)
        else:
            self.initial_anomaly = initial_anomaly

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

        # Initialize argument of the periapsis
        self._periapsis_arg_lw = None
        self._periapsis_arg = None
        if periapsis_arg is None or periapsis_arg == 0.:
            self.periapsis_arg = Angle(0., unit=u.rad)
        else:
            self.periapsis_arg = periapsis_arg

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def semimajor_axis(self):
        return self._semimajor_axis

    @semimajor_axis.setter
    def semimajor_axis(self, semimajor_axis):
        if isinstance(semimajor_axis, u.Quantity) or isinstance(semimajor_axis, Distance):
            self._semimajor_axis = semimajor_axis.to(u.au)
        else:
            self._semimajor_axis = Distance(semimajor_axis, unit=u.au)
            warnings.warn("Casting semimajor axis, input as "+str(semimajor_axis)+", to AU", AstropyUserWarning)
        self._semimajor_axis_lw = self._semimajor_axis.to(lw_distance_unit).value

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
        self._orbital_period = (2 * np.pi * self._semimajor_axis**1.5 * np.sqrt(1/self._grav_param)).decompose().to(u.s)
        self._orbital_frequency = np.sqrt(self._grav_param/self._semimajor_axis**3).decompose().to(u.Hz)*u.rad

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def periapsis_arg(self):
        return self._periapsis_arg

    @periapsis_arg.setter
    def periapsis_arg(self, periapsis_arg):
        if periapsis_arg == 0.:
            self._periapsis_arg = Angle(0., unit=u.rad)
        else:
            if isinstance(periapsis_arg, Angle) or isinstance(periapsis_arg, u.Quantity):
                self._periapsis_arg = periapsis_arg.to(u.rad)
            else:
                self._periapsis_arg = Angle(periapsis_arg, unit=u.rad)
                warnings.warn("Casting periapsis_arg, input as " + str(periapsis_arg) + ", to radians",
                              AstropyUserWarning)
        self._periapsis_arg_lw = self._periapsis_arg.value

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def initial_anomaly(self):
        return self._initial_anomaly

    @initial_anomaly.setter
    def initial_anomaly(self, initial_anomaly):
        if initial_anomaly == 0.:
            self._initial_anomaly = Angle(0., unit=u.rad)
        else:
            if isinstance(initial_anomaly, Angle) or isinstance(initial_anomaly, u.Quantity):
                self._initial_anomaly = initial_anomaly.to(u.rad)
            else:
                self._initial_anomaly = Angle(initial_anomaly, unit=u.rad)
                warnings.warn("Casting initial_anomaly, input as " + str(initial_anomaly) + ", to radians",
                              AstropyUserWarning)
        self._initial_anomaly_lw = self._initial_anomaly.value

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def orbital_period(self):
        return self._orbital_period

    # ------------------------------------------------------------------------------------------------------------ #
    def about_orbit(self):
        """
        Prints out information about the Keplerian orbit
        :return: None
        """
        if self._name != "":
            print("Planet name: ", self._name)
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
        :param Angle theta: Angle, or array of angles, relative to periapsis to evaluate the distance at
        :return u.Quantity: Returns the distance from the star as a function of angle
        """
        if not (isinstance(theta, Angle) or isinstance(theta, u.Quantity)):
            theta = Angle(theta, unit=u.rad)
            warnings.warn("Casting theta, input as " + str(theta) + ", to radians", AstropyUserWarning)
        return self._semimajor_axis * (1 - self._eccentricity ** 2) / \
               (1 + self._eccentricity * np.cos(theta - self._periapsis_arg))

    def _distance_at_angle_radians_lw(self, theta):
        """
        Lightweight version, provides distance at a radian angle or np.array of angles without unit checking
        :param float theta: Angle, or array of angles, relative to periapsis to evaluate the distance at
        :return float: Returns the distance from the star as a function of angle
        """
        return self._semimajor_axis_lw * (1 - self._eccentricity ** 2) / \
               (1 + self._eccentricity * np.cos(theta - self._periapsis_arg_lw))

    # ------------------------------------------------------------------------------------------------------------ #
    def distance_at_time(self, t0):
        """
        Provides the distance at a time, or np.array of times
        :param u.Quantity t0: Time for evaluation
        :return u.Quantity: Distance from star at the designated time(s)
        """
        mean_anomaly = (self._orbital_frequency*t0 + self._initial_anomaly)
        root = optimize.newton(kepler, mean_anomaly, args=(self._eccentricity, mean_anomaly),
                               tol=1.48e-09, maxiter=500,
                               fprime=D_kepler, fprime2=D2_kepler)
        root %= 2*np.pi
        theta = 2*np.arctan(self._eccentric_factor*np.tan(root/2))
        return u.Quantity(self._distance_at_angle_radians_lw(theta), unit=u.au)

    # ------------------------------------------------------------------------------------------------------------ #
    def calc_xyz_at_angle_au(self, theta):
        """
        Provides the 3D position of the object relative to star for a given angle
        :param Angle theta: true anomaly of the object
        :return np.array: xyz array for position as a function of angle
        """
        true_anom = theta + self._periapsis_arg
        r = self.distance_at_angle(true_anom)
        x = r*(np.cos(self._longitude)*np.cos(true_anom)
               - np.sin(self._longitude)*np.sin(true_anom)*np.cos(self._inclination))
        y = r*(np.sin(self._longitude)*np.cos(true_anom)
               + np.cos(self._longitude)*np.sin(true_anom)*np.cos(self._inclination))
        z = r*(np.sin(self._inclination)*np.sin(true_anom))
        return np.array((x.to(u.au).value, y.to(u.au).value, z.to(u.au).value))

    def _calc_xyz_at_angle_au_lw(self, theta):
        """
        Provides the unitless 3D position of the object relative to star for a given angle
        :param Angle theta: true anomaly of the object
        :return np.array: xyz array for position as a function of angle
        """
        true_anom = theta + self._periapsis_arg_lw
        r = self.distance_at_angle(true_anom)
        x = r*(np.cos(self._longitude)*np.cos(true_anom)
               - np.sin(self._longitude)*np.sin(true_anom)*np.cos(self._inclination))
        y = r*(np.sin(self._longitude)*np.cos(true_anom)
               + np.cos(self._longitude)*np.sin(true_anom)*np.cos(self._inclination))
        z = r*(np.sin(self._inclination)*np.sin(true_anom))
        return np.array((x.to(u.au).value, y.to(u.au).value, z.to(u.au).value))

    # ------------------------------------------------------------------------------------------------------------ #
    def calc_xyz_at_time_au(self, t0):
        """
        Provides the distance at a time, or np.array of times
        :param u.Quantity t0: Time for evaluation
        :return u.Quantity: Distance from star at the designated time(s)
        """
        mean_anomaly = (self._orbital_frequency * t0 + self._initial_anomaly).to(u.rad).value
        root = optimize.newton(kepler, mean_anomaly, args=(self._eccentricity, mean_anomaly),
                               tol=1.48e-09, maxiter=500,
                               fprime=D_kepler, fprime2=D2_kepler)
        root %= 2 * np.pi
        theta = 2 * np.arctan(self._eccentric_factor * np.tan(root / 2))
        return self.calc_xyz_at_angle_au(theta*u.rad)

    # ------------------------------------------------------------------------------------------------------------ #
    def calc_xyz_ascending_node_au(self):
        """
        Provides the 3D position of the ascending node, hopefully
        :return np.array:
        """
        theta = 0*u.deg
        true_anom = theta
        r = self.distance_at_angle(true_anom)
        x = r*(np.cos(self._longitude)*np.cos(true_anom)
               - np.sin(self._longitude)*np.sin(true_anom)*np.cos(self._inclination))
        y = r*(np.sin(self._longitude)*np.cos(true_anom)
               + np.cos(self._longitude)*np.sin(true_anom)*np.cos(self._inclination))
        z = r*(np.sin(self._inclination)*np.sin(true_anom))
        return np.array((x.to(u.au).value, y.to(u.au).value, z.to(u.au).value))

    # ------------------------------------------------------------------------------------------------------------ #
    def generate_orbital_positions_by_angle(self, num_points):
        """
        Returns a list of positions along the path for a single orbit
        :param int num_points: Number of points to plot
        :return list:
        """
        all_angles = np.linspace(0 * u.rad, 2 * pi_u, num_points)
        orbit_positions = [self.calc_xyz_at_angle_au(ang + self.initial_anomaly) for ang in all_angles]
        return orbit_positions

    # ------------------------------------------------------------------------------------------------------------ #
    def generate_orbital_positions_by_time(self, num_points):
        """
        Returns a list of positions along the path for a single orbit
        :param int num_points: Number of points to plot
        :return list:
        """
        all_times = np.linspace(0 * u.s, self.orbital_period, num_points)
        orbit_positions = [self.calc_xyz_at_time_au(ti) for ti in all_times]
        return orbit_positions

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


# Some internal functions for solving Kepler's equation by finding the root, using the pre-computed derivatives
def kepler(x, e, M):
    return x - e * np.sin(x) - M


def D_kepler(x, e, M):
    return 1 - e*np.cos(x)


def D2_kepler(x, e, M):
    return e*np.sin(x)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def true_anomaly_from_mean(mean_anomaly, e):
    """
    Calculates the true anomaly from the mean anomaly.  Based on https://en.wikipedia.org/wiki/True_anomaly
    :param Angle mean_anomaly:
    :param float e: eccentricity
    :return u.Quantity:
    """
    if isinstance(mean_anomaly, Angle) or isinstance(mean_anomaly, u.Quantity):
        _M = mean_anomaly.to(u.rad)
    else:
        _M = Angle(mean_anomaly, unit=u.rad)
        warnings.warn("Casting initial_anomaly, input as " + str(mean_anomaly) + ", to radians",
                      AstropyUserWarning)
    return_val = _M + (2*e-.25*e**3)*np.sin(_M)*u.rad+5*e**2/4*np.sin(2*_M)*u.rad+13*e**3/12*np.sin(3*_M)*u.rad
    return return_val


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

