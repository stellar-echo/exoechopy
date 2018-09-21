
"""
This module provides Planet classes and methods for simulations and analysis.
"""

import warnings

from astropy import units as u
from astropy.coordinates import Angle
from astropy.coordinates import Distance
from astropy.utils.exceptions import AstropyUserWarning

from .orbital_physics import *
from .spectral import *
from ..utils.plottables import *


__all__ = ['Exoplanet', 'KeplerianExoplanet']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class Exoplanet(Plottable):
    """Exoplanet base class."""

    # ------------------------------------------------------------------------------------------------------------ #
    def __init__(self,
                 contrast: float=None,
                 albedo: (float, Albedo)=None,
                 semimajor_axis: u.Quantity=None,
                 radius: u.Quantity=None,
                 mass: u.Quantity=None,
                 **kwargs
                 ):
        """Simplest class of exoplanets.

        Parameters
        ----------
        contrast
            Exoplanet-star contrast, *overrides albedo if provided*
        albedo
            Albedo instance or float in [0, 1]
        semimajor_axis
            Semimajor axis of orbit
        radius
            Radius of exoplanet
        mass
            Mass of exoplanet
        kwargs
            kwargs are currently only passed to Plottable
        """

        super().__init__(**kwargs)

        if contrast is None:
            self._contrast = None
            if isinstance(albedo, (int, float)):
                self._albedo = float(albedo)
            elif isinstance(albedo, Albedo):
                self._albedo = albedo
            elif albedo is None:
                self._albedo = None
            else:
                raise TypeError("Albedo, input as" + str(albedo) + ", doesn't make sense to Exoplanet.__init__.")
        else:
            if isinstance(contrast, float):
                self._contrast = contrast
                self._albedo = None

        if semimajor_axis is None:
            self._semimajor_axis = Distance(1, unit=u.au)
        else:
            if isinstance(semimajor_axis, u.Quantity) or isinstance(semimajor_axis, Distance):
                self._semimajor_axis = semimajor_axis.to(u.au)
            else:
                self._semimajor_axis = Distance(semimajor_axis, unit=u.au)
                warnings.warn("Casting semimajor axis, input as "+str(semimajor_axis)+", to AU", AstropyUserWarning)

        if radius is None:
            warnings.warn("Exoplanet has no radius", AstropyUserWarning)
            self._radius = None
        else:
            if isinstance(radius, u.Quantity):
                self._radius = radius.to(u.R_jup)
            else:
                self._radius = u.Quantity(radius, unit=u.R_jup)
                warnings.warn("Casting radius, input as " + str(radius) + ", to Jupiter radii", AstropyUserWarning)

        if mass is None:
            self._mass = None
        else:
            if isinstance(mass, u.Quantity):
                self._mass = mass.to(u.M_jup)
            else:
                self._mass = u.Quantity(mass, unit=u.M_jup)
                warnings.warn("Casting mass, input as " + str(mass) + ", to Jupiter masses", AstropyUserWarning)

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def semimajor_axis(self):
        return self._semimajor_axis

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def radius(self):
        return self._radius

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def mass(self):
        return self._mass

    # ------------------------------------------------------------------------------------------------------------ #
    def get_echo_magnitude(self,
                           dist_to_source: (Distance, u.Quantity)=None,
                           earth_angle: (u.Quantity, Angle)=None,
                           *phase_law_args) -> float:
        """Calculate the echo magnitude from the planet

        If contrast was specified, will return that value
        Otherwise, will use the albedo and planet properties to calculate

        Parameters
        ----------
        dist_to_source
            Distance between planet and light source
            If None, will use semimajor_axis
        earth_angle
            Used for phase law calculations

        Returns
        -------

        """
        if self._contrast is not None:
            return self._contrast
        else:
            if dist_to_source is None:
                dist_to_source = self.semimajor_axis
                warnings.warn("No distance specified, using semimajor axis for echo magnitude", AstropyUserWarning)
            # echo_relative_magnitude works with Albedo objects to compute phase law:
            return echo_relative_magnitude(distance_to_star=dist_to_source,
                                           phase_angle=earth_angle,
                                           geometric_albedo=self._albedo,
                                           planet_radius=self.radius,
                                           *phase_law_args)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class KeplerianExoplanet(KeplerianOrbit, Exoplanet):
    """Simple planet class, follows a Keplerian orbit."""

    # ------------------------------------------------------------------------------------------------------------ #
    def __init__(self,
                 semimajor_axis=None,
                 eccentricity=None,
                 star_mass=None,
                 initial_anomaly=None,
                 inclination=None,
                 longitude=None,
                 periapsis_arg=None,
                 albedo=None,
                 radius=None,
                 **kwargs
                 ):
        """
        Provides a Keplerian exoplanet for simulations and models.

        :param Distance semimajor_axis: semimajor axis as an astropy quantity
        Default 1 AU
        :param float eccentricity: orbit eccentricity
        :param u.Quantity star_mass: mass of star as an astropy quantity
        Default 1 M_sun
        :param Angle initial_anomaly: initial anomaly relative to star
        :param Angle inclination: orbital inclination relative to star
        :param Angle longitude: longitude of ascending node
        :param Angle periapsis_arg: argument of the periapsis
        :param Albedo albedo: an Albedo class, which can include special phase laws
        :param u.quantity radius:  Radius of exoplanet
        """
        super().__init__(semimajor_axis=semimajor_axis, eccentricity=eccentricity, star_mass=star_mass,
                         initial_anomaly=initial_anomaly, inclination=inclination, longitude=longitude,
                         periapsis_arg=periapsis_arg, albedo=albedo, radius=radius, **kwargs)




