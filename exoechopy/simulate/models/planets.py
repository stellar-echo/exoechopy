
"""
This module provides Planet classes and methods for simulations and analysis.
"""

import warnings
from astropy import units as u
from exoechopy.utils.spectral import *
from exoechopy.utils.orbital_physics import *
from exoechopy.utils.plottables import *
from astropy.coordinates import Angle
from astropy.coordinates import Distance
from astropy.utils.exceptions import AstropyUserWarning

__all__ = ['Exoplanet', 'KeplerianExoplanet']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class Exoplanet(Plottable):
    """Exoplanet base class."""

    # ------------------------------------------------------------------------------------------------------------ #
    def __init__(self,
                 contrast=None,
                 albedo=None,
                 semimajor_axis=None,
                 radius=None,
                 mass=None,
                 **kwargs
                 ):
        """
        Simplest class of exoplanets.
        :param float contrast: exoplanet-star contrast, overrides albedo if provided
        :param float albedo: [0, 1]
        :param u.Quantity semimajor_axis:
        :param u.Quantity radius:
        :param u.Quantity mass:
        """
        super().__init__(**kwargs)

        if contrast is None:
            self._contrast = None
            if isinstance(albedo, float) or albedo == 0 or albedo == 1:
                self._albedo = float(albedo)
            elif isinstance(albedo, Albedo):
                self._albedo = albedo
            elif albedo is None:
                self._albedo = None
            else:
                raise TypeError("Albedo, input as" + str(albedo) + ", doesn't make sense.")
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




