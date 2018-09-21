
"""
This module provides models for stellar limb opacity and intensity.
"""

import warnings
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from astropy.utils.exceptions import AstropyUserWarning

from ..utils.constants import *


__all__ = ['no_limb_darkening',
           'calculate_basic_limb_darkening', '_calculate_basic_limb_darkening_lw', 'limb_darkened_radial_position']

# TODO Implement spectral dependencies
# TODO Create a Limb class?  Could handle spectral stuff and other models more easily?
# TODO Handle transit cases a little
# TODO Handle flare depth somehow
# TODO Handle flares slightly behind star, but still visible through limb

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def no_limb_darkening(angular_position: Angle,
                      star_radius_over_distance: float=0,
                      **kwargs) -> float:
    """Returns 1 if visible from the position, 0 if on the far side of the star.

    Parameters
    ----------
    angular_position
        Angle between star_origin-->observer and star_origin-->point_on_star
    star_radius_over_distance
        Radius of star / Distance to star
    kwargs
        To cooperate with other limb darkening arguments

    Returns
    -------
    float
        0 or 1 depending on if the point is on the far side of the star
    """
    if isinstance(angular_position, Angle) or isinstance(angular_position, u.Quantity):
        _angular_position = angular_position.to(u.rad)
    else:
        _angular_position = Angle(angular_position, unit=u.rad)
        warnings.warn("Casting angular_position, input as " + str(angular_position) + ", to radians",
                      AstropyUserWarning)

    mod_angle = np.mod(_angular_position.value, 2 * np.pi)*u.rad
    if pi_u / 2 <= mod_angle <= 3 * pi_u / 2:  # If it's beyond the limb from any distance...
        return 0.

    max_limb_angle = np.arcsin(star_radius_over_distance)*u.rad
    # If viewed from Earth, max_limb_angle is effectively 0
    if max_limb_angle.value == 0:
        # Already ruled out that it's not behind the star in previous if test
        return 1.

    # If user requests an angle on the other side of the star, currently assume star is opaque and blocks light:
    max_angle = pi_u/2 - max_limb_angle
    if np.abs(_angular_position) > max_angle:
        return 0.

    # If viewed within the star system at a realistic angle:
    else:
        return 1.


#  =============================================================  #


def calculate_basic_limb_darkening(angular_position: Angle,
                                   star_radius_over_distance: float=0,
                                   **kwargs) -> float:
    """Calculates the effective intensity of a point on a surface relative to an observer position.

    Parameters
    ----------
    angular_position
        Angle between star_origin-->observer and star_origin-->point_on_star
    star_radius_over_distance
        Radius of star / Distance to star
    kwargs
        Arguments for limb darkening equation

    Returns
    -------
    float
        Value from [0, 1] for visibility of the point on the surface

    """
    if isinstance(angular_position, Angle) or isinstance(angular_position, u.Quantity):
        _angular_position = angular_position.to(u.rad)
    else:
        _angular_position = Angle(angular_position, unit=u.rad)
        warnings.warn("Casting angular_position, input as " + str(angular_position) + ", to radians",
                      AstropyUserWarning)

    mod_angle = np.mod(_angular_position.value, 2 * np.pi)*u.rad
    if pi_u / 2 <= mod_angle <= 3 * pi_u / 2:  # If it's beyond the limb from any distance...
        return 0

    max_limb_angle = np.arcsin(star_radius_over_distance)*u.rad
    # If viewed from Earth, max_limb_angle is effectively 0
    if max_limb_angle.value == 0:
        return limb_darkened_radial_position(np.sin(_angular_position), **kwargs)

    # If user requests an angle on the other side of the star, currently assume star is opaque and blocks light:
    max_angle = pi_u/2 - max_limb_angle
    if np.abs(_angular_position) > max_angle:
        return 0.

    # If viewed within the star system at a realistic angle:
    else:
        theta_angle = np.arctan2(np.sin(_angular_position), 1 / star_radius_over_distance - np.cos(_angular_position))
        theta_angle = np.mod(theta_angle, 2 * pi_u)
    # if max_limb_angle <= theta_angle <= 2*np.pi-max_limb_angle:
    #     return 0.
    # else:
        return limb_darkened_phase(theta_angle, max_limb_angle, **kwargs)


def _calculate_basic_limb_darkening_lw(angular_position, star_radius_over_distance=0, **kwargs):
    """
    Lightweight version, does not track or verify units
    Calculates the effective intensity of a point on a surface relative to an observer position.

    :param angular_position: Angle between star_origin-->observer and star_origin-->point_on_star
    :param star_radius_over_distance: radius of star / Distance to star
    :return:
    """
    mod_angle = np.mod(angular_position, 2 * np.pi)
    if np.pi / 2 <= mod_angle <= 3 * np.pi / 2:  # If it's beyond the limb from any distance...
        return 0

    max_limb_angle = np.arcsin(star_radius_over_distance)
    # If viewed from Earth, max_limb_angle is effectively 0
    if max_limb_angle == 0:
        return limb_darkened_radial_position(np.sin(angular_position), **kwargs)

    # If user requests an angle on the other side of the star, currently assume star is opaque and blocks light:
    max_angle = np.pi/2 - max_limb_angle
    if np.abs(angular_position) > max_angle:
        return 0.

    # If viewed within the star system at a realistic angle:
    else:
        theta_angle = np.arctan2(np.sin(angular_position), 1 / star_radius_over_distance - np.cos(angular_position))
        theta_angle = np.mod(theta_angle, 2 * np.pi)
    # if max_limb_angle <= theta_angle <= 2*np.pi-max_limb_angle:
    #     return 0.
    # else:
        return _limb_darkened_phase_lw(theta_angle, max_limb_angle, **kwargs)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def limb_darkened_radial_position(radial_position, a0=.3, a1=.93, a2=-.23, **kwargs):
    """
    Cosine expansion version of relative intensity as a function of fractional radial position on the star.
    https://en.wikipedia.org/wiki/Limb_darkening
    :param float radial_position: fraction of radius on star where flare occurred, in polar coordinates.  [0-1]
    :param float a0: 0th order term
    :param float a1: 1st order term
    :param float a2: 2nd order term
    :return float: relative limb intensity
    """
    cos_phi_squared = 1 - radial_position ** 2
    return a0 + a1*cos_phi_squared**.5 + a2*cos_phi_squared


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def limb_darkened_phase(observation_angle, max_limb_angle, a0=.3, a1=.93, a2=-.23, **kwargs):
    """
    Computes the intensity of the star based on
    :param Angle observation_angle: Angle between observer-->star and observer-->point_on_star
    :param Angle max_limb_angle:
    :param float a0: 0th order term
    :param float a1: 1st order term
    :param float a2: 2nd order term
    :return float:
    """
    if observation_angle == 0.:
        _observation_angle = Angle(0., unit=u.rad)
    else:
        if isinstance(observation_angle, Angle) or isinstance(observation_angle, u.Quantity):
            _observation_angle = observation_angle.to(u.rad)
        else:
            _observation_angle = Angle(observation_angle, unit=u.rad)
            warnings.warn("Casting observation_angle, input as " + str(observation_angle) + ", to radians",
                          AstropyUserWarning)

    if isinstance(max_limb_angle, Angle) or isinstance(max_limb_angle, u.Quantity):
        _max_limb_angle = max_limb_angle.to(u.rad)
    else:
        _max_limb_angle = Angle(max_limb_angle, unit=u.rad)
        warnings.warn("Casting max_limb_angle, input as " + str(max_limb_angle) + ", to radians",
                      AstropyUserWarning)

    cos_phi_squared = 1 - (np.sin(_observation_angle) / np.sin(_max_limb_angle)) ** 2
    try:
        return_val = a0 + a1*cos_phi_squared**.5 + a2*cos_phi_squared
    except FloatingPointError:
        print("returnVal: ", return_val)
        print("cosPhiSquared: ", cos_phi_squared)
        print("relativeAngle: ", _observation_angle)
        print("maxAngle: ", _max_limb_angle)
        return_val = 0
    return return_val


def _limb_darkened_phase_lw(observation_angle, max_limb_angle, a0=.3, a1=.93, a2=-.23, **kwargs):
    """
    Lightweight version, does not track units
    :param float observation_angle: Angle between observer-->star_origin and observer-->point_on_star
    :param float max_limb_angle:
    :param float a0: 0th order term
    :param float a1: 1st order term
    :param float a2: 2nd order term
    :return:
    """
    cos_phi_squared = 1 - (np.sin(observation_angle) / np.sin(max_limb_angle)) ** 2
    try:
        return_val = a0 + a1*cos_phi_squared**.5 + a2*cos_phi_squared
    except FloatingPointError:
        print("returnVal: ", return_val)
        print("cosPhiSquared: ", cos_phi_squared)
        print("relativeAngle: ", observation_angle)
        print("maxAngle: ", max_limb_angle)
        return_val = 0
    return return_val

