
"""
This module provides models for stellar limb opacity and intensity.
"""

import warnings
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from astropy.utils.exceptions import AstropyUserWarning
from exoechopy.utils.globals import *


__all__ = ['calculate_basic_limb_darkening', '_calculate_basic_limb_darkening_lw', 'limb_darkened_radial_position']

# TODO Implement spectral dependencies
# TODO Create a Limb class?  Could handle spectral stuff and other models more easily?
# TODO Handle transit cases a little
# TODO Handle flare depth somehow
# TODO Handle flares slightly behind star, but still visible through limb

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def calculate_basic_limb_darkening(angular_position, star_radius_over_distance=0, **kwargs):
    """
    Calculates the effective intensity of a point on a surface relative to an observer position.

    :param angular_position: Angle between star_origin-->observer and star_origin-->point_on_star
    :param star_radius_over_distance: radius of star / Distance to star
    :return:
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


##############################################  TEST & DEMO CODE  ######################################################


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2)

    angle_array = np.linspace(-np.pi, np.pi, 1000)*u.rad
    sun_radius = (1*u.R_sun).to(u.au)

    ax1.set_xlabel("Angle on star surface")
    ax1.set_ylabel("Relative star intensity")
    ax1.set_title("Solar intensity approx. by angle", y=1.5)

    #  ---------------------------------------------------------  #
    infinity_limb = [calculate_basic_limb_darkening(angular_position=ai,
                                                    max_limb_angle=0)
                     for ai in angle_array]
    ax1.plot(angle_array, infinity_limb, color='k', lw=1, label="Infinity")

    #  ^^^^^^^^^^^^^^^^^^^^^^  #
    parker_probe_min_distance = 0.041*u.au
    parker_limb = [calculate_basic_limb_darkening(angular_position=ai,
                                                  star_radius_over_distance=(sun_radius/parker_probe_min_distance).value)
                   for ai in angle_array]
    ax1.plot(angle_array, parker_limb, color='r', lw=1, label="Parker probe min dist")

    #  ^^^^^^^^^^^^^^^^^^^^^^  #
    mercury_distance = 0.39*u.au
    mercury_limb = [calculate_basic_limb_darkening(angular_position=ai,
                                                   star_radius_over_distance=(sun_radius/mercury_distance).value)
                    for ai in angle_array]
    ax1.plot(angle_array, mercury_limb, color='darkred', lw=1, label="Mercury")

    #  ^^^^^^^^^^^^^^^^^^^^^^  #
    earth_distance = 1*u.au
    earth_limb = [calculate_basic_limb_darkening(angular_position=ai,
                                                 star_radius_over_distance=(sun_radius/earth_distance).value)
                  for ai in angle_array]
    ax1.plot(angle_array, earth_limb, color='b', lw=1, label="Earth")

    #  ---------------------------------------------------------  #

    radial_value_array = np.linspace(-1, 1, 1000)
    ax2.plot(radial_value_array, limb_darkened_radial_position(radial_value_array), color='k')
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Radial position (r/R)")
    ax2.set_ylabel("Relative star intensity")
    ax2.set_title("Relative star intensity by radial position", y=1.1)

    #  ---------------------------------------------------------  #

    ax1.set_aspect(2*np.pi)
    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0.)

    ax2.set_aspect(2)

    plt.tight_layout()
    plt.show()



