
"""
This module provides models for stellar limb opacity and intensity.
"""

import warnings
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from astropy.utils.exceptions import AstropyUserWarning


__all__ = ['_calculate_basic_limb_darkening_lw']

# TODO Reconcile angle vs radial position limb darkening models !!

# TODO Implement spectral dependencies
# TODO Create a Limb class?  Could handle spectral stuff and other models more easily?
# TODO Handle transit cases a little
# TODO Handle flare depth somehow
# TODO Handle flares behind star, but still visible through limb

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def _calculate_basic_limb_darkening_lw(relative_angle, max_limb_angle=0, dist_rad_ratio=10.,
                                       **kwargs):
    """
    Lightweight version, does not track or verify units
    Calculates the effective intensity of a point on a surface relative to an observer position.

    :param relative_angle: Angle drawn between star origin-->observer and star origin-->point on the star
    :param max_limb_angle: Angle between observer and very edge (limb) of the star, 0 from infinity
    :param dist_rad_ratio: Distance to star / radius of star
    :return:
    """
    mod_angle = np.mod(relative_angle, 2 * np.pi)
    if np.pi / 2 <= mod_angle <= 3 * np.pi / 2:  # If it's beyond the limb from any distance...
        return 0
    # If viewed from Earth, max_limb_angle is effectively 0
    if max_limb_angle == 0:
        return limb_darkened_radial_position(np.sin(relative_angle), **kwargs)
    # If viewed within the star system, max_limb_angle depends on actual geometry of orbit
    else:
        theta_angle = np.arctan2(np.sin(relative_angle), dist_rad_ratio - np.cos(relative_angle))
        theta_angle %= 2 * np.pi
    if max_limb_angle <= theta_angle <= 2*np.pi-max_limb_angle:
        return 0.
    else:
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
    x = np.sqrt(1 - radial_position ** 2)
    return a0 + a1*x**.5 + a2*x


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def _limb_darkened_phase_lw(relative_angle, max_limb_angle, a0=.3, a1=.93, a2=-.23, **kwargs):
    """
    Lightweight version, does not track units
    :param float relative_angle:
    :param float max_limb_angle:
    :param float a0: 0th order term
    :param float a1: 1st order term
    :param float a2: 2nd order term
    :return:
    """
    cos_phi_squared = 1 - (np.sin(relative_angle) / np.sin(max_limb_angle)) ** 2
    try:
        return_val = a0 + a1*cos_phi_squared**.5 + a2*cos_phi_squared
    except FloatingPointError:
        print("returnVal: ", return_val)
        print("cosPhiSquared: ", cos_phi_squared)
        print("relativeAngle: ", relative_angle)
        print("maxAngle: ", max_limb_angle)
        return_val = 0
    return return_val


def limb_darkened_phase(relative_angle, max_limb_angle, a0=.3, a1=.93, a2=-.23, **kwargs):
    """
    Computes the intensity of the star based on
    :param Angle relative_angle:
    :param Angle max_limb_angle:
    :param float a0: 0th order term
    :param float a1: 1st order term
    :param float a2: 2nd order term
    :return float:
    """
    if relative_angle == 0.:
        _relative_angle = Angle(0., unit=u.rad)
    else:
        if isinstance(relative_angle, Angle) or isinstance(relative_angle, u.Quantity):
            _relative_angle = relative_angle.to(u.rad)
        else:
            _relative_angle = Angle(relative_angle, unit=u.rad)
            warnings.warn("Casting relative_angle, input as " + str(relative_angle) + ", to radians",
                          AstropyUserWarning)

    if isinstance(max_limb_angle, Angle) or isinstance(max_limb_angle, u.Quantity):
        _max_limb_angle = max_limb_angle.to(u.rad)
    else:
        _max_limb_angle = Angle(max_limb_angle, unit=u.rad)
        warnings.warn("Casting max_limb_angle, input as " + str(max_limb_angle) + ", to radians",
                      AstropyUserWarning)

    cos_phi_squared = 1 - (np.sin(_relative_angle) / np.sin(_max_limb_angle)) ** 2
    try:
        return_val = a0 + a1*cos_phi_squared**.5 + a2*cos_phi_squared
    except FloatingPointError:
        print("returnVal: ", return_val)
        print("cosPhiSquared: ", cos_phi_squared)
        print("relativeAngle: ", _relative_angle)
        print("maxAngle: ", _max_limb_angle)
        return_val = 0
    return return_val


##############################################  TEST & DEMO CODE  ######################################################


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # radial_value_array = np.linspace(-1, 1, 1000)
    # angle_from_radius_array = np.arcsin(radial_value_array)
    angle_array = np.linspace(-np.pi, np.pi, 1000)
    sun_radius = (1*u.R_sun).to(u.au)

    #  ---------------------------------------------------------  #
    infinity_limb = [_calculate_basic_limb_darkening_lw(relative_angle=ai,
                                                        max_limb_angle=0)
                     for ai in angle_array]
    ax1.plot(angle_array, infinity_limb, color='k', lw=1, label="From infinity")

    #  ^^^^^^^^^^^^^^^^^^^^^^  #
    parker_probe_min_distance = 0.041*u.au
    max_limb_angle_pp = np.arcsin(sun_radius.value/parker_probe_min_distance.value)
    parker_limb = [_calculate_basic_limb_darkening_lw(relative_angle=ai,
                                                      max_limb_angle=max_limb_angle_pp,
                                                      dist_rad_ratio=(parker_probe_min_distance/sun_radius).value)
                   for ai in angle_array]
    ax1.plot(angle_array, parker_limb, color='r', lw=1, label="From Parker probe min dist")

    #  ^^^^^^^^^^^^^^^^^^^^^^  #
    mercury_distance = 0.39*u.au
    max_limb_angle_Hg = np.arcsin(sun_radius.value/mercury_distance.value)
    mercury_limb = [_calculate_basic_limb_darkening_lw(relative_angle=ai,
                                                       max_limb_angle=max_limb_angle_Hg,
                                                       dist_rad_ratio=(mercury_distance/sun_radius).value)
                    for ai in angle_array]
    ax1.plot(angle_array, mercury_limb, color='darkred', lw=1, label="From Mercury")

    #  ^^^^^^^^^^^^^^^^^^^^^^  #
    earth_distance = 1*u.au
    max_limb_angle_Earth = np.arcsin(sun_radius.value/earth_distance.value)
    earth_limb = [_calculate_basic_limb_darkening_lw(relative_angle=ai,
                                                     max_limb_angle=max_limb_angle_Earth,
                                                     dist_rad_ratio=(earth_distance/sun_radius).value)
                  for ai in angle_array]
    ax1.plot(angle_array, earth_limb, color='b', lw=1, label="From Earth")

    ax1.set_xlabel("Angular orientation")
    ax1.set_ylabel("Relative star intensity")
    ax1.set_title("Solar intensity approximation")
    ax1.legend()

    #  ---------------------------------------------------------  #

    ax2.plot()
    ax2.set_xlabel("Radial position (r/R)")
    ax2.set_ylabel("Relative star intensity")
    ax2.set_title("Solar intensity approximation, viewed from Mercury")

    #  ---------------------------------------------------------  #
    plt.tight_layout()
    plt.show()

    # plt.plot(radial_value_array, angle_from_radius_array)
    # plt.show()


