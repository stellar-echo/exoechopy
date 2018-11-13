
"""
This module provides standard functions useful in computing radiometry.
"""

import warnings
import numpy as np
from astropy import units as u
from astropy.utils.exceptions import AstropyUserWarning
from astropy.coordinates import Angle
from astropy.coordinates import Distance

from ...utils.constants import *
from .simple import Albedo

__all__ = ['lambertian_phase_law', 'echo_relative_magnitude']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def lambertian_phase_law(angle: Angle):
    """Lambertian sphere phase law

    Parameters
    ----------
    angle
        [0, 2 pi]

    Returns
    -------
    u.Quantity

    """
    if not isinstance(angle, (Angle, u.Quantity)):
        angle = Angle(angle, u.rad)
        warnings.warn("Casting phase angle, input as " + str(angle) + ", to radians", AstropyUserWarning)
    angle %= (2 * pi_u)
    return (np.sin(angle)*u.rad + (pi_u - angle) * np.cos(angle)) / pi_u


# --------------------------------------------------------------------- #

def echo_relative_magnitude(distance_to_star: Distance,
                            phase_angle: Angle,
                            geometric_albedo: (float, Albedo),
                            planet_radius: u.Quantity,
                            *phase_law_args):
    """Approximate echo relative magnitude

    From 'Direct Imaging of Exoplanets' by Traub & Oppenheimer

    Parameters
    ----------
    distance_to_star
        Separation between the source (like the flare) and the planet
    phase_angle
        Angle relative to Earth, 0 = superior conjunction, Pi/2 = max elongation, Pi = inferior conjunction
    geometric_albedo
        In VIS, Earth is 0.367, Venus is 0.84, moon is 0.113, jupiter is 0.52
    planet_radius
    phase_law_args
        Pass additional variables to the phase_law, if relevant

    Returns
    -------
    float
        Reflected-light contrast of an exoplanet

    """
    if isinstance(geometric_albedo, (int, float)):
        alb = geometric_albedo
    else:
        alb = geometric_albedo.calculate_albedo_from_phase_law(phase_angle, *phase_law_args)
    inverse_falloff = (planet_radius/distance_to_star)**2
    return alb*inverse_falloff

