
"""
This module provides standard functions useful in computing radiometry.
"""

import warnings
import numpy as np
from astropy import units as u
from astropy.utils.exceptions import AstropyUserWarning
from astropy.coordinates import Angle
from astropy.coordinates import Distance

from ....utils.constants import *

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
    if not isinstance(angle, Angle):
        angle = Angle(angle, u.rad)
        warnings.warn("Casting phase angle, input as " + str(angle) + ", to radians", AstropyUserWarning)
    angle %= (2 * pi_u * u.rad)
    return (np.sin(angle) + (pi_u - angle) * np.cos(angle)) / pi_u


# --------------------------------------------------------------------- #


def echo_relative_magnitude(distance_to_star: Distance,
                            phase_angle: Angle,
                            geometric_albedo: float,
                            planet_radius: u.Quantity,
                            phase_law: FunctionType=lambertian_phase_law):
    """Approximate echo relative magnitude

    From 'Direct Imaging of Exoplanets' by Traub & Oppenheimer

    Parameters
    ----------
    distance_to_star
    phase_angle
        Angle relative to Earth, 0 = superior conjunction, Pi/2 = max elongation, Pi = inferior conjunction
    geometric_albedo
        In VIS, Earth is 0.367, Venus is 0.84, moon is 0.113, jupiter is 0.52
    planet_radius
    phase_law
        Which phase law to use

    Returns
    -------
    float
        Reflected-light contrast of an exoplanet

    """

    phase = phase_law(phase_angle)  # Fix to enable *flare_kwargs
    inverse_falloff = (planet_radius/distance_to_star)**2
    return geometric_albedo*phase*inverse_falloff

