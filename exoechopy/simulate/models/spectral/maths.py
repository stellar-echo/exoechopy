
"""
This module provides standard functions useful in computing radiometry.
"""

import warnings
import numpy as np
from astropy import units as u
from astropy.utils.exceptions import AstropyUserWarning
from astropy.coordinates import Angle
from astropy.coordinates import Distance
from exoechopy.utils.globals import *

__all__ = ['lambertian_phase_law', 'echo_relative_magnitude']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def lambertian_phase_law(self, angle):
    """
    Lambertian sphere phase law
    :param Angle angle: [0, 2 pi]
    :return float:
    """
    if not isinstance(angle, Angle):
        angle = Angle(angle, u.rad)
        warnings.warn("Casting phase angle, input as " + str(angle) + ", to radians", AstropyUserWarning)
    angle %= (2 * pi_u * u.rad)
    return (np.sin(angle) + (pi_u - angle) * np.cos(angle)) / pi_u


# --------------------------------------------------------------------- #


def echo_relative_magnitude(distance_to_star,
                            phase_angle,
                            geometric_albedo,
                            planet_radius,
                            phase_law=lambertian_phase_law):
    """
    From Direct Imaging of Exoplanets by Traub & Oppenheimer
    :param Distance distance_to_star:
    :param Angle phase_angle: angle relative to Earth, 0 = superior conjunction, Pi/2 = max elongation, Pi = inferior conjunct
    :param float geometric_albedo: In VIS, Earth is 0.367, Venus is 0.84, moon is 0.113, jupiter is 0.52
    :param u.Quantity planet_radius:
    :param function phase_law: Which phase law to use
    :return float: reflected-light contrast of an exoplanet
    """
    phase = phase_law(phase_angle)  # Fix to enable *args
    inverse_falloff = (planet_radius/distance_to_star)**2
    return geometric_albedo*phase*inverse_falloff

