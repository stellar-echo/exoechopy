
"""
This module provides models for stellar limb opacity and intensity.
"""

import warnings
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from astropy.utils.exceptions import AstropyUserWarning

from ..utils.constants import *
from .spectral.spectral_dicts import limb_dict


__all__ = ['Limb']


# TODO Implement spectral dependencies
# TODO Handle transit cases a little
# TODO Handle flare depth somehow
# TODO Handle flares slightly behind star, but still visible through limb?

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class Limb:
    def __init__(self,
                 limb_model: str='quadratic',
                 coeffs: list=[],
                 func: FunctionType=None):
        """A class for handling limb darkening functions

        Based on BATMAN:
        https://www.cfa.harvard.edu/~lkreidberg/batman/tutorial.html#limb-darkening-options

        Parameters
        ----------
        limb_model
            Name of the limb model.  Options:
                - uniform
                - linear
                - quadratic (default)
                - square-root
                - logarithmic
                - exponential
                - power2
                - nonlinear
                - custom (this requires inputing a custom func, coeffs will be passed as *args to func)
        coeffs
            Coefficients to pass to the limb model
        func
            A custom limb model provided by the user--ignored if limb_model is not custom!
        """
        try:
            self._limb_function = limb_dict[limb_model]
            self._limb_model = limb_model
            self._limb_args = coeffs
        except KeyError:
            if func is not None:
                self._limb_function = func
                self._limb_model = limb_model
                self._limb_args = coeffs
            else:
                raise AttributeError("Unknown limb model")

    #  =============================================================  #
    def calculate_limb_intensity(self,
                                 angle: u.Quantity,
                                 star_radius_over_distance: float = 0):
        """Calculates the relative intensity at a point on the surface of a star relative to an observation angle.

        Parameters
        ----------
        angle
            Angle between star_origin-->observer and star_origin-->point_on_star
        star_radius_over_distance
            Radius of star / Distance to star, defaults to 0 (meaning Earth is effectively at infinity)

        Returns
        -------
        float
            Value from [0, 1] for visibility of the point on the surface
        """

        if isinstance(angle, Angle) or isinstance(angle, u.Quantity):
            _angular_position = angle.to(u.rad)
        else:
            _angular_position = Angle(angle, unit=u.rad)
            warnings.warn("Casting angular_position, input as " + str(angle) + ", to radians",
                          AstropyUserWarning)

        mod_angle = np.mod(_angular_position.value, 2 * np.pi) * u.rad
        if pi_u / 2 <= mod_angle <= 3 * pi_u / 2:  # If it's beyond the limb from any distance...
            return 0

        max_limb_angle = np.arcsin(star_radius_over_distance) * u.rad
        # If viewed from Earth, max_limb_angle is effectively 0
        if max_limb_angle.value == 0:
            return self.limb_darkened_radial_position(np.sin(_angular_position))

        # If user requests an angle on the other side of the star, currently assume star is opaque and blocks light:
        max_angle = pi_u / 2 - max_limb_angle
        if np.abs(_angular_position) > max_angle:
            return 0.

        # If viewed within the star system at a realistic angle:
        else:
            theta_angle = np.arctan2(np.sin(_angular_position),
                                     1 / star_radius_over_distance - np.cos(_angular_position))
            theta_angle = np.mod(theta_angle, 2 * pi_u)
            # if max_limb_angle <= theta_angle <= 2*np.pi-max_limb_angle:
            #     return 0.
            # else:
            return self._limb_darkened_within_system_lw(theta_angle, max_limb_angle)

    #  =============================================================  #
    def _limb_darkened_within_system_lw(self, theta, max_limb_angle):
        """

        Parameters
        ----------
        theta
        max_limb_angle

        Returns
        -------

        """
        cos_phi = (1 - (np.sin(theta) / np.sin(max_limb_angle)) ** 2)**.5
        try:
            return_val = self._limb_function(cos_phi, *self._limb_args)
        except FloatingPointError:
            print("return_val: ", return_val)
            print("cos_phi: ", cos_phi)
            print("max_limb_angle: ", max_limb_angle)
            return_val = 0
        return return_val

    #  =============================================================  #
    def limb_darkened_radial_position(self, radial_position: float) -> float:
        """Relative intensity as a function of fractional radial position on the star.

        Parameters
        ----------
        radial_position
            Fraction of radius on star, in polar coordinates.  [0-1]

        Returns
        -------
        float
            Relative limb intensity
        """
        cos_phi = (1 - radial_position ** 2)**.5
        return self._limb_function(cos_phi, *self._limb_args)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

class SpectralLimb(Limb):
    def __init__(self,
                 limb_model: str,
                 coeff_funcs: list,
                 model_func: FunctionType=None):
        """Extension of Limb to allow each coefficient to be a function of wavelength

        Not currently implemented, this is a placeholder!

        Parameters
        ----------
        limb_model
        coeff_funcs
        model_func
        """
        raise NotImplementedError

