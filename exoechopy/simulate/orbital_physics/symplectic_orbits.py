
"""
This module provides 6th order Yoshida symplectic integration of orbits for simulations and analysis.

References
----------
Yoshida, H. (1990). "Construction of higher order symplectic integrators". Phys. Lett. A. 150 (5–7): 262.
https://en.wikipedia.org/wiki/Symplectic_integrator

"""

import warnings
import numpy as np
from scipy import optimize
from astropy import units as u
from astropy.coordinates import Angle
from astropy.coordinates import Distance
from astropy import constants as const
from astropy.utils.exceptions import AstropyUserWarning

from ...utils import *
from .yoshida_coeffs import *

__all__ = ['SymplecticSolver']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class SymplecticSolver:
    """
    Base class for numerically computing many-body orbits, solved with 6th order Yoshida integration.
    """
    # ------------------------------------------------------------------------------------------------------------ #
    def __init__(self,
                 dt: u.Quantity = None,
                 *args):
        """
        Accepts a list of planets and stars and a timestep

        Parameters
        ----------
        dt
            Time step
        args
            List of orbital objects, such as planet or star classes
        """
        if isinstance(dt, u.Quantity):
            self._dt = dt.to(lw_time_unit)
        elif isinstance(dt, float):
            self._dt = u.Quantity(dt, u.s).to(lw_time_unit)
            warnings.warn("Casting SymplecticSolver timestep dt, input as " + str(dt) + ", to "+u_str(lw_time_unit),
                          AstropyUserWarning)
        self._dt_lw = self._dt.value

        all_objects = args

        self._mass_list = []
        for obj in all_objects:
            self._mass_list.append(obj.mass)

    # ------------------------------------------------------------------------------------------------------------ #
    # TODO make this operate pseudo-statically
    def yoshida_6_integrate(self, all_pos, all_vel, all_mass, dt):
        new_pos = all_pos.copy()
        new_vel = all_vel.copy()

        for c_i, d_i in zip(c_vect, d_vect):
            accel_vect = get_accel_vect(new_pos, all_mass)
            new_vel += c_i * accel_vect * dt
            new_pos += d_i * new_vel * dt

        return new_pos, new_vel


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
def accel_no_mass(pos_1_vect, pos_2_vect):
    dv = pos_1_vect - pos_2_vect
    r3_2_inv = np.dot(dv, dv)
    r3_2_inv **= -(3 / 2)
    return -dv * G_const * r3_2_inv

def get_accel_vect(all_pos, all_mass):
    accel_vect = np.zeros(shape=all_pos.shape)
    for body_i in range(len(all_pos)):
        for body_j in range(body_i + 1, len(all_pos)):
            a_ij = accel_no_mass(all_pos[body_i], all_pos[body_j])
            accel_vect[body_i] += a_ij * all_mass[body_j]
            accel_vect[body_j] -= a_ij * all_mass[body_i]
    return accel_vect