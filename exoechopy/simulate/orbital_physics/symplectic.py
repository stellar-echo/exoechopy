
"""
This module provides 6th order Yoshida symplectic integration of orbits for simulations and analysis.

References
----------
Yoshida, H. (1990). "Construction of higher order symplectic integrators". Phys. Lett. A. 150 (5–7): 262.
https://en.wikipedia.org/wiki/Symplectic_integrator

"""

import warnings
import numpy as np

from astropy import units as u
from astropy import constants as const
from astropy.utils.exceptions import AstropyUserWarning

from ...utils import *
from .yoshida_coeffs import *

__all__ = ['SymplecticSolver', 'accel_no_mass', 'accel_no_mass_array']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

G_const = const.G.to(lw_distance_unit**3/lw_mass_unit/lw_time_unit**2).value


class SymplecticSolver:
    """
    Base class for numerically computing many-body orbits, solved with 6th order Yoshida integration.
    """
    # ------------------------------------------------------------------------------------------------------------ #
    def __init__(self,
                 *args,
                 dt: u.Quantity = None):
        """
        Accepts a list of planets, stars, and other massive objects and a timestep

        Uses the position and velocity as defined in each object.
        Subtracts the center of momentum velocity before beginning computations.

        Must pass each object separately, even though some objects may be satellites of others.
        This is just to prevent mistaken duplicate planets, and could be changed in the future.

        When running calculations, stores position results in each object's ._all_positions field

        Parameters
        ----------
        dt
            Time step
        args
            List of orbital objects, such as planet or star classes
        """
        self._dt = None
        self._dt_lw = None
        self.update_timestep(dt)

        self._all_objects = args

        self._mass_list = []
        self._mass_list_lw = []
        for obj in self._all_objects:
            self._mass_list.append(obj.mass)
            self._mass_list_lw.append(obj.mass.to(lw_mass_unit).value)
            obj._time_domain = None
            obj._all_positions = None
            obj._all_velocities = None

        self._n_bodies = len(self._mass_list)

        self._reset_system_momentum()

        # Used in computations, initialize as good practice:
        self._pos_state = None
        self._vel_state = None
        self._accel_state = None

    # ------------------------------------------------------------------------------------------------------------ #
    def update_timestep(self, dt: u.Quantity):
        """Sets the timestep for the solver

        Parameters
        ----------
        dt
            New timestep
        """
        if isinstance(dt, u.Quantity):
            self._dt = dt.to(lw_time_unit)
        elif isinstance(dt, float):
            self._dt = u.Quantity(dt, u.s).to(lw_time_unit)
            warnings.warn("Casting SymplecticSolver timestep dt, input as " + str(dt) + ", to "+u_str(lw_time_unit),
                          AstropyUserWarning)
        self._dt_lw = self._dt.value

    # ------------------------------------------------------------------------------------------------------------ #
    def _reset_system_momentum(self):
        """Forces the system's total momentum to zero to prevent walk-off due to initial conditions

        """
        system_momentum = np.zeros(3)*lw_mass_unit*lw_distance_unit/lw_time_unit
        total_mass = u.Quantity(0, lw_mass_unit)
        for obj in self._all_objects:
            system_momentum += obj.mass * obj.velocity
            total_mass += obj.mass
        velocity_center_of_mass = system_momentum/total_mass
        for obj in self._all_objects:
            obj.velocity -= velocity_center_of_mass

    # ------------------------------------------------------------------------------------------------------------ #
    def calculate_orbits(self,
                         total_time: u.Quantity,
                         steps_per_save: int = 1):
        """Runs the solver over total_time (to within dt) and saves the position every steps_per_save timesteps

        Parameters
        ----------
        total_time
            Maximum time to simulate
        steps_per_save
            How many steps before saving the position.  Useful if timestep is very small, but only sample every so often.
            Default to every step

        """
        # Initialize parameters
        num_steps = int((total_time/self._dt).value)
        num_steps -= num_steps % steps_per_save

        num_saves = num_steps // steps_per_save

        pos_array = np.zeros((num_saves, self._n_bodies, 3))
        self._pos_state = np.zeros((self._n_bodies, 3))
        vel_array = np.zeros((num_saves, self._n_bodies, 3))
        self._vel_state = np.zeros((self._n_bodies, 3))
        self._accel_state = np.zeros((self._n_bodies, 3))

        # Generate the time domain, save into each object, and initialize the vectors
        time_domain = self._dt*steps_per_save*np.arange(num_saves)
        for o_i, obj in enumerate(self._all_objects):
            obj._time_domain = time_domain
            pos_array[0, o_i] = obj.position.to(lw_distance_unit)
            self._pos_state[o_i] = obj.position.to(lw_distance_unit)
            vel_array[0, o_i] = obj.velocity.to(lw_distance_unit/lw_time_unit)
            self._vel_state[o_i] = obj.velocity.to(lw_distance_unit/lw_time_unit)

        ct = 1
        for step in range(1, num_steps):
            # Do the actual integration:
            self._yoshida_6_integrate()
            # Do we save this step?
            if step % steps_per_save == 0:
                pos_array[ct] = self._pos_state
                vel_array[ct] = self._vel_state
                ct += 1

        for o_i, obj in enumerate(self._all_objects):
            obj._all_positions = u.Quantity(pos_array[:, o_i, :], lw_distance_unit).to(u.au)
            obj._all_velocities = u.Quantity(vel_array[:, o_i, :], lw_distance_unit/lw_time_unit)

    # ------------------------------------------------------------------------------------------------------------ #
    def _yoshida_6_integrate(self):
        """Run the 6th order symplectic integration using the Yoshida coefficients
        """
        for c_i, d_i in zip(c_vect, d_vect):
            self._update_accel_vect()
            self._vel_state += c_i * self._accel_state * self._dt_lw
            self._pos_state += d_i * self._vel_state * self._dt_lw

    def _update_accel_vect(self):
        accel_vect = np.zeros(self._accel_state.shape)
        for body_i in range(self._n_bodies):
            for body_j in range(body_i + 1, self._n_bodies):
                a_ij = accel_no_mass_lw(self._pos_state[body_i], self._pos_state[body_j])
                accel_vect[body_i] += a_ij * self._mass_list_lw[body_j]
                accel_vect[body_j] -= a_ij * self._mass_list_lw[body_i]
        self._accel_state = accel_vect.copy()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
def accel_no_mass_lw(pos_1_vect: np.ndarray,
                     pos_2_vect: np.ndarray) -> np.ndarray:
    """Acceleration between two bodies, ignoring mass, in lw units

    Parameters
    ----------
    pos_1_vect
        Body 1 position
    pos_2_vect
        Body 2 position

    Returns
    -------
    np.ndarray
        Mass-normalized acceleration--multiply by the mass to get acceleration
    """
    dv = pos_1_vect - pos_2_vect
    r3_2_inv = np.vdot(dv, dv)
    r3_2_inv **= -(3 / 2)
    return -dv * G_const * r3_2_inv


def accel_no_mass(pos_1_vect: u.Quantity,
                  pos_2_vect: u.Quantity) -> u.Quantity:
    """Acceleration between two bodies, ignoring mass

    Parameters
    ----------
    pos_1_vect
        Body 1 position
    pos_2_vect
        Body 2 position

    Returns
    -------
    u.Quantity
        Mass-normalized acceleration--multiply by the mass to get acceleration
    """
    dv = pos_1_vect - pos_2_vect
    r3_2_inv = np.vdot(dv, dv)
    r3_2_inv **= -(3 / 2)
    return -dv * const.G * r3_2_inv


def accel_no_mass_array(pos_1_vect_array: u.Quantity,
                        pos_2_vect_array: u.Quantity) -> u.Quantity:
    """Acceleration between two bodies, ignoring mass

    Parameters
    ----------
    pos_1_vect_array
        Array of body 1 positions
    pos_2_vect_array
        Array of body 2 positions

    Returns
    -------
    u.Quantity
        Mass-normalized acceleration array--multiply by the mass to get acceleration
    """
    dv = pos_1_vect_array - pos_2_vect_array
    r3_2_inv = np.sum(dv*dv, axis=1)
    r3_2_inv **= -(3 / 2)
    return -dv * const.G * r3_2_inv




