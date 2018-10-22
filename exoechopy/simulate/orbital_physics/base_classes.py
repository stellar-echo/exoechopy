
"""
This module provides base classes for orbital physics objects

"""

import warnings

import numpy as np
from scipy import interpolate
from astropy import units as u
from astropy import constants as const
from astropy.utils.exceptions import AstropyUserWarning

from ...utils import *

__all__ = ['MassiveObject', 'MultiStarSystem']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class MassiveObject:
    """Object with mass, position, velocity, satellites, an optional list of positions and times"""

    # ------------------------------------------------------------------------------------------------------------ #
    def __init__(self,
                 mass: u.Quantity = None,
                 position: u.Quantity = None,
                 velocity: u.Quantity = None,
                 **kwargs
                 ):
        """Base instance for orbital objects with mass, positions, velocities, and satellites

        Parameters
        ----------
        mass
            Mass of object
        position
            Initial position of object
        velocity
            Initial velocity of object
        kwargs
        """
        super().__init__(**kwargs)

        self._position = None
        self.position = position

        # These values are reserved for when the positions are calculated by an external solver
        self._time_domain = None
        self._all_positions = None  # Stored as u.Quantity, should be in u.au typically
        self._all_velocities = None  # Stored as u.Quantity, should be in m/s typically
        self._position_interpolator = None
        self._velocity_interpolator = None

        self._velocity = None
        self.velocity = velocity

        self._orbiting_bodies_list = []

        #  Initialize mass, then pass to the set function:
        self._mass = None
        if mass is not None:
            self.mass = mass

        self._parent_mass = None
        self._grav_param = None
        self._orbital_period = None
        self._orbital_frequency = None

    # ------------------------------------------------------------------------------------------------------------ #
    def get_position_at_time(self, time: u.Quantity) -> u.Quantity:
        """Returns the 3d position at the requested time

        Performs a quadratic interpolation between points using scipy.interpolate.interp1d

        Parameters
        ----------
        time
            Time to evaluate the position at

        Returns
        -------
        u.Quantity
            The position at the requested time
        """
        time = time.to(self._time_domain.unit)
        if self._position_interpolator is None:
            self._position_interpolator = interpolate.interp1d(self._time_domain, self._all_positions,
                                                               axis=0, kind='quadratic', copy=False, assume_sorted=True)
        return u.Quantity(self._position_interpolator(time), self._all_positions.unit)

    def get_position_at_time_lw(self, time: float) -> np.ndarray:
        """Returns the 3d position at the requested time without units

        Performs a quadratic interpolation between points using scipy.interpolate.interp1d

        Parameters
        ----------
        time
            Time to evaluate the position at

        Returns
        -------
        np.ndarray
            The position at the requested time
        """
        if self._position_interpolator is None:
            self._position_interpolator = interpolate.interp1d(self._time_domain, self._all_positions,
                                                               axis=0, kind='quadratic', copy=False, assume_sorted=True)
        return self._position_interpolator(time)

    # ------------------------------------------------------------------------------------------------------------ #
    def get_velocity_at_time(self, time: u.Quantity) -> u.Quantity:
        """Returns the 3d velocity at the requested time

        Performs a quadratic interpolation between points using scipy.interpolate.interp1d

        Parameters
        ----------
        time
            Time to evaluate the position at

        Returns
        -------
        u.Quantity
            The position at the requested time
        """
        time = time.to(self._time_domain.unit)
        if self._velocity_interpolator is None:
            self._velocity_interpolator = interpolate.interp1d(self._time_domain, self._all_velocities,
                                                               axis=0, kind='quadratic', copy=False, assume_sorted=True)
        return u.Quantity(self._velocity_interpolator(time), self._all_velocities.unit)

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def mass(self):
        """Mass of object, used for simulations

        Returns
        -------
        u.Quantity
            Mass of object
        """
        return self._mass

    @mass.setter
    def mass(self, mass):
        """Update the mass and the associated orbiting body's orbits, if relevant

        Parameters
        ----------
        mass
            New mass of object
        """
        if mass is None:
            self._mass = None
        else:
            if isinstance(mass, u.Quantity):
                self._mass = mass.to(u.solMass)
            else:
                self._mass = u.Quantity(mass, unit=u.solMass)
                warnings.warn("Casting mass, input as " + str(mass) + ", to solar masses", AstropyUserWarning)

        for orbiting_body in self._orbiting_bodies_list:
            orbiting_body.set_star_mass(self._mass)

    @property
    def system_mass(self):
        """Mass of object and all child objects, such as star + planets, or planets + moons, etc.

        Calculates recursively.

        Returns
        -------
        u.Quantity
            Mass of all objects in system
        """
        mass = self.mass
        for orbiting_body in self._orbiting_bodies_list:
            _mass = orbiting_body.system_mass
            if _mass is not None:
                mass += mass
        return mass

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        if position is None:
            self._position = u.Quantity(np.zeros(3), u.au)
        elif isinstance(position, u.Quantity):
            self._position = position.copy()
        else:
            self._position = u.Quantity(position, u.au)
            warnings.warn("Casting position, input as " + str(position) + ", to "+u_str(u.au),
                          AstropyUserWarning)

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        if velocity is None:
            self._velocity = u.Quantity(np.zeros(3), lw_distance_unit/lw_time_unit)
        elif isinstance(velocity, u.Quantity):
            self._velocity = velocity.copy()
        else:
            self._velocity = u.Quantity(velocity, lw_distance_unit/lw_time_unit)
            warnings.warn("Casting Planet velocity, input as " + str(velocity) + ", to "
                          + u_str(lw_distance_unit/lw_time_unit), AstropyUserWarning)

    # ------------------------------------------------------------------------------------------------------------ #
    def add_orbiting_object(self, new_object: 'MassiveObject'):
        """Add an orbiting object to the body, e.g., add an exoplanet to a star

        Parameters
        ----------
        new_object
            MassiveObject object to be added to the parent MassiveObject
        """
        if self._mass is None:
            warnings.warn("Adding an orbital body to a Star that has no mass.", AstropyUserWarning)
        if isinstance(new_object, MassiveObject):
            new_object.parent_mass = self._mass
            self._orbiting_bodies_list.append(new_object)
        else:
            raise TypeError("Currently, orbiting objects must be MassiveObject class instance")

    # ------------------------------------------------------------------------------------------------------------ #
    def get_all_orbiting_objects(self) -> ['MassiveObject']:
        """Return a *copy* of the list of exoplanets held by the Star

        Returns
        -------
        [*MassiveObject]
            List of MassiveObject instances
        """
        return self._orbiting_bodies_list.copy()

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def parent_mass(self):
        """Mass of the object that this object orbits.

        Useful for updating Keplerian orbits

        Returns
        -------
        u.Quantity
            Mass of the object this object orbits
        """
        return self._parent_mass

    @parent_mass.setter
    def parent_mass(self, parent_mass: u.Quantity):
        """Updates this object's copy of the parent's mass

        Parameters
        ----------
        parent_mass
            Mass to use as this object's parent's mass
        """
        if parent_mass is None:
            self._parent_mass = None
        else:
            if isinstance(parent_mass, u.Quantity):
                self._parent_mass = parent_mass
            else:
                self._parent_mass = u.Quantity(parent_mass, u.M_sun)
                warnings.warn("Casting star mass, input as " + str(parent_mass) + ", to M_sun", AstropyUserWarning)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class MultiStarSystem(MassiveObject):
    """Holder of multiple stars, currently a placeholder for future work"""
    # ------------------------------------------------------------------------------------------------------------ #
    def __init__(self, **kwargs):
        super().__init__(mass=0*u.M_sun, **kwargs)
        self.radius = -1*u.au


