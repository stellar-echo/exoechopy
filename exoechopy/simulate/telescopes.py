
"""
This module provides telescopes for acquiring data.
"""

import warnings

import numpy as np
from pathlib import Path
from astropy import units as u
import astropy.constants as const
from astropy.utils.exceptions import AstropyUserWarning

from .stars import *
from .flares.active_regions import *
from ..utils import *
from .orbital_physics import BaseSolver

__all__ = ['Telescope']

# TODO Handle cases with externally computed orbital positions for planets and stars

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class Telescope:
    """Basic telescope object for acquiring data from a star system."""

    # ------------------------------------------------------------------------------------------------------------ #
    def __init__(self,
                 collection_area: u.Quantity=None,
                 efficiency: float=None,
                 cadence: u.Quantity=None,
                 observation_target: DeltaStar=None,
                 random_seed: int=None,
                 name: str=None):
        """Defines a simple telescope with a variety of properties.

        Can be given a star system for observation to generate a synthetic light curve.


        Parameters
        ----------
        collection_area
            Total collection area of telescope.  Subtract out an obstructions to collection area.
        efficiency
            Total telescope efficiency, including detectors
        cadence
            Data is integrated and binned at this cadence
        observation_target
            Star system for observation
        random_seed
            Optional random seed for this telescope to make results precisely reproducible
        name
            Optional name for the telescope
        """

        self._area = None
        if collection_area is not None:
            self.collection_area = collection_area

        self._efficiency = efficiency

        self._cadence = None
        if cadence is not None:
            self.cadence = cadence

        self._observation_target = None
        if observation_target is not None:
            self.observation_target = observation_target

        self._random_seed = None
        if random_seed is not None:
            self.random_seed = random_seed

        if name is None:
            self._name = "Telescope"
        else:
            self._name = name

        self._observation_time = None
        self._observation_total_bins = None
        self._time_domain = None  # np.array of the start-times of each time bin
        self._pure_lightcurve = None  # np.array of the values in the lightcurve, before adding noise
        self._background_lightcurve = None  # np.array of the slowly-varying background
        self._noisy_lightcurve = None  # np.array of the values in the lightcurve after adding noise
        self._all_flares = FlareCollection({})  # FlareCollection object, regenerated by star for each observing run

        # Variables used when save_diagnostic_data is True in collect_data() call:
        self._all_earth_flare_angles = None  # np.array of earth-flare angles, for diagnostics
        self._earth_flare_visibility = None  # np.array of the earth-flare visibility, based on the limb function
        self._all_exoplanet_flare_angles = None  # np.array of exoplanet-flare visibility, based on the limb function
        self._all_exoplanet_contrasts = None  # np.array of exoplanet phase function values at flare times
        self._all_exoplanet_lags = None

        self._quiescent_intensity_sec = None
        self._quiescent_intensity_sample = None
        if self._update_flux_ready():
            self._update_flux()

        # Used to notify a few functions to look for pre-computed data, instead of using Keplerian approximation
        self.SOLVER_FLAG = False

    # ------------------------------------------------------------------------------------------------------------ #
    def _prep_observations(self,
                           cadence=None,
                           observation_time=None,
                           solver=None):

        self.observation_time = observation_time

        if solver is not None and observation_time is not None:
            solver.calculate_orbits(observation_time)
            self.SOLVER_FLAG = True

        if cadence is not None:
            self.cadence = cadence
        if self.cadence is None:
            raise ValueError("A cadence is required to run an observation.")
        if self.observation_target.system_mass is None:
            warnings.warn("observation_target does not have mass, some parameters may not be initialized",
                          AstropyUserWarning)
        self._update_flux()

    def prepare_continuous_observational_run(self,
                                             observation_time: u.Quantity,
                                             cadence: u.Quantity=None,
                                             print_report: bool=False,
                                             solver: BaseSolver=None):
        """Prepares to generate a synthetic light curve over a pre-determined period of time.

        Parameters
        ----------
        observation_time
            Total observation time to simulate
        cadence
            Observation cadence (will overwrite initialized cadence!)
        print_report
            Whether or not to print() some info about the observation
        solver
            Optional numerical solver for computing orbits
        """
        self._prep_observations(cadence=cadence, observation_time=observation_time, solver=solver)

        #  Determine observational parameters:
        self._time_domain = np.arange(0., self.observation_time.to(lw_time_unit).value,
                                      step=self.cadence.to(lw_time_unit).value)*lw_time_unit
        self._observation_total_bins = len(self._time_domain)

        #  Initialize the lightcurves:
        self._pure_lightcurve = np.ones(self._observation_total_bins) * self._quiescent_intensity_sample
        self._background_lightcurve = np.zeros(self._observation_total_bins)
        self._noisy_lightcurve = np.zeros(self._observation_total_bins)

        #  Generate flares
        self._all_flares = self._observation_target.generate_flares_over_time(self._observation_time)
        if self._all_flares.num_flares == 0:
            warnings.warn("No flares generated!", AstropyUserWarning)

        if print_report:
            self.observation_report()

    def prepare_observe_n_flares(self,
                                 num_flares: int,
                                 duration: u.Quantity=None,
                                 cadence: u.Quantity=None,
                                 solver: BaseSolver=None):
        """Prepares to generate N observable flares.

        If a duration is provided, will include background variability associated with the star's activity.
        Flares are distributed uniformly across the duration (for now)

        Parameters
        ----------
        num_flares
            How many flares to observe
        duration
            Optional duration, tunes background variability to realistically match
        cadence
            Observation cadence (will overwrite initialized cadence!)
        solver
            Optional numerical solver for computing orbits
        """
        #  Generate flares
        if duration is None:
            exo_list = self._observation_target.get_all_orbiting_objects()
            if len(exo_list) > 0:
                duration = exo_list[0].orbital_period
            else:
                try:
                    duration = 1/self._observation_target.rotation_rate
                except AttributeError:
                    duration = num_flares*u.Quantity(60, u.s)

        self._prep_observations(cadence=cadence, observation_time=duration, solver=solver)

        self._all_flares = self._observation_target.generate_n_flares_at_times(np.linspace(0,
                                                                               duration.to(lw_time_unit).value,
                                                                               num_flares))
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------ #
    def collect_data(self,
                     output_folder: Path=None,
                     base_filename: str=None,
                     save_diagnostic_data: bool=False):
        """Generate a synthetic light curve

        Parameters
        ----------
        output_folder
            Optional location to save the data
        base_filename
            Filename for the dataset
        save_diagnostic_data
            Whether or not to store additional data about the lightcurve
        """

        # TODO Handle multi-star systems

        flares = self._all_flares.all_flares
        flare_times = self._all_flares.all_flare_times
        flare_positions = self._all_flares.all_flare_vectors
        if flare_positions is None:
            flare_positions = [None for f in flares]
        flare_intensities = self._all_flares.all_flare_intensities
        num_flares = len(flares)

        # Shorten a few commonly used attributes:
        target = self.observation_target
        exo_list = self.observation_target.get_all_orbiting_objects()
        earth_vect = self.observation_target.earth_direction_vector
        dt = self.cadence

        if save_diagnostic_data:
            # A few lists of things that are computed, useful for diagnostics:
            all_earth_flare_angles = np.zeros(num_flares)*u.rad
            earth_flare_visibility = np.zeros(num_flares)
            all_exoplanet_flare_angles = np.zeros((num_flares, len(exo_list)))*u.rad
            all_exoplanet_contrasts = np.zeros((num_flares, len(exo_list)))
            all_exoplanet_lags = np.zeros((num_flares, len(exo_list)))

        for f_i, (flare, flare_time, flare_vect, flare_mag) in enumerate(zip(flares,
                                                                             flare_times,
                                                                             flare_positions,
                                                                             flare_intensities)):
            flare_duration = flare.flare_duration
            num_plot_points = max(int(flare_duration/dt), 1)
            # Determine start/stop indices for flare, pad by an index on either side:
            i0 = int(flare_time / dt)-1
            i1 = i0 + num_plot_points + 2
            # Handle edges of lightcurve:
            i0 = max(i0, 0)
            i1 = min(i1, len(self._time_domain))

            local_flare_times = self._time_domain[i0: i1]

            #  =============================================================  #
            #  Determine what is observed at Earth from the flares:
            if flare_vect is not None:
                earth_flare_angle = u.Quantity(angle_between_vectors(earth_vect, flare_vect.value), u.rad)
            else:
                earth_flare_angle = 0
            earth_flare_limb = target.star_limb(earth_flare_angle)

            if save_diagnostic_data:
                all_earth_flare_angles[f_i] = earth_flare_angle
                earth_flare_visibility[f_i] = earth_flare_limb

            # If flare is on the other side of star, no need to do more math.  Otherwise:
            if earth_flare_limb > 0:
                #  flare.evaluate_over_array_lw provides a scale factor without units,
                #  so use this version since flare_mag also has units.
                flare_mag = flare_mag.to(u.ph/u.s/u.m**2)
                flare_signal = flare.evaluate_over_array_lw(local_flare_times - flare_time) * flare_mag * dt
                flare_signal *= earth_flare_limb
                self._pure_lightcurve[i0: i1] += flare_signal*self.collection_area

            #  =============================================================  #
            #  Calculate echoes from exoplanets and what is then observed at Earth:
            for e_i, exoplanet in enumerate(exo_list):
                # Find out where the exoplanet was at the time of the flare
                if not self.SOLVER_FLAG:
                    exo_vect = exoplanet.calc_xyz_at_time(flare_time)
                    exo_vel = exoplanet.calc_vel_at_time(flare_time)
                else:
                    obs_vect = self.observation_target.get_position_at_time(flare_time)
                    obs_vel = self.observation_target.get_velocity_at_time(flare_time)
                    exo_vect = exoplanet.get_position_at_time(flare_time) - obs_vect
                    exo_vel = exoplanet.get_velocity_at_time(flare_time) - obs_vel

                if flare_vect is not None:
                    # Just for completeness, move the planet slightly based on the light travel time from the star:
                    exo_vect += exo_vel * u.Quantity(np.linalg.norm(exo_vect - flare_vect),
                                                     exo_vect.unit) / const.c
                    # Determine the visibility of the flare from the exoplanet:
                    flare_exo_angle = u.Quantity(angle_between_vectors(exo_vect.value, flare_vect.value), u.rad)
                    exo_flare_limb = target.star_limb(flare_exo_angle)
                    if save_diagnostic_data:
                        all_exoplanet_flare_angles[f_i, e_i] = flare_exo_angle
                else:
                    flare_vect = u.Quantity(np.zeros(3), exo_vect.unit)
                    # Just for completeness, move the planet slightly based on the light travel time from the star:
                    exo_vect += exo_vel * u.Quantity(np.linalg.norm(exo_vect - flare_vect), exo_vect.unit) / const.c
                    exo_flare_limb = 1

                # If flare is on the other side of star from planet, no need to do more math.  Otherwise:
                if exo_flare_limb > 0:
                    # Get the lag:
                    echo_lag, ev_norm = compute_lag(start_vect=flare_vect,
                                                    echo_vect=exo_vect,
                                                    detect_vect=earth_vect,
                                                    return_v2_norm=True)

                    # Slightly speed up calculations by reducing number of norms:
                    exo_flare_vect = exo_vect-flare_vect
                    earth_exo_phase_angle = u.Quantity(np.arccos(np.dot(earth_vect, exo_flare_vect/ev_norm)), 'rad')
                    exoplanet_earth_visibility = exoplanet.get_echo_magnitude(dist_to_source=ev_norm,
                                                                              earth_angle=earth_exo_phase_angle)
                    if save_diagnostic_data:
                        all_exoplanet_contrasts[f_i, e_i] = exoplanet_earth_visibility
                        all_exoplanet_lags[f_i, e_i] = echo_lag.to(u.s).value

                    flare_echo_time = flare_time + echo_lag
                    # Determine start/stop indices for echo, pad by an index on either side:
                    i2 = int(flare_echo_time / dt) - 1
                    i3 = i2 + num_plot_points + 2
                    # Handle edges of lightcurve:
                    i2 = min(max(i2, 0), len(self._time_domain)-2)
                    i3 = min(i3, len(self._time_domain))

                    # Note, this is not a perfect reproduction of the flare:
                    # It may be digitized into slightly different bins during telescope readout!
                    # So we explicitly re-calculate the flare at the time of the echo lag for maximum realism
                    local_flare_times = self._time_domain[i2: i3]
                    echo_signal = flare.evaluate_over_array_lw(local_flare_times - flare_echo_time) * flare_mag * dt

                    # Reduce echo strength by the flare visibility and the Earth visibility:
                    echo_signal *= exo_flare_limb*exoplanet_earth_visibility
                    # Calculate influence on the lightcurve:
                    self._pure_lightcurve[i2: i3] += echo_signal*self.collection_area

        # Store some useful diagnostic data in the Telescope.
        # May be better to replace with a dictionary to be more flexible?
        if save_diagnostic_data:
            self._all_earth_flare_angles = all_earth_flare_angles
            self._earth_flare_visibility = earth_flare_visibility
            self._all_exoplanet_flare_angles = all_exoplanet_flare_angles
            self._all_exoplanet_contrasts = all_exoplanet_contrasts
            self._all_exoplanet_lags = all_exoplanet_lags

        #  Save files, if a Path is given:
        if output_folder is not None:
            output_folder = Path(output_folder)
            if save_diagnostic_data:
                np.save(output_folder / (base_filename + " all_earth_flare_angles.npy"), all_earth_flare_angles)
                np.save(output_folder / (base_filename + " earth_flare_visibility.npy"), earth_flare_visibility)
                np.save(output_folder / (base_filename + " all_exoplanet_flare_angles.npy"), all_exoplanet_flare_angles)
                np.save(output_folder / (base_filename + " all_exoplanet_contrasts.npy"), all_exoplanet_contrasts)
                np.save(output_folder / (base_filename + " all_exoplanet_lags.npy"), all_exoplanet_lags)
            # To add: Echo lag, echo magnitude, flare location, etc.
            np.save(output_folder / (base_filename + " pure_signal.npy"), self._pure_lightcurve)
            # To add: stellar variability, degraded signals

    # ------------------------------------------------------------------------------------------------------------ #
    def get_degraded_lightcurve(self, *args):
        """Degrade the pure lightcurve, saving the result to self._noisy_lightcurve

        Parameters
        ----------
        args
            Format should be a list of functions to apply, in the order desired
            If the function requires args, they should be passed as a tuple with a kwargs dictionary:
            (func, kwargs)
            Args are applied as:
            new_lightcurve = func(old_lightcurve)
            or
            new_lightcurve = func(old_lightcurve, **kwargs)

        Returns
        -------

        """
        if self._pure_lightcurve is None:
            raise ValueError("Lightcurve is uninitialized, run collect_data() first")
        self._noisy_lightcurve = self._pure_lightcurve.copy()
        for arg in args:
            try:
                # Test to see if this is a (func, kwarg) pair:
                _ = iter(arg)
                self._noisy_lightcurve = arg[0](self._noisy_lightcurve, **arg[1])
            except TypeError:
                self._noisy_lightcurve = arg(self._noisy_lightcurve)
        return self._noisy_lightcurve.copy()

    # ------------------------------------------------------------------------------------------------------------ #
    def observation_report(self):
        print(self._name+" total observation time: "+u_str(self._observation_time.to(u.d)))
        print("Quiescent photons/sample: "+u_str(self._quiescent_intensity_sample))
        print("Total number of flares generated: ", self._all_flares.num_flares)

    # ------------------------------------------------------------------------------------------------------------ #
    def _update_flux_ready(self):
        flux_list = [self._area, self._efficiency, self._cadence, self._observation_target]
        update_ready = True
        for item in flux_list:
            if item is None:
                update_ready = False
        return update_ready

    def _update_flux(self):
        self._quiescent_intensity_sec = self._observation_target.get_flux()*self._area*self._efficiency
        self._quiescent_intensity_sample = self._quiescent_intensity_sec*self._cadence

    def observation_feasibility_report(self):
        if self._update_flux_ready():
            self._update_flux()
            print("Quiescent photons/sample: " + u_str(self._quiescent_intensity_sample))
            print("Quiescent photons/sec: " + u_str(self._quiescent_intensity_sec))
        elif self._area is not None and self._efficiency is not None and self._observation_target is not None:
            self._quiescent_intensity_sec = self._observation_target.get_flux() * self._area * self._efficiency
            print("Quiescent photons/sec: "+u_str(self._quiescent_intensity_sec))

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def collection_area(self):
        return self._area

    @collection_area.setter
    def collection_area(self, collection_area):
        if isinstance(collection_area, u.Quantity):
            self._area = collection_area.to("m2")
        else:
            self._area = u.Quantity(collection_area, unit="m2")
            warnings.warn("Casting collection_area, input as "+str(collection_area)+", to m??.", AstropyUserWarning)

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def cadence(self):
        return self._cadence

    @cadence.setter
    def cadence(self, cadence):
        if isinstance(cadence, u.Quantity):
            self._cadence = cadence.to(u.s)
        else:
            self._cadence = u.Quantity(cadence, unit=u.s)
            warnings.warn("Casting cadence, input as "+str(cadence)+", to s.", AstropyUserWarning)

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def observation_target(self):
        return self._observation_target

    @observation_target.setter
    def observation_target(self, observation_target):
        if isinstance(observation_target, DeltaStar):
            self._observation_target = observation_target
        else:
            raise TypeError("observation_target must be a DeltaStar or Star class instance")

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, random_seed):
        np.random.seed(random_seed)
        self._random_seed = random_seed

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def observation_time(self):
        return self._observation_time

    @observation_time.setter
    def observation_time(self, observation_time):
        if isinstance(observation_time, u.Quantity):
            self._observation_time = observation_time.to(u.s)
        else:
            self._observation_time = u.Quantity(observation_time, unit=u.s)
            warnings.warn("Casting observation_time, input as "+str(observation_time)+", to s.", AstropyUserWarning)

    # ------------------------------------------------------------------------------------------------------------ #
    def get_time_domain(self):
        if self._time_domain is None:
            raise ValueError("Time domain is not initialized")
        else:
            return self._time_domain.copy()
