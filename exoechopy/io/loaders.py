
"""
This module provides tools for importing .exo files and generating systems
"""

import os
import json
import copy

from pathlib import Path, PurePath
from exoechopy.utils import astropyio, is_nested
from exoechopy.utils.pdf_parser import *
from exoechopy.simulate import *


__all__ = ["generate_system_from_file", "SystemGenerator"]


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
keplerian_parameters = ["semimajor_axis", "eccentricity", "inclination", "longitude",
                        "periapsis_arg", "initial_anomaly"]


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
def generate_system_from_file(file_path: (str, Path)):
    return SystemGenerator(file_path)


class SystemGenerator(dict):
    def __init__(self, file_path: (str, Path)):
        """Import a system from a json file and provide some functionality to modify and use it

        Keys
        ----------
        'telescope_parameters'
        'star_parameters'
        'planet_parameters'
        'post_process_parameters'
        'solver_parameters'

        Parameters
        ----------
        file_path
            Location to load the system from


        """
        self._file_path = Path(file_path)
        with open(str(self._file_path)) as json_file:
            raw_json = json.load(json_file)
            super().__init__(raw_json)
        self._planets = None
        self._stars = None
        self._telescopes = None

    # ------------------------------------------------------------------------------------------------------------ #
    def initialize_experiments(self):
        """Generates a list of experiments that can then be run in parallel (or series)

        Returns
        -------
        List
            List of telescope objects with prepare_continuous_observational_run already set
            To run, implement .collect_data() on each telescope
        """
        self._stars = self.gen_stars()
        if not is_nested(self._stars):
            self._stars = [self._stars]

        self._planets = self.gen_planets()
        if not is_nested(self._planets):
            self._planets = [self._planets]

        self._telescopes = self.gen_telescope()

        all_experiments = []
        for telescope in self._telescopes:
            for star_list in self._stars:
                for planet_list in self._planets:
                    _stars = copy.deepcopy(star_list)
                    _planets = copy.deepcopy(planet_list)
                    _telescope = copy.deepcopy(telescope)
                    for s in _stars:
                        for p in _planets:
                            s.add_orbiting_object(p)
                    if len(_stars) > 1:
                        raise NotImplementedError("Not sure how to do this yet")
                    else:
                        _telescope.observation_target = _stars[0]
                    solver = self["solver_parameters"]["solver"]
                    if solver is None or solver == "Keplerian":
                        solver = None
                    elif solver == "y6solver":
                        dt = self["solver_parameters"]["solver_timestep"]
                        steps_per_save = self["solver_parameters"]["steps_per_save"]
                        simulation_objects = [_stars[0]]
                        simulation_objects.extend(_stars[0].get_all_orbiting_objects())
                        solver = SymplecticSolver(*simulation_objects, dt=dt, steps_per_save=steps_per_save)
                    observation_time = astropyio.to_quantity(self['telescope_parameters']['observation_time'])

                    _telescope.prepare_continuous_observational_run(observation_time=observation_time,
                                                                    solver=solver)
                    all_experiments.append(_telescope)
        return all_experiments

    # ------------------------------------------------------------------------------------------------------------ #
    def get_system_objects(self, full_experiment: bool=False):
        """Generates the files required to run a simulation

        Parameters
        ----------
        full_experiment
            If True, and the system contains iterables, such as a range of star magnitudes,
            will return all values.
            If False, will only return the very first instance from the iterables.

        Returns
        -------
        All stars, all planets, and all telescope class instances
            If running a full_experiment, they will NOT be attached to each other
            For instance, if the star mass is iterated over, the planet is not attached
            and the orbital parameters do not change--it must be manually attached for each run.
            Similarly, each star or planet iteration must be attached to the telescope before running.
        """
        if full_experiment:
            raise NotImplementedError("Not yet implemented, currently only handles specific stars")
        else:
            return self.gen_stars(), self.gen_planets(), self.gen_telescope()

    # ------------------------------------------------------------------------------------------------------------ #
    def gen_planets(self):
        """Generates planet objects requested

        Parameters
        ----------

        Returns
        -------
        Planet or list of Planets
        """
        # Sometimes will have multi-planet systems, so treat as list of planets always for forward compatibility
        if not is_nested(self['planet_parameters']):
            all_planets = [self['planet_parameters']]
        else:
            all_planets = self['planet_parameters']

        #  =============================================================  #
        #  Check if any parameters are iterables, rather than than fixed values:
        all_planets_iter_dict = {}
        #  These are items that are expected to iterate, even if they are not iteration experiments,
        #  and require manual processing:
        no_iter_list = []

        # Read out the planet properties from the JSON dictionary translation, see if there are any iterated parameters
        for p_i, planet in enumerate(all_planets):
            # Maintain a list of planet parameters that are iterated over:
            planet_iterables = {}
            all_planets_iter_dict[p_i] = planet_iterables

            for k, v in planet.items():
                if is_nested(v) and k not in no_iter_list:
                    planet_iterables[k] = astropyio.to_quantity_list(v)

        #  =============================================================  #
        #  Implement planet generation:
        include_iterable = False
        for k, v in all_planets_iter_dict.items():
            if len(v) > 0:
                include_iterable = True
        if include_iterable:
            raise NotImplementedError("Cannot yet handle iterated parameters")
            all_iterations = []
            for iterated_key in all_planets_iter_dict.keys():
                pass
        # Base case, no iterated parameters:
        else:
            return_planet_list = []
            for p_i, planet in enumerate(all_planets):
                # Standard parameters:
                _type = planet['planet_type']
                # Special planet keywords, depend on planet type
                planet_kw = {}
                if _type == "KeplerianExoplanet":
                    _Planet = KeplerianExoplanet
                    for param in keplerian_parameters:
                        try:
                            planet_kw[param] = astropyio.to_quantity(planet[param])
                        except TypeError:
                            planet_kw[param] = planet[param]

                elif _type == "Exoplanet":
                    _Planet = Exoplanet
                else:
                    raise TypeError("Unknown planet type specified: ", self['planet_parameters']["planet_type"],
                                    ", must be KeplerianExoplanet or Exoplanet")

                _spectral = _interpret_spectrum(planet['planet_spectral_band'])
                _albedo = planet['planet_albedo']
                planet_albedo = Albedo(_spectral, _albedo)

                point_color = planet['planet_point_color']
                planet_name = planet['planet_name']

                mass = astropyio.to_quantity(planet['planet_mass'])
                radius = astropyio.to_quantity(planet['planet_radius'])
                parent_mass = astropyio.to_quantity(planet['parent_mass'])

                current_planet = _Planet(mass=mass, radius=radius, albedo=planet_albedo,
                                         parent_mass=parent_mass,
                                         point_color=point_color, name=planet_name,
                                         **planet_kw)

                return_planet_list.append(current_planet)
            return return_planet_list

    # ------------------------------------------------------------------------------------------------------------ #
    def gen_telescope(self):
        """Generates telescope objects requested

        Parameters
        ----------

        Returns
        -------
        Telescope or list of Telescopes
        """

        #  These are items that are expected to iterate, even if they are not iteration experiments,
        #  and require manual processing:
        no_iter_list = []

        # Read out the telescope properties from the JSON dictionary translation, look for iterated parameters
        # Maintain a list of telescope parameters that are iterated over:
        telescope_iterables = {}

        for k, v in self['telescope_parameters'].items():
            if is_nested(v) and k not in no_iter_list:
                telescope_iterables[k] = astropyio.to_quantity_list(v)

        #  =============================================================  #
        #  Implement telescope generation:
        if len(telescope_iterables) > 0:
            raise NotImplementedError("Cannot yet handle iterated parameters")
            all_iterations = []
            for iterated_key in telescope_iterables.keys():
                pass
        # Base case, no iterated parameters:
        else:
            return_telescope = None
            telescope = self['telescope_parameters']

            # Special telescope keywords, depend on telescope type (placeholder)
            telescope_kw = {}
            non_kw_list = ["observation_time"]
            for k, v in telescope.items():
                if k not in non_kw_list:
                    try:
                        _ = iter(v)
                        if len(v) == 2:
                            telescope_kw[k] = astropyio.to_quantity(v)
                        else:
                            telescope_kw[k] = v
                    except TypeError:
                        telescope_kw[k] = v
            _Telescope = Telescope

            current_telescope = Telescope(**telescope_kw)

            return_telescope = [current_telescope]

            return return_telescope

    # ------------------------------------------------------------------------------------------------------------ #
    def gen_stars(self):
        """Generates star objects requested

        Parameters
        ----------

        Returns
        -------
        Star or list of Stars
        """
        # Sometimes will have binary systems, so treat as list of stars always for forward compatibility
        if not is_nested(self['star_parameters']):
            all_stars = [self['star_parameters']]
        else:
            all_stars = self['star_parameters']

        #  =============================================================  #
        #  Check if any parameters are iterables, rather than than fixed values:
        all_stars_iter_dict = {}
        #  These are items that are expected to iterate, even if they are not iteration experiments,
        #  and require manual processing:
        no_iter_list = ['active_region']

        # Read out the star properties from the JSON dictionary translation, see if there are any iterated parameters
        for s_i, star in enumerate(all_stars):
            # Maintain a list of star parameters that are iterated over:
            star_iterables = {}
            all_stars_iter_dict[s_i] = star_iterables

            for k, v in star.items():
                if is_nested(v) and k not in no_iter_list:
                    star_iterables[k] = astropyio.to_quantity_list(v)

            #  =============================================================  #
            # Sub-dicts:

            # Often have multiple active regions on the same star, treat as a list of regions:
            if not is_nested(star['active_region']):
                all_active_regions = [star['active_region']]
            else:
                all_active_regions = star['active_region']
            all_regions_iter_dict = {}
            iterable_regions = False
            for r_i, active_region in enumerate(all_active_regions):
                # Maintain a list of star parameters that are iterated over:
                region_iterables = {}
                all_regions_iter_dict[r_i] = region_iterables

                # Potentially iterable parameters:
                _intensity_pdf = _interpret_pdf(active_region['intensity_pdf'])
                _occurrence_freq_pdf = _interpret_pdf(active_region['occurrence_freq_pdf'])
                _region = _interpret_pdf(active_region['region'])
                _flare_kwargs = active_region['flare_kwargs']

                if is_nested(_intensity_pdf):
                    region_iterables["intensity_pdf"] = _intensity_pdf
                    iterable_regions = True
                if is_nested(_occurrence_freq_pdf):
                    region_iterables["occurrence_freq_pdf"] = _occurrence_freq_pdf
                    iterable_regions = True
                if is_nested(_region):
                    region_iterables["region"] = _region
                    iterable_regions = True
                if is_nested(_flare_kwargs):
                    region_iterables["flare_kwargs"] = _flare_kwargs
                    iterable_regions = True
            if iterable_regions:
                star_iterables['active_region'] = all_regions_iter_dict

        #  =============================================================  #
        #  Implement star generation:
        include_iterable = False
        for k, v in all_stars_iter_dict.items():
            if len(v) > 0:
                include_iterable = True
        if include_iterable:
            raise NotImplementedError("Cannot yet handle iterated parameters")
            all_iterations = []
            for iterated_key in all_stars_iter_dict.keys():
                pass
        # Base case, no iterated parameters:
        else:
            return_stars_list = []
            for s_i, star in enumerate(all_stars):
                # Standard parameters:
                _type = star['star_type']
                # Special star keywords, depend on star type
                star_kw = {}
                if _type == "DeltaStar":
                    _Star = DeltaStar
                elif _type == "Star":
                    _Star = Star
                    star_kw["limb"] = _interpret_limb(star['limb'])
                    star_kw["radius"] = astropyio.to_quantity(star['star_radius'])
                else:
                    raise TypeError("Unknown star type specified: ", self['star_parameters']["star_type"],
                                    ", must be DeltaStar or Star")

                _spectral = _interpret_spectrum(star['spectral_properties'])
                _star_magnitude = star['star_magnitude']
                emission_type = SpectralEmitter(_spectral, _star_magnitude)

                point_color = star['point_color']
                star_name = star['star_name']

                mass = astropyio.to_quantity(star['star_mass'])
                earth_long = astropyio.to_quantity(star['earth_longitude'])
                earth_lat = astropyio.to_quantity(star['earth_latitude'])

                current_star = _Star(mass=mass, spectral_type=emission_type,
                                     point_color=point_color, name=star_name, **star_kw)
                current_star.set_view_from_earth(earth_long, earth_lat)

                #  =============================================================  #
                # Often have multiple active regions on the same star, treat as a list of regions:
                if not is_nested(star['active_region']):
                    all_active_regions = [star['active_region']]
                else:
                    all_active_regions = star['active_region']

                for r_i, active_region in enumerate(all_active_regions):
                    _flare_type = active_region['flare_type']
                    if _flare_type == "DeltaFlare":
                        _Flare = DeltaFlare
                    elif _flare_type in ("ExponentialFlare1", "ParabolicRiseExponentialDecay", "PRED"):
                        _Flare = ParabolicRiseExponentialDecay
                    else:
                        raise TypeError("Unknown flare type: ", _flare_type)
                    intensity_pdf = _interpret_pdf(active_region['intensity_pdf'])
                    occurrence_freq_pdf = _interpret_pdf(active_region['occurrence_freq_pdf'])
                    if _type == "DeltaStar":
                        region = None
                    else:
                        region_kw = _interpret_pdf(active_region['region'])
                        region = Region(**region_kw)
                    flare_kwargs = _interpret_kw_pdf(active_region['flare_kwargs'])
                    flare_activity = FlareActivity(_Flare, intensity_pdf=intensity_pdf,
                                                   **flare_kwargs)
                    ar = ActiveRegion(flare_activity=flare_activity,
                                      occurrence_freq_pdf=occurrence_freq_pdf,
                                      region=region)
                    current_star.add_active_regions(ar)

                return_stars_list.append(current_star)
            return return_stars_list

    # ------------------------------------------------------------------------------------------------------------ #
    def save_system(self,
                    folder: (str, Path),
                    filename: (str, PurePath)):
        """
        Save the system file as a .exo JSON file

        Parameters
        ----------
        folder
            Location to generate the system template
        filename
            Filename for the system template
        """
        folder = Path(folder)
        if not folder.exists():
            os.mkdir(folder)

        filename = PurePath(filename)
        if filename.suffix == "":
            filename = filename.with_suffix(".exo")

        with open(str(folder/filename), 'w') as json_export_file:
            json.dump(self.copy(), json_export_file)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Raw input parsers:
def _interpret_pdf(for_interpretation: (dict, float, list)):
    if not is_nested(for_interpretation):
        all_pdfs = [for_interpretation]
    else:
        all_pdfs = for_interpretation
    return_vals = []
    for pdf in all_pdfs:
        if isinstance(pdf, dict):
            _pdf = _interpret_kw_pdf(pdf)
            if _pdf is None:
                _pdf, unit = parse_pdf(pdf, warnings_on=False)
                if unit is None:
                    return_vals.append(_pdf)
                else:
                    return_vals.append((_pdf, unit))
            else:
                return_vals.append(_pdf)
        else:
            _pdf, unit = parse_pdf(pdf, warnings_on=False)
            if unit is None:
                return_vals.append(_pdf)
            else:
                return_vals.append((_pdf, unit))
    if len(return_vals) == 1:
        return return_vals[0]
    else:
        return return_vals


#  =============================================================  #
def _interpret_kw_pdf(for_interpretation: (dict, float, list)):
    return_val = for_interpretation
    if isinstance(for_interpretation, dict):
        return_val = {}
        for k, v in for_interpretation.items():
            _val = v
            if k in pdf_dict:
                # Then we're likely in a recursion:
                _val = pdf_dict[k](**v)
                return _val
            elif isinstance(v, dict):
                _val = _interpret_kw_pdf(v)
                if 'unit' in v:
                    _val = (_val, v['unit'])
                return_val[k] = _val
            else:
                _pdf, unit = parse_pdf([k, v], warnings_on=False)
                if unit is None:
                    return_val[_pdf[0]] = _pdf[1]
                else:
                    return_val[_pdf[0]] = (_pdf[1], unit)
    return return_val


#  =============================================================  #
def _interpret_spectrum(for_interpretation: (dict, float)):
    if isinstance(for_interpretation, dict):
        kv = [(k, v) for k, v in for_interpretation.items()]
        all_spectra = []
        for spectral_type, spectral_band in kv:
            if spectral_type == "JohnsonPhotometricBand":
                all_spectra.append(JohnsonPhotometricBand(spectral_band))
            else:
                raise NotImplementedError("Currently only supports JohnsonPhotometricBand")
        if len(all_spectra) == 1:
            return all_spectra[0]
        else:
            return all_spectra
    else:
        # Just a count, most likely:
        return float(for_interpretation)


#  =============================================================  #
def _interpret_limb(for_interpretation: dict):
    # TODO add spectral_limb once it exists
    kv = [(k, v) for k, v in for_interpretation.items()][0]
    return Limb(limb_model=kv[0], coeffs=kv[1])
