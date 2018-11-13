
"""
This generates a template that includes (hopefully) all relevant star system parameters.
The template can be loaded by various ExoEchoPy modules.

In several cases, the template generates default values.  Many of these are provided as examples.

For analysis, see the analysis_template

Can be initialized from command line to provide a working folder and a filename

"""

import json
import sys
import os

from pathlib import Path, PurePath
from numpy import pi


__all__ = ["create_system_template"]


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Initialization parameters
init_args = sys.argv
if len(init_args) > 1:
    working_folder = Path(init_args[1])
    if not working_folder.exists():
        os.mkdir(working_folder)
else:
    working_folder = Path.cwd()
if len(init_args) > 2:
    output_file_name = PurePath(init_args[2])
    if output_file_name.suffix == "":
        output_file_name = output_file_name.with_suffix(".exo")
else:
    output_file_name = "system_template.exo"


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Telescope parameters
telescope_parameters = {
    "collection_area": (pi*2.**2, "m2"),
    "efficiency": 1.,
    "cadence": (1., "s"),
    "observation_time": (4., "hr"),
    "random_seed": 0,
    "name": ""
}


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Degradation parameters
post_process_parameters = {
    "add_poisson_noise": True
}


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Solver parameters
solver_parameters = {
    "solver": "Keplerian",
    "solver_timestep": None,
    "steps_per_save": None
}


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Star parameters
star_parameters = {
    # For delta-star, the radius is actually ignored, it's just provided in the template for compatibility with Star
    "star_radius": (.6, "R_sun"),
    "star_mass": (.6, "M_sun"),
    "star_type": "DeltaStar",

    # Point towards Earth from Star:
    "earth_longitude": (0, "rad"),
    "earth_latitude": (0, "rad"),

    # Star spectral parameters
    "star_magnitude": 16,
    "spectral_properties": {"JohnsonPhotometricBand": "U"},
    "limb": {"quadratic": [.1, .3]},

    # Flare parameters
    "active_region": {
        "flare_type": "ExponentialFlare1",
        # When input like this, represents a uniform distribution:
        "intensity_pdf": [20000., 200000.],
        "occurrence_freq_pdf": {"uniform": {"loc": 50, "scale": 900}},
        # For delta-star, the region is ignored, we just provide an example in the template for compatibility with Star
        "region": {"longitude_pdf": [0, 2 * pi], "latitude_pdf": [pi / 4, 2 * pi - pi / 4]},
        # To iterate over a single kwarg, create a list of flare_kwarg dicts
        "flare_kwargs": {
            "onset_pdf": ([1, 5], "s"),
            # Example of how to pass a more complicated distribution:
            "decay_pdf": {"rayleigh": {"scale": 6}, "unit": "s"}}
    },

    # Star draw parameters
    "point_color": 'saddlebrown',
    "star_name": "My Default Star"
}


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Planet parameters
planet_parameters = {
    "planet_type": "KeplerianExoplanet",
    "planet_mass": (10., "M_jup"),
    "planet_radius": (2, "R_jup"),
    # Hint the star mass, just in case it gets separated from star later on:
    "planet_host_star_mass": (0.6, "M_sun"),

    # Keplerian Orbital parameters
    "eccentricity": 0.,
    "semimajor_axis": (.06, "au"),
    "inclination": (0, "rad"),
    "longitude": (0, "rad"),
    "periapsis_arg": (0, "rad"),
    "initial_anomaly": (0, "rad"),

    # Planet spectral parameters
    "planet_spectral_band": {"JohnsonPhotometricBand": "U"},
    "planet_albedo": 1.,

    # Planet draw parameters
    "planet_name": "My Default Planet",
    "planet_point_color": 'k',
    "planet_path_color": 'dimgray'
}

# To add multiple stars, for instance, turn values into lists of dicts
all_parameters = {
    "telescope_parameters": telescope_parameters,
    "star_parameters": star_parameters,
    "planet_parameters": planet_parameters,
    "post_process_parameters": post_process_parameters,
    "solver_parameters": solver_parameters
}


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
def create_system_template(folder: (str, Path),
                           filename: (str, PurePath)):
    """Generates a star system template file in folder with name filename

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

    with open(folder/filename, 'w') as json_export_file:
        json.dump(all_parameters, json_export_file)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

if __name__ == "__main__":
    print("Generating template at: ", working_folder/output_file_name)
    create_system_template(working_folder, output_file_name)
