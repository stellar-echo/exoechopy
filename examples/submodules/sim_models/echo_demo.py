

"""Shows how the telescopes module functions through examples."""

import numpy as np
from astropy import units as u
from scipy import stats

import exoechopy as eep
from exoechopy.utils.constants import *


def run():

    print("""
    Detecting echoes is hard--we start by showing what results look like using optimistic values.
    First, create a delta-function star (zero radius) and give it a big bright planet
    """)

    telescope_area = np.pi * (5*u.m)**2
    total_efficiency = 0.8
    observation_cadence = 2 * u.s
    observation_duration = 1 * u.hr
    telescope_random_seed = 99
    approximate_num_flares = 10

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    #  Create an observation target:
    spectral_band = eep.simulate.spectral.JohnsonPhotometricBand('U')
    emission_type = eep.simulate.spectral.SpectralEmitter(spectral_band, magnitude=16)

    MyStar = eep.simulate.DeltaStar(spectral_type=emission_type, point_color='saddlebrown')

    # Face-on circular
    MyStar.set_view_from_earth(0*u.deg, 0*u.deg)

    #  =============================================================  #
    #  Create a planet to orbit the star
    planet_albedo = eep.simulate.spectral.Albedo(spectral_band, 1.)
    planet_radius = 2*u.R_jup
    e_1 = 0.  # eccentricity
    a_1 = 0.02 * u.au  # semimajor axis
    i_1 = 0 * u.deg  # inclination
    L_1 = 0 * u.deg  # longitude
    w_1 = 0 * u.deg  # arg of periapsis
    m0_1 = 0 * u.deg  # initial anomaly

    planet_name = "HelloExoWorld"

    Planet1 = eep.simulate.KeplerianExoplanet(semimajor_axis=a_1,
                                              eccentricity=e_1,
                                              inclination=i_1,
                                              longitude=L_1,
                                              periapsis_arg=w_1,
                                              initial_anomaly=m0_1,
                                              albedo=planet_albedo,
                                              radius=planet_radius,
                                              point_color='k', path_color='dimgray',
                                              name=planet_name)
    MyStar.add_exoplanet(Planet1)

    eep.visualize.render_3d_planetary_system(MyStar)

    #  =============================================================  #

    #  Approx. max flare intensity:
    max_intensity = 10*MyStar.get_flux()
    min_intensity = max_intensity/10

    #  To explicitly specify units, use a tuple of (pdf, unit):
    flare_intensities = ([min_intensity, max_intensity], max_intensity.unit)

    MyExpFlareActivity = eep.simulate.active_regions.FlareActivity(eep.simulate.flares.ExponentialFlare1,
                                                                   intensity_pdf=flare_intensities,
                                                                   onset_pdf=[1, 4] * u.s,
                                                                   decay_pdf=(stats.rayleigh(scale=5), u.s))

    Starspot = eep.simulate.active_regions.ActiveRegion(flare_activity=MyExpFlareActivity,
                                                occurrence_freq_pdf=approximate_num_flares/observation_duration)

    MyStar.add_active_regions(Starspot)

    #  =============================================================  #
    MyTelescope = eep.simulate.Telescope(collection_area=telescope_area,
                                         efficiency=total_efficiency,
                                         cadence=observation_cadence,
                                         observation_target=MyStar,
                                         random_seed=telescope_random_seed,
                                         name="My Telescope")

    MyTelescope.prepare_continuous_observational_run(observation_duration, print_report=True)
    MyTelescope.collect_data(save_diagnostic_data=True)

    print("MyTelescope._all_exoplanet_contrasts: ", MyTelescope._all_exoplanet_contrasts)
    print("MyTelescope._all_exoplanet_lags: ", MyTelescope._all_exoplanet_lags)

    eep.visualize.interactive_lightcurve(MyTelescope._time_domain, MyTelescope._pure_lightcurve)

# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #

if __name__ == "__main__":

    run()
