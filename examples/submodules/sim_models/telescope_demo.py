

"""Shows how the telescopes module functions through examples."""

import numpy as np
from astropy import units as u
from scipy import stats

import exoechopy as eep
from exoechopy.utils.constants import *


def run():

    telescope_area = np.pi * (1*u.m)**2
    total_efficiency = 0.7
    observation_cadence = 5 * u.s
    observation_duration = 2 * u.hr
    telescope_random_seed = 99
    approximate_num_flares = 60

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    #  Create an observation target:
    spectral_band = eep.simulate.spectral.JohnsonPhotometricBand('U')
    emission_type = eep.simulate.spectral.SpectralEmitter(spectral_band, magnitude=16)

    star_radius = 1.*u.R_sun
    star_rotation_rate = 2*pi_u/observation_duration  # Spin quickly for demo purposes

    MyStar = eep.simulate.Star(radius=star_radius, spectral_type=emission_type, rotation_rate=star_rotation_rate,
                               limb=eep.simulate.Limb('nonlinear', coeffs=[0.5, 0.1, 0.1, -0.1]),
                               name="My Star", point_color='saddlebrown')

    MyStar.set_view_from_earth(0*u.deg, 90*u.deg)

    # eep.visualize.render_3d_planetary_system(MyStar)

    #  =============================================================  #

    #  Delta-function-like region centered on equator:
    long_1 = pi_u
    lat_1 = pi_u/2
    region_1 = eep.simulate.active_regions.Region(long_1, lat_1)
    region_1_name = 'DelFlares Reg1'

    #  Approx. max flare intensity:
    max_intensity = MyStar.get_flux()
    min_intensity = max_intensity/10

    #  To explicitly specify units, use a tuple of (pdf, unit):
    delta_flare_intensities = ([min_intensity, max_intensity], max_intensity.unit)

    MyDeltaFlareActivity = eep.simulate.active_regions.FlareActivity(
        eep.simulate.flares.DeltaFlare, intensity_pdf=delta_flare_intensities,
        label=region_1_name)

    ActiveRegion1 = eep.simulate.active_regions.ActiveRegion(flare_activity=MyDeltaFlareActivity,
        occurrence_freq_pdf=approximate_num_flares/2/observation_duration,
        region=region_1)

    MyStar.add_active_regions(ActiveRegion1)

    #  =============================================================  #
    MyTelescope = eep.simulate.Telescope(collection_area=telescope_area,
                                         efficiency=total_efficiency,
                                         cadence=observation_cadence,
                                         observation_target=MyStar,
                                         random_seed=telescope_random_seed,
                                         name="My First Telescope")

    MyTelescope.prepare_continuous_observational_run(observation_duration, print_report=True)
    MyTelescope.collect_data(save_diagnostic_data=True)

    eep.visualize.render_telescope_lightcurve(MyTelescope, flare_color='orange')

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    print("""
    Now we'll create a second active regions on opposite side of the star, 
    and give it different phenomenology to make it distinct.  This active region will have a wider range
    of flare locations, so it won't disappear behind the star quite as cleanly.
    """)

    #  Hemispherical region:
    min_long_2 = -pi_u / 2
    max_long_2 = pi_u / 2
    #  Center around equator, but do not extend all the way to the poles:
    min_lat_2 = pi_u / 12
    max_lat_2 = pi_u - min_lat_2
    region_2 = eep.simulate.active_regions.Region([min_long_2, max_long_2], [min_lat_2, max_lat_2])
    region_2_name = 'ExpFlares Reg2'

    #  To allow default units of ph/s-m², do not specify a unit (will throw a warning as a reminder):
    exponential_flare_intensities = stats.expon(scale=max_intensity)

    MyExpFlareActivity = eep.simulate.active_regions.FlareActivity(
        eep.simulate.flares.ParabolicRiseExponentialDecay,
        intensity_pdf=exponential_flare_intensities,
        onset_pdf=[1, 4] * u.s,
        decay_pdf=(stats.rayleigh(scale=10), u.s),
        label=region_2_name)

    ActiveRegion2 = eep.simulate.active_regions.ActiveRegion(flare_activity=MyExpFlareActivity,
                                                             occurrence_freq_pdf=approximate_num_flares/2/observation_duration,
                                                             region=region_2)

    MyStar.add_active_regions(ActiveRegion2)

    #  =============================================================  #

    #  Telescope is already defined and already knows changes in star, just need to re-prepare observations:

    MyTelescope.prepare_continuous_observational_run(observation_duration, print_report=True)
    MyTelescope.collect_data(save_diagnostic_data=True)

    eep.visualize.render_telescope_lightcurve(MyTelescope, flare_color={region_1_name: "orange",
                                                                        region_2_name: "b"})

    #  =============================================================  #
    MyTelescope.prepare_continuous_observational_run(observation_duration*5)
    MyTelescope.collect_data(save_diagnostic_data=True)

    print("""
    Upon close inspection of the lightcurve, you can see the general trend of alternating between 
    delta-like flares and exponential flares, which matches the star's rotation.""")

    eep.visualize.interactive_lightcurve(MyTelescope._time_domain, MyTelescope._pure_lightcurve)

# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #


if __name__ == "__main__":

    run()
