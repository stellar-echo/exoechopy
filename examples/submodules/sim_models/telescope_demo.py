

"""Shows how the telescopes module functions through examples."""

import numpy as np
from astropy import units as u
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
from exoechopy.visualize.standard_3d_plots import *
from exoechopy.simulate.models import *
from exoechopy.utils.constants import *


def run():

    telescope_area = np.pi * (1*u.m)**2
    total_efficiency = 0.7
    observation_cadence = 5 * u.s
    observation_duration = 2 * u.hr
    telescope_random_seed = 99

    #  =============================================================  #
    #  Create an observation target:
    spectral_band = spectral.JohnsonPhotometricBand('U')
    emission_type = spectral.SpectralEmitter(spectral_band, magnitude=16)

    star_radius = 1.*u.R_sun
    star_rotation_rate = 2*pi_u/observation_duration  # Spin quickly for demo purposes

    MyStar = Star(radius=star_radius, spectral_type=emission_type, rotation_rate=star_rotation_rate,
                  limb_function=no_limb_darkening,
                  name="My Star", point_color='saddlebrown')

    MyStar.set_view_from_earth(0*u.deg, 90*u.deg)

    #  =============================================================  #
    #  Create two active regions on opposite sides of the star, give different phenomenology to distinguish:
    min_long_1 = np.pi - np.pi/12
    max_long_1 = np.pi + np.pi/12
    min_lat_1 = np.pi / 4
    max_lat_1 = min_lat_1 + np.pi / 6
    region_1 = active_regions.Region([min_long_1, max_long_1], [min_lat_1, max_lat_1])
    region_1_name = 'DelFlares Reg1'

    min_long_2 = -np.pi / 12
    max_long_2 = np.pi / 12
    min_lat_2 = np.pi / 2 + np.pi / 6
    max_lat_2 = min_lat_2 + np.pi / 6
    region_2 = active_regions.Region([min_long_2, max_long_2], [min_lat_2, max_lat_2])
    region_2_name = 'ExpFlares Reg2'

    MyDeltaFlareActivity = active_regions.FlareActivity(flares.DeltaFlare, intensity_pdf=[1, 10],
                                                        label=region_1_name)

    MyExpFlareActivity = active_regions.FlareActivity(flares.ExponentialFlare1,
                                                      intensity_pdf=stats.expon(scale=10 * u.s),
                                                      onset_pdf=[1, 4] * u.s,
                                                      decay_pdf=(stats.rayleigh(scale=5), u.s),
                                                      label=region_2_name)

    ActiveRegion1 = active_regions.ActiveRegion(flare_activity=MyDeltaFlareActivity,
                                                occurrence_freq_pdf=20/observation_duration,
                                                region=region_1)

    ActiveRegion2 = active_regions.ActiveRegion(flare_activity=MyExpFlareActivity,
                                                occurrence_freq_pdf=20/observation_duration,
                                                region=region_2)

    MyStar.add_active_regions(ActiveRegion1, ActiveRegion2)

    #  =============================================================  #

    MyTelescope = Telescope(collection_area=telescope_area,
                            efficiency=total_efficiency,
                            cadence=observation_cadence,
                            observation_target=MyStar,
                            random_seed=telescope_random_seed,
                            name="My First Telescope")

    MyTelescope.prepare_continuous_observational_run(observation_duration, print_report=True)
    earth_flare_angles, earth_flare_visibility = MyTelescope.collect_data()

    # TODO add axes to plot visibility at the same time to verify it aligns
    # TODO Fix the ultra-warning

    flare_times = MyTelescope._all_flares.all_flare_times

    fig, ax_array = plt.subplots(nrows=3, figsize=(12, 6))
    ax_array[0].plot(MyTelescope._time_domain, MyTelescope._pure_lightcurve,
                     color='k', lw=1., drawstyle='steps-post', label="Lightcurve")

    flare_colors = ['b' if f.label == region_2_name else 'orange' for f in MyTelescope._all_flares.all_flares]
    ax_array[0].scatter(flare_times, np.ones(len(flare_times))*np.median(MyTelescope._pure_lightcurve),
                        marker="^", color=flare_colors)

    ax_array[1].scatter(flare_times, earth_flare_angles,
                        color=flare_colors, marker='.', s=8, label="earth_flare_angles")
    ax_array[1].plot(MyTelescope._time_domain, 1-np.cos(MyStar.get_rotation(MyTelescope._time_domain)),
                     color='k', ls='--', lw=1)

    ax_array[2].scatter(flare_times, earth_flare_visibility,
                        color=flare_colors, marker='.', s=8, label="earth_flare_visibility")

    fig.subplots_adjust(hspace=0)

    plt.show()


# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #

if __name__ == "__main__":

    run()
