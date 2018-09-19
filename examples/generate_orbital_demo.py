
"""Display the various orbital parameters and how they affect the orbits."""

from astropy import units as u

import exoechopy as eep
from exoechopy.utils import u_str

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def run():
    spectral_band = eep.simulate.models.spectral.JohnsonPhotometricBand('U')
    emission_type = eep.simulate.models.spectral.SpectralEmitter(spectral_band, magnitude=16)

    MyStar = eep.simulate.Star(mass=.6*u.M_sun, radius=.6*u.R_sun, spectral_type=emission_type,
                               name="My Star", point_color='saddlebrown')
    MyStar.set_view_from_earth(180*u.deg, 120*u.deg)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

    planet_albedo = eep.simulate.models.spectral.Albedo(spectral_band, 1.)

    #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   #
    e_1 = 0.3
    a_1 = 0.01 * u.au
    i_1 = 20*u.deg
    L_1 = 0*u.deg
    w_1 = 0*u.deg
    m0_1 = 180*u.deg
    name_1 = "e: "+str(e_1)+", a: "+u_str(a_1)+", i: "+u_str(i_1)+", Ω: "+u_str(L_1)+", ω: "+u_str(w_1)+", M0: "+u_str(m0_1)
    Planet1 = eep.simulate.KeplerianExoplanet(semimajor_axis=a_1,
                                              eccentricity=e_1,
                                              inclination=i_1,
                                              longitude=L_1,
                                              periapsis_arg=w_1,
                                              initial_anomaly=m0_1,
                                              albedo=planet_albedo,
                                              point_color='.1', path_color='.5',
                                              name=name_1)
    MyStar.add_exoplanet(Planet1)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

    Planet1.about_orbit()

    eep.visualize.orbit_plot(Planet1)

    eep.visualize.animate_3d_planetary_system(MyStar, num_frames=100)

# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #


if __name__ == "__main__":
    run()

