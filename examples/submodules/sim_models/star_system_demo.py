
"""Display the various orbital parameters and how they affect the orbits."""

from astropy import units as u

import exoechopy as eep
from exoechopy.utils import u_str


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def run():

    spectral_band = eep.simulate.spectral.JohnsonPhotometricBand('U')
    emission_type = eep.simulate.spectral.SpectralEmitter(spectral_band, magnitude=16)

    MyStar = eep.simulate.Star(mass=.6*u.M_sun, radius=.6*u.R_sun, spectral_type=emission_type,
                               name="My Star", point_color='saddlebrown')
    MyStar.set_view_from_earth(180*u.deg, 120*u.deg)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

    planet_albedo = eep.simulate.spectral.Albedo(spectral_band, 1.)

    #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   #
    e_1 = 0.2
    a_1 = 0.01 * u.au
    i_1 = 20*u.deg
    L_1 = 0*u.deg
    w_1 = 0*u.deg
    m0_1 = 0*u.deg
    name_1 = "e: "+str(e_1)+", a: "\
             +u_str(a_1)+", i: "+u_str(i_1)+", Ω: "\
             +u_str(L_1)+", ω: "+u_str(w_1)+", M0: "+u_str(m0_1)
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

    #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   #
    e_2 = 0.2
    a_2 = 0.02 * u.au
    i_2 = 20*u.deg
    L_2 = 60*u.deg
    w_2 = 0*u.deg
    m0_2 = 0*u.deg
    name_2 = "e: "+str(e_2)+", a: "+u_str(a_2)+", i: "\
             +u_str(i_2)+", Ω: "+u_str(L_2)+", ω: "\
             +u_str(w_2)+", M0: "+u_str(m0_2)
    Planet2 = eep.simulate.KeplerianExoplanet(semimajor_axis=a_2,
                                              eccentricity=e_2,
                                              inclination=i_2,
                                              longitude=L_2,
                                              periapsis_arg=w_2,
                                              initial_anomaly=m0_2,
                                              albedo=planet_albedo,
                                              point_color='r', path_color='salmon',
                                              name=name_2)
    MyStar.add_exoplanet(Planet2)

    #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   #
    e_3 = 0.
    a_3 = 0.03 * u.au
    i_3 = 20*u.deg
    L_3 = 60*u.deg
    w_3 = 90*u.deg
    m0_3 = 0*u.deg
    name_3 = "e: "+str(e_3)+", a: "+u_str(a_3)+", i: "\
             +u_str(i_3)+", Ω: "+u_str(L_3)+", ω: "\
             +u_str(w_3)+", M0: "+u_str(m0_3)
    Planet3 = eep.simulate.KeplerianExoplanet(semimajor_axis=a_3,
                                              eccentricity=e_3,
                                              inclination=i_3,
                                              longitude=L_3,
                                              periapsis_arg=w_3,
                                              initial_anomaly=m0_3,
                                              albedo=planet_albedo,
                                              point_color='b', path_color='powderblue',
                                              name=name_3)
    MyStar.add_exoplanet(Planet3)

    #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   #
    e_4 = 0.2
    a_4 = 0.04 * u.au
    i_4 = 20*u.deg
    L_4 = 60*u.deg
    w_4 = 90*u.deg
    m0_4 = 0*u.deg
    name_4 = "e: "+str(e_4)+", a: "+u_str(a_4)+", i: "\
             +u_str(i_4)+", Ω: "+u_str(L_4)+", ω: "\
             +u_str(w_4)+", M0: "+u_str(m0_4)
    Planet4 = eep.simulate.KeplerianExoplanet(semimajor_axis=a_4,
                                              eccentricity=e_4,
                                              inclination=i_4,
                                              longitude=L_4,
                                              periapsis_arg=w_4,
                                              initial_anomaly=m0_4,
                                              albedo=planet_albedo,
                                              point_color='g', path_color='darkseagreen',
                                              name=name_4)
    MyStar.add_exoplanet(Planet4)

    #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   #
    e_5 = 0.2
    a_5 = 0.05 * u.au
    i_5 = 20*u.deg
    L_5 = 60*u.deg
    w_5 = 90*u.deg
    m0_5 = 120*u.deg
    name_5 = "e: "+str(e_5)+", a: "+u_str(a_5)+", i: "\
             +u_str(i_5)+", Ω: "+u_str(L_5)+", ω: "\
             +u_str(w_5)+", M0: "+u_str(m0_5)
    Planet5 = eep.simulate.KeplerianExoplanet(semimajor_axis=a_5,
                                              eccentricity=e_5,
                                              inclination=i_5,
                                              longitude=L_5,
                                              periapsis_arg=w_5,
                                              initial_anomaly=m0_5,
                                              albedo=planet_albedo,
                                              point_color='darkviolet', path_color='thistle',
                                              name=name_5)
    MyStar.add_exoplanet(Planet5)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

    Planet1.about_orbit()

    eep.visualize.render_3d_planetary_system(MyStar)

    eep.visualize.animate_3d_planetary_system(MyStar)


# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #


if __name__ == "__main__":
    run()

