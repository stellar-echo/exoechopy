
"""Display the various orbital parameters and how they affect the orbits."""

# import numpy as np
import exoechopy as eep
from astropy import units as u
from exoechopy.utils.orbital_physics import true_anomaly_from_mean

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

spectral_band = eep.utils.spectral.JohnsonPhotometricBand('U')
emission_type = eep.utils.spectral.SpectralEmitter(spectral_band, 16)

MyStar = eep.simulate.Star(mass=.6*u.M_sun, radius=.6*u.R_sun, spectral_type=emission_type)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

planet_albedo = eep.utils.spectral.Albedo(spectral_band, 1.)

e_1 = 0.2
a_1 = 0.01 * u.au
i_1 = 20*u.deg
L_1 = 0*u.deg
w_1 = 0*u.deg
m0_1 = 0*u.deg
Planet1 = eep.simulate.KeplerianExoplanet(semimajor_axis=a_1,
                                          eccentricity=e_1,
                                          inclination=i_1,
                                          longitude=L_1,
                                          periapsis_arg=w_1,
                                          initial_anomaly=m0_1,
                                          albedo=planet_albedo,
                                          point_color='.1', path_color='.5',
                                          name='20deg inc. e.2')
MyStar.add_exoplanet(Planet1)


e_2 = 0.2
a_2 = 0.02 * u.au
i_2 = 20*u.deg
L_2 = 60*u.deg
w_2 = 0*u.deg
m0_2 = 0*u.deg
Planet2 = eep.simulate.KeplerianExoplanet(semimajor_axis=a_2,
                                          eccentricity=e_2,
                                          inclination=i_2,
                                          longitude=L_2,
                                          periapsis_arg=w_2,
                                          initial_anomaly=m0_2,
                                          albedo=planet_albedo,
                                          point_color='r', path_color='salmon',
                                          name='20deg inc. 60deg long e.2')
MyStar.add_exoplanet(Planet2)


e_3 = 0.
a_3 = 0.03 * u.au
i_3 = 20*u.deg
L_3 = 60*u.deg
w_3 = 90*u.deg
m0_3 = 0*u.deg
Planet3 = eep.simulate.KeplerianExoplanet(semimajor_axis=a_3,
                                          eccentricity=e_3,
                                          inclination=i_3,
                                          longitude=L_3,
                                          periapsis_arg=w_3,
                                          initial_anomaly=m0_3,
                                          albedo=planet_albedo,
                                          point_color='b', path_color='powderblue',
                                          name='20deg inc. 60deg long 90deg peri')
MyStar.add_exoplanet(Planet3)


e_4 = 0.2
a_4 = 0.04 * u.au
i_4 = 20*u.deg
L_4 = 60*u.deg
w_4 = 90*u.deg
m0_4 = 0*u.deg
Planet4 = eep.simulate.KeplerianExoplanet(semimajor_axis=a_4,
                                          eccentricity=e_4,
                                          inclination=i_4,
                                          longitude=L_4,
                                          periapsis_arg=w_4,
                                          initial_anomaly=m0_4,
                                          albedo=planet_albedo,
                                          point_color='g', path_color='darkseagreen',
                                          name='20deg inc. 60deg long 90deg peri e.2')
MyStar.add_exoplanet(Planet4)


e_5 = 0.2
a_5 = 0.05 * u.au
i_5 = 20*u.deg
L_5 = 60*u.deg
w_5 = 90*u.deg
m0_5 = 120*u.deg
Planet5 = eep.simulate.KeplerianExoplanet(semimajor_axis=a_5,
                                          eccentricity=e_5,
                                          inclination=i_5,
                                          longitude=L_5,
                                          periapsis_arg=w_5,
                                          initial_anomaly=m0_5,
                                          albedo=planet_albedo,
                                          point_color='darkviolet', path_color='thistle',
                                          name='20deg inc. 60deg long 90deg peri e.2 m0=120deg')
MyStar.add_exoplanet(Planet5)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


Planet1.about_orbit()

eep.visualize.render_3d_planetary_system(MyStar)
