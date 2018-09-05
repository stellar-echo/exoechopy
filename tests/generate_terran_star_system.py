
"""Test orbital dynamics with a solar system much like our own..."""

# import numpy as np
import exoechopy as eep
from astropy import units as u
from exoechopy.utils.orbital_physics import true_anomaly_from_mean

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

spectral_band = eep.utils.spectral.JohnsonPhotometricBand('U')
emission_type = eep.utils.spectral.SpectralEmitter(spectral_band, 16)

MyStar = eep.simulate.Star(mass=1*u.M_sun, radius=1.*u.R_sun, spectral_type=emission_type,
                           name="Solly", point_color='darkorange')

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

planet_albedo = eep.utils.spectral.Albedo(spectral_band, 1.)

#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   #
a_M1 = 0.38709893*u.au
e_M1 = 0.20563069
i_M1 = 3.38*u.deg  # relative to sun equator
L_M1 = 48.33167*u.deg
w_M1 = 77.45645*u.deg - L_M1
M_M1 = true_anomaly_from_mean(252.25084*u.deg - w_M1 - L_M1, e_M1)
MercurialPlanet = eep.simulate.KeplerianExoplanet(semimajor_axis=a_M1,
                                                  eccentricity=e_M1,
                                                  inclination=i_M1,
                                                  longitude=L_M1,
                                                  periapsis_arg=w_M1,
                                                  initial_anomaly=M_M1,
                                                  albedo=planet_albedo,
                                                  point_color='.1', path_color='.5',
                                                  name='Mercurial')
MyStar.add_exoplanet(MercurialPlanet)

#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   #
a_V = 0.72333199*u.au
e_V = 0.006772
i_V = 3.86*u.deg  # relative to sun equator
L_V = 76.68069*u.deg
w_V = 131.53298*u.deg - L_V
M_V = true_anomaly_from_mean(181.97973*u.deg - w_V - L_V, e_V)
VenusianPlanet = eep.simulate.KeplerianExoplanet(semimajor_axis=a_V,
                                                 eccentricity=e_V,
                                                 inclination=i_V,
                                                 longitude=L_V,
                                                 periapsis_arg=w_V,
                                                 initial_anomaly=M_V,
                                                 albedo=planet_albedo,
                                                 point_color='orange', path_color='darkgoldenrod',
                                                 name='Venusian')
MyStar.add_exoplanet(VenusianPlanet)

#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   #
a_E = 1.00000011*u.au
e_E = 0.01671022
i_E = 7.155*u.deg  # relative to sun equator
L_E = -11.26064*u.deg
w_E = 102.94719*u.deg - L_E
M_E = true_anomaly_from_mean(100.46435*u.deg - w_E - L_E, e_E)
EarthyPlanet = eep.simulate.KeplerianExoplanet(semimajor_axis=a_E,
                                               eccentricity=e_E,
                                               inclination=i_E,
                                               longitude=L_E,
                                               periapsis_arg=w_E,
                                               initial_anomaly=M_E,
                                               albedo=planet_albedo,
                                               point_color='b', path_color='lightskyblue',
                                               name='Earthy')
MyStar.add_exoplanet(EarthyPlanet)

#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   #
a_M2 = 1.52366231*u.au
e_M2 = 0.09341233
i_M2 = 5.65*u.deg  # relative to sun equator
L_M2 = 49.57854*u.deg
w_M2 = 336.04084*u.deg - L_M2
M_M2 = true_anomaly_from_mean(355.45332*u.deg - w_M2 - L_M2, e_M2)
MarsyPlanet = eep.simulate.KeplerianExoplanet(semimajor_axis=a_M2,
                                              eccentricity=e_M2,
                                              inclination=i_M2,
                                              longitude=L_M2,
                                              periapsis_arg=w_M2,
                                              initial_anomaly=M_M2,
                                              albedo=planet_albedo,
                                              point_color='darkred', path_color='salmon',
                                              name='Martian')
MyStar.add_exoplanet(MarsyPlanet)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


EarthyPlanet.about_orbit()

eep.visualize.render_3d_planetary_system(MyStar)
