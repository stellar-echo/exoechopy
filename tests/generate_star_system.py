
"""First test"""

import numpy as np
import exoechopy as eep
from astropy import units as u

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

spectral_band = eep.utils.spectral.JohnsonPhotometricBand('U')
emission_type = eep.utils.spectral.SpectralEmitter(spectral_band, 16)

MyStar = eep.simulate.Star(mass=1*u.M_sun, radius=1.*u.R_sun, spectral_type=emission_type)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

planet_albedo = eep.utils.spectral.Albedo(spectral_band, 1.)


## Need to convert the initial_anomaly from mean anomaly into true anomaly

MercurialPlanet = eep.simulate.KeplerianExoplanet(semimajor_axis=0.38709893 * u.au,
                                                  eccentricity=0.20563069,
                                                  inclination=3.38*u.deg,
                                                  longitude=48.33*u.deg,
                                                  periapsis_arg=(77.46-48.33)*u.deg,
                                                  initial_anomaly=(252.25-77.46)*u.deg,
                                                  albedo=planet_albedo,
                                                  point_color='.1', path_color='.5')
MyStar.add_exoplanet(MercurialPlanet)


VenusyPlanet = eep.simulate.KeplerianExoplanet(semimajor_axis=0.723332 * u.au,
                                               eccentricity=0.006772,
                                               inclination=3.86*u.deg,
                                               longitude=76.67*u.deg,
                                               periapsis_arg=(131.77-76.67)*u.deg,
                                               initial_anomaly=(181.98-131.77)*u.deg,
                                               albedo=planet_albedo,
                                               point_color='orange', path_color='darkgoldenrod')
MyStar.add_exoplanet(VenusyPlanet)


EarthyPlanet = eep.simulate.KeplerianExoplanet(semimajor_axis=1.000001018 * u.au,
                                               eccentricity=0.0167086,
                                               inclination=7.155*u.deg,
                                               longitude=-11.26*u.deg,
                                               periapsis_arg=(102.93+11.26)*u.deg,
                                               initial_anomaly=(100.46-102.93)*u.deg,
                                               albedo=planet_albedo,
                                               point_color='darkblue', path_color='lightskyblue')
MyStar.add_exoplanet(EarthyPlanet)


MarsyPlanet = eep.simulate.KeplerianExoplanet(semimajor_axis=1.52366231 * u.au,
                                              eccentricity=0.09341233,
                                              inclination=5.65*u.deg,
                                              longitude=49.58*u.deg,
                                              periapsis_arg=(336.04-49.58)*u.deg,
                                              initial_anomaly=(355.45-336.04)*u.deg,
                                              albedo=planet_albedo,
                                              point_color='darkred', path_color='salmon')
MyStar.add_exoplanet(MarsyPlanet)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


EarthyPlanet.about_orbit()

eep.visualize.render_3d_planetary_system(MyStar)
