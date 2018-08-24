
"""First test"""

import numpy as np
import exoechopy as eep
from astropy import units as u

spectral_band = eep.utils.spectral.JohnsonPhotometricBand('U')

emission_type = eep.utils.spectral.SpectralEmitter(spectral_band, 16)
MyStar = eep.simulate.Star(mass=1*u.M_sun, radius=1.*u.R_sun, spectral_type=emission_type)

planet_albedo = eep.utils.spectral.Albedo(spectral_band, 1.)
MyPlanet = eep.simulate.KeplerianExoplanet(semimajor_axis=1.*u.au, eccentricity=.6, inclination=np.pi/4*u.rad,
                                           albedo=planet_albedo)

MyStar.add_exoplanet(MyPlanet)

eep.visualize.render_3d_planetary_system(MyStar)
