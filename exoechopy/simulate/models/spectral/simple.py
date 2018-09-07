
"""
This module provides support for simple static spectra, including single-band containers and methods.
Includes emission and reflection objects.
"""

import warnings
from astropy import units as u
from astropy.utils.exceptions import AstropyUserWarning
from .values import *
from .maths import *
from exoechopy.utils.globals import *


__all__ = ['SpectralBand', 'JohnsonPhotometricBand', 'SpectralEmitter', 'Albedo']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class SpectralBand:
    """
    Base class for a single spectral band.
    """

    # ------------------------------------------------------------------------------------------------------------ #
    def __init__(self,
                 wavelength=None,
                 bandwidth=None,
                 flux_m0=None):
        """
        The SpectralBand class provides a single spectral band.

        :param u.Quantity wavelength: Center wavelength of the band
        :param u.Quantity bandwidth: Width of band
        :param u.Quantity flux_m0: Flux of band at m=0
        """
        self._band = None
        if wavelength is None and flux_m0 is None:
            # In future, may have some reasons to provide an empty spectral container, but not currently.
            raise ValueError("No wavelength provided.")
        elif wavelength is None and flux_m0 is not None:
            if isinstance(flux_m0, u.Quantity):
                self._flux_m0 = flux_m0.to(u.ct/u.s/u.m**2)
                self._wavelength = None
                self._bandwidth = None
            else:
                self._flux_m0 = flux_m0 * u.ct/u.s/u.m**2
                warnings.warn("Casting flux_m0, input as "+str(flux_m0)+", to ct/s/m²", AstropyUserWarning)
        else:
            if isinstance(wavelength, u.Quantity):
                self._wavelength = wavelength
            else:
                self._wavelength = wavelength * u.nm
                warnings.warn("Casting wavelength, input as "+str(wavelength)+", to nm", AstropyUserWarning)
            if bandwidth is None:
                self._bandwidth = self._wavelength / 5
                warnings.warn("No bandwidth provided, using"+self._bandwidth.tostring(), AstropyUserWarning)
            else:
                if isinstance(bandwidth, u.Quantity):
                    self._bandwidth = bandwidth
                else:
                    self._bandwidth = bandwidth * u.nm
                    warnings.warn("Casting bandwidth, input as " + str(bandwidth) + ", to nm", AstropyUserWarning)

    # ------------------------------------------------------------------------------------------------------------ #
    def get_flux(self, magnitude):
        """
        Returns the flux in photons/sec for the given band
        :param float magnitude: star magnitude
        :return float: Photons/sec
        """
        if self._wavelength is None:  # flux is ct/s, no spectral dependencies
            return self._flux_m0 * 10**(-.4*magnitude)
        else:  # flux has a spectral dependence
            return self._flux_m0.to(u.photon/u.m**2/u.s/u.nm, equivalencies=u.spectral_density(self._wavelength)) \
                   * self._bandwidth * 10**(-.4*magnitude)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class JohnsonPhotometricBand(SpectralBand):
    """
    Base class for a single Johnson photometric band, built on the SpectralBand, with a built-in library
    """
    def __init__(self, band):
        """
        The PhotometricBand references the common pseudo-standard photometric series (UBVgR).

        :param str band: Which band to use, e.g., 'U', and is case-sensitive
        """
        if band in johnson_band_center_dict:
            self._band = band
            SpectralBand.__init__(self,
                                  johnson_band_center_dict[self._band],
                                  johnson_bandwidth_dict[self._band],
                                  johnson_band_flux_dict[self._band]
                                  )
        else:
            raise TypeError("Band "+str(band)+" is unknown.")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class SpectralEmitter:
    """
    Base class for a spectral source.
    """

    # ------------------------------------------------------------------------------------------------------------ #
    def __init__(self, spectra, magnitude, abs_mag=False):
        """
        Enables an object to emit a specific band at a given magnitude.

        :param SpectralBand spectra: SpectralBand class that describes its emission
        :param float magnitude: Magnitude of star, default of apparent magnitude
        :param bool abs_mag: Convert to absolute magnitude.  Not currently implemented.
        """
        if isinstance(spectra, SpectralBand):
            self._spectra = spectra
        if isinstance(magnitude, float):
            self._magnitude = magnitude

    def relative_flux(self):
        """Returns the relative flux from the SpectralEmitter"""
        return self._spectra.get_flux(self._magnitude)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class Albedo:
    """
    Base class for a spectral reflector.
    """

    # ------------------------------------------------------------------------------------------------------------ #
    def __init__(self, spectra, albedo, phase_function=None):
        """
        Enables an object to reflect a specific band at a given magnitude.

        :param SpectralBand spectra: SpectralBand class that describes its reflection
        :param float albedo: [0, 1]
        :param phase_function: Currently only supporting a Lambertian, later may create PhaseFunction classes
        """
        if isinstance(spectra, SpectralBand):
            self._spectra = spectra
        if isinstance(albedo, float):
            self._albedo = albedo
        self._phase_function = "Lambertian"

    # ------------------------------------------------------------------------------------------------------------ #
    def phase_law(self, angle):
        if self._phase_function == "Lambertian":
            return lambertian_phase_law(angle)

    # ------------------------------------------------------------------------------------------------------------ #
    def phase_law_bidirectional(self, angle_incident, angle_observed):
        """Placeholder for a future thing"""
        pass

    # ------------------------------------------------------------------------------------------------------------ #
    def phase_law_inhomogeneous(self, angle, albedo_map):
        """Placeholder for a future thing"""
        pass
