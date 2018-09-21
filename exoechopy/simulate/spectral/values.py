
"""
This module provides specific spectral band properties.
"""

from astropy import units as u

johnson_band_center_dict = {"U": 360 * u.nm,
                            "B": 440 * u.nm,
                            "V": 550 * u.nm,
                            "R": 640 * u.nm,
                            "I": 790 * u.nm,
                            "J": 1260 * u.nm,
                            "H": 1600 * u.nm,
                            "K": 2220 * u.nm,
                            "g": 520 * u.nm,
                            "r": 670 * u.nm,
                            "i": 790 * u.nm,
                            "z": 910 * u.nm
                            }

johnson_bandwidth_dict = {"U": 54 * u.nm,
                          "B": 97 * u.nm,
                          "V": 88 * u.nm,
                          "R": 147 * u.nm,
                          "I": 150 * u.nm,
                          "J": 202 * u.nm,
                          "H": 368 * u.nm,
                          "K": 511 * u.nm,
                          "g": 73 * u.nm,
                          "r": 94 * u.nm,
                          "i": 126 * u.nm,
                          "z": 118 * u.nm
                          }

johnson_band_flux_dict = {"U": 1810 * u.Jy,
                          "B": 4260 * u.Jy,
                          "V": 3640 * u.Jy,
                          "R": 3080 * u.Jy,
                          "I": 2550 * u.Jy,
                          "J": 1600 * u.Jy,
                          "H": 1080 * u.Jy,
                          "K": 670 * u.Jy,
                          "g": 3730 * u.Jy,
                          "r": 4490 * u.Jy,
                          "i": 4760 * u.Jy,
                          "z": 4810 * u.Jy
                          }
