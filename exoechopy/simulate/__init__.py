
"""
This subpackage provides functionality for simulating stars with variability, including starspots and flares.
Functionality is also provided to add echoes and interactions with exoplanets and structures such as accretion disks.
Packages contains different classes and models for stars, exoplanets, and observation platforms.
"""

from .flares import *
from .orbital_physics import *
from .spectral import *

from .limbs import *
from .planets import *
from .stars import *
from .telescopes import *

from . import methods
