
"""Generate and analyze some flares and echoes from exoplanets."""

import examples.submodules.sim_models.flares_demo as flare_demo
import examples.submodules.sim_models.active_regions_demo as active_regions_demo
import examples.submodules.sim_models.limbs_demo as limbs_demo
import examples.submodules.sim_models.star_system_demo as star_system_demo
import examples.submodules.sim_models.telescope_demo as telescope_demo

print("""
ExoEchoPy provides modeling and analysis tools to support the development of a new exoplanet hunting technique.
Flares echo off planets, creating a delayed copy of the flare in the lightcurve.
To model these systems, we must generate synthetic flares and synthetic echoes in a synthetic light curves.
The 'simulate' package provides the requisite tools for producing synthetic light curves.
""")

flare_demo.run()

print("""
Collections of flares are handled by the active_regions module, which enables statistical distributions 
in intensity, character, and location on the star.
""")

active_regions_demo.run()

print("""
Active regions are built on Stars, which have limb darkening that is caused by optical opacity of the star.
To handle the limb of the star, the limb module is used.
""")

limbs_demo.run()

print("""
Stars are also able to support multiple orbiting exoplanets.  
These exoplanets can currently be driven only be Keplerian orbits.
""")

star_system_demo.run()

print("""
The telescope module provides classes for generating data from stars, active regions, and planets.
""")

telescope_demo.run()

