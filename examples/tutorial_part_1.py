
"""Generate and analyze some flares and echoes from exoplanets."""

import examples.submodules.sim_models.flares_demo as flare_demo
import examples.submodules.sim_models.active_regions_demo as active_regions_demo
import examples.submodules.sim_models.limbs_demo as limbs_demo
import examples.submodules.sim_models.star_system_demo as star_system_demo
import examples.submodules.sim_models.nonkeplerian_orbit_demo as nonkeplerian_demo
import examples.submodules.sim_models.telescope_demo as telescope_demo
import examples.submodules.math_methods.autocorr_demos as autocorr_demos
import examples.submodules.echo_demo_1 as echo_demo_1
import examples.submodules.echo_demo_2 as echo_demo_2

print("""
ExoEchoPy provides modeling and analysis tools to support the development of a new exoplanet hunting technique.
Flares echo off planets, creating a delayed copy of the flare in the lightcurve.
To model these systems, we must generate synthetic flares and synthetic echoes in a synthetic light curves.
The 'simulate' package provides the requisite tools for producing synthetic light curves.

#  =============================================================  #
The simulate.flares.flares.py module demo:
""")

flare_demo.run()

print("""
Collections of flares are handled by the active_regions module, which enables statistical distributions 
in intensity, character, and location on the star.

#  =============================================================  #
The simulate.flares.active_regions.py module demo:
""")

active_regions_demo.run()

print("""
Active regions are built on Stars, which have limb darkening that is caused by optical opacity of the star.
To handle the limb of the star, the limb module is used.

#  =============================================================  #
The simulate.limbs.py module demo:
""")

limbs_demo.run()

print("""
Stars are also able to support multiple orbiting exoplanets.  
The simplest systems use Keplerian physics and do not cause the stars to move

#  =============================================================  #
The simulate.orbital_physics.keplerian.py module demo:
""")

star_system_demo.run()

print("""
Using the SymplecticSolver, it's possible to handle more complex systems including multi-planet and multi-star options.

#  =============================================================  #
The simulate.orbital_physics.solvers.py module demo:
""")

nonkeplerian_demo.run()

print("""
The telescope module provides classes for generating data from stars, active regions, and planets.

#  =============================================================  #
The simulate.telescopes.py module demo:
""")

telescope_demo.run()

print("""
The 'analyze' subpackage has methods for working with flares and light curves.

#  =============================================================  #
The analyze.autocorrelation.py module demo:
""")

autocorr_demos.run()

print("""
Using these modules, it's possible to create and analyze synthetic light curves.
The echo_demo_1 shows an approach to getting started, as well as illustrating why the analysis is challenging.

#  =============================================================  #
The examples.submodules.echo_demo_1.py demo:
""")

echo_demo_1.run()

print("""
The echo_demo_2 shows a slightly more realistic scenario, as well as illustrating why the analysis gets harder.

#  =============================================================  #
The examples.submodules.echo_demo_2.py demo:
""")

echo_demo_2.run()

print("""
More to come!
""")
