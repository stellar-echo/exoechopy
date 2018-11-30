
"""Display the various orbital parameters and how they affect the orbits."""

import numpy as np

from astropy import units as u
import astropy.constants as const

import exoechopy as eep
from exoechopy.utils import u_str


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def run():

    print("""In addition to simple Keplerian motion, exoechopy has a 6th order symplectic physics solver.
    This allows more complete simulation, including the ability to estimate Doppler parameters more accurately.
    
    In the Keplerian motion, the star is not treated as moving, but the SymplecticSolver accurately models
    the star wobble.  
    Or stars.  The engine is not optimized for large numbers of bodies, but N<10 is fairly straightforward.
    
    In this first example, we show a pair of equal mass stars orbiting each other.  
    """)

    star_mass = .6*u.M_sun
    a0 = 0.05*u.au
    period = (2*np.pi*np.sqrt((2*a0)**3/(const.G*2*star_mass))).decompose()
    v0 = 2*np.pi*a0/period

    star_1_pos = u.Quantity(np.array((a0.value, 0, 0)), a0.unit)
    star_2_pos = u.Quantity(np.array((-a0.value, 0, 0)), a0.unit)
    star_1_vel = u.Quantity(np.array((0, v0.value, 0)), v0.unit).decompose()
    star_2_vel = u.Quantity(np.array((0, -v0.value, 0)), v0.unit).decompose()

    MyStar1 = eep.simulate.Star(mass=star_mass, radius=.6*u.R_sun,
                                position=star_1_pos, velocity=star_1_vel,
                                name="My Star 1", point_color='saddlebrown', path_color='burlywood')
    MyStar2 = eep.simulate.Star(mass=star_mass, radius=.4 * u.R_sun,
                                position=star_2_pos, velocity=star_2_vel,
                                name="My Star 2", point_color='mediumslateblue', path_color='powderblue')

    # Initialize the solver:
    y6solver = eep.simulate.SymplecticSolver(MyStar1, MyStar2, dt=100*u.s, steps_per_save=10)
    # Run the solver:
    y6solver.calculate_orbits(period*2)

    # Visualize results:
    # Since we have two stars, we need to hold them in an appropriate container for processing purposes:
    star_system = eep.simulate.MultiStarSystem()
    star_system.add_orbiting_object(MyStar1)
    star_system.add_orbiting_object(MyStar2)

    eep.visualize.animate_3d_planetary_system(star_system)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

    print("""It can be helpful to generate Keplerian orbits as initial conditions for planets.
    While the Keplerian parameters will not remain exact in the SymplecticSolver, 
    it simplifies the system generation process.
    """)

    #  Initialize an exoplanet as KeplerianExoplanet to get an approximately Keplerian orbit:
    e_1 = 0.25          # eccentricity
    a_1 = 0.03 * u.au   # semimajor axis
    i_1 = 45*u.deg      # inclination
    L_1 = 0*u.deg       # longitude
    w_1 = 0*u.deg      # arg of periapsis
    m0_1 = 0*u.deg      # initial anomaly
    planet_mass = 30.*u.M_jup
    name_1 = "Exo_1"
    Planet1 = eep.simulate.KeplerianExoplanet(semimajor_axis=a_1,
                                              eccentricity=e_1,
                                              inclination=i_1,
                                              longitude=L_1,
                                              periapsis_arg=w_1,
                                              initial_anomaly=m0_1,
                                              point_color='k', path_color='dimgray',
                                              name=name_1,
                                              mass=planet_mass,
                                              parent_mass=star_mass + planet_mass)

    # Helper function to convert the Keplerian orbit into Cartesian coordinates
    planet_pos, star_pos, planet_vel, star_vel = eep.simulate.reduced_keplerian_to_2body(Planet1,
                                                                                         planet_mass,
                                                                                         star_mass)

    Planet1.position = planet_pos
    Planet1.velocity = planet_vel

    MyStar1.position = star_pos
    MyStar1.velocity = star_vel

    star_system = eep.simulate.MultiStarSystem()
    star_system.add_orbiting_object(MyStar1)
    star_system.add_orbiting_object(Planet1)

    #  =============================================================  #

    # Initialize the solver:
    y6solver = eep.simulate.SymplecticSolver(MyStar1, Planet1, dt=100*u.s, steps_per_save=20)

    period = (2 * np.pi * np.sqrt(a_1 ** 3 / (const.G * (star_mass+planet_mass)))).decompose()

    # Run the solver:
    y6solver.calculate_orbits(2*period)

    eep.visualize.animate_3d_planetary_system(star_system)

    eep.visualize.precomputed_orbit_plot(Planet1)

    eep.visualize.precomputed_system_plot(Planet1, MyStar1)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

    print("""Complex multi-star, multi-planet systems can be handled with minimal additional effort.
    """)

    # Reset stars
    MyStar1 = eep.simulate.Star(mass=star_mass, radius=.6*u.R_sun,
                                position=star_1_pos, velocity=star_1_vel,
                                name="My Star 1", point_color='saddlebrown', path_color='burlywood')
    MyStar2 = eep.simulate.Star(mass=star_mass, radius=.4 * u.R_sun,
                                position=star_2_pos, velocity=star_2_vel*.8,
                                name="My Star 2", point_color='mediumslateblue', path_color='powderblue')

    Planet1.position += MyStar1.position
    Planet1.velocity += MyStar1.velocity
    Planet1.mass = 1*u.M_jup

    star_system = eep.simulate.MultiStarSystem()
    star_system.add_orbiting_object(MyStar1)
    star_system.add_orbiting_object(MyStar2)
    star_system.add_orbiting_object(Planet1)

    #  =============================================================  #

    # Initialize the solver:
    y6solver = eep.simulate.SymplecticSolver(MyStar1, MyStar2, Planet1, dt=100*u.s, steps_per_save=20)

    period = (2 * np.pi * np.sqrt((2*a0)**3 / (const.G * (2*star_mass+planet_mass)))).decompose()

    # Run the solver:
    y6solver.calculate_orbits(2*period)

    eep.visualize.animate_3d_planetary_system(star_system)

    eep.visualize.precomputed_system_plot(Planet1, MyStar1, MyStar2)


# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #

if __name__ == "__main__":
    run()

