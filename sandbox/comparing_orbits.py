"""Display the various orbital parameters and how they affect the orbits."""

import numpy as np

from astropy import units as u
import astropy.constants as const

import exoechopy as eep
import matplotlib.pyplot as plt


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def run():
    star_mass = .6 * u.M_sun
    star_radius = .6 * u.R_sun

    #  Initialize an exoplanet as KeplerianExoplanet to get an approximately Keplerian orbit:
    e_1 = 0.            # eccentricity
    a_1 = 0.03 * u.au   # semimajor axis
    i_1 = 0 * u.deg     # inclination
    L_1 = 0 * u.deg     # longitude
    w_1 = 0 * u.deg     # arg of periapsis
    m0_1 = 0 * u.deg    # initial anomaly
    planet_mass = 30. * u.M_jup

    name_1 = "Exo1"

    print("Separation in star radii: ", (a_1/star_radius).decompose())

    #  =============================================================  #

    Planet1 = eep.simulate.KeplerianExoplanet(semimajor_axis=a_1,
                                              eccentricity=e_1,
                                              inclination=i_1,
                                              longitude=L_1,
                                              periapsis_arg=w_1,
                                              initial_anomaly=m0_1,
                                              point_color='k', path_color='dimgray',
                                              name=name_1,
                                              mass=planet_mass,
                                              star_mass=star_mass+planet_mass)


    planet_pos, star_pos, planet_vel, star_vel = eep.simulate.reduced_keplerian_to_2body(Planet1,
                                                                                         planet_mass,
                                                                                         star_mass)

    Planet1.position = planet_pos
    Planet1.velocity = planet_vel

    MyStar1 = eep.simulate.Star(mass=star_mass, radius=star_radius,
                                position=star_pos, velocity=star_vel,
                                name="My Star 1", point_color='saddlebrown', path_color='burlywood')
    # Initialize the solver:
    y6solver = eep.simulate.SymplecticSolver(MyStar1, Planet1, dt=100 * u.s)
    # Run the solver:
    period = (2 * np.pi * np.sqrt(a_1 ** 3 / (const.G * (star_mass+planet_mass)))).decompose()
    y6solver.calculate_orbits(2*period, steps_per_save=10)

    # Visualize results:
    # Since we have two stars, we need to hold them in an appropriate container for processing purposes:

    time_domain = MyStar1._time_domain
    star_positions = np.transpose(MyStar1._all_positions)
    planet_positions = np.transpose(Planet1._all_positions)

    plt.plot(time_domain/period, (star_positions[0]/star_radius).decompose(), color='b', lw=1, ls='-', label="x-pos")
    plt.plot(time_domain/period, (star_positions[1]/star_radius).decompose(), color='r', lw=1, ls='--', label="y-pos")
    plt.plot(time_domain/period, (star_positions[2] / star_radius).decompose(), color='g', lw=1, ls='-.', label="z-pos")
    plt.xlabel("Time (t/T)")
    plt.ylabel("Displacement (stellar radii)")
    plt.title("Star displacement in stellar radii vs time")
    plt.legend()
    plt.show()

    planet_star_separation = planet_positions-star_positions

    plt.plot(time_domain/period, 100*(a_1.value-np.linalg.norm(planet_star_separation, axis=0))/a_1.value,
             color='k', lw=1)
    plt.title("Planet-star separation vs time")
    plt.xlabel("Time (t/T)")
    plt.ylabel("Relative error (%)")
    plt.show()

    Planet2 = eep.simulate.KeplerianExoplanet(semimajor_axis=a_1,
                                              eccentricity=e_1,
                                              inclination=i_1,
                                              longitude=L_1,
                                              periapsis_arg=w_1,
                                              initial_anomaly=m0_1,
                                              point_color='k', path_color='dimgray',
                                              name=name_1,
                                              mass=planet_mass,
                                              star_mass=star_mass+planet_mass)
    keplerian_positions = np.transpose(np.array([Planet2.calc_xyz_at_time(ti) for ti in time_domain]))

    plt.plot(time_domain / period, keplerian_positions[0] - planet_star_separation[0].to(u.au).value,
             color='b', lw=1, ls='-', label="x-err")
    plt.plot(time_domain / period, keplerian_positions[1] - planet_star_separation[1].to(u.au).value,
             color='r', lw=1, ls='--', label="y-err")
    plt.plot(time_domain / period, keplerian_positions[2] - planet_star_separation[2].to(u.au).value,
             color='g', lw=1, ls='-.', label="z-err")
    plt.title("Difference between keplerian approximation and integration")
    plt.show()

    #  =============================================================  #

    star_system = eep.simulate.MultiStarSystem()
    star_system.add_orbiting_object(MyStar1)
    star_system.add_orbiting_object(Planet1)

    eep.visualize.animate_3d_planetary_system(star_system)

    eep.visualize.precomputed_orbit_plot(Planet1)

    eep.visualize.precomputed_system_plot(Planet1, MyStar1)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #

if __name__ == '__main__':
    run()

