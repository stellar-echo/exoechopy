
import exoechopy as eep
from astropy import units as u
from astropy.constants import c
import numpy as np
import matplotlib.pyplot as plt
import time


def run():

    semimajor_axis = u.Quantity(0.05, 'au')
    eccentricity = 0.0
    star_mass = u.Quantity(1, 'M_sun')
    initial_anomaly = u.Quantity(0, 'rad')
    inclination = u.Quantity(.5, 'rad')
    longitude = u.Quantity(0, 'rad')
    periapsis = u.Quantity(0, 'rad')

    planet = eep.simulate.KeplerianOrbit(semimajor_axis=semimajor_axis,
                                         eccentricity=eccentricity,
                                         star_mass=star_mass,
                                         initial_anomaly=initial_anomaly,
                                         inclination=inclination,
                                         longitude=longitude,
                                         periapsis_arg=periapsis)

    star = eep.simulate.DeltaStar(mass=star_mass)

    high_resolution = 1000
    low_resolution = 20
    period = planet.orbital_period
    evaluate_times = u.Quantity(np.linspace(0, 4*period.to(u.s).value, high_resolution), 's')
    t0 = time.clock()
    xyz_positions = np.array([planet.calc_xyz_at_time(f_i) for f_i in evaluate_times])
    t1 = time.clock()
    approx_xyz = planet.evaluate_positions_at_times_lw(evaluate_times.value, num_points=low_resolution)
    t2 = time.clock()

    print("Exact calculation: ", t1-t0)
    print("Interpolated flare times: ", t2-t1)

    error = u.Quantity(np.linalg.norm(xyz_positions-approx_xyz, axis=1), 'au')
    plt.plot(evaluate_times, (error/c).to(u.s))
    plt.xlabel("Time (s)")
    plt.ylabel("Position error (seconds)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()