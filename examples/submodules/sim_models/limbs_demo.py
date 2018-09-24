

"""Shows how the Limb module functions through examples."""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from exoechopy.simulate import limbs


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def run():
    fig, (ax1, ax2) = plt.subplots(1, 2)

    angle_array = np.linspace(-np.pi, np.pi, 1000) * u.rad
    sun_radius = (1 * u.R_sun).to(u.au)

    ax1.set_xlabel("Angle on star surface")
    ax1.set_ylabel("Relative star intensity")
    ax1.set_title("Solar intensity approx. by angle", y=1.5)

    #  =============================================================  #
    zero_limb_infinity = [limbs.no_limb_darkening(angular_position=ai)
                          for ai in angle_array]
    ax1.plot(angle_array, zero_limb_infinity, color='gray', lw=1, ls='--', label="Sharp limb, inf")

    #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  #

    infinity_limb = [limbs.calculate_basic_limb_darkening(angular_position=ai,
                                                          star_radius_over_distance=0)
                     for ai in angle_array]
    ax1.plot(angle_array, infinity_limb, color='k', lw=1, label="Infinity")

    #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  #
    parker_probe_min_distance = 0.041 * u.au
    parker_limb = [limbs.calculate_basic_limb_darkening(angular_position=ai,
                                                        star_radius_over_distance=(
                                                        sun_radius / parker_probe_min_distance).value)
                   for ai in angle_array]
    ax1.plot(angle_array, parker_limb, color='r', lw=1, label="Parker probe min dist")

    #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  #
    close_dist = 0.01 * u.au
    zero_limb = [limbs.no_limb_darkening(angular_position=ai,
                                         star_radius_over_distance=(
                                             sun_radius / close_dist).value)
                 for ai in angle_array]
    ax1.plot(angle_array, zero_limb, color='salmon', lw=1, ls='--', label="Sharp limb, 0.01au")

    #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  #
    mercury_distance = 0.39 * u.au
    mercury_limb = [limbs.calculate_basic_limb_darkening(angular_position=ai,
                                                         star_radius_over_distance=(sun_radius/mercury_distance).value)
                    for ai in angle_array]
    ax1.plot(angle_array, mercury_limb, color='darkred', lw=1, label="Mercury")

    #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  #
    earth_distance = 1 * u.au
    earth_limb = [limbs.calculate_basic_limb_darkening(angular_position=ai,
                                                       star_radius_over_distance=(sun_radius/earth_distance).value)
                  for ai in angle_array]
    ax1.plot(angle_array, earth_limb, color='b', lw=1, label="Earth")

    #  =============================================================  #

    radial_value_array = np.linspace(-1, 1, 1000)
    ax2.plot(radial_value_array, limbs.limb_darkened_radial_position(radial_value_array), color='k')
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Radial position (r/R)")
    ax2.set_ylabel("Relative star intensity")
    ax2.set_title("Relative star intensity by radial position", y=1.1)

    #  =============================================================  #

    ax1.set_aspect(2 * np.pi)
    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0.)

    ax2.set_aspect(2)

    plt.tight_layout()
    plt.show()


# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #

if __name__ == "__main__":

    run()
