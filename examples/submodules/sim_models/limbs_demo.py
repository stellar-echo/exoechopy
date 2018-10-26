

"""Shows how the Limb module functions through examples."""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from exoechopy.simulate import Limb


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def run():
    fig, ax = plt.subplots(figsize=(8, 6))

    angle_array = np.linspace(-np.pi, np.pi, 1000) * u.rad
    sun_radius = (1 * u.R_sun).to(u.au)

    ax.set_xlabel("Angle on star surface")
    ax.set_ylabel("Relative star intensity")
    plt.suptitle("Approximate star intensity as a function of angle", y=1)

    #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  #

    uniform_limb = Limb(limb_model='uniform')
    uniform_limb_infinity_vals = [uniform_limb.calculate_limb_intensity(ai) for ai in angle_array]

    ax.plot(angle_array, uniform_limb_infinity_vals, color='gray', lw=1, ls='--', label="Sharp limb, infinity")

    #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  #

    quadratic_limb = Limb(limb_model='quadratic', coeffs=[.1, .3])
    infinity_limb_vals = [quadratic_limb.calculate_limb_intensity(ai) for ai in angle_array]
    ax.plot(angle_array, infinity_limb_vals, color='k', lw=1, label="Infinity quadratic limb")

    #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  #
    parker_probe_min_distance = 0.041 * u.au
    parker_limb = [quadratic_limb.calculate_limb_intensity(angle=ai,
                                                           star_radius_over_distance=
                                                           (sun_radius / parker_probe_min_distance).value)
                   for ai in angle_array]
    ax.plot(angle_array, parker_limb, color='r', lw=1, label="Quadratic, Parker probe min dist")

    #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  #
    close_dist = 0.01 * u.au
    zero_limb = [uniform_limb.calculate_limb_intensity(angle=ai,
                                                       star_radius_over_distance=(sun_radius / close_dist).value)
                 for ai in angle_array]
    ax.plot(angle_array, zero_limb, color='salmon', lw=1, ls='--', label="Sharp limb, 0.01au")

    #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  #
    mercury_distance = 0.39 * u.au
    mercury_limb = [quadratic_limb.calculate_limb_intensity(angle=ai,
                                                            star_radius_over_distance=
                                                            (sun_radius/mercury_distance).value)
                    for ai in angle_array]
    ax.plot(angle_array, mercury_limb, color='darkred', lw=1, label="Quadratic from Mercury")

    #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  #
    earth_distance = 1 * u.au
    earth_limb = [quadratic_limb.calculate_limb_intensity(angle=ai,
                                                          star_radius_over_distance=(sun_radius/earth_distance).value)
                  for ai in angle_array]
    ax.plot(angle_array, earth_limb, color='b', lw=1, label="Quadratic from Earth")

    #  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  #

    # ax.set_aspect(2 * np.pi)
    ax.legend(ncol=2, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)

    plt.subplots_adjust(left=.1, top=.8, right=.95)

    plt.show()

    #  =============================================================  #
    fig, ax = plt.subplots(figsize=(8, 6))

    # Also try logarithmic, exponential, or a custom function
    limb_options = ['uniform', 'linear', 'quadratic', 'square-root',
                    'power2', 'nonlinear']

    coeff_list = [[], [.3], [.1, .3], [.1, .3],
                  [.4, .8], [.5, .1, .1, -.1]]

    radial_value_array = np.linspace(-1, 1, 1000)

    for limb_type, coeffs in zip(limb_options, coeff_list):
        current_limb = Limb(limb_model=limb_type, coeffs=coeffs)
        ax.plot(radial_value_array, current_limb.limb_darkened_radial_position(radial_value_array), label=limb_type)
    ax.set_ylim(0)
    ax.legend(loc='best')
    ax.set_xlabel("Radial position (r/R)")
    ax.set_ylabel("Relative star intensity")
    ax.set_title("Relative star intensity by radial position")

    plt.show()

# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #


if __name__ == "__main__":

    run()
