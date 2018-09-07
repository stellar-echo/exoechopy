
"""
This module provides functionality for modeling flare light curve profiles.
"""

import warnings
import numpy as np
from astropy import units as u
from astropy.utils.exceptions import AstropyUserWarning

__all__ = []

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class DeltaFlare:
    """A flare that occurs in a single time bin"""
    def __init__(self, counts=None, **kwargs):
        """
        Adds counts to a single time bin.

        :param int counts:
        """
        if counts:
            self._counts = float(counts)
        else:
            self._counts = None

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def integrated_counts(self):
        """
        Total integrated counts of the flare
        :return float:
        """
        return self._counts


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class ExponentialFlare1(DeltaFlare):
    """A flare with a quadratic onset and exponential decay and C0 continuity."""
    def __init__(self, onset, decay, max_decay=10, **kwargs):
        """
        Standard impulsive flare with a sharp onset and exponential decay.
        Multiply values by intensity to scale for different scenarios

        :param u.Quantity onset: Onset time
        :param u.Quantity decay: Exponential decay constant, exp(-t/decay)
        :param float max_decay: How many decay constants to include-- influences the C1 discontinuity at the end of tail
        """

        super().__init__(**kwargs)

        if isinstance(onset, u.Quantity):
            self._onset = onset.to(u.s).value
        else:
            self._onset = u.Quantity(onset, u.s).value
            warnings.warn("Casting onset, input as " + str(onset) + ", to seconds", AstropyUserWarning)

        if isinstance(decay, u.Quantity):
            self._decay = decay.to(u.s).value
        else:
            self._decay = u.Quantity(decay, u.s).value
            warnings.warn("Casting decay, input as " + str(decay) + ", to seconds", AstropyUserWarning)

        if self._decay < self._onset:
            warnings.warn("Decay, " + str(decay) + ", shorter than onset, " + str(onset), AstropyUserWarning)

        self._decay_duration = self._decay*max_decay
        self._epsilon = np.exp(-self._decay_duration/self._decay)

        # Total counts, as determined through analytical integration of the piecewise flare function:
        self._counts = self._onset/3. + self._decay - self._decay_duration*self._epsilon/(1.-self._epsilon)

    # ------------------------------------------------------------------------------------------------------------ #
    def evaluate_at_time(self, time):
        if isinstance(time, u.Quantity):
            _time = time.to(u.s).value
        else:
            _time = u.Quantity(time, u.s).value
            warnings.warn("Casting time, input as " + str(time) + ", to seconds", AstropyUserWarning)
        return self._evaluate_at_time_lw(_time)*u.ct/u.s

    def _evaluate_at_time_lw(self, t):
        """
        Unitless version of actual flare function.
        :param float t: Time in seconds
        :return float: Ct/s
        """
        if 0 < t < self._onset + self._decay_duration:
            if t <= self._onset:
                return np.square(t/self._onset)
            else:
                return (np.exp(-(t-self._onset)/self._decay)-self._epsilon)/(1.-self._epsilon)
        else:
            return 0.

    def evaluate_over_array_lw(self, time_array, cadence=None):
        # Need to implement with analytical integration between data points, do not handle units
        if cadence:  # Means constant time step between points
            raise NotImplementedError
        else:  # Means explicitly evaluate time step between points
            raise NotImplementedError


##############################################  TEST & DEMO CODE  ######################################################


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.cm as mplcm
    color_map = mplcm.copper_r

    onset = 3.5 * u.s
    decay = 4 * u.s
    time_domain = np.linspace(-onset.value, (onset + 11 * decay).value, 1000) * u.s

    fig, (ax_deriv, ax_plt) = plt.subplots(2, 1)

    max_decay_list = [2, 4, 6, 10]
    for max_num_decay in max_decay_list:
        MyFlare = ExponentialFlare1(onset=onset, decay=decay, max_decay=max_num_decay)
        print('MyFlare total counts: ', MyFlare.integrated_counts)
        flare_intensity_list = np.array([MyFlare.evaluate_at_time(ti).value for ti in time_domain])
        flare_intensity_derivative = np.diff(flare_intensity_list)/np.diff(time_domain)
        ax_plt.plot(time_domain, flare_intensity_list,
                    color=color_map(max_num_decay/max(max_decay_list)),
                    lw=1, label="max_decay="+str(max_num_decay))
        ax_deriv.plot(time_domain[:-1], flare_intensity_derivative,
                      color=color_map(max_num_decay/max(max_decay_list)),
                      lw=1)

    ax_plt.set_ylabel("Flare intensity (ct/s)")
    ax_plt.legend()
    ax_plt.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)

    ax_deriv.set_xlabel("Time (s)")
    ax_deriv.set_ylabel("Diff[Flare intensity (ct/s)]")

    plt.suptitle("Prototypical ExponentialFlare1")
    plt.tight_layout()
    plt.show()
