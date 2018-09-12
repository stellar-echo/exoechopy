
"""
This module provides functionality for modeling flare light curve profiles.
"""

import warnings
import numpy as np
from astropy import units as u
from astropy.utils.exceptions import AstropyUserWarning
from scipy import stats

__all__ = ['ProtoFlare', 'DeltaFlare', 'ExponentialFlare1']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class ProtoFlare:
    """Template class for flares, generates no signal."""
    def __init__(self):
        self._counts = None
        # Explicit definition of where the function is piecewise/discontinuous:
        self._discontinuities = None
        self.discontinuities = [0.]

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def integrated_counts(self):
        """
        Total integrated counts of the flare
        :return float:
        """
        return self._counts

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def discontinuities(self):
        return self._discontinuities

    @discontinuities.setter
    def discontinuities(self, discontinuities):
        if discontinuities:
            _discont = [x for x in discontinuities]
            _discont.sort()
            self._discontinuities = np.array(_discont)
        else:
            self._discontinuities = None

    # ------------------------------------------------------------------------------------------------------------ #
    def evaluate_over_array_lw(self, time_array):
        """
        Given a time domain, evaluate the flare.  This discretizes and 'analytically' integrates the time bins
        Need to re-implement with faster array handling, such as np.where().  For now, just getting it working correctly
        Each bin is integrated over [x_i, x_i+1).
        :param np.array time_array:
        :return np.array:
        """
        dt = time_array[1] - time_array[0]
        # Provide an end-point for the last bin to integrate over:
        edge_adjusted_time_array = np.append(time_array, time_array[-1]+dt)
        return np.array([self._integrate_between_times_lw(t0, t1)
                         for t0, t1 in zip(edge_adjusted_time_array[:-1], edge_adjusted_time_array[1:])])

    # ------------------------------------------------------------------------------------------------------------ #
    def _integrate_between_times_lw(self, t0, t1):
        """
        Evaluate the flare between times t0 and t0+dt in sec
        May want to upgrade to be more array-friendly, for now just trying to get it right
        :param float t0:
        :param float t1:
        :return float:
        """
        # Case 1: Both sides of integration are outside domain of flare function, nothing occurs
        if t1 <= self.discontinuities[0] or self.discontinuities[-1] < t0:
            return 0.
        # Case 2: Time step is longer than entire flare duration, can ignore piecewise complications
        elif t0 <= self.discontinuities[0] and self.discontinuities[-1] < t1:
            return self.integrated_counts
        # Case 3: Evaluate within the piecewise flare function itself
        else:
            return self._integrated_flare_lw(t1) - self._integrated_flare_lw(t0)

    # ------------------------------------------------------------------------------------------------------------ #
    def _integrated_flare_lw(self, t0):
        """
        Integrated flare, analytically pre-computed
        :param t0:
        :return:
        """
        return 0.

    # ------------------------------------------------------------------------------------------------------------ #
    def evaluate_at_time(self, time):
        """
        Evaluate the flare function at the given time
        :param u.Quantity time:
        :return u.Quantity: ct/sec
        """
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
        return 0.
    # ------------------------------------------------------------------------------------------------------------ #


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class DeltaFlare(ProtoFlare):
    """A flare that occurs in a single time bin"""
    def __init__(self, counts=None, **kwargs):
        """
        Adds counts to a single time bin.

        :param int counts:
        """
        super().__init__()

        if counts:
            self._counts = float(counts)
        else:
            self._counts = None

    # ------------------------------------------------------------------------------------------------------------ #
    def _integrated_flare_lw(self, t0):
        """
        Integrated flare, analytically pre-computed
        :param t0:
        :return:
        """
        if self.discontinuities[0] <= t0 < self.discontinuities[-1]:
            return self.integrated_counts
        else:
            return 0


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class ExponentialFlare1(ProtoFlare):
    """A flare with a quadratic onset and exponential decay and C0 continuity."""
    def __init__(self,
                 onset,
                 decay,
                 max_decay=10,
                 **kwargs):
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

        # Explicit definition of where the function is piecewise/discontinuous:
        self.discontinuities = [0, self._onset, self._onset+self._decay_duration]

        # Total counts, as determined through analytical integration of the piecewise flare function:
        self._counts = self._onset/3. + self._decay - self._decay_duration*self._epsilon/(1.-self._epsilon)

    # ------------------------------------------------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------------------------------------------------ #
    def _integrated_flare_lw(self, t0):
        """
        Analytically pre-computed flare integration
        :param float t0: time of evaluation in sec ('unitless')
        :return float:
        """
        if t0 < self.discontinuities[0]:
            return 0
        if self.discontinuities[0] <= t0 < self.discontinuities[1]:
            return np.power(t0, 3)/(3*self._onset**2)
        elif self.discontinuities[1] <= t0 < self.discontinuities[2]:
            return self._onset/3 \
                   + (self._decay - self._decay*np.exp(-(t0-self._onset)/self._decay) - (t0-self._onset)*self._epsilon)\
                   / (1-self._epsilon)
        else:
            return self.integrated_counts


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class MultipeakFlare(ProtoFlare):
    """A container for generating multiple sequential flares of the same type"""
    def __init__(self,
                 flare_type=None,
                 num_subflare_pdf=None,
                 **kwargs):
        """
        Creates a multi-peaked flare based on input parameters and a designated flare type
        :param DeltaFlare flare_type:
        :param stats.rv_discrete num_subflare_pdf: Distribution function for determining how many subflares to include
        :param kwargs:
        """
        super().__init__(**kwargs)

        if isinstance(flare_type, DeltaFlare):
            self._flare_type = flare_type
        if isinstance(num_subflare_pdf, int):
            self._num_flares = num_subflare_pdf
        elif isinstance(num_subflare_pdf, stats.rv_discrete):
            self._num_flares = num_subflare_pdf.rvs()
        raise NotImplementedError


##############################################  TEST & DEMO CODE  ######################################################


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.cm as mplcm
    color_map = mplcm.copper_r
    color_map2 = mplcm.winter

    # ------------------------------------------------------------------------------------------------------------ #
    print("""
    It is important to avoid discontinuities after the flare peak, many detection filters are sensitive to them.
    They can be interpreted as false echoes, or can otherwise muck up detection.  
    ~10 decay constants seems to be adequate for our prototypical flare. 
    """)

    onset = 1.5 * u.s
    decay = 4 * u.s
    time_domain = np.linspace(-2*onset.value, (onset + 11 * decay).value, 1000) * u.s

    max_decay_list = [4, 10]
    for max_num_decay in max_decay_list:
        MyFlare = ExponentialFlare1(onset=onset, decay=decay, max_decay=max_num_decay)
        flare_intensity_list = np.array([MyFlare.evaluate_at_time(ti).value for ti in time_domain])
        plt.plot(time_domain, flare_intensity_list,
                    color=color_map(max_num_decay/max(max_decay_list)),
                    lw=1, label="max_decay="+str(max_num_decay))
        plt.arrow((onset+max_num_decay*decay).value, .25, 0, -.15,
                  head_width=.5, head_length=.05,
                  color=color_map(max_num_decay/max(max_decay_list)))

    plt.ylabel("Flare intensity (ct/s)")
    plt.legend()

    plt.title("Prototypical ExponentialFlare1")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------------------------------------------------ #
    print("""
    Using evaluate_over_array_lw() preserves the total integrated counts, as shown for delta and exponential flares.
    The flares can be count-normalized by dividing by MyFlare.integrated_counts
    """)
    # ----------------------------------- #
    divs = [10, 11, 51]
    MyDeltaFlare = DeltaFlare(500)
    fig, ax_list = plt.subplots(1, len(divs), figsize=(6, 4))
    for di, ax in zip(divs, ax_list):
        times = np.linspace(-5, 5, di)
        integrated_flare = MyDeltaFlare.evaluate_over_array_lw(times)
        ax.plot(times, integrated_flare,
                 label=str(di)+"points", color='k', lw=1,
                 marker='.', drawstyle='steps-post')
        ax.set_title(str(di)+"points")
        ax.tick_params('x', top=True, direction='in')
        print("Time division: ", di, "\tFlare integration: ", np.sum(integrated_flare),
              "\tExact:", MyDeltaFlare.integrated_counts)
    fig.suptitle("Delta-function flare discretization")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()

    # ----------------------------------- #
    divs = [10, 50, 100, 1000]
    MyExponentialFlare = ExponentialFlare1(onset=onset, decay=decay)
    for di in divs:
        times = np.linspace(-40 * decay.value, (onset + 40 * decay).value, di)
        integrated_flare = MyExponentialFlare.evaluate_over_array_lw(times)
        plt.plot(times, integrated_flare,
                 label=str(di)+"points",
                 marker='.', drawstyle='steps-post')
        print("Time division:", di, "\tFlare counts:", np.sum(integrated_flare),
              "\tExact:", MyExponentialFlare.integrated_counts)
    plt.title("Exponential flare discretization")
    plt.legend()
    plt.show()


