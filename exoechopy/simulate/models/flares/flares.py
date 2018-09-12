
"""
This module provides functionality for modeling flare light curve profiles.

To create your own flare:
 Create child class of ProtoFlare
 Common overrides:
    - self.discontinuities:         list of points where the function is piecewise evaluated
    - self._counts:                 total integrated counts, analytically evaluated if possible
    - self._integrated_flare_lw():  the analytically computed (if possible) piecewise integration of the flare
    - self._evaluate_at_time_lw():  the analytical form of the flare's lightcurve

"""

import warnings
import numpy as np
from astropy import units as u
from astropy.utils.exceptions import AstropyUserWarning
from exoechopy.utils.globals import *
from scipy import stats

__all__ = ['ProtoFlare', 'DeltaFlare', 'ExponentialFlare1']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class ProtoFlare:
    """Template class for flares, generates no signal."""
    def __init__(self, **kwargs):
        self._counts = None
        self.integrated_counts = None
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

    @integrated_counts.setter
    def integrated_counts(self, counts):
        if counts:
            if isinstance(counts, CountType):
                self._counts = float(counts)
            else:
                raise ValueError("counts must be int/float/u.Quantity or None")
        else:
            self._counts = None

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
    def __init__(self,
                 counts: CountType=None,
                 **kwargs):
        """
        Adds counts to a single time bin.

        :param CountType counts:
        """
        super().__init__()

        self.integrated_counts = counts

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
                 onset: u.Quantity,
                 decay: u.Quantity,
                 max_decay: IntFloat=10,
                 **kwargs):
        """
        Standard impulsive flare with a sharp onset and exponential decay.
        Multiply values by intensity to scale for different scenarios

        :param u.Quantity onset: Onset time
        :param u.Quantity decay: Exponential decay constant, exp(-t/decay)
        :param float max_decay: How many decay constants to trail the flare peak,
        influences the C1 discontinuity at the end of tail
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

        if isinstance(max_decay, IntFloat):
            self._decay_duration = self._decay * max_decay
        else:
            raise TypeError("max_decay in Class ExponentialFlare1 must be an integer or a float")

        self._epsilon = np.exp(-self._decay_duration/self._decay)

        # Explicit definition of where the function is piecewise/discontinuous:
        self.discontinuities = [0, self._onset, self._onset+self._decay_duration]

        # Total counts, as determined through analytical integration of the piecewise flare function:
        self.integrated_counts = self._onset/3. + self._decay - self._decay_duration*self._epsilon/(1.-self._epsilon)

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
                 flare_type: ProtoFlare=None,
                 num_subflare_pdf: PDFType=None,
                 **kwargs):
        """
        Creates a multi-peaked flare based on input parameters and a designated flare type
        :param ProtoFlare flare_type:
        :param PDFType num_subflare_pdf: value, tuple of [min, max], or frozen PDF for determining number of subflares
        """
        super().__init__(**kwargs)

        if isinstance(flare_type, DeltaFlare):
            self._flare_type = flare_type
        if isinstance(num_subflare_pdf, int):
            self._num_flares = num_subflare_pdf
        elif isinstance(num_subflare_pdf, stats.rv_discrete):
            self._num_flares = num_subflare_pdf.rvs()
        raise NotImplementedError


# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.cm as mplcm
    color_map = mplcm.copper_r

    #  =============================================================  #
    print("""
    It is important to avoid discontinuities after the flare peak, where an echo may occur.
    Many detection filters are sensitive to them, and they can be interpreted as false echoes or alter the analysis.
    Jump discontinuities are killer, so we have implemented an exponential decay that eliminates the jump.
    C1 discontinuities can be a problem as well, so we push them far from the flare peak;  
    ~10 decay constants seems to be adequate for our prototypical flare. 
    """)

    onset_time = 1.5 * u.s
    decay_const = 4 * u.s
    time_domain = np.linspace(-2 * onset_time.value, (onset_time + 11 * decay_const).value, 1000) * u.s

    max_decay_list = [4, 10]
    for max_num_decay in max_decay_list:
        MyFlare = ExponentialFlare1(onset=onset_time, decay=decay_const, max_decay=max_num_decay)
        flare_intensity_list = np.array([MyFlare.evaluate_at_time(ti).value for ti in time_domain])
        plt.plot(time_domain, flare_intensity_list,
                 color=color_map(max_num_decay/max(max_decay_list)),
                 lw=1, label="max_decay="+str(max_num_decay))
        plt.arrow((onset_time + max_num_decay * decay_const).value, .25, 0, -.15,
                  head_width=.5, head_length=.05,
                  color=color_map(max_num_decay/max(max_decay_list)))

    plt.xlabel('Time (sec)')
    plt.ylabel('Flare intensity (ct/s)')
    plt.legend()

    plt.title('Prototypical ExponentialFlare1')
    plt.tight_layout()
    plt.show()

    #  =============================================================  #
    print("""
    The delta-function flare narrows as the discretization increases, preserving the total peak count.
    To normalize to flux rather than integrated counts, divide by delta_t
    Depending on discretization, the point may occur before or on the zero-point of the time domain:
    delta_function(t_0, t_1) =  {  1 if 0 in [t_0, t_1)  }
                                {  0 else                }
    """)
    divs = [10, 11, 51]
    MyDeltaFlare = DeltaFlare(500)
    fig, ax_list = plt.subplots(1, len(divs), figsize=(10, 4))
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
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Counts')
    fig.suptitle("Delta-function flare discretization")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()

    #  =============================================================  #
    print("""
    Using evaluate_over_array_lw() preserves the total integrated counts, as shown for delta and exponential flares.
    The flares can be count-normalized by dividing by MyFlare.integrated_counts
    The flares can be converted to rates (ct/sec) by dividing by dt
    """)
    divs = [10, 50, 100, 1000]
    fig, (ax_ct, ax_ctsec) = plt.subplots(1, 2, figsize=(10, 4))
    MyExponentialFlare = ExponentialFlare1(onset=onset_time, decay=decay_const)
    for di in divs:
        times = np.linspace(-40 * decay_const.value, (onset_time + 40 * decay_const).value, di)
        integrated_flare = MyExponentialFlare.evaluate_over_array_lw(times)
        ax_ct.plot(times, integrated_flare,
                   label=str(di)+"points",
                   marker='.', drawstyle='steps-post')
        ax_ctsec.plot(times, integrated_flare/(times[1]-times[0]),
                      label=str(di)+"points",
                      marker='.', drawstyle='steps-post')
        print("Time division:", di, "\tFlare counts:", np.sum(integrated_flare),
              "\tExact:", MyExponentialFlare.integrated_counts)

    ax_ct.set_xlabel("Time (s)")
    ax_ct.set_ylabel("Counts")
    ax_ct.set_title("Integrated counts")
    ax_ct.legend()

    ax_ctsec.set_xlabel("Time (s)")
    ax_ctsec.set_ylabel("Counts/sec")
    ax_ctsec.set_title("Counts/second")
    ax_ctsec.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Exponential flare discretization")

    plt.show()


