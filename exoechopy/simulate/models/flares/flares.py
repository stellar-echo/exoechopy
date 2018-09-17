
"""
This module provides functionality for modeling flare light curve profiles.

"""

import warnings
import numpy as np
from astropy import units as u
from astropy.utils.exceptions import AstropyUserWarning
from exoechopy.utils.constants import *
from scipy import stats

__all__ = ['ProtoFlare', 'DeltaFlare', 'ExponentialFlare1']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class ProtoFlare:
    """Template class for flares, generates no signal."""
    def __init__(self, **kwargs):
        """Provides a generic flare flass meant for subclassing.

        Parameters
        ----------

        Attributes
        ----------
        integrated_counts
        discontinuities

        Methods
        ----------
        evaluate_over_array_lw
        evaluate_at_time

        **Methods that can be overwritten by subclasses**
        _integrated_flare_lw
        _evaluate_at_time_lw
        integrated_counts

        Notes
        ----------
        To create your own flare:
            Create child class of ProtoFlare
            Common overrides:
                - self.discontinuities:         list of points that separate different piecewise regions
                - self._counts:                 total integrated counts, analytically evaluated if possible
                - self._integrated_flare_lw():  the analytically computed piecewise integration of the flare
                - self._evaluate_at_time_lw():  the analytical form of the flare's lightcurve

        Currently, flares are designed to be normalized to one at their peak intensity.
        Alternatively, they can be normalized by total intensity by dividing by flare.integrated_counts

        See Also
        ----------
        DeltaFlare
        ExponentialFlare1
        MultipeakFlare
        """
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
            self._counts = 1.

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def discontinuities(self) -> list:
        """Times that the flare light curve requires piecewise considerations

        Returns
        -------
        list
            List of floats, times that the analytical function is piecewise (not necessarily C0/C1 discontinuous)
        """
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
    def evaluate_over_array_lw(self, time_array: np.ndarray) -> np.ndarray:
        """Evaluate the flare over a unitless time domain.

        This discretizes and 'analytically' integrates the time bins
        Need to re-implement with faster array handling, such as np.where().  For now, just getting it working correctly
        Each bin is integrated over [x_i, x_i+1).

        Parameters
        ----------
        time_array

        Returns
        -------
        np.ndarray
            Provides the discretized values for the times provided by the time_array
        """
        dt = time_array[1] - time_array[0]
        # Provide an end-point for the last bin to integrate over:
        edge_adjusted_time_array = np.append(time_array, time_array[-1]+dt)
        return np.array([self._integrate_between_times_lw(t0, t1)
                         for t0, t1 in zip(edge_adjusted_time_array[:-1], edge_adjusted_time_array[1:])])

    def evaluate_over_array(self, time_array: u.Quantity) -> u.Quantity:
        """Evaluate the flare over a time domain.

        This discretizes and 'analytically' integrates the time bins
        Need to re-implement with faster array handling, such as np.where().  For now, just getting it working correctly
        Each bin is integrated over [x_i, x_i+1).

        Parameters
        ----------
        time_array

        Returns
        -------
        np.ndarray
            Provides the discretized values for the times provided by the time_array
        """
        time_array_lw = time_array.to(lw_time_unit).value
        return self.evaluate_over_array_lw(time_array_lw)*lw_time_unit

    # ------------------------------------------------------------------------------------------------------------ #
    def _integrate_between_times_lw(self, t0: float, t1: float) -> float:
        """Evaluate the flare between times t0 and t0+dt in sec

        May want to upgrade to be more array-friendly, for now just trying to get it right

        Parameters
        ----------
        t0
            Integration start time
        t1
            Integration end time

        Returns
        -------
        float
            Difference between _integrated_flare_lw at the two times, after checking for shortcuts
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
    def _integrated_flare_lw(self, t0: float) -> float:
        """Integrated flare, analytically pre-computed

        Parameters
        ----------
        t0

        Returns
        -------

        """
        return 0.

    # ------------------------------------------------------------------------------------------------------------ #
    def evaluate_at_time(self, time: u.Quantity) -> u.Quantity:
        """Evaluate the flare function at the given time, tracking units

        Parameters
        ----------
        time
            Evaluate the flare at the given time

        Returns
        -------
        u.Quantity
            Returns ct/sec
        """
        if isinstance(time, u.Quantity):
            _time = time.to(u.s).value
        else:
            _time = u.Quantity(time, u.s).value
            warnings.warn("Casting time, input as " + str(time) + ", to seconds", AstropyUserWarning)
        return self._evaluate_at_time_lw(_time)*u.ct/u.s

    def _evaluate_at_time_lw(self, t: float) -> float:
        """Unitless version of flare lightcurve.

        Parameters
        ----------
        t
            Time of evaluation in seconds

        Returns
        -------
        float
            Represents ct/s

        """
        return 0.
    # ------------------------------------------------------------------------------------------------------------ #


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


class DeltaFlare(ProtoFlare):
    """A flare that occurs in a single time bin"""
    def __init__(self,
                 counts: CountType=None,
                 **kwargs):
        """Produces a flare that fires all photons within a single time bin.

        Adds counts to a single time bin.
        Note, the standard convention for this library is to treat these flare classes as normalized
        and rescale externally, so .

        delta_function(t_0, t_1) =  {  counts if 0 in [t_0, t_1)  }
                                    {  0 else                     }

        Parameters
        ----------
        counts
            Defaults to 1 count within one bin


        See Also
        ----------
        ProtoFlare
        ExponentialFlare1
        MultipeakFlare
        """

        super().__init__()

        self.integrated_counts = counts

    # ------------------------------------------------------------------------------------------------------------ #
    def _integrated_flare_lw(self, t0):
        """Integrated flare, analytically pre-computed

        Parameters
        ----------
        t0
            Time to evaluate the flare at.
        Returns
        -------
        float
            delta_function(t_0, t_1) =  {  counts if 0 in [t_0, t_1)  }
                                        {  0 else                     }

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
        """Standard impulsive flare with a sharp onset and exponential decay.

        Can multiply values by an intensity to scale for different scenarios.
        Can divide by .integrated_counts to normalize the total area under the flare curve.


        Parameters
        ----------
        onset
            Onset time, follows a parabolic rise in intensity
        decay
            Exponential decay constant, exp(-t/decay)
        max_decay
            How many decay constants trail the flare peak, which influences the C1 discontinuity at the end of tail
        kwargs

        See Also
        ----------
        ProtoFlare
        DeltaFlare
        MultipeakFlare
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

        if self._decay < self._onset/2:
            warnings.warn("Decay, " + str(decay) + ", shorter than onset/2, " + str(onset), AstropyUserWarning)

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
    def _evaluate_at_time_lw(self, t: float) -> float:
        """Evaluates the flare without considering units

        Parameters
        ----------
        t
            Time in seconds at which to evaluate the flare.

        Returns
        -------
        float
            Represents ct/s

        """

        if 0 < t < self._onset + self._decay_duration:
            if t <= self._onset:
                return np.square(t/self._onset)
            else:
                return (np.exp(-(t-self._onset)/self._decay)-self._epsilon)/(1.-self._epsilon)
        else:
            return 0.

    # ------------------------------------------------------------------------------------------------------------ #
    def _integrated_flare_lw(self, t0: float) -> float:
        """Analytically pre-computed flare integration

        Parameters
        ----------
        t0
            Time of evaluation in sec ('unitless')
        Returns
        -------

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
        """Creates a multi-peaked flare based on input parameters and a designated flare type.

        Parameters
        ----------
        flare_type
            Type of underlying flare to repeat within the multipeak function
        num_subflare_pdf
            Value, tuple of [min, max], or frozen PDF for determining number of subflares
        kwargs

        See Also
        ----------
        ProtoFlare
        DeltaFlare
        ExponentialFlare1
        """

        super().__init__(**kwargs)

        if isinstance(flare_type, DeltaFlare):
            self._flare_type = flare_type
        if isinstance(num_subflare_pdf, int):
            self._num_flares = num_subflare_pdf
        elif isinstance(num_subflare_pdf, stats.rv_discrete):
            self._num_flares = num_subflare_pdf.rvs()
        raise NotImplementedError


