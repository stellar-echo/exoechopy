import numpy as np
import exoechopy as eep
from astropy import units as u
from scipy import optimize

__all__ = ['FitFlareSuperresolutionPRED']

# TODO - handle double/multi-peak flares
# TODO - handle deweighting pixels

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def sum_frames(frames, frames_per_sum, frame_offset=0):
    all_vals = []
    for f_i, frame in enumerate(frames[frame_offset:]):
        if f_i % frames_per_sum == 0:
            all_vals.append(frame)
        else:
            all_vals[-1] += frame
    return u.Quantity(all_vals, all_vals[0].unit)


def sum_frames_lw(frames, frames_per_sum, frame_offset=0):
    all_vals = []
    for f_i, frame in enumerate(frames[frame_offset:]):
        if f_i % frames_per_sum == 0:
            all_vals.append(frame)
        else:
            all_vals[-1] += frame
    return np.array(all_vals)


class FitFlareSuperresolutionPRED:
    """Class for identifying probable flares signatures based on an observed time-coarsened flare"""

    def __init__(self,
                 time_domain: u.Quantity,
                 flux_array: u.Quantity,
                 int_time: u.Quantity,
                 read_time: u.Quantity,
                 frame_sum: int):
        self._time_domain_lw = time_domain.to('s').value
        self._time_domain = time_domain.copy()
        self._flux_array = flux_array.copy()
        self._flux_array_lw = self._flux_array.to('ct').value
        self._int_time = int_time.copy()
        self._int_time_lw = int_time.to('s').value
        self._read_time = read_time.copy()
        self._read_time_lw = read_time.to('s').value
        self._frame_cadence = self._int_time + self._read_time
        self._frame_cadence_lw = self._frame_cadence.to('s').value
        self._data_cadence = self._frame_cadence * frame_sum
        self._data_cadence_lw = self._frame_cadence_lw * frame_sum
        self._frame_sum = frame_sum
        self._bounds = [(1, np.inf),  # amplitude, in counts
                        (1E-1, np.inf),  # rise time
                        (1E-2, np.inf),  # decay const
                        (self._time_domain_lw[0], self._time_domain_lw[-1])]  # flare time

        if np.abs((self._time_domain[1] - self._time_domain[0]) - self._data_cadence).to(u.s).value > 1E-6:
            raise AttributeError("Time domain and cadence do not match within tolerance: "
                                 "\nCadence: ", self._data_cadence,
                                 "\ndt", self._time_domain[1] - self._time_domain[0])

    # ------------------------------------------------------------------------------------------------------------ #
    @property
    def frame_cadence(self):
        return self._frame_cadence.copy()

    @property
    def data_cadence(self):
        return self._data_cadence.copy()

    @property
    def int_time(self):
        return self._int_time.copy()

    @property
    def read_time(self):
        return self._read_time.copy()

    @property
    def time_domain(self):
        return self._time_domain.copy()

    @property
    def flux_array(self):
        return self._flux_array.copy()

    # ------------------------------------------------------------------------------------------------------------ #
    # TODO - consolidate common sections of exact_flare, integrate_PRED_flare_with_readout?
    def exact_flare(self,
                    amplitude: u.Quantity,
                    peak_time_abs: u.Quantity,
                    rise_time: u.Quantity,
                    decay_const: u.Quantity,
                    max_decay_const=10,
                    resolution=5):
        flare_model = eep.simulate.ParabolicRiseExponentialDecay(onset=rise_time,
                                                                 decay=decay_const,
                                                                 max_decay=max_decay_const)
        local_cadence = self.data_cadence / resolution
        flare_duration = flare_model.flare_duration + local_cadence
        num_frames = int(flare_duration / local_cadence)
        # Relative time local to flare:
        # all_times = u.Quantity(np.arange(0,
        #                                  (local_cadence * num_frames).to('s').value,
        #                                  local_cadence.value), 's')[:num_frames]
        time_domain_HD = u.Quantity(np.linspace(self._time_domain[0].value,
                                                self._time_domain[-1].value,
                                                len(self._time_domain) * resolution), 's')
        # print("Exact flare all_times: ", all_times)
        # print("Exact flare time_domain_HD: ", time_domain_HD)
        flare_start_time_exact = peak_time_abs - rise_time
        print("flare_start_time_exact: ", flare_start_time_exact)
        # TODO - handle case when flare start time is prior to data start
        flare_bin_start_ind = eep.utils.find_near_index_floor(time_domain_HD, flare_start_time_exact)
        print("Exact flare flare_bin_start_ind: ", flare_bin_start_ind)
        flare_start_time_binned = time_domain_HD[flare_bin_start_ind]
        print("exact flare_start_time_binned: ", flare_start_time_binned)
        time_shift = time_domain_HD[0] + flare_start_time_exact - flare_start_time_binned
        print("exact time_shift: ", time_shift)
        # all_times -= local_cadence + time_shift
        # time_domain_HD -= local_cadence + time_shift
        integrated_flare = amplitude * flare_model.evaluate_over_array(time_domain_HD - local_cadence - time_shift)

        max_ind = min(len(time_domain_HD), flare_bin_start_ind + num_frames - 1)
        flare_times = time_domain_HD[flare_bin_start_ind:max_ind]

        return_values = np.zeros(len(time_domain_HD))
        return_values[flare_bin_start_ind:max_ind] = integrated_flare[:len(flare_times)]
        return time_domain_HD, return_values

    def integrate_PRED_flare_with_readout(self,
                                          amplitude: u.Quantity,
                                          peak_time_abs: u.Quantity,
                                          rise_time: u.Quantity,
                                          decay_const: u.Quantity,
                                          max_decay_const=10):
        """
        Integrates a flare, dropping lost photons during readout.


        Parameters
        ----------
        amplitude
            Flare amplitude in counts
        peak_time_abs
            Flare peak time, referenced to class time_domain
        rise_time
            Parabolic rise--peak of flare occurs at rise_time
        decay_const
            Exponential decay
        max_decay_const
            How many decay constants to calculate after the flare to avoid C0 discontinuities

        Returns
        -------
        u.Quantity(flux values) registered with the class's time_domain

        """
        flare_model = eep.simulate.ParabolicRiseExponentialDecay(onset=rise_time,
                                                                 decay=decay_const,
                                                                 max_decay=max_decay_const)
        print("flare_model.flare_duration: ", flare_model.flare_duration)
        flare_duration = flare_model.flare_duration + self._frame_cadence
        num_frames = int(flare_duration / self.frame_cadence)

        # Ensure we have an integer-divisible number of frames:
        if num_frames % self._frame_sum != 0:
            num_frames += self._frame_sum - num_frames % self._frame_sum

        integrated_frames = num_frames // self._frame_sum

        # Relative time local to flare:
        flare_local_time = u.Quantity(np.zeros(num_frames * 2), 's')
        int_times = u.Quantity(np.arange(0,
                                         (self.frame_cadence * num_frames).to('s').value,
                                         self._frame_cadence_lw), 's')[:num_frames]
        read_times = int_times + self.int_time
        flare_local_time[::2] = int_times
        flare_local_time[1::2] = read_times

        flare_start_time_exact = peak_time_abs - rise_time
        flare_bin_start_ind = eep.utils.find_near_index_floor(self._time_domain,
                                                              flare_start_time_exact)
        flare_start_time_binned = self._time_domain[flare_bin_start_ind]

        time_shift = flare_start_time_exact - flare_start_time_binned
        flare_local_time -= time_shift
        integrated_flare = amplitude * flare_model.evaluate_over_array(flare_local_time)
        keep_vals = integrated_flare[::2]

        # TODO - verify max ind is selected correctly
        max_ind = min(len(self._time_domain), flare_bin_start_ind + integrated_frames)
        flare_times = self._time_domain[flare_bin_start_ind:max_ind]

        return_values = np.zeros(len(self._time_domain))
        return_values[flare_bin_start_ind:max_ind] = sum_frames(keep_vals, self._frame_sum)[:len(flare_times)]

        return return_values

    def _integrate_PRED_flare_with_readout_lw(self,
                                              amplitude: float,
                                              peak_time_abs: float,
                                              rise_time: float,
                                              decay_const: float,
                                              max_decay_const=10):
        """
        Integrates a flare, dropping lost photons during readout.


        Parameters
        ----------
        peak_time_abs
            Flare peak time, referenced to class time_domain
        rise_time
            Parabolic rise--peak of flare occurs at rise_time
        decay_const
            Exponential decay
        max_decay_const
            How many decay constants to calculate after the flare to avoid C0 discontinuities

        Returns
        -------
        u.Quantity(flux values) registered with the class's time_domain

        """
        flare_model = eep.simulate.ParabolicRiseExponentialDecay(onset=u.Quantity(rise_time, 's'),
                                                                 decay=u.Quantity(decay_const, 's'),
                                                                 max_decay=max_decay_const)
        print("flare_model.flare_duration: ", flare_model.flare_duration)
        flare_duration = flare_model.flare_duration.to(u.s).value + self._frame_cadence_lw
        num_frames = int(flare_duration / self._frame_cadence_lw)

        # Ensure we have an integer-divisible number of frames:
        if num_frames % self._frame_sum != 0:
            num_frames += self._frame_sum - num_frames % self._frame_sum

        integrated_frames = num_frames // self._frame_sum

        # Relative time local to flare:
        flare_local_time = np.zeros(num_frames * 2)
        int_times = np.arange(0, self._frame_cadence_lw * num_frames, self._frame_cadence_lw)[:num_frames]
        read_times = int_times + self._int_time_lw
        flare_local_time[::2] = int_times
        flare_local_time[1::2] = read_times

        flare_start_time_exact = peak_time_abs - rise_time
        print("peak_time_abs: ", peak_time_abs)
        print("flare_start_time_exact: ", flare_start_time_exact)
        # TODO - handle case when flare start time is prior to data start
        flare_bin_start_ind = eep.utils.find_near_index_floor(self._time_domain,
                                                              u.Quantity(flare_start_time_exact, u.s))
        flare_start_time_binned = self._time_domain_lw[flare_bin_start_ind]

        time_shift = flare_start_time_exact - flare_start_time_binned
        flare_local_time -= time_shift
        integrated_flare = amplitude * flare_model.evaluate_over_array(u.Quantity(flare_local_time, 's')).value
        keep_vals = integrated_flare[::2]

        # TODO - verify max ind is selected correctly
        max_ind = min(len(self._time_domain_lw), flare_bin_start_ind + integrated_frames)
        flare_times = self._time_domain_lw[flare_bin_start_ind:max_ind]

        return_values = np.zeros(len(self._time_domain_lw))
        return_values[flare_bin_start_ind:max_ind] = sum_frames_lw(keep_vals, self._frame_sum)[:len(flare_times)]

        return return_values

    def fitting_cost_function_PRED(self,
                                   opt_params):
        """
        Draft cost function for minimization of difference with real light curve.  Anticipate indexing issues.

        Parameters
        ----------
        opt_params
            Packed optimization parameters:
            rise_time_s
                Rise time, float, specified in seconds.  To be optimized.
            decay_const_s
                Decay constant, float, specified in seconds.  To be optimized.
            seg_peak_time_s
                Time of flare peak in the relative frame of the obs_time_domain, float, specified in seconds.  To be optimized.
        obs_time_domain
            Observation time domain
        obs_lightcurve
            Actual observed lightcurve
        int_time
            Integration time for the detector
        read_time
            Readout time for the detector
        frames_per_sum
            Number of frames summed into the actual data product
        frame_offset
            How many cadences to shift the frame summation by
            Note, this requires compensation of the flare position externally (-cadence * frame_offset)

        Returns
        -------

        """
        amplitude_ct, rise_time_s, decay_const_s, seg_peak_time_s = opt_params
        est_flare = self._integrate_PRED_flare_with_readout_lw(amplitude_ct,
                                                               seg_peak_time_s,
                                                               rise_time_s,
                                                               decay_const_s)

        # Ignore last datapoint just in case the est_flare frame shift artificially reduced the signal in the last bin
        diff = (self._flux_array_lw[:len(est_flare) - 1] - est_flare[:-1])
        cost_func = np.dot(diff, diff)
        return cost_func

    def fit_flare(self,
                  init_guess_amplitude: u.Quantity,
                  init_guess_rise_time: u.Quantity,
                  init_guess_decay_const: u.Quantity,
                  init_guess_flare_time: u.Quantity):
        initial_guesses = np.array((init_guess_amplitude.to(u.ct).value,
                                    init_guess_rise_time.to(u.s).value,
                                    init_guess_decay_const.to(u.s).value,
                                    init_guess_flare_time.to(u.s).value))
        result = optimize.minimize(self.fitting_cost_function_PRED,
                                   x0=initial_guesses,
                                   bounds=self._bounds)
        params = result.x
        amp = u.Quantity(params[0], 'ct')
        rise = u.Quantity(params[1], 's')
        decay = u.Quantity(params[2], 's')
        flare_time = u.Quantity(params[3], 's')
        result_dict = {'amplitude': amp,
                       'rise_time': rise,
                       'decay_const': decay,
                       'flare_time': flare_time,
                       'cost': result.fun}
        return result_dict


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
if __name__ == '__main__':
    from matplotlib import pyplot as plt

    int_time = 5.
    read_time = 0.5
    cadence = int_time + read_time
    frame_sum = 1

    my_time = u.Quantity(np.arange(-50, 150, cadence * frame_sum), 's')

    my_flare_time = u.Quantity(7, 's')
    my_flare_onset = u.Quantity(14, 's')
    my_flare_decay = u.Quantity(12, 's')
    my_flare_amplitude = u.Quantity(100, 'ct')

    initializer = FitFlareSuperresolutionPRED(time_domain=my_time,
                                              flux_array=u.Quantity(0, 'ct'),
                                              int_time=u.Quantity(int_time, 's'),
                                              read_time=u.Quantity(read_time, 's'),
                                              frame_sum=frame_sum)
    vals = initializer.integrate_PRED_flare_with_readout(my_flare_amplitude,
                                                         my_flare_time,
                                                         my_flare_onset,
                                                         my_flare_decay)
    exact_time, exact_flare = initializer.exact_flare(my_flare_amplitude,
                                                      my_flare_time,
                                                      my_flare_onset,
                                                      my_flare_decay,
                                                      resolution=40)

    my_lightcurve = 10 + vals
    my_lightcurve = np.random.poisson(my_lightcurve.astype(int))-10

    plt.plot(exact_time, exact_flare,
             drawstyle='steps-post', color='k', lw=1, label='Exact flare')
    plt.plot(initializer.time_domain, my_lightcurve,
             drawstyle='steps-post', color='k', lw=1, label='Observed lightcurve')

    fitter = FitFlareSuperresolutionPRED(time_domain=my_time,
                                         flux_array=u.Quantity(my_lightcurve, u.ct),
                                         int_time=u.Quantity(int_time, 's'),
                                         read_time=u.Quantity(read_time, 's'),
                                         frame_sum=frame_sum)
    results = fitter.fit_flare(my_flare_amplitude,
                               my_flare_onset,
                               my_flare_decay,
                               my_flare_time)

    fitted_vals = fitter.integrate_PRED_flare_with_readout(amplitude=results['amplitude'],
                                                           peak_time_abs=results['flare_time'],
                                                           rise_time=results['rise_time'],
                                                           decay_const=results['decay_const'])

    plt.plot(fitter.time_domain, fitted_vals,
             drawstyle='steps-post', color='r', lw=1, label='Fitted lightcurve')

    exact_time_fit, exact_flare_fit = initializer.exact_flare(amplitude=results['amplitude'],
                                                              peak_time_abs=results['flare_time'],
                                                              rise_time=results['rise_time'],
                                                              decay_const=results['decay_const'],
                                                              resolution=40)
    plt.plot(exact_time_fit, exact_flare_fit,
             drawstyle='steps-post', color='g', lw=1, label='Fitted flare implied shape')

    plt.legend()

    plt.plot()

    print("Results:\n", results)

    print("np.sum(exact_flare): ", np.mean(exact_flare))
    print("np.sum(vals): ", np.mean(vals))
