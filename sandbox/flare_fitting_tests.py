import numpy as np
import exoechopy as eep
from matplotlib import pyplot as plt
from astropy import units as u
from scipy import optimize


def integrate_flare_with_readout(start_time: u.Quantity,
                                 int_time: u.Quantity,
                                 read_time: u.Quantity,
                                 total_time: u.Quantity,
                                 flare_model: eep.simulate.ProtoFlare):
    """
    Integrates a flare, dropping lost photons during readout.
    Start time is relative to flare_model, so 0 is typically the flare_model start
    Cadence is determined from int_time_s + read_time_s
    The duration is determined from total_time_s and may have some rounding differences in array length

    Parameters
    ----------
    start_time
        First time point in the integrated flare model
    int_time
        Time to integrate, for each frame
    read_time
        Time between integrations where light is lost
    total_time
        Duration of the simulation
    flare_model
        A pre-generated flare model, such as eep.simulate.ParabolicRiseExponentialDecay

    Returns
    -------
    u.Quantity(integration times), u.Quantity(integrated flux)
    """
    cadence = int_time + read_time
    num_frames = int(total_time // cadence)
    all_times = u.Quantity(np.zeros(num_frames * 2), 's')
    int_times = u.Quantity(np.arange(0, (cadence * num_frames).to('s').value, cadence.to('s').value), 's')[:num_frames]
    read_times = int_times + int_time
    all_times[::2] = int_times
    all_times[1::2] = read_times
    all_times += start_time
    integrated_flare = flare_model.evaluate_over_array(all_times)
    keep_vals = integrated_flare[::2]
    keep_times = all_times[::2]
    return keep_times, keep_vals


def sum_frames(frame_times: u.Quantity,
               frame_vals: u.Quantity,
               frames_per_sum: int,
               frame_offset=0):
    """
    Function that sums frames, allowing an offset

    Parameters
    ----------
    frame_times
        Array of times corresponding to frame values
    frame_vals
        Array of values to be summed
    frames_per_sum
        Number of frames to sum per bin
    frame_offset
        Number of frames to offset the summation by

    Returns
    -------
    u.Quantity(coarsened data)

    Note, the final frame may have a partial summation depending on offset/num frames.
    Recommend dropping it.
    """
    _times = []
    _integrated_lc = []
    meas_unit = frame_vals.unit
    time_unit = frame_times.unit
    for f_i, (time, measurement) in enumerate(zip(frame_times[frame_offset:],
                                                  frame_vals[frame_offset:])):
        if f_i % frames_per_sum == 0:
            _times.append(time.value)
            _integrated_lc.append(measurement.value)
        else:
            _integrated_lc[-1] += measurement.value
    return_times = u.Quantity(_times, time_unit)
    return_measures = u.Quantity(_integrated_lc, meas_unit)
    return return_times, return_measures


def processed_PRED_signal(rise_time: u.Quantity,
                          decay_const: u.Quantity,
                          int_time: u.Quantity,
                          read_time: u.Quantity,
                          seg_start_time: u.Quantity,
                          seg_end_time: u.Quantity,
                          seg_peak_time: u.Quantity,
                          frames_per_sum: int,
                          frame_offset: int = 0):
    """

    Parameters
    ----------
    rise_time
        Flare rise time
    decay_const
        Flare decay constant
    seg_peak_time
        Flare peak intensity in time domain of seg_start_time, seg_end_time
    int_time
        Frame integration time
    read_time
        Frame readout time
    seg_start_time
        Flare segment start time
        Can be relative, like 0, or absolute, like JDs since obs start
    seg_end_time
        Flare segment end time, same time domain as seg_start_time
    frames_per_sum
        Number of integrated frames per summation
    frame_offset
        How many cadences to shift the frame summation by
        Note, this requires compensation of the flare position externally (-cadence * frame_offset)

    Returns
    -------
    u.Quantity(summed_frame_times), u.Quantity(summed_frame_values)
    """
    assert frame_offset >= 0
    test_flare = eep.simulate.ParabolicRiseExponentialDecay(rise_time,
                                                            decay_const)
    relative_peak_time = seg_peak_time - seg_start_time
    assert relative_peak_time > 0

    cadence = int_time + read_time
    start_time = - relative_peak_time + rise_time
    total_time = seg_end_time - seg_start_time

    frame_times, frame_values = integrate_flare_with_readout(start_time=start_time,
                                                             int_time=int_time,
                                                             read_time=read_time,
                                                             total_time=total_time,
                                                             flare_model=test_flare)

    # print('start_time: ', start_time)
    # print("relative_peak_time: ", relative_peak_time)
    # print("frame_times: ", frame_times)

    summed_times, summed_values = sum_frames(frame_times, frame_values, frames_per_sum, frame_offset)
    summed_times -= cadence * frame_offset
    return summed_times, summed_values


def fitting_cost_function_PRED(opt_params,
                               obs_time_domain: u.Quantity,
                               obs_lightcurve,
                               int_time: u.Quantity,
                               read_time: u.Quantity,
                               frames_per_sum: int,
                               frame_offset: int = 0):
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
    rise_time_s, decay_const_s, seg_peak_time_s = opt_params
    seg_start_time = obs_time_domain[0].to('s')
    seg_end_time = obs_time_domain[-1].to('s')
    _, est_flare = processed_PRED_signal(u.Quantity(rise_time_s, 's'),
                                         u.Quantity(decay_const_s, 's'),
                                         int_time=int_time,
                                         read_time=read_time,
                                         seg_start_time=seg_start_time,
                                         seg_end_time=seg_end_time,
                                         seg_peak_time=u.Quantity(seg_peak_time_s, 's'),
                                         frames_per_sum=frames_per_sum,
                                         frame_offset=frame_offset)
    # Ignore last datapoint just in case the est_flare frame shift artificially reduced the signal in the last bin
    diff = (obs_lightcurve[:len(est_flare) - 1] - est_flare[:-1])
    cost_func = np.dot(diff, diff)
    return cost_func


# TODO:
# Create the cost function that will be used in minimization
# Then, pick parameters from best outcome (analyze other outcomes too, though)
# Repeat for each frame offset option


kepler_int_time = u.Quantity(6.02, 's')
kepler_read_time = u.Quantity(0.52, 's')

kepler_sc_num_frame_sum = 9

segment_start_time = u.Quantity(-180, 's')
segment_end_time = u.Quantity(720, 's')

test_flare_rise_time = u.Quantity(32, 's')
test_flare_decay_const = u.Quantity(64, 's')

test_flare_start_time = 2.5 * test_flare_rise_time

my_flare = eep.simulate.ParabolicRiseExponentialDecay(test_flare_rise_time,
                                                      test_flare_decay_const)
times, values = integrate_flare_with_readout(start_time=segment_start_time,
                                             int_time=kepler_int_time,
                                             read_time=kepler_read_time,
                                             total_time=segment_end_time - segment_start_time,
                                             flare_model=my_flare)

cadence = kepler_int_time + kepler_read_time

f, ax = plt.subplots(ncols=9)
for test_offset in range(9):
    summed_times_0, summed_values_0 = sum_frames(times, values, kepler_sc_num_frame_sum, frame_offset=test_offset)
    ax[test_offset].plot(times - test_offset * cadence, values, drawstyle='steps-post', color='k')
    # ax[test_offset].plot(summed_times_0, summed_values_0, drawstyle='steps-post', color='r')

    summed_times_1, summed_values_1 = processed_PRED_signal(test_flare_rise_time,
                                                            test_flare_decay_const,
                                                            int_time=kepler_int_time,
                                                            read_time=kepler_read_time,
                                                            seg_start_time=segment_start_time,
                                                            seg_end_time=segment_end_time,
                                                            seg_peak_time=test_flare_rise_time,  # Note, tweak this
                                                            frames_per_sum=kepler_sc_num_frame_sum,
                                                            frame_offset=test_offset)

    ax[test_offset].plot(summed_times_1, summed_values_1, drawstyle='steps-post', color='g')

    print("Difference: ", np.sum((summed_values_0 - summed_values_1)) ** 2)
plt.close()

test_offset = 5
time_offset = u.Quantity(165, 's')


init_guess_rise_time = 8
init_guess_decay_const = 10
init_guess_flare_time = 180
in_obs_time_domain, in_obs_lightcurve = sum_frames(times, values, kepler_sc_num_frame_sum, frame_offset=test_offset)

in_obs_time_domain += time_offset

obs_unit = in_obs_lightcurve.unit
test_noise_level = 100

in_obs_lightcurve *= test_noise_level
in_obs_lightcurve += u.Quantity(test_noise_level, obs_unit)
in_obs_lightcurve = np.random.poisson(in_obs_lightcurve.value.astype(int)).astype(float)
in_obs_lightcurve -= test_noise_level
in_obs_lightcurve /= test_noise_level
in_obs_lightcurve = u.Quantity(in_obs_lightcurve, obs_unit)

initial_guesses = np.array((init_guess_rise_time, init_guess_decay_const, init_guess_flare_time))
bounds = [(1E-1, np.inf), (1E-2, np.inf), (-np.inf, np.inf)]  # May need to make last bound constrained by window?


# TODO:
#  - Create a solution dict
#  - Take the best answer(s) and provide that as the 'result' for the underlying flare
#  - Move fitting operations into a class to reduce redundancies and chances of making a mistake

def fit_flares(raw_data_time,
               raw_data_values,
               init_guess_rise_time,
               init_guess_decay_const,
               init_guess_flare_time,
               int_time=kepler_int_time,
               read_time=kepler_read_time,
               num_frame_sum=kepler_sc_num_frame_sum):
    result_dict = {}
    cadence = read_time + int_time
    for offset in range(num_frame_sum):
        input_args = (raw_data_time, raw_data_values,
                      int_time, read_time, num_frame_sum,
                      offset)
        initial_guesses = np.array((init_guess_rise_time, init_guess_decay_const, init_guess_flare_time))
        result = optimize.minimize(fitting_cost_function_PRED, x0=initial_guesses, args=input_args, bounds=bounds)
        params = result.x
        rise = u.Quantity(params[0], 's')
        decay = u.Quantity(params[1], 's')
        flare_time = u.Quantity(params[2], 's') - offset * cadence
        result_dict[offset] = {'rise_time': rise,
                               'decay_const': decay,
                               'flare_time': flare_time,
                               'offset': offset,
                               'cost': result.fun}
    return result_dict


def select_best_fit(res_dict: dict):
    best_result = None
    best_cost = np.inf
    for k, v in res_dict.items():
        if v['cost'] < best_cost:
            best_cost = v['cost']
            best_result = v
    return best_result


def best_fit_flare(fit_dict, start_time, int_time, read_time, total_time):
    fit_flare = eep.simulate.ParabolicRiseExponentialDecay(fit_dict['rise_time'],
                                                           fit_dict['decay_const'])
    print("start_time: ", start_time)
    print("start_time - fit_dict['flare_time']: ", start_time - fit_dict['flare_time'])
    print("fit_dict['flare_time']: ", fit_dict['flare_time'])
    fit_times, fit_values = integrate_flare_with_readout(
        start_time=start_time - fit_dict['flare_time'],
        int_time=int_time,
        read_time=read_time,
        total_time=total_time,
        flare_model=fit_flare)
    fit_times += fit_dict['flare_time'] - fit_dict['rise_time']
    return fit_times, fit_values


def best_fit_to_obs(fit_dict, start_time, int_time, read_time, total_time, frames_per_sum):
    cadence = read_time + int_time
    fit_times, fit_values = processed_PRED_signal(fit_dict['rise_time'],
                                                  fit_dict['decay_const'],
                                                  int_time=int_time,
                                                  read_time=read_time,
                                                  seg_start_time=start_time,
                                                  seg_end_time=total_time + start_time,
                                                  seg_peak_time=fit_dict['flare_time'],
                                                  frames_per_sum=frames_per_sum,
                                                  frame_offset=0)
    fit_times += fit_dict['flare_time'] - fit_dict['rise_time']
    return fit_times, fit_values


d = fit_flares(in_obs_time_domain, in_obs_lightcurve,
               init_guess_rise_time, init_guess_decay_const, init_guess_flare_time)
best_fit_dict = select_best_fit(d)

print("Best fit: ", best_fit_dict)

plt.plot(times + time_offset, values, drawstyle='steps-post', color='k', lw=1,
         label='Exact solution')
plt.plot(in_obs_time_domain, in_obs_lightcurve, drawstyle='steps-post', color='g', lw=1,
         label='Actual data')
fit_flare_times, fit_flare_values = best_fit_flare(best_fit_dict,
                                                   in_obs_time_domain[0],
                                                   kepler_int_time,
                                                   kepler_read_time,
                                                   in_obs_time_domain[-1] - in_obs_time_domain[0])
plt.plot(fit_flare_times, fit_flare_values, drawstyle='steps-post', color='r', lw=1,
         label='Represented solution')

summed_times_solved, summed_values_solved = best_fit_to_obs(best_fit_dict,
                                                            in_obs_time_domain[0],
                                                            kepler_int_time,
                                                            kepler_read_time,
                                                            in_obs_time_domain[-1] - in_obs_time_domain[0],
                                                            kepler_sc_num_frame_sum)
plt.plot(summed_times_solved, summed_values_solved,
         drawstyle='steps-post', color='b', lw=1.5,
         label='Solved')

plt.legend()

# f, ax = plt.subplots(ncols=9)
# for test_offset in range(9):
#     # Plot the easy stuff first:
#     ax[test_offset].plot(times + time_offset, values, drawstyle='steps-post', color='k', lw=1,
#                          label='Exact solution')
#     ax[test_offset].plot(in_obs_time_domain, in_obs_lightcurve, drawstyle='steps-post', color='g', lw=1,
#                          label='Actual data')
#     print("test_offset: ", test_offset)
#     input_args = (in_obs_time_domain, in_obs_lightcurve,
#                   kepler_int_time, kepler_read_time, kepler_sc_num_frame_sum,
#                   test_offset)
#     print("len(input_args): ", len(input_args))
#
#     res = optimize.minimize(fitting_cost_function_PRED, x0=initial_guesses, args=input_args, bounds=bounds)
#     print("Exact: ", test_flare_rise_time.value, test_flare_decay_const.value, test_flare_rise_time.value)
#     print("Solved: ", res.x)
#     print("Objective function value: ", res.fun)
#     soln_str = ["{:.2f}".format(x) for x in res.x]
#     print(res.message)
#     solved_flare_rise_time, solved_flare_decay_const, solved_peak_time = res.x
#     solved_flare_rise_time = u.Quantity(solved_flare_rise_time, 's')
#     solved_flare_decay_const = u.Quantity(solved_flare_decay_const, 's')
#     solved_peak_time = u.Quantity(solved_peak_time, 's')
#     summed_times_solved, summed_values_solved = processed_PRED_signal(solved_flare_rise_time,
#                                                                       solved_flare_decay_const,
#                                                                       int_time=kepler_int_time,
#                                                                       read_time=kepler_read_time,
#                                                                       seg_start_time=segment_start_time + time_offset,
#                                                                       seg_end_time=segment_end_time + time_offset,
#                                                                       seg_peak_time=solved_peak_time,
#                                                                       frames_per_sum=kepler_sc_num_frame_sum,
#                                                                       frame_offset=test_offset)
#     ax[test_offset].plot(summed_times_solved + test_offset * cadence, summed_values_solved,
#                          drawstyle='steps-post', color='b', lw=1.5,
#                          label='Solved')
#
#     my_flare = eep.simulate.ParabolicRiseExponentialDecay(solved_flare_rise_time,
#                                                           solved_flare_decay_const)
#     new_times, new_values = integrate_flare_with_readout(
#         start_time=segment_start_time + test_offset * cadence - solved_peak_time + time_offset,
#         int_time=kepler_int_time,
#         read_time=kepler_read_time,
#         total_time=segment_end_time - segment_start_time,
#         flare_model=my_flare)
#
#     ax[test_offset].plot(new_times, new_values,
#                          drawstyle='steps-post', color='r', lw=1,
#                          label='Represented solution')
#     ax[test_offset].set_title(str(test_offset) + ", " + str(res.fun) + "\n" + str(soln_str))
#     ax[test_offset].legend()
