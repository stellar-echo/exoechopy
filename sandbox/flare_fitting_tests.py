import numpy as np
import exoechopy as eep
from matplotlib import pyplot as plt
from astropy import units as u


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
    int_times = u.Quantity(np.arange(0, (cadence * num_frames).to('s').value, cadence.to('s').value), 's')
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

    Returns
    -------

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

    print('start_time: ', start_time)
    print("relative_peak_time: ", relative_peak_time)
    print("frame_times: ", frame_times)

    summed_times, summed_values = sum_frames(frame_times, frame_values, frames_per_sum, frame_offset)
    summed_times -= cadence*frame_offset
    return summed_times, summed_values

# TODO: generate synthetic flare, allow shifting within 1 cadence
# Then, use frame summing (will test each frame-shift as a separate minimization)
# Then, create the cost function that will be used in minimization
# Then, pick parameters from best outcome (analyze other outcomes too, though)


kepler_int_time = u.Quantity(6.02, 's')
kepler_read_time = u.Quantity(0.52, 's')

kepler_sc_num_frame_sum = 9

segment_start_time = u.Quantity(-180, 's')
segment_end_time = u.Quantity(240, 's')

test_flare_rise_time = u.Quantity(12, 's')
test_flare_decay_const = u.Quantity(24, 's')



my_flare = eep.simulate.ParabolicRiseExponentialDecay(test_flare_rise_time,
                                                      test_flare_decay_const)
times, values = integrate_flare_with_readout(start_time=segment_start_time,
                                             int_time=kepler_int_time,
                                             read_time=kepler_read_time,
                                             total_time=segment_end_time - segment_start_time,
                                             flare_model=my_flare)

f, ax = plt.subplots(ncols=9)
for test_offset in range(9):

    summed_times_0, summed_values_0 = sum_frames(times, values, kepler_sc_num_frame_sum, frame_offset=test_offset)
    cadence = kepler_int_time+kepler_read_time
    ax[test_offset].plot(times-test_offset*cadence, values, drawstyle='steps-post', color='k')
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

    print("Difference: ", np.sum((summed_values_0-summed_values_1))**2)

