from scipy import signal, optimize
import numpy as np

from matplotlib import pyplot as plt
import matplotlib
from exoechopy.simulate import ParabolicRiseExponentialDecay as pred
from astropy import units as u

sig_noise_ratio = 30
cadence = 1.5
signal_mag = 1

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

flare = pred(u.Quantity(0.5, 's'), u.Quantity(1.5, 's'))
time_series = u.Quantity(np.arange(-10, 20, cadence), 's')
flare_signal = flare.evaluate_over_array(time_series)

kernel = np.zeros(7)

kernel[2] = 1
kernel[4] = 1


def make_echo(flare_signal, kernel):
    return np.convolve(flare_signal, kernel, mode='same')


echo_pristine = make_echo(flare_signal, kernel)
scale_factor = echo_pristine.max()
echo_pristine /= scale_factor
echo_pristine *= sig_noise_ratio ** 2
echo_measurement = echo_pristine + np.random.poisson(
    np.ones(len(echo_pristine)) * sig_noise_ratio ** 2) - sig_noise_ratio ** 2


# echo_measurement = np.random.poisson(echo_pristine.value).astype(float)
# echo_measurement /= np.sum(kernel)


def opt_func(values, observation, flare, reg_param=0.001):
    return np.sum((make_echo(flare, values) - observation) ** 2) ** .5 + reg_param * np.sum(np.diff(values) ** 2) ** .5


res = optimize.minimize(opt_func, np.ones(len(kernel)), args=(echo_measurement, flare_signal.value))

font = {'family': 'normal',
        'size': 10}

matplotlib.rc('font', **font)

f, ax = plt.subplots(ncols=3, figsize=(12, 4))

ax[0].plot(time_series[:-1], flare_signal, color='k')
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Signal magnitude")
ax[0].set_title("Input flare")

ax[1].plot(time_series[:-1], echo_pristine, color='gray', label='Pristine echo', ls='--')
ax[1].plot(time_series[:-1], echo_measurement, color='k', label="Echo with noise")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Signal magnitude")
ax[1].legend(bbox_to_anchor=(0, 0, 1, 0.2), loc="lower left",
             mode="expand", borderaxespad=0, ncol=2)
ax[1].set_title("Signal measured from exoplanet")

ax[2].plot(cadence * np.arange(len(kernel)), res.x / sig_noise_ratio ** 2 * scale_factor.value, color='k',
           label="Estimated kernel")
ax[2].plot(cadence * np.arange(len(kernel)), kernel, color='gray', label="Exact kernel", ls='--')
ax[2].set_title("Planetary system kernel estimate")
ax[2].set_xlabel("Echo lag (s)")
ax[2].set_ylabel("Estimated echo magnitude")
ax[2].legend(bbox_to_anchor=(0, 0, 1, 0.2), loc="lower left",
             mode="expand", borderaxespad=0, ncol=2)

plt.tight_layout()
plt.show(block=True)

#
#
# blur_kernel = signal.windows.blackman(lowpass_width)
# blur_kernel /= np.sum(blur_kernel)
#
# star_intensity = signal.convolve(template, blur_kernel, mode='same')
#
# # plt.plot(template)
# # plt.plot(convolved)
# # plt.show()
#
# planet_kernel = 1/np.linspace(.5, 1, planet_width)**2
# planet_kernel /= np.sum(planet_kernel)
#
# # Convolve 'shape' of planet with star intensity:
# signal_from_planet = np.convolve(star_intensity, planet_kernel, mode='same')
#
# plt.plot(signal_from_planet, label='planet_echo', ls='-')
# plt.plot(star_intensity, label='Star truth', ls='--')
# plt.legend()
# plt.show()
#
# combined_signal = star_intensity[echo_lag:] + planet_background_contrast*signal_from_planet[:-echo_lag]
#
# meanval = np.mean(combined_signal)
#
# noisy_signal = combined_signal / meanval
# noisy_signal *= sig_noise_ratio**2
#
# noisy_signal = np.random.poisson(noisy_signal.astype(int))
# noisy_signal = noisy_signal / sig_noise_ratio**2
# noisy_signal *= meanval
#
# plt.plot(noisy_signal, label='Signal measured', ls='-')
# plt.plot(combined_signal, label='Star truth', ls='--')
# plt.legend()
# plt.show()
#
#
# def autocorrelate_array(data_array,
#                         max_lag: int,
#                         min_lag: int=0) -> np.ndarray:
#     """Computes the unnormalized autocorrelation at multiple lags for a dataset
#
#     Parameters
#     ----------
#     data_array
#         Preprocessed data array for analysis
#     max_lag
#         Largest lag value to compute
#     min_lag
#         Smallest lag value to compute
#
#     Returns
#     -------
#     np.ndarray
#         Array of autocorrelation values for lags in [min_lag, max_lag]
#
#     """
#     data_array = data_array - np.mean(data_array)
#     corr_vals = np.correlate(data_array, data_array, mode='same')
#     # Need to center the data before returning:
#     return corr_vals[len(corr_vals)//2+min_lag:len(corr_vals)//2+max_lag+1]/corr_vals[len(corr_vals)//2]
#
#
# autocorr = autocorrelate_array(noisy_signal, max_lag=2*echo_lag)
#
# f, ax = plt.subplots(ncols=2, figsize=(10, 4))
# ax[0].plot(autocorr)
# ax[0].set_title("Autocorrelation")
# ax[1].plot(planet_kernel)
# ax[1].set_ylim(0, np.max(planet_kernel))
# ax[1].set_title("Planet kernel")
# plt.tight_layout()
# plt.show()
