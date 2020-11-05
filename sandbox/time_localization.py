
import numpy as np
from matplotlib import pyplot as plt
import exoechopy as eep
from astropy import units as u


flare_onset = 1
flare_decay = 1
my_flare = eep.simulate.ParabolicRiseExponentialDecay(u.Quantity(flare_onset, 's'), u.Quantity(flare_decay, 's'))

time_domain = u.Quantity(np.linspace(0, flare_decay*10, 1000), 's')

flare_lc = my_flare.evaluate_over_array(time_domain)

fft = np.fft.rfft(flare_lc)
fft_freqs = np.fft.rfftfreq(len(flare_lc), time_domain[1].value-time_domain[0].value)

f, ax = plt.subplots(ncols=2, figsize=(8, 4))
ax[0].plot(time_domain[:-1], flare_lc)
ax[1].plot(fft_freqs, np.abs(fft))
plt.tight_layout()
plt.show()

