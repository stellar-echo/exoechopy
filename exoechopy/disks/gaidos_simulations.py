# Simulations and analysis of flares and disk echoes using Gaidos (1994) model of an optically thin disk
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const
from exoechopy.simulate import flares
from sklearn import metrics


# Autocorrelation Function
def autocorrelate_array(data_array,
                        max_lag: int,
                        min_lag: int = 0) -> np.ndarray:
    """Computes the unnormalized autocorrelation at multiple lags for a dataset
    Parameters
    ----------
    data_array
        Preprocessed data array for analysis
    max_lag
        Largest lag value to compute
    min_lag
        Smallest lag value to compute
    Returns
    -------
    np.ndarray
        Array of autocorrelation values for lags in [min_lag, max_lag]
    """
    data_array = data_array - np.mean(data_array)
    corr_vals = np.correlate(data_array, data_array, mode='same')
    # Need to center the data before returning:
    return corr_vals[len(corr_vals)//2+min_lag:len(corr_vals)//2+max_lag+1]/corr_vals[len(corr_vals)//2]


class SimFlare:
    """ Generates simulated flares and disk echoes. Uses code from ExoEchoPy to generate flares, and generates disk
    echoes based on Gaidos (1994) model of echo strength for an optically thin disk. Has member functions that visualize
    and run detection algorithms such as autocorrelation, along with evaluation via ROC curve."""
    
    def __init__(self, echo_strength, quiescent_flux, onset, decay, flare_amp, inner_radius, pristine_array=None,
                 noisy_array=None, final_time_array=None, index=None,  arr_size=200, echo=True):
        """
        Parameters:

        echo_strength: strength of the echo, as a fraction of the flare luminosity. In practice, this parameter is
        normally about 1%, but it can be tuned here for the purpose of refining detection algorithms.
        quiescent_flux: value of quiescent flux (arbitrary flux units)
        onset: flare onset time (seconds)
        decay: flare decay time (seconds)
        flare_amp: amplitude of flare (arbitrary flux units)
        inner_radius: inner radius of the disk (AU) used to calculate echo delay times
        arr_size: length of quiescent flux array
        echo: optional; if False, no echo is generated
        """
        self.quiescent_flux = quiescent_flux
        self.onset = onset
        self.decay = decay
        self.flare_amp = flare_amp
        self.inner_radius = inner_radius
        self.arr_size = arr_size
        self.echo = echo
        self.echo_strength = echo_strength
        self.pristine_array = pristine_array
        self.noisy_array = noisy_array
        self.final_time_array = final_time_array
        self.index = index

    def gen_sim_flare_echo(self):
        """ Sim flare and echo function. The bread and butter. """

        # Generate quiescent flux - the background/baseline
        quiescent_list = list(np.zeros(self.arr_size) + self.quiescent_flux)  # Just a list of numbers, no units

        # Generate flare
        onset_time = self.onset * u.s
        decay_time = self.decay * u.s
        init_flare = flares.ExponentialFlare1(onset_time, decay_time)  # Onset and decay in seconds

        # Time domain
        flare_time_domain = np.linspace(0, (60 * self.arr_size), self.arr_size)*u.s  # Array in units of seconds

        # Flare intensity
        flare_arr = u.Quantity([init_flare.evaluate_at_time(t) for t in flare_time_domain]).value
        flare_intensity = flare_arr * self.flare_amp
        
        # Insert flare into quiescent flux (the position is arbitrary and can be changed without affecting anything)
        pos = 50
        flare_list = list(flare_intensity + self.quiescent_flux)
        for i in range(len(flare_list)):
            quiescent_list.insert(i + pos, flare_list[i])
        
        if self.echo:
            # Create echo
            _echo_ = (flare_intensity * self.echo_strength) + self.quiescent_flux

            # Calculate delay based on Gaidos (1994) model of echo strength
            # this value was calculated assuming an optically thin disk
            dist = self.inner_radius * (1.496e11 * u.m)
            tau = (dist/const.c).value * u.s
            delay = (tau.value/(np.sqrt(self.echo_strength) * np.sqrt(tau.value/init_flare._flare_duration.value)))*u.s
           
            # The place to insert the echo in the time array
            self.index = pos + int(round(delay.value/60))

            # Insert the echo at the delay time
            for i in range(len(_echo_)):
                quiescent_list.insert(i + self.index, _echo_[i])

        # Add noise based on a Poisson distribution
        self.pristine_array = np.asarray(quiescent_list)
        self.noisy_array = np.random.poisson(self.pristine_array)

        # Create new time array with new shape
        self.final_time_array = np.linspace(0, 60*len(self.pristine_array), len(self.pristine_array)) * u.s

        return self.noisy_array

    def plot(self, with_pristine=False):
        # Plotting function. If with_pristine = True, plots the pristine light curve as well as the noisy light curve.
        plt.figure(figsize=(12, 6))
        plt.plot(self.final_time_array, self.noisy_array, c='red', alpha=0.8, label="With noise")
        
        if with_pristine:
            plt.plot(self.final_time_array, self.pristine_array, c='black', alpha=0.7, label="Pre-noise")
        
        plt.xlabel("Time (s)")
        plt.ylabel("Flux (e-/s)")
        plt.title("Simulated Flare and Echo with ExoEchoPy \n With Random Poisson Noise")
        plt.legend()
        plt.show()
        
    def plot_ac(self, with_pristine=False):
        # Plot the autocorrelation function for a SimFlare object.

        # Generate pristine AC
        pr_fl_echo = self.pristine_array[0:200]
        pr_ac = autocorrelate_array(pr_fl_echo, max_lag=160)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(autocorrelate_array(self.noisy_array[0:200], max_lag=100), c='b', label="Noisy Array AC", alpha=0.7)

        if with_pristine:
            plt.plot(pr_ac, c='k', label="Pristine Array AC")

        plt.title("Autocorrelation - Pristine Array & Noisy Array")
        plt.axvline(x=self.index-50)
        plt.legend()
        plt.show()

    def plot_ac_summed(self):
        # Generates autocorrelation plot of repeat flare events. Uses the mean autocorrelation value at each lag
        # Plots the 90th and 95th percentiles of all autocorrelations as well

        # Generate identical flares to add
        twenty_flares = []
        for i in range(0, 20):
            sim_fl = SimFlare(self.echo_strength, self.quiescent_flux, self.onset, self.decay, self.flare_amp)
            sim_flare = sim_fl.gen_sim_flare_echo()
            twenty_flares.append(sim_flare[0:200])

        seventy_five_flares = []
        for i in range(0, 75):
            sim_fl = SimFlare(self.echo_strength, self.quiescent_flux, self.onset, self.decay, self.flare_amp)
            sim_flare = sim_fl.gen_sim_flare_echo()
            seventy_five_flares.append(sim_flare[0:200])

        three_hundred_flares = []
        for i in range(0, 300):
            sim_fl = SimFlare(self.echo_strength, self.quiescent_flux, self.onset, self.decay, self.flare_amp)
            sim_flare = sim_fl.gen_sim_flare_echo()
            three_hundred_flares.append(sim_flare[0:200])

        # Generate autocorrelations
        ac_twenty = []
        for flare in twenty_flares:
            ac = autocorrelate_array(flare, max_lag=100)
            ac_twenty.append(ac)

        ac_seventy_five = []
        for flare in seventy_five_flares:
            ac = autocorrelate_array(flare, max_lag=100)
            ac_seventy_five.append(ac)

        ac_three_hundred = []
        for flare in three_hundred_flares:
            ac = autocorrelate_array(flare, max_lag=100)
            ac_three_hundred.append(ac)

        # Generate arrays for plotting means

        # n = 20
        ac_twenty_array = np.zeros((20, 100))  # 20 rows, 100 columns to store each value at each index
        for index, row in enumerate(ac_twenty_array):
            ac_twenty_array[index] = ac_twenty_array[index] + ac_twenty[index]

        # Grab the means and percentiles
        ac_twenty_means = []
        ac_twenty_90 = []
        ac_twenty_95 = []
        for i in range(np.shape(ac_twenty_array)[1]):
            mean = np.mean(ac_twenty_array[:, i])
            ac_twenty_means.append(mean)

            per90 = np.percentile(ac_twenty_array[:, i], 90)
            ac_twenty_90.append(per90)

            per95 = np.percentile(ac_twenty_array[:, i], 95)
            ac_twenty_95.append(per95)

        # n = 75
        ac_75_array = np.zeros((75, 100))  # 20 rows, 100 columns to store each value at each index
        for index, row in enumerate(ac_75_array):
            ac_75_array[index] = ac_75_array[index] + ac_seventy_five[index]

        ac_75_means = []
        ac_75_90 = []
        ac_75_95 = []
        for i in range(np.shape(ac_75_array)[1]):
            mean = np.mean(ac_75_array[:, i])
            ac_75_means.append(mean)

            per90 = np.percentile(ac_75_array[:, i], 90)
            ac_75_90.append(per90)

            per95 = np.percentile(ac_75_array[:, i], 95)
            ac_75_95.append(per95)

        # n = 300
        ac_300_array = np.zeros((300, 100))  # 20 rows, 100 columns to store each value at each index
        for index, row in enumerate(ac_300_array):
            ac_300_array[index] = ac_300_array[index] + ac_three_hundred[index]

        ac_300_means = []
        ac_300_90 = []
        ac_300_95 = []
        for i in range(np.shape(ac_300_array)[1]):
            mean = np.mean(ac_300_array[:, i])
            ac_300_means.append(mean)

            per90 = np.percentile(ac_300_array[:, i], 90)
            ac_300_90.append(per90)

            per95 = np.percentile(ac_300_array[:, i], 95)
            ac_300_95.append(per95)

        # Generate stacked plot - different number of repeat events
        fig, ax = plt.subplots(3, figsize=(10, 10), sharex=True, sharey=True)

        ax[0].set_title("Summed Autocorrelation")
        ax[0].plot(ac_twenty_means, c="k", label="n = 20 - mean")
        ax[0].plot(ac_twenty_90, label="n = 20 - 90th percentile", linestyle="dashed", color="0.5")
        ax[0].plot(ac_twenty_95, label="n = 20 - 95th percentile", linestyle="dotted", color="0.3")
        ax[1].plot(ac_75_means, c="k", label="n = 75 - means")
        ax[1].plot(ac_75_90, label="n = 75 - 90th percentile", linestyle="dashed", color="0.5")
        ax[1].plot(ac_75_95, label="n = 75 - 95th percentile", linestyle="dotted", color="0.3")
        ax[2].plot(ac_300_means, c="k", label="n = 300 - means")
        ax[2].plot(ac_300_90, label="n = 300 - 90th percentile", linestyle="dashed", color="0.5")
        ax[2].plot(ac_300_95, label="n = 300 - 95th percentile", linestyle="dotted", color="0.3")
        plt.xlabel("Lag")
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()

        fig.subplots_adjust(hspace=0.1)

        plt.show()

    def gen_roc(self, n, to_sum=0):
        """For a SimFlare object, generate n flares and generate an ROC curve based on the supplied echo strength.

        :param n: int, determines number of flares used to generate curve
        :param to_sum: int, optional. If not 0, determines the amount of flares to sum to boost the signal of the curve.
        For example, to_sum = 5 would add every 5 flares generated.

        """

        # Start with the control case -- No echoes
        no_echo_flares = []
        for i in range(0, n):
            fl = SimFlare(self.echo_strength, self.quiescent_flux, self.onset, self.decay, self.flare_amp,
                          self.inner_radius, echo=False)
            flare = fl.gen_sim_flare_echo()
            no_echo_flares.append(flare)

        # Perform autocorrelation on each flare
        no_echo_ac = []
        for flare in no_echo_flares:
            ac = autocorrelate_array(flare, max_lag=100)
            no_echo_ac.append(ac)

        # Grab all AC values from the echo index
        echo_indices = []
        for array in no_echo_ac:
            echo_indices.append(array[self.index - 50])

        # Repeat with flares

        # Generate n flares
        echo_flares = []
        for i in range(0, n):
            fl = SimFlare(self.echo_strength, self.quiescent_flux, self.onset, self.decay, self.flare_amp,
                          self.inner_radius)
            flare = fl.gen_sim_flare_echo()
            echo_flares.append(flare)

        # Run autocorrelation
        echo_ac = []
        for flare in echo_flares:
            ac = autocorrelate_array(flare, max_lag=100)
            echo_ac.append(ac)

        # Grab echo index
        true_echo_indices = []
        for array in echo_ac:
            true_echo_indices.append(array[self.index - 50])

        # Grab true positives and false positives
        tp = []
        fp = []

        for ac_val in true_echo_indices:
            n_true_hits = np.sum(true_echo_indices >= ac_val)/n
            n_false_hits = np.sum(echo_indices >= ac_val)/n
            tp.append(n_true_hits)
            fp.append(n_false_hits)

        # Calculate AUC score
        fp.sort()
        tp.sort()

        auc = metrics.auc(fp, tp)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve - {}% Echo Strength \n 10K Flares".format(self.echo_strength*100))
        plt.plot(fp, tp, c="b", alpha=0.7, label="ROC Curve, AUC = {}".format(round(auc, 2)))
        plt.plot([0, 1], [0, 1], c='k', linestyle="dashed", label="random predictor")

        if to_sum > 0:
            no_echo_flare_sums = np.add.reduceat(no_echo_flares, np.arange(0, len(no_echo_flares), to_sum))
            no_echo_flare_sums = list(no_echo_flare_sums)

            no_echo_sums_ac = []
            for flare in no_echo_flare_sums:
                ac = autocorrelate_array(flare, max_lag=100)
                no_echo_sums_ac.append(ac)

            no_echo_sums_indices = []
            for arr in no_echo_sums_ac:
                no_echo_sums_indices.append(arr[self.index - 50])

            echo_flare_sums = np.add.reduceat(echo_flares, np.arange(0, len(echo_flares), to_sum))
            echo_flare_sums = list(echo_flare_sums)

            echo_sums_ac = []
            for flare in echo_flare_sums:
                ac = autocorrelate_array(flare, max_lag=100)
                echo_sums_ac.append(ac)

            true_echo_sums_indices = []
            for arr in echo_sums_ac:
                true_echo_sums_indices.append(arr[self.index - 50])

            tp_sum = []
            fp_sum = []

            for ac_val in true_echo_sums_indices:
                n_true = np.sum(true_echo_sums_indices >= ac_val) / (n / to_sum)
                n_false = np.sum(no_echo_sums_indices >= ac_val) / (n / to_sum)
                tp_sum.append(n_true)
                fp_sum.append(n_false)

            # If the last data point in the summed curve is not (1,1), append it
            if tp_sum[-1] != 1:
                tp_sum.append(1)

            if fp_sum[-1] != 1:
                fp_sum.append(1)

            fp_sum.sort()
            tp_sum.sort()

            auc_sum = metrics.auc(fp_sum, tp_sum)

            plt.plot(fp_sum, tp_sum, c="r", label="Every %s Flares Summed, AUC = %s" % (to_sum, (round(auc_sum, 2))))

        plt.legend(loc=4)
        plt.show()

    def plot_auc(self):

        n_flares = [0, 9, 16, 25, 36, 49, 64, 81, 100]
        auc = [0.57, 0.7, 0.77, 0.81, 0.81, 0.89, 0.91, 0.94, 0.97]

        plt.figure(figsize=(12, 6))
        plt.xlabel("Number of Flares Summed")
        plt.ylabel("AUC Value")
        plt.title("Summed Flares vs. AUC value")
        plt.plot(n_flares, auc, c="k")
        plt.ylim(0.5, 1)
        plt.show()


ex = SimFlare(echo_strength=0.01, quiescent_flux=100, onset=30, decay=15,
              arr_size=200, flare_amp=3000, inner_radius=0.01)
ex.gen_sim_flare_echo()
ex.plot(with_pristine=True)
ex.plot_ac(with_pristine=True)
ex.gen_roc(10000, 4)
ex.plot_auc()


