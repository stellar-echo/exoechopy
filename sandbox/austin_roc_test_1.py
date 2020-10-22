from exoechopy.simulate import flares
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from astropy import units as u

plt.style.use("seaborn")


# Relevant functions
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
    return corr_vals[len(corr_vals) // 2 + min_lag:len(corr_vals) // 2 + max_lag + 1] / corr_vals[len(corr_vals) // 2]


# My simulated flare function
def generate_sim_flare_echo(quiescent_flux, onset, decay, flare_amp, echo_strength, arr_size=200, echo=True, plot=False,
                            plot_ac=False):
    """ Generates a simulated exponential flare and echo based on Gaidos (1994) model of
    echo luminosity for an optically thin disk.
    Uses K2 cadence of 30 seconds. Assumes the host star has a disk with an inner radius of 1 AU.

    Parameters:

    quiescent_flux: value of quiescent flux (arbitrary flux units)
    onset: flare onset time (seconds)
    decay: flare decay time (seconds)
    flare_amp: amplitude of flare (arbitrary flux units)
    echo_strength: strength of echo, in terms of a percentage of flare strength. In practice, this is normally about 1% of
    the flare's amplitude, but here this parameter can be tuned for the purpose of refining echo detection algorithms.
    arr_size: size of quiescent flux array
    echo: optional; if False, no echo is generated
    plot: optional; if True, generates a plot of the flare + echo
    plot_ac: optional; if True, generates a simple plot of the autocorrelation function for the flare and echo

    Returns:

    noisy_array: light curve array with flare, echo, and noise
    """

    # Set Random Seed
    # np.random.seed(13)

    # Generate quiescent flux
    quiescent_list = list(np.zeros(arr_size) + quiescent_flux)  # Just a list of numbers, no units

    # Generate flare
    onset_time = onset * u.s
    decay_time = decay * u.s
    init_flare = flares.ExponentialFlare1(onset_time, decay_time)  # Onset and decay in seconds

    # Time domain
    flare_time_domain = np.linspace(0, (30 * arr_size), arr_size) * u.s  # Array in units of seconds

    # Flare intensity
    flare_arr = u.Quantity([init_flare.evaluate_at_time(t) for t in flare_time_domain]).value
    flare_intensity = flare_arr * flare_amp

    # Insert flare into quiescent flux
    pos = 50
    flare_list = list(flare_intensity + quiescent_flux)
    for i in range(len(flare_list)):
        quiescent_list.insert(i + pos, flare_list[i])

    if echo:
        # Create echo
        # echo = [(0.01*i)+((flare_list[0])*0.9) for i in flare_list[0:500]]
        echo = (flare_intensity * echo_strength) + quiescent_flux
        # return(echo)

        # Calculate delay based on Gaidos (1994) model of echo strength
        generate_sim_flare_echo.delay = 219.18 * np.sqrt(init_flare._flare_duration.value) * u.s
        index = round(generate_sim_flare_echo.delay.value / 30)  # The place to insert the echo in the time array

        # Insert the echo at the delay time
        echo_pos = index
        for i in range(len(echo)):
            quiescent_list.insert(i + echo_pos, echo[i])

    # Add noise
    generate_sim_flare_echo.pristine_array = np.asarray(quiescent_list)
    generate_sim_flare_echo.noisy_array = np.random.poisson(generate_sim_flare_echo.pristine_array)

    # Create new time array with new shape
    generate_sim_flare_echo.final_time_array = np.linspace(0, 30 * len(generate_sim_flare_echo.pristine_array),
                                                           len(generate_sim_flare_echo.pristine_array)) * u.s
    if plot:
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(generate_sim_flare_echo.final_time_array, generate_sim_flare_echo.noisy_array, c='red', alpha=0.8,
                 label="With noise")
        plt.plot(generate_sim_flare_echo.final_time_array, generate_sim_flare_echo.pristine_array, c='black', alpha=0.7,
                 label="Pre-noise")
        plt.xlabel("Time (s)")
        plt.ylabel("Flux (e-/s)")
        plt.title("Simulated Flare and Echo with ExoEchoPy \n With Random Poisson Noise")
        plt.legend()
        plt.show()

    if plot_ac:
        pr_fl_echo = generate_sim_flare_echo.pristine_array[0:200]
        pr_ac = autocorrelate_array(pr_fl_echo, max_lag=160)
        plt.figure(figsize=(12, 6))
        plt.plot(pr_ac, c='k', label="Pristine Array AC")
        plt.plot(autocorrelate_array(generate_sim_flare_echo.noisy_array[0:200], max_lag=100), c='b',
                 label="Noisy Array AC", alpha=0.7)
        plt.title("Autocorrelation - Pristine Array & Noisy Array")
        plt.legend()
        plt.show()

    return (generate_sim_flare_echo.noisy_array)

    # Generate ROC Curve


def gen_roc_curve():
    # Generates an ROC Curve based on autocorrelation from flares and echoes and stuff

    # Test echo strengths from 1% to 20%
    range_of_echo_strengths = [round(i, 2) for i in np.arange(0.01, 0.21, 0.01)]

    # False Positive Rate list
    fpr_list = []
    # True Positive Rate list
    tpr_list = []

    for current_echo_strength in range_of_echo_strengths:

        # Flare and echo generation - generated 500 flares without echoes, and 500 with echoes
        flares_echo = []
        for i in range(0, 500):
            fl = generate_sim_flare_echo(100, 30, 15, 300, current_echo_strength)  # sim flare function I wrote
            flares_echo.append(fl)

        flares_no_echo = []
        for i in range(0, 500):
            fl = generate_sim_flare_echo(100, 30, 15, 300, current_echo_strength, echo=False)
            flares_no_echo.append(fl)

        # Autocorrelation on each flare
        fl_ac = []
        for fl in flares_echo:
            ac = autocorrelate_array(fl, max_lag=100)
            fl_ac.append(ac)

        fl_noecho_ac = []
        for fl in flares_no_echo:
            ac = autocorrelate_array(fl, max_lag=100)
            fl_noecho_ac.append(ac)

        # Generate Data Points
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        # If the AC on a flare with an echo is higher than the echo strength being tested, record it as a true positive
        # otherwise, it's a false negative (predicted no echo when echo was there)
        for ac in fl_ac:
            if max(ac[1:]) >= current_echo_strength - 0.01:
                tp += 1
            else:
                fn += 1

        # If the AC on a no-echo flare is higher than the echo strength being tested, record it as a false positive
        # Otherwise, it's a true negative
        for ac in fl_noecho_ac:
            if max(ac[1:]) >= current_echo_strength:
                fp += 1
            else:
                tn += 1

        # Calc sensitivity and specificity for data points
        tpr = tp / (tp + fn)
        fpr = 1 - (tn / (tn + fp))
        print("Data point at echo strength {}:".format(current_echo_strength), (fpr, tpr))
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_list, tpr_list, c="b", label="ROC Curve")
    plt.plot([0, 1], [0, 1], c="k", alpha=0.8, linestyle="dashed", label="Random Predictor")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curve as a Function of Echo Strength")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    gen_roc_curve()
