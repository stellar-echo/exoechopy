# Simulations using Gaidos (1994) model of echo strength/time delay for an optically thin disk

from exoechopy.simulate import flares
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
plt.style.use("seaborn")

# The autocorrelation function 
def autocorrelate_array(data_array,
                        max_lag: int,
                        min_lag: int=0) -> np.ndarray:
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

def generate_sim_flare_echo(quiescent_flux, onset, decay, flare_amp, arr_size = 200, echo = True, plot = False,
                           plot_ac = False):
    ''' Generates a simulated exponential flare and echo based on Gaidos (1994) model of echo luminosity for an optically thin disk.
     Uses K2 cadence of 30 seconds. Assumes echo luminosity is 10% of flare luminosity, for purposes of testing detection algorithms. Will be toned down to a more
     realistic 1% later. 
     
     Parameters:
     
     quiescent_flux: value of quiescent flux (arbitrary flux units)
     onset: flare onset time (seconds)
     decay: flare decay time (seconds)
     flare_amp: amplitude of flare (arbitrary flux units)
     arr_size: size of quiescent flux array
     echo: optional; if False, no echo is generated
     plot: optional; if True, generates a plot of the flare + echo 
     plot_ac: optional; if True, generates a simple plot of the autocorrelation function for the flare and echo 
     
     Returns:
     
     noisy_array: light curve array with flare, echo, and noise
     '''
    
    # Set Random Seed 
    #np.random.seed(13)
    
    # Generate quiescent flux
    quiescent_list = list(np.zeros(arr_size) + quiescent_flux) # Just a list of numbers, no units
    
    #Generate flare
    onset_time = onset*u.s
    decay_time = decay*u.s
    init_flare = flares.ParabolicRiseExponentialDecay(onset_time, decay_time) # Onset and decay in seconds
    
    # Time domain
    flare_time_domain = np.linspace(0, (30*arr_size), arr_size)*u.s # Array in units of seconds
    
    # Flare intensity
    flare_arr = u.Quantity([init_flare.evaluate_at_time(t) for t in flare_time_domain]).value
    flare_intensity = flare_arr*flare_amp
    
    # Insert flare into quiescent flux
    pos = 50 
    flare_list = list(flare_intensity + quiescent_flux)
    for i in range(len(flare_list)):
        quiescent_list.insert(i + pos, flare_list[i])
    
    
    if echo:
        # Create echo
        #echo = [(0.01*i)+((flare_list[0])*0.9) for i in flare_list[0:500]]
        echo_amp = 0.1
        echo = (flare_intensity*echo_amp) + quiescent_flux
        #return(echo)
    
        # Calculate delay based on Gaidos (1994) model of echo strength
        generate_sim_flare_echo.delay = 219.18*np.sqrt(init_flare._flare_duration.value) * u.s
        index = round(generate_sim_flare_echo.delay.value/30) # The place to insert the echo in the time array
    
        # Insert the echo at the delay time
        echo_pos = index 
        for i in range(len(echo)):
            quiescent_list.insert(i+echo_pos, echo[i])
        
    # Add noise
    generate_sim_flare_echo.pristine_array = np.asarray(quiescent_list)
    generate_sim_flare_echo.noisy_array = np.random.poisson(generate_sim_flare_echo.pristine_array).astype(float)
    
    # Create new time array with new shape 
    generate_sim_flare_echo.final_time_array = np.linspace(0, 30*len(generate_sim_flare_echo.pristine_array),
                                                           len(generate_sim_flare_echo.pristine_array)) * u.s    
    if plot:
        # Plot
        plt.figure(figsize = (12,6))
        plt.plot(generate_sim_flare_echo.final_time_array, generate_sim_flare_echo.noisy_array, c = 'red', alpha = 0.8, label = "With noise")
        plt.plot(generate_sim_flare_echo.final_time_array, generate_sim_flare_echo.pristine_array, c = 'black', alpha = 0.7, label = "Pre-noise")
        plt.xlabel("Time (s)")
        plt.ylabel("Flux (e-/s)")
        plt.title("Simulated Flare and Echo with ExoEchoPy \n With Random Poisson Noise")
        plt.legend()
        plt.show()
        
    if plot_ac:
        pr_fl_echo = generate_sim_flare_echo.pristine_array[0:200]
        pr_ac = autocorrelate_array(pr_fl_echo, max_lag = 160)
        plt.figure(figsize=(12,6))
        plt.plot(pr_ac, c = 'k', label = "Pristine Array AC")
        plt.plot(autocorrelate_array(generate_sim_flare_echo.noisy_array[0:200], max_lag = 100), c = 'b', label = "Noisy Array AC", alpha = 0.7)
        plt.title("Autocorrelation - Pristine Array & Noisy Array")
        plt.legend()
        plt.show()
        
    return(generate_sim_flare_echo.noisy_array)
    


def plot_ac():
    # Generates autocorrelation plot of repeat flare events. Uses the mean autocorrelation value at each lag 
    # Now comes with 90th and 95th percentiles as well
    
    
    
    # Generate identical flares to add
    twenty_flares = []
    for i in range(0,20):
        sim_fl = generate_sim_flare_echo(100, 30, 15, 300)
        twenty_flares.append(sim_fl[0:200])
    
    seventy_five_flares = []
    for i in range(0,75):
        sim_fl = generate_sim_flare_echo(100, 30, 15, 300)
        seventy_five_flares.append(sim_fl[0:200])
    
    three_hundred_flares = []
    for i in range(0, 300):
        sim_fl = generate_sim_flare_echo(100, 30, 15, 300)
        three_hundred_flares.append(sim_fl[0:200])
        
        
    # Generate autocorrelations
    ac_twenty = []
    for flare in twenty_flares:
        ac = autocorrelate_array(flare, max_lag = 100)
        ac_twenty.append(ac)
    
    ac_seventy_five = []
    for flare in seventy_five_flares:
        ac = autocorrelate_array(flare, max_lag = 100)
        ac_seventy_five.append(ac)
    
    ac_three_hundred = []
    for flare in three_hundred_flares:
        ac = autocorrelate_array(flare, max_lag = 100)
        ac_three_hundred.append(ac)
        
        
    # Generate arrays for plotting means
    
    # n = 20
    ac_twenty_array = np.zeros((20, 100)) # 20 rows, 100 columns to store each value at each index
    for index, row in enumerate(ac_twenty_array):
        ac_twenty_array[index] = ac_twenty_array[index] + ac_twenty[index]
    
    # Grab the means and percentiles
    ac_twenty_means = []
    ac_twenty_90 = []
    ac_twenty_95 = []
    for i in range(np.shape(ac_twenty_array)[1]):
        mean = np.mean(ac_twenty_array[:,i])
        ac_twenty_means.append(mean)
        
        per90 = np.percentile(ac_twenty_array[:,i], 90)
        ac_twenty_90.append(per90)
        
        per95 = np.percentile(ac_twenty_array[:,i], 95)
        ac_twenty_95.append(per95)

    #n = 75
    ac_75_array = np.zeros((75, 100)) # 20 rows, 100 columns to store each value at each index
    for index, row in enumerate(ac_75_array):
        ac_75_array[index] = ac_75_array[index] + ac_seventy_five[index]
    
    ac_75_means = []
    ac_75_90 = []
    ac_75_95 = []
    for i in range(np.shape(ac_75_array)[1]):
        mean = np.mean(ac_75_array[:,i])
        ac_75_means.append(mean)
        
        per90 = np.percentile(ac_75_array[:,i], 90)
        ac_75_90.append(per90)
        
        per95 = np.percentile(ac_75_array[:,i], 95)
        ac_75_95.append(per95)
        
    #n = 300
    ac_300_array = np.zeros((300, 100)) # 20 rows, 100 columns to store each value at each index
    for index, row in enumerate(ac_300_array):
        ac_300_array[index] = ac_300_array[index] + ac_three_hundred[index]
    
    ac_300_means = []
    ac_300_90 = []
    ac_300_95 = []
    for i in range(np.shape(ac_300_array)[1]):
        mean = np.mean(ac_300_array[:,i])
        ac_300_means.append(mean)
        
        per90 = np.percentile(ac_300_array[:,i], 90)
        ac_300_90.append(per90)
        
        per95 = np.percentile(ac_300_array[:,i], 95)
        ac_300_95.append(per95)
    
    # Generate stacked plot - different number of repeat events
    fig, ax = plt.subplots(3, figsize = (10,10), sharex = True, sharey = True)

    ax[0].set_title("Summed Autocorrelation")
    ax[0].plot(ac_twenty_means, c = "k", label = "n = 20 - mean")
    ax[0].plot(ac_twenty_90, label = "n = 20 - 90th percentile", linestyle = "dashed", color = "0.5")
    ax[0].plot(ac_twenty_95, label = "n = 20 - 95th percentile", linestyle = "dotted", color = "0.3")
    ax[1].plot(ac_75_means, c = "k", label = "n = 75 - means")
    ax[1].plot(ac_75_90, label = "n = 75 - 90th percentile", linestyle = "dashed", color = "0.5")
    ax[1].plot(ac_75_95, label = "n = 75 - 95th percentile", linestyle = "dotted", color = "0.3")
    ax[2].plot(ac_300_means, c = "k", label = "n = 300 - means")
    ax[2].plot(ac_300_90, label = "n = 300 - 90th percentile", linestyle = "dashed", color = "0.5")
    ax[2].plot(ac_300_95, label = "n = 300 - 95th percentile", linestyle = "dotted", color = "0.3")
    plt.xlabel("Lag")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    fig.subplots_adjust(hspace=0.1)
    
    plt.show()


generate_sim_flare_echo(1000, 30, 15, 3000, plot_ac = True)
plot_ac()
