# Simulations using Gaidos (1994) model of echo strength/time delay

from exoechopy.simulate import flares
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
plt.style.use("seaborn")


# Generate simulated flares and echoes
def generate_sim_flare_echo(quiescent_flux, onset, decay, flare_amp, arr_size = 200):
    ''' Generates a simulated exponential flare and echo based on Gaidos (1994) model of echo luminosity.
     Uses K2 cadence of 30 seconds. Assumes echo luminosity is 1% of flare luminosity. '''
    
    # Set Random Seed 
    #np.random.seed(13)
    
    # Generate quiescent flux
    quiescent_list = list(np.zeros(arr_size) + quiescent_flux) # Just a list of numbers, no units
    
    #Generate flare
    onset_time = onset * u.s
    decay_time = decay * u.s
    init_flare = flares.ExponentialFlare1(onset_time, decay_time) # Onset and decay in seconds
    
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
    
    # Plot
    #plt.figure(figsize = (12,6))
    #plt.plot(generate_sim_flare_echo.final_time_array, generate_sim_flare_echo.noisy_array, c = 'red', alpha = 0.8, label = "With noise")
    #plt.plot(generate_sim_flare_echo.final_time_array, generate_sim_flare_echo.pristine_array, c = 'black', alpha = 0.7, label = "Pre-noise")
    #plt.xlabel("Time (s)")
    #plt.ylabel("Flux (e-/s)")
    #plt.title("Simulated Flare and Echo with ExoEchoPy \n With Random Poisson Noise")
    #plt.legend()
    #plt.xlim(0,5000)
    #plt.ylim(395,450)
    #print("Flare Duration:", init_flare._flare_duration)
    #print("Time Delay based on Gaidos:", generate_sim_flare_echo.delay)
    #print(generate_sim_flare_echo.pristine_array)
    #plt.axhline(y = 1300)
    #plt.show()
    
    return(generate_sim_flare_echo.noisy_array)
    
generate_sim_flare_echo(1000, 30, 15, 3000)

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

# Generate identical flares to add, each with independent random noise
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

# Generate autocorrelation arrays with sums of flare arrays
a20 = autocorrelate_array(sum(twenty_flares), max_lag = 100)
a75 = autocorrelate_array(sum(seventy_five_flares), max_lag = 100)
a300 = autocorrelate_array(sum(three_hundred_flares), max_lag = 100)

# Stacked AC plot
fig, ax = plt.subplots(3, figsize = (10,10), sharex = True, sharey = True)

ax[0].set_title("Summed Autocorrelation")
ax[0].plot(a20, c = "r", label = "n = 20")
#ax[0].axvline(x = 48, linestyle = "dotted", c = "k", alpha = 0.7)
ax[1].plot(a75, c = "b", label = "n = 75")
#ax[1].axvline(x = 48, linestyle = "dotted", c = "k", alpha = 0.7)
ax[2].plot(a300, c = "k", label = "n = 300")
#ax[2].axvline(x = 48, linestyle = "dotted", c = "k", alpha = 0.7)
plt.xlabel("Lag")
ax[0].legend()
ax[1].legend()
ax[2].legend()

fig.subplots_adjust(hspace=0.1)
