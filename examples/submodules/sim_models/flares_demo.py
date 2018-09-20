

"""Shows how the flares module functions through examples."""

import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
from exoechopy.simulate.models import flares
from exoechopy.utils import *

color_map = mplcm.copper_r


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def run():
    #  =============================================================  #
    print("""
    It is important to avoid discontinuities after the flare peak, where an echo may occur.
    Many detection filters are sensitive to them, and they can be interpreted as false echoes or alter the analysis.
    Jump discontinuities are killer, so we have implemented an exponential decay that eliminates the jump.
    C1 discontinuities can be a problem as well, so we push them far from the flare peak;  
    ~10 decay constants seems to be adequate for our prototypical flare. 
    """)

    onset_time = 1.5 * u.s
    decay_const = 4 * u.s
    time_domain = np.linspace(-2 * onset_time.value, (onset_time + 11 * decay_const).value, 1000) * u.s

    max_decay_list = [4, 10]
    for max_num_decay in max_decay_list:
        MyFlare = flares.ExponentialFlare1(onset=onset_time, decay=decay_const, max_decay=max_num_decay)
        flare_intensity_list = u.Quantity([MyFlare.evaluate_at_time(ti) for ti in time_domain])
        plt.plot(time_domain, flare_intensity_list,
                 color=color_map(max_num_decay/max(max_decay_list)),
                 lw=1, label="max_decay="+str(max_num_decay))
        plt.arrow((onset_time + max_num_decay * decay_const).value, .25, 0, -.15,
                  head_width=.5, head_length=.05,
                  color=color_map(max_num_decay/max(max_decay_list)))

    plt.xlabel('Time (sec)')
    plt.ylabel('Flare intensity ('+u_labelstr(flare_intensity_list)+")")
    plt.legend()

    plt.title('Prototypical ExponentialFlare1')
    plt.tight_layout()
    plt.show()

    #  =============================================================  #
    print("""
    The delta-function flare narrows as the discretization increases, preserving the total peak count.
    To normalize to flux rather than integrated counts, divide by delta_t
    Depending on discretization, the point may occur before or on the zero-point of the time domain:
    delta_function(t_0, t_1) =  {  1 if 0 in [t_0, t_1)  }
                                {  0 else                }
    """)
    divs = [10, 11, 51]
    MyDeltaFlare = flares.DeltaFlare(500)
    fig, ax_list = plt.subplots(1, len(divs), figsize=(10, 4))
    for di, ax in zip(divs, ax_list):
        times = np.linspace(-5, 5, di)*u.s
        evaluated_flare = MyDeltaFlare.evaluate_over_array(times)
        ax.plot(times, evaluated_flare,
                 label=str(di)+"points", color='k', lw=1,
                 marker='.', drawstyle='steps-post')
        ax.set_title(str(di)+"points")
        ax.tick_params('x', top=True, direction='in')
        print("Time division: ", di, "\tFlare integration: ", np.sum(evaluated_flare)*(times[1]-times[0]),
              "\tExact:", MyDeltaFlare.integrated_counts)
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel(u_labelstr(evaluated_flare))
    fig.suptitle("Delta-function flare discretization")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()

    #  =============================================================  #
    print("""
    Using evaluate_over_array() preserves the total integrated counts, as shown for delta and exponential flares.
    The flares can be count-normalized by dividing by MyFlare.integrated_counts
    The flares can be converted to integrated counts by multiplying by dt
    """)
    divs = [10, 20, 50, 200]
    fig, (ax_ct, ax_ctsec) = plt.subplots(1, 2, figsize=(10, 4))
    MyExponentialFlare = flares.ExponentialFlare1(onset=onset_time, decay=decay_const)
    for di in divs:
        times = np.linspace(-5 * decay_const.value, (onset_time + 10 * decay_const).value, di)*u.s
        evaluated_flare = MyExponentialFlare.evaluate_over_array(times)
        integrated_flare = evaluated_flare*(times[1]-times[0])
        ax_ct.plot(times, integrated_flare,
                   label=str(di)+"points",
                   marker='.', drawstyle='steps-post')
        flare_rate = evaluated_flare
        ax_ctsec.plot(times, flare_rate,
                      label=str(di)+"points",
                      marker='.', drawstyle='steps-post')
        print("Time division:", di, "\tFlare counts:", np.sum(evaluated_flare)*(times[1]-times[0]),
              "\tExact:", MyExponentialFlare.integrated_counts)

    ax_ct.set_xlabel("Time (s)")
    ax_ct.set_ylabel(u_labelstr(integrated_flare))
    ax_ct.set_title("Integrated counts")
    ax_ct.legend()

    ax_ctsec.set_xlabel("Time (s)")
    ax_ctsec.set_ylabel(u_labelstr(flare_rate))
    ax_ctsec.set_title("Counts/second")
    ax_ctsec.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Exponential flare discretization")

    plt.show()

# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #


if __name__ == "__main__":

    run()



