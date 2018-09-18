

"""Shows how the active_regions module functions through examples."""

import numpy as np
from astropy import units as u
from scipy import stats
import matplotlib.pyplot as plt
from exoechopy.visualize.standard_3d_plots import *
from exoechopy.simulate.models import flares, active_regions
from exoechopy.utils.plottables import PointCloud


def run():
    print("""
    Regions are a Class that supports determining where on a star a flare occurs.
    They can be deterministic, such as a single point, or probability distributions.
    """)

    min_long = np.pi / 4
    max_long = 2 * np.pi - np.pi / 4
    min_lat = np.pi / 4
    max_lat = min_lat + np.pi / 6
    MyRegion = active_regions.Region([min_long, max_long], [min_lat, max_lat])

    number_of_flares = 1000

    vect_points = MyRegion.gen_xyz_vectors(number_of_flares)

    MyPointCloud = PointCloud(vect_points, point_color="k", display_marker='.', point_size=3, linewidth=0,
                              name="Flare locations")

    ax_dic = scatter_plot_3d(MyPointCloud, savefile='hold')

    plot_sphere(ax_dic['ax'], rad=.99, sphere_color='y')

    set_3d_axes_equal(ax_dic['ax'])

    plt.suptitle("Region demo")
    plt.legend()
    plt.show()

    #  =============================================================  #

    print("""FlareActivity allows modeling multiple flares without requiring an explicit time or regional dependence.
    The FlareActivity class can be passed to the ActiveRegion with a Region to produce a full 3d model of the flares.
    It allows the use of custom Flare instances and can handle different probability generators for each argument.
    """)

    np.random.seed(101)

    num_test_flares = 6
    MyDeltaFlareActivity = active_regions.FlareActivity(flares.DeltaFlare, intensity_pdf=[1, 10], name='DeltaFlare demo')

    delta_flare_collection = MyDeltaFlareActivity.generate_n_flares(num_test_flares)
    print("Number of flares: ", delta_flare_collection.num_flares)
    print("All flare intensities: ", delta_flare_collection.all_flare_intensities)
    print("All flare times (uninitialized): ", delta_flare_collection.all_flare_times)
    print("All flare vectors (uninitialized): ", delta_flare_collection.all_flare_vectors)

    fig, ax_list = plt.subplots(1, num_test_flares, figsize=(10, 4))
    time_scale = 10 * u.s
    num_plot_points = 20
    times = np.linspace(-time_scale.value / 2, time_scale.value / 2, num_plot_points)

    for flare, flare_mag, ax in zip(delta_flare_collection.all_flares,
                                    delta_flare_collection.all_flare_intensities,
                                    ax_list):
        integrated_flare = flare.evaluate_over_array_lw(times) * flare_mag
        ax.plot(times, integrated_flare,
                color='.4', lw=1, drawstyle='steps-post',
                marker='s', markersize=3, markerfacecolor='k', markeredgewidth=0)
        ax.tick_params('x', top=True, direction='in')
        ax.set_ylim(0, 1.1 * max(delta_flare_collection.all_flare_intensities))
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Counts')

    fig.suptitle("DeltaFlare FlareActivity demo")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    #  =============================================================  #

    print("""To pass a special set of flare_kwargs, add them when declaring the FlareActivity:  
    For ExponentialFlare1, has keyworks onset= and decay=, so replace with onset_pdf and decay_pdf.
    FlareActivity will recognize '_pdf' as the indicator of a probability density function and will remove '_pdf' 
    from the keyword string before passing to the Flare class.  
    If '_pdf' is not included, it will pass the entire string (so 'onset' and 'onset_pdf' will behave the same), 
    but the best practice is to explicitly remind readers of your code that it's 
    a probability density function generator, not a single value, that is being passed.
    To explicitly include a unit to a frozen pdf, pass a tuple of the pdf with the irreducible unit as the second arg. 

    """)

    num_test_flares = 6
    MyExpFlareActivity = active_regions.FlareActivity(flares.ExponentialFlare1,
                                                      intensity_pdf=stats.expon(scale=10 * u.s),
                                                      onset_pdf=[1, 4] * u.s,
                                                      decay_pdf=(stats.rayleigh(scale=5), u.s),
                                                      name='ExponentialFlare1 demo')

    exp_flare_collection = MyExpFlareActivity.generate_n_flares(num_test_flares)
    print("Number of flares: ", exp_flare_collection.num_flares)
    print("All flare intensities: ", exp_flare_collection.all_flare_intensities)
    print("All flare times (uninitialized): ", exp_flare_collection.all_flare_times)
    print("All flare vectors (uninitialized): ", exp_flare_collection.all_flare_vectors)

    fig, ax_list = plt.subplots(1, num_test_flares, figsize=(10, 4))
    time_scale = 30 * u.s
    num_plot_points = 40
    times = np.linspace(-time_scale.value / 6, 5 * time_scale.value / 6, num_plot_points)

    for flare, flare_mag, ax in zip(exp_flare_collection.all_flares,
                                    exp_flare_collection.all_flare_intensities,
                                    ax_list):
        # Normalize to integrated flare peak for display purposes:
        integrated_flare = flare.evaluate_over_array_lw(times) * flare_mag / np.max(
            flare.evaluate_over_array_lw(times))
        ax.plot(times, integrated_flare,
                color='.4', lw=1, drawstyle='steps-post',
                marker='s', markersize=3, markerfacecolor='k', markeredgewidth=0)
        ax.tick_params('x', top=True, direction='in')
        ax.set_ylim(0, 1.1 * max(exp_flare_collection.all_flare_intensities))
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Counts')

    fig.suptitle("ExponentialFlare1 FlareActivity demo")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    #  =============================================================  #

    print("""
    The ActiveRegion is basically a holder to align FlareActivity and Regions.
    Multiple types of FlareActivity can be present, they will be selected based on a probability ratio.
    To get the FlareCollection from the region, call the instance.
    """)

    # Without explicit times:
    num_test_flares = 6
    MyActiveRegion = active_regions.ActiveRegion(flare_activity=[MyExpFlareActivity, MyDeltaFlareActivity],
                                                 num_flares=num_test_flares,
                                                 region=MyRegion,
                                                 flare_activity_ratios=[.6, .4])
    ar_flare_collection = MyActiveRegion()
    print("Number of flares: ", ar_flare_collection.num_flares)
    print("All flare intensities: ", ar_flare_collection.all_flare_intensities)
    print("All flare times (uninitialized): ", ar_flare_collection.all_flare_times)
    print("All flare vectors: ", ar_flare_collection.all_flare_vectors)

    fig, ax_list = plt.subplots(1, num_test_flares, figsize=(12, 4))
    time_scale = 30 * u.s
    num_plot_points = 40
    times = np.linspace(-time_scale.value / 6, 5 * time_scale.value / 6, num_plot_points)

    for flare, flare_mag, ax in zip(ar_flare_collection.all_flares,
                                    ar_flare_collection.all_flare_intensities,
                                    ax_list):
        # Normalize to integrated flare peak for display purposes:
        integrated_flare = flare.evaluate_over_array_lw(times) * flare_mag / np.max(
            flare.evaluate_over_array_lw(times))
        ax.plot(times, integrated_flare,
                color='.4', lw=1, drawstyle='steps-post',
                marker='s', markersize=3, markerfacecolor='k', markeredgewidth=0)
        ax.tick_params('x', top=True, direction='in')
        ax.set_ylim(0, 1.1 * max(ar_flare_collection.all_flare_intensities))
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Counts')

    fig.suptitle("MyActiveRegion ActiveRegion demo")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    MyPointCloud = PointCloud(ar_flare_collection.all_flare_vectors, point_color="k", display_marker='.',
                              point_size=4,
                              linewidth=0,
                              name="Flare locations")

    ax_dic = scatter_plot_3d(MyPointCloud, savefile='hold')

    plot_sphere(ax_dic['ax'], rad=.99, sphere_color='y')

    set_3d_axes_equal(ax_dic['ax'])

    plt.suptitle("MyActiveRegion flare locations")
    plt.legend()
    plt.show()

    #  =============================================================  #

    print("""
    Time scales can be explicitly given, as well, which results in an unspecified number of flares that occur 
    according to occurrence_freq_pdf
    """)

    # With explicit times:
    observation_duration = .2 * u.hr
    print("observation_duration.to(u.s): ", observation_duration.to(u.s))
    MyActiveRegion = active_regions.ActiveRegion(flare_activity=[MyExpFlareActivity, MyDeltaFlareActivity],
                                                 occurrence_freq_pdf=10 / observation_duration,
                                                 region=MyRegion,
                                                 flare_activity_ratios=[.5, .5])
    ar_flare_collection = MyActiveRegion(observation_duration)
    print("Number of flares: ", ar_flare_collection.num_flares)
    print("All flare intensities: ", ar_flare_collection.all_flare_intensities)
    print("All flare times: ", ar_flare_collection.all_flare_times)
    print("All flare vectors: ", ar_flare_collection.all_flare_vectors)

    fig = plt.figure(figsize=(14, 4))
    ax = fig.add_subplot(111)

    num_plot_points = 2000
    all_time = np.linspace(0, observation_duration.to(u.s), num_plot_points)
    lightcurve = np.zeros(num_plot_points)

    dt = all_time[1] - all_time[0]
    print("dt: ", dt)
    num_plot_points_per_flare = int(40/dt.value)

    for flare, flare_mag, flare_time in zip(ar_flare_collection.all_flares,
                                            ar_flare_collection.all_flare_intensities,
                                            ar_flare_collection.all_flare_times):
        # Normalize to integrated flare peak for display purposes:
        i0 = int(flare_time/dt)
        i1 = int(flare_time/dt)+num_plot_points_per_flare
        local_flare_times = all_time[i0: i1]
        integrated_flare = flare.evaluate_over_array(local_flare_times-flare_time) * flare_mag
        lightcurve[i0: i1] += integrated_flare.value
        ax.scatter([flare_time.value], [0], color='orange', marker='^')

    ax.plot(all_time, lightcurve,
            color='.4', lw=1, drawstyle='steps-post',
            marker='s', markersize=3, markerfacecolor='k', markeredgewidth=0)
    ax.tick_params('x', top=True, direction='in')
    # ax.set_ylim(0, 1.1 * max(ar_flare_collection.all_flare_intensities))
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Counts')

    fig.suptitle("MyActiveRegion ActiveRegion synthetic light curve demo")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ******************************************************************************************************************** #
# ************************************************  TEST & DEMO CODE  ************************************************ #

if __name__ == "__main__":

    run()
