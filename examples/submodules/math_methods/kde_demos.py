import numpy as np
import matplotlib.pyplot as plt
from exoechopy.analyze import *
from scipy.stats import rayleigh

my_distribution = rayleigh()

np.random.seed(10)
test_data = my_distribution.rvs(size=2)
x_data = np.linspace(-1, max(test_data)+1, 10)
x_data_fine = np.linspace(-1, max(test_data)+1, 1000)
dx = x_data[1]-x_data[0]

gausskde = GaussianKDE(test_data)
gauss_yvals = gausskde(x_data)
print("Approximate integral of Gaussian KDE: ", np.sum(gauss_yvals)*dx)

tophatkde = TophatKDE(test_data)

tophat_yvals = tophatkde(x_data)
tophat_yvals_fine = tophatkde(x_data_fine)
print("Approximate integral of Tophat KDE: ", np.sum(tophat_yvals)*dx)

lower = 0
upper = 1.5
print("Computed tophat integral from ", lower, "to", upper, ": ", tophatkde.integrate_between(lower, upper))
print("Computed gaussian integral from", lower, "to", upper, ": ", gausskde.integrate_between(lower, upper))
lower = -1
upper = 5
print("Computed tophat integral from ", lower, "to", upper, ": ", tophatkde.integrate_between(lower, upper))
print("Computed gaussian integral from", lower, "to", upper, ": ", gausskde.integrate_between(lower, upper))

fig, ax = plt.subplots()
ax.plot(x_data, gauss_yvals, color='r', lw=1, label="Gaussian KDE")
ax.plot(x_data, tophat_yvals, color='k', lw=1, drawstyle='steps-post', label="Tophat KDE")
ax.plot(x_data_fine, tophat_yvals_fine, color='gray', lw=1, ls='--', drawstyle='steps-post', label="Tophat KDE 'exact'")
ax.scatter(test_data, np.zeros(len(test_data)), marker='+', color='k', zorder=10)
ax.hist(test_data, bins=len(x_data), density=True, histtype='stepfilled', color='powderblue', zorder=0,
        label="matplotlib histogram")
ax.legend(loc='best')
ax.annotate("Depending on discretization,\n"
            "the tophat bin heights and\n"
            "widths may not be equal;\n"
            "their integrals are equal", xy=(0, .6), xytext=(-1, 2),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.3"))
ax.annotate("", xy=(1.5, .4), xytext=(.5, 2),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-.3"))
for x_val in x_data:
    ax.axvline(x=x_val, color='lightgray', zorder=-1, lw=1)
plt.show()

#  =============================================================  #
test_data = my_distribution.rvs(size=200)
x_data = np.linspace(-1, max(test_data)+1, len(test_data)//2)

dx = x_data[1]-x_data[0]

gausskde = GaussianKDE(test_data)
gauss_yvals = gausskde(x_data)
print("Approximate integral of Gaussian KDE: ", np.sum(gauss_yvals)*dx)

tophatkde = TophatKDE(test_data)
tophat_yvals = tophatkde(x_data)
print("Approximate integral of Tophat KDE: ", np.sum(tophat_yvals)*dx)

lower = 0
upper = 1.5
print("Computed tophat integral from ", lower, "to", upper, ": ", tophatkde.integrate_between(lower, upper))
print("Computed gaussian integral from", lower, "to", upper, ": ", gausskde.integrate_between(lower, upper))
lower = -1
upper = 5
print("Computed tophat integral from ", lower, "to", upper, ": ", tophatkde.integrate_between(lower, upper))
print("Computed gaussian integral from", lower, "to", upper, ": ", gausskde.integrate_between(lower, upper))

fig, ax = plt.subplots()
ax.plot(x_data, gauss_yvals, color='r', lw=1, label="Gaussian KDE")
ax.plot(x_data, tophat_yvals, color='k', lw=1, ls='--', drawstyle='steps-post', label="Tophat KDE")
ax.scatter(test_data, np.zeros(len(test_data)), marker='+', color='k', zorder=10)
ax.hist(test_data, bins=len(test_data)//5, density=True, histtype='stepfilled', color='powderblue', zorder=0,
        label="matplotlib histogram")
ax.legend(loc='best')
# ax.annotate("Due to discretization,\n"
#             "the tophat bin heights \n"
#             "and widths are not always equal;\n"
#             "their integrals are equal", xy=(0, .6), xytext=(-1, 2),
#             arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.3"))
# ax.annotate("", xy=(1.5, .4), xytext=(.5, 2),
#             arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-.3"))
plt.show()
