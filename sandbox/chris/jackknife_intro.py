
import numpy as np
from matplotlib import pyplot as plt

# Also see: https://en.wikipedia.org/wiki/Jackknife_resampling

# Inputs:
num_samples = 20
outlier_mag = 1.65
exact_mean = 0.5
exact_std_dev = .1

# Confidence interval, used later:
conf_value = 0.99

np.random.seed(0)

# Let it run:
# my_random_data = np.random.random(num_samples)
my_random_data = np.random.normal(exact_mean, exact_std_dev, num_samples)
# Create one value higher than all the others:
sneaky_index = num_samples//2
my_random_data[sneaky_index] = outlier_mag

# Naive sample mean is just the mean of all values:
sample_mean = np.mean(my_random_data)
sample_SEofMean = np.std(my_random_data) / np.sqrt(num_samples)
print("Sample mean: ", sample_mean)
print("Sample std dev of mean: ", sample_SEofMean)

# Leave-one-out (the Jackknife) analysis:
all_indices = np.arange(num_samples)
indices_to_skip = np.arange(num_samples)

# This is not the most efficient way to do this!
# I'm just showing it in a pedagogical way
# Astropy has a jackknife function that should be used:
# https://docs.astropy.org/en/stable/api/astropy.stats.jackknife_stats.html
jk_means = np.zeros(num_samples)
for ind in indices_to_skip:
    jk_sample = np.delete(all_indices, ind)
    # New mean is the mean of all data points EXCEPT the value of interest
    jk_means[ind] = np.mean(my_random_data[jk_sample])

# Now, when we plot it, you'll notice that when the high-value outlier is removed,
# the mean is considerably lower (though perhaps not definitively so for a given test)
plt.plot([0, num_samples-1], [sample_mean, sample_mean],
         color='k', ls='--', label='Raw mean')
plt.plot([0, num_samples-1], [sample_mean + sample_SEofMean, sample_mean + sample_SEofMean],
         color='gray', ls='--', label='Raw mean ±1SE of mean')
plt.plot([0, num_samples-1], [sample_mean - sample_SEofMean, sample_mean - sample_SEofMean],
         color='gray', ls='--')
plt.scatter(indices_to_skip, jk_means, color='r', marker='.',
            label='Resampled means')
plt.scatter(sneaky_index, jk_means[sneaky_index], color='b', marker='+', label='Outlier removed from mean')
plt.ylim(exact_mean-2*exact_std_dev, exact_mean+2*exact_std_dev)
plt.legend()
plt.xlabel("Removed index")
plt.ylabel("Recomputed mean without removed index")
plt.title("Jackknifed means for each index test")
plt.tight_layout()
plt.show(block=True)

# So we can use a metric like standard error of the mean to identify outliers and get closer to the correct mean


# -- Now, let's use it another way -- #
from astropy import stats

mean, bias, jk_std_err, conf_interval = stats.jackknife_stats(my_random_data, np.mean, confidence_level=conf_value)
# Note, bias of the mean is meaningless, it's more useful for nonlinear values like std. dev.
print("Astropy jk_std_err: ", jk_std_err)
print("Astropy conf_interval: ", conf_interval)

# So from the astropy results, the conf_value% confidence interval
my_random_data_outlier_mask = (my_random_data < conf_interval[0]) | (my_random_data > conf_interval[1])
jk_outliers = my_random_data[my_random_data_outlier_mask]


print("Outliers identified: ", jk_outliers)
new_mean = np.mean(my_random_data[~my_random_data_outlier_mask])
print("Mean without outliers: ", new_mean)
print("Exact mean of distribution: ", exact_mean)
print("Sample mean: ", sample_mean)
original_error = sample_mean-exact_mean
new_error = new_mean-exact_mean
print("Removing outliers and recomputing the mean "
      "caused the error in the mean to change from ", original_error, " to ", new_error)

# This section is used to show that choice of your confidence interval impacts the mean
# Try different random seeds or number of samples and you'll see that the results
# are not uniformly improved by performing rejection unless you have significant outliers
# However, for rejecting micro-flares inside our echo region, it will very likely be a significant outlier,
# so this approach should work as long as we don't push it too hard
c_level_test = np.linspace(0.75, 0.9995, 500)
updated_means = np.zeros_like(c_level_test)
errors = np.zeros_like(c_level_test)
for c_i, c_lev in enumerate(c_level_test):
    mean, bias, jk_std_err, conf_interval = stats.jackknife_stats(my_random_data, np.mean, confidence_level=c_lev)

    my_random_data_outlier_mask = (my_random_data < conf_interval[0]) | (my_random_data > conf_interval[1])
    jk_outliers = my_random_data[my_random_data_outlier_mask]
    new_mean = np.mean(my_random_data[~my_random_data_outlier_mask])
    err = new_mean - exact_mean
    updated_means[c_i] = new_mean
    errors[c_i] = err

plt.plot(c_level_test, updated_means, color='k')
plt.plot([c_level_test[0], c_level_test[-1]], [exact_mean, exact_mean],
         color='gray', ls='--', label='Actual mean')
plt.xlabel("Confidence interval used for outlier detection")
plt.ylabel("Computed mean")
plt.title("Mean as a function of confidence interval")
plt.show()
