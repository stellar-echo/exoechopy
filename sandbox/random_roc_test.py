import numpy as np
from matplotlib import pyplot as plt

n_samples = 500

scale = 0.5

mean_1 = 0
mean_2 = scale

dist_fp = np.random.normal(mean_1, scale=scale, size=n_samples)
dist_tp = np.random.normal(mean_2, scale=scale, size=n_samples)

plt.hist(dist_fp)
plt.hist(dist_tp)
plt.show()
plt.close()

tp_hit = []
fp_hit = []

minval = min(np.min(dist_tp), np.min(dist_fp))

for tp in dist_tp:
    N_true_hits = np.sum(dist_tp >= tp) / len(dist_tp)
    N_false_hist = np.sum(dist_fp >= tp) / len(dist_fp)
    tp_hit.append(N_true_hits)
    fp_hit.append(N_false_hist)

plt.scatter(fp_hit, tp_hit, marker='.', color='k')
plt.show()
