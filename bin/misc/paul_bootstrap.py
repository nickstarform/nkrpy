"""Quick code for Bootstrap."""

# import modules
import numpy as np
import scipy.optimize as opt
from time import time
from scipy.stats import chisquare
import matplotlib.pyplot as plt


def linear(x, a, b):
    """Linear Function."""
    return a * x + b

# --------------------------------------------------------------------------------
# generate data
np.random.seed(900)

num_data = 1000

x = np.array(list(np.arange(0, num_data, 0.1)))

y = np.array(list(map(lambda x: 2. * x + 4., x)))

noise = np.random.random(x.shape)

noisy_y = y + np.average(y) * np.random.normal(size=x.shape)

frac = 0.1

# generate a list of indices to exclude. Turn in into a set for O(1) lookup time
inds = list(set(np.random.choice(list(range(len(noisy_y))), int(frac*len(noisy_y)))))

print(inds)

data = np.vstack((x[inds], noisy_y[inds]))

print(f'Data size:{data.shape}')  # Data size:(2, 10000)
print(f'First row:{data[:, 0]}')  # array([0.        , 1.73280611])

# --------------------------------------------------------------------------------
# now for bootstrapping
samples = int(num_data / 2.)
bootstraps = int(num_data / 5.)
guess = (1., 1.)

results = []
errf = []
for boot in range(bootstraps):
    t1 = time()
    # can only choose 1d
    sampled_data_i = np.random.choice(list(range(data.shape[-1])), samples)
    sampled_data = data[:, sampled_data_i]
    popt, pcov = opt.curve_fit(linear, sampled_data[0, :],
                               sampled_data[1, :], p0=guess)
    x2 = chisquare(sampled_data[1, :],
                   linear(sampled_data[0, :], *popt))[0]
    results.append(popt)
    errf.append(x2)
    print(f'Finished #{boot}, with params:{popt} and fit of:{x2})')
    print(f'Time:{time() - t1}')

results = np.array(results).reshape(-1, 2)                                  

# --------------------------------------------------------------------------------
# plotting
plt.figure(figsize=[15, 15])
plt.scatter(data[0, :], data[1, :], color='black', s=10, label='Raw')
# all results
for i, result in enumerate(results):
    a, b = result
    if i == 0:
        plt.plot(data[0, :], linear(data[0, :], a, b), alpha=0.2,
                 color='red', label='fits')
    else:
        plt.plot(data[0, :], linear(data[0, :], a, b), alpha=0.2, color='red')

# lowest x2
i = errf.index(np.min(errf))
plt.plot(data[0, :], linear(data[0, :], results[i, 0], results[i, 1]), alpha=0.7,
         color='blue', label=r'lowest $\chi^{2}$')
# median
i = np.median(results, axis=0)
plt.plot(data[0, :], linear(data[0, :], i[0], i[1]), alpha=0.8,
         color='green', label=r'Median Fit')

plt.legend()
plt.tight_layout()
plt.xlim(0, num_data)
plt.show()

# end of file
