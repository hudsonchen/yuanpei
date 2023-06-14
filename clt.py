import numpy as np
import matplotlib.pyplot as plt

# Choose the case of coin flips (0, 1).
# Variable 'random_data' is the population of results from 10000 coin flippings.
# Variable 'sample_mean' is the sample mean of each sample.

samples_mean = []

# Choose 1000 sample data randomly from the popuation data.
for i in range(10000):
    random_data = np.random.randint(0, 2, 10000)
    samples_mean.append(random_data.mean())

sample_mean = np.array(samples_mean)

# Plot the distribution of the sample means.
plt.hist(sample_mean, bins =30, color ='b', alpha=0.8, density=True)

# Plot the Standard normal distibution.
mu = 0.5
variance = 0.25 / 10000
sigma = np.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)

# samples_std = np.random.normal(mu, sigma, 10000)
# plt.hist(samples_std, bins=30, color='red', alpha=0.8, density=True)

import scipy
from scipy import stats
plt.plot(x, stats.norm.pdf(x, mu, sigma), color='black', linewidth=5)
plt.grid()
plt.show()