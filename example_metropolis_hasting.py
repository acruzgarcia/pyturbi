# This is a sample Python script.
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from metropolis.metropolis import MetropolisAlgorithm

# Please we aware that i am not generating here exactly the same random numbers than in R
np.random.seed(568)
x = np.random.normal(loc=0.0, scale=2.0, size=500)
print('Sampled mean and std: mean={:2.2f}, std={:2.2f}'.format(np.mean(x), np.std(x)))

lupost = lambda param: np.sum(stats.norm.logpdf(x, loc=param[0], scale=np.exp(param[1])))
metrop = MetropolisAlgorithm(lupost=lupost,
                             initial=np.array([-2, np.log(7)]),
                             scale=np.array([0.04, 0.04]),
                             burn_cont=100)

# Sample the distribution of parameters
result = metrop.run(batches=1000)

# second parameter requires transform
result.batch[:, 1] = np.exp(result.batch[:, 1])

means = [np.mean(result.batch[:, i]) for i in [0, 1]]
modes = [stats.mode(result.batch[:, i]).mode[0] for i in [0, 1]]

# Print results
print('Mean for parameters: mean={:2.2f}, std={:2.2f}'.format(*means))
print('Mode for parameters: mean={:2.2f}, std={:2.2f}'.format(*modes))
print('Accept rate is {:2.2f}'.format(result.accept))

# Plot simulated markov chains, the parameters
plt.figure()
for i in [0, 1]:
    ax = plt.subplot(1, 2, i + 1)
    ax.plot(result.batch[:, i])
plt.show()

# Plot histograms of the parameters
plt.figure()
for i in [0, 1]:
    ax = plt.subplot(1, 2, i + 1)
    ax.hist(result.batch[:, i])
plt.show()

pepe = 1
