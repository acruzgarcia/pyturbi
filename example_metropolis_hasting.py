# This is a sample Python script.
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from metropolis.metropolis import MetropolisAlgorithm

# Please we aware that i am not generating here exactly the same random numbers than in R
np.random.seed(568)
x = np.random.normal(loc=0.0, scale=2.0, size=500)

logprior1 = lambda mu: stats.norm.logpdf(mu, loc=0, scale=0.5)
logprior2 = lambda sigma: stats.norm.logpdf(sigma, loc=2, scale=0.5)

lupost = lambda param: np.sum(stats.norm.logpdf(x, loc=param[0], scale=np.exp(param[1]))) \
                       * logprior1(param[0]) \
                       * logprior2(np.exp(param[1]))
metrop = MetropolisAlgorithm(lupost=lupost,
                             initial=np.array([-2, np.log(7)]),
                             scale=np.array([0.04, 0.04]),
                             burn_cont=100)
res = metrop.run(batches=1000)

# second parameter requires transform
res.batch[:, 1] = np.exp(res.batch[:, 1])

print('Sampled mean and std: mean={:2.2f}, std={:2.2f}'.format(np.mean(x), np.std(x)))
means = [np.mean(res.batch[:, i]) for i in [0, 1]]
print('Mean for parameters: mean={:2.2f}, std={:2.2f}'.format(*means))
modes = [stats.mode(res.batch[:, i]).mode[0] for i in [0, 1]]
print('Mode for parameters: mean={:2.2f}, std={:2.2f}'.format(*modes))
print('Accept rate is {:2.2f}'.format(res.accept))

# Plot simulated params
plt.figure()
for i in [0, 1]:
    ax = plt.subplot(1, 2, i + 1)
    ax.plot(res.batch[:, i])
plt.show()

plt.figure()
for i in [0, 1]:
    ax = plt.subplot(1, 2, i + 1)
    ax.hist(res.batch[:, i])
plt.show()

pepe = 1
