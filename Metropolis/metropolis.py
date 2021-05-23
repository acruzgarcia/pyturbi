import numpy as np
from typing import Callable
from dataclasses import dataclass


@dataclass
class MetropolisAlgorithmRun:
    batch: np.ndarray
    accept: float


class MetropolisAlgorithm:

    def __init__(self,
                 lupost: Callable[[np.array], float],
                 initial: np.array,
                 scale: np.array = np.array([1.0]),
                 burn_cont=0):
        """
        Create an object that supports the execution of the metropolis algorithm.

        Parameters
        ----------
        lupost function proportional to the objective probability distribution to be sampled from.
        initial initial vector state
        scale scale vector that defines the jump distribution (which is normal)
        burn_cont number of samples for the burn-in period.
        """

        self._lupost = lupost
        self._state = initial
        self.n_params = len(initial)

        if len(scale) == self.n_params:
            self._scale = scale
        elif len(scale) == 1:
            self._scale = np.full((1, self.n_params), scale)

        self.burn = False
        self.burn_count = burn_cont
        self.burn_batch = np.full((burn_cont, self.n_params), np.nan)

    def run(self, batches: int) -> MetropolisAlgorithmRun:

        # prepare for putputs
        batch = np.full((batches, self.n_params), np.nan)
        count_accept = 0

        start = - self.burn_count if not self.burn else 0
        for i in range(start, batches):

            if i >= 0:
                batch[i, ] = self._state
            else:
                self.burn_batch[i + self.burn_count, ] = self._state

            log_pxt = self._lupost(self._state)
            proposal = self._gen_proposal()
            log_px = self._lupost(proposal)

            if log_px >= log_pxt:
                # no need to simulate the uniform, acceptance
                self._state = proposal
                if i >= 0:
                    count_accept += 1
            else:
                ratio = np.exp(log_px - log_pxt)
                u = np.random.uniform(low=0, high=1, size=1)
                if u <= ratio:
                    # acceptance
                    self._state = proposal
                    if i >= 0:
                        count_accept += 1

        return MetropolisAlgorithmRun(batch=batch, accept=count_accept/batches)

    def _gen_proposal(self) -> np.ndarray:

        # Generate candidate with random walk
        z = np.random.normal(loc=0.0, scale=1.0, size=self.n_params)
        return self._state + self._scale * z
