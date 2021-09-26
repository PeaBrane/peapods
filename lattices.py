import time
import numpy as np
from numpy.random import rand, randn, choice
import itertools

import utils


class Ising():
    def __init__(self, dims, couplings='ferro'):
        self.n_dims = len(dims)
        self.dims = dims
        self.coupling_dims = dims + (self.n_dims,)

        if couplings == 'ferro':
            self.couplings = np.ones(self.coupling_dims)
        elif couplings == 'bimodal':
            self.couplings = rand(*self.coupling_dims).around()
        elif couplings == 'gauss':
            self.couplings = randn(*self.coupling_dims)

        self.reset()

    def reset(self):
        self.spins = -1 + 2 * rand(*self.dims)
        self.magnetizations = np.zeros(self.dims)
        self.correlations = np.zeros(self.dims)
        self.correlations_truncated = np.zeros(self.dims)
        self.energy = 0

    def sweep(self, beta, n_sweeps=1):
        for _ in range(n_sweeps):
            rand_list = -np.log(rand(*self.dims)) / self.spins / beta / 2

            for coordinate in itertools.product(*[list(range(dim)) for dim in self.dims]):
                local_field = utils.get_local_field(self.spins, self.couplings, self.dims, coordinate)
                if rand_list[coordinate] >= local_field:
                    self.spins[coordinate] = -self.spins[coordinate]

            self.magnetizations += self.spins
            self.correlations += self.spins[tuple([0]*self.n_dims)] * self.spins
            self.energy += utils.get_energy(self.spins, self.couplings)

        self.magnetizations /= n_sweeps
        self.correlations /= n_sweeps
        self.energy /= n_sweeps
        self.correlations_truncated = self.correlations - self.magnetizations**2

    def simulated_annealing(self, temperature_high, temperature_low, n_sweeps, space='linear'):
        beta_low, beta_high = 1 / temperature_high, 1 / temperature_low
        if space == 'linear':
            beta_list = np.linspace(beta_low, beta_high, n_sweeps)
        elif space == 'log':
            beta_list = np.geomspace(beta_low, beta_high, n_sweeps)
        else:
            raise ValueError("space argument has to be either 'linear' or 'log'")

        for sweep in range(n_sweeps):
            self.sweep(beta_list[sweep])


model = Ising((40, 40))
start = time.time()
model.simulated_annealing(10, 0.1, 2**6)
end = time.time()
print(end - start)
print(model.energy)
