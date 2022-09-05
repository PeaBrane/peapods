from tempfile import TemporaryDirectory
import time
import numpy as np
from numpy.random import rand, randn
import itertools

import utils


class Statistics():
    def __init__(self, reduce_dims=None, power=1):
        self.reduce_dims = reduce_dims
        self.power = power
        self.count = 0
        self.aggregate = 0

    def update(self, new_input):
        self.count += 1

        if self.reduce_dims is not None:
            new_input = new_input.mean(self.reduce_dims)
        if self.power != 1:
            new_input = new_input ** self.power

        self.aggregate += new_input

    @property
    def average(self):
        return self.aggregate / self.count


class Ising():
    def __init__(self, lattice_shape, couplings='ferro', temperatures=np.geomspace(0.1, 10, 30)):
        self.n_dims = len(lattice_shape)
        self.lattice_shape = tuple(lattice_shape)
        self.coupling_dims = lattice_shape + (self.n_dims,)

        self.n_replicas = len(temperatures)
        self.temp_list = temperatures.copy()
        self.temp_ids = np.arange(self.n_replicas)
        self.replica_ids = np.arange(self.n_replicas)

        if couplings == 'ferro':
            self.couplings = np.ones(self.coupling_dims)
        elif couplings == 'bimodal':
            self.couplings = -1 + 2*rand(*self.coupling_dims).round()
        elif couplings == 'gaussian':
            self.couplings = randn(*self.coupling_dims)

        couplings = [coupling for coupling in np.moveaxis(self.couplings, -1, 0)]
        couplings_clone = [np.roll(self.couplings[..., i], 1, i) for i in range(self.n_dims)]
        self.couplings_doubled = np.stack(couplings + couplings_clone, axis=-1)

        self.neighbors = utils.get_neighbors(lattice_shape)

        self.reset()

    def reset(self):
        self.n_sweeps = 0

        self.temp_ids = np.arange(self.n_replicas)
        self.replica_ids = np.arange(self.n_replicas)

        self.spins = -1 + 2 * rand(self.n_replicas, *self.lattice_shape).round()
        self.energies, self.interactions = utils.get_energy(self.spins, self.couplings)

        self.mags_stat = Statistics(reduce_dims=tuple(range(-self.n_dims, 0)))
        self.mags2_stat, self.mags4_stat = \
            Statistics(reduce_dims=tuple(range(-self.n_dims, 0)), power=2), Statistics(reduce_dims=tuple(range(-self.n_dims, 0)), power=4)

        self.energies_stat = Statistics()
        self.energies2_stat = Statistics(power=2)

    def update(self):
        self.n_sweeps += 1

        spins = self.spins[self.temp_ids]
        self.energies, self.interactions = utils.get_energy(spins, self.couplings)
        
        self.mags_stat.update(spins)
        self.mags2_stat.update(spins)
        self.mags4_stat.update(spins)

        self.energies_stat.update(self.energies)
        self.energies2_stat.update(self.energies)

        if self.n_sweeps != 0 and self.n_sweeps % 10 == 0:
            self.binder_cumulant = 1 - (self.mags4_stat.average) / (3 * self.mags2_stat.average**2)
            self.heat_capacity = (self.energies2_stat.average - self.energies_stat.average**2) / self.temp_list**2

    def update_spins(self, n_sweeps=1, cluster_update=False): 
        for _ in range(n_sweeps):
            self.spins = utils.sweep(self.spins, self.couplings_doubled, self.neighbors, self.temp_list[self.replica_ids])
            self.update()

            if cluster_update:
                self.cluster_update()

    def cluster_update(self):
        spins = self.spins.reshape([self.n_replicas, -1])

        for (replica_id, temp, interaction) in zip(self.replica_ids, self.temp_list, self.interactions):
            interaction = (1 - np.exp(-2 * interaction / temp)) >= rand(*self.lattice_shape, self.n_dims)
            clusters = utils.get_clusters(interaction)
            cluster_id = clusters[np.random.choice(clusters.size)]
            
            spins[replica_id][clusters == cluster_id] = -spins[replica_id][clusters == cluster_id]
            self.spins = spins.reshape(self.n_replicas, *self.lattice_shape)

        self.update()

    def parallel_tempering(self, n_sweeps, cluster_update=False, exchange_interval=1):
        for sweep_id in range(n_sweeps):
            self.update_spins(cluster_update=cluster_update)
            
            if sweep_id % exchange_interval == 0:
                temp_id = np.random.choice(self.n_replicas - 1)
                temp_1, temp_2 = self.temp_list[temp_id], self.temp_list[temp_id + 1]
                energy_1, energy_2 = self.energies[temp_id], self.energies[temp_id+1]
                replica_id_1, replica_id_2 = self.replica_ids[temp_id], self.replica_ids[temp_id + 1]
                
                if (energy_2 - energy_1) * (1 / temp_1 - 1 / temp_2) >= np.log(rand()):
                    self.temp_ids[replica_id_1], self.temp_ids[replica_id_2] = self.temp_ids[replica_id_2], self.temp_ids[replica_id_1]
                    self.replica_ids = (self.temp_ids == np.arange(self.n_replicas)[..., np.newaxis]).argmax(1)

    # def simulated_annealing(self, T_high, T_low, n_sweeps, space='linear'):
    #     beta_low, beta_high = 1 / T_high, 1 / T_low
    #     if space == 'linear':
    #         temp_list = 1 / np.linspace(beta_low, beta_high, n_sweeps)
    #     elif space == 'log':
    #         temp_list = 1 / np.geomspace(beta_low, beta_high, n_sweeps)
    #     else:
    #         raise ValueError("space argument has to be either 'linear' or 'log'")

    #     for temp in temp_list:
    #         self.sweep(temp)
                    

class IsingOverlap(Ising):
    def __init__(self, lattice_shape, couplings='ferro', temperatures=np.geomspace(0.1, 10, 30)):
        super().__init__(lattice_shape, couplings, temperatures)

    def reset(self):
        super().reset()