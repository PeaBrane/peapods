import time
import numpy as np
from numpy.random import rand, randn
import itertools

import utils


class Ising():
    def __init__(self, lattice_shape, couplings='ferro', temp_list=np.geomspace(0.1, 10, 20)):
        self.n_dims = len(lattice_shape)
        self.lattice_shape = tuple(lattice_shape)
        self.coupling_dims = lattice_shape + (self.n_dims,)

        self.n_replicas = len(temp_list)
        self.temp_list = temp_list
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

        self.reset()

    def reset(self):
        self.temp_ids = np.arange(self.n_replicas)
        self.replica_ids = np.arange(self.n_replicas)

        self.n_sweeps = 0
        self.spins = -1 + 2 * rand(self.n_replicas, *self.lattice_shape).round()
        self.energies, self.interactions = utils.get_energy(self.spins, self.couplings)
 
        self.mags_aggregate = 0
        self.energies_aggregate = 0
        self.spins_aggregate = np.zeros_like(self.spins)
        self.correlations_aggregate = np.zeros_like(self.spins)

        self.mags_average = np.zeros(self.n_replicas)
        self.energies_average = np.zeros(self.n_replicas)

    def update(self):
        self.n_sweeps += 1
        self.energies, self.interactions = utils.get_energy(self.spins[self.temp_ids], self.couplings)
        
        self.spins_aggregate += self.spins
        self.mags_aggregate += self.spins[self.temp_ids].mean(tuple(range(1, self.n_dims+1)))
        self.energies_aggregate += self.energies
        self.correlations_aggregate += self.spins[tuple([0]*self.n_dims)] * self.spins

        self.mags_average = self.mags_aggregate / self.n_sweeps
        self.energies_average = self.energies_aggregate / self.n_sweeps

    def sweep(self, n_sweeps=1, cluster_update=False): 
        rand_block = np.log(rand(n_sweeps, self.n_replicas, *(self.lattice_shape))) \
            * np.expand_dims(self.temp_list[self.replica_ids], tuple(range(1, self.n_dims+1))) / 2

        for sweep_id in range(n_sweeps):
            spins_transposed = np.moveaxis(self.spins, 0, -1)
            rand_list = np.moveaxis(rand_block[sweep_id], 0, -1)

            for coordinate in itertools.product(*[list(range(dim)) for dim in self.lattice_shape]):
                local_field = \
                    utils.get_local_field(spins_transposed, self.couplings_doubled, self.n_replicas, self.n_dims, self.lattice_shape, coordinate)
                update_mask = -spins_transposed[coordinate] * local_field >= rand_list[coordinate]
                spins_transposed[coordinate][update_mask] = -spins_transposed[coordinate][update_mask]
            
            self.spins = np.moveaxis(spins_transposed, -1, 0)
            self.update()

            if cluster_update:
                self.cluster_update()

    def cluster_update(self):
        spins = self.spins.reshape([self.n_replicas, -1])

        for (replica_id, temp, interaction) in zip(self.replica_ids, self.temp_list, self.interactions):
            interaction = (1 - np.exp(-2 * interaction / temp)) >= rand(self.n_dims, *self.lattice_shape)
            clusters = utils.get_clusters(interaction)
            cluster_id = clusters[np.random.choice(clusters.size)]
            
            spins[replica_id][clusters == cluster_id] = -spins[replica_id][clusters == cluster_id]
            self.spins = spins.reshape(self.n_replicas, *self.lattice_shape)

        self.update()

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

    def parallel_tempering(self, n_sweeps, cluster_update=False, exchange_interval=1):
        for sweep_id in range(n_sweeps):
            self.sweep(cluster_update=cluster_update)
            
            if sweep_id % exchange_interval == 0:
                temp_id = np.random.choice(self.n_replicas - 1)
                temp_1, temp_2 = self.temp_list[temp_id], self.temp_list[temp_id + 1]
                index_1, index_2 = (np.argwhere(self.temp_ids == temp_idx)[0, 0] for temp_idx in [temp_id, temp_id+1])
                energy_1, energy_2 = (self.energies[temp_idx] for temp_idx in [temp_id, temp_id+1])
                
                if (energy_2 - energy_1) * (1 / temp_1 - 1 / temp_2) >= np.log(rand()):
                    self.temp_ids[index_1], self.temp_ids[index_2] = self.temp_ids[index_2], self.temp_ids[index_1]
                    self.replica_ids = (self.temp_ids == np.arange(self.n_replicas)[..., np.newaxis]).argmax(1)
                    