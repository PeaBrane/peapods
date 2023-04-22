from math import prod

import numpy as np
from numpy.random import rand, randn
from tqdm import tqdm
import joblib

from clusters import get_clusters
import sweeps
import utils


class Statistics():
    def __init__(self, 
                 reduce_dims=None, 
                 normalize_dims=None, 
                 power=1):
        self.reduce_dims = reduce_dims
        self.normalize_dims = normalize_dims
        self.power = power
        self.count = 0
        self.aggregate = 0

    def update(self, new_input):
        self.count += 1

        if self.reduce_dims is not None:
            new_input = new_input.mean(self.reduce_dims)
        if self.power != 1:
            new_input = new_input**self.power

        self.aggregate += new_input

    @property
    def average(self):
        average = self.aggregate / self.count
        if self.normalize_dims is not None:
            average = average / average.sum(self.normalize_dims, keepdims=True)
        return average
    
    def reset_states(self):
        self.count, self.aggregate = 0, 0


class Ising():
    def __init__(self, 
                 lattice_shape, 
                 couplings='ferro', 
                 temperatures=np.geomspace(0.1, 10, 30)):
        self.n_dims = len(lattice_shape)
        self.lattice_shape = tuple(lattice_shape)
        self.coupling_dims = lattice_shape + (self.n_dims,)

        self.n_replicas = len(temperatures)
        self.temp_list = temperatures.copy()
        self.temp_ids = np.arange(self.n_replicas)
        self.replica_ids = np.arange(self.n_replicas)
        
        match couplings:
            case 'ferro':
                self.couplings = np.ones(self.coupling_dims)
            case 'bimodal':
                self.couplings = -1 + 2*rand(*self.coupling_dims).round()
            case 'gaussian':
                self.couplings = randn(*self.coupling_dims)
            case _:
                raise ValueError("Invalid mode for couplings.")

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
        self.mags = self.spins.mean(tuple(range(-self.n_dims, 0)))
        self.energies, self.interactions = utils.get_energy(self.spins, self.couplings)
        self.csds = np.zeros((self.n_replicas, prod(self.lattice_shape)))

        self.mags_stat, self.mags2_stat, self.mags4_stat = [Statistics(power=power) for power in [1, 2, 4]]
        self.energies_stat, self.energies2_stat = [Statistics(power=power) for power in [1, 2]]
        
        self.csds_stat = Statistics(normalize_dims=(-1,))

    def update(self):
        self.n_sweeps += 1
        
        self.energies, self.interactions = utils.get_energy(self.spins, self.couplings)
        self.mags = self.spins.mean(tuple(range(-self.n_dims, 0)))[self.replica_ids]
        
        self.mags_stat.update(self.mags)
        self.mags2_stat.update(self.mags)
        self.mags4_stat.update(self.mags)

        self.energies_stat.update(self.energies[self.replica_ids])
        self.energies2_stat.update(self.energies[self.replica_ids])
        
        self.csds_stat.update(self.csds[self.replica_ids])

        if self.n_sweeps != 0 and self.n_sweeps % 2**3 == 0:
            self.binder_cumulant = 1 - (self.mags4_stat.average) / (3 * self.mags2_stat.average**2)
            self.heat_capacity = (self.energies2_stat.average - self.energies_stat.average**2) / self.temp_list**2
            
    def get_energies(self):
        return self.energies_stat.average
    
    def get_csds(self):
        return self.csds_stat.average

    def sweep(self, mode='metropolis'):
        self.spins = sweeps.sweep(self.spins, self.couplings_doubled, self.neighbors, self.temp_list[self.temp_ids], mode=mode)
    
    def cluster_update(self, record=True):
        spins = self.spins.reshape([self.n_replicas, -1])

        for replica_id, (temp, interaction) in enumerate(zip(self.temp_list[self.temp_ids], self.interactions)):            
            cluster_labels = get_clusters(interaction, temp)
            cluster_id = cluster_labels[np.random.choice(cluster_labels.size)]
            
            if record:
                csd = np.bincount(np.bincount(cluster_labels) - 1)
                self.csds[replica_id, :len(csd)] = csd
            
            spins[replica_id, cluster_labels == cluster_id] = -spins[replica_id, cluster_labels == cluster_id]
            self.spins = spins.reshape(self.n_replicas, *self.lattice_shape)

    def parallel_tempering(self):
        temp_id = np.random.choice(self.n_replicas - 1)
        temp_1, temp_2 = self.temp_list[temp_id], self.temp_list[temp_id + 1]
        energy_1, energy_2 = self.energies[temp_id], self.energies[temp_id + 1]
        
        if (energy_2 - energy_1) * (1 / temp_1 - 1 / temp_2) >= np.log(rand()):
            temp_id_0 = temp_id - 1 if temp_id != 0 else None
            self.replica_ids[temp_id:temp_id+2] = self.replica_ids[temp_id+1:temp_id_0:-1]
            self.temp_ids = np.argsort(self.replica_ids)
    
    def sample(self, 
               n_sweeps, 
               mode='metropolis',
               cluster_update_interval=None, 
               pt_interval=None):
        for sweep_id in range(n_sweeps):
            self.sweep(mode=mode)
            self.update()
            
            if (cluster_update_interval is not None) and (sweep_id % cluster_update_interval == 0):
                self.cluster_update()
                self.update()
            
            if (pt_interval is not None) and (sweep_id % pt_interval == 0):
                self.parallel_tempering()


class IsingEnsemble():
    def __init__(self,
                 lattice_shape,
                 n_ensemble=2,
                 **kwargs):
        self.n_ensemble = n_ensemble
        self.ising_ensemble = [Ising(lattice_shape, **kwargs) for _ in range(n_ensemble)]
        
    def sample(self, n_sweeps, **kwargs):
        def run_sample(ising):
            ising.sample(n_sweeps, **kwargs)
            return ising
        
        self.ising_ensemble = joblib.Parallel(n_jobs=self.n_ensemble) \
            (joblib.delayed(run_sample)(ising) for ising in self.ising_ensemble)
        
    def get_energies(self):
        energies_ensemble = np.array([ising.get_energies() for ising in self.ising_ensemble])
        return energies_ensemble.mean(0)
    