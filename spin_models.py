from math import prod

import joblib
import numpy as np
from numpy.random import rand, randn

import sweeps
import utils
from clusters import get_clusters
from utils import Statistics, swap


class Ising():
    def __init__(self, 
                 lattice_shape, 
                 couplings='ferro', 
                 temperatures=np.geomspace(0.1, 10, 32)):
        self.n_dims = len(lattice_shape)
        self.lattice_shape = tuple(lattice_shape)
        self.coupling_dims = lattice_shape + (self.n_dims,)

        self.n_temps = len(temperatures)
        self.temperatures = temperatures.copy()
        self.temp_ids = np.arange(self.n_temps)
        self.system_ids = np.arange(self.n_temps)
        
        match couplings:
            case 'ferro':
                self.couplings = np.ones(self.coupling_dims)
            case 'bimodal':
                self.couplings = -1 + 2*rand(*self.coupling_dims).round()
            case 'gaussian':
                self.couplings = randn(*self.coupling_dims)
            case _:
                raise ValueError("Invalid mode for couplings.")

        self.couplings = self.couplings.astype(np.float32)
        couplings = [coupling for coupling in np.moveaxis(self.couplings, -1, 0)]
        couplings_clone = [np.roll(self.couplings[..., i], 1, i) for i in range(self.n_dims)]
        self.couplings_doubled = np.stack(couplings + couplings_clone, axis=-1)

        self.neighbors = utils.get_neighbors(lattice_shape)
        
        self.reset()

    def reset(self):
        self.n_swept = 0

        self.temp_ids = np.arange(self.n_temps)
        self.system_ids = np.arange(self.n_temps)

        self.spins = -1 + 2 * rand(self.n_temps, *self.lattice_shape).round().astype(np.float32)
        self.mags = self.spins.mean(tuple(range(-self.n_dims, 0)))
        self.energies, self.interactions = utils.get_energy(self.spins, self.couplings)
        self.csds = np.zeros((self.n_temps, prod(self.lattice_shape)))

        self.mags_stat, self.mags2_stat, self.mags4_stat = [Statistics(power=power) for power in [1, 2, 4]]
        self.energies_stat, self.energies2_stat = [Statistics(power=power) for power in [1, 2]]
        
        self.csds_stat = Statistics()

    def update(self, record=True, csd_update=False):
        self.n_swept += 1
        self.energies, self.interactions = utils.get_energy(self.spins, self.couplings)
        
        if record:
            self.mags = self.spins.mean(tuple(range(-self.n_dims, 0)))[self.system_ids]
            
            self.mags_stat.update(self.mags)
            self.mags2_stat.update(self.mags)
            self.mags4_stat.update(self.mags)

            self.energies_stat.update(self.energies[self.system_ids])
            self.energies2_stat.update(self.energies[self.system_ids])
            
            if csd_update:
                self.csds_stat.update(self.csds)

            if self.n_swept != 0 and self.n_swept % 2**3 == 0:
                self.binder_cumulant = 1 - (self.mags4_stat.average) / (3 * self.mags2_stat.average**2)
                self.heat_capacity = (self.energies2_stat.average - self.energies_stat.average**2) / self.temperatures**2
            
    def get_energies(self):
        return self.energies_stat.average
    
    def get_csds(self, normalized=True):
        csds = self.csds_stat.average
        if normalized:
            csds = csds / csds.sum(-1, keepdim=True)
        return csds

    def sweep(self, mode='metropolis'):
        self.spins = sweeps.sweep(self.spins, self.couplings_doubled, self.neighbors, 
                                  self.temperatures[self.temp_ids], mode=mode)
    
    def record_clusters(self, clusters, temp_id):
        csd = np.bincount(np.bincount(clusters) - 1)
        self.csds[temp_id, :len(csd)] = csd
    
    def sw_update(self, update=True, record=True):
        """
        Performs a cluster update of the spins.
        """
        self.csds.fill(0)
        spins = self.spins.reshape(self.n_temps, -1)

        for temp_id, (interaction, temp) in enumerate(zip(self.interactions[self.system_ids], self.temperatures)):
            clusters = get_clusters(interaction, temp)
            
            # flips the spin clusters
            system_id = self.system_ids[temp_id]
            cluster_id = clusters[np.random.choice(clusters.size)]
            flip_mask = (clusters == cluster_id)
            if update:
                spins[system_id, flip_mask] = -spins[system_id, flip_mask]
                
            # records the cluster size distribution
            if record:                    
                self.record_clusters(clusters, temp_id)
            
        self.spins = spins.reshape(self.n_temps, *self.lattice_shape)
        
    def replica_cluster_update(self, update=True, record=True, cluster_mode='cmr'):
        raise NotImplementedError("Replica cluster update requires two replicas.")

    def parallel_tempering(self):
        """
        Exchanges the spins of two randomly selected adjacent temperatures.
        """
        temp_id = np.random.choice(self.n_temps - 1)
        ids = [temp_id, temp_id + 1]
        temp_1, temp_2 = self.temperatures[ids]
        energy_1, energy_2 = self.energies[self.system_ids[ids]]
        
        if (energy_2 - energy_1) * (1 / temp_1 - 1 / temp_2) >= np.log(rand()):
            self.system_ids = swap(self.system_ids, *ids)
            self.temp_ids = np.argsort(self.system_ids)
    
    def sample(self, 
               n_sweeps, 
               sweep_mode='metropolis',
               cluster_update_interval=None,
               cluster_mode='sw',
               pt_interval=None,
               warmup_ratio=0.25):
        warmup_sweeps = round(n_sweeps * warmup_ratio)
        
        for sweep_id in range(n_sweeps):
            record = sweep_id >= warmup_sweeps
            
            self.sweep(mode=sweep_mode)
            self.update(record=record)
            
            if (cluster_update_interval is not None) and (sweep_id % cluster_update_interval == 0):
                match cluster_mode:
                    case 'sw':
                        self.sw_update(record=record)
                        self.update(record=record, csd_update=True)
                    case 'cmr' | 'houd':
                        self.replica_cluster_update(record=record, cluster_mode=cluster_mode)
                        self.update(record=record, csd_update=True)
            
            if (pt_interval is not None) and (sweep_id % pt_interval == 0):
                self.parallel_tempering()


class IsingReplicas(Ising):
    def __init__(self, 
                 lattice_shape,
                 **kwargs):
        """
        Similar to the base Ising class, but instead keeps 2 replicas per temperature.
        Mainly used for simulating spin glasses.
        """
        super().__init__(lattice_shape, **kwargs)
        
        self.n_temps = len(self.temperatures) * 2
        self.temperatures = np.repeat(self.temperatures, 2)
        self.temp_ids = np.arange(self.n_temps)
        self.system_ids = np.arange(self.n_temps)
        self.reset()
        
    def get_energies(self):
        energies = super().get_energies()
        return energies.reshape((-1, 2)).mean(1)
    
    def get_csds(self, normalized=True):
        csds = self.csds_stat.average
        if normalized:
            csds = csds / csds.sum(-1, keepdim=True)
        return csds
    
    def replica_cluster_update(self, update=True, record=True, cluster_mode='cmr'):
        """
        TODO: implement cluster update routine
        """
        self.csds.fill(0)
        spins = self.spins.reshape(self.n_temps, -1)
        
        interactions_pairs = self.interactions[self.system_ids]
        interactions_pairs = np.stack([interactions_pairs[0::2], 
                                       interactions_pairs[1::2]], axis=1)
        
        for temp_id, (interaction, temp) in enumerate(zip(interactions_pairs, self.temperatures[::2])):
            clusters, _ = get_clusters(interaction, temp, cluster_mode=cluster_mode)
                
            # records the cluster size distribution
            if record:                    
                super().record_clusters(clusters, temp_id)
            
        self.spins = spins.reshape(self.n_temps, *self.lattice_shape)
    
    def parallel_tempering(self):
        for replica_id in range(2):
            temp_id = np.random.choice(self.n_temps // 2 - 1)
            temp_1, temp_2 = self.temperatures[[2*temp_id, 2*temp_id+2]]
            ids = [2 * temp_id + replica_id, 2 * temp_id + 2 + replica_id]
            energy_1, energy_2 = self.energies[self.system_ids[ids]]
            
            if (energy_2 - energy_1) * (1 / temp_1 - 1 / temp_2) >= np.log(rand()):
                self.system_ids = swap(self.system_ids, *ids)
                
        self.temp_ids = np.argsort(self.system_ids)
        

class IsingEnsemble():
    def __init__(self,
                 lattice_shape,
                 n_ensemble=2,
                 replicas=False,
                 **kwargs):
        self.n_ensemble = n_ensemble
        
        if replicas:
            self.ising_ensemble = [IsingReplicas(lattice_shape, **kwargs) for _ in range(n_ensemble)]
        else:
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
    
    def get_csds(self):        
        csds_ensemble = np.stack([ising.get_csds(normalized=False) for ising in self.ising_ensemble])
        csds = csds_ensemble.sum(0)
        return csds / csds.sum(-1, keepdims=True)
    