from math import prod
import numpy as np


def get_local_field(spins_transposed, couplings_doubled, n_replicas, n_dims, lattice_shape, coordinate):
    neighbors = np.array(coordinate)[..., np.newaxis] + np.hstack([np.eye(n_dims), -np.eye(n_dims)]).astype('int64')
    neighbors = np.mod(neighbors, np.array(lattice_shape)[..., np.newaxis])
    neighbors = np.ravel_multi_index(neighbors, lattice_shape)
    neighbors = (neighbors * n_replicas) + np.arange(n_replicas)[..., np.newaxis]  # [n_replicas, neighbors]

    local_field = (spins_transposed.take(neighbors) * couplings_doubled[coordinate]).sum(1)
    return local_field


def get_energy(spins, couplings):
    n_dims = len(spins.shape) - 1

    spins_rolled = np.stack([np.roll(spins, -1, i) for i in range(1, n_dims+1)], axis=1)  # [n_replicas, n_dims, (shape)]
    energies = spins[:, np.newaxis, ...] * spins_rolled * np.moveaxis(couplings, -1, 0)  # [n_replicas, n_dims, (shape)]
    energies = energies.sum(tuple(range(1, n_dims+2))) / spins[0].size

    return energies
