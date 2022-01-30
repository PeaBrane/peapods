from math import prod
import numpy as np
from pyparsing import col
from scipy.sparse import csr_matrix, csgraph


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
    interactions = spins[:, np.newaxis, ...] * spins_rolled * np.moveaxis(couplings, -1, 0)  # [n_replicas, n_dims, (shape)]
    energies = interactions.sum(tuple(range(1, n_dims+2))) / spins[0].size

    return energies, interactions


def get_clusters(interaction):
    n_spins = interaction[0].size
    spin_ids = np.arange(n_spins).reshape(interaction[0].shape)

    rows = [interact.flatten().nonzero()[0] for interact in interaction]
    columns = [np.roll(spin_ids, -1, dim).take(row) for (dim, row) in enumerate(rows)]
    rows, columns = np.hstack(rows), np.hstack(columns)

    G_sparse = csr_matrix((np.ones_like(rows, dtype='bool'), (rows, columns)), shape=(n_spins, n_spins))
    clusters = csgraph.connected_components(G_sparse)[1]
    return clusters
