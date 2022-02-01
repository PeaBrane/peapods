from math import prod
import numpy as np
from scipy.sparse import csr_matrix, csgraph
import numba


def get_neighbors(lattice_shape):
    inds = np.arange(prod(lattice_shape))
    inds = list(np.unravel_index(inds, lattice_shape))
    neighbors = []

    for dim, dim_size in enumerate(lattice_shape):
        inds_new = inds.copy()
        inds_new[dim] = (inds_new[dim] + 1) % dim_size
        neighbor = np.ravel_multi_index(inds_new, lattice_shape)
        neighbors.append(neighbor)

    for dim, dim_size in enumerate(lattice_shape):
        inds_new = inds.copy()
        inds_new[dim] = (inds_new[dim] - 1) % dim_size
        neighbor = np.ravel_multi_index(inds_new, lattice_shape)
        neighbors.append(neighbor)

    return np.vstack(neighbors).T


@numba.njit()
def sweep(spins, couplings_doubled, neighbors, temp_list):
    n_replicas = spins.shape[0]
    lattice_shape = spins.shape[1:]

    spins = spins.reshape((n_replicas, -1))  # [n_replicas, num_spins]
    num_spins = spins.shape[1]
    couplings_doubled = couplings_doubled.reshape((-1, couplings_doubled.shape[-1]))  # [num_spins, n_coupling]
    rand_block = (np.log(np.random.rand(num_spins, n_replicas)) * temp_list).T / 2  # [n_replicas, num_spins]

    for replica_id in range(n_replicas):
        spin = spins[replica_id]
        rand_list = rand_block[replica_id]
        for spin_id in range(num_spins):
            if -(spin[spin_id] * spin.take(neighbors[spin_id]) * couplings_doubled[spin_id]).sum() >= rand_list[spin_id]:
                spin[spin_id] = -spin[spin_id]
        spins[replica_id] = spin

    return spins.reshape(n_replicas, *lattice_shape)


# def get_local_field(spins_transposed, couplings_doubled, n_replicas, n_dims, lattice_shape, coordinate):
#     neighbors = np.array(coordinate)[..., np.newaxis] + np.hstack([np.eye(n_dims), -np.eye(n_dims)]).astype('int64')
#     neighbors = np.mod(neighbors, np.array(lattice_shape)[..., np.newaxis])
#     neighbors = np.ravel_multi_index(neighbors, lattice_shape)
#     neighbors = (neighbors * n_replicas) + np.arange(n_replicas)[..., np.newaxis]  # [n_replicas, neighbors]

#     local_field = (spins_transposed.take(neighbors) * couplings_doubled[coordinate]).sum(1)
#     return local_field


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
