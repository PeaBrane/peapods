from math import prod

import numba
import numpy as np
from scipy.sparse import csgraph, csr_matrix


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


def get_energy(spins, couplings):
    """returns the energy and interactions given the current spin and couplings configurations

    Args:
        spins: shaped (..., *lattice_shape)
        couplings: shaped (*lattice_shape, n_dims)
    """
    n_dims = couplings.ndim - 1

    spins_rolled = np.stack([np.roll(spins, -1, i) 
                             for i in range(-n_dims, 0)], axis=-1)  # (..., *lattice_shape, n_dims)
    interactions = spins[..., np.newaxis] * spins_rolled * couplings  # (..., *lattice_shape, n_dims)    
    energies = interactions.sum(tuple(range(-n_dims - 1, 0))) / prod(spins.shape[-n_dims:])

    return energies, interactions


def get_clusters(interaction):
    n_spins = prod(interaction.shape[:-1])
    spin_ids = np.arange(n_spins).reshape(interaction.shape[:-1])

    rows = [interact.flatten().nonzero()[0] for interact in np.moveaxis(interaction, -1, 0)]
    columns = [np.roll(spin_ids, -1, dim).take(row) for (dim, row) in enumerate(rows)]
    rows, columns = np.hstack(rows), np.hstack(columns)

    G_sparse = csr_matrix((np.ones_like(rows, dtype='bool'), (rows, columns)), shape=(n_spins, n_spins))
    clusters = csgraph.connected_components(G_sparse)[1]
    return clusters
