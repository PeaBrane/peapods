from math import prod

import numpy as np
from numpy.random import rand
from scipy.sparse import csgraph, csr_matrix


def bond_clusters(bonds):
    n_spins = prod(bonds.shape[:-1])
    spin_ids = np.arange(n_spins).reshape(bonds.shape[:-1])

    rows = [bond.flatten().nonzero()[0] for bond in np.moveaxis(bonds, -1, 0)]
    columns = [np.roll(spin_ids, -1, dim).take(row) for (dim, row) in enumerate(rows)]
    rows, columns = np.hstack(rows), np.hstack(columns)

    G_sparse = csr_matrix((np.ones_like(rows, dtype='bool'), (rows, columns)), shape=(n_spins, n_spins))
    clusters = csgraph.connected_components(G_sparse)[1]
    return clusters


def site_clusters(sites):
    ndim = sites.ndim
    bonds = np.stack([sites & np.roll(sites, -1, dim) for dim in range(ndim)], axis=-1)
    clusters = bond_clusters(bonds)
    clusters[sites.flatten() == 0] = -1
    return clusters


def get_clusters(interaction, temp, cluster_mode='sw'):
    match cluster_mode:
        case 'sw':
            p = np.exp(-2 * interaction / temp)
            bonds = ((1 - p) >= rand(*p.shape))
            return bond_clusters(bonds)
        
        case 'cmr':
            p = np.exp(-2 * np.abs(interaction[0]) / temp)
            rand_block = rand(2, *p.shape)
            interaction_positive = (interaction > 0)
            
            interaction_double = np.logical_and(*interaction_positive)
            interaction_single = np.logical_xor(*interaction_positive)
            
            bonds_blue = interaction_double & ((1 - p**2) >= rand_block[0])
            bonds_red = interaction_single & ((1 - p) >= rand_block[1])
            bonds_grey = np.logical_or(bonds_blue, bonds_red)
            
            return bond_clusters(bonds_blue), bond_clusters(bonds_grey)
            
        case 'houd':
            interaction_positive = (interaction > 0)
            interaction_consistent = ~np.logical_xor(*interaction_positive)
            return bond_clusters(interaction_consistent), None
        
        case _:
            raise ValueError("Invalid cluster mode.")
            