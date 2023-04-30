from math import prod
import numpy as np
from numpy.random import rand

from scipy.sparse import csgraph, csr_matrix


def sw_interaction(interaction, temp):
    """
    Returns a boolean interaction array denoting the openess of
    bonds under the SW algorithm:
    https://en.wikipedia.org/wiki/Swendsen%E2%80%93Wang_algorithm
    """
    return (1 - np.exp(-2 * interaction / temp)) >= rand(*interaction.shape)


def get_clusters(interaction, temp, mode='sw'):
    match mode:
        case 'sw':
            interaction = sw_interaction(interaction, temp)
    
    n_spins = prod(interaction.shape[:-1])
    spin_ids = np.arange(n_spins).reshape(interaction.shape[:-1])

    rows = [interact.flatten().nonzero()[0] for interact in np.moveaxis(interaction, -1, 0)]
    columns = [np.roll(spin_ids, -1, dim).take(row) for (dim, row) in enumerate(rows)]
    rows, columns = np.hstack(rows), np.hstack(columns)

    G_sparse = csr_matrix((np.ones_like(rows, dtype='bool'), (rows, columns)), shape=(n_spins, n_spins))
    clusters = csgraph.connected_components(G_sparse)[1]
    return clusters
