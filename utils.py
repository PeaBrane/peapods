from math import prod

import numpy as np


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
