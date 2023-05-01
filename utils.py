from math import prod

import numpy as np


def swap(arr, ind1, ind2):
    arr[ind1], arr[ind2] = arr[ind2], arr[ind1]
    return arr


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

    return np.stack(neighbors, -1).astype(np.int32)


def get_energy(spins, couplings):
    """returns the energy and interactions given the current spin and couplings configurations

    Args:
        spins: shaped (..., *lattice_shape)
        couplings: shaped (*lattice_shape, n_dims)
    """
    n_dims = couplings.ndim - 1

    # (..., *lattice_shape, n_dims)  
    spins_rolled = np.stack([np.roll(spins, -1, i) for i in range(-n_dims, 0)], axis=-1)
    interactions = spins[..., np.newaxis] * spins_rolled * couplings
    
    # (...)
    energies = interactions.sum(tuple(range(-n_dims - 1, 0))) / prod(spins.shape[-n_dims:])

    return energies, interactions


class Statistics():
    def __init__(self, 
                 reduce_dims=None, 
                 power=1):
        self.reduce_dims = reduce_dims
        self.power = power
        
        self.count, self.aggregate = 0, 0

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
        return average
    
    def reset_states(self):
        self.count, self.aggregate = 0, 0
