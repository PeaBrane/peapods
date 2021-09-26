import numpy as np


def get_local_field(spins, couplings, dims, coordinate):
    n_dims = len(dims)
    local_field = 0

    for i in range(n_dims):
        neighbor = list(coordinate)
        neighbor[i] = (coordinate[i] + 1) % dims[i]
        neighbor = tuple(neighbor)
        local_field += spins[neighbor] * couplings[coordinate + (i,)]
    for i in range(n_dims):
        neighbor = list(coordinate)
        neighbor[i] = (coordinate[i] - 1) % dims[i]
        neighbor = tuple(neighbor)
        local_field += spins[neighbor] * couplings[neighbor + (i,)]

    return local_field


def get_energy(spins, couplings):
    n_dims = spins.ndim
    energy = sum([np.sum(spins * np.roll(spins, -1, i) * couplings[..., i])
                  for i in range(n_dims)])
    return energy
