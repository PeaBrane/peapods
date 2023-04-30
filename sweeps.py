import numba
import numpy as np
from scipy.special import logit


def sweep(spins: np.ndarray, 
          couplings_doubled: np.ndarray, 
          neighbors: np.ndarray, 
          temperatures: np.ndarray, 
          mode='metropolis'):
    """Perform a single-spin-flip sweep over the lattice.
    TODO: generalize for arbitrary preceding dimensions

    Args:
        spins (np.ndarray): the Ising spins, shaped (..., *lattice_shape)
        couplings_doubled (np.ndarray): the couplings connected to each spin, 
            shaped (*lattice_shape, 2 * n_dims), assuming square lattice
        neighbors (np.ndarray): the neighbors adjacent to each spin,
            shaped (*lattice_shape, 2 * n_dims)
        temperatures (np.ndarray): the temperature list
        mode (str, optional): the update rule. Defaults to 'metropolis'.

    Returns:
        np.ndarray: the updated spins
    """
    n_replicas = spins.shape[0]
    lattice_shape = spins.shape[1:]

    spins = spins.reshape((n_replicas, -1)).T  # [num_spins, n_replicas]
    num_spins = spins.shape[0]
    couplings_doubled = couplings_doubled.reshape((-1, couplings_doubled.shape[-1]))  # [num_spins, n_coupling]
    
    match mode:
        case 'metropolis':
            rand_block = np.log(np.random.rand(num_spins, n_replicas)) * temperatures
            rand_block = (rand_block / 2).astype(np.float32)
        case 'gibbs':
            rand_block = logit(np.random.rand(num_spins, n_replicas)) * temperatures
            rand_block = (rand_block / 2).astype(np.float32)
        case _:
            raise ValueError("Invalid sampling mode.")

    spins = sweep_numba(spins, couplings_doubled, neighbors, rand_block)
    # return spins.reshape(n_replicas, *lattice_shape)
    return spins.T.reshape(n_replicas, *lattice_shape)


@numba.njit("f4[:, :](f4[:, :], f4[:, :], i4[:, :], f4[:, :])")
def sweep_numba(spins, couplings_doubled, neighbors, rand_block):    
    num_spins = spins.shape[0]
    for spin_id in range(num_spins):
        spin = spins[spin_id]
        
        local_fields = (spins[neighbors[spin_id]] * np.expand_dims(couplings_doubled[spin_id], 1)).sum(0)
        eng_changes = -spin * local_fields
        flip_mask = eng_changes >= rand_block[spin_id]
        spin[flip_mask] = -spin[flip_mask]
        
        spins[spin_id] = spin
        
    return spins
        