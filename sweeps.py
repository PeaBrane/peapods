import numba
import numpy as np
from scipy.special import logit


def sweep(spins: np.ndarray, 
          couplings_doubled: np.ndarray, 
          neighbors: np.ndarray, 
          temp_list: np.ndarray, 
          mode='metropolis'):
    """Perform a single-spin-flip sweep over the lattice.
    TODO: generalize for arbitrary preceding dimensions

    Args:
        spins (np.ndarray): the Ising spins, shaped (..., *lattice_shape)
        couplings_doubled (np.ndarray): the couplings connected to each spin, 
            shaped (*lattice_shape, 2 * n_dims), assuming square lattice
        neighbors (np.ndarray): the neighbors adjacent to each spin,
            shaped (*lattice_shape, 2 * n_dims)
        temp_list (np.ndarray): the temperature list
        mode (str, optional): the update rule. Defaults to 'metropolis'.

    Returns:
        np.ndarray: the updated spins
    """
    n_replicas = spins.shape[0]
    lattice_shape = spins.shape[1:]

    spins = spins.reshape((n_replicas, -1))  # [n_replicas, num_spins]
    num_spins = spins.shape[1]
    couplings_doubled = couplings_doubled.reshape((-1, couplings_doubled.shape[-1]))  # [num_spins, n_coupling]
    
    match mode:
        case 'metropolis':
            rand_block = np.log(np.random.rand(num_spins, n_replicas)) * temp_list
            rand_block = rand_block.T / 2
        case 'gibbs':
            rand_block = logit(np.random.rand(num_spins, n_replicas)) * temp_list
            rand_block = rand_block.T / 2
        case _:
            raise ValueError("Invalid sampling mode.")

    spins = sweep_numba(spins, couplings_doubled, neighbors, rand_block)
    return spins.reshape(n_replicas, *lattice_shape)


@numba.njit()
def sweep_numba(spins, couplings_doubled, neighbors, rand_block):
    n_replicas, num_spins = spins.shape    
    for replica_id in range(n_replicas):
        spin = spins[replica_id]
        rand_list = rand_block[replica_id]
        for spin_id in range(num_spins):
            if -(spin[spin_id] * spin.take(neighbors[spin_id]) * couplings_doubled[spin_id]).sum() >= rand_list[spin_id]:
                spin[spin_id] = -spin[spin_id]
        spins[replica_id] = spin
        
    return spins
        