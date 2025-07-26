import numba
import numpy as np
from scipy.special import logit


def sweep(
    spins: np.ndarray,
    couplings_doubled: np.ndarray,
    neighbors: np.ndarray,
    temperatures: np.ndarray,
    mode="metropolis",
):
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
    num_spins = np.prod(lattice_shape)

    # Create views without transpose - shape: (n_replicas, num_spins)
    spins_flat = spins.reshape((n_replicas, num_spins))
    couplings_flat = couplings_doubled.reshape((num_spins, -1))
    neighbors_flat = neighbors.reshape((num_spins, -1))

    match mode:
        case "metropolis":
            # Shape: (n_replicas, num_spins) - better memory layout
            rand_block = (
                np.log(np.random.rand(n_replicas, num_spins))
                * temperatures[:, np.newaxis]
            )
            rand_block = (rand_block / 2).astype(np.float32)
        case "gibbs":
            rand_block = (
                logit(np.random.rand(n_replicas, num_spins))
                * temperatures[:, np.newaxis]
            )
            rand_block = (rand_block / 2).astype(np.float32)
        case _:
            raise ValueError("Invalid sampling mode.")

    sweep_numba(spins_flat, couplings_flat, neighbors_flat, rand_block)
    return spins


@numba.njit("void(f4[:, :], f4[:, :], i4[:, :], f4[:, :])", fastmath=True)
def sweep_numba(spins, couplings_doubled, neighbors, rand_block):
    """
    Modified to work with shape (n_replicas, num_spins) instead of (num_spins, n_replicas)
    This avoids transpose operations and improves cache locality.
    """
    n_replicas, num_spins = spins.shape

    for spin_id in range(num_spins):
        # Get neighbors for this spin
        neighbor_indices = neighbors[spin_id]
        couplings = couplings_doubled[spin_id]

        # Process all replicas for this spin
        for replica_id in range(n_replicas):
            spin = spins[replica_id, spin_id]

            # Calculate local field
            local_field = 0.0
            for i in range(len(neighbor_indices)):
                local_field += spins[replica_id, neighbor_indices[i]] * couplings[i]

            # Energy change if we flip this spin
            eng_change = -spin * local_field

            # Metropolis criterion
            if eng_change >= rand_block[replica_id, spin_id]:
                spins[replica_id, spin_id] = -spin
