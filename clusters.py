from math import prod

import numba
import numpy as np
from numpy.random import rand


@numba.njit
def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@numba.njit
def _union(parent, rank, x, y):
    rx, ry = _find(parent, x), _find(parent, y)
    if rx == ry:
        return
    if rank[rx] < rank[ry]:
        rx, ry = ry, rx
    parent[ry] = rx
    if rank[rx] == rank[ry]:
        rank[rx] += 1


@numba.njit("i4[:](b1[:, :], i4[:, :])")
def _bond_clusters_uf(bonds_flat, neighbors):
    """Union-find on a flat bond array.

    Args:
        bonds_flat: (n_spins, n_dims) — True if bond from spin to its +1 neighbor in that dim
        neighbors: (n_spins, n_dims) — neighbor index for each spin/dim (forward neighbors only)
    """
    n_spins, n_dims = bonds_flat.shape
    parent = np.arange(n_spins, dtype=np.int32)
    rank = np.zeros(n_spins, dtype=np.int32)

    for spin_id in range(n_spins):
        for dim in range(n_dims):
            if bonds_flat[spin_id, dim]:
                _union(parent, rank, spin_id, neighbors[spin_id, dim])

    for i in range(n_spins):
        parent[i] = _find(parent, i)

    return parent


def _get_forward_neighbors(lattice_shape):
    """Precompute forward (+1) neighbor indices for each dim."""
    n_dims = len(lattice_shape)
    n_spins = prod(lattice_shape)
    spin_ids = np.arange(n_spins).reshape(lattice_shape)
    neighbors = np.stack(
        [np.roll(spin_ids, -1, dim).ravel() for dim in range(n_dims)], axis=-1
    )
    return neighbors.astype(np.int32)


def bond_clusters(bonds):
    lattice_shape = bonds.shape[:-1]
    neighbors = _get_forward_neighbors(lattice_shape)
    bonds_flat = bonds.reshape(-1, bonds.shape[-1])
    return _bond_clusters_uf(bonds_flat, neighbors)


@numba.njit("i4[:](b1[:], i4[:, :])")
def _site_clusters_uf(sites_flat, neighbors):
    n_spins = sites_flat.shape[0]
    n_dims = neighbors.shape[1]
    parent = np.arange(n_spins, dtype=np.int32)
    rank = np.zeros(n_spins, dtype=np.int32)

    for spin_id in range(n_spins):
        if not sites_flat[spin_id]:
            continue
        for dim in range(n_dims):
            neighbor = neighbors[spin_id, dim]
            if sites_flat[neighbor]:
                _union(parent, rank, spin_id, neighbor)

    for i in range(n_spins):
        parent[i] = _find(parent, i)

    return parent


def site_clusters(sites):
    neighbors = _get_forward_neighbors(sites.shape)
    sites_flat = sites.ravel()
    clusters = _site_clusters_uf(sites_flat, neighbors)
    clusters[~sites_flat] = -1
    return clusters


def get_clusters(interaction, temp, cluster_mode="sw"):
    match cluster_mode:
        case "sw":
            p = np.exp(-2 * interaction / temp)
            bonds = (1 - p) >= rand(*p.shape)
            return bond_clusters(bonds)

        case "cmr":
            p = np.exp(-2 * np.abs(interaction[0]) / temp)
            rand_block = rand(2, *p.shape)
            interaction_positive = interaction > 0

            interaction_double = np.logical_and(*interaction_positive)
            interaction_single = np.logical_xor(*interaction_positive)

            bonds_blue = interaction_double & ((1 - p**2) >= rand_block[0])
            bonds_red = interaction_single & ((1 - p) >= rand_block[1])
            bonds_grey = np.logical_or(bonds_blue, bonds_red)

            return bond_clusters(bonds_blue), bond_clusters(bonds_grey)

        case "houd":
            interaction_positive = interaction > 0
            interaction_consistent = ~np.logical_xor(*interaction_positive)
            return bond_clusters(interaction_consistent), None

        case _:
            raise ValueError("Invalid cluster mode.")
