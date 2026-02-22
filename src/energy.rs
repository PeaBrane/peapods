use crate::lattice::Lattice;

/// Compute per-replica average energy and per-spin forward interactions.
///
/// `spins`: flat (n_replicas * n_spins), i8 values +1/-1
/// `couplings`: flat (n_spins * n_dims), forward couplings only
///
/// Returns:
///   energies: Vec<f32> of length n_replicas (average energy per spin)
///   interactions: Vec<f32> of length (n_replicas * n_spins * n_dims)
///     interactions[r * n_spins * n_dims + i * n_dims + d] =
///       spin[r,i] * spin[r,neighbor_fwd(i,d)] * coupling[i,d]
pub fn compute_energies(
    lattice: &Lattice,
    spins: &[i8],
    couplings: &[f32],
    n_replicas: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n_spins = lattice.n_spins;
    let n_dims = lattice.n_dims;
    let mut energies = vec![0.0f32; n_replicas];
    let mut interactions = vec![0.0f32; n_replicas * n_spins * n_dims];

    #[allow(clippy::needless_range_loop)]
    for r in 0..n_replicas {
        let spin_base = r * n_spins;
        let inter_base = r * n_spins * n_dims;
        let mut total = 0.0f32;

        for i in 0..n_spins {
            let si = spins[spin_base + i] as f32;
            for d in 0..n_dims {
                let j = lattice.neighbor(i, d, true);
                let sj = spins[spin_base + j] as f32;
                let c = couplings[i * n_dims + d];
                let inter = si * sj * c;
                interactions[inter_base + i * n_dims + d] = inter;
                total += inter;
            }
        }

        energies[r] = total / n_spins as f32;
    }

    (energies, interactions)
}

/// Compute per-replica average energy only (no interactions), for use after sweeps.
pub fn compute_energies_only(
    lattice: &Lattice,
    spins: &[i8],
    couplings: &[f32],
    n_replicas: usize,
) -> Vec<f32> {
    let n_spins = lattice.n_spins;
    let n_dims = lattice.n_dims;
    let mut energies = vec![0.0f32; n_replicas];

    #[allow(clippy::needless_range_loop)]
    for r in 0..n_replicas {
        let spin_base = r * n_spins;
        let mut total = 0.0f32;

        for i in 0..n_spins {
            let si = spins[spin_base + i] as f32;
            for d in 0..n_dims {
                let j = lattice.neighbor(i, d, true);
                let sj = spins[spin_base + j] as f32;
                let c = couplings[i * n_dims + d];
                total += si * sj * c;
            }
        }

        energies[r] = total / n_spins as f32;
    }

    energies
}
