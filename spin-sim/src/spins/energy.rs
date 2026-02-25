use crate::geometry::Lattice;

/// Compute per-system average energy, and optionally per-spin forward interactions.
///
/// `spins`: flat (n_systems * n_spins), i8 values +1/-1
/// `couplings`: flat (n_spins * n_neighbors), forward couplings only
///
/// Returns:
///   energies: Vec<f32> of length n_systems (average energy per spin)
///   interactions: Option<Vec<f32>> of length (n_systems * n_spins * n_neighbors)
///     interactions[r * n_spins * n_neighbors + i * n_neighbors + d] =
///       spin[r,i] * spin[r,neighbor_fwd(i,d)] * coupling[i,d]
pub fn compute_energies(
    lattice: &Lattice,
    spins: &[i8],
    couplings: &[f32],
    n_systems: usize,
    with_interactions: bool,
) -> (Vec<f32>, Option<Vec<f32>>) {
    let n_spins = lattice.n_spins;
    let n_neighbors = lattice.n_neighbors;
    let mut energies = vec![0.0f32; n_systems];
    let mut interactions = if with_interactions {
        Some(vec![0.0f32; n_systems * n_spins * n_neighbors])
    } else {
        None
    };

    #[allow(clippy::needless_range_loop)]
    for r in 0..n_systems {
        let spin_base = r * n_spins;
        let inter_base = r * n_spins * n_neighbors;
        let mut total = 0.0f32;

        for i in 0..n_spins {
            let si = spins[spin_base + i] as f32;
            for d in 0..n_neighbors {
                let j = lattice.neighbor(i, d, true);
                let sj = spins[spin_base + j] as f32;
                let c = couplings[i * n_neighbors + d];
                let inter = si * sj * c;
                if let Some(ref mut inters) = interactions {
                    inters[inter_base + i * n_neighbors + d] = inter;
                }
                total += inter;
            }
        }

        energies[r] = total / n_spins as f32;
    }

    (energies, interactions)
}
