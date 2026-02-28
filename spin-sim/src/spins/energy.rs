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
#[cfg_attr(feature = "profile", inline(never))]
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
                let j = lattice.neighbor_fwd(i, d);
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

/// Compute the link overlap q_l between replica pairs at each temperature.
///
/// q_l(T) = (1/N_b) Σ_{<ij>} (σ_i^a σ_j^a)(σ_i^b σ_j^b), averaged over pairs.
/// Returns `Vec<f32>` of length `n_temps`.
pub fn compute_link_overlaps(
    lattice: &Lattice,
    spins: &[i8],
    system_ids: &[usize],
    n_replicas: usize,
    n_temps: usize,
) -> Vec<f32> {
    let n_spins = lattice.n_spins;
    let n_neighbors = lattice.n_neighbors;
    let n_bonds = n_spins * n_neighbors;
    let n_pairs = n_replicas / 2;
    let mut result = vec![0.0f32; n_temps];

    if n_pairs == 0 {
        return result;
    }

    for pair_idx in 0..n_pairs {
        let r_a = 2 * pair_idx;
        let r_b = 2 * pair_idx + 1;
        for t in 0..n_temps {
            let sys_a = system_ids[r_a * n_temps + t];
            let sys_b = system_ids[r_b * n_temps + t];
            let base_a = sys_a * n_spins;
            let base_b = sys_b * n_spins;

            let mut sum = 0i64;
            for i in 0..n_spins {
                let si_a = spins[base_a + i] as i64;
                let si_b = spins[base_b + i] as i64;
                for d in 0..n_neighbors {
                    let j = lattice.neighbor_fwd(i, d);
                    let sj_a = spins[base_a + j] as i64;
                    let sj_b = spins[base_b + j] as i64;
                    sum += (si_a * sj_a) * (si_b * sj_b);
                }
            }
            result[t] += sum as f32 / n_bonds as f32;
        }
    }

    let inv = 1.0 / n_pairs as f32;
    for v in result.iter_mut() {
        *v *= inv;
    }

    result
}
