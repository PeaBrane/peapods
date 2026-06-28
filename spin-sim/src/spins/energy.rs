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
    let mut energies = vec![0.0f32; n_systems];
    if !with_interactions {
        compute_energies_into(lattice, spins, couplings, &mut energies);
        return (energies, None);
    }

    let n_neighbors = lattice.n_neighbors;
    let mut interactions = vec![0.0f32; n_systems * lattice.n_spins * n_neighbors];
    compute_energies_inner(
        lattice,
        spins,
        couplings,
        &mut energies,
        |_, _| {},
        |r, i, d, interaction| {
            interactions[(r * lattice.n_spins + i) * n_neighbors + d] = interaction;
        },
    );
    (energies, Some(interactions))
}

#[cfg_attr(feature = "profile", inline(never))]
pub fn compute_energies_into(
    lattice: &Lattice,
    spins: &[i8],
    couplings: &[f32],
    energies: &mut [f32],
) {
    compute_energies_inner(
        lattice,
        spins,
        couplings,
        energies,
        |_, _| {},
        |_, _, _, _| {},
    );
}

pub fn compute_energies_and_magnetizations_into(
    lattice: &Lattice,
    spins: &[i8],
    couplings: &[f32],
    energies: &mut [f32],
    magnetization_sums: &mut [i64],
) {
    assert_eq!(energies.len(), magnetization_sums.len());
    magnetization_sums.fill(0);
    compute_energies_inner(
        lattice,
        spins,
        couplings,
        energies,
        |r, spin| magnetization_sums[r] += spin as i64,
        |_, _, _, _| {},
    );
}

fn compute_energies_inner(
    lattice: &Lattice,
    spins: &[i8],
    couplings: &[f32],
    energies: &mut [f32],
    mut record_spin: impl FnMut(usize, i8),
    mut record_interaction: impl FnMut(usize, usize, usize, f32),
) {
    let n_spins = lattice.n_spins;
    let n_neighbors = lattice.n_neighbors;
    assert_eq!(spins.len(), energies.len() * n_spins);
    assert_eq!(couplings.len(), n_spins * n_neighbors);
    for (r, energy) in energies.iter_mut().enumerate() {
        let spin_base = r * n_spins;
        let mut total = 0.0f32;
        for i in 0..n_spins {
            let spin = spins[spin_base + i];
            record_spin(r, spin);
            let si = spin as f32;
            for d in 0..n_neighbors {
                let j = lattice.neighbor_fwd(i, d);
                let sj = spins[spin_base + j] as f32;
                let c = couplings[i * n_neighbors + d];
                let interaction = si * sj * c;
                record_interaction(r, i, d, interaction);
                total += interaction;
            }
        }
        *energy = total / n_spins as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn into_path_matches_interaction_path() {
        let lattice = Lattice::new(vec![2, 3]);
        let couplings = vec![1.0; lattice.n_spins * lattice.n_neighbors];
        let spins = vec![
            1, 1, 1, 1, 1, 1, // aligned system
            1, -1, 1, -1, 1, -1, // mixed system
        ];

        let (expected, interactions) = compute_energies(&lattice, &spins, &couplings, 2, true);
        let mut actual = vec![0.0; 2];
        compute_energies_into(&lattice, &spins, &couplings, &mut actual);
        let mut magnetization_sums = vec![0; 2];
        compute_energies_and_magnetizations_into(
            &lattice,
            &spins,
            &couplings,
            &mut actual,
            &mut magnetization_sums,
        );

        assert_eq!(actual, expected);
        assert_eq!(magnetization_sums, vec![6, 0]);
        let interactions = interactions.unwrap();
        let per_system = lattice.n_spins * lattice.n_neighbors;
        for (energy, system_interactions) in expected.iter().zip(interactions.chunks(per_system)) {
            assert_eq!(
                *energy,
                system_interactions.iter().sum::<f32>() / lattice.n_spins as f32
            );
        }
    }
}
