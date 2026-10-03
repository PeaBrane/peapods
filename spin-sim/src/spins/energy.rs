use crate::geometry::Lattice;
use rayon::prelude::*;

/// Compute per-system average energy, and optionally per-spin forward interactions.
///
/// `spins`: flat (n_systems * n_spins), i8 values +1/-1
/// `couplings`: flat (n_spins * n_neighbors), forward couplings only
///
/// Returns:
///   energies: Vec<f32> of length n_systems (interaction sum per spin, -H/N)
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

/// Per-system energies (-H/N) and magnetization sums, parallel over systems unless
/// `sequential`. Results match [`compute_energies_and_magnetizations_into`] exactly.
pub(crate) fn refresh_energies_and_magnetizations(
    lattice: &Lattice,
    spins: &[i8],
    couplings: &[f32],
    energies: &mut [f32],
    magnetization_sums: &mut [i64],
    sequential: bool,
) {
    let n_spins = lattice.n_spins;
    assert_eq!(spins.len(), energies.len() * n_spins);
    assert_eq!(energies.len(), magnetization_sums.len());
    let refresh = |((system, energy), magnetization): ((&[i8], &mut f32), &mut i64)| {
        compute_energies_and_magnetizations_into(
            lattice,
            system,
            couplings,
            std::slice::from_mut(energy),
            std::slice::from_mut(magnetization),
        );
    };
    let chunks = spins.chunks_exact(n_spins).zip(energies.iter_mut());
    if sequential {
        chunks.zip(magnetization_sums.iter_mut()).for_each(refresh);
        return;
    }
    spins
        .par_chunks_exact(n_spins)
        .zip(energies.par_iter_mut())
        .zip(magnetization_sums.par_iter_mut())
        .for_each(refresh);
}

/// Exact -H totals and magnetization sums for couplings in {-1, 0, 1}, parallel over
/// systems unless `sequential`.
pub(crate) fn refresh_unit_totals(
    lattice: &Lattice,
    spins: &[i8],
    couplings: &[i8],
    totals: &mut [i64],
    magnetization_sums: &mut [i64],
    sequential: bool,
) {
    let n_spins = lattice.n_spins;
    let n_neighbors = lattice.n_neighbors;
    assert_eq!(spins.len(), totals.len() * n_spins);
    assert_eq!(totals.len(), magnetization_sums.len());
    assert_eq!(couplings.len(), n_spins * n_neighbors);
    let refresh = |((system, total), magnetization): ((&[i8], &mut i64), &mut i64)| {
        let mut bonds = 0i64;
        let mut spin_sum = 0i64;
        for i in 0..n_spins {
            let spin = system[i];
            spin_sum += i64::from(spin);
            let mut field = 0i32;
            for d in 0..n_neighbors {
                let j = lattice.neighbor_fwd(i, d);
                field += i32::from(system[j] * couplings[i * n_neighbors + d]);
            }
            bonds += i64::from(i32::from(spin) * field);
        }
        *total = bonds;
        *magnetization = spin_sum;
    };
    let chunks = spins.chunks_exact(n_spins).zip(totals.iter_mut());
    if sequential {
        chunks.zip(magnetization_sums.iter_mut()).for_each(refresh);
        return;
    }
    spins
        .par_chunks_exact(n_spins)
        .zip(totals.par_iter_mut())
        .zip(magnetization_sums.par_iter_mut())
        .for_each(refresh);
}

/// The cached -H/N for an exact -H total, identical to [`compute_energies_into`].
#[inline]
pub(crate) fn unit_energy(total: i64, n_spins: usize) -> f32 {
    (total as f64 / n_spins as f64) as f32
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
    // Keep one traversal for all geometries; a duplicate canonical-square path
    // produced only a modest improvement.
    for (r, energy) in energies.iter_mut().enumerate() {
        let spin_base = r * n_spins;
        // f64 keeps continuous-coupling totals accurate; PT multiplies them back by N.
        let mut total = 0.0f64;
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
                total += f64::from(interaction);
            }
        }
        *energy = (total / n_spins as f64) as f32;
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

    #[test]
    fn refresh_paths_match_serial_energies() {
        let lattice = Lattice::new(vec![5, 4]);
        let n = lattice.n_spins;
        let unit: Vec<i8> = (0..n * 2).map(|i| [1, -1, 0][i % 3]).collect();
        let couplings: Vec<f32> = unit.iter().map(|&c| f32::from(c)).collect();
        let spins: Vec<i8> = (0..3 * n)
            .map(|i| if (i * 31) % 7 < 3 { -1 } else { 1 })
            .collect();
        let mut expected = vec![0.0; 3];
        let mut expected_mags = vec![0; 3];
        compute_energies_and_magnetizations_into(
            &lattice,
            &spins,
            &couplings,
            &mut expected,
            &mut expected_mags,
        );
        for sequential in [false, true] {
            let mut energies = vec![0.0; 3];
            let mut mags = vec![0; 3];
            refresh_energies_and_magnetizations(
                &lattice,
                &spins,
                &couplings,
                &mut energies,
                &mut mags,
                sequential,
            );
            assert_eq!((&energies, &mags), (&expected, &expected_mags));

            let mut totals = vec![0; 3];
            let mut unit_mags = vec![0; 3];
            refresh_unit_totals(
                &lattice,
                &spins,
                &unit,
                &mut totals,
                &mut unit_mags,
                sequential,
            );
            let unit_energies: Vec<f32> = totals.iter().map(|&t| unit_energy(t, n)).collect();
            assert_eq!((&unit_energies, &unit_mags), (&expected, &expected_mags));
        }
    }

    /// Parallel tempering multiplies the cached energy by N, so the total must stay
    /// accurate for continuous couplings on large lattices (f32 summation drifts by O(1)).
    #[test]
    fn continuous_coupling_totals_stay_accurate() {
        use rand::{Rng, SeedableRng};
        use rand_xoshiro::Xoshiro256StarStar;

        let lattice = Lattice::new(vec![256, 256]);
        let mut rng = Xoshiro256StarStar::seed_from_u64(9);
        let couplings: Vec<f32> = (0..lattice.n_spins * lattice.n_neighbors)
            .map(|_| {
                let u1: f64 = 1.0 - rng.gen::<f64>();
                let u2: f64 = rng.gen();
                ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
            })
            .collect();
        let spins: Vec<i8> = (0..lattice.n_spins)
            .map(|i| if (i * 7919) % 11 < 4 { -1 } else { 1 })
            .collect();

        let mut exact = 0.0f64;
        for i in 0..lattice.n_spins {
            for d in 0..lattice.n_neighbors {
                let j = lattice.neighbor_fwd(i, d);
                exact += f64::from(couplings[i * lattice.n_neighbors + d])
                    * f64::from(spins[i] * spins[j]);
            }
        }
        let mut energy = [0.0f32];
        compute_energies_into(&lattice, &spins, &couplings, &mut energy);
        let total_error = (f64::from(energy[0]) * lattice.n_spins as f64 - exact).abs();
        // Only the final f32 rounding of the per-spin value remains: |e| * 2^-24 * N.
        let bound = exact.abs() * 2f64.powi(-23);
        assert!(total_error <= bound, "total error {total_error} > {bound}");
    }
}
