use super::threshold;
use crate::geometry::Lattice;
use crate::parallel::{par_over_replicas, par_over_replicas_with};
use rand::RngCore;
use rand_xoshiro::Xoshiro256StarStar;
use std::ops::Add;

/// Coupling arithmetic for local fields; unit couplings use exact integers.
trait Coupling: Copy + Send + Sync {
    type Field: Copy + Default + Add<Output = Self::Field>;
    fn term(spin: i8, coupling: Self) -> Self::Field;
}

impl Coupling for f32 {
    type Field = f32;
    #[inline]
    fn term(spin: i8, coupling: f32) -> f32 {
        spin as f32 * coupling
    }
}

impl Coupling for i8 {
    type Field = i32;
    #[inline]
    fn term(spin: i8, coupling: i8) -> i32 {
        i32::from(spin * coupling)
    }
}

/// Compute local field for spin `i` from all `2 * n_neighbors` neighbors.
#[inline]
fn local_field<C: Coupling>(
    lattice: &Lattice,
    spin_slice: &[i8],
    couplings: &[C],
    i: usize,
) -> C::Field {
    let n_neighbors = lattice.n_neighbors;
    let mut h = C::Field::default();
    for d in 0..n_neighbors {
        let j_fwd = lattice.neighbor_fwd(i, d);
        h = h + C::term(spin_slice[j_fwd], couplings[i * n_neighbors + d]);

        let j_bwd = lattice.neighbor_bwd(i, d);
        h = h + C::term(spin_slice[j_bwd], couplings[j_bwd * n_neighbors + d]);
    }
    h
}

#[inline]
fn square_interior_field<C: Coupling>(
    spin_slice: &[i8],
    couplings: &[C],
    i: usize,
    width: usize,
) -> C::Field {
    // Valid only for non-boundary sites on a canonical 2D lattice, whose
    // directions have strides `width` and 1. Backward terms use the neighboring
    // site's coupling because couplings own forward bonds.
    let mut h = C::Field::default();
    h = h + C::term(spin_slice[i + width], couplings[i * 2]);
    h = h + C::term(spin_slice[i - width], couplings[(i - width) * 2]);
    h = h + C::term(spin_slice[i + 1], couplings[i * 2 + 1]);
    h = h + C::term(spin_slice[i - 1], couplings[(i - 1) * 2 + 1]);
    h
}

#[inline]
fn sweep_sites<C: Coupling>(
    lattice: &Lattice,
    spin_slice: &mut [i8],
    couplings: &[C],
    mut attempt: impl FnMut(&mut [i8], usize, C::Field),
) {
    let Some((height, width)) = lattice
        .square_shape()
        .filter(|&(height, width)| height >= 3 && width >= 3)
    else {
        for i in 0..lattice.n_spins {
            let h = local_field(lattice, spin_slice, couplings, i);
            attempt(spin_slice, i, h);
        }
        return;
    };

    for i in 0..width {
        let h = local_field(lattice, spin_slice, couplings, i);
        attempt(spin_slice, i, h);
    }

    for row in 1..height - 1 {
        let row_start = row * width;
        let h = local_field(lattice, spin_slice, couplings, row_start);
        attempt(spin_slice, row_start, h);

        for i in row_start + 1..row_start + width - 1 {
            let h = square_interior_field(spin_slice, couplings, i, width);
            attempt(spin_slice, i, h);
        }

        let row_end = row_start + width - 1;
        let h = local_field(lattice, spin_slice, couplings, row_end);
        attempt(spin_slice, row_end, h);
    }

    for i in (height - 1) * width..height * width {
        let h = local_field(lattice, spin_slice, couplings, i);
        attempt(spin_slice, i, h);
    }
}

/// Single-spin acceptance rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Acceptance {
    Metropolis,
    Gibbs,
}

impl Acceptance {
    /// Flip probability for `gain = -s_i h_i` (a flip changes -H by `2 * gain`)
    /// at `beta2 = 2 / T`.
    ///
    /// Lookup tables and the generic kernel share this f32 arithmetic, so both
    /// paths make bit-identical decisions for unit couplings.
    #[inline]
    fn probability(self, gain: f32, beta2: f32) -> f32 {
        match self {
            Self::Metropolis => (gain * beta2).exp(),
            Self::Gibbs => 1.0 / (1.0 + (-gain * beta2).exp()),
        }
    }
}

/// Acceptance thresholds for couplings in {-1, 0, 1}, indexed by temperature
/// slot and integer gain.
pub(crate) struct UnitCouplingLookup {
    thresholds: Vec<u64>,
    couplings: Vec<i8>,
    offset: i32,
    table_width: usize,
}

impl UnitCouplingLookup {
    pub(crate) fn new(
        couplings: &[f32],
        temperatures: &[f32],
        n_neighbors: usize,
        acceptance: Acceptance,
    ) -> Option<Self> {
        if !couplings
            .iter()
            .all(|&coupling| coupling == -1.0 || coupling == 0.0 || coupling == 1.0)
            || !temperatures
                .iter()
                .all(|&temperature| (2.0 / temperature).is_finite() && temperature > 0.0)
        {
            return None;
        }

        let offset = i32::try_from(n_neighbors.checked_mul(2)?).ok()?;
        let table_width = usize::try_from(offset.checked_mul(2)?.checked_add(1)?).ok()?;
        let mut thresholds = Vec::with_capacity(temperatures.len().checked_mul(table_width)?);
        for &temperature in temperatures {
            let beta2 = 2.0 / temperature;
            for gain in -offset..=offset {
                thresholds.push(threshold(acceptance.probability(gain as f32, beta2)));
            }
        }

        Some(Self {
            thresholds,
            couplings: couplings.iter().map(|&coupling| coupling as i8).collect(),
            offset,
            table_width,
        })
    }

    pub(crate) fn couplings(&self) -> &[i8] {
        &self.couplings
    }

    #[inline]
    fn row(&self, temperature_id: usize) -> &[u64] {
        let start = temperature_id * self.table_width;
        &self.thresholds[start..start + self.table_width]
    }
}

/// Branchless conditional flip: acceptance is data dependent and poorly predicted.
#[inline]
fn flip_if(spin_slice: &mut [i8], i: usize, accept: bool) {
    spin_slice[i] *= 1 - 2 * i8::from(accept);
}

/// Exact change of one system's -H total and magnetization sum over a unit-coupling sweep.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct UnitSweepDelta {
    pub interaction: i64,
    pub magnetization: i64,
}

/// Integer sweep of one system with unit couplings; with `TRACK` it also returns the
/// exact observable change.
#[inline(always)]
fn unit_kernel<const TRACK: bool>(
    lattice: &Lattice,
    spin_slice: &mut [i8],
    rng: &mut Xoshiro256StarStar,
    lookup: &UnitCouplingLookup,
    temperature_id: usize,
) -> UnitSweepDelta {
    let row = lookup.row(temperature_id);
    let mut gain_sum = 0i64;
    let mut magnetization = 0i64;
    sweep_sites(lattice, spin_slice, &lookup.couplings, |spins, i, h| {
        let spin = i32::from(spins[i]);
        let gain = -spin * h;
        let accept = rng.next_u64() < row[(gain + lookup.offset) as usize];
        if TRACK {
            let flip = i32::from(accept);
            gain_sum += i64::from(gain * flip);
            magnetization -= i64::from(2 * spin * flip);
        }
        flip_if(spins, i, accept);
    });
    // A flip with gain g = -s h raises -H by 2g.
    UnitSweepDelta {
        interaction: 2 * gain_sum,
        magnetization,
    }
}

/// Single-spin-flip sweep over all replicas.
///
/// Uses the integer lookup kernel when `lookup` is given; it must have been built
/// from the same couplings, temperatures and acceptance rule. That kernel also writes
/// each system's exact observable change into `deltas` (indexed by system id).
#[cfg_attr(feature = "profile", inline(never))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn single_spin_sweep(
    lattice: &Lattice,
    spins: &mut [i8],
    couplings: &[f32],
    temperatures: &[f32],
    system_ids: &[usize],
    rngs: &mut [Xoshiro256StarStar],
    sequential: bool,
    acceptance: Acceptance,
    lookup: Option<&UnitCouplingLookup>,
    deltas: Option<&mut [UnitSweepDelta]>,
) {
    let n_spins = lattice.n_spins;
    if let Some(lookup) = lookup {
        let Some(deltas) = deltas else {
            par_over_replicas(
                spins,
                rngs,
                temperatures,
                system_ids,
                n_spins,
                sequential,
                |spin_slice, rng, _temperature, temperature_id, _system_id| {
                    unit_kernel::<false>(lattice, spin_slice, rng, lookup, temperature_id);
                },
            );
            return;
        };
        par_over_replicas_with(
            spins,
            rngs,
            temperatures,
            system_ids,
            n_spins,
            sequential,
            deltas,
            |spin_slice, rng, _temperature, temperature_id, _system_id, delta| {
                *delta = unit_kernel::<true>(lattice, spin_slice, rng, lookup, temperature_id);
            },
        );
        return;
    }

    par_over_replicas(
        spins,
        rngs,
        temperatures,
        system_ids,
        n_spins,
        sequential,
        |spin_slice, rng, temperature, _, _| {
            let beta2 = 2.0 / temperature;
            sweep_sites(lattice, spin_slice, couplings, |spins, i, h| {
                let gain = -(spins[i] as f32) * h;
                let accept = rng.next_u64() < threshold(acceptance.probability(gain, beta2));
                flip_if(spins, i, accept);
            });
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::hypercubic;
    use crate::test_utils::{chi2_cutoff, ExactLaw};
    use rand::SeedableRng;

    #[allow(clippy::too_many_arguments)]
    fn sweep(
        lattice: &Lattice,
        spins: &mut [i8],
        couplings: &[f32],
        temperatures: &[f32],
        system_ids: &[usize],
        rngs: &mut [Xoshiro256StarStar],
        acceptance: Acceptance,
        lookup: Option<&UnitCouplingLookup>,
    ) {
        single_spin_sweep(
            lattice,
            spins,
            couplings,
            temperatures,
            system_ids,
            rngs,
            true,
            acceptance,
            lookup,
            None,
        );
    }

    #[test]
    fn square_specialization_matches_generic_trajectory() {
        let shape = vec![5, 7];
        let specialized = Lattice::new(shape.clone());
        let generic = Lattice::with_offsets(shape, hypercubic(2));
        let n_spins = specialized.n_spins;
        let initial_spins: Vec<i8> = (0..2 * n_spins)
            .map(|i| if i % 3 == 0 { -1 } else { 1 })
            .collect();
        let couplings: Vec<f32> = (0..n_spins * 2)
            .map(|i| ((i % 7) as f32 - 3.0) / 3.0)
            .collect();
        let temperatures = [0.8, 2.5];
        let system_ids = [0, 1];
        let initial_rngs = [
            Xoshiro256StarStar::seed_from_u64(42),
            Xoshiro256StarStar::seed_from_u64(43),
        ];

        for acceptance in [Acceptance::Metropolis, Acceptance::Gibbs] {
            let mut specialized_spins = initial_spins.clone();
            let mut generic_spins = initial_spins.clone();
            let mut specialized_rngs = initial_rngs.clone();
            let mut generic_rngs = initial_rngs.clone();
            for (lattice, spins, rngs) in [
                (&specialized, &mut specialized_spins, &mut specialized_rngs),
                (&generic, &mut generic_spins, &mut generic_rngs),
            ] {
                sweep(
                    lattice,
                    spins,
                    &couplings,
                    &temperatures,
                    &system_ids,
                    rngs,
                    acceptance,
                    None,
                );
            }

            assert_eq!(specialized_spins, generic_spins);
            for (specialized_rng, generic_rng) in
                specialized_rngs.iter_mut().zip(generic_rngs.iter_mut())
            {
                assert_eq!(specialized_rng.next_u64(), generic_rng.next_u64());
            }
        }
    }

    #[test]
    fn unit_lookup_is_fail_closed() {
        let rule = Acceptance::Metropolis;
        assert!(UnitCouplingLookup::new(&[-1.0, 0.0, 1.0], &[0.5, 2.0], 2, rule).is_some());
        assert!(UnitCouplingLookup::new(&[-2.0, 2.0], &[1.0], 2, rule).is_none());
        assert!(UnitCouplingLookup::new(&[-1.0, f32::NAN], &[1.0], 2, rule).is_none());
        assert!(UnitCouplingLookup::new(&[-1.0, 1.0], &[0.0], 2, rule).is_none());
        assert!(UnitCouplingLookup::new(&[-1.0, 1.0], &[f32::from_bits(1)], 2, rule).is_none());
    }

    /// `draw < threshold` accepts with probability threshold / 2^64, which must match the
    /// exact rule in relative terms, including rates far below the old 2^-24 floor.
    #[test]
    fn acceptance_thresholds_are_relatively_exact() {
        for temperature in [0.1f32, 0.35, 1.0, 2.269, 7.0] {
            let lookup_metropolis =
                UnitCouplingLookup::new(&[1.0], &[temperature], 3, Acceptance::Metropolis).unwrap();
            let lookup_gibbs =
                UnitCouplingLookup::new(&[1.0], &[temperature], 3, Acceptance::Gibbs).unwrap();
            let temperature = f64::from(temperature);
            for gain in -6i32..=6 {
                let delta_h = -2.0 * gain as f64;
                // f32 exponent arithmetic limits the relative accuracy to ~|dH/T| * 2^-24.
                let tolerance = 1e-6 + 8.0 * (delta_h / temperature).abs() * 2f64.powi(-24);
                let exact_metropolis = (-delta_h / temperature).exp().min(1.0);
                let exact_gibbs = 1.0 / (1.0 + (delta_h / temperature).exp());
                for (lookup, exact) in [
                    (&lookup_metropolis, exact_metropolis),
                    (&lookup_gibbs, exact_gibbs),
                ] {
                    let rate = lookup.row(0)[(gain + 6) as usize] as f64 / 2f64.powi(64);
                    if exact < 1e-37 {
                        assert!(rate <= exact.max(2f64.powi(-64)) * 2.0);
                        continue;
                    }
                    assert!(
                        (rate - exact).abs() <= tolerance * exact + 2f64.powi(-64),
                        "T={temperature} gain={gain}: rate {rate:e} vs exact {exact:e}"
                    );
                }
            }
        }
    }

    fn assert_lookup_matches_generic(lattice: &Lattice, acceptance: Acceptance) {
        let n_spins = lattice.n_spins;
        let couplings: Vec<f32> = (0..n_spins * lattice.n_neighbors)
            .map(|i| match i % 3 {
                0 => -1.0,
                1 => 0.0,
                _ => 1.0,
            })
            .collect();
        let temperatures = [0.7, 2.0, 5.0];
        let system_ids = [2, 0, 1];
        let initial_spins: Vec<i8> = (0..3 * n_spins)
            .map(|i| if i % 5 == 0 { -1 } else { 1 })
            .collect();
        let initial_rngs = [
            Xoshiro256StarStar::seed_from_u64(71),
            Xoshiro256StarStar::seed_from_u64(72),
            Xoshiro256StarStar::seed_from_u64(73),
        ];
        let lookup =
            UnitCouplingLookup::new(&couplings, &temperatures, lattice.n_neighbors, acceptance)
                .unwrap();
        let mut table_spins = initial_spins.clone();
        let mut generic_spins = initial_spins;
        let mut table_rngs = initial_rngs.clone();
        let mut generic_rngs = initial_rngs;

        for _ in 0..20 {
            sweep(
                lattice,
                &mut table_spins,
                &couplings,
                &temperatures,
                &system_ids,
                &mut table_rngs,
                acceptance,
                Some(&lookup),
            );
            sweep(
                lattice,
                &mut generic_spins,
                &couplings,
                &temperatures,
                &system_ids,
                &mut generic_rngs,
                acceptance,
                None,
            );
        }

        assert_eq!(table_spins, generic_spins);
        for (table_rng, generic_rng) in table_rngs.iter_mut().zip(generic_rngs.iter_mut()) {
            assert_eq!(table_rng.next_u64(), generic_rng.next_u64());
        }
    }

    #[test]
    fn unit_lookup_matches_generic_with_permuted_systems() {
        for acceptance in [Acceptance::Metropolis, Acceptance::Gibbs] {
            assert_lookup_matches_generic(&Lattice::new(vec![8, 8]), acceptance);
            assert_lookup_matches_generic(
                &Lattice::with_offsets(vec![8, 8], hypercubic(2)),
                acceptance,
            );
            assert_lookup_matches_generic(&Lattice::new(vec![4, 4, 4]), acceptance);
        }
    }

    /// Draws configurations exactly from the Boltzmann law on a small torus, applies one
    /// sweep and compares the full state histogram to the exact law. A sweep that
    /// satisfies (total) balance leaves the law invariant after a single step.
    fn assert_sweep_preserves_boltzmann(
        couplings: &[f32],
        temperature: f32,
        acceptance: Acceptance,
        use_lookup: bool,
    ) {
        const TRIALS: usize = 60_000;
        let lattice = Lattice::new(vec![3, 2]);
        let law = ExactLaw::new(&lattice, couplings, temperature);
        let lookup = use_lookup.then(|| {
            UnitCouplingLookup::new(couplings, &[temperature], lattice.n_neighbors, acceptance)
                .unwrap()
        });

        let mut draw_rng = Xoshiro256StarStar::seed_from_u64(3);
        let mut rngs = vec![Xoshiro256StarStar::seed_from_u64(5)];
        let mut counts = vec![0usize; law.states.len()];
        for _ in 0..TRIALS {
            let mut spins = law.sample(&mut draw_rng).to_vec();
            sweep(
                &lattice,
                &mut spins,
                couplings,
                &[temperature],
                &[0],
                &mut rngs,
                acceptance,
                lookup.as_ref(),
            );
            counts[ExactLaw::index(&spins)] += 1;
        }

        let chi2 = law.chi2(&counts);
        assert!(
            chi2 < chi2_cutoff(counts.len() - 1),
            "{acceptance:?} lookup={use_lookup} T={temperature}: chi2 {chi2:.1}"
        );
    }

    #[test]
    fn sweeps_preserve_boltzmann() {
        let gaussian_like: Vec<f32> = [
            0.9, -1.3, 0.4, -0.7, 1.1, -0.2, 0.6, -1.5, 0.3, 1.2, -0.5, 0.8,
        ]
        .to_vec();
        let unit: Vec<f32> = [
            1.0, -1.0, 1.0, 1.0, 0.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0,
        ]
        .to_vec();
        for acceptance in [Acceptance::Metropolis, Acceptance::Gibbs] {
            for temperature in [0.8, 2.5] {
                assert_sweep_preserves_boltzmann(&gaussian_like, temperature, acceptance, false);
                assert_sweep_preserves_boltzmann(&unit, temperature, acceptance, false);
                assert_sweep_preserves_boltzmann(&unit, temperature, acceptance, true);
            }
        }
    }
}
