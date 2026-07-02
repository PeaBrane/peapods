use crate::geometry::Lattice;
use crate::parallel::par_over_replicas;
use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;

/// Compute local field for spin `i` from all `2 * n_neighbors` neighbors.
#[inline]
fn local_field(lattice: &Lattice, spin_slice: &[i8], couplings: &[f32], i: usize) -> f32 {
    let n_neighbors = lattice.n_neighbors;
    let mut h = 0.0f32;
    for d in 0..n_neighbors {
        let j_fwd = lattice.neighbor_fwd(i, d);
        h += spin_slice[j_fwd] as f32 * couplings[i * n_neighbors + d];

        let j_bwd = lattice.neighbor_bwd(i, d);
        h += spin_slice[j_bwd] as f32 * couplings[j_bwd * n_neighbors + d];
    }
    h
}

#[inline]
fn square_interior_field(spin_slice: &[i8], couplings: &[f32], i: usize, width: usize) -> f32 {
    // Valid only for non-boundary sites on a canonical 2D lattice, whose
    // directions have strides `width` and 1. Backward terms use the neighboring
    // site's coupling because couplings own forward bonds.
    let mut h = 0.0f32;
    h += spin_slice[i + width] as f32 * couplings[i * 2];
    h += spin_slice[i - width] as f32 * couplings[(i - width) * 2];
    h += spin_slice[i + 1] as f32 * couplings[i * 2 + 1];
    h += spin_slice[i - 1] as f32 * couplings[(i - 1) * 2 + 1];
    h
}

#[inline]
fn attempt_flip(
    spin_slice: &mut [i8],
    rng: &mut Xoshiro256StarStar,
    temp: f32,
    i: usize,
    h: f32,
    threshold_fn: &impl Fn(&mut Xoshiro256StarStar, f32) -> f32,
) {
    let si = spin_slice[i] as f32;
    let eng_change = -si * h;
    if eng_change >= threshold_fn(rng, temp) {
        spin_slice[i] = -spin_slice[i];
    }
}

#[inline]
fn sweep_sites(
    lattice: &Lattice,
    spin_slice: &mut [i8],
    couplings: &[f32],
    mut attempt: impl FnMut(&mut [i8], usize, f32),
) {
    let Some((height, width)) = lattice.square_shape() else {
        for i in 0..lattice.n_spins {
            let h = local_field(lattice, spin_slice, couplings, i);
            attempt(spin_slice, i, h);
        }
        return;
    };

    if height < 3 || width < 3 {
        for i in 0..lattice.n_spins {
            let h = local_field(lattice, spin_slice, couplings, i);
            attempt(spin_slice, i, h);
        }
        return;
    }

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

const F32_UNIFORM_BITS: u32 = 24;
const F32_UNIFORM_VALUES: u32 = 1 << F32_UNIFORM_BITS;

pub(crate) struct UnitCouplingMetropolisLookup {
    accepted_counts: Vec<u32>,
    offset: i32,
    table_width: usize,
}

impl UnitCouplingMetropolisLookup {
    pub(crate) fn new(couplings: &[f32], temperatures: &[f32], n_neighbors: usize) -> Option<Self> {
        if !couplings
            .iter()
            .all(|&coupling| coupling == -1.0 || coupling == 0.0 || coupling == 1.0)
            || !temperatures
                .iter()
                .all(|temperature| temperature.is_finite() && *temperature / 2.0 > 0.0)
        {
            return None;
        }

        let offset = n_neighbors.checked_mul(2)?;
        if offset > F32_UNIFORM_VALUES as usize {
            return None;
        }
        let table_width = offset.checked_mul(2)?.checked_add(1)?;
        let offset_i32 = i32::try_from(offset).ok()?;
        let capacity = temperatures.len().checked_mul(table_width)?;
        let mut accepted_counts = Vec::with_capacity(capacity);
        for &temperature in temperatures {
            for energy_change in -offset_i32..=offset_i32 {
                accepted_counts.push(Self::accepted_count(temperature, energy_change));
            }
        }

        Some(Self {
            accepted_counts,
            offset: offset_i32,
            table_width,
        })
    }

    fn accepted_count(temperature: f32, energy_change: i32) -> u32 {
        let mut low = 0u32;
        let mut high = F32_UNIFORM_VALUES;
        while low < high {
            let midpoint = low + (high - low) / 2;
            if Self::legacy_accepts(temperature, energy_change, midpoint) {
                low = midpoint + 1;
            } else {
                high = midpoint;
            }
        }
        low
    }

    #[inline]
    fn legacy_accepts(temperature: f32, energy_change: i32, draw: u32) -> bool {
        let uniform = draw as f32 / F32_UNIFORM_VALUES as f32;
        energy_change as f32 >= (temperature / 2.0) * uniform.ln()
    }

    #[inline]
    fn get(&self, temperature_id: usize, energy_change: i32) -> u32 {
        let energy_index = usize::try_from(energy_change + self.offset)
            .expect("unit coupling energy change is out of range");
        self.accepted_counts[temperature_id * self.table_width + energy_index]
    }
}

#[inline]
fn attempt_flip_lookup(
    spin_slice: &mut [i8],
    rng: &mut Xoshiro256StarStar,
    i: usize,
    h: f32,
    accepted_counts: &UnitCouplingMetropolisLookup,
    temperature_id: usize,
) {
    let energy_change = (-spin_slice[i] as f32 * h) as i32;
    // rand's Standard f32 uses the high 24 bits of one u32 draw. Comparing the
    // same integer grid preserves the legacy logarithmic decision and RNG use.
    let draw = rng.gen::<u32>() >> (u32::BITS - F32_UNIFORM_BITS);
    if draw < accepted_counts.get(temperature_id, energy_change) {
        spin_slice[i] = -spin_slice[i];
    }
}

/// Single-spin-flip sweep with a generic acceptance threshold.
///
/// `threshold_fn(rng, temp)` returns the value that `eng_change` is compared against.
#[allow(clippy::too_many_arguments)]
fn sweep_generic(
    lattice: &Lattice,
    spins: &mut [i8],
    couplings: &[f32],
    temperatures: &[f32],
    system_ids: &[usize],
    rngs: &mut [Xoshiro256StarStar],
    sequential: bool,
    threshold_fn: impl Fn(&mut Xoshiro256StarStar, f32) -> f32 + Send + Sync,
) {
    let n_spins = lattice.n_spins;
    par_over_replicas(
        spins,
        rngs,
        temperatures,
        system_ids,
        n_spins,
        sequential,
        |spin_slice, rng, temp, _, _| {
            sweep_sites(lattice, spin_slice, couplings, |spins, i, h| {
                attempt_flip(spins, rng, temp, i, h, &threshold_fn);
            });
        },
    );
}

/// Metropolis single-spin-flip sweep over all replicas.
#[cfg_attr(feature = "profile", inline(never))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn metropolis_sweep(
    lattice: &Lattice,
    spins: &mut [i8],
    couplings: &[f32],
    temperatures: &[f32],
    system_ids: &[usize],
    rngs: &mut [Xoshiro256StarStar],
    sequential: bool,
    lookup: Option<&UnitCouplingMetropolisLookup>,
) {
    if let Some(lookup) = lookup {
        par_over_replicas(
            spins,
            rngs,
            temperatures,
            system_ids,
            lattice.n_spins,
            sequential,
            |spin_slice, rng, _temperature, temperature_id, _system_id| {
                sweep_sites(lattice, spin_slice, couplings, |spins, i, h| {
                    attempt_flip_lookup(spins, rng, i, h, lookup, temperature_id);
                });
            },
        );
        return;
    }

    sweep_generic(
        lattice,
        spins,
        couplings,
        temperatures,
        system_ids,
        rngs,
        sequential,
        // A guaranteed-accept branch intended to skip RNG/logarithm work was benchmarked and regressed, so this path is intentional.
        |rng, temp| (temp / 2.0) * rng.gen::<f32>().ln(),
    );
}

/// Gibbs single-spin-flip sweep over all replicas.
#[cfg_attr(feature = "profile", inline(never))]
pub fn gibbs_sweep(
    lattice: &Lattice,
    spins: &mut [i8],
    couplings: &[f32],
    temperatures: &[f32],
    system_ids: &[usize],
    rngs: &mut [Xoshiro256StarStar],
    sequential: bool,
) {
    sweep_generic(
        lattice,
        spins,
        couplings,
        temperatures,
        system_ids,
        rngs,
        sequential,
        |rng, temp| {
            let u: f32 = rng.gen();
            (temp / 2.0) * (u / (1.0 - u)).ln()
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::hypercubic;
    use rand::{RngCore, SeedableRng};

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

        let mut specialized_spins = initial_spins.clone();
        let mut generic_spins = initial_spins;
        let mut specialized_rngs = initial_rngs.clone();
        let mut generic_rngs = initial_rngs;

        metropolis_sweep(
            &specialized,
            &mut specialized_spins,
            &couplings,
            &temperatures,
            &system_ids,
            &mut specialized_rngs,
            true,
            None,
        );
        metropolis_sweep(
            &generic,
            &mut generic_spins,
            &couplings,
            &temperatures,
            &system_ids,
            &mut generic_rngs,
            true,
            None,
        );

        assert_eq!(specialized_spins, generic_spins);
        for (specialized_rng, generic_rng) in
            specialized_rngs.iter_mut().zip(generic_rngs.iter_mut())
        {
            assert_eq!(specialized_rng.next_u64(), generic_rng.next_u64());
        }
    }

    #[test]
    fn unit_lookup_is_fail_closed() {
        assert!(UnitCouplingMetropolisLookup::new(&[-1.0, 0.0, 1.0], &[0.5, 2.0], 2).is_some());
        assert!(UnitCouplingMetropolisLookup::new(&[-2.0, 2.0], &[1.0], 2).is_none());
        assert!(UnitCouplingMetropolisLookup::new(&[-1.0, f32::NAN], &[1.0], 2).is_none());
        assert!(UnitCouplingMetropolisLookup::new(&[-1.0, 1.0], &[0.0], 2).is_none());
        assert!(
            UnitCouplingMetropolisLookup::new(&[-1.0, 1.0], &[f32::from_bits(1)], 2,).is_none()
        );
    }

    #[test]
    fn unit_lookup_cutoffs_match_legacy_boundaries() {
        let temperatures = [0.7, 2.0, 5.0];
        let lookup =
            UnitCouplingMetropolisLookup::new(&[-1.0, 0.0, 1.0], &temperatures, 3).unwrap();
        for (temperature_id, &temperature) in temperatures.iter().enumerate() {
            for energy_change in -6..=6 {
                let count = lookup.get(temperature_id, energy_change);
                if count > 0 {
                    assert!(UnitCouplingMetropolisLookup::legacy_accepts(
                        temperature,
                        energy_change,
                        count - 1,
                    ));
                }
                if count < F32_UNIFORM_VALUES {
                    assert!(!UnitCouplingMetropolisLookup::legacy_accepts(
                        temperature,
                        energy_change,
                        count,
                    ));
                }
            }
        }
    }

    fn assert_lookup_matches_log(lattice: &Lattice) {
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
            UnitCouplingMetropolisLookup::new(&couplings, &temperatures, lattice.n_neighbors)
                .unwrap();
        let mut table_spins = initial_spins.clone();
        let mut log_spins = initial_spins;
        let mut table_rngs = initial_rngs.clone();
        let mut log_rngs = initial_rngs;

        for _ in 0..20 {
            metropolis_sweep(
                lattice,
                &mut table_spins,
                &couplings,
                &temperatures,
                &system_ids,
                &mut table_rngs,
                true,
                Some(&lookup),
            );
            metropolis_sweep(
                lattice,
                &mut log_spins,
                &couplings,
                &temperatures,
                &system_ids,
                &mut log_rngs,
                true,
                None,
            );
        }

        assert_eq!(table_spins, log_spins);
        for (table_rng, log_rng) in table_rngs.iter_mut().zip(log_rngs.iter_mut()) {
            assert_eq!(table_rng.next_u64(), log_rng.next_u64());
        }
    }

    #[test]
    fn unit_lookup_matches_log_with_permuted_systems() {
        assert_lookup_matches_log(&Lattice::new(vec![8, 8]));
        assert_lookup_matches_log(&Lattice::with_offsets(vec![8, 8], hypercubic(2)));
    }
}
