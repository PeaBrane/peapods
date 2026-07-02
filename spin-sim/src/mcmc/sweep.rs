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
    rng: &mut Xoshiro256StarStar,
    temp: f32,
    threshold_fn: &impl Fn(&mut Xoshiro256StarStar, f32) -> f32,
) {
    let Some((height, width)) = lattice.square_shape() else {
        for i in 0..lattice.n_spins {
            let h = local_field(lattice, spin_slice, couplings, i);
            attempt_flip(spin_slice, rng, temp, i, h, threshold_fn);
        }
        return;
    };

    if height < 3 || width < 3 {
        for i in 0..lattice.n_spins {
            let h = local_field(lattice, spin_slice, couplings, i);
            attempt_flip(spin_slice, rng, temp, i, h, threshold_fn);
        }
        return;
    }

    for i in 0..width {
        let h = local_field(lattice, spin_slice, couplings, i);
        attempt_flip(spin_slice, rng, temp, i, h, threshold_fn);
    }

    for row in 1..height - 1 {
        let row_start = row * width;
        let h = local_field(lattice, spin_slice, couplings, row_start);
        attempt_flip(spin_slice, rng, temp, row_start, h, threshold_fn);

        for i in row_start + 1..row_start + width - 1 {
            let h = square_interior_field(spin_slice, couplings, i, width);
            attempt_flip(spin_slice, rng, temp, i, h, threshold_fn);
        }

        let row_end = row_start + width - 1;
        let h = local_field(lattice, spin_slice, couplings, row_end);
        attempt_flip(spin_slice, rng, temp, row_end, h, threshold_fn);
    }

    for i in (height - 1) * width..height * width {
        let h = local_field(lattice, spin_slice, couplings, i);
        attempt_flip(spin_slice, rng, temp, i, h, threshold_fn);
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
        |spin_slice, rng, temp, _| {
            sweep_sites(lattice, spin_slice, couplings, rng, temp, &threshold_fn);
        },
    );
}

/// Metropolis single-spin-flip sweep over all replicas.
#[cfg_attr(feature = "profile", inline(never))]
pub fn metropolis_sweep(
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
        );
        metropolis_sweep(
            &generic,
            &mut generic_spins,
            &couplings,
            &temperatures,
            &system_ids,
            &mut generic_rngs,
            true,
        );

        assert_eq!(specialized_spins, generic_spins);
        for (specialized_rng, generic_rng) in
            specialized_rngs.iter_mut().zip(generic_rngs.iter_mut())
        {
            assert_eq!(specialized_rng.next_u64(), generic_rng.next_u64());
        }
    }
}
