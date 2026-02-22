use crate::lattice::Lattice;
use crate::parallel::par_over_replicas;
use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;

/// Compute local field for spin `i` from all 2*n_dims neighbors.
#[inline]
fn local_field(lattice: &Lattice, spin_slice: &[i8], couplings: &[f32], i: usize) -> f32 {
    let n_dims = lattice.n_dims;
    let mut h = 0.0f32;
    for d in 0..n_dims {
        let j_fwd = lattice.neighbor(i, d, true);
        h += spin_slice[j_fwd] as f32 * couplings[i * n_dims + d];

        let j_back = lattice.neighbor(i, d, false);
        h += spin_slice[j_back] as f32 * couplings[j_back * n_dims + d];
    }
    h
}

/// Single-spin-flip sweep with a generic acceptance threshold.
///
/// `threshold_fn(rng, temp)` returns the value that `eng_change` is compared against.
fn sweep_generic(
    lattice: &Lattice,
    spins: &mut [i8],
    couplings: &[f32],
    temperatures: &[f32],
    system_ids: &[usize],
    rngs: &mut [Xoshiro256StarStar],
    threshold_fn: impl Fn(&mut Xoshiro256StarStar, f32) -> f32 + Send + Sync,
) {
    let n_spins = lattice.n_spins;
    par_over_replicas(
        spins,
        rngs,
        temperatures,
        system_ids,
        n_spins,
        |spin_slice, rng, temp, _| {
            for i in 0..n_spins {
                let si = spin_slice[i] as f32;
                let h = local_field(lattice, spin_slice, couplings, i);
                let eng_change = -si * h;
                if eng_change >= threshold_fn(rng, temp) {
                    spin_slice[i] = -spin_slice[i];
                }
            }
        },
    );
}

/// Metropolis single-spin-flip sweep over all replicas in parallel.
pub fn metropolis_sweep(
    lattice: &Lattice,
    spins: &mut [i8],
    couplings: &[f32],
    temperatures: &[f32],
    system_ids: &[usize],
    rngs: &mut [Xoshiro256StarStar],
) {
    sweep_generic(
        lattice,
        spins,
        couplings,
        temperatures,
        system_ids,
        rngs,
        |rng, temp| (temp / 2.0) * rng.gen::<f32>().ln(),
    );
}

/// Gibbs single-spin-flip sweep over all replicas in parallel.
pub fn gibbs_sweep(
    lattice: &Lattice,
    spins: &mut [i8],
    couplings: &[f32],
    temperatures: &[f32],
    system_ids: &[usize],
    rngs: &mut [Xoshiro256StarStar],
) {
    sweep_generic(
        lattice,
        spins,
        couplings,
        temperatures,
        system_ids,
        rngs,
        |rng, temp| {
            let u: f32 = rng.gen();
            (temp / 2.0) * (u / (1.0 - u)).ln()
        },
    );
}
