//! Model-specific representations, with a common interaction-energy convention.
use crate::geometry::Lattice;
use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;

/// Arithmetic retained at each model's native precision.
pub trait Real: Copy + Default + PartialOrd + Send + Sync + std::fmt::Debug {
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
    fn exchange_accept(
        e1: Self,
        e2: Self,
        t1: Self,
        t2: Self,
        n: usize,
        rng: &mut Xoshiro256StarStar,
    ) -> bool;
}
macro_rules! real_impl {
    ($t:ty, $uniform:expr) => {
        impl Real for $t {
            fn from_f64(value: f64) -> Self {
                value as Self
            }
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn exchange_accept(
                e1: Self,
                e2: Self,
                t1: Self,
                t2: Self,
                n: usize,
                rng: &mut Xoshiro256StarStar,
            ) -> bool {
                // Caches store -H/N in both models.
                let delta = (n as Self) * (e2 - e1) * (1.0 / t1 - 1.0 / t2);
                let uniform: fn(&mut Xoshiro256StarStar) -> Self = $uniform;
                delta >= uniform(rng).ln()
            }
        }
    };
}
// A 24-bit f32 uniform is exactly zero once in 2^24 draws, which would accept any
// swap; the 62-bit draw keeps rare swaps at their own rate, as the sweeps do.
real_impl!(f32, crate::mcmc::uniform_f32);
real_impl!(f64, |rng| rng.gen::<f64>());

/// A spin representation and its statically dispatched model kernels.
pub trait Spin: Copy + Default + Send + Sync {
    type Value: Real;
    fn random(rng: &mut Xoshiro256StarStar) -> Self;
    /// Recompute the interaction sum per site, -H/N, indexed by system.
    fn interactions(
        lattice: &Lattice,
        spins: &[Self],
        couplings: &[Self::Value],
        out: &mut [Self::Value],
    );
    fn components(self) -> [f64; 2];
}
impl Spin for i8 {
    type Value = f32;
    fn random(rng: &mut Xoshiro256StarStar) -> Self {
        if rng.gen::<f32>() < 0.5 {
            -1
        } else {
            1
        }
    }
    fn interactions(lattice: &Lattice, spins: &[Self], couplings: &[f32], out: &mut [f32]) {
        super::energy::compute_energies_into(lattice, spins, couplings, out);
    }
    fn components(self) -> [f64; 2] {
        [self as f64, 0.0]
    }
}
impl Spin for [f64; 2] {
    type Value = f64;
    fn random(rng: &mut Xoshiro256StarStar) -> Self {
        let (s, c) = (std::f64::consts::TAU * rng.gen::<f64>()).sin_cos();
        [c, s]
    }
    fn interactions(lattice: &Lattice, spins: &[Self], couplings: &[f64], out: &mut [f64]) {
        for (configuration, energy) in spins.chunks_exact(lattice.n_spins).zip(out) {
            *energy =
                super::xy::interaction(lattice, configuration, couplings) / lattice.n_spins as f64;
        }
    }
    fn components(self) -> [f64; 2] {
        self
    }
}
