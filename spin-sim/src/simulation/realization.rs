use crate::geometry::Lattice;
use crate::spins;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;

/// Mutable state for one disorder realization.
///
/// Holds the coupling array (fixed after construction), spin configurations for
/// every replica at every temperature, and bookkeeping for parallel tempering.
///
/// With `n_replicas` replicas and `n_temps` temperatures there are
/// `n_systems = n_replicas * n_temps` independent spin configurations.
/// Spins are stored in a single flat `Vec` of length `n_systems * n_spins`,
/// where system `i` occupies `spins[i*n_spins .. (i+1)*n_spins]`.
pub struct Realization {
    /// Forward couplings, length `n_spins * n_neighbors`.
    pub couplings: Vec<f32>,
    /// All spin configurations, length `n_systems * n_spins` (+1/−1).
    pub spins: Vec<i8>,
    /// Temperature assigned to each system slot, length `n_systems`.
    pub temperatures: Vec<f32>,
    /// Parallel-tempering permutation: `system_ids[slot]` is the system index
    /// currently occupying temperature slot `slot`.
    pub system_ids: Vec<usize>,
    /// One PRNG per system.
    pub rngs: Vec<Xoshiro256StarStar>,
    /// One PRNG per overlap-update pair slot, length `n_temps * (n_replicas / 2)`.
    pub pair_rngs: Vec<Xoshiro256StarStar>,
    /// Cached total energy per system (E / N), length `n_systems`.
    pub energies: Vec<f32>,
}

impl Realization {
    /// Initialize a realization with random ±1 spins.
    ///
    /// Seeds replica RNGs deterministically as `base_seed, base_seed+1, …`.
    pub fn new(
        lattice: &Lattice,
        couplings: Vec<f32>,
        temps: &[f32],
        n_replicas: usize,
        base_seed: u64,
    ) -> Self {
        let n_spins = lattice.n_spins;
        let n_temps = temps.len();
        let n_systems = n_replicas * n_temps;

        let temperatures = temps.repeat(n_replicas);

        let mut rngs = Vec::with_capacity(n_systems);
        for i in 0..n_systems {
            rngs.push(Xoshiro256StarStar::seed_from_u64(base_seed + i as u64));
        }

        let mut spins = vec![0i8; n_systems * n_spins];
        for (i, rng) in rngs.iter_mut().enumerate() {
            for j in 0..n_spins {
                spins[i * n_spins + j] = if rng.gen::<f32>() < 0.5 { -1 } else { 1 };
            }
        }

        let system_ids: Vec<usize> = (0..n_systems).collect();

        let n_pairs = n_replicas / 2;
        let mut pair_rngs = Vec::with_capacity(n_temps * n_pairs);
        for i in 0..n_temps * n_pairs {
            pair_rngs.push(Xoshiro256StarStar::seed_from_u64(
                base_seed + n_systems as u64 + i as u64,
            ));
        }

        let (energies, _) =
            spins::energy::compute_energies(lattice, &spins, &couplings, n_systems, false);

        Self {
            couplings,
            spins,
            temperatures,
            system_ids,
            rngs,
            pair_rngs,
            energies,
        }
    }

    /// Re-randomize all spins and reset the tempering permutation.
    pub fn reset(&mut self, lattice: &Lattice, n_replicas: usize, n_temps: usize, base_seed: u64) {
        let n_spins = lattice.n_spins;
        let n_systems = n_replicas * n_temps;

        for i in 0..n_systems {
            self.rngs[i] = Xoshiro256StarStar::seed_from_u64(base_seed + i as u64);
            for j in 0..n_spins {
                self.spins[i * n_spins + j] = if self.rngs[i].gen::<f32>() < 0.5 {
                    -1
                } else {
                    1
                };
            }
        }

        self.system_ids = (0..n_systems).collect();

        let n_pairs = n_replicas / 2;
        for i in 0..n_temps * n_pairs {
            self.pair_rngs[i] =
                Xoshiro256StarStar::seed_from_u64(base_seed + n_systems as u64 + i as u64);
        }

        let (energies, _) = spins::energy::compute_energies(
            lattice,
            &self.spins,
            &self.couplings,
            n_systems,
            false,
        );
        self.energies = energies;
    }
}
