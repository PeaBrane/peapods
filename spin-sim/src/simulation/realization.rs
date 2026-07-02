use crate::geometry::Lattice;
use crate::spins;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;

const SYSTEM_SEED_DOMAIN: u64 = 0x53A9_17E1_4C2D_8B6F;
const PAIR_SEED_DOMAIN: u64 = 0xA147_5EED_91C3_02D7;

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}

fn child_seed(root: u64, domain: u64, index: usize) -> u64 {
    splitmix64(root ^ domain ^ splitmix64(index as u64))
}

pub(super) struct PtState {
    edge_attempts: Vec<u64>,
    edge_acceptances: Vec<u64>,
    round_trips: Vec<u64>,
    trip_state: Vec<u8>,
    next_parity: usize,
    cold_slot: usize,
    hot_slot: usize,
}

impl PtState {
    fn new(n_replicas: usize, n_temps: usize, system_ids: &[usize], temperatures: &[f32]) -> Self {
        let mut state = Self {
            edge_attempts: Vec::new(),
            edge_acceptances: Vec::new(),
            round_trips: Vec::new(),
            trip_state: Vec::new(),
            next_parity: 0,
            cold_slot: 0,
            hot_slot: 0,
        };
        state.reset(n_replicas, n_temps, system_ids, temperatures);
        state
    }

    fn reset(
        &mut self,
        n_replicas: usize,
        n_temps: usize,
        system_ids: &[usize],
        temperatures: &[f32],
    ) {
        let n_edges = n_temps.saturating_sub(1);
        self.edge_attempts.resize(n_edges, 0);
        self.edge_attempts.fill(0);
        self.edge_acceptances.resize(n_edges, 0);
        self.edge_acceptances.fill(0);
        self.round_trips.resize(n_replicas * n_temps, 0);
        self.round_trips.fill(0);
        self.trip_state.resize(n_replicas * n_temps, 0);
        self.trip_state.fill(0);
        self.next_parity = 0;
        (self.cold_slot, self.hot_slot) = Self::extreme_temperature_slots(temperatures, n_temps);

        if n_temps == 0 {
            return;
        }
        for replica in 0..n_replicas {
            self.trip_state[system_ids[replica * n_temps + self.hot_slot]] = 1;
        }
    }

    pub(super) fn record_attempt(&mut self, attempt: crate::mcmc::tempering::TemperingAttempt) {
        self.edge_attempts[attempt.edge] += 1;
        if !attempt.accepted {
            return;
        }
        self.edge_acceptances[attempt.edge] += 1;

        self.record_arrival(attempt.left_system, attempt.edge + 1);
        self.record_arrival(attempt.right_system, attempt.edge);
    }

    pub(super) fn first_parity(&self) -> usize {
        self.next_parity
    }

    pub(super) fn advance_parity(&mut self) {
        self.next_parity = 1 - self.next_parity;
    }

    fn extreme_temperature_slots(temperatures: &[f32], n_temps: usize) -> (usize, usize) {
        if n_temps == 0 {
            return (0, 0);
        }
        let mut cold_slot = 0;
        let mut hot_slot = 0;
        for slot in 1..n_temps {
            if temperatures[slot] < temperatures[cold_slot] {
                cold_slot = slot;
            }
            if temperatures[slot] > temperatures[hot_slot] {
                hot_slot = slot;
            }
        }
        (cold_slot, hot_slot)
    }

    fn record_arrival(&mut self, system: usize, slot: usize) {
        if slot == self.hot_slot {
            if self.trip_state[system] == 2 {
                self.round_trips[system] += 1;
            }
            self.trip_state[system] = 1;
            return;
        }
        if slot == self.cold_slot && self.trip_state[system] == 1 {
            self.trip_state[system] = 2;
        }
    }
}

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
    pub(super) pt: PtState,
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
            rngs.push(Xoshiro256StarStar::seed_from_u64(child_seed(
                base_seed,
                SYSTEM_SEED_DOMAIN,
                i,
            )));
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
            pair_rngs.push(Xoshiro256StarStar::seed_from_u64(child_seed(
                base_seed,
                PAIR_SEED_DOMAIN,
                i,
            )));
        }

        let mut energies = vec![0.0; n_systems];
        spins::energy::compute_energies_into(lattice, &spins, &couplings, &mut energies);

        let pt = PtState::new(n_replicas, n_temps, &system_ids, &temperatures);
        Self {
            couplings,
            spins,
            temperatures,
            system_ids,
            rngs,
            pair_rngs,
            pt,
            energies,
        }
    }

    /// Re-randomize all spins and reset the tempering permutation.
    pub fn reset(&mut self, lattice: &Lattice, n_replicas: usize, n_temps: usize, base_seed: u64) {
        let n_spins = lattice.n_spins;
        let n_systems = n_replicas * n_temps;

        for i in 0..n_systems {
            self.rngs[i] =
                Xoshiro256StarStar::seed_from_u64(child_seed(base_seed, SYSTEM_SEED_DOMAIN, i));
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
                Xoshiro256StarStar::seed_from_u64(child_seed(base_seed, PAIR_SEED_DOMAIN, i));
        }

        self.energies.resize(n_systems, 0.0);
        spins::energy::compute_energies_into(
            lattice,
            &self.spins,
            &self.couplings,
            &mut self.energies,
        );
        self.pt
            .reset(n_replicas, n_temps, &self.system_ids, &self.temperatures);
    }

    pub fn pt_edge_attempts(&self) -> &[u64] {
        &self.pt.edge_attempts
    }

    pub fn pt_edge_acceptances(&self) -> &[u64] {
        &self.pt.edge_acceptances
    }

    pub fn pt_round_trips(&self) -> &[u64] {
        &self.pt.round_trips
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcmc::tempering::TemperingAttempt;

    #[test]
    fn reset_replays_spins_and_pt_state() {
        let lattice = Lattice::new(vec![3, 3]);
        let couplings = vec![1.0; lattice.n_spins * lattice.n_neighbors];
        let mut realization = Realization::new(&lattice, couplings, &[1.0, 2.0], 2, 17);
        let initial_spins = realization.spins.clone();
        realization.system_ids.swap(0, 1);
        realization.pt.edge_attempts.fill(9);
        realization.pt.advance_parity();

        realization.reset(&lattice, 2, 2, 17);

        assert_eq!(realization.spins, initial_spins);
        assert_eq!(realization.system_ids, vec![0, 1, 2, 3]);
        assert_eq!(realization.pt_edge_attempts(), &[0]);
        assert_eq!(realization.pt.first_parity(), 0);
    }

    #[test]
    fn tracks_hot_cold_hot_across_attempts() {
        let lattice = Lattice::new(vec![2, 2]);
        let couplings = vec![1.0; lattice.n_spins * lattice.n_neighbors];
        let mut realization = Realization::new(&lattice, couplings, &[0.5, 1.0, 2.0], 1, 7);

        for (edge, left_system, right_system) in [(1, 1, 2), (0, 0, 2), (0, 2, 0), (1, 2, 1)] {
            realization.pt.record_attempt(TemperingAttempt {
                edge,
                accepted: true,
                left_system,
                right_system,
            });
        }

        assert_eq!(realization.pt_edge_attempts(), &[2, 2]);
        assert_eq!(realization.pt_edge_acceptances(), &[2, 2]);
        assert_eq!(realization.pt_round_trips()[2], 1);
    }
}
