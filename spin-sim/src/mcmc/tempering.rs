use crate::spins::model::Real;
use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TemperingAttempt {
    pub edge: usize,
    pub accepted: bool,
    pub left_system: usize,
    pub right_system: usize,
}

/// Parallel tempering: attempt to swap adjacent temperature pairs.
///
/// Picks a random adjacent pair (temp_id, temp_id+1) and applies the
/// Metropolis criterion using total energies.
///
/// `energies`: cached -H/N per system id
/// `n_spins`: total number of spins (for converting to total energy)
#[cfg_attr(feature = "profile", inline(never))]
pub fn parallel_tempering<T: Real>(
    energies: &[T],
    temperatures: &[T],
    system_ids: &mut [usize],
    n_spins: usize,
    rng: &mut Xoshiro256StarStar,
    mut on_attempt: impl FnMut(TemperingAttempt),
) {
    let n_temps = system_ids.len();
    if n_temps < 2 {
        return;
    }

    let temp_id = rng.gen_range(0..n_temps - 1);
    on_attempt(attempt_edge(
        energies,
        temperatures,
        system_ids,
        n_spins,
        rng,
        temp_id,
    ));
}

#[cfg_attr(feature = "profile", inline(never))]
pub fn parallel_tempering_full_ladder<T: Real>(
    energies: &[T],
    temperatures: &[T],
    system_ids: &mut [usize],
    n_spins: usize,
    rng: &mut Xoshiro256StarStar,
    first_parity: usize,
    mut on_attempt: impl FnMut(TemperingAttempt),
) {
    let n_temps = system_ids.len();
    if n_temps < 2 {
        return;
    }

    for parity in [first_parity, 1 - first_parity] {
        for edge in (parity..n_temps - 1).step_by(2) {
            on_attempt(attempt_edge(
                energies,
                temperatures,
                system_ids,
                n_spins,
                rng,
                edge,
            ));
        }
    }
}

fn attempt_edge<T: Real>(
    energies: &[T],
    temperatures: &[T],
    system_ids: &mut [usize],
    n_spins: usize,
    rng: &mut Xoshiro256StarStar,
    temp_id: usize,
) -> TemperingAttempt {
    let temp_1 = temperatures[temp_id];
    let temp_2 = temperatures[temp_id + 1];
    let energy_1 = energies[system_ids[temp_id]];
    let energy_2 = energies[system_ids[temp_id + 1]];
    let left_system = system_ids[temp_id];
    let right_system = system_ids[temp_id + 1];

    let accepted = T::exchange_accept(energy_1, energy_2, temp_1, temp_2, n_spins, rng);
    if accepted {
        system_ids.swap(temp_id, temp_id + 1);
    }

    TemperingAttempt {
        edge: temp_id,
        accepted,
        left_system,
        right_system,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn full_ladder_attempts_every_edge_in_requested_parity_order() {
        let energies = [0.0; 5];
        let temperatures = [0.5, 0.8, 1.2, 2.0, 4.0];
        let mut system_ids = vec![0, 1, 2, 3, 4];
        let mut rng = Xoshiro256StarStar::seed_from_u64(17);
        let mut edges = Vec::new();
        parallel_tempering_full_ladder(
            &energies,
            &temperatures,
            &mut system_ids,
            64,
            &mut rng,
            0,
            |attempt| edges.push(attempt.edge),
        );
        assert_eq!(edges, vec![0, 2, 1, 3]);

        edges.clear();
        parallel_tempering_full_ladder(
            &energies,
            &temperatures,
            &mut system_ids,
            64,
            &mut rng,
            1,
            |attempt| edges.push(attempt.edge),
        );
        assert_eq!(edges, vec![1, 3, 0, 2]);
    }
}

#[cfg(test)]
mod stationarity_tests {
    use super::*;
    use crate::geometry::Lattice;
    use crate::test_utils::{bond_sum, chi2_cutoff, ExactLaw};
    use rand::SeedableRng;

    /// Swapping exact draws from two Boltzmann laws must leave both marginals invariant.
    #[test]
    fn exchange_preserves_both_temperatures() {
        const TRIALS: usize = 60_000;
        let lattice = Lattice::new(vec![3, 2]);
        let couplings = [
            0.9, -1.3, 0.4, -0.7, 1.1, -0.2, 0.6, -1.5, 0.3, 1.2, -0.5, 0.8,
        ];
        let temperatures = [0.9f32, 1.6];
        let laws = temperatures.map(|t| ExactLaw::new(&lattice, &couplings, t));
        let n = lattice.n_spins;
        let mut draw_rng = Xoshiro256StarStar::seed_from_u64(19);
        let mut rng = Xoshiro256StarStar::seed_from_u64(23);
        let mut counts = [vec![0usize; 1 << n], vec![0usize; 1 << n]];
        for _ in 0..TRIALS {
            let states = [
                laws[0].sample(&mut draw_rng).to_vec(),
                laws[1].sample(&mut draw_rng).to_vec(),
            ];
            let energies = states
                .clone()
                .map(|s| (bond_sum(&lattice, &couplings, &s) / n as f64) as f32);
            let mut ids = vec![0, 1];
            attempt_edge(&energies, &temperatures, &mut ids, n, &mut rng, 0);
            for (slot, &id) in ids.iter().enumerate() {
                counts[slot][ExactLaw::index(&states[id])] += 1;
            }
        }
        for (law, counts) in laws.iter().zip(&counts) {
            let chi2 = law.chi2(counts);
            assert!(chi2 < chi2_cutoff(counts.len() - 1), "chi2 {chi2:.1}");
        }
    }
}

#[cfg(test)]
mod sign_tests {
    use super::*;
    use rand::SeedableRng;
    #[test]
    fn exchange_uses_negative_physical_energy_cache() {
        // H=(-100,+100) is favorable at (cold,hot), so swapping must reject.
        let mut ids = vec![0, 1];
        let mut rng = Xoshiro256StarStar::seed_from_u64(5);
        let rejected = attempt_edge(&[100.0f64, -100.0], &[0.5, 2.0], &mut ids, 1, &mut rng, 0);
        assert!(!rejected.accepted);
        assert_eq!(ids, vec![0, 1]);
        let accepted = attempt_edge(&[-100.0f64, 100.0], &[0.5, 2.0], &mut ids, 1, &mut rng, 0);
        assert!(accepted.accepted);
        assert_eq!(ids, vec![1, 0]);
    }
}
