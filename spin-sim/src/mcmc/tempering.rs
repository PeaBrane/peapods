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
/// `energies`: per-replica average energy (energy per spin)
/// `n_spins`: total number of spins (for converting to total energy)
#[cfg_attr(feature = "profile", inline(never))]
pub fn parallel_tempering(
    energies: &[f32],
    temperatures: &[f32],
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
pub fn parallel_tempering_full_ladder(
    energies: &[f32],
    temperatures: &[f32],
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

fn attempt_edge(
    energies: &[f32],
    temperatures: &[f32],
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

    let delta = (n_spins as f32) * (energy_2 - energy_1) * (1.0 / temp_1 - 1.0 / temp_2);
    let log_rand = (rng.gen::<f32>()).ln();

    let accepted = delta >= log_rand;
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
