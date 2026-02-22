use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;

/// Parallel tempering: attempt to swap adjacent temperature pairs.
///
/// Picks a random adjacent pair (temp_id, temp_id+1) and applies the
/// Metropolis criterion using total energies.
///
/// `energies`: per-replica average energy (energy per spin)
/// `n_spins`: total number of spins (for converting to total energy)
pub fn parallel_tempering(
    energies: &[f32],
    temperatures: &[f32],
    system_ids: &mut [usize],
    n_spins: usize,
    rng: &mut Xoshiro256StarStar,
) {
    let n_temps = system_ids.len();
    if n_temps < 2 {
        return;
    }

    let temp_id = rng.gen_range(0..n_temps - 1);
    let temp_1 = temperatures[temp_id];
    let temp_2 = temperatures[temp_id + 1];
    let energy_1 = energies[system_ids[temp_id]];
    let energy_2 = energies[system_ids[temp_id + 1]];

    let delta = (n_spins as f32) * (energy_2 - energy_1) * (1.0 / temp_1 - 1.0 / temp_2);
    let log_rand = (rng.gen::<f32>()).ln();

    if delta >= log_rand {
        system_ids.swap(temp_id, temp_id + 1);
    }
}
