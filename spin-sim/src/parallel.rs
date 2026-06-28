use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;

/// Dispatch a per-replica closure over replicas, optionally in parallel.
///
/// Each replica gets a mutable spin slice and its own RNG. The closure receives
/// `(spin_slice, rng, temp)`.
///
/// When `sequential` is true, replicas are processed on the current thread
/// (no rayon overhead, best when outer-level parallelism over disorder
/// realizations already saturates all physical cores).
///
/// SAFETY: relies on `system_ids` mapping each temp_id to a unique system_id
/// so that each parallel task touches a disjoint spin slice and RNG.
pub fn par_over_replicas(
    spins: &mut [i8],
    rngs: &mut [Xoshiro256StarStar],
    temperatures: &[f32],
    system_ids: &[usize],
    n_spins: usize,
    sequential: bool,
    body: impl Fn(&mut [i8], &mut Xoshiro256StarStar, f32, usize) + Send + Sync,
) {
    let sp = spins.as_mut_ptr() as usize;
    let rp = rngs.as_mut_ptr() as usize;

    let work = |temp_id: usize| unsafe {
        let system_id = system_ids[temp_id];
        let spin_slice =
            std::slice::from_raw_parts_mut((sp as *mut i8).add(system_id * n_spins), n_spins);
        let rng = &mut *(rp as *mut Xoshiro256StarStar).add(system_id);
        let temp = temperatures[temp_id];
        body(spin_slice, rng, temp, system_id);
    };

    if sequential {
        (0..system_ids.len()).for_each(work);
    } else {
        (0..system_ids.len()).into_par_iter().for_each(work);
    }
}
