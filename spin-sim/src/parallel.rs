use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;

/// Dispatch a per-replica closure over replicas, optionally in parallel.
///
/// Each replica gets a mutable spin slice and its own RNG. The closure receives
/// `(spin_slice, rng, temp, temp_id, system_id)`.
///
/// When `sequential` is true, replicas are processed on the current thread
/// (no rayon overhead, best when outer-level parallelism over disorder
/// realizations already saturates all physical cores).
///
/// SAFETY: relies on `system_ids` mapping each temp_id to a unique system_id
/// so that each parallel task touches a disjoint spin slice and RNG.
pub(crate) fn par_over_replicas<S: Copy + Send + Sync, T: Copy + Send + Sync>(
    spins: &mut [S],
    rngs: &mut [Xoshiro256StarStar],
    temperatures: &[T],
    system_ids: &[usize],
    n_spins: usize,
    sequential: bool,
    body: impl Fn(&mut [S], &mut Xoshiro256StarStar, T, usize, usize) + Send + Sync,
) {
    let mut outputs = vec![(); rngs.len()];
    par_over_replicas_with(
        spins,
        rngs,
        temperatures,
        system_ids,
        n_spins,
        sequential,
        &mut outputs,
        |spin_slice, rng, temp, temp_id, system_id, _| {
            body(spin_slice, rng, temp, temp_id, system_id)
        },
    );
}

/// Like [`par_over_replicas`], also handing each task the `outputs` entry of its system.
#[allow(clippy::too_many_arguments)]
pub(crate) fn par_over_replicas_with<S: Copy + Send + Sync, T: Copy + Send + Sync, O: Send>(
    spins: &mut [S],
    rngs: &mut [Xoshiro256StarStar],
    temperatures: &[T],
    system_ids: &[usize],
    n_spins: usize,
    sequential: bool,
    outputs: &mut [O],
    body: impl Fn(&mut [S], &mut Xoshiro256StarStar, T, usize, usize, &mut O) + Send + Sync,
) {
    let n_systems = rngs.len();
    assert!(
        system_ids.len() <= n_systems
            && temperatures.len() >= system_ids.len()
            && outputs.len() == n_systems
            && spins.len() >= n_systems * n_spins,
        "replica buffers disagree with the system count"
    );
    let sp = spins.as_mut_ptr() as usize;
    let rp = rngs.as_mut_ptr() as usize;
    let op = outputs.as_mut_ptr() as usize;

    let work = |temp_id: usize| unsafe {
        let system_id = system_ids[temp_id];
        assert!(system_id < n_systems, "system id out of range");
        let spin_slice =
            std::slice::from_raw_parts_mut((sp as *mut S).add(system_id * n_spins), n_spins);
        let rng = &mut *(rp as *mut Xoshiro256StarStar).add(system_id);
        let output = &mut *(op as *mut O).add(system_id);
        let temp = temperatures[temp_id];
        body(spin_slice, rng, temp, temp_id, system_id, output);
    };

    if sequential {
        (0..system_ids.len()).for_each(work);
    } else {
        (0..system_ids.len()).into_par_iter().for_each(work);
    }
}
