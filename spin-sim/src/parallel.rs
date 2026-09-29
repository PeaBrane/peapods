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

/// One replica's mutable state inside a grouped task.
pub(crate) struct ReplicaSlot<'a, S, T, O> {
    pub spins: &'a mut [S],
    pub rng: &'a mut Xoshiro256StarStar,
    pub temp: T,
    pub temp_id: usize,
    pub output: &'a mut O,
}

/// Like [`par_over_replicas_with`], but hands each task up to `group` consecutive
/// temperature slots so kernels can interleave independent systems on one core.
///
/// SAFETY: as for [`par_over_replicas`], `system_ids` must not repeat a system.
#[allow(clippy::too_many_arguments)]
pub(crate) fn par_over_replica_groups<S: Copy + Send + Sync, T: Copy + Send + Sync, O: Send>(
    spins: &mut [S],
    rngs: &mut [Xoshiro256StarStar],
    temperatures: &[T],
    system_ids: &[usize],
    n_spins: usize,
    sequential: bool,
    group: usize,
    outputs: &mut [O],
    body: impl Fn(&mut [ReplicaSlot<'_, S, T, O>]) + Send + Sync,
) {
    let n_systems = rngs.len();
    assert!(
        group > 0
            && system_ids.len() <= n_systems
            && temperatures.len() >= system_ids.len()
            && outputs.len() == n_systems
            && spins.len() >= n_systems * n_spins,
        "replica buffers disagree with the system count"
    );
    let sp = spins.as_mut_ptr() as usize;
    let rp = rngs.as_mut_ptr() as usize;
    let op = outputs.as_mut_ptr() as usize;
    let n_slots = system_ids.len();

    let work = |chunk: usize| unsafe {
        let start = chunk * group;
        let end = (start + group).min(n_slots);
        let mut slots = Vec::with_capacity(end - start);
        for temp_id in start..end {
            let system_id = system_ids[temp_id];
            assert!(system_id < n_systems, "system id out of range");
            slots.push(ReplicaSlot {
                spins: std::slice::from_raw_parts_mut(
                    (sp as *mut S).add(system_id * n_spins),
                    n_spins,
                ),
                rng: &mut *(rp as *mut Xoshiro256StarStar).add(system_id),
                temp: temperatures[temp_id],
                temp_id,
                output: &mut *(op as *mut O).add(system_id),
            });
        }
        body(&mut slots);
    };

    let n_chunks = n_slots.div_ceil(group);
    if sequential {
        (0..n_chunks).for_each(work);
    } else {
        (0..n_chunks).into_par_iter().for_each(work);
    }
}
