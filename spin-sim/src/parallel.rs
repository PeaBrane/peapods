use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;
use std::marker::PhantomData;

/// Runs `work` for every task index, on the calling thread when `sequential` (no
/// rayon overhead, best when outer-level parallelism over disorder realizations
/// already saturates all physical cores).
pub(crate) fn for_each_task(n_tasks: usize, sequential: bool, work: impl Fn(usize) + Send + Sync) {
    if sequential {
        (0..n_tasks).for_each(work);
        return;
    }
    (0..n_tasks).into_par_iter().for_each(work);
}

/// An optional output slice whose elements are written by disjoint parallel tasks.
pub(crate) struct Slots<'a, T> {
    ptr: *mut T,
    len: usize,
    _slice: PhantomData<&'a mut [T]>,
}

// SAFETY: elements are reached only through `get` and `at`, whose callers guarantee
// that no two tasks name the same slot.
unsafe impl<T: Send> Sync for Slots<'_, T> {}

impl<'a, T> Slots<'a, T> {
    pub(crate) fn new(out: Option<&'a mut [T]>) -> Self {
        let (ptr, len) = out.map_or((std::ptr::null_mut(), 0), |out| {
            (out.as_mut_ptr(), out.len())
        });
        Self {
            ptr,
            len,
            _slice: PhantomData,
        }
    }

    pub(crate) fn is_some(&self) -> bool {
        !self.ptr.is_null()
    }

    /// The element at `slot`, or `None` when the output was not requested.
    ///
    /// # Safety
    /// No other live reference may name the same slot.
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn get(&self, slot: usize) -> Option<&mut T> {
        if self.ptr.is_null() {
            return None;
        }
        assert!(slot < self.len, "output slot out of range");
        Some(&mut *self.ptr.add(slot))
    }

    /// The element at `slot` of a slice that is always present.
    ///
    /// # Safety
    /// No other live reference may name the same slot.
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn at(&self, slot: usize) -> &mut T {
        self.get(slot).expect("slot storage is always present")
    }
}

/// One replica's mutable state inside a task.
pub(crate) struct ReplicaSlot<'a, S, T, O> {
    pub spins: &'a mut [S],
    pub rng: &'a mut Xoshiro256StarStar,
    pub temp: T,
    pub temp_id: usize,
    pub system_id: usize,
    pub output: &'a mut O,
}

/// Dispatch a per-replica closure over replicas, optionally in parallel.
///
/// Each replica gets a mutable spin slice and its own RNG. The closure receives
/// `(spin_slice, rng, temp, temp_id, system_id)`.
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
    par_over_replica_groups(
        spins,
        rngs,
        temperatures,
        system_ids,
        n_spins,
        sequential,
        1,
        &mut outputs,
        |slots| {
            let slot = &mut slots[0];
            body(
                slot.spins,
                slot.rng,
                slot.temp,
                slot.temp_id,
                slot.system_id,
            );
        },
    );
}

/// Hands each task up to `group` consecutive temperature slots, so kernels can
/// interleave independent systems on one core, with the `outputs` entry of each
/// system.
///
/// Every system's spins, generator and output are taken exactly once, so a repeated
/// or out-of-range entry of `system_ids` panics rather than aliasing.
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
            && n_spins > 0
            && system_ids.len() <= n_systems
            && temperatures.len() >= system_ids.len()
            && outputs.len() == n_systems
            && spins.len() >= n_systems * n_spins,
        "replica buffers disagree with the system count"
    );
    let mut spin_blocks: Vec<_> = spins
        .chunks_exact_mut(n_spins)
        .take(n_systems)
        .map(Some)
        .collect();
    let mut rngs: Vec<_> = rngs.iter_mut().map(Some).collect();
    let mut outputs: Vec<_> = outputs.iter_mut().map(Some).collect();
    let mut slots: Vec<_> = system_ids
        .iter()
        .enumerate()
        .map(|(temp_id, &system_id)| {
            assert!(system_id < n_systems, "system id out of range");
            let repeated = "system id repeated";
            ReplicaSlot {
                spins: spin_blocks[system_id].take().expect(repeated),
                rng: rngs[system_id].take().expect(repeated),
                temp: temperatures[temp_id],
                temp_id,
                system_id,
                output: outputs[system_id].take().expect(repeated),
            }
        })
        .collect();

    if sequential {
        slots.chunks_mut(group).for_each(body);
        return;
    }
    slots.par_chunks_mut(group).for_each(body);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    #[should_panic(expected = "system id repeated")]
    fn repeated_system_ids_panic_instead_of_aliasing() {
        let mut spins = vec![1i8; 8];
        let mut rngs = vec![Xoshiro256StarStar::seed_from_u64(0); 2];
        par_over_replicas(
            &mut spins,
            &mut rngs,
            &[1.0f32, 2.0],
            &[0, 0],
            4,
            true,
            |_, _, _, _, _| {},
        );
    }
}
