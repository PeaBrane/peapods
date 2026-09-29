use super::utils::{
    dfs_cluster, find_seed, top4_sizes, uf_bonds, uf_bonds_extend, uf_bonds_with,
    uf_flatten_counts, uf_histogram, BondMetrics, BondSampler, GraphObservationSlot, PooledUf,
};
use crate::config::{ClusterAction, ClusterMode, OverlapClusterBuildMode};
use crate::geometry::Lattice;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;

struct GroupTasks {
    systems: Vec<usize>,
    n_replicas: usize,
    n_groups: usize,
    group_size: usize,
}

impl GroupTasks {
    #[inline]
    fn len(&self) -> usize {
        self.systems.len() / self.n_replicas * self.n_groups
    }

    #[inline]
    fn group(&self, task_idx: usize) -> (usize, usize, &[usize]) {
        let t = task_idx / self.n_groups;
        let g = task_idx % self.n_groups;
        let start = t * self.n_replicas + g * self.group_size;
        (t, g, &self.systems[start..start + self.group_size])
    }
}

/// Build shuffled per-temperature group assignments.
fn build_tasks(
    system_ids: &[usize],
    n_replicas: usize,
    n_temps: usize,
    group_size: usize,
    rngs: &mut [Xoshiro256StarStar],
    n_pairs: usize,
) -> GroupTasks {
    let n_groups = n_replicas / group_size;
    let mut systems = Vec::with_capacity(n_temps * n_replicas);
    for t in 0..n_temps {
        let start = systems.len();
        systems.extend((0..n_replicas).map(|k| system_ids[k * n_temps + t]));
        systems[start..].shuffle(&mut rngs[t * n_pairs]);
    }
    GroupTasks {
        systems,
        n_replicas,
        n_groups,
        group_size,
    }
}

/// Top-level overlap cluster update dispatcher.
///
/// Selects the appropriate per-mode function based on `mode`, then runs it
/// over all temperature-group tasks in parallel (or sequentially).
#[cfg_attr(feature = "profile", inline(never))]
#[allow(clippy::too_many_arguments)]
pub fn overlap_update(
    lattice: &Lattice,
    spins: &mut [i8],
    couplings: &[f32],
    temperatures: &[f32],
    system_ids: &[usize],
    n_replicas: usize,
    n_temps: usize,
    rngs: &mut [Xoshiro256StarStar],
    mode: &OverlapClusterBuildMode,
    cluster_mode: ClusterMode,
    action: ClusterAction,
    csd_out: Option<&mut [Vec<u64>]>,
    top4_out: Option<&mut [[u32; 4]]>,
    observation_out: Option<&mut [GraphObservationSlot]>,
    sequential: bool,
    snap_out: Option<&mut [Vec<u32>]>,
    blue_snap_out: Option<&mut [Vec<u32>]>,
    spin_snap_out: Option<&mut [Vec<[Vec<i8>; 2]>]>,
    sid_snap_out: Option<&mut [Vec<[usize; 2]>]>,
) {
    match mode {
        OverlapClusterBuildMode::Houdayer(group_size) => houdayer_step(
            lattice,
            spins,
            system_ids,
            n_replicas,
            n_temps,
            rngs,
            *group_size,
            cluster_mode,
            action,
            csd_out,
            top4_out,
            observation_out,
            sequential,
            snap_out,
            spin_snap_out,
            sid_snap_out,
        ),
        OverlapClusterBuildMode::Jorg => jorg_step(
            lattice,
            spins,
            couplings,
            temperatures,
            system_ids,
            n_replicas,
            n_temps,
            rngs,
            cluster_mode,
            action,
            csd_out,
            top4_out,
            observation_out,
            sequential,
            snap_out,
            spin_snap_out,
            sid_snap_out,
        ),
        OverlapClusterBuildMode::Cmr => cmr_step(
            lattice,
            spins,
            couplings,
            temperatures,
            system_ids,
            n_replicas,
            n_temps,
            rngs,
            cluster_mode,
            action,
            csd_out,
            top4_out,
            observation_out,
            sequential,
            snap_out,
            blue_snap_out,
            spin_snap_out,
            sid_snap_out,
        ),
    }
}

/// Houdayer-N isoenergetic overlap cluster update.
///
/// For each group of N replicas at a given temperature:
/// 1. Active sites: spin sum across all N replicas = 0 (balanced)
/// 2. Deterministic bonds (p=1) between pairs of active sites
/// 3. Flip all N replicas on cluster sites
#[allow(clippy::too_many_arguments)]
fn houdayer_step(
    lattice: &Lattice,
    spins: &mut [i8],
    system_ids: &[usize],
    n_replicas: usize,
    n_temps: usize,
    rngs: &mut [Xoshiro256StarStar],
    group_size: usize,
    cluster_mode: ClusterMode,
    action: ClusterAction,
    mut csd_out: Option<&mut [Vec<u64>]>,
    mut top4_out: Option<&mut [[u32; 4]]>,
    mut observation_out: Option<&mut [GraphObservationSlot]>,
    sequential: bool,
    mut snap_out: Option<&mut [Vec<u32>]>,
    mut spin_snap_out: Option<&mut [Vec<[Vec<i8>; 2]>]>,
    mut sid_snap_out: Option<&mut [Vec<[usize; 2]>]>,
) {
    let n_spins = lattice.n_spins;
    let n_pairs = n_replicas / 2;
    let wolff = cluster_mode == ClusterMode::Wolff;

    let tasks = build_tasks(system_ids, n_replicas, n_temps, group_size, rngs, n_pairs);

    let sp = spins.as_mut_ptr() as usize;
    let rp = rngs.as_mut_ptr() as usize;
    let use_uf = action == ClusterAction::Observe
        || !wolff
        || csd_out.is_some()
        || top4_out.is_some()
        || snap_out.is_some();

    let cp = csd_out
        .as_mut()
        .map(|s| s.as_mut_ptr() as usize)
        .unwrap_or(0);
    let has_csd = csd_out.is_some();
    let tp = top4_out
        .as_mut()
        .map(|s| s.as_mut_ptr() as usize)
        .unwrap_or(0);
    let has_top4 = top4_out.is_some();
    let op = observation_out
        .as_mut()
        .map(|s| s.as_mut_ptr() as usize)
        .unwrap_or(0);
    let has_observation = observation_out.is_some();
    let snp = snap_out
        .as_mut()
        .map(|s| s.as_mut_ptr() as usize)
        .unwrap_or(0);
    let has_snap = snap_out.is_some();
    let spp = spin_snap_out
        .as_mut()
        .map(|s| s.as_mut_ptr() as usize)
        .unwrap_or(0);
    let sidp = sid_snap_out
        .as_mut()
        .map(|s| s.as_mut_ptr() as usize)
        .unwrap_or(0);

    let work = |task_idx: usize| unsafe {
        let (t, g, systems) = tasks.group(task_idx);
        let rng = &mut *(rp as *mut Xoshiro256StarStar).add(t * n_pairs + g);
        let sp_ptr = sp as *mut i8;
        let slot = t * n_pairs + g;

        // Only the first group per temperature is published as a snapshot.
        if spp != 0 && sidp != 0 && g == 0 && systems.len() >= 2 {
            let base_a = systems[0] * n_spins;
            let base_b = systems[1] * n_spins;
            let spin_slot = &mut *(spp as *mut Vec<[Vec<i8>; 2]>).add(slot);
            spin_slot.push([
                std::slice::from_raw_parts(sp_ptr.add(base_a), n_spins).to_vec(),
                std::slice::from_raw_parts(sp_ptr.add(base_b), n_spins).to_vec(),
            ]);
            let sid_slot = &mut *(sidp as *mut Vec<[usize; 2]>).add(slot);
            sid_slot.push([systems[0], systems[1]]);
        }

        let is_active = |i: usize| -> bool {
            let mut sum: i32 = 0;
            for &system in systems {
                sum += *sp_ptr.add(system * n_spins + i) as i32;
            }
            sum == 0
        };

        if use_uf {
            // The O(N) graph pass would otherwise re-sum the group for both endpoints
            // of every bond.
            let active: Vec<bool> = (0..n_spins).map(is_active).collect();
            let is_active = |i: usize| active[i];
            let mut should_bond = |i: usize, d: usize| {
                let j = lattice.neighbor_fwd(i, d);
                is_active(i) && is_active(j)
            };
            let mut metrics = has_observation.then(|| BondMetrics::new(lattice));
            let mut uf = if let Some(ref mut metrics) = metrics {
                uf_bonds_with(lattice, &mut should_bond, |site, dim| {
                    metrics.record_bond(lattice, site, dim);
                })
            } else {
                uf_bonds(lattice, &mut should_bond)
            };

            // Statistics and snapshots describe the graph, so record them before the
            // Wolff seed search, which may find no active site.
            let counts = uf_flatten_counts(&mut uf.parent);
            if has_csd {
                let csd_slot = &mut *(cp as *mut Vec<u64>).add(slot);
                uf_histogram(&counts, csd_slot.as_mut_slice());
            }
            if has_top4 {
                let out = &mut *(tp as *mut [u32; 4]).add(slot);
                *out = top4_sizes(&counts);
            }
            if has_snap {
                let snap_slot = &mut *(snp as *mut Vec<u32>).add(slot);
                snap_slot.clear();
                snap_slot.extend_from_slice(&uf.parent[..n_spins]);
            }
            if let Some(metrics) = metrics {
                let observation_slot = &mut *(op as *mut GraphObservationSlot).add(slot);
                *observation_slot = metrics.finish(&counts);
            }
            if action == ClusterAction::Observe {
                return;
            }

            if wolff {
                let Some(seed) = find_seed(n_spins, rng, &is_active) else {
                    return;
                };
                let seed_root = uf.parent[seed];
                for (i, &p) in uf.parent.iter().enumerate().take(n_spins) {
                    if p == seed_root {
                        for &system in systems {
                            *sp_ptr.add(system * n_spins + i) *= -1;
                        }
                    }
                }
            } else {
                let storage = &mut *uf;
                storage.rank.fill(u8::MAX);
                for &p in storage.parent.iter().take(n_spins) {
                    let root = p as usize;
                    if counts[root] > 1 && storage.rank[root] == u8::MAX {
                        storage.rank[root] = u8::from(rng.gen::<f32>() < 0.5);
                    }
                }
                for (i, &p) in storage.parent.iter().enumerate().take(n_spins) {
                    if storage.rank[p as usize] == 1 {
                        for &system in systems {
                            *sp_ptr.add(system * n_spins + i) *= -1;
                        }
                    }
                }
            }
        } else {
            let Some(seed) = find_seed(n_spins, rng, &is_active) else {
                return;
            };
            let mut in_cluster = vec![false; n_spins];
            let mut stack = Vec::with_capacity(n_spins);
            dfs_cluster(
                lattice,
                seed,
                &mut in_cluster,
                &mut stack,
                |site, nb, _d, _fwd| is_active(site) && is_active(nb),
            );
            for (i, &in_c) in in_cluster.iter().enumerate() {
                if in_c {
                    for &system in systems {
                        *sp_ptr.add(system * n_spins + i) *= -1;
                    }
                }
            }
        }
    };

    if sequential {
        (0..tasks.len()).for_each(work);
    } else {
        (0..tasks.len()).into_par_iter().for_each(work);
    }
}

/// Jörg stochastic overlap cluster update.
///
/// For each pair of replicas at a given temperature:
/// 1. Active sites: σ_i ≠ τ_i (negative overlap)
/// 2. Stochastic FK bonds on active sites: p = 1 - exp(-4 J σ_i σ_j / T)
/// 3. Flip both replicas on cluster sites
#[allow(clippy::too_many_arguments)]
fn jorg_step(
    lattice: &Lattice,
    spins: &mut [i8],
    couplings: &[f32],
    temperatures: &[f32],
    system_ids: &[usize],
    n_replicas: usize,
    n_temps: usize,
    rngs: &mut [Xoshiro256StarStar],
    cluster_mode: ClusterMode,
    action: ClusterAction,
    mut csd_out: Option<&mut [Vec<u64>]>,
    mut top4_out: Option<&mut [[u32; 4]]>,
    mut observation_out: Option<&mut [GraphObservationSlot]>,
    sequential: bool,
    mut snap_out: Option<&mut [Vec<u32>]>,
    mut spin_snap_out: Option<&mut [Vec<[Vec<i8>; 2]>]>,
    mut sid_snap_out: Option<&mut [Vec<[usize; 2]>]>,
) {
    let n_spins = lattice.n_spins;
    let n_neighbors = lattice.n_neighbors;
    let n_pairs = n_replicas / 2;
    let wolff = cluster_mode == ClusterMode::Wolff;

    let tasks = build_tasks(system_ids, n_replicas, n_temps, 2, rngs, n_pairs);

    let sp = spins.as_mut_ptr() as usize;
    let rp = rngs.as_mut_ptr() as usize;
    let use_uf = action == ClusterAction::Observe
        || !wolff
        || csd_out.is_some()
        || top4_out.is_some()
        || snap_out.is_some();

    let cp = csd_out
        .as_mut()
        .map(|s| s.as_mut_ptr() as usize)
        .unwrap_or(0);
    let has_csd = csd_out.is_some();
    let tp = top4_out
        .as_mut()
        .map(|s| s.as_mut_ptr() as usize)
        .unwrap_or(0);
    let has_top4 = top4_out.is_some();
    let op = observation_out
        .as_mut()
        .map(|s| s.as_mut_ptr() as usize)
        .unwrap_or(0);
    let has_observation = observation_out.is_some();
    let snp = snap_out
        .as_mut()
        .map(|s| s.as_mut_ptr() as usize)
        .unwrap_or(0);
    let has_snap = snap_out.is_some();
    let spp = spin_snap_out
        .as_mut()
        .map(|s| s.as_mut_ptr() as usize)
        .unwrap_or(0);
    let sidp = sid_snap_out
        .as_mut()
        .map(|s| s.as_mut_ptr() as usize)
        .unwrap_or(0);

    let work = |task_idx: usize| unsafe {
        let (t, g, systems) = tasks.group(task_idx);
        let rng = &mut *(rp as *mut Xoshiro256StarStar).add(t * n_pairs + g);
        let jorg_bond = BondSampler::new(4.0 / temperatures[t]);
        let base_a = systems[0] * n_spins;
        let base_b = systems[1] * n_spins;
        let sp_ptr = sp as *mut i8;
        let slot = t * n_pairs + g;

        // Only the first pair per temperature is published as a snapshot.
        if spp != 0 && sidp != 0 && g == 0 {
            let spin_slot = &mut *(spp as *mut Vec<[Vec<i8>; 2]>).add(slot);
            spin_slot.push([
                std::slice::from_raw_parts(sp_ptr.add(base_a), n_spins).to_vec(),
                std::slice::from_raw_parts(sp_ptr.add(base_b), n_spins).to_vec(),
            ]);
            let sid_slot = &mut *(sidp as *mut Vec<[usize; 2]>).add(slot);
            sid_slot.push([systems[0], systems[1]]);
        }

        let is_active = |i: usize| -> bool { *sp_ptr.add(base_a + i) != *sp_ptr.add(base_b + i) };

        if use_uf {
            let active: Vec<bool> = (0..n_spins).map(is_active).collect();
            let is_active = |i: usize| active[i];
            let mut should_bond = |i: usize, d: usize| {
                let j = lattice.neighbor_fwd(i, d);
                if !is_active(i) || !is_active(j) {
                    return false;
                }
                let inter = *sp_ptr.add(base_a + i) as f32
                    * *sp_ptr.add(base_a + j) as f32
                    * couplings[i * n_neighbors + d];
                if inter <= 0.0 {
                    return false;
                }
                jorg_bond.sample(inter, rng)
            };
            let mut metrics = has_observation.then(|| BondMetrics::new(lattice));
            let mut uf = if let Some(ref mut metrics) = metrics {
                uf_bonds_with(lattice, &mut should_bond, |site, dim| {
                    metrics.record_bond(lattice, site, dim);
                })
            } else {
                uf_bonds(lattice, &mut should_bond)
            };

            // Statistics and snapshots describe the graph, so record them before the
            // Wolff seed search, which may find no active site.
            let counts = uf_flatten_counts(&mut uf.parent);
            if has_csd {
                let csd_slot = &mut *(cp as *mut Vec<u64>).add(slot);
                uf_histogram(&counts, csd_slot.as_mut_slice());
            }
            if has_top4 {
                let out = &mut *(tp as *mut [u32; 4]).add(slot);
                *out = top4_sizes(&counts);
            }
            if has_snap {
                let snap_slot = &mut *(snp as *mut Vec<u32>).add(slot);
                snap_slot.clear();
                snap_slot.extend_from_slice(&uf.parent[..n_spins]);
            }
            if let Some(metrics) = metrics {
                let observation_slot = &mut *(op as *mut GraphObservationSlot).add(slot);
                *observation_slot = metrics.finish(&counts);
            }
            if action == ClusterAction::Observe {
                return;
            }

            if wolff {
                let Some(seed) = find_seed(n_spins, rng, &is_active) else {
                    return;
                };
                let seed_root = uf.parent[seed];
                for (i, &p) in uf.parent.iter().enumerate().take(n_spins) {
                    if p == seed_root {
                        *sp_ptr.add(base_a + i) *= -1;
                        *sp_ptr.add(base_b + i) *= -1;
                    }
                }
            } else {
                let storage = &mut *uf;
                storage.rank.fill(u8::MAX);
                for &p in storage.parent.iter().take(n_spins) {
                    let root = p as usize;
                    if counts[root] > 1 && storage.rank[root] == u8::MAX {
                        storage.rank[root] = u8::from(rng.gen::<f32>() < 0.5);
                    }
                }
                for (i, &p) in storage.parent.iter().enumerate().take(n_spins) {
                    if storage.rank[p as usize] == 1 {
                        *sp_ptr.add(base_a + i) *= -1;
                        *sp_ptr.add(base_b + i) *= -1;
                    }
                }
            }
        } else {
            let Some(seed) = find_seed(n_spins, rng, &is_active) else {
                return;
            };
            let mut in_cluster = vec![false; n_spins];
            let mut stack = Vec::with_capacity(n_spins);
            dfs_cluster(
                lattice,
                seed,
                &mut in_cluster,
                &mut stack,
                |site, nb, d, fwd| {
                    if !is_active(nb) {
                        return false;
                    }
                    let coupling = if fwd {
                        couplings[site * n_neighbors + d]
                    } else {
                        couplings[nb * n_neighbors + d]
                    };
                    let inter = *sp_ptr.add(base_a + site) as f32
                        * *sp_ptr.add(base_a + nb) as f32
                        * coupling;
                    if inter <= 0.0 {
                        return false;
                    }
                    jorg_bond.sample(inter, rng)
                },
            );
            for (i, &in_c) in in_cluster.iter().enumerate() {
                if in_c {
                    *sp_ptr.add(base_a + i) *= -1;
                    *sp_ptr.add(base_b + i) *= -1;
                }
            }
        }
    };

    if sequential {
        (0..tasks.len()).for_each(work);
    } else {
        (0..tasks.len()).into_par_iter().for_each(work);
    }
}

/// CMR two-phase overlap cluster update (Machta-Newman-Stein 2007, eqs 10-11).
///
/// Phase 1 — Blue clusters:
/// 1. Blue bond: doubly-satisfied edges (both replicas satisfied) with prob 1-r²
///    where r = exp(-2|J_ij|/T).
/// 2. SW: flip each non-singleton blue cluster (both replicas) with prob 1/2.
///    Wolff: flip seed's blue cluster (both replicas, always).
/// 3. CSD/top4 from blue clusters.
///
/// Phase 2 — Grey clusters (extend blue UF with red bonds):
/// 4. Red bond: singly-satisfied edges (exactly one replica satisfied, evaluated on
///    post-blue-flip spins) with prob 1-r. Blue flips negate both replicas, which
///    swaps which replica is satisfied on a singly-satisfied edge but preserves the
///    singly-satisfied classification. So red bonds can be evaluated on post-blue-flip
///    spins.
/// 5. Grey = Blue ∪ Red (blue ⊂ grey always).
/// 6. SW: flip each non-singleton grey cluster with k ∈ {0,1,2,3}.
///    Wolff: flip seed's grey cluster with k ∈ {1,2,3}.
///
/// Grey clusters are supersets of blue clusters; sites in blue clusters receive both
/// the blue flip and the grey flip. This composition is the correct CMR update.
#[allow(clippy::too_many_arguments)]
unsafe fn build_cmr_blue_graph(
    lattice: &Lattice,
    sp_ptr: *mut i8,
    base_a: usize,
    couplings: &[f32],
    n_neighbors: usize,
    overlap: &[i8],
    blue_bond: BondSampler,
    rng: &mut Xoshiro256StarStar,
    on_bond: impl FnMut(usize, usize),
) -> PooledUf {
    uf_bonds_with(
        lattice,
        |i, d| {
            let j = lattice.neighbor_fwd(i, d);
            // Equal site overlaps make replica b satisfied exactly when a is.
            if overlap[i] != overlap[j] {
                return false;
            }
            let coupling = couplings[i * n_neighbors + d];
            let satisfied =
                *sp_ptr.add(base_a + i) as f32 * *sp_ptr.add(base_a + j) as f32 * coupling > 0.0;
            satisfied && blue_bond.sample(coupling.abs(), rng)
        },
        on_bond,
    )
}

#[allow(clippy::too_many_arguments)]
fn cmr_step(
    lattice: &Lattice,
    spins: &mut [i8],
    couplings: &[f32],
    temperatures: &[f32],
    system_ids: &[usize],
    n_replicas: usize,
    n_temps: usize,
    rngs: &mut [Xoshiro256StarStar],
    cluster_mode: ClusterMode,
    action: ClusterAction,
    mut csd_out: Option<&mut [Vec<u64>]>,
    mut top4_out: Option<&mut [[u32; 4]]>,
    mut observation_out: Option<&mut [GraphObservationSlot]>,
    sequential: bool,
    mut snap_out: Option<&mut [Vec<u32>]>,
    mut blue_snap_out: Option<&mut [Vec<u32>]>,
    mut spin_snap_out: Option<&mut [Vec<[Vec<i8>; 2]>]>,
    mut sid_snap_out: Option<&mut [Vec<[usize; 2]>]>,
) {
    let n_spins = lattice.n_spins;
    let n_neighbors = lattice.n_neighbors;
    let n_pairs = n_replicas / 2;
    let wolff = cluster_mode == ClusterMode::Wolff;

    let tasks = build_tasks(system_ids, n_replicas, n_temps, 2, rngs, n_pairs);

    let sp = spins.as_mut_ptr() as usize;
    let rp = rngs.as_mut_ptr() as usize;
    let use_uf = action == ClusterAction::Observe
        || !wolff
        || csd_out.is_some()
        || top4_out.is_some()
        || snap_out.is_some();

    let cp = csd_out
        .as_mut()
        .map(|s| s.as_mut_ptr() as usize)
        .unwrap_or(0);
    let has_csd = csd_out.is_some();
    let tp = top4_out
        .as_mut()
        .map(|s| s.as_mut_ptr() as usize)
        .unwrap_or(0);
    let has_top4 = top4_out.is_some();
    let op = observation_out
        .as_mut()
        .map(|s| s.as_mut_ptr() as usize)
        .unwrap_or(0);
    let has_observation = observation_out.is_some();
    let snp = snap_out
        .as_mut()
        .map(|s| s.as_mut_ptr() as usize)
        .unwrap_or(0);
    let has_snap = snap_out.is_some();
    let bsnp = blue_snap_out
        .as_mut()
        .map(|s| s.as_mut_ptr() as usize)
        .unwrap_or(0);
    let has_blue_snap = blue_snap_out.is_some();
    let spp = spin_snap_out
        .as_mut()
        .map(|s| s.as_mut_ptr() as usize)
        .unwrap_or(0);
    let sidp = sid_snap_out
        .as_mut()
        .map(|s| s.as_mut_ptr() as usize)
        .unwrap_or(0);

    let work = |task_idx: usize| unsafe {
        let (t, g, systems) = tasks.group(task_idx);
        let rng = &mut *(rp as *mut Xoshiro256StarStar).add(t * n_pairs + g);
        // Blue bonds use 1 - r^2 and red bonds 1 - r, with r = exp(-2|J|/T).
        let blue_bond = BondSampler::new(4.0 / temperatures[t]);
        let red_bond = BondSampler::new(2.0 / temperatures[t]);
        let base_a = systems[0] * n_spins;
        let base_b = systems[1] * n_spins;
        let sp_ptr = sp as *mut i8;
        let slot = t * n_pairs + g;

        // Only the first pair per temperature is published as a snapshot.
        if spp != 0 && sidp != 0 && g == 0 {
            let spin_slot = &mut *(spp as *mut Vec<[Vec<i8>; 2]>).add(slot);
            spin_slot.push([
                std::slice::from_raw_parts(sp_ptr.add(base_a), n_spins).to_vec(),
                std::slice::from_raw_parts(sp_ptr.add(base_b), n_spins).to_vec(),
            ]);
            let sid_slot = &mut *(sidp as *mut Vec<[usize; 2]>).add(slot);
            sid_slot.push([systems[0], systems[1]]);
        }

        if use_uf {
            let seed = if wolff {
                rng.gen_range(0..n_spins)
            } else {
                0 // unused
            };

            // Site overlaps q_i = a_i b_i classify bonds: an edge is singly satisfied
            // iff q_i != q_j (J != 0), and blue flips negate both replicas, leaving q.
            let overlap: Vec<i8> = (0..n_spins)
                .map(|i| *sp_ptr.add(base_a + i) * *sp_ptr.add(base_b + i))
                .collect();

            // === Phase 1: Blue clusters ===
            let mut metrics = has_observation.then(|| BondMetrics::new(lattice));
            let mut uf = if let Some(ref mut metrics) = metrics {
                build_cmr_blue_graph(
                    lattice,
                    sp_ptr,
                    base_a,
                    couplings,
                    n_neighbors,
                    &overlap,
                    blue_bond,
                    rng,
                    |site, dim| metrics.record_bond(lattice, site, dim),
                )
            } else {
                build_cmr_blue_graph(
                    lattice,
                    sp_ptr,
                    base_a,
                    couplings,
                    n_neighbors,
                    &overlap,
                    blue_bond,
                    rng,
                    |_site, _dim| {},
                )
            };

            let counts = uf_flatten_counts(&mut uf.parent);
            if has_csd {
                let csd_slot = &mut *(cp as *mut Vec<u64>).add(slot);
                uf_histogram(&counts, csd_slot.as_mut_slice());
            }
            if has_top4 {
                let out = &mut *(tp as *mut [u32; 4]).add(slot);
                *out = top4_sizes(&counts);
            }
            if has_blue_snap {
                let blue_slot = &mut *(bsnp as *mut Vec<u32>).add(slot);
                blue_slot.clear();
                blue_slot.extend_from_slice(&uf.parent[..n_spins]);
            }
            if let Some(metrics) = metrics {
                let observation_slot = &mut *(op as *mut GraphObservationSlot).add(slot);
                *observation_slot = metrics.finish(&counts);
            }
            if action == ClusterAction::Observe {
                return;
            }

            // Flip blue clusters (both replicas jointly)
            if wolff {
                let seed_root = uf.parent[seed] as usize;
                for (i, &p) in uf.parent.iter().enumerate().take(n_spins) {
                    if p as usize == seed_root {
                        *sp_ptr.add(base_a + i) *= -1;
                        *sp_ptr.add(base_b + i) *= -1;
                    }
                }
            } else {
                // Keep UF ranks intact until red bonds extend the blue clusters.
                let mut do_flip = vec![u8::MAX; n_spins];
                for &p in uf.parent.iter().take(n_spins) {
                    let root = p as usize;
                    if counts[root] > 1 && do_flip[root] == u8::MAX {
                        do_flip[root] = u8::from(rng.gen::<f32>() < 0.5);
                    }
                }
                for (i, &p) in uf.parent.iter().enumerate().take(n_spins) {
                    if do_flip[p as usize] == 1 {
                        *sp_ptr.add(base_a + i) *= -1;
                        *sp_ptr.add(base_b + i) *= -1;
                    }
                }
            }

            // === Phase 2: Grey clusters (extend blue UF with red bonds) ===
            // Red bond: singly-satisfied edges on post-flip spins, prob 1 - r.
            // No clone needed: blue flips preserve the singly-satisfied classification.
            // Return blue counts so grey counts can reuse the pooled allocation.
            drop(counts);
            let storage = &mut *uf;
            uf_bonds_extend(&mut storage.parent, &mut storage.rank, lattice, |i, d| {
                let j = lattice.neighbor_fwd(i, d);
                let coupling = couplings[i * n_neighbors + d];
                overlap[i] != overlap[j] && coupling != 0.0 && red_bond.sample(coupling.abs(), rng)
            });

            let grey_counts = uf_flatten_counts(&mut uf.parent);

            if has_snap {
                let snap_slot = &mut *(snp as *mut Vec<u32>).add(slot);
                snap_slot.clear();
                snap_slot.extend_from_slice(&uf.parent[..n_spins]);
            }

            // Flip grey clusters (each replica independently)
            if wolff {
                let seed_root = uf.parent[seed] as usize;
                let k: u8 = rng.gen_range(1..=3);
                let flip_a = k & 1 != 0;
                let flip_b = k & 2 != 0;
                for (i, &p) in uf.parent.iter().enumerate().take(n_spins) {
                    if p as usize == seed_root {
                        if flip_a {
                            *sp_ptr.add(base_a + i) *= -1;
                        }
                        if flip_b {
                            *sp_ptr.add(base_b + i) *= -1;
                        }
                    }
                }
            } else {
                let storage = &mut *uf;
                storage.rank.fill(u8::MAX);
                for &p in storage.parent.iter().take(n_spins) {
                    let root = p as usize;
                    if grey_counts[root] > 1 && storage.rank[root] == u8::MAX {
                        storage.rank[root] = rng.gen_range(0..=3);
                    }
                }
                for (i, &p) in storage.parent.iter().enumerate().take(n_spins) {
                    let k = storage.rank[p as usize];
                    if k == 0 || k == u8::MAX {
                        continue;
                    }
                    if k & 1 != 0 {
                        *sp_ptr.add(base_a + i) *= -1;
                    }
                    if k & 2 != 0 {
                        *sp_ptr.add(base_b + i) *= -1;
                    }
                }
            }
        } else {
            // Pure Wolff BFS two-phase (no stats needed)
            let seed = rng.gen_range(0..n_spins);

            // === Phase 1: Blue cluster BFS ===
            let mut in_cluster = vec![false; n_spins];
            let mut stack = Vec::with_capacity(n_spins);
            dfs_cluster(
                lattice,
                seed,
                &mut in_cluster,
                &mut stack,
                |site, nb, d, fwd| {
                    let coupling_idx = if fwd {
                        site * n_neighbors + d
                    } else {
                        nb * n_neighbors + d
                    };
                    let coupling = couplings[coupling_idx];

                    let a_sat = *sp_ptr.add(base_a + site) as f32
                        * *sp_ptr.add(base_a + nb) as f32
                        * coupling
                        > 0.0;
                    let b_sat = *sp_ptr.add(base_b + site) as f32
                        * *sp_ptr.add(base_b + nb) as f32
                        * coupling
                        > 0.0;

                    if !a_sat || !b_sat {
                        return false;
                    }
                    blue_bond.sample(coupling.abs(), rng)
                },
            );

            // Flip blue cluster (both replicas always)
            for (i, &in_c) in in_cluster.iter().enumerate() {
                if in_c {
                    *sp_ptr.add(base_a + i) *= -1;
                    *sp_ptr.add(base_b + i) *= -1;
                }
            }

            // === Phase 2: Grey cluster (extend from the flipped blue cluster) ===
            // Edges at the blue cluster had their blue bond decided in phase 1, so only
            // red bonds can still attach there. Sites reached through red bonds sample
            // both bond types afresh: blue on doubly satisfied edges (prob 1-r²) and
            // red on singly satisfied ones (prob 1-r). The grey cluster must absorb
            // both, or the grey flip breaks detailed balance.
            let in_blue = in_cluster.clone();
            stack.extend((0..n_spins).filter(|&i| in_blue[i]));

            let mut grey_bond = |site: usize, nb: usize, coupling: f32| -> bool {
                let a_sat =
                    *sp_ptr.add(base_a + site) as f32 * *sp_ptr.add(base_a + nb) as f32 * coupling
                        > 0.0;
                let b_sat =
                    *sp_ptr.add(base_b + site) as f32 * *sp_ptr.add(base_b + nb) as f32 * coupling
                        > 0.0;
                if a_sat != b_sat {
                    return red_bond.sample(coupling.abs(), rng);
                }
                if !a_sat || in_blue[site] {
                    return false;
                }
                blue_bond.sample(coupling.abs(), rng)
            };

            while let Some(site) = stack.pop() {
                for d in 0..lattice.n_neighbors {
                    let fwd = lattice.neighbor_fwd(site, d);
                    if !in_cluster[fwd] && grey_bond(site, fwd, couplings[site * n_neighbors + d]) {
                        in_cluster[fwd] = true;
                        stack.push(fwd);
                    }

                    let bwd = lattice.neighbor_bwd(site, d);
                    if !in_cluster[bwd] && grey_bond(site, bwd, couplings[bwd * n_neighbors + d]) {
                        in_cluster[bwd] = true;
                        stack.push(bwd);
                    }
                }
            }

            // Flip grey cluster with k ∈ {1,2,3}
            let k: u8 = rng.gen_range(1..=3);
            let flip_a = k & 1 != 0;
            let flip_b = k & 2 != 0;
            for (i, &in_c) in in_cluster.iter().enumerate() {
                if in_c {
                    if flip_a {
                        *sp_ptr.add(base_a + i) *= -1;
                    }
                    if flip_b {
                        *sp_ptr.add(base_b + i) *= -1;
                    }
                }
            }
        }
    };

    if sequential {
        (0..tasks.len()).for_each(work);
    } else {
        (0..tasks.len()).into_par_iter().for_each(work);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    // At this temperature one step of a red-only grey closure, or of one that re-samples
    // blue bonds at the flipped blue cluster, misses the exact energy by more than 15
    // standard errors on the 2x2 torus.
    const TEMP: f32 = 3.0;
    const TRIALS: usize = 40_000;

    fn couplings(lattice: &Lattice) -> Vec<f32> {
        const VALUES: [f32; 8] = [0.9, -1.3, 0.4, -0.7, 1.1, -0.2, 0.6, -1.5];
        (0..lattice.n_spins * lattice.n_neighbors)
            .map(|k| VALUES[k % VALUES.len()] * (1.0 + 0.1 * (k / VALUES.len()) as f32))
            .collect()
    }

    fn bond_sum(lattice: &Lattice, couplings: &[f32], s: &[i8]) -> f64 {
        let mut total = 0.0;
        for i in 0..lattice.n_spins {
            for d in 0..lattice.n_neighbors {
                let j = lattice.neighbor_fwd(i, d);
                total +=
                    (couplings[i * lattice.n_neighbors + d] * s[i] as f32 * s[j] as f32) as f64;
            }
        }
        total
    }

    fn state(n: usize, bits: usize) -> Vec<i8> {
        (0..n)
            .map(|i| if (bits >> i) & 1 == 1 { 1 } else { -1 })
            .collect()
    }

    /// Mean and variance of (e_a + e_b, q^2) under two independent Boltzmann replicas.
    fn exact_moments(lattice: &Lattice, couplings: &[f32], p: &[f64]) -> [(f64, f64); 2] {
        let n = lattice.n_spins;
        let states: Vec<Vec<i8>> = (0..p.len()).map(|b| state(n, b)).collect();
        let e: Vec<f64> = states
            .iter()
            .map(|s| bond_sum(lattice, couplings, s) / n as f64)
            .collect();
        let (mut m, mut m2) = ([0.0; 2], [0.0; 2]);
        for (x, sx) in states.iter().enumerate() {
            for (y, sy) in states.iter().enumerate() {
                let w = p[x] * p[y];
                let q = sx.iter().zip(sy).map(|(a, b)| (a * b) as f64).sum::<f64>() / n as f64;
                for (k, obs) in [e[x] + e[y], q * q].into_iter().enumerate() {
                    m[k] += w * obs;
                    m2[k] += w * obs * obs;
                }
            }
        }
        [(m[0], m2[0] - m[0] * m[0]), (m[1], m2[1] - m[1] * m[1])]
    }

    /// Draws replica pairs exactly from the Boltzmann law, applies one overlap move and
    /// checks that energy and q^2 keep their Boltzmann means. A stationary move must
    /// preserve every expectation after a single step.
    fn assert_preserves_boltzmann(
        shape: Vec<usize>,
        mode: OverlapClusterBuildMode,
        cluster_mode: ClusterMode,
        with_stats: bool,
    ) {
        let lattice = Lattice::new(shape.clone());
        let n = lattice.n_spins;
        let couplings = couplings(&lattice);
        let weights: Vec<f64> = (0..1usize << n)
            .map(|b| (bond_sum(&lattice, &couplings, &state(n, b)) / TEMP as f64).exp())
            .collect();
        let z: f64 = weights.iter().sum();
        let p: Vec<f64> = weights.iter().map(|w| w / z).collect();
        let cdf: Vec<f64> = p
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x;
                Some(*acc)
            })
            .collect();
        let exact = exact_moments(&lattice, &couplings, &p);

        let mut draw_rng = Xoshiro256StarStar::seed_from_u64(7);
        let mut rngs = vec![Xoshiro256StarStar::seed_from_u64(11)];
        let mut top4 = vec![[0u32; 4]; 1];
        let mut spins = vec![0i8; 2 * n];
        let mut sums = [0.0f64; 2];
        for _ in 0..TRIALS {
            for replica in 0..2 {
                let u: f64 = draw_rng.gen();
                let bits = cdf.partition_point(|&c| c < u).min(p.len() - 1);
                spins[replica * n..(replica + 1) * n].copy_from_slice(&state(n, bits));
            }
            overlap_update(
                &lattice,
                &mut spins,
                &couplings,
                &[TEMP],
                &[0, 1],
                2,
                1,
                &mut rngs,
                &mode,
                cluster_mode,
                ClusterAction::Update,
                None,
                with_stats.then_some(top4.as_mut_slice()),
                None,
                true,
                None,
                None,
                None,
                None,
            );
            let (a, b) = spins.split_at(n);
            let q = a.iter().zip(b).map(|(x, y)| (x * y) as f64).sum::<f64>() / n as f64;
            sums[0] +=
                (bond_sum(&lattice, &couplings, a) + bond_sum(&lattice, &couplings, b)) / n as f64;
            sums[1] += q * q;
        }
        for (k, name) in ["energy", "q^2"].iter().enumerate() {
            let (mean, var) = exact[k];
            let observed = sums[k] / TRIALS as f64;
            let se = (var / TRIALS as f64).sqrt();
            assert!(
                (observed - mean).abs() < 5.0 * se,
                "{mode:?} {cluster_mode:?} stats={with_stats} {shape:?}: {name} {observed:.5} vs exact {mean:.5} (se {se:.5})"
            );
        }
    }

    /// With identical replicas no site is active, so the Wolff seed search fails; the
    /// all-singleton cluster statistics must still be written for every build mode.
    #[test]
    fn wolff_records_statistics_without_active_sites() {
        let lattice = Lattice::new(vec![4, 4]);
        let n = lattice.n_spins;
        let couplings = vec![1.0; n * lattice.n_neighbors];
        for mode in [
            OverlapClusterBuildMode::Houdayer(2),
            OverlapClusterBuildMode::Jorg,
        ] {
            let mut spins = vec![1i8; 2 * n];
            let mut rngs = vec![Xoshiro256StarStar::seed_from_u64(1)];
            let mut csd = vec![vec![0u64; n + 1]];
            let mut top4 = vec![[0u32; 4]];
            let mut snapshot = vec![Vec::new()];
            overlap_update(
                &lattice,
                &mut spins,
                &couplings,
                &[1.0],
                &[0, 1],
                2,
                1,
                &mut rngs,
                &mode,
                ClusterMode::Wolff,
                ClusterAction::Update,
                Some(&mut csd),
                Some(&mut top4),
                None,
                true,
                Some(&mut snapshot),
                None,
                None,
                None,
            );
            assert_eq!(csd[0][1], n as u64, "{mode:?}");
            assert_eq!(top4[0], [1; 4], "{mode:?}");
            assert_eq!(snapshot[0], (0..n as u32).collect::<Vec<_>>(), "{mode:?}");
            assert!(spins.iter().all(|&s| s == 1));
        }
    }

    #[test]
    fn overlap_moves_preserve_boltzmann() {
        let modes = [
            OverlapClusterBuildMode::Houdayer(2),
            OverlapClusterBuildMode::Jorg,
            OverlapClusterBuildMode::Cmr,
        ];
        for shape in [vec![2, 2], vec![3, 2]] {
            for mode in &modes {
                for cluster_mode in [ClusterMode::Wolff, ClusterMode::Sw] {
                    for with_stats in [false, true] {
                        assert_preserves_boltzmann(shape.clone(), *mode, cluster_mode, with_stats);
                    }
                }
            }
        }
    }
}
