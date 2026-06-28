use super::utils::{
    dfs_cluster, find, find_seed, top4_sizes, uf_bonds, uf_bonds_extend, uf_flatten_counts,
    uf_histogram,
};
use crate::config::{ClusterMode, OverlapClusterBuildMode};
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
    csd_out: Option<&mut [Vec<u64>]>,
    top4_out: Option<&mut [[u32; 4]]>,
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
            csd_out,
            top4_out,
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
            csd_out,
            top4_out,
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
            csd_out,
            top4_out,
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
    csd_out: Option<&mut [Vec<u64>]>,
    top4_out: Option<&mut [[u32; 4]]>,
    sequential: bool,
    snap_out: Option<&mut [Vec<u32>]>,
    spin_snap_out: Option<&mut [Vec<[Vec<i8>; 2]>]>,
    sid_snap_out: Option<&mut [Vec<[usize; 2]>]>,
) {
    let n_spins = lattice.n_spins;
    let n_pairs = n_replicas / 2;
    let wolff = cluster_mode == ClusterMode::Wolff;

    let tasks = build_tasks(system_ids, n_replicas, n_temps, group_size, rngs, n_pairs);

    let sp = spins.as_mut_ptr() as usize;
    let rp = rngs.as_mut_ptr() as usize;
    let use_uf = !wolff || csd_out.is_some() || top4_out.is_some() || snap_out.is_some();

    let cp = csd_out.as_ref().map(|s| s.as_ptr() as usize).unwrap_or(0);
    let has_csd = csd_out.is_some();
    let tp = top4_out.as_ref().map(|s| s.as_ptr() as usize).unwrap_or(0);
    let has_top4 = top4_out.is_some();
    let snp = snap_out.as_ref().map(|s| s.as_ptr() as usize).unwrap_or(0);
    let has_snap = snap_out.is_some();
    let spp = spin_snap_out
        .as_ref()
        .map(|s| s.as_ptr() as usize)
        .unwrap_or(0);
    let sidp = sid_snap_out
        .as_ref()
        .map(|s| s.as_ptr() as usize)
        .unwrap_or(0);

    let work = |task_idx: usize| unsafe {
        let (t, g, systems) = tasks.group(task_idx);
        let rng = &mut *(rp as *mut Xoshiro256StarStar).add(t * n_pairs + g);
        let sp_ptr = sp as *mut i8;
        let slot = t * n_pairs + g;

        if has_snap && systems.len() >= 2 {
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
            let mut uf = uf_bonds(lattice, |i, d| {
                let j = lattice.neighbor_fwd(i, d);
                is_active(i) && is_active(j)
            });

            if wolff {
                let Some(seed) = find_seed(n_spins, rng, &is_active) else {
                    return;
                };
                let seed_root = find(&mut uf.parent, seed as u32);
                for i in 0..n_spins {
                    if find(&mut uf.parent, i as u32) == seed_root {
                        for &system in systems {
                            *sp_ptr.add(system * n_spins + i) *= -1;
                        }
                    }
                }
                if has_csd || has_top4 || has_snap {
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
                }
            } else {
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
    csd_out: Option<&mut [Vec<u64>]>,
    top4_out: Option<&mut [[u32; 4]]>,
    sequential: bool,
    snap_out: Option<&mut [Vec<u32>]>,
    spin_snap_out: Option<&mut [Vec<[Vec<i8>; 2]>]>,
    sid_snap_out: Option<&mut [Vec<[usize; 2]>]>,
) {
    let n_spins = lattice.n_spins;
    let n_neighbors = lattice.n_neighbors;
    let n_pairs = n_replicas / 2;
    let wolff = cluster_mode == ClusterMode::Wolff;

    let tasks = build_tasks(system_ids, n_replicas, n_temps, 2, rngs, n_pairs);

    let sp = spins.as_mut_ptr() as usize;
    let rp = rngs.as_mut_ptr() as usize;
    let use_uf = !wolff || csd_out.is_some() || top4_out.is_some() || snap_out.is_some();

    let cp = csd_out.as_ref().map(|s| s.as_ptr() as usize).unwrap_or(0);
    let has_csd = csd_out.is_some();
    let tp = top4_out.as_ref().map(|s| s.as_ptr() as usize).unwrap_or(0);
    let has_top4 = top4_out.is_some();
    let snp = snap_out.as_ref().map(|s| s.as_ptr() as usize).unwrap_or(0);
    let has_snap = snap_out.is_some();
    let spp = spin_snap_out
        .as_ref()
        .map(|s| s.as_ptr() as usize)
        .unwrap_or(0);
    let sidp = sid_snap_out
        .as_ref()
        .map(|s| s.as_ptr() as usize)
        .unwrap_or(0);

    let work = |task_idx: usize| unsafe {
        let (t, g, systems) = tasks.group(task_idx);
        let rng = &mut *(rp as *mut Xoshiro256StarStar).add(t * n_pairs + g);
        let temp = temperatures[t];
        let base_a = systems[0] * n_spins;
        let base_b = systems[1] * n_spins;
        let sp_ptr = sp as *mut i8;
        let slot = t * n_pairs + g;

        if has_snap {
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
            let mut uf = uf_bonds(lattice, |i, d| {
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
                rng.gen::<f32>() < 1.0 - (-4.0 * inter / temp).exp()
            });

            if wolff {
                let Some(seed) = find_seed(n_spins, rng, &is_active) else {
                    return;
                };
                let seed_root = find(&mut uf.parent, seed as u32);
                for i in 0..n_spins {
                    if find(&mut uf.parent, i as u32) == seed_root {
                        *sp_ptr.add(base_a + i) *= -1;
                        *sp_ptr.add(base_b + i) *= -1;
                    }
                }
                if has_csd || has_top4 || has_snap {
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
                }
            } else {
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
                    rng.gen::<f32>() < 1.0 - (-4.0 * inter / temp).exp()
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
    csd_out: Option<&mut [Vec<u64>]>,
    top4_out: Option<&mut [[u32; 4]]>,
    sequential: bool,
    snap_out: Option<&mut [Vec<u32>]>,
    blue_snap_out: Option<&mut [Vec<u32>]>,
    spin_snap_out: Option<&mut [Vec<[Vec<i8>; 2]>]>,
    sid_snap_out: Option<&mut [Vec<[usize; 2]>]>,
) {
    let n_spins = lattice.n_spins;
    let n_neighbors = lattice.n_neighbors;
    let n_pairs = n_replicas / 2;
    let wolff = cluster_mode == ClusterMode::Wolff;

    let tasks = build_tasks(system_ids, n_replicas, n_temps, 2, rngs, n_pairs);

    let sp = spins.as_mut_ptr() as usize;
    let rp = rngs.as_mut_ptr() as usize;
    let use_uf = !wolff || csd_out.is_some() || top4_out.is_some() || snap_out.is_some();

    let cp = csd_out.as_ref().map(|s| s.as_ptr() as usize).unwrap_or(0);
    let has_csd = csd_out.is_some();
    let tp = top4_out.as_ref().map(|s| s.as_ptr() as usize).unwrap_or(0);
    let has_top4 = top4_out.is_some();
    let snp = snap_out.as_ref().map(|s| s.as_ptr() as usize).unwrap_or(0);
    let has_snap = snap_out.is_some();
    let bsnp = blue_snap_out
        .as_ref()
        .map(|s| s.as_ptr() as usize)
        .unwrap_or(0);
    let has_blue_snap = blue_snap_out.is_some();
    let spp = spin_snap_out
        .as_ref()
        .map(|s| s.as_ptr() as usize)
        .unwrap_or(0);
    let sidp = sid_snap_out
        .as_ref()
        .map(|s| s.as_ptr() as usize)
        .unwrap_or(0);

    let work = |task_idx: usize| unsafe {
        let (t, g, systems) = tasks.group(task_idx);
        let rng = &mut *(rp as *mut Xoshiro256StarStar).add(t * n_pairs + g);
        let temp = temperatures[t];
        let base_a = systems[0] * n_spins;
        let base_b = systems[1] * n_spins;
        let sp_ptr = sp as *mut i8;
        let slot = t * n_pairs + g;

        if has_snap {
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

            // === Phase 1: Blue clusters ===
            // Blue bond: doubly-satisfied edges with prob 1 - r²
            let mut uf = uf_bonds(lattice, |i, d| {
                let j = lattice.neighbor_fwd(i, d);
                let coupling = couplings[i * n_neighbors + d];

                let a_sat =
                    *sp_ptr.add(base_a + i) as f32 * *sp_ptr.add(base_a + j) as f32 * coupling
                        > 0.0;
                let b_sat =
                    *sp_ptr.add(base_b + i) as f32 * *sp_ptr.add(base_b + j) as f32 * coupling
                        > 0.0;

                if !a_sat || !b_sat {
                    return false;
                }
                let r = (-2.0 * coupling.abs() / temp).exp();
                rng.gen::<f32>() < 1.0 - r * r
            });

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

                let a_sat =
                    *sp_ptr.add(base_a + i) as f32 * *sp_ptr.add(base_a + j) as f32 * coupling
                        > 0.0;
                let b_sat =
                    *sp_ptr.add(base_b + i) as f32 * *sp_ptr.add(base_b + j) as f32 * coupling
                        > 0.0;

                if a_sat == b_sat {
                    return false;
                }
                let r = (-2.0 * coupling.abs() / temp).exp();
                rng.gen::<f32>() < 1.0 - r
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
                    let r = (-2.0 * coupling.abs() / temp).exp();
                    rng.gen::<f32>() < 1.0 - r * r
                },
            );

            // Flip blue cluster (both replicas always)
            for (i, &in_c) in in_cluster.iter().enumerate() {
                if in_c {
                    *sp_ptr.add(base_a + i) *= -1;
                    *sp_ptr.add(base_b + i) *= -1;
                }
            }

            // === Phase 2: Grey cluster (extend from blue frontier) ===
            // Re-seed BFS from all blue sites; in_cluster prevents re-adding them.
            for (i, &in_c) in in_cluster.iter().enumerate() {
                if in_c {
                    stack.push(i);
                }
            }

            // Continue BFS with red bond rule (singly-satisfied, prob 1-r)
            while let Some(site) = stack.pop() {
                for d in 0..lattice.n_neighbors {
                    let fwd = lattice.neighbor_fwd(site, d);
                    if !in_cluster[fwd] {
                        let coupling = couplings[site * n_neighbors + d];

                        let a_sat = *sp_ptr.add(base_a + site) as f32
                            * *sp_ptr.add(base_a + fwd) as f32
                            * coupling
                            > 0.0;
                        let b_sat = *sp_ptr.add(base_b + site) as f32
                            * *sp_ptr.add(base_b + fwd) as f32
                            * coupling
                            > 0.0;

                        let eligible = a_sat != b_sat;
                        let activate = eligible && {
                            let r = (-2.0 * coupling.abs() / temp).exp();
                            rng.gen::<f32>() < 1.0 - r
                        };
                        if activate {
                            in_cluster[fwd] = true;
                            stack.push(fwd);
                        }
                    }

                    let bwd = lattice.neighbor_bwd(site, d);
                    if !in_cluster[bwd] {
                        let coupling = couplings[bwd * n_neighbors + d];

                        let a_sat = *sp_ptr.add(base_a + site) as f32
                            * *sp_ptr.add(base_a + bwd) as f32
                            * coupling
                            > 0.0;
                        let b_sat = *sp_ptr.add(base_b + site) as f32
                            * *sp_ptr.add(base_b + bwd) as f32
                            * coupling
                            > 0.0;

                        let eligible = a_sat != b_sat;
                        let activate = eligible && {
                            let r = (-2.0 * coupling.abs() / temp).exp();
                            rng.gen::<f32>() < 1.0 - r
                        };
                        if activate {
                            in_cluster[bwd] = true;
                            stack.push(bwd);
                        }
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
