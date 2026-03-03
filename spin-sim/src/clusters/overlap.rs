use super::utils::{
    bfs_cluster, find, find_seed, top4_sizes, uf_bonds, uf_flatten_counts, uf_histogram,
};
use crate::config::{ClusterMode, OverlapClusterBuildMode};
use crate::geometry::Lattice;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;

/// Build per-temperature group assignments: shuffle replicas and chunk into groups.
///
/// Returns `Vec<(temp_idx, group_idx, system_ids_in_group)>`.
fn build_tasks(
    system_ids: &[usize],
    n_replicas: usize,
    n_temps: usize,
    group_size: usize,
    rngs: &mut [Xoshiro256StarStar],
    n_pairs: usize,
) -> Vec<(usize, usize, Vec<usize>)> {
    let n_groups = n_replicas / group_size;
    let mut tasks = Vec::with_capacity(n_temps * n_groups);
    for t in 0..n_temps {
        let mut replica_systems: Vec<usize> = (0..n_replicas)
            .map(|k| system_ids[k * n_temps + t])
            .collect();
        replica_systems.shuffle(&mut rngs[t * n_pairs]);
        for (g, chunk) in replica_systems.chunks_exact(group_size).enumerate() {
            tasks.push((t, g, chunk.to_vec()));
        }
    }
    tasks
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
) {
    let n_spins = lattice.n_spins;
    let n_pairs = n_replicas / 2;
    let wolff = cluster_mode == ClusterMode::Wolff;

    let tasks = build_tasks(system_ids, n_replicas, n_temps, group_size, rngs, n_pairs);

    let sp = spins.as_mut_ptr() as usize;
    let rp = rngs.as_mut_ptr() as usize;
    let use_uf = !wolff || csd_out.is_some() || top4_out.is_some();

    let cp = csd_out.as_ref().map(|s| s.as_ptr() as usize).unwrap_or(0);
    let has_csd = csd_out.is_some();
    let tp = top4_out.as_ref().map(|s| s.as_ptr() as usize).unwrap_or(0);
    let has_top4 = top4_out.is_some();

    let work = |(t, g, systems): &(usize, usize, Vec<usize>)| unsafe {
        let rng = &mut *(rp as *mut Xoshiro256StarStar).add(t * n_pairs + g);
        let bases: Vec<usize> = systems.iter().map(|&s| s * n_spins).collect();
        let sp_ptr = sp as *mut i8;

        let is_active = |i: usize| -> bool {
            let mut sum: i32 = 0;
            for &base in &bases {
                sum += *sp_ptr.add(base + i) as i32;
            }
            sum == 0
        };

        if use_uf {
            let (mut parent, _) = uf_bonds(lattice, |i, d| {
                let j = lattice.neighbor_fwd(i, d);
                is_active(i) && is_active(j)
            });

            if wolff {
                let Some(seed) = find_seed(n_spins, rng, &is_active) else {
                    return;
                };
                let seed_root = find(&mut parent, seed as u32);
                for i in 0..n_spins {
                    if find(&mut parent, i as u32) == seed_root {
                        for &base in &bases {
                            *sp_ptr.add(base + i) *= -1;
                        }
                    }
                }
                if has_csd || has_top4 {
                    let counts = uf_flatten_counts(&mut parent);
                    if has_csd {
                        let csd_slot = &mut *(cp as *mut Vec<u64>).add(t * n_pairs + g);
                        uf_histogram(&counts, csd_slot.as_mut_slice());
                    }
                    if has_top4 {
                        let out = &mut *(tp as *mut [u32; 4]).add(t * n_pairs + g);
                        *out = top4_sizes(&counts);
                    }
                }
            } else {
                let counts = uf_flatten_counts(&mut parent);
                if has_csd {
                    let csd_slot = &mut *(cp as *mut Vec<u64>).add(t * n_pairs + g);
                    uf_histogram(&counts, csd_slot.as_mut_slice());
                }
                if has_top4 {
                    let out = &mut *(tp as *mut [u32; 4]).add(t * n_pairs + g);
                    *out = top4_sizes(&counts);
                }
                let mut do_flip = vec![u8::MAX; n_spins];
                for &p in parent.iter().take(n_spins) {
                    let root = p as usize;
                    if counts[root] > 1 && do_flip[root] == u8::MAX {
                        do_flip[root] = u8::from(rng.gen::<f32>() < 0.5);
                    }
                }
                for (i, &p) in parent.iter().enumerate().take(n_spins) {
                    if do_flip[p as usize] == 1 {
                        for &base in &bases {
                            *sp_ptr.add(base + i) *= -1;
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
            bfs_cluster(
                lattice,
                seed,
                &mut in_cluster,
                &mut stack,
                |site, nb, _d, _fwd| is_active(site) && is_active(nb),
            );
            for (i, &in_c) in in_cluster.iter().enumerate() {
                if in_c {
                    for &base in &bases {
                        *sp_ptr.add(base + i) *= -1;
                    }
                }
            }
        }
    };

    if sequential {
        tasks.iter().for_each(work);
    } else {
        tasks.par_iter().for_each(work);
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
) {
    let n_spins = lattice.n_spins;
    let n_neighbors = lattice.n_neighbors;
    let n_pairs = n_replicas / 2;
    let wolff = cluster_mode == ClusterMode::Wolff;

    let tasks = build_tasks(system_ids, n_replicas, n_temps, 2, rngs, n_pairs);

    let sp = spins.as_mut_ptr() as usize;
    let rp = rngs.as_mut_ptr() as usize;
    let use_uf = !wolff || csd_out.is_some() || top4_out.is_some();

    let cp = csd_out.as_ref().map(|s| s.as_ptr() as usize).unwrap_or(0);
    let has_csd = csd_out.is_some();
    let tp = top4_out.as_ref().map(|s| s.as_ptr() as usize).unwrap_or(0);
    let has_top4 = top4_out.is_some();

    let work = |(t, g, systems): &(usize, usize, Vec<usize>)| unsafe {
        let rng = &mut *(rp as *mut Xoshiro256StarStar).add(t * n_pairs + g);
        let temp = temperatures[*t];
        let base_a = systems[0] * n_spins;
        let base_b = systems[1] * n_spins;
        let sp_ptr = sp as *mut i8;

        let is_active = |i: usize| -> bool { *sp_ptr.add(base_a + i) != *sp_ptr.add(base_b + i) };

        if use_uf {
            let (mut parent, _) = uf_bonds(lattice, |i, d| {
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
                let seed_root = find(&mut parent, seed as u32);
                for i in 0..n_spins {
                    if find(&mut parent, i as u32) == seed_root {
                        *sp_ptr.add(base_a + i) *= -1;
                        *sp_ptr.add(base_b + i) *= -1;
                    }
                }
                if has_csd || has_top4 {
                    let counts = uf_flatten_counts(&mut parent);
                    if has_csd {
                        let csd_slot = &mut *(cp as *mut Vec<u64>).add(t * n_pairs + g);
                        uf_histogram(&counts, csd_slot.as_mut_slice());
                    }
                    if has_top4 {
                        let out = &mut *(tp as *mut [u32; 4]).add(t * n_pairs + g);
                        *out = top4_sizes(&counts);
                    }
                }
            } else {
                let counts = uf_flatten_counts(&mut parent);
                if has_csd {
                    let csd_slot = &mut *(cp as *mut Vec<u64>).add(t * n_pairs + g);
                    uf_histogram(&counts, csd_slot.as_mut_slice());
                }
                if has_top4 {
                    let out = &mut *(tp as *mut [u32; 4]).add(t * n_pairs + g);
                    *out = top4_sizes(&counts);
                }
                let mut do_flip = vec![u8::MAX; n_spins];
                for &p in parent.iter().take(n_spins) {
                    let root = p as usize;
                    if counts[root] > 1 && do_flip[root] == u8::MAX {
                        do_flip[root] = u8::from(rng.gen::<f32>() < 0.5);
                    }
                }
                for (i, &p) in parent.iter().enumerate().take(n_spins) {
                    if do_flip[p as usize] == 1 {
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
            bfs_cluster(
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
        tasks.iter().for_each(work);
    } else {
        tasks.par_iter().for_each(work);
    }
}

/// CMR two-phase overlap cluster update.
///
/// Phase A — Grey cluster randomization:
/// 1. Grey bond: union of two per-replica FK graphs with bond prob 1-r
///    where r = exp(-2|J_ij|/T). Bond if (a_sat AND rand < 1-r) OR (b_sat AND rand < 1-r).
/// 2. SW: flip each replica independently with prob 1/2 per non-singleton cluster.
///    Wolff: pick random seed, grow grey cluster, pick k ∈ {1,2,3} uniformly,
///    flip replica a if k&1, flip replica b if k&2.
///
/// Phase B — Blue cluster flip (same as Jörg but on updated spins):
/// 3. Re-evaluate is_active: σ_i ≠ τ_i after phase A.
/// 4. Blue bond: stochastic, both endpoints same active status,
///    p = 1 - exp(-4 J σ_i σ_j / T) on satisfied bonds.
/// 5. SW: flip each non-singleton blue cluster (both replicas) with prob 1/2.
///    Wolff: grow from same seed as phase A, flip both replicas.
/// 6. CSD/top4 from blue clusters only.
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
) {
    let n_spins = lattice.n_spins;
    let n_neighbors = lattice.n_neighbors;
    let n_pairs = n_replicas / 2;
    let wolff = cluster_mode == ClusterMode::Wolff;

    let tasks = build_tasks(system_ids, n_replicas, n_temps, 2, rngs, n_pairs);

    let sp = spins.as_mut_ptr() as usize;
    let rp = rngs.as_mut_ptr() as usize;
    let use_uf_blue = !wolff || csd_out.is_some() || top4_out.is_some();

    let cp = csd_out.as_ref().map(|s| s.as_ptr() as usize).unwrap_or(0);
    let has_csd = csd_out.is_some();
    let tp = top4_out.as_ref().map(|s| s.as_ptr() as usize).unwrap_or(0);
    let has_top4 = top4_out.is_some();

    let work = |(t, g, systems): &(usize, usize, Vec<usize>)| unsafe {
        let rng = &mut *(rp as *mut Xoshiro256StarStar).add(t * n_pairs + g);
        let temp = temperatures[*t];
        let base_a = systems[0] * n_spins;
        let base_b = systems[1] * n_spins;
        let sp_ptr = sp as *mut i8;

        // === Phase A: Grey cluster randomization ===
        let grey_seed: usize;

        if wolff {
            // Wolff grey: BFS from random seed, flip with k ∈ {1,2,3}
            let Some(seed) = find_seed(n_spins, rng, |_| true) else {
                return;
            };
            grey_seed = seed;

            let mut in_cluster = vec![false; n_spins];
            let mut stack = Vec::with_capacity(n_spins);
            bfs_cluster(
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
                    let j_abs = couplings[coupling_idx].abs();
                    let r = (-2.0 * j_abs / temp).exp();

                    let a_sat_site = *sp_ptr.add(base_a + site) as f32
                        * *sp_ptr.add(base_a + nb) as f32
                        * couplings[coupling_idx]
                        > 0.0;
                    let b_sat_site = *sp_ptr.add(base_b + site) as f32
                        * *sp_ptr.add(base_b + nb) as f32
                        * couplings[coupling_idx]
                        > 0.0;

                    (a_sat_site && rng.gen::<f32>() < 1.0 - r)
                        || (b_sat_site && rng.gen::<f32>() < 1.0 - r)
                },
            );

            // Pick k ∈ {1,2,3}: flip replica a if k&1, replica b if k&2
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
        } else {
            // SW grey: UF decomposition, flip each replica independently with prob 1/2
            grey_seed = rng.gen_range(0..n_spins);

            let (mut parent, _) = uf_bonds(lattice, |i, d| {
                let j = lattice.neighbor_fwd(i, d);
                let j_abs = couplings[i * n_neighbors + d].abs();
                let r = (-2.0 * j_abs / temp).exp();

                let a_sat = *sp_ptr.add(base_a + i) as f32
                    * *sp_ptr.add(base_a + j) as f32
                    * couplings[i * n_neighbors + d]
                    > 0.0;
                let b_sat = *sp_ptr.add(base_b + i) as f32
                    * *sp_ptr.add(base_b + j) as f32
                    * couplings[i * n_neighbors + d]
                    > 0.0;

                (a_sat && rng.gen::<f32>() < 1.0 - r) || (b_sat && rng.gen::<f32>() < 1.0 - r)
            });

            let counts = uf_flatten_counts(&mut parent);

            // For each non-singleton grey cluster, flip each replica independently with prob 1/2
            // Generate one random per cluster root (2 bits: flip_a, flip_b)
            let mut cluster_flip = vec![0u8; n_spins];
            for (i, &p) in parent.iter().enumerate().take(n_spins) {
                let root = p as usize;
                if counts[root] > 1 && root == i {
                    cluster_flip[root] = rng.gen_range(0..=3);
                }
            }

            for (i, &p) in parent.iter().enumerate().take(n_spins) {
                let k = cluster_flip[p as usize];
                if k == 0 {
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

        // === Phase B: Blue cluster flip ===
        let is_active = |i: usize| -> bool { *sp_ptr.add(base_a + i) != *sp_ptr.add(base_b + i) };

        if use_uf_blue {
            let (mut parent, _) = uf_bonds(lattice, |i, d| {
                let j = lattice.neighbor_fwd(i, d);
                if is_active(i) != is_active(j) {
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
                // Grow blue cluster from same seed as grey
                let seed_root = find(&mut parent, grey_seed as u32);
                for i in 0..n_spins {
                    if find(&mut parent, i as u32) == seed_root {
                        *sp_ptr.add(base_a + i) *= -1;
                        *sp_ptr.add(base_b + i) *= -1;
                    }
                }
                if has_csd || has_top4 {
                    let counts = uf_flatten_counts(&mut parent);
                    if has_csd {
                        let csd_slot = &mut *(cp as *mut Vec<u64>).add(t * n_pairs + g);
                        uf_histogram(&counts, csd_slot.as_mut_slice());
                    }
                    if has_top4 {
                        let out = &mut *(tp as *mut [u32; 4]).add(t * n_pairs + g);
                        *out = top4_sizes(&counts);
                    }
                }
            } else {
                let counts = uf_flatten_counts(&mut parent);
                if has_csd {
                    let csd_slot = &mut *(cp as *mut Vec<u64>).add(t * n_pairs + g);
                    uf_histogram(&counts, csd_slot.as_mut_slice());
                }
                if has_top4 {
                    let out = &mut *(tp as *mut [u32; 4]).add(t * n_pairs + g);
                    *out = top4_sizes(&counts);
                }
                // SW: flip each non-singleton blue cluster with prob 1/2
                let mut cluster_do_flip = vec![u8::MAX; n_spins]; // sentinel = not decided
                for &p in parent.iter().take(n_spins) {
                    let root = p as usize;
                    if counts[root] > 1 && cluster_do_flip[root] == u8::MAX {
                        cluster_do_flip[root] = rng.gen_range(0..=1);
                    }
                }
                for (i, &p) in parent.iter().enumerate().take(n_spins) {
                    if cluster_do_flip[p as usize] == 1 {
                        *sp_ptr.add(base_a + i) *= -1;
                        *sp_ptr.add(base_b + i) *= -1;
                    }
                }
            }
        } else {
            // Pure Wolff blue without stats — grow BFS from grey_seed
            let mut in_cluster = vec![false; n_spins];
            let mut stack = Vec::with_capacity(n_spins);
            bfs_cluster(
                lattice,
                grey_seed,
                &mut in_cluster,
                &mut stack,
                |site, nb, d, fwd| {
                    if is_active(site) != is_active(nb) {
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
        tasks.iter().for_each(work);
    } else {
        tasks.par_iter().for_each(work);
    }
}
