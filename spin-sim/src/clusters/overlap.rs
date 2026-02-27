use super::utils::{bfs_cluster, find, find_seed, top4_sizes, uf_bonds};
use crate::geometry::Lattice;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;

/// Overlap cluster update (Houdayer ICM, Jörg, CMR, or CMR3), parallelized over groups.
///
/// For each temperature, shuffles replicas into random groups of `group_size`
/// (2 for Houdayer/Jörg/CMR, 3 for CMR3). For each group, grows a cluster on
/// the overlap subgraph and either swaps or freely assigns it.
///
/// `rngs` has length `n_temps * n_pairs` (one RNG per pair slot, positional).
/// The first slot's RNG at each temperature is also used for the shuffle.
/// Since `n_groups = n_replicas / group_size <= n_pairs`, the array is always
/// large enough.
///
/// When `stochastic` is false (Houdayer), bonds are deterministic (prob=1).
/// When `stochastic` is true (Jörg/CMR/CMR3), bond activation is
/// `p = 1 - exp(-4 * J * σ_i * σ_j / T)` on satisfied bonds (group_size=2) or
/// `p = 1 - exp(-6 * |J| / T)` on triply-satisfied bonds (group_size=3).
///
/// When `restrict_to_negative` is true (Houdayer/Jörg), only negative-overlap
/// sites are eligible. When false (CMR/CMR3), all same-sign overlap sites are
/// bonded (group_size=2) or all sites are eligible (group_size=3).
///
/// When `wolff` is true, uses BFS single-cluster (one seed per group).
/// When `wolff` is false, uses union-find global decomposition and swaps/flips
/// all non-singleton clusters. CSD/top4 collection forces UF even when `wolff`.
///
/// When `free_assign` is true (free CMR/CMR3), instead of swapping σ^a ↔ σ^b,
/// each replica is independently flipped. For group_size=2: SW picks per-cluster
/// coin flips, Wolff picks uniformly from {flip_a, flip_b, flip_both}. For
/// group_size=3: SW picks 3 independent coin flips per cluster, Wolff picks
/// uniformly from the 7 non-identity states.
///
/// When `csd_out` is `Some`, forces UF path and histograms per-group cluster
/// sizes. Slice length must be `n_temps * n_pairs`, indexed by
/// `t * n_pairs + g`. Each inner vec must be pre-sized to `n_spins + 1`.
///
/// When `top4_out` is `Some`, forces UF path and writes the 4 largest cluster
/// sizes (descending) per group. Indexed by `t * n_pairs + g`.
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
    stochastic: bool,
    restrict_to_negative: bool,
    wolff: bool,
    free_assign: bool,
    group_size: usize,
    csd_out: Option<&mut [Vec<u64>]>,
    top4_out: Option<&mut [[u32; 4]]>,
) {
    let n_spins = lattice.n_spins;
    let n_neighbors = lattice.n_neighbors;
    let n_pairs = n_replicas / 2;
    let n_groups = n_replicas / group_size;

    // Phase 1 (sequential): determine all groups via per-temperature shuffle
    // Task: (temp_idx, group_idx, [sys_a, sys_b, sys_c]) where sys_c == usize::MAX when group_size == 2
    let mut tasks: Vec<(usize, usize, [usize; 3])> = Vec::with_capacity(n_temps * n_groups);
    for t in 0..n_temps {
        let mut replica_systems: Vec<usize> = (0..n_replicas)
            .map(|k| system_ids[k * n_temps + t])
            .collect();
        replica_systems.shuffle(&mut rngs[t * n_pairs]);
        for (g, chunk) in replica_systems.chunks_exact(group_size).enumerate() {
            let systems = [
                chunk[0],
                chunk[1],
                if group_size >= 3 {
                    chunk[2]
                } else {
                    usize::MAX
                },
            ];
            tasks.push((t, g, systems));
        }
    }

    // Phase 2 (parallel): process all pairs
    let sp = spins.as_mut_ptr() as usize;
    let rp = rngs.as_mut_ptr() as usize;
    let use_uf = !wolff || csd_out.is_some() || top4_out.is_some();

    let cp = csd_out.as_ref().map(|s| s.as_ptr() as usize).unwrap_or(0);
    let has_csd = csd_out.is_some();

    let tp = top4_out.as_ref().map(|s| s.as_ptr() as usize).unwrap_or(0);
    let has_top4 = top4_out.is_some();

    tasks.par_iter().for_each(|&(t, g, systems)| unsafe {
        let rng = &mut *(rp as *mut Xoshiro256StarStar).add(t * n_pairs + g);
        let temp = temperatures[t];
        let base_a = systems[0] * n_spins;
        let base_b = systems[1] * n_spins;
        let base_c = if group_size >= 3 {
            systems[2] * n_spins
        } else {
            0
        };
        let sp_ptr = sp as *mut i8;

        let overlap_sign =
            |i: usize| -> bool { *sp_ptr.add(base_a + i) != *sp_ptr.add(base_b + i) };

        if use_uf {
            let csd_slot = if has_csd {
                let slot = &mut *(cp as *mut Vec<u64>).add(t * n_pairs + g);
                Some(slot.as_mut_slice())
            } else {
                None
            };

            let (mut parent, _) = uf_bonds(
                lattice,
                |i, d| {
                    let j = lattice.neighbor_fwd(i, d);
                    if group_size >= 3 {
                        // Triple satisfaction: all 3 replicas must have satisfied bond
                        let sa_i = *sp_ptr.add(base_a + i) as f32;
                        let sa_j = *sp_ptr.add(base_a + j) as f32;
                        let sb_i = *sp_ptr.add(base_b + i) as f32;
                        let sb_j = *sp_ptr.add(base_b + j) as f32;
                        let sc_i = *sp_ptr.add(base_c + i) as f32;
                        let sc_j = *sp_ptr.add(base_c + j) as f32;
                        let coup = couplings[i * n_neighbors + d];
                        let inter_a = sa_i * sa_j * coup;
                        let inter_b = sb_i * sb_j * coup;
                        let inter_c = sc_i * sc_j * coup;
                        if inter_a <= 0.0 || inter_b <= 0.0 || inter_c <= 0.0 {
                            return false;
                        }
                        rng.gen::<f32>() < 1.0 - (-6.0 * inter_a.abs() / temp).exp()
                    } else {
                        if restrict_to_negative {
                            if !overlap_sign(i) || !overlap_sign(j) {
                                return false;
                            }
                        } else if overlap_sign(i) != overlap_sign(j) {
                            return false;
                        }
                        if !stochastic {
                            return true;
                        }
                        let inter = *sp_ptr.add(base_a + i) as f32
                            * *sp_ptr.add(base_a + j) as f32
                            * couplings[i * n_neighbors + d];
                        if inter <= 0.0 {
                            return false;
                        }
                        rng.gen::<f32>() < 1.0 - (-4.0 * inter / temp).exp()
                    }
                },
                csd_slot,
            );

            if wolff {
                // Wolff + CSD/top4: UF decomposition already done, operate on seed's cluster
                let Some(seed) = find_seed(n_spins, rng, |i| {
                    if group_size >= 3 {
                        true
                    } else if restrict_to_negative {
                        overlap_sign(i)
                    } else {
                        true
                    }
                }) else {
                    return;
                };
                let seed_root = find(&mut parent, seed as u32);

                if free_assign {
                    if group_size >= 3 {
                        let flip_mask = rng.gen_range(1u8..8);
                        let do_flip_a = flip_mask & 1 != 0;
                        let do_flip_b = flip_mask & 2 != 0;
                        let do_flip_c = flip_mask & 4 != 0;
                        for i in 0..n_spins {
                            if find(&mut parent, i as u32) == seed_root {
                                if do_flip_a {
                                    *sp_ptr.add(base_a + i) *= -1;
                                }
                                if do_flip_b {
                                    *sp_ptr.add(base_b + i) *= -1;
                                }
                                if do_flip_c {
                                    *sp_ptr.add(base_c + i) *= -1;
                                }
                            }
                        }
                    } else {
                        let flip_choice = rng.gen_range(0u8..3);
                        let do_flip_a = flip_choice != 1;
                        let do_flip_b = flip_choice != 0;
                        for i in 0..n_spins {
                            if find(&mut parent, i as u32) == seed_root {
                                if do_flip_a {
                                    *sp_ptr.add(base_a + i) *= -1;
                                }
                                if do_flip_b {
                                    *sp_ptr.add(base_b + i) *= -1;
                                }
                            }
                        }
                    }
                } else {
                    for i in 0..n_spins {
                        if find(&mut parent, i as u32) == seed_root {
                            std::ptr::swap(sp_ptr.add(base_a + i), sp_ptr.add(base_b + i));
                        }
                    }
                }
            } else {
                // SW: flatten parents, compute counts, operate on all non-singletons
                for i in 0..n_spins {
                    parent[i] = find(&mut parent, i as u32);
                }
                let mut counts = vec![0usize; n_spins];
                for i in 0..n_spins {
                    counts[parent[i] as usize] += 1;
                }

                if has_top4 {
                    let out = &mut *(tp as *mut [u32; 4]).add(t * n_pairs + g);
                    *out = top4_sizes(&counts);
                }

                if free_assign {
                    if group_size >= 3 {
                        let mut flip_a = vec![u8::MAX; n_spins];
                        let mut flip_b = vec![u8::MAX; n_spins];
                        let mut flip_c = vec![u8::MAX; n_spins];
                        for (i, &p) in parent.iter().enumerate().take(n_spins) {
                            let root = p as usize;
                            if counts[root] > 1 {
                                if flip_a[root] == u8::MAX {
                                    flip_a[root] = rng.gen::<u8>() & 1;
                                    flip_b[root] = rng.gen::<u8>() & 1;
                                    flip_c[root] = rng.gen::<u8>() & 1;
                                }
                                if flip_a[root] == 1 {
                                    *sp_ptr.add(base_a + i) *= -1;
                                }
                                if flip_b[root] == 1 {
                                    *sp_ptr.add(base_b + i) *= -1;
                                }
                                if flip_c[root] == 1 {
                                    *sp_ptr.add(base_c + i) *= -1;
                                }
                            }
                        }
                    } else {
                        let mut flip_a = vec![u8::MAX; n_spins];
                        let mut flip_b = vec![u8::MAX; n_spins];
                        for (i, &p) in parent.iter().enumerate().take(n_spins) {
                            let root = p as usize;
                            if counts[root] > 1 {
                                if flip_a[root] == u8::MAX {
                                    flip_a[root] = rng.gen::<u8>() & 1;
                                    flip_b[root] = rng.gen::<u8>() & 1;
                                }
                                if flip_a[root] == 1 {
                                    *sp_ptr.add(base_a + i) *= -1;
                                }
                                if flip_b[root] == 1 {
                                    *sp_ptr.add(base_b + i) *= -1;
                                }
                            }
                        }
                    }
                } else {
                    for i in 0..n_spins {
                        if counts[parent[i] as usize] > 1 {
                            std::ptr::swap(sp_ptr.add(base_a + i), sp_ptr.add(base_b + i));
                        }
                    }
                }
            }
        } else {
            // Pure Wolff without CSD/top4
            let Some(seed) = find_seed(n_spins, rng, |i| {
                if group_size >= 3 {
                    true
                } else if restrict_to_negative {
                    overlap_sign(i)
                } else {
                    true
                }
            }) else {
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
                    if group_size >= 3 {
                        let coup = if fwd {
                            couplings[site * n_neighbors + d]
                        } else {
                            couplings[nb * n_neighbors + d]
                        };
                        let inter_a = *sp_ptr.add(base_a + site) as f32
                            * *sp_ptr.add(base_a + nb) as f32
                            * coup;
                        let inter_b = *sp_ptr.add(base_b + site) as f32
                            * *sp_ptr.add(base_b + nb) as f32
                            * coup;
                        let inter_c = *sp_ptr.add(base_c + site) as f32
                            * *sp_ptr.add(base_c + nb) as f32
                            * coup;
                        if inter_a <= 0.0 || inter_b <= 0.0 || inter_c <= 0.0 {
                            return false;
                        }
                        rng.gen::<f32>() < 1.0 - (-6.0 * inter_a.abs() / temp).exp()
                    } else {
                        if restrict_to_negative {
                            if !overlap_sign(nb) {
                                return false;
                            }
                        } else if overlap_sign(site) != overlap_sign(nb) {
                            return false;
                        }
                        if !stochastic {
                            return true;
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
                    }
                },
            );

            if free_assign {
                if group_size >= 3 {
                    let flip_mask = rng.gen_range(1u8..8);
                    let do_flip_a = flip_mask & 1 != 0;
                    let do_flip_b = flip_mask & 2 != 0;
                    let do_flip_c = flip_mask & 4 != 0;
                    for (i, &in_c) in in_cluster.iter().enumerate() {
                        if in_c {
                            if do_flip_a {
                                *sp_ptr.add(base_a + i) *= -1;
                            }
                            if do_flip_b {
                                *sp_ptr.add(base_b + i) *= -1;
                            }
                            if do_flip_c {
                                *sp_ptr.add(base_c + i) *= -1;
                            }
                        }
                    }
                } else {
                    let flip_choice = rng.gen_range(0u8..3);
                    let do_flip_a = flip_choice != 1;
                    let do_flip_b = flip_choice != 0;
                    for (i, &in_c) in in_cluster.iter().enumerate() {
                        if in_c {
                            if do_flip_a {
                                *sp_ptr.add(base_a + i) *= -1;
                            }
                            if do_flip_b {
                                *sp_ptr.add(base_b + i) *= -1;
                            }
                        }
                    }
                }
            } else {
                for (i, &in_c) in in_cluster.iter().enumerate() {
                    if in_c {
                        std::ptr::swap(sp_ptr.add(base_a + i), sp_ptr.add(base_b + i));
                    }
                }
            }
        }
    });
}
