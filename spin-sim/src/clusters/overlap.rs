use super::utils::{bfs_cluster, find, find_seed, top4_sizes, uf_bonds};
use crate::geometry::Lattice;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;

/// Overlap cluster update (Houdayer ICM, Jörg, or CMR-N), parallelized over groups.
///
/// For each temperature, shuffles replicas into random groups of `group_size`
/// (2 for Houdayer/Jörg/CMR, N for CMR-N). For each group, grows a cluster on
/// the overlap subgraph and either swaps or freely assigns it.
///
/// `rngs` has length `n_temps * n_pairs` (one RNG per pair slot, positional).
/// The first slot's RNG at each temperature is also used for the shuffle.
/// Since `n_groups = n_replicas / group_size <= n_pairs`, the array is always
/// large enough.
///
/// When `stochastic` is false (Houdayer), bonds are deterministic (prob=1).
/// When `stochastic` is true (Jörg/CMR-N), bond activation is
/// `p = 1 - exp(-4 * J * σ_i * σ_j / T)` on satisfied bonds (group_size=2) or
/// `p = 1 - exp(-2N * |J| / T)` on N-fold satisfied bonds (group_size=N>=3).
///
/// When `restrict_to_negative` is true (Houdayer/Jörg), only negative-overlap
/// sites are eligible. When false (CMR-N), all same-sign overlap sites are
/// bonded (group_size=2) or all sites are eligible (group_size>=3).
///
/// When `wolff` is true, uses BFS single-cluster (one seed per group).
/// When `wolff` is false, uses union-find global decomposition and swaps/flips
/// all non-singleton clusters. CSD/top4 collection forces UF even when `wolff`.
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
    group_size: usize,
    csd_out: Option<&mut [Vec<u64>]>,
    top4_out: Option<&mut [[u32; 4]]>,
    sequential: bool,
) {
    let n_spins = lattice.n_spins;
    let n_neighbors = lattice.n_neighbors;
    let n_pairs = n_replicas / 2;
    let n_groups = n_replicas / group_size;

    // Phase 1 (sequential): determine all groups via per-temperature shuffle
    let mut tasks: Vec<(usize, usize, Vec<usize>)> = Vec::with_capacity(n_temps * n_groups);
    for t in 0..n_temps {
        let mut replica_systems: Vec<usize> = (0..n_replicas)
            .map(|k| system_ids[k * n_temps + t])
            .collect();
        replica_systems.shuffle(&mut rngs[t * n_pairs]);
        for (g, chunk) in replica_systems.chunks_exact(group_size).enumerate() {
            tasks.push((t, g, chunk.to_vec()));
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

    let work = |(t, g, systems): &(usize, usize, Vec<usize>)| unsafe {
        let rng = &mut *(rp as *mut Xoshiro256StarStar).add(t * n_pairs + g);
        let temp = temperatures[*t];
        let bases: Vec<usize> = systems.iter().map(|&s| s * n_spins).collect();
        let sp_ptr = sp as *mut i8;

        let overlap_sign =
            |i: usize| -> bool { *sp_ptr.add(bases[0] + i) != *sp_ptr.add(bases[1] + i) };

        let n_bond_coeff = -((2 * group_size) as f32);

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
                        let coup = couplings[i * n_neighbors + d];
                        for &base in &bases {
                            let inter =
                                *sp_ptr.add(base + i) as f32 * *sp_ptr.add(base + j) as f32 * coup;
                            if inter <= 0.0 {
                                return false;
                            }
                        }
                        rng.gen::<f32>() < 1.0 - (n_bond_coeff * coup.abs() / temp).exp()
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
                        let inter = *sp_ptr.add(bases[0] + i) as f32
                            * *sp_ptr.add(bases[0] + j) as f32
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

                if group_size >= 3 {
                    let flip_mask = rng.gen_range(1u64..(1u64 << group_size));
                    for i in 0..n_spins {
                        if find(&mut parent, i as u32) == seed_root {
                            for (k, &base) in bases.iter().enumerate() {
                                if flip_mask & (1u64 << k) != 0 {
                                    *sp_ptr.add(base + i) *= -1;
                                }
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
                                *sp_ptr.add(bases[0] + i) *= -1;
                            }
                            if do_flip_b {
                                *sp_ptr.add(bases[1] + i) *= -1;
                            }
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

                if group_size >= 3 {
                    let mut flips: Vec<Vec<u8>> =
                        (0..group_size).map(|_| vec![u8::MAX; n_spins]).collect();
                    for (i, &p) in parent.iter().enumerate().take(n_spins) {
                        let root = p as usize;
                        if counts[root] > 1 {
                            if flips[0][root] == u8::MAX {
                                for flip in flips.iter_mut() {
                                    flip[root] = rng.gen::<u8>() & 1;
                                }
                            }
                            for (flip, &base) in flips.iter().zip(&bases) {
                                if flip[root] == 1 {
                                    *sp_ptr.add(base + i) *= -1;
                                }
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
                                *sp_ptr.add(bases[0] + i) *= -1;
                            }
                            if flip_b[root] == 1 {
                                *sp_ptr.add(bases[1] + i) *= -1;
                            }
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
                        for &base in &bases {
                            let inter = *sp_ptr.add(base + site) as f32
                                * *sp_ptr.add(base + nb) as f32
                                * coup;
                            if inter <= 0.0 {
                                return false;
                            }
                        }
                        rng.gen::<f32>() < 1.0 - (n_bond_coeff * coup.abs() / temp).exp()
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
                        let inter = *sp_ptr.add(bases[0] + site) as f32
                            * *sp_ptr.add(bases[0] + nb) as f32
                            * coupling;
                        if inter <= 0.0 {
                            return false;
                        }
                        rng.gen::<f32>() < 1.0 - (-4.0 * inter / temp).exp()
                    }
                },
            );

            if group_size >= 3 {
                let flip_mask = rng.gen_range(1u64..(1u64 << group_size));
                for (i, &in_c) in in_cluster.iter().enumerate() {
                    if in_c {
                        for (k, &base) in bases.iter().enumerate() {
                            if flip_mask & (1u64 << k) != 0 {
                                *sp_ptr.add(base + i) *= -1;
                            }
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
                            *sp_ptr.add(bases[0] + i) *= -1;
                        }
                        if do_flip_b {
                            *sp_ptr.add(bases[1] + i) *= -1;
                        }
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
