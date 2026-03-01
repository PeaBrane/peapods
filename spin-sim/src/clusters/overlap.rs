use super::utils::{
    bfs_cluster, find, find_seed, top4_sizes, uf_bonds, uf_flatten_counts, uf_histogram,
};
use crate::geometry::Lattice;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;

/// Overlap cluster update (Houdayer-N, Jörg, or CMR), parallelized over groups.
///
/// For each temperature, shuffles replicas into random groups of `group_size`
/// (2 for Jörg/CMR, N for Houdayer-N). For each group, grows a cluster on
/// the overlap subgraph and flips all replicas in the group on cluster sites.
///
/// Houdayer-N uses an isoenergetic criterion: a site is "active" (balanced)
/// when the spin sum across the group is zero. For group_size=2 this reduces
/// to the standard negative overlap σ₁ ≠ σ₂. For group_size=2k, it requires
/// a k-k split. Flipping all replicas preserves the balanced multiset, so the
/// move is isoenergetic — no FK decomposition needed.
///
/// `rngs` has length `n_temps * n_pairs` (one RNG per pair slot, positional).
/// The first slot's RNG at each temperature is also used for the shuffle.
/// Since `n_groups = n_replicas / group_size <= n_pairs`, the array is always
/// large enough.
///
/// When `stochastic` is false (Houdayer-N), bonds are deterministic (prob=1).
/// When `stochastic` is true (Jörg/CMR), bond activation is
/// `p = 1 - exp(-4 * J * σ_i * σ_j / T)` on satisfied bonds (group_size=2).
///
/// When `restrict_to_negative` is true (Houdayer-N/Jörg), only active
/// (balanced) sites are eligible. When false (CMR), all same-sign overlap
/// sites are bonded.
///
/// When `wolff` is true, uses BFS single-cluster (one seed per group).
/// When `wolff` is false, uses union-find global decomposition and flips
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
                if restrict_to_negative {
                    if !is_active(i) || !is_active(j) {
                        return false;
                    }
                } else if is_active(i) != is_active(j) {
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
            });

            if wolff {
                let Some(seed) = find_seed(n_spins, rng, |i| {
                    if restrict_to_negative {
                        is_active(i)
                    } else {
                        true
                    }
                }) else {
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

                for (i, &p) in parent.iter().enumerate().take(n_spins) {
                    if counts[p as usize] > 1 {
                        for &base in &bases {
                            *sp_ptr.add(base + i) *= -1;
                        }
                    }
                }
            }
        } else {
            // Pure Wolff without CSD/top4
            let Some(seed) = find_seed(n_spins, rng, |i| {
                if restrict_to_negative {
                    is_active(i)
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
                    if restrict_to_negative {
                        if !is_active(nb) {
                            return false;
                        }
                    } else if is_active(site) != is_active(nb) {
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
                },
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
