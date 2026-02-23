use super::utils::{bfs_cluster, find, find_seed, uf_bonds};
use crate::lattice::Lattice;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;

/// Overlap cluster update (Houdayer ICM, Jörg, or CMR), parallelized over pairs.
///
/// For each temperature, shuffles replicas into random pairs. For each pair,
/// grows a cluster on the overlap subgraph and swaps it.
///
/// `rngs` has length `n_temps * n_pairs` (one RNG per pair slot, positional).
/// The first pair slot's RNG at each temperature is also used for the shuffle.
///
/// When `stochastic` is false (Houdayer), bonds are deterministic (prob=1).
/// When `stochastic` is true (Jörg/CMR), bond activation is
/// `p = 1 - exp(-4 * J * σ_i * σ_j / T)` on satisfied bonds.
///
/// When `restrict_to_negative` is true (Houdayer/Jörg), only negative-overlap
/// sites are eligible. When false (CMR), all same-sign overlap sites are bonded.
///
/// When `wolff` is true, uses BFS single-cluster (one seed per pair).
/// When `wolff` is false, uses union-find global decomposition and swaps all
/// non-singleton clusters. CSD collection forces UF even when `wolff` is true.
///
/// When `csd_out` is `Some`, forces UF path and writes per-pair cluster size
/// distributions. Slice length must be `n_temps * n_pairs`, indexed by
/// `t * n_pairs + p`.
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
    csd_out: Option<&mut [Vec<usize>]>,
) {
    let n_spins = lattice.n_spins;
    let n_dims = lattice.n_dims;
    let n_pairs = n_replicas / 2;

    // Phase 1 (sequential): determine all pairs via per-temperature shuffle
    let mut tasks: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(n_temps * n_pairs);
    for t in 0..n_temps {
        let mut replica_systems: Vec<usize> = (0..n_replicas)
            .map(|k| system_ids[k * n_temps + t])
            .collect();
        replica_systems.shuffle(&mut rngs[t * n_pairs]);
        for (p, pair) in replica_systems.chunks_exact(2).enumerate() {
            tasks.push((t, p, pair[0], pair[1]));
        }
    }

    // Phase 2 (parallel): process all pairs
    let sp = spins.as_mut_ptr() as usize;
    let rp = rngs.as_mut_ptr() as usize;
    let use_uf = !wolff || csd_out.is_some();

    let cp = csd_out.as_ref().map(|s| s.as_ptr() as usize).unwrap_or(0);
    let has_csd = csd_out.is_some();

    tasks.par_iter().for_each(|&(t, p, sys_a, sys_b)| unsafe {
        let rng = &mut *(rp as *mut Xoshiro256StarStar).add(t * n_pairs + p);
        let temp = temperatures[t];
        let base_a = sys_a * n_spins;
        let base_b = sys_b * n_spins;
        let sp_ptr = sp as *mut i8;

        let overlap_sign =
            |i: usize| -> bool { *sp_ptr.add(base_a + i) != *sp_ptr.add(base_b + i) };

        if use_uf {
            let csd_slot = if has_csd {
                Some(&mut *(cp as *mut Vec<usize>).add(t * n_pairs + p))
            } else {
                None
            };

            let (mut parent, _) = uf_bonds(
                lattice,
                |i, d| {
                    let j = lattice.neighbor(i, d, true);
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
                        * couplings[i * n_dims + d];
                    if inter <= 0.0 {
                        return false;
                    }
                    rng.gen::<f32>() < 1.0 - (-4.0 * inter / temp).exp()
                },
                csd_slot,
            );

            if wolff {
                // Wolff + CSD: UF decomposition already done, swap only seed's cluster
                let Some(seed) = find_seed(n_spins, rng, |i| {
                    if restrict_to_negative {
                        overlap_sign(i)
                    } else {
                        true
                    }
                }) else {
                    return;
                };
                let seed_root = find(&mut parent, seed as u32);
                for i in 0..n_spins {
                    if find(&mut parent, i as u32) == seed_root {
                        std::ptr::swap(sp_ptr.add(base_a + i), sp_ptr.add(base_b + i));
                    }
                }
            } else {
                for i in 0..n_spins {
                    parent[i] = find(&mut parent, i as u32);
                }
                let mut counts = vec![0usize; n_spins];
                for i in 0..n_spins {
                    counts[parent[i] as usize] += 1;
                }
                for i in 0..n_spins {
                    if counts[parent[i] as usize] > 1 {
                        std::ptr::swap(sp_ptr.add(base_a + i), sp_ptr.add(base_b + i));
                    }
                }
            }
        } else {
            // Pure Wolff without CSD
            let Some(seed) = find_seed(n_spins, rng, |i| {
                if restrict_to_negative {
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
                        couplings[site * n_dims + d]
                    } else {
                        couplings[nb * n_dims + d]
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
                    std::ptr::swap(sp_ptr.add(base_a + i), sp_ptr.add(base_b + i));
                }
            }
        }
    });
}
