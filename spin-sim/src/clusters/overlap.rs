use super::utils::{bfs_cluster, find, find_seed, uf_bonds};
use crate::lattice::Lattice;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;

/// Overlap cluster update (Houdayer ICM or Jörg), single-threaded.
///
/// For each temperature, shuffles replicas into random pairs. For each pair,
/// grows a cluster on the negative-overlap subgraph and swaps it.
///
/// When `stochastic` is false (Houdayer), bonds are deterministic (prob=1).
/// When `stochastic` is true (Jörg), bond activation is
/// `p = 1 - exp(-4 * J * σ_i * σ_j / T)` on satisfied bonds.
///
/// When `wolff` is true, uses BFS single-cluster (one seed per pair).
/// When `wolff` is false, uses union-find global decomposition and swaps all
/// non-singleton clusters.
#[allow(clippy::too_many_arguments)]
pub fn overlap_update(
    lattice: &Lattice,
    spins: &mut [i8],
    couplings: &[f32],
    temperatures: &[f32],
    system_ids: &[usize],
    n_replicas: usize,
    n_temps: usize,
    rng: &mut Xoshiro256StarStar,
    stochastic: bool,
    wolff: bool,
) {
    let n_spins = lattice.n_spins;
    let n_dims = lattice.n_dims;

    let mut in_cluster = vec![false; n_spins];
    let mut stack = Vec::with_capacity(n_spins);

    for t in 0..n_temps {
        let temp = temperatures[t];
        let mut replica_systems: Vec<usize> = (0..n_replicas)
            .map(|k| system_ids[k * n_temps + t])
            .collect();
        replica_systems.shuffle(rng);

        for pair in replica_systems.chunks_exact(2) {
            let sys_a = pair[0];
            let sys_b = pair[1];
            let base_a = sys_a * n_spins;
            let base_b = sys_b * n_spins;

            if wolff {
                let Some(seed) =
                    find_seed(n_spins, rng, |i| spins[base_a + i] != spins[base_b + i])
                else {
                    continue;
                };

                in_cluster.fill(false);
                stack.clear();

                bfs_cluster(
                    lattice,
                    seed,
                    &mut in_cluster,
                    &mut stack,
                    |site, nb, d, fwd| {
                        if spins[base_a + nb] == spins[base_b + nb] {
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
                        let inter =
                            spins[base_a + site] as f32 * spins[base_a + nb] as f32 * coupling;
                        if inter <= 0.0 {
                            return false;
                        }
                        rng.gen::<f32>() < 1.0 - (-4.0 * inter / temp).exp()
                    },
                );

                for (i, &in_c) in in_cluster.iter().enumerate() {
                    if in_c {
                        spins.swap(base_a + i, base_b + i);
                    }
                }
            } else {
                let (mut parent, _) = uf_bonds(
                    lattice,
                    |i, d| {
                        let j = lattice.neighbor(i, d, true);
                        if spins[base_a + i] == spins[base_b + i]
                            || spins[base_a + j] == spins[base_b + j]
                        {
                            return false;
                        }
                        if !stochastic {
                            return true;
                        }
                        let inter = spins[base_a + i] as f32
                            * spins[base_a + j] as f32
                            * couplings[i * n_dims + d];
                        if inter <= 0.0 {
                            return false;
                        }
                        rng.gen::<f32>() < 1.0 - (-4.0 * inter / temp).exp()
                    },
                    None,
                );

                for i in 0..n_spins {
                    parent[i] = find(&mut parent, i as u32);
                }
                let mut counts = vec![0usize; n_spins];
                for i in 0..n_spins {
                    counts[parent[i] as usize] += 1;
                }
                for i in 0..n_spins {
                    if counts[parent[i] as usize] > 1 {
                        spins.swap(base_a + i, base_b + i);
                    }
                }
            }
        }
    }
}
