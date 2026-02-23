use super::utils::{bfs_cluster, find, uf_bonds};
use crate::parallel::par_over_replicas;
use crate::spins::Lattice;
use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;

/// Fortuin-Kasteleyn cluster update (SW or Wolff), parallelized over replicas.
///
/// When `wolff` is false, performs Swendsen-Wang (flip each cluster with p=0.5).
/// When `wolff` is true, performs Wolff (flip only the seed's cluster).
///
/// Uses a BFS fast path when `wolff && csd_out.is_none()`. Otherwise uses
/// union-find, computing interactions on-the-fly from `couplings`.
///
/// When `csd_out` is `Some`, FK cluster sizes (sorted descending) are written
/// into the pre-allocated per-system slots. The slice length must equal the
/// number of systems (i.e. `system_ids.len()`); each vec is *appended to*,
/// so the caller should clear them beforehand if a fresh collection is wanted.
#[allow(clippy::too_many_arguments)]
pub fn fk_update(
    lattice: &Lattice,
    spins: &mut [i8],
    couplings: &[f32],
    temperatures: &[f32],
    system_ids: &[usize],
    rngs: &mut [Xoshiro256StarStar],
    wolff: bool,
    csd_out: Option<&mut [Vec<usize>]>,
) {
    let n_spins = lattice.n_spins;
    let n_dims = lattice.n_dims;

    // BFS fast path: Wolff without CSD collection
    if wolff && csd_out.is_none() {
        par_over_replicas(
            spins,
            rngs,
            temperatures,
            system_ids,
            n_spins,
            |spin_slice, rng, temp, _| {
                let seed = rng.gen_range(0..n_spins);
                let mut in_cluster = vec![false; n_spins];
                let mut stack = Vec::with_capacity(n_spins);

                bfs_cluster(
                    lattice,
                    seed,
                    &mut in_cluster,
                    &mut stack,
                    |site, nb, d, fwd| {
                        let coupling = if fwd {
                            couplings[site * n_dims + d]
                        } else {
                            couplings[nb * n_dims + d]
                        };
                        let interaction =
                            spin_slice[site] as f32 * spin_slice[nb] as f32 * coupling;
                        if interaction <= 0.0 {
                            return false;
                        }
                        rng.gen::<f32>() < 1.0 - (-2.0 * interaction / temp).exp()
                    },
                );

                for i in 0..n_spins {
                    if in_cluster[i] {
                        spin_slice[i] = -spin_slice[i];
                    }
                }
            },
        );
        return;
    }

    // UF path: SW, or Wolff + CSD
    let chunks: Vec<(usize, usize)> = (0..system_ids.len())
        .map(|temp_id| (system_ids[temp_id], temp_id))
        .collect();

    let sp = spins.as_mut_ptr() as usize;
    let rp = rngs.as_mut_ptr() as usize;
    let cp = csd_out.as_ref().map(|s| s.as_ptr() as usize).unwrap_or(0);
    let has_csd = csd_out.is_some();

    chunks.par_iter().for_each(|&(system_id, temp_id)| unsafe {
        let spin_slice =
            std::slice::from_raw_parts_mut((sp as *mut i8).add(system_id * n_spins), n_spins);
        let rng = &mut *(rp as *mut Xoshiro256StarStar).add(system_id);
        let temp = temperatures[temp_id];

        let csd_slot = if has_csd {
            Some(&mut *(cp as *mut Vec<usize>).add(temp_id))
        } else {
            None
        };

        let (mut parent, _) = uf_bonds(
            lattice,
            |i, d| {
                let j = lattice.neighbor(i, d, true);
                let inter = spin_slice[i] as f32 * spin_slice[j] as f32 * couplings[i * n_dims + d];
                if inter <= 0.0 {
                    return false;
                }
                rng.gen::<f32>() < 1.0 - (-2.0 * inter / temp).exp()
            },
            csd_slot,
        );

        if !has_csd {
            for i in 0..n_spins {
                parent[i] = find(&mut parent, i as u32);
            }
        }

        if wolff {
            let seed = rng.gen_range(0..n_spins);
            let seed_root = parent[seed];
            for i in 0..n_spins {
                if parent[i] == seed_root {
                    spin_slice[i] = -spin_slice[i];
                }
            }
        } else {
            let mut flip_decision = vec![2u8; n_spins]; // 2 = undecided
            for i in 0..n_spins {
                let root = parent[i] as usize;
                if flip_decision[root] == 2 {
                    flip_decision[root] = u8::from(rng.gen::<f32>() < 0.5);
                }
                if flip_decision[root] == 1 {
                    spin_slice[i] = -spin_slice[i];
                }
            }
        }
    });
}
