use crate::lattice::Lattice;
use crate::parallel::par_over_replicas;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;

// --- Union-Find ---

#[inline]
fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

#[inline]
fn union(parent: &mut [u32], rank: &mut [u8], x: u32, y: u32) {
    let rx = find(parent, x);
    let ry = find(parent, y);
    if rx == ry {
        return;
    }
    if rank[rx as usize] < rank[ry as usize] {
        parent[rx as usize] = ry;
    } else {
        parent[ry as usize] = rx;
        if rank[rx as usize] == rank[ry as usize] {
            rank[rx as usize] += 1;
        }
    }
}

// --- Generic cluster helpers ---
//
// `bfs_cluster` and `uf_bonds` factor out the two cluster-building patterns
// (BFS single-cluster and union-find global decomposition). Each takes a
// closure for the algorithm-specific activation logic. The closure is
// `impl FnMut`, so it is monomorphized at each call site — zero overhead
// vs hand-inlined code. The closure can capture mutable state (e.g. an RNG
// for probabilistic bond activation).
//
// To build a new cluster algorithm, use `find_seed` for seed selection and
// `bfs_cluster` for growth. For example, a Jörg cluster (negative-overlap
// sites + probabilistic bond activation) as a BFS single-cluster:
//
//   let Some(seed) = find_seed(n_spins, rng, |i| {
//       spins[base_a + i] != spins[base_b + i]
//   }) else { continue; };
//   bfs_cluster(lattice, seed, &mut in_cluster, &mut stack,
//       |site, nb, d, fwd| {
//           if spins[base_a + nb] == spins[base_b + nb] { return false; }
//           let coupling = if fwd { couplings[site * n_dims + d] }
//                          else   { couplings[nb * n_dims + d] };
//           let inter = spins[base_a + site] as f32 * spins[base_a + nb] as f32 * coupling;
//           if inter <= 0.0 { return false; }
//           rng.gen::<f32>() < 1.0 - (-4.0 * inter / temp).exp()
//       },
//   );
//
// Or equivalently as a global SW-style decomposition via `uf_bonds`:
//
//   let (mut parent, _) = uf_bonds(lattice, |i, d| {
//       let j = lattice.neighbor(i, d, true);
//       if spins[base_a + i] == spins[base_b + i]
//       || spins[base_a + j] == spins[base_b + j] { return false; }
//       let inter = spins[base_a + i] as f32 * spins[base_a + j] as f32
//                 * couplings[i * n_dims + d];
//       if inter <= 0.0 { return false; }
//       rng.gen::<f32>() < 1.0 - (-4.0 * inter / temp).exp()
//   }, None);  // pass Some(&mut sizes) to collect CSD

/// Find an eligible seed site via 64 random probes.
///
/// Returns `None` when no probe passes `eligible`. This is a fast
/// probabilistic search — if eligible sites are rare it may miss them.
#[inline]
fn find_seed(
    n_spins: usize,
    rng: &mut Xoshiro256StarStar,
    mut eligible: impl FnMut(usize) -> bool,
) -> Option<usize> {
    for _ in 0..64 {
        let i = rng.gen_range(0..n_spins);
        if eligible(i) {
            return Some(i);
        }
    }
    None
}

/// Grow a BFS cluster from `seed`. `should_add(site, neighbor, dim, forward)`
/// decides whether to add each not-yet-visited neighbor.
/// Caller owns buffers: `in_cluster` must be all-false, `stack` must be empty.
#[inline]
fn bfs_cluster(
    lattice: &Lattice,
    seed: usize,
    in_cluster: &mut [bool],
    stack: &mut Vec<usize>,
    mut should_add: impl FnMut(usize, usize, usize, bool) -> bool,
) {
    in_cluster[seed] = true;
    stack.push(seed);

    while let Some(site) = stack.pop() {
        for d in 0..lattice.n_dims {
            for &fwd in &[true, false] {
                let nb = lattice.neighbor(site, d, fwd);
                if !in_cluster[nb] && should_add(site, nb, d, fwd) {
                    in_cluster[nb] = true;
                    stack.push(nb);
                }
            }
        }
    }
}

/// Activate forward bonds via union-find. `should_bond(site, dim)` decides
/// whether to activate the bond from `site` to its forward neighbor in `dim`.
/// Returns `(parent, rank)`. When `csd` is `Some`, parent is flattened in-place
/// and sorted cluster sizes (descending) are appended to the vec.
#[inline]
fn uf_bonds(
    lattice: &Lattice,
    mut should_bond: impl FnMut(usize, usize) -> bool,
    csd: Option<&mut Vec<usize>>,
) -> (Vec<u32>, Vec<u8>) {
    let n_spins = lattice.n_spins;
    let mut parent: Vec<u32> = (0..n_spins as u32).collect();
    let mut rank = vec![0u8; n_spins];

    for i in 0..n_spins {
        for d in 0..lattice.n_dims {
            if should_bond(i, d) {
                let j = lattice.neighbor(i, d, true);
                union(&mut parent, &mut rank, i as u32, j as u32);
            }
        }
    }

    if let Some(sizes) = csd {
        for i in 0..n_spins {
            parent[i] = find(&mut parent, i as u32);
        }
        let mut counts = vec![0usize; n_spins];
        for i in 0..n_spins {
            counts[parent[i] as usize] += 1;
        }
        let mut cluster_sizes: Vec<usize> = counts.into_iter().filter(|&c| c > 0).collect();
        cluster_sizes.sort_unstable_by(|a, b| b.cmp(a));
        sizes.extend(cluster_sizes);
    }

    (parent, rank)
}

// --- Public cluster updates ---

/// Swendsen-Wang cluster update, parallelized over replicas.
///
/// When `csd_out` is `Some`, FK cluster sizes (sorted descending) are written
/// into the pre-allocated per-system slots. The slice length must equal the
/// number of systems (i.e. `system_ids.len()`); each vec is *appended to*,
/// so the caller should clear them beforehand if a fresh collection is wanted.
pub fn sw_update(
    lattice: &Lattice,
    spins: &mut [i8],
    interactions: &[f32],
    temperatures: &[f32],
    system_ids: &[usize],
    rngs: &mut [Xoshiro256StarStar],
    csd_out: Option<&mut [Vec<usize>]>,
) {
    let n_spins = lattice.n_spins;
    let n_dims = lattice.n_dims;

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

        let inter_base = system_id * n_spins * n_dims;

        let (mut parent, _) = uf_bonds(
            lattice,
            |i, d| {
                let inter = interactions[inter_base + i * n_dims + d];
                if inter <= 0.0 {
                    return false;
                }
                let p = 1.0 - (-2.0 * inter / temp).exp();
                rng.gen::<f32>() < p
            },
            csd_slot,
        );

        if !has_csd {
            for i in 0..n_spins {
                parent[i] = find(&mut parent, i as u32);
            }
        }

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
    });
}

/// Wolff single-cluster update, parallelized over replicas.
pub fn wolff_update(
    lattice: &Lattice,
    spins: &mut [i8],
    couplings: &[f32],
    temperatures: &[f32],
    system_ids: &[usize],
    rngs: &mut [Xoshiro256StarStar],
) {
    let n_spins = lattice.n_spins;
    let n_dims = lattice.n_dims;

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
                    let interaction = spin_slice[site] as f32 * spin_slice[nb] as f32 * coupling;
                    if interaction <= 0.0 {
                        return false;
                    }
                    let p = 1.0 - (-2.0 * interaction / temp).exp();
                    rng.gen::<f32>() < p
                },
            );

            for i in 0..n_spins {
                if in_cluster[i] {
                    spin_slice[i] = -spin_slice[i];
                }
            }
        },
    );
}

/// Jörg cluster move — temperature-dependent variant of Houdayer ICM.
///
/// Like Houdayer, pairs replicas at each temperature, seeds on negative-overlap
/// sites, and swaps the grown cluster. Unlike Houdayer, bond activation is
/// stochastic: `p = 1 - exp(-4 * J * σ_i * σ_j / T)` on satisfied bonds
/// between negative-overlap sites. The factor of 4 (vs Wolff's 2) accounts for
/// both replicas changing simultaneously at boundary bonds.
#[allow(clippy::too_many_arguments)]
pub fn jorg_update(
    lattice: &Lattice,
    spins: &mut [i8],
    couplings: &[f32],
    temperatures: &[f32],
    system_ids: &[usize],
    n_replicas: usize,
    n_temps: usize,
    rng: &mut Xoshiro256StarStar,
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

            let Some(seed) = find_seed(n_spins, rng, |i| spins[base_a + i] != spins[base_b + i])
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
                    let coupling = if fwd {
                        couplings[site * n_dims + d]
                    } else {
                        couplings[nb * n_dims + d]
                    };
                    let inter = spins[base_a + site] as f32 * spins[base_a + nb] as f32 * coupling;
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
        }
    }
}

/// Houdayer isoenergetic cluster move (ICM).
///
/// For each temperature, shuffle replicas into random pairs. For each pair,
/// identify negative-overlap sites (where spins disagree), grow a BFS
/// cluster on that subgraph (prob=1), and exchange the cluster between replicas.
/// Preserves each replica's energy → always accepted.
pub fn houdayer_update(
    lattice: &Lattice,
    spins: &mut [i8],
    system_ids: &[usize],
    n_replicas: usize,
    n_temps: usize,
    rng: &mut Xoshiro256StarStar,
) {
    let n_spins = lattice.n_spins;

    let mut in_cluster = vec![false; n_spins];
    let mut stack = Vec::with_capacity(n_spins);

    for t in 0..n_temps {
        let mut replica_systems: Vec<usize> = (0..n_replicas)
            .map(|k| system_ids[k * n_temps + t])
            .collect();
        replica_systems.shuffle(rng);

        for pair in replica_systems.chunks_exact(2) {
            let sys_a = pair[0];
            let sys_b = pair[1];
            let base_a = sys_a * n_spins;
            let base_b = sys_b * n_spins;

            let Some(seed) = find_seed(n_spins, rng, |i| spins[base_a + i] != spins[base_b + i])
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
                |_site, nb, _d, _fwd| spins[base_a + nb] != spins[base_b + nb],
            );

            for (i, &in_c) in in_cluster.iter().enumerate() {
                if in_c {
                    spins.swap(base_a + i, base_b + i);
                }
            }
        }
    }
}
