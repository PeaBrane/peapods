use super::utils::{dfs_cluster, uf_bonds, uf_flatten};
use crate::config::ClusterMode;
use crate::geometry::Lattice;
use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;

#[inline]
fn metropolis(delta: f64, rng: &mut Xoshiro256StarStar) -> bool {
    delta <= 0.0 || rng.gen::<f64>() < (-delta).exp()
}

/// Replica Monte Carlo (Swendsen & Wang, PRL 57, 2607 (1986)) between the systems at
/// adjacent temperature slots `t` and `t + 1` of every replica row.
///
/// With τ_i = s^a_i s^b_i held fixed, β_a H(s^a) + β_b H(s^b) is an Ising model in
/// s^a with couplings J (β_a + β_b τ_i τ_j). Negating both replicas on a connected
/// τ-domain keeps τ, and with it the domain decomposition, unchanged and alters only
/// the boundary bonds, whose coupling is J (β_a - β_b). Each flip is accepted with
/// the Metropolis probability of that boundary energy.
///
/// `Sw` updates every τ-domain. Adjacent domains carry opposite τ, so all τ = +1
/// domains are conditionally independent given the τ = -1 ones: they are decided
/// together, then the τ = -1 domains on the updated boundary. `Wolff` proposes the
/// domain of one uniformly random site. Edges `(0, 1), (2, 3), ...` are updated before
/// `(1, 2), (3, 4), ...`; each task draws from the generator of its lower slot.
#[cfg_attr(feature = "profile", inline(never))]
#[allow(clippy::too_many_arguments)]
pub fn rmc_update(
    lattice: &Lattice,
    spins: &mut [i8],
    couplings: &[f32],
    temperatures: &[f32],
    system_ids: &[usize],
    n_replicas: usize,
    n_temps: usize,
    active_temps: &[bool],
    rngs: &mut [Xoshiro256StarStar],
    cluster_mode: ClusterMode,
    sequential: bool,
) {
    let n_spins = lattice.n_spins;
    let n_neighbors = lattice.n_neighbors;
    let sp = spins.as_mut_ptr() as usize;
    let rp = rngs.as_mut_ptr() as usize;

    for parity in 0..2 {
        let n_edges = n_temps.saturating_sub(parity) / 2;
        let work = |task: usize| unsafe {
            let t = parity + 2 * (task % n_edges);
            if !(active_temps[t] && active_temps[t + 1]) {
                return;
            }
            let slot = (task / n_edges) * n_temps + t;
            let rng = &mut *(rp as *mut Xoshiro256StarStar).add(slot);
            let sp_ptr = sp as *mut i8;
            let base_a = system_ids[slot] * n_spins;
            let base_b = system_ids[slot + 1] * n_spins;
            let tau = |i: usize| *sp_ptr.add(base_a + i) * *sp_ptr.add(base_b + i);
            // Change of β_a H_a + β_b H_b per unit of J s^a_i s^a_j on a flipped boundary.
            let scale = 2.0
                * (1.0 / f64::from(temperatures[slot]) - 1.0 / f64::from(temperatures[slot + 1]));
            let boundary = |i: usize, j: usize, coupling: f32| {
                scale
                    * f64::from(coupling)
                    * f64::from(*sp_ptr.add(base_a + i) * *sp_ptr.add(base_a + j))
            };
            let flip = |i: usize| {
                *sp_ptr.add(base_a + i) *= -1;
                *sp_ptr.add(base_b + i) *= -1;
            };

            if cluster_mode == ClusterMode::Wolff {
                let seed = rng.gen_range(0..n_spins);
                let mut in_cluster = vec![false; n_spins];
                let mut stack = Vec::with_capacity(n_spins);
                dfs_cluster(
                    lattice,
                    seed,
                    &mut in_cluster,
                    &mut stack,
                    |site, nb, _d, _fwd| tau(site) == tau(nb),
                );
                let mut delta = 0.0;
                for i in (0..n_spins).filter(|&i| in_cluster[i]) {
                    for d in 0..n_neighbors {
                        let fwd = lattice.neighbor_fwd(i, d);
                        if !in_cluster[fwd] {
                            delta += boundary(i, fwd, couplings[i * n_neighbors + d]);
                        }
                        let bwd = lattice.neighbor_bwd(i, d);
                        if !in_cluster[bwd] {
                            delta += boundary(i, bwd, couplings[bwd * n_neighbors + d]);
                        }
                    }
                }
                if metropolis(delta, rng) {
                    (0..n_spins).filter(|&i| in_cluster[i]).for_each(flip);
                }
                return;
            }

            // τ, and with it the domains and their walls, is invariant under the move.
            let tau: Vec<i8> = (0..n_spins).map(tau).collect();
            let mut uf = uf_bonds(lattice, |i, d| tau[i] == tau[lattice.neighbor_fwd(i, d)]);
            let storage = &mut *uf;
            uf_flatten(&mut storage.parent);
            let mut walls = Vec::new();
            for i in 0..n_spins {
                for d in 0..n_neighbors {
                    let j = lattice.neighbor_fwd(i, d);
                    if tau[i] != tau[j] {
                        walls.push((i, j, couplings[i * n_neighbors + d]));
                    }
                }
            }
            let mut delta = vec![0.0f64; n_spins];
            for sign in [1i8, -1] {
                delta.fill(0.0);
                for &(i, j, coupling) in &walls {
                    let inside = if tau[i] == sign { i } else { j };
                    delta[storage.parent[inside] as usize] += boundary(i, j, coupling);
                }
                // `rank` holds each domain's decision: u8::MAX undecided, 1 flip.
                storage.rank.fill(u8::MAX);
                for (&p, &t) in storage.parent.iter().zip(&tau) {
                    let root = p as usize;
                    if t == sign && storage.rank[root] == u8::MAX {
                        storage.rank[root] = u8::from(metropolis(delta[root], rng));
                    }
                }
                for (i, &p) in storage.parent.iter().enumerate() {
                    if storage.rank[p as usize] == 1 {
                        flip(i);
                    }
                }
            }
        };

        let n_tasks = n_replicas * n_edges;
        if sequential {
            (0..n_tasks).for_each(work);
        } else {
            (0..n_tasks).into_par_iter().for_each(work);
        }
    }
}
