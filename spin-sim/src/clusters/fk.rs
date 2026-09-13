use super::utils::{
    dfs_cluster, uf_bonds_fresh, uf_bonds_fresh_with, uf_flatten, uf_flatten_counts_fresh,
    uf_histogram, BondMetrics, GraphObservationSlot,
};
use crate::config::ClusterAction;
use crate::geometry::Lattice;
use crate::parallel::par_over_replicas;
use crate::spins::model::Spin;
use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;

/// Fortuin-Kasteleyn cluster update (SW or Wolff).
///
/// When `wolff` is false, performs Swendsen-Wang (flip each cluster with p=0.5).
/// When `wolff` is true, performs Wolff (flip only the seed's cluster).
///
/// Uses a BFS fast path when `wolff && csd_out.is_none()`. Otherwise uses
/// union-find, computing interactions on-the-fly from `couplings`.
///
/// When `csd_out` is `Some`, cluster sizes are histogrammed into the
/// pre-allocated per-system slots (`hist[s]` += 1 for each cluster of size
/// `s`). The slice length must equal the number of systems (i.e.
/// `system_ids.len()`); each inner vec must be pre-sized to `n_spins + 1`.
///
/// When `sequential` is true, replicas are processed on the current thread.
#[cfg_attr(feature = "profile", inline(never))]
#[allow(clippy::too_many_arguments)]
pub fn fk_update(
    lattice: &Lattice,
    spins: &mut [i8],
    couplings: &[f32],
    temperatures: &[f32],
    system_ids: &[usize],
    rngs: &mut [Xoshiro256StarStar],
    wolff: bool,
    action: ClusterAction,
    csd_out: Option<&mut [Vec<u64>]>,
    observation_out: Option<&mut [GraphObservationSlot]>,
    sequential: bool,
) {
    embedded_update::<IsingEmbedding>(
        lattice,
        spins,
        couplings,
        temperatures,
        system_ids,
        rngs,
        wolff,
        action,
        csd_out,
        observation_out,
        sequential,
        None,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn embedded_update<E: Embedding>(
    lattice: &Lattice,
    spins: &mut [E::Spin],
    couplings: &[<E::Spin as Spin>::Value],
    temperatures: &[<E::Spin as Spin>::Value],
    system_ids: &[usize],
    rngs: &mut [Xoshiro256StarStar],
    wolff: bool,
    action: ClusterAction,
    mut csd_out: Option<&mut [Vec<u64>]>,
    mut observation_out: Option<&mut [GraphObservationSlot]>,
    sequential: bool,
    mut visited_out: Option<&mut [u64]>,
) {
    let n_spins = lattice.n_spins;
    let n_neighbors = lattice.n_neighbors;
    let vp = visited_out.as_mut().map_or(0, |v| v.as_mut_ptr() as usize);

    // BFS fast path: Wolff without CSD collection
    if action == ClusterAction::Update && wolff && csd_out.is_none() {
        par_over_replicas(
            spins,
            rngs,
            temperatures,
            system_ids,
            n_spins,
            sequential,
            |spin_slice, rng, temp, temp_id, _| {
                let axis = E::axis(rng);
                let seed = rng.gen_range(0..n_spins);
                let mut in_cluster = vec![false; n_spins];
                let mut stack = Vec::with_capacity(n_spins);

                dfs_cluster(
                    lattice,
                    seed,
                    &mut in_cluster,
                    &mut stack,
                    |site, nb, d, fwd| {
                        let coupling = if fwd {
                            couplings[site * n_neighbors + d]
                        } else {
                            couplings[nb * n_neighbors + d]
                        };
                        E::bond(spin_slice[site], spin_slice[nb], coupling, temp, &axis, rng)
                    },
                );

                let mut visited = 0;
                for i in 0..n_spins {
                    if in_cluster[i] {
                        E::reflect(&mut spin_slice[i], &axis);
                        visited += 1;
                    }
                }
                if vp != 0 {
                    unsafe {
                        *(vp as *mut u64).add(temp_id) += visited;
                    }
                }
            },
        );
        return;
    }

    // UF path: SW, or Wolff + CSD
    let sp = spins.as_mut_ptr() as usize;
    let rp = rngs.as_mut_ptr() as usize;
    let cp = csd_out
        .as_mut()
        .map(|s| s.as_mut_ptr() as usize)
        .unwrap_or(0);
    let has_csd = csd_out.is_some();
    let op = observation_out
        .as_mut()
        .map(|s| s.as_mut_ptr() as usize)
        .unwrap_or(0);
    let has_observation = observation_out.is_some();

    let work = |temp_id: usize| unsafe {
        let system_id = system_ids[temp_id];
        let spin_slice =
            std::slice::from_raw_parts_mut((sp as *mut E::Spin).add(system_id * n_spins), n_spins);
        let rng = &mut *(rp as *mut Xoshiro256StarStar).add(system_id);
        let temp = temperatures[temp_id];
        let axis = E::axis(rng);

        let mut should_bond = |i: usize, d: usize| {
            let j = lattice.neighbor_fwd(i, d);
            E::bond(
                spin_slice[i],
                spin_slice[j],
                couplings[i * n_neighbors + d],
                temp,
                &axis,
                rng,
            )
        };

        // Fresh storage is intentional: pooling regressed FK/SW throughput.
        let mut metrics = has_observation.then(|| BondMetrics::new(lattice));
        let (mut parent, mut scratch) = if let Some(ref mut metrics) = metrics {
            uf_bonds_fresh_with(lattice, &mut should_bond, |site, dim| {
                metrics.record_bond(lattice, site, dim);
            })
        } else {
            uf_bonds_fresh(lattice, &mut should_bond)
        };

        if has_csd || has_observation {
            let counts = uf_flatten_counts_fresh(&mut parent);
            if has_csd {
                let csd_slot = &mut *(cp as *mut Vec<u64>).add(temp_id);
                uf_histogram(&counts, csd_slot.as_mut_slice());
            }
            if let Some(metrics) = metrics {
                let observation_slot = &mut *(op as *mut GraphObservationSlot).add(temp_id);
                *observation_slot = metrics.finish(&counts);
            }
        } else {
            uf_flatten(&mut parent);
        }

        if action == ClusterAction::Observe {
            return;
        }

        let mut visited = 0;
        if wolff {
            let seed = rng.gen_range(0..n_spins);
            let seed_root = parent[seed];
            for (&site_parent, spin) in parent.iter().zip(spin_slice.iter_mut()) {
                if site_parent == seed_root {
                    E::reflect(spin, &axis);
                    visited += 1;
                }
            }
        } else {
            scratch.fill(2); // 2 = undecided
            for (&site_parent, spin) in parent.iter().zip(spin_slice.iter_mut()) {
                let root = site_parent as usize;
                if scratch[root] == 2 {
                    scratch[root] = u8::from(E::coin(rng));
                }
                if scratch[root] == 1 {
                    E::reflect(spin, &axis);
                    visited += 1;
                }
            }
        }
        if vp != 0 {
            *(vp as *mut u64).add(temp_id) += if wolff { visited } else { n_spins as u64 };
        }
    };

    if sequential {
        (0..system_ids.len()).for_each(work);
    } else {
        (0..system_ids.len()).into_par_iter().for_each(work);
    }
}

/// Embedded Ising bonds, with model-specific arithmetic and reflection.
pub(crate) trait Embedding {
    type Spin: Spin;
    type Axis;
    fn axis(rng: &mut Xoshiro256StarStar) -> Self::Axis;
    fn bond(
        a: Self::Spin,
        b: Self::Spin,
        j: <Self::Spin as Spin>::Value,
        temperature: <Self::Spin as Spin>::Value,
        axis: &Self::Axis,
        rng: &mut Xoshiro256StarStar,
    ) -> bool;
    fn reflect(spin: &mut Self::Spin, axis: &Self::Axis);
    fn coin(rng: &mut Xoshiro256StarStar) -> bool;
}
struct IsingEmbedding;
impl Embedding for IsingEmbedding {
    type Spin = i8;
    type Axis = ();
    fn axis(_: &mut Xoshiro256StarStar) {}
    fn bond(a: i8, b: i8, j: f32, t: f32, _: &(), rng: &mut Xoshiro256StarStar) -> bool {
        let interaction = a as f32 * b as f32 * j;
        interaction > 0.0 && rng.gen::<f32>() < 1.0 - (-2.0 * interaction / t).exp()
    }
    fn reflect(spin: &mut i8, _: &()) {
        *spin = -*spin;
    }
    fn coin(rng: &mut Xoshiro256StarStar) -> bool {
        rng.gen::<f32>() < 0.5
    }
}
