use super::utils::{
    build_graph, decide_clusters, dfs_cluster, find_seed, top4_sizes, uf_bonds_extend,
    uf_flatten_counts, uf_histogram, BondMetrics, BondSampler, GraphObservationSlot, UNDECIDED,
};
use crate::config::{ClusterAction, ClusterMode, OverlapClusterBuildMode, MAX_PERM_GROUP};
use crate::geometry::Lattice;
use crate::parallel::{for_each_task, Slots};
use rand::seq::SliceRandom;
use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;

/// Which sites of a replica group the deterministic overlap moves act on.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ActiveRule {
    /// `houdN`: sites where the group's spins sum to zero (not detailed-balanced for N > 2).
    Balanced,
    /// `pairN`: sites where any pair `(g0 g1)(g2 g3)...` disagrees.
    Pairwise,
    /// `permN`: sites where the group does not agree.
    Permutation,
    /// `gpermN`: sites whose pattern differs from a random reference site's up to sign.
    GaugedPermutation,
}

struct GroupTasks {
    systems: Vec<usize>,
    /// Temperature slots that receive moves.
    temps: Vec<usize>,
    n_replicas: usize,
    n_groups: usize,
    group_size: usize,
}

impl GroupTasks {
    #[inline]
    fn len(&self) -> usize {
        self.temps.len() * self.n_groups
    }

    #[inline]
    fn group(&self, task_idx: usize) -> (usize, usize, &[usize]) {
        let t = self.temps[task_idx / self.n_groups];
        let g = task_idx % self.n_groups;
        let start = t * self.n_replicas + g * self.group_size;
        (t, g, &self.systems[start..start + self.group_size])
    }
}

/// Build shuffled per-temperature group assignments.
fn build_tasks(
    system_ids: &[usize],
    n_replicas: usize,
    n_temps: usize,
    group_size: usize,
    rngs: &mut [Xoshiro256StarStar],
    n_pairs: usize,
    active_temps: &[bool],
) -> GroupTasks {
    let n_groups = n_replicas / group_size;
    let mut systems = Vec::with_capacity(n_temps * n_replicas);
    for t in 0..n_temps {
        let start = systems.len();
        systems.extend((0..n_replicas).map(|k| system_ids[k * n_temps + t]));
        systems[start..].shuffle(&mut rngs[t * n_pairs]);
    }
    GroupTasks {
        systems,
        // Every temperature is still shuffled above, so masking keeps RNG use unchanged.
        temps: (0..n_temps).filter(|&t| active_temps[t]).collect(),
        n_replicas,
        n_groups,
        group_size,
    }
}

/// Everything an overlap move may publish, indexed by the task's slot
/// `t * n_pairs + g`.
struct SlotOutputs<'a> {
    csd: Slots<'a, Vec<u64>>,
    top4: Slots<'a, [u32; 4]>,
    observation: Slots<'a, GraphObservationSlot>,
    cluster_ids: Slots<'a, Vec<u32>>,
    blue_ids: Slots<'a, Vec<u32>>,
    spins: Slots<'a, Vec<[Vec<i8>; 2]>>,
    system_ids: Slots<'a, Vec<[usize; 2]>>,
}

impl SlotOutputs<'_> {
    /// Whether the move must build the full graph: Swendsen-Wang, an observation, or
    /// any graph statistic. Otherwise Wolff grows only the seed's cluster.
    fn needs_graph(&self, action: ClusterAction, wolff: bool) -> bool {
        action == ClusterAction::Observe
            || !wolff
            || self.csd.is_some()
            || self.top4.is_some()
            || self.cluster_ids.is_some()
    }

    /// Publishes the first two replicas of the first group at each temperature, as
    /// they are before the move.
    ///
    /// # Safety
    /// `slot` must belong to the calling task, and `systems` must index `sp_ptr`.
    unsafe fn snapshot_spins(
        &self,
        slot: usize,
        group: usize,
        sp_ptr: *const i8,
        systems: &[usize],
        n_spins: usize,
    ) {
        if group != 0 {
            return;
        }
        let (Some(spins), Some(ids)) = (self.spins.get(slot), self.system_ids.get(slot)) else {
            return;
        };
        let replica = |system: usize| {
            std::slice::from_raw_parts(sp_ptr.add(system * n_spins), n_spins).to_vec()
        };
        spins.push([replica(systems[0]), replica(systems[1])]);
        ids.push([systems[0], systems[1]]);
    }

    /// Records a built graph: its cluster-size histogram, four largest clusters,
    /// cluster ids (into `ids`) and bond metrics. With `active`, only clusters of
    /// active sites count: the sites a move leaves alone are not clusters of it.
    ///
    /// # Safety
    /// `slot` must belong to the calling task.
    unsafe fn publish_graph(
        &self,
        slot: usize,
        ids: &Slots<'_, Vec<u32>>,
        parent: &[u32],
        counts: &[u32],
        active: Option<&[bool]>,
        metrics: Option<BondMetrics>,
    ) {
        let sizes = counts
            .iter()
            .enumerate()
            .filter(|&(root, _)| active.is_none_or(|active| active[root]))
            .map(|(_, &size)| size);
        if let Some(csd) = self.csd.get(slot) {
            uf_histogram(sizes.clone(), csd);
        }
        if let Some(top4) = self.top4.get(slot) {
            *top4 = top4_sizes(sizes.clone());
        }
        if let Some(ids) = ids.get(slot) {
            ids.clear();
            ids.extend_from_slice(parent);
        }
        if let (Some(metrics), Some(observation)) = (metrics, self.observation.get(slot)) {
            *observation = metrics.finish(parent.len(), sizes);
        }
    }
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
    active_temps: &[bool],
    rngs: &mut [Xoshiro256StarStar],
    mode: &OverlapClusterBuildMode,
    cluster_mode: ClusterMode,
    action: ClusterAction,
    csd_out: Option<&mut [Vec<u64>]>,
    top4_out: Option<&mut [[u32; 4]]>,
    observation_out: Option<&mut [GraphObservationSlot]>,
    sequential: bool,
    snap_out: Option<&mut [Vec<u32>]>,
    blue_snap_out: Option<&mut [Vec<u32>]>,
    spin_snap_out: Option<&mut [Vec<[Vec<i8>; 2]>]>,
    sid_snap_out: Option<&mut [Vec<[usize; 2]>]>,
) {
    let n_spins = lattice.n_spins;
    // The tasks reach replicas through raw pointers, so the layout is checked here.
    assert!(
        system_ids.len() == n_replicas * n_temps
            && system_ids
                .iter()
                .all(|&id| (id + 1) * n_spins <= spins.len()),
        "overlap buffers disagree with the replica layout"
    );
    let outs = SlotOutputs {
        csd: Slots::new(csd_out),
        top4: Slots::new(top4_out),
        observation: Slots::new(observation_out),
        cluster_ids: Slots::new(snap_out),
        blue_ids: Slots::new(blue_snap_out),
        spins: Slots::new(spin_snap_out),
        system_ids: Slots::new(sid_snap_out),
    };
    let (group_size, rule) = match *mode {
        OverlapClusterBuildMode::Houdayer(n) => (n, ActiveRule::Balanced),
        OverlapClusterBuildMode::Pair(n) => (n, ActiveRule::Pairwise),
        OverlapClusterBuildMode::Perm(n) => (n, ActiveRule::Permutation),
        OverlapClusterBuildMode::GaugedPerm(n) => (n, ActiveRule::GaugedPermutation),
        OverlapClusterBuildMode::Jorg(group_size) => {
            return jorg_step(
                lattice,
                spins,
                couplings,
                temperatures,
                system_ids,
                n_replicas,
                n_temps,
                active_temps,
                rngs,
                group_size,
                cluster_mode,
                action,
                &outs,
                sequential,
            )
        }
        OverlapClusterBuildMode::Cmr => {
            return cmr_step(
                lattice,
                spins,
                couplings,
                temperatures,
                system_ids,
                n_replicas,
                n_temps,
                active_temps,
                rngs,
                cluster_mode,
                action,
                &outs,
                sequential,
            )
        }
        OverlapClusterBuildMode::Rmc => {
            unreachable!("rmc runs through rmc_update, which owns its task layout")
        }
    };
    isoenergetic_step(
        lattice,
        spins,
        system_ids,
        n_replicas,
        n_temps,
        active_temps,
        rngs,
        group_size,
        rule,
        cluster_mode,
        action,
        &outs,
        sequential,
    );
}

/// Whether any replica pair `(systems[2k], systems[2k+1])` disagrees at site `i`.
#[inline]
unsafe fn any_pair_differs(sp_ptr: *const i8, systems: &[usize], n_spins: usize, i: usize) -> bool {
    systems
        .as_chunks::<2>()
        .0
        .iter()
        .any(|pair| *sp_ptr.add(pair[0] * n_spins + i) != *sp_ptr.add(pair[1] * n_spins + i))
}

/// Acts on site `i` of every replica pair in `systems`. With `pairwise` each pair is
/// swapped, which negates both replicas where they disagree and leaves agreeing pairs
/// alone; otherwise every replica is negated.
#[inline]
unsafe fn flip_group_site(
    sp_ptr: *mut i8,
    systems: &[usize],
    n_spins: usize,
    i: usize,
    pairwise: bool,
) {
    for pair in systems.as_chunks::<2>().0 {
        let a = sp_ptr.add(pair[0] * n_spins + i);
        let b = sp_ptr.add(pair[1] * n_spins + i);
        if pairwise && *a == *b {
            continue;
        }
        *a = -*a;
        *b = -*b;
    }
}

/// Fills `out` with a uniformly random permutation of `0..out.len()`.
fn random_permutation(rng: &mut Xoshiro256StarStar, out: &mut [u8]) {
    for (k, v) in out.iter_mut().enumerate() {
        *v = k as u8;
    }
    out.shuffle(rng);
}

/// Fills `out` with a uniformly random permutation other than the identity.
fn random_non_identity(rng: &mut Xoshiro256StarStar, out: &mut [u8]) {
    loop {
        random_permutation(rng, out);
        if out.iter().enumerate().any(|(k, &v)| usize::from(v) != k) {
            return;
        }
    }
}

/// Permutes the gauged spins `gauge[k] s^k` of the group at site `i`: replica `k`
/// receives `gauge[k] gauge[perm[k]] s^perm[k]`. An all-ones gauge moves raw spins.
#[inline]
unsafe fn permute_group_site(
    sp_ptr: *mut i8,
    systems: &[usize],
    n_spins: usize,
    i: usize,
    perm: &[u8],
    gauge: &[i8],
) {
    let mut values = [0i8; MAX_PERM_GROUP];
    for ((value, &system), &g) in values.iter_mut().zip(systems).zip(gauge) {
        *value = g * *sp_ptr.add(system * n_spins + i);
    }
    for ((&system, &source), &g) in systems.iter().zip(perm).zip(gauge) {
        *sp_ptr.add(system * n_spins + i) = g * values[usize::from(source)];
    }
}

/// Houdayer-N isoenergetic overlap cluster update.
///
/// For each group of N replicas at a given temperature:
/// 1. Active sites: spin sum across all N replicas = 0 (balanced)
/// 2. Deterministic bonds (p=1) between pairs of active sites
/// 3. Flip all N replicas on cluster sites
///
/// With `pairwise` (the `pairN` mode) the group is split into pairs
/// `(g0 g1)(g2 g3)...` and the move applies the product of pair swaps `g` instead:
/// active sites are those where any pair disagrees, and a cluster flip swaps every
/// pair on it. Every neighbor of a connected active component is fixed by `g`, so the
/// swap conserves the summed energy exactly. For N = 2 both rules coincide.
///
/// With [`ActiveRule::Permutation`] (the `permN` mode) active sites are those where the
/// group does not agree. Every neighbor of an active component agrees, so every
/// replica permutation fixes it, and permuting the group's spins on the component
/// conserves the summed energy. The components are the same afterwards, so drawing
/// the permutation uniformly (SW: one per component, Wolff: one non-identity
/// permutation for the seed's component) is a heat bath over each component's orbit.
///
/// [`ActiveRule::GaugedPermutation`] (`gpermN`) first reads the pattern `P` of a
/// uniformly random reference site and permutes the gauged spins `P_k s^k` instead.
/// Sites with pattern `±P` are fixed by every gauged permutation and no other site
/// acquires that pattern, so their count, which is the probability of choosing this
/// gauge, is the same before and after the move.
#[allow(clippy::too_many_arguments)]
fn isoenergetic_step(
    lattice: &Lattice,
    spins: &mut [i8],
    system_ids: &[usize],
    n_replicas: usize,
    n_temps: usize,
    active_temps: &[bool],
    rngs: &mut [Xoshiro256StarStar],
    group_size: usize,
    rule: ActiveRule,
    cluster_mode: ClusterMode,
    action: ClusterAction,
    outs: &SlotOutputs<'_>,
    sequential: bool,
) {
    let n_spins = lattice.n_spins;
    let n_pairs = n_replicas / 2;
    let pairwise = rule == ActiveRule::Pairwise;
    let permutation = matches!(
        rule,
        ActiveRule::Permutation | ActiveRule::GaugedPermutation
    );
    let wolff = cluster_mode == ClusterMode::Wolff;

    let tasks = build_tasks(
        system_ids,
        n_replicas,
        n_temps,
        group_size,
        rngs,
        n_pairs,
        active_temps,
    );
    let use_uf = outs.needs_graph(action, wolff);
    let sp = spins.as_mut_ptr() as usize;
    let rngs = Slots::new(Some(rngs));

    // SAFETY: tasks act on disjoint replica groups, and task `(t, g)` alone uses the
    // generator and the output slot `t * n_pairs + g`.
    for_each_task(tasks.len(), sequential, |task_idx| unsafe {
        let (t, g, systems) = tasks.group(task_idx);
        let slot = t * n_pairs + g;
        let rng = rngs.at(slot);
        let sp_ptr = sp as *mut i8;
        outs.snapshot_spins(slot, g, sp_ptr, systems, n_spins);

        let mut gauge_store = [1i8; MAX_PERM_GROUP];
        if rule == ActiveRule::GaugedPermutation {
            let reference = rng.gen_range(0..n_spins);
            for (sign, &system) in gauge_store.iter_mut().zip(systems) {
                *sign = *sp_ptr.add(system * n_spins + reference);
            }
        }
        // Only the permutation rules read the gauge; pair and balanced groups may
        // exceed `MAX_PERM_GROUP`.
        let gauge: &[i8] = if permutation {
            &gauge_store[..systems.len()]
        } else {
            &[]
        };
        let is_active = |i: usize| -> bool {
            if pairwise {
                return any_pair_differs(sp_ptr, systems, n_spins, i);
            }
            if permutation {
                let first = gauge[0] * *sp_ptr.add(systems[0] * n_spins + i);
                return systems[1..]
                    .iter()
                    .zip(&gauge[1..])
                    .any(|(&system, &sign)| sign * *sp_ptr.add(system * n_spins + i) != first);
            }
            let mut sum: i32 = 0;
            for &system in systems {
                sum += *sp_ptr.add(system * n_spins + i) as i32;
            }
            sum == 0
        };
        // Acts on one site of the chosen cluster; `perm` matters only to the
        // permutation rules.
        let apply = |i: usize, perm: &[u8; MAX_PERM_GROUP]| {
            if permutation {
                permute_group_site(sp_ptr, systems, n_spins, i, &perm[..systems.len()], gauge);
                return;
            }
            flip_group_site(sp_ptr, systems, n_spins, i, pairwise);
        };
        let mut perm = [0u8; MAX_PERM_GROUP];

        if !use_uf {
            let Some(seed) = find_seed(n_spins, rng, &is_active) else {
                return;
            };
            let mut in_cluster = vec![false; n_spins];
            let mut stack = Vec::with_capacity(n_spins);
            dfs_cluster(
                lattice,
                seed,
                &mut in_cluster,
                &mut stack,
                // Every site already in the cluster is active.
                |_site, nb, _d, _fwd| is_active(nb),
            );
            if permutation {
                random_non_identity(rng, &mut perm[..group_size]);
            }
            for i in (0..n_spins).filter(|&i| in_cluster[i]) {
                apply(i, &perm);
            }
            return;
        }

        // The O(N) graph pass would otherwise re-sum the group for both endpoints
        // of every bond.
        let active: Vec<bool> = (0..n_spins).map(is_active).collect();
        let (mut uf, metrics) = build_graph(lattice, outs.observation.is_some(), |i, d| {
            active[i] && active[lattice.neighbor_fwd(i, d)]
        });

        // Statistics and snapshots describe the graph, so record them before the
        // Wolff seed search, which may find no active site.
        let counts = uf_flatten_counts(&mut uf.parent);
        outs.publish_graph(
            slot,
            &outs.cluster_ids,
            &uf.parent,
            &counts,
            Some(&active),
            metrics,
        );
        if action == ClusterAction::Observe {
            return;
        }

        if wolff {
            let Some(seed) = find_seed(n_spins, rng, |i| active[i]) else {
                return;
            };
            if permutation {
                random_non_identity(rng, &mut perm[..group_size]);
            }
            let seed_root = uf.parent[seed];
            for i in (0..n_spins).filter(|&i| uf.parent[i] == seed_root) {
                apply(i, &perm);
            }
            return;
        }
        if permutation {
            // One uniform permutation per component, singletons included.
            let mut perms = vec![UNDECIDED; n_spins * group_size];
            for i in (0..n_spins).filter(|&i| active[i]) {
                let root = uf.parent[i] as usize;
                let perm = &mut perms[root * group_size..(root + 1) * group_size];
                if perm[0] == UNDECIDED {
                    random_permutation(rng, perm);
                }
                permute_group_site(sp_ptr, systems, n_spins, i, perm, gauge);
            }
            return;
        }
        let storage = &mut *uf;
        decide_clusters(
            &storage.parent,
            &mut storage.rank,
            |_, root| counts[root] > 1,
            |_| u8::from(rng.gen::<f32>() < 0.5),
            |site, flip| {
                if flip == 1 {
                    flip_group_site(sp_ptr, systems, n_spins, site, pairwise);
                }
            },
        );
    });
}

/// Jörg stochastic overlap cluster update.
///
/// For each pair of replicas at a given temperature:
/// 1. Active sites: σ_i ≠ τ_i (negative overlap)
/// 2. Stochastic FK bonds on active sites: p = 1 - exp(-4 J σ_i σ_j / T)
/// 3. Flip both replicas on cluster sites
///
/// For a group of N > 2 replicas the move applies the product `g` of the pair swaps
/// `(g0 g1)(g2 g3)...` (Niedermayer / Kandel-Domany embedding): a bond is activated
/// with p = 1 - exp(-β max(0, ΔE)), where ΔE = E(σ_i, gσ_j) - E(σ_i, σ_j) =
/// 4 J Σ σ^a_i σ^a_j sums over the pairs `(a, b)` disagreeing at both endpoints, and
/// every pair is swapped on the flipped clusters.
#[allow(clippy::too_many_arguments)]
fn jorg_step(
    lattice: &Lattice,
    spins: &mut [i8],
    couplings: &[f32],
    temperatures: &[f32],
    system_ids: &[usize],
    n_replicas: usize,
    n_temps: usize,
    active_temps: &[bool],
    rngs: &mut [Xoshiro256StarStar],
    group_size: usize,
    cluster_mode: ClusterMode,
    action: ClusterAction,
    outs: &SlotOutputs<'_>,
    sequential: bool,
) {
    let n_spins = lattice.n_spins;
    let n_neighbors = lattice.n_neighbors;
    let n_pairs = n_replicas / 2;
    let wolff = cluster_mode == ClusterMode::Wolff;

    let tasks = build_tasks(
        system_ids,
        n_replicas,
        n_temps,
        group_size,
        rngs,
        n_pairs,
        active_temps,
    );
    let use_uf = outs.needs_graph(action, wolff);
    let sp = spins.as_mut_ptr() as usize;
    let rngs = Slots::new(Some(rngs));

    // SAFETY: as in `isoenergetic_step`.
    for_each_task(tasks.len(), sequential, |task_idx| unsafe {
        let (t, g, systems) = tasks.group(task_idx);
        let slot = t * n_pairs + g;
        let rng = rngs.at(slot);
        let jorg_bond = BondSampler::new(4.0 / temperatures[t]);
        let sp_ptr = sp as *mut i8;
        // ΔE / 4J of swapping the pairs at one endpoint of an active-active bond.
        let aligned = |i: usize, j: usize| -> i32 {
            systems
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| {
                    let (a, b) = (pair[0] * n_spins, pair[1] * n_spins);
                    let (ai, aj) = (*sp_ptr.add(a + i), *sp_ptr.add(a + j));
                    if ai == *sp_ptr.add(b + i) || aj == *sp_ptr.add(b + j) {
                        return 0;
                    }
                    i32::from(ai * aj)
                })
                .sum()
        };
        outs.snapshot_spins(slot, g, sp_ptr, systems, n_spins);
        let is_active = |i: usize| -> bool { any_pair_differs(sp_ptr, systems, n_spins, i) };

        if !use_uf {
            let Some(seed) = find_seed(n_spins, rng, &is_active) else {
                return;
            };
            let mut in_cluster = vec![false; n_spins];
            let mut stack = Vec::with_capacity(n_spins);
            dfs_cluster(
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
                    let inter = aligned(site, nb) as f32 * coupling;
                    if inter <= 0.0 {
                        return false;
                    }
                    jorg_bond.sample(inter, rng)
                },
            );
            for i in (0..n_spins).filter(|&i| in_cluster[i]) {
                flip_group_site(sp_ptr, systems, n_spins, i, true);
            }
            return;
        }

        let active: Vec<bool> = (0..n_spins).map(is_active).collect();
        let (mut uf, metrics) = build_graph(lattice, outs.observation.is_some(), |i, d| {
            if !active[i] {
                return false;
            }
            let j = lattice.neighbor_fwd(i, d);
            if !active[j] {
                return false;
            }
            let inter = aligned(i, j) as f32 * couplings[i * n_neighbors + d];
            if inter <= 0.0 {
                return false;
            }
            jorg_bond.sample(inter, rng)
        });

        // Statistics and snapshots describe the graph, so record them before the
        // Wolff seed search, which may find no active site.
        let counts = uf_flatten_counts(&mut uf.parent);
        outs.publish_graph(
            slot,
            &outs.cluster_ids,
            &uf.parent,
            &counts,
            Some(&active),
            metrics,
        );
        if action == ClusterAction::Observe {
            return;
        }

        if wolff {
            let Some(seed) = find_seed(n_spins, rng, |i| active[i]) else {
                return;
            };
            let seed_root = uf.parent[seed];
            for i in (0..n_spins).filter(|&i| uf.parent[i] == seed_root) {
                flip_group_site(sp_ptr, systems, n_spins, i, true);
            }
            return;
        }
        let storage = &mut *uf;
        decide_clusters(
            &storage.parent,
            &mut storage.rank,
            |_, root| counts[root] > 1,
            |_| u8::from(rng.gen::<f32>() < 0.5),
            |site, flip| {
                if flip == 1 {
                    flip_group_site(sp_ptr, systems, n_spins, site, true);
                }
            },
        );
    });
}

/// CMR two-phase overlap cluster update (Machta-Newman-Stein 2007, eqs 10-11).
///
/// Phase 1 — Blue clusters:
/// 1. Blue bond: doubly-satisfied edges (both replicas satisfied) with prob 1-r²
///    where r = exp(-2|J_ij|/T).
/// 2. SW: flip each non-singleton blue cluster (both replicas) with prob 1/2.
///    Wolff: flip seed's blue cluster (both replicas, always).
/// 3. CSD/top4 from blue clusters.
///
/// Phase 2 — Grey clusters (extend blue UF with red bonds):
/// 4. Red bond: singly-satisfied edges (exactly one replica satisfied, evaluated on
///    post-blue-flip spins) with prob 1-r. Blue flips negate both replicas, which
///    swaps which replica is satisfied on a singly-satisfied edge but preserves the
///    singly-satisfied classification. So red bonds can be evaluated on post-blue-flip
///    spins.
/// 5. Grey = Blue ∪ Red (blue ⊂ grey always).
/// 6. SW: flip each non-singleton grey cluster with k ∈ {0,1,2,3}.
///    Wolff: flip seed's grey cluster with k ∈ {1,2,3}.
///
/// Grey clusters are supersets of blue clusters; sites in blue clusters receive both
/// the blue flip and the grey flip. This composition is the correct CMR update.
#[allow(clippy::too_many_arguments)]
fn cmr_step(
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
    action: ClusterAction,
    outs: &SlotOutputs<'_>,
    sequential: bool,
) {
    let n_spins = lattice.n_spins;
    let n_neighbors = lattice.n_neighbors;
    let n_pairs = n_replicas / 2;
    let wolff = cluster_mode == ClusterMode::Wolff;

    let tasks = build_tasks(
        system_ids,
        n_replicas,
        n_temps,
        2,
        rngs,
        n_pairs,
        active_temps,
    );
    let use_uf = outs.needs_graph(action, wolff);
    let sp = spins.as_mut_ptr() as usize;
    let rngs = Slots::new(Some(rngs));

    // SAFETY: as in `isoenergetic_step`.
    for_each_task(tasks.len(), sequential, |task_idx| unsafe {
        let (t, g, systems) = tasks.group(task_idx);
        let slot = t * n_pairs + g;
        let rng = rngs.at(slot);
        // Blue bonds use 1 - r^2 and red bonds 1 - r, with r = exp(-2|J|/T).
        let blue_bond = BondSampler::new(4.0 / temperatures[t]);
        let red_bond = BondSampler::new(2.0 / temperatures[t]);
        let base_a = systems[0] * n_spins;
        let base_b = systems[1] * n_spins;
        let sp_ptr = sp as *mut i8;
        outs.snapshot_spins(slot, g, sp_ptr, systems, n_spins);
        // Bit 0 of `k` flips replica a at the site, bit 1 replica b.
        let flip = |i: usize, k: u8| {
            if k & 1 != 0 {
                *sp_ptr.add(base_a + i) *= -1;
            }
            if k & 2 != 0 {
                *sp_ptr.add(base_b + i) *= -1;
            }
        };

        if !use_uf {
            // Wolff two-phase update by depth-first search (no statistics needed).
            let seed = rng.gen_range(0..n_spins);

            // === Phase 1: Blue cluster ===
            let mut in_cluster = vec![false; n_spins];
            let mut stack = Vec::with_capacity(n_spins);
            dfs_cluster(
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
                    let coupling = couplings[coupling_idx];

                    let a_sat = *sp_ptr.add(base_a + site) as f32
                        * *sp_ptr.add(base_a + nb) as f32
                        * coupling
                        > 0.0;
                    let b_sat = *sp_ptr.add(base_b + site) as f32
                        * *sp_ptr.add(base_b + nb) as f32
                        * coupling
                        > 0.0;

                    if !a_sat || !b_sat {
                        return false;
                    }
                    blue_bond.sample(coupling.abs(), rng)
                },
            );

            // Flip blue cluster (both replicas always)
            for i in (0..n_spins).filter(|&i| in_cluster[i]) {
                flip(i, 3);
            }

            // === Phase 2: Grey cluster (extend from the flipped blue cluster) ===
            // Edges at the blue cluster had their blue bond decided in phase 1, so only
            // red bonds can still attach there. Sites reached through red bonds sample
            // both bond types afresh: blue on doubly satisfied edges (prob 1-r²) and
            // red on singly satisfied ones (prob 1-r). The grey cluster must absorb
            // both, or the grey flip breaks detailed balance.
            let in_blue = in_cluster.clone();
            stack.extend((0..n_spins).filter(|&i| in_blue[i]));

            let mut grey_bond = |site: usize, nb: usize, coupling: f32| -> bool {
                let a_sat =
                    *sp_ptr.add(base_a + site) as f32 * *sp_ptr.add(base_a + nb) as f32 * coupling
                        > 0.0;
                let b_sat =
                    *sp_ptr.add(base_b + site) as f32 * *sp_ptr.add(base_b + nb) as f32 * coupling
                        > 0.0;
                if a_sat != b_sat {
                    return red_bond.sample(coupling.abs(), rng);
                }
                if !a_sat || in_blue[site] {
                    return false;
                }
                blue_bond.sample(coupling.abs(), rng)
            };

            while let Some(site) = stack.pop() {
                for d in 0..lattice.n_neighbors {
                    let fwd = lattice.neighbor_fwd(site, d);
                    if !in_cluster[fwd] && grey_bond(site, fwd, couplings[site * n_neighbors + d]) {
                        in_cluster[fwd] = true;
                        stack.push(fwd);
                    }

                    let bwd = lattice.neighbor_bwd(site, d);
                    if !in_cluster[bwd] && grey_bond(site, bwd, couplings[bwd * n_neighbors + d]) {
                        in_cluster[bwd] = true;
                        stack.push(bwd);
                    }
                }
            }

            // Flip grey cluster with k ∈ {1,2,3}
            let k: u8 = rng.gen_range(1..=3);
            for i in (0..n_spins).filter(|&i| in_cluster[i]) {
                flip(i, k);
            }
            return;
        }

        // Wolff draws its seed before any bond.
        let seed = wolff.then(|| rng.gen_range(0..n_spins));

        // Site overlaps q_i = a_i b_i classify bonds: an edge is singly satisfied
        // iff q_i != q_j (J != 0), and blue flips negate both replicas, leaving q.
        let overlap: Vec<i8> = (0..n_spins)
            .map(|i| *sp_ptr.add(base_a + i) * *sp_ptr.add(base_b + i))
            .collect();

        // === Phase 1: Blue clusters ===
        let (mut uf, metrics) = build_graph(lattice, outs.observation.is_some(), |i, d| {
            let j = lattice.neighbor_fwd(i, d);
            // Equal site overlaps make replica b satisfied exactly when a is.
            if overlap[i] != overlap[j] {
                return false;
            }
            let coupling = couplings[i * n_neighbors + d];
            let satisfied =
                *sp_ptr.add(base_a + i) as f32 * *sp_ptr.add(base_a + j) as f32 * coupling > 0.0;
            satisfied && blue_bond.sample(coupling.abs(), rng)
        });

        let counts = uf_flatten_counts(&mut uf.parent);
        outs.publish_graph(slot, &outs.blue_ids, &uf.parent, &counts, None, metrics);
        if action == ClusterAction::Observe {
            return;
        }

        // Flip blue clusters (both replicas jointly)
        if let Some(seed) = seed {
            let seed_root = uf.parent[seed];
            for i in (0..n_spins).filter(|&i| uf.parent[i] == seed_root) {
                flip(i, 3);
            }
        } else {
            // Keep UF ranks intact until red bonds extend the blue clusters.
            let mut do_flip = vec![UNDECIDED; n_spins];
            decide_clusters(
                &uf.parent,
                &mut do_flip,
                |_, root| counts[root] > 1,
                |_| u8::from(rng.gen::<f32>() < 0.5),
                |site, both| {
                    if both == 1 {
                        flip(site, 3);
                    }
                },
            );
        }

        // === Phase 2: Grey clusters (extend blue UF with red bonds) ===
        // Red bond: singly-satisfied edges on post-flip spins, prob 1 - r.
        // No clone needed: blue flips preserve the singly-satisfied classification.
        // Return blue counts so grey counts can reuse the pooled allocation.
        drop(counts);
        let storage = &mut *uf;
        uf_bonds_extend(&mut storage.parent, &mut storage.rank, lattice, |i, d| {
            let j = lattice.neighbor_fwd(i, d);
            let coupling = couplings[i * n_neighbors + d];
            overlap[i] != overlap[j] && coupling != 0.0 && red_bond.sample(coupling.abs(), rng)
        });

        let grey_counts = uf_flatten_counts(&mut uf.parent);
        if let Some(ids) = outs.cluster_ids.get(slot) {
            ids.clear();
            ids.extend_from_slice(&uf.parent);
        }

        // Flip grey clusters (each replica independently)
        if let Some(seed) = seed {
            let seed_root = uf.parent[seed];
            let k: u8 = rng.gen_range(1..=3);
            for i in (0..n_spins).filter(|&i| uf.parent[i] == seed_root) {
                flip(i, k);
            }
            return;
        }
        let storage = &mut *uf;
        decide_clusters(
            &storage.parent,
            &mut storage.rank,
            |_, root| grey_counts[root] > 1,
            |_| rng.gen_range(0..=3),
            flip,
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spins::energy::compute_energies_into;
    use rand::SeedableRng;

    /// The pair rule has no fixed-size state, so its groups may exceed
    /// `MAX_PERM_GROUP`; the move conserves the group's summed energy.
    #[test]
    fn pair_groups_may_exceed_the_permutation_limit() {
        let lattice = Lattice::new(vec![4, 3]);
        let n = lattice.n_spins;
        let n_replicas = MAX_PERM_GROUP + 2;
        let mut rng = Xoshiro256StarStar::seed_from_u64(3);
        let couplings: Vec<f32> = (0..n * lattice.n_neighbors)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let total_energy = |spins: &[i8]| {
            let mut energies = vec![0.0; n_replicas];
            compute_energies_into(&lattice, spins, &couplings, &mut energies);
            energies.iter().map(|&e| f64::from(e)).sum::<f64>()
        };
        for cluster_mode in [ClusterMode::Sw, ClusterMode::Wolff] {
            let mut spins: Vec<i8> = (0..n_replicas * n)
                .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
                .collect();
            let initial = spins.clone();
            let before = total_energy(&spins);
            let mut rngs: Vec<_> = (0..n_replicas as u64 / 2)
                .map(Xoshiro256StarStar::seed_from_u64)
                .collect();
            let system_ids: Vec<usize> = (0..n_replicas).collect();
            for _ in 0..8 {
                overlap_update(
                    &lattice,
                    &mut spins,
                    &couplings,
                    &[1.0],
                    &system_ids,
                    n_replicas,
                    1,
                    &[true],
                    &mut rngs,
                    &OverlapClusterBuildMode::Pair(n_replicas),
                    cluster_mode,
                    ClusterAction::Update,
                    None,
                    None,
                    None,
                    true,
                    None,
                    None,
                    None,
                    None,
                );
            }
            assert_ne!(spins, initial, "{cluster_mode:?}");
            assert!(
                (total_energy(&spins) - before).abs() < 1e-4,
                "{cluster_mode:?}"
            );
        }
    }

    /// With identical replicas no site is active, so the Wolff seed search fails and
    /// the move has no cluster; the statistics must still be written, and empty.
    #[test]
    fn wolff_records_statistics_without_active_sites() {
        let lattice = Lattice::new(vec![4, 4]);
        let n = lattice.n_spins;
        let couplings = vec![1.0; n * lattice.n_neighbors];
        for mode in [
            OverlapClusterBuildMode::Houdayer(2),
            OverlapClusterBuildMode::Jorg(2),
        ] {
            let mut spins = vec![1i8; 2 * n];
            let mut rngs = vec![Xoshiro256StarStar::seed_from_u64(1)];
            let mut csd = vec![vec![0u64; n + 1]];
            let mut top4 = vec![[7u32; 4]];
            let mut snapshot = vec![Vec::new()];
            overlap_update(
                &lattice,
                &mut spins,
                &couplings,
                &[1.0],
                &[0, 1],
                2,
                1,
                &[true],
                &mut rngs,
                &mode,
                ClusterMode::Wolff,
                ClusterAction::Update,
                Some(&mut csd),
                Some(&mut top4),
                None,
                true,
                Some(&mut snapshot),
                None,
                None,
                None,
            );
            assert!(csd[0].iter().all(|&count| count == 0), "{mode:?}");
            assert_eq!(top4[0], [0; 4], "{mode:?}");
            assert_eq!(snapshot[0], (0..n as u32).collect::<Vec<_>>(), "{mode:?}");
            assert!(spins.iter().all(|&s| s == 1));
        }
    }
}
