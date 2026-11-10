use crate::geometry::Lattice;
use crate::mcmc::{threshold_f64, uniform_draw};
use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;
use std::cell::RefCell;
use std::ops::{Deref, DerefMut};

// --- Bond activation ---

/// Samples bonds with probability `1 - exp(-rate * x)` for bond weight `x > 0`.
///
/// Draws compare a uniform integer against an exact threshold; unit weights reuse a
/// precomputed threshold, which is bit-identical to the general formula.
#[derive(Clone, Copy)]
pub(crate) struct BondSampler {
    neg_rate: f64,
    unit: i64,
}

impl BondSampler {
    #[inline]
    pub(crate) fn new(rate: f32) -> Self {
        let neg_rate = -f64::from(rate);
        Self {
            neg_rate,
            unit: Self::general_threshold(neg_rate, 1.0),
        }
    }

    #[inline]
    fn general_threshold(neg_rate: f64, x: f32) -> i64 {
        // f64 exp keeps 1 - e^-y accurate to ~1e-16 / y (better than f32 for y > 1e-9)
        // and is far cheaper than expm1f.
        threshold_f64(1.0 - (f64::from(x) * neg_rate).exp())
    }

    #[inline]
    pub(crate) fn sample(self, x: f32, rng: &mut Xoshiro256StarStar) -> bool {
        let threshold = if x == 1.0 {
            self.unit
        } else {
            Self::general_threshold(self.neg_rate, x)
        };
        uniform_draw(rng) < threshold
    }
}

// --- Union-Find ---

#[derive(Default)]
pub(super) struct UfStorage {
    pub(super) parent: Vec<u32>,
    pub(super) rank: Vec<u8>,
}

pub(super) struct PooledUf(Option<UfStorage>);

impl Deref for PooledUf {
    type Target = UfStorage;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref().unwrap()
    }
}

impl DerefMut for PooledUf {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut().unwrap()
    }
}

impl Drop for PooledUf {
    fn drop(&mut self) {
        let storage = self.0.take().unwrap();
        UF_POOL.with(|pool| pool.borrow_mut().push(storage));
    }
}

pub(super) struct PooledCounts(Option<Vec<u32>>);

impl Deref for PooledCounts {
    type Target = [u32];

    fn deref(&self) -> &Self::Target {
        self.0.as_ref().unwrap()
    }
}

impl DerefMut for PooledCounts {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut().unwrap()
    }
}

impl Drop for PooledCounts {
    fn drop(&mut self) {
        let counts = self.0.take().unwrap();
        COUNTS_POOL.with(|pool| pool.borrow_mut().push(counts));
    }
}

// Pooling is intentionally limited to overlap clusters because extending it to ordinary FK/SW regressed performance.
thread_local! {
    static UF_POOL: RefCell<Vec<UfStorage>> = const { RefCell::new(Vec::new()) };
    static COUNTS_POOL: RefCell<Vec<Vec<u32>>> = const { RefCell::new(Vec::new()) };
}

#[inline]
pub(super) fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

#[inline]
pub(super) fn union(parent: &mut [u32], rank: &mut [u8], x: u32, y: u32) {
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
// `dfs_cluster` and `uf_bonds` factor out the two cluster-building patterns
// (DFS single-cluster and union-find global decomposition). Each takes a
// closure for the algorithm-specific activation logic. The closure is
// `impl FnMut`, so it is monomorphized at each call site — zero overhead
// vs hand-inlined code. The closure can capture mutable state (e.g. an RNG
// for probabilistic bond activation).

/// Find an eligible seed site via 64 random probes.
///
/// Returns `None` when no probe passes `eligible`. This is a fast
/// probabilistic search — if eligible sites are rare it may miss them.
#[inline]
pub(super) fn find_seed(
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

/// Grow a DFS cluster from `seed`. `should_add(site, neighbor, dim, forward)`
/// decides whether to add each not-yet-visited neighbor.
/// Caller owns buffers: `in_cluster` must be all-false, `stack` must be empty.
#[inline]
pub(super) fn dfs_cluster(
    lattice: &Lattice,
    seed: usize,
    in_cluster: &mut [bool],
    stack: &mut Vec<usize>,
    mut should_add: impl FnMut(usize, usize, usize, bool) -> bool,
) {
    in_cluster[seed] = true;
    stack.push(seed);

    while let Some(site) = stack.pop() {
        for d in 0..lattice.n_neighbors {
            let fwd = lattice.neighbor_fwd(site, d);
            if !in_cluster[fwd] && should_add(site, fwd, d, true) {
                in_cluster[fwd] = true;
                stack.push(fwd);
            }
            let bwd = lattice.neighbor_bwd(site, d);
            if !in_cluster[bwd] && should_add(site, bwd, d, false) {
                in_cluster[bwd] = true;
                stack.push(bwd);
            }
        }
    }
}

/// Activate forward bonds via union-find. `should_bond(site, dim)` decides
/// whether to activate the bond from `site` to its forward neighbor in `dim`.
/// Returns pooled `(parent, rank)` storage.
#[inline]
pub(super) fn uf_bonds(
    lattice: &Lattice,
    should_bond: impl FnMut(usize, usize) -> bool,
) -> PooledUf {
    uf_bonds_with(lattice, should_bond, |_site, _dim| {})
}

#[inline]
pub(super) fn uf_bonds_with(
    lattice: &Lattice,
    mut should_bond: impl FnMut(usize, usize) -> bool,
    mut on_bond: impl FnMut(usize, usize),
) -> PooledUf {
    let n_spins = lattice.n_spins;
    let mut storage = UF_POOL
        .with(|pool| pool.borrow_mut().pop())
        .unwrap_or_default();
    reset_uf(&mut storage.parent, &mut storage.rank, n_spins);
    activate_bonds(
        &mut storage.parent,
        &mut storage.rank,
        lattice,
        &mut should_bond,
        &mut on_bond,
    );

    PooledUf(Some(storage))
}

#[inline]
pub(super) fn uf_bonds_fresh(
    lattice: &Lattice,
    should_bond: impl FnMut(usize, usize) -> bool,
) -> (Vec<u32>, Vec<u8>) {
    uf_bonds_fresh_with(lattice, should_bond, |_site, _dim| {})
}

#[inline]
pub(super) fn uf_bonds_fresh_with(
    lattice: &Lattice,
    mut should_bond: impl FnMut(usize, usize) -> bool,
    mut on_bond: impl FnMut(usize, usize),
) -> (Vec<u32>, Vec<u8>) {
    let mut parent = Vec::new();
    let mut rank = Vec::new();
    reset_uf(&mut parent, &mut rank, lattice.n_spins);
    activate_bonds(
        &mut parent,
        &mut rank,
        lattice,
        &mut should_bond,
        &mut on_bond,
    );
    (parent, rank)
}

#[inline]
fn reset_uf(parent: &mut Vec<u32>, rank: &mut Vec<u8>, n_spins: usize) {
    parent.resize(n_spins, 0);
    for (i, parent) in parent.iter_mut().enumerate() {
        *parent = i as u32;
    }
    rank.resize(n_spins, 0);
    rank.fill(0);
}

#[inline]
fn activate_bonds(
    parent: &mut [u32],
    rank: &mut [u8],
    lattice: &Lattice,
    should_bond: &mut impl FnMut(usize, usize) -> bool,
    on_bond: &mut impl FnMut(usize, usize),
) {
    for i in 0..lattice.n_spins {
        for d in 0..lattice.n_neighbors {
            if should_bond(i, d) {
                let j = lattice.neighbor_fwd(i, d);
                union(parent, rank, i as u32, j as u32);
                on_bond(i, d);
            }
        }
    }
}

/// Extend an existing union-find by activating additional forward bonds.
#[inline]
pub(super) fn uf_bonds_extend(
    parent: &mut [u32],
    rank: &mut [u8],
    lattice: &Lattice,
    mut should_bond: impl FnMut(usize, usize) -> bool,
) {
    activate_bonds(parent, rank, lattice, &mut should_bond, &mut |_, _| {});
}

/// Pooled union-find of `should_bond`, recording [`BondMetrics`] only when `observe`,
/// so the unobserved build keeps its bond loop free of bookkeeping.
#[inline]
pub(super) fn build_graph(
    lattice: &Lattice,
    observe: bool,
    should_bond: impl FnMut(usize, usize) -> bool,
) -> (PooledUf, Option<BondMetrics>) {
    if !observe {
        return (uf_bonds(lattice, should_bond), None);
    }
    let mut metrics = BondMetrics::new(lattice);
    let uf = uf_bonds_with(lattice, should_bond, |site, dim| {
        metrics.record_bond(lattice, site, dim);
    });
    (uf, Some(metrics))
}

/// Flatten UF parent array in-place and return per-root counts.
#[inline]
pub(super) fn uf_flatten_counts(parent: &mut [u32]) -> PooledCounts {
    let n = parent.len();
    uf_flatten(parent);
    let mut counts = COUNTS_POOL
        .with(|pool| pool.borrow_mut().pop())
        .unwrap_or_default();
    counts.resize(n, 0);
    counts.fill(0);
    count_roots(parent, &mut counts);
    PooledCounts(Some(counts))
}

#[inline]
pub(super) fn uf_flatten_counts_fresh(parent: &mut [u32]) -> Vec<u32> {
    let mut counts = vec![0u32; parent.len()];
    uf_flatten(parent);
    count_roots(parent, &mut counts);
    counts
}

#[inline]
fn count_roots(parent: &[u32], counts: &mut [u32]) {
    for &root in parent {
        counts[root as usize] += 1;
    }
}

/// Flatten a UF parent array in-place.
#[inline]
pub(super) fn uf_flatten(parent: &mut [u32]) {
    for i in 0..parent.len() {
        parent[i] = find(parent, i as u32);
    }
}

/// Marks a cluster whose update has not been drawn yet.
pub(super) const UNDECIDED: u8 = u8::MAX;

/// Swendsen-Wang pass over a flattened union-find: sites are visited in order, each
/// cluster draws its decision at its first eligible site, and `apply` sees every
/// eligible site with its cluster's decision. `eligible` and `decide` take the root.
#[inline]
pub(super) fn decide_clusters(
    parent: &[u32],
    decisions: &mut [u8],
    mut eligible: impl FnMut(usize, usize) -> bool,
    mut decide: impl FnMut(usize) -> u8,
    mut apply: impl FnMut(usize, u8),
) {
    decisions.fill(UNDECIDED);
    for (site, &root) in parent.iter().enumerate() {
        let root = root as usize;
        if !eligible(site, root) {
            continue;
        }
        if decisions[root] == UNDECIDED {
            decisions[root] = decide(root);
        }
        apply(site, decisions[root]);
    }
}

/// Histogram cluster sizes into `hist[s] += 1`; zero entries are not clusters.
#[inline]
pub(super) fn uf_histogram(sizes: impl IntoIterator<Item = u32>, hist: &mut [u64]) {
    for size in sizes {
        if size > 0 {
            hist[size as usize] += 1;
        }
    }
}

pub(super) fn top4_sizes(sizes: impl IntoIterator<Item = u32>) -> [u32; 4] {
    let mut top = [0u32; 4]; // ascending; top[0] = current minimum of top-4
    for c in sizes {
        if c > top[0] {
            top[0] = c;
            top.sort_unstable();
        }
    }
    top.reverse(); // descending: top[0] = largest
    top
}

/// Most axes along which wrapping is tracked.
pub(crate) const MAX_WINDING_DIMS: usize = 3;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct GraphObservationSlot {
    pub top4: [u32; 4],
    pub active_bonds: u32,
    /// Whether some cluster wraps the torus along each lattice dimension.
    pub wraps: [bool; MAX_WINDING_DIMS],
    /// Whether one cluster wraps along every dimension.
    pub wraps_all: bool,
    pub wrapping_clusters: u32,
    /// Sites in wrapping clusters.
    pub wrapping_sites: u32,
    pub large_components: u32,
    /// `Σ_C |C|² / N²` and `Σ_C |C|⁴ / N⁴` over the clusters.
    pub s2: f64,
    pub s4: f64,
    pub observed: bool,
}

/// A component counts as large from this fraction of the sites.
const LARGE_COMPONENT_FRACTION: f64 = 0.05;

// Component and winding bookkeeping stay separate because combining their
// union-find paths regressed this observer workload.
enum Winding {
    Untracked,
    Square(WindingUf<2>),
    Cubic(WindingUf<3>),
}

pub(super) struct BondMetrics {
    active_bonds: u32,
    winding: Winding,
}

impl BondMetrics {
    pub fn new(lattice: &Lattice) -> Self {
        let n_sites = lattice.n_spins;
        let winding = if let Some((l0, l1)) = lattice.square_shape() {
            Winding::Square(WindingUf::new(n_sites, [l0, l1]))
        } else if let Some(shape) = lattice.cubic_shape() {
            Winding::Cubic(WindingUf::new(n_sites, shape))
        } else {
            Winding::Untracked
        };
        Self {
            active_bonds: 0,
            winding,
        }
    }

    #[inline]
    pub fn record_bond(&mut self, lattice: &Lattice, site: usize, dim: usize) {
        self.active_bonds += 1;
        let neighbor = || lattice.neighbor_fwd(site, dim);
        match &mut self.winding {
            Winding::Untracked => {}
            Winding::Square(winding) => winding.union_forward(site, neighbor(), dim),
            Winding::Cubic(winding) => winding.union_forward(site, neighbor(), dim),
        }
    }

    /// `clusters` are the `(root, size)` pairs of a graph on `n_sites` sites, as left by
    /// the union-find this recorder followed bond by bond (zero sizes are ignored).
    pub fn finish(
        self,
        n_sites: usize,
        clusters: impl Iterator<Item = (usize, u32)> + Clone,
    ) -> GraphObservationSlot {
        let threshold = ((n_sites as f64) * LARGE_COMPONENT_FRACTION).ceil() as u32;
        let sizes = clusters.clone().map(|(_, size)| size);
        let fraction = |size: u32| f64::from(size) / n_sites as f64;
        let mut slot = GraphObservationSlot {
            top4: top4_sizes(sizes.clone()),
            active_bonds: self.active_bonds,
            large_components: sizes.clone().filter(|&size| size >= threshold).count() as u32,
            s2: sizes.clone().map(|size| fraction(size).powi(2)).sum(),
            s4: sizes.map(|size| fraction(size).powi(4)).sum(),
            observed: true,
            ..Default::default()
        };
        match &self.winding {
            Winding::Untracked => {}
            Winding::Square(winding) => winding.summarize(clusters, &mut slot),
            Winding::Cubic(winding) => winding.summarize(clusters, &mut slot),
        }
        slot
    }
}

/// Union-find that tracks each site's displacement to its root, so that closing a
/// cycle reveals whether it winds around the torus.
struct WindingUf<const D: usize> {
    parent: Vec<u32>,
    rank: Vec<u8>,
    displacement: Vec<[i64; D]>,
    wrap: Vec<[bool; D]>,
    shape: [i64; D],
}

impl<const D: usize> WindingUf<D> {
    fn new(n_sites: usize, shape: [usize; D]) -> Self {
        Self {
            parent: (0..n_sites as u32).collect(),
            rank: vec![0; n_sites],
            displacement: vec![[0; D]; n_sites],
            wrap: vec![[false; D]; n_sites],
            shape: shape.map(|extent| extent as i64),
        }
    }

    fn find(&mut self, site: usize) -> (usize, [i64; D]) {
        let parent = self.parent[site] as usize;
        if parent == site {
            return (site, [0; D]);
        }

        let own = self.displacement[site];
        let (root, parent_to_root) = self.find(parent);
        let to_root = std::array::from_fn(|axis| own[axis] + parent_to_root[axis]);
        self.parent[site] = root as u32;
        self.displacement[site] = to_root;
        (root, to_root)
    }

    fn union_forward(&mut self, site: usize, neighbor: usize, dim: usize) {
        let (root_site, disp_site) = self.find(site);
        let (root_neighbor, disp_neighbor) = self.find(neighbor);
        let mut neighbor_from_site: [i64; D] =
            std::array::from_fn(|axis| disp_site[axis] - disp_neighbor[axis]);
        neighbor_from_site[dim] += 1;

        if root_site == root_neighbor {
            for (axis, &displacement) in neighbor_from_site.iter().enumerate() {
                debug_assert_eq!(displacement % self.shape[axis], 0);
                self.wrap[root_site][axis] |= displacement != 0;
            }
            return;
        }

        let combined_wrap: [bool; D] = std::array::from_fn(|axis| {
            self.wrap[root_site][axis] || self.wrap[root_neighbor][axis]
        });
        // The same rank rule as `union`, so both structures keep the same roots.
        if self.rank[root_site] < self.rank[root_neighbor] {
            self.parent[root_site] = root_neighbor as u32;
            self.displacement[root_site] = neighbor_from_site.map(|d| -d);
            self.wrap[root_neighbor] = combined_wrap;
            return;
        }

        self.parent[root_neighbor] = root_site as u32;
        self.displacement[root_neighbor] = neighbor_from_site;
        self.wrap[root_site] = combined_wrap;
        if self.rank[root_site] == self.rank[root_neighbor] {
            self.rank[root_site] += 1;
        }
    }

    /// Fills the wrapping fields of `slot` from the `(root, size)` pairs of the
    /// component union-find. Wrap flags live on roots, which path compression never
    /// changes, and this structure merges by the same rule as the component one, so
    /// the two agree on every root.
    fn summarize(
        &self,
        clusters: impl Iterator<Item = (usize, u32)>,
        slot: &mut GraphObservationSlot,
    ) {
        for (root, size) in clusters.filter(|&(_, size)| size > 0) {
            debug_assert_eq!(self.parent[root] as usize, root);
            let wrap = self.wrap[root];
            if !wrap.contains(&true) {
                continue;
            }
            slot.wrapping_clusters += 1;
            slot.wrapping_sites += size;
            slot.wraps_all |= !wrap.contains(&false);
            for (seen, &wraps) in slot.wraps.iter_mut().zip(&wrap) {
                *seen |= wraps;
            }
        }
    }

    #[cfg(test)]
    fn winding(&self) -> [bool; D] {
        let mut seen = [false; D];
        for (site, wrap) in self.wrap.iter().enumerate() {
            if self.parent[site] as usize != site {
                continue;
            }
            for (seen, &wraps) in seen.iter_mut().zip(wrap) {
                *seen |= wraps;
            }
        }
        seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // 4×4 periodic lattice:
    //
    //    0  1  2  3
    //    4  5  6  7
    //    8  9 10 11
    //   12 13 14 15
    //
    // Forward neighbors (periodic):
    //   dim 0 (→): row*4 + (col+1)%4   — so 3→0, 7→4, 11→8, 15→12
    //   dim 1 (↓): ((row+1)%4)*4 + col  — so 12→0, 13→1, 14→2, 15→3

    fn lattice_4x4() -> Lattice {
        Lattice::new(vec![4, 4])
    }

    // Bond-based activation: specific (site, dim) forward bonds.
    // dim 0 = ↓ (stride 4), dim 1 = → (stride 1)
    //
    //    0 ─→ 1    2    3
    //    ↓         (periodic →) ╲
    //    4    5    6    7        ╲
    //                             ╲
    //    8    9   10 ─→ 11        ╱
    //              ↓             ╱
    //   12   13   14   15 ←────╱  (bond 3,1: 3→0 wraps right)
    //
    // Active forward bonds: (0,1)=0→1, (0,0)=0→4, (3,1)=3→0, (10,1)=10→11, (10,0)=10→14
    // Clusters: {0,1,3,4}, {10,11,14}, 9 singletons

    fn bond_set() -> HashSet<(usize, usize)> {
        [(0, 0), (0, 1), (3, 1), (10, 0), (10, 1)]
            .into_iter()
            .collect()
    }

    fn dfs_bond_closure(
        bonds: &HashSet<(usize, usize)>,
    ) -> impl FnMut(usize, usize, usize, bool) -> bool + '_ {
        |site, nb, d, fwd| {
            let src = if fwd { site } else { nb };
            bonds.contains(&(src, d))
        }
    }

    #[test]
    fn test_dfs_bond_based() {
        let lattice = lattice_4x4();
        let n = lattice.n_spins;
        let bonds = bond_set();

        // Seed at 0 — should grow {0, 1, 3, 4}
        let mut in_cluster = vec![false; n];
        let mut stack = Vec::new();
        dfs_cluster(
            &lattice,
            0,
            &mut in_cluster,
            &mut stack,
            dfs_bond_closure(&bonds),
        );
        let cluster: HashSet<usize> = (0..n).filter(|&i| in_cluster[i]).collect();
        assert_eq!(cluster, [0, 1, 3, 4].into_iter().collect());

        // Seed at 10 — should grow {10, 11, 14}
        in_cluster.fill(false);
        stack.clear();
        dfs_cluster(
            &lattice,
            10,
            &mut in_cluster,
            &mut stack,
            dfs_bond_closure(&bonds),
        );
        let cluster: HashSet<usize> = (0..n).filter(|&i| in_cluster[i]).collect();
        assert_eq!(cluster, [10, 11, 14].into_iter().collect());

        // Seed at 7 — isolated singleton
        in_cluster.fill(false);
        stack.clear();
        dfs_cluster(
            &lattice,
            7,
            &mut in_cluster,
            &mut stack,
            dfs_bond_closure(&bonds),
        );
        let cluster: HashSet<usize> = (0..n).filter(|&i| in_cluster[i]).collect();
        assert_eq!(cluster, [7].into_iter().collect());
    }

    // Site-based activation: all forward bonds from active sites.
    // dim 0 = ↓ (stride 4), dim 1 = → (stride 1)
    //
    //    0 ─→ 1    2    3 ─→ 0  (periodic →)
    //    ↓              ↓
    //    4    5    6    7
    //
    //    8    9   10 ─→ 11
    //              ↓
    //   12   13   14   15
    //
    // Active sites: {0, 3, 10}
    // Active forward bonds: (0,1)=0→1, (0,0)=0→4, (3,1)=3→0, (3,0)=3→7,
    //                       (10,1)=10→11, (10,0)=10→14
    // Clusters: {0,1,3,4,7}, {10,11,14}, 9 singletons

    fn active_sites() -> HashSet<usize> {
        [0, 3, 10].into_iter().collect()
    }

    fn dfs_site_closure(
        sites: &HashSet<usize>,
    ) -> impl FnMut(usize, usize, usize, bool) -> bool + '_ {
        |site, nb, _d, fwd| {
            let src = if fwd { site } else { nb };
            sites.contains(&src)
        }
    }

    #[test]
    fn test_dfs_site_based() {
        let lattice = lattice_4x4();
        let n = lattice.n_spins;
        let sites = active_sites();

        // Seed at 0 — should grow {0, 1, 3, 4, 7} (3→7 adds site 7)
        let mut in_cluster = vec![false; n];
        let mut stack = Vec::new();
        dfs_cluster(
            &lattice,
            0,
            &mut in_cluster,
            &mut stack,
            dfs_site_closure(&sites),
        );
        let cluster: HashSet<usize> = (0..n).filter(|&i| in_cluster[i]).collect();
        assert_eq!(cluster, [0, 1, 3, 4, 7].into_iter().collect());

        // Seed at 10 — should grow {10, 11, 14}
        in_cluster.fill(false);
        stack.clear();
        dfs_cluster(
            &lattice,
            10,
            &mut in_cluster,
            &mut stack,
            dfs_site_closure(&sites),
        );
        let cluster: HashSet<usize> = (0..n).filter(|&i| in_cluster[i]).collect();
        assert_eq!(cluster, [10, 11, 14].into_iter().collect());
    }

    #[test]
    fn test_uf_bond_based() {
        let lattice = lattice_4x4();
        let n = lattice.n_spins;
        let bonds = bond_set();

        let mut uf = uf_bonds(&lattice, |i, d| bonds.contains(&(i, d)));
        let parent = &mut uf.parent;

        // Flatten parents for easy inspection
        for i in 0..n {
            let root = find(parent, i as u32);
            parent[i] = root;
        }

        // Sites in cluster A must share a root
        let root_a = parent[0];
        for &s in &[0, 1, 3, 4] {
            assert_eq!(parent[s], root_a, "site {s} should be in cluster A");
        }

        // Sites in cluster B must share a root
        let root_b = parent[10];
        for &s in &[10, 11, 14] {
            assert_eq!(parent[s], root_b, "site {s} should be in cluster B");
        }

        // Clusters A and B are distinct
        assert_ne!(root_a, root_b);

        // Remaining sites are singletons
        let clustered: HashSet<usize> = [0, 1, 3, 4, 10, 11, 14].into_iter().collect();
        for (i, &site_parent) in parent.iter().enumerate().take(n) {
            if !clustered.contains(&i) {
                assert_eq!(site_parent, i as u32, "site {i} should be a singleton");
            }
        }
    }

    #[test]
    fn test_uf_site_based() {
        let lattice = lattice_4x4();
        let n = lattice.n_spins;
        let sites = active_sites();

        let mut uf = uf_bonds(&lattice, |i, _d| sites.contains(&i));
        let parent = &mut uf.parent;

        for i in 0..n {
            let root = find(parent, i as u32);
            parent[i] = root;
        }

        // Cluster A: {0, 1, 3, 4, 7}
        let root_a = parent[0];
        for &s in &[0, 1, 3, 4, 7] {
            assert_eq!(parent[s], root_a, "site {s} should be in cluster A");
        }

        // Cluster B: {10, 11, 14}
        let root_b = parent[10];
        for &s in &[10, 11, 14] {
            assert_eq!(parent[s], root_b, "site {s} should be in cluster B");
        }

        assert_ne!(root_a, root_b);

        let clustered: HashSet<usize> = [0, 1, 3, 4, 7, 10, 11, 14].into_iter().collect();
        for (i, &site_parent) in parent.iter().enumerate().take(n) {
            if !clustered.contains(&i) {
                assert_eq!(site_parent, i as u32, "site {i} should be a singleton");
            }
        }
    }

    #[test]
    fn test_uf_csd_bond_based() {
        let lattice = lattice_4x4();
        let n = lattice.n_spins;
        let bonds = bond_set();

        let mut uf = uf_bonds(&lattice, |i, d| bonds.contains(&(i, d)));
        let counts = uf_flatten_counts(&mut uf.parent);
        let mut hist = vec![0u64; n + 1];
        uf_histogram(counts.iter().copied(), &mut hist);

        // Clusters: size 4 ×1, size 3 ×1, size 1 ×9
        assert_eq!(hist[1], 9);
        assert_eq!(hist[3], 1);
        assert_eq!(hist[4], 1);
        assert_eq!(hist.iter().sum::<u64>(), 11);
    }

    #[test]
    fn test_uf_csd_site_based() {
        let lattice = lattice_4x4();
        let n = lattice.n_spins;
        let sites = active_sites();

        let mut uf = uf_bonds(&lattice, |i, _d| sites.contains(&i));
        let counts = uf_flatten_counts(&mut uf.parent);
        let mut hist = vec![0u64; n + 1];
        uf_histogram(counts.iter().copied(), &mut hist);

        // Clusters: size 5 ×1, size 3 ×1, size 1 ×8
        assert_eq!(hist[1], 8);
        assert_eq!(hist[3], 1);
        assert_eq!(hist[5], 1);
        assert_eq!(hist.iter().sum::<u64>(), 10);
    }

    #[test]
    fn pooled_uf_resets_after_size_change() {
        let larger = Lattice::new(vec![4, 4]);
        {
            let mut uf = uf_bonds(&larger, |_i, _d| true);
            let counts = uf_flatten_counts(&mut uf.parent);
            assert_eq!(counts.iter().sum::<u32>(), 16);
        }

        let smaller = Lattice::new(vec![3, 3]);
        let mut uf = uf_bonds(&smaller, |_i, _d| false);
        assert_eq!(uf.parent, (0..9).collect::<Vec<_>>());
        assert!(uf.rank.iter().all(|&rank| rank == 0));
        let counts = uf_flatten_counts(&mut uf.parent);
        assert!(counts.iter().all(|&count| count == 1));
    }

    #[test]
    fn winding_distinguishes_seams_and_noncontractible_cycles() {
        let index = |row: usize, col: usize| 4 * row + col;
        let mut seam = WindingUf::new(16, [4, 4]);
        seam.union_forward(index(0, 3), index(0, 0), 1);
        assert_eq!(seam.winding(), [false, false]);

        let mut row = WindingUf::new(16, [4, 4]);
        for col in 0..4 {
            row.union_forward(index(1, col), index(1, (col + 1) % 4), 1);
        }
        assert_eq!(row.winding(), [false, true]);

        let mut both = WindingUf::new(16, [4, 4]);
        for col in 0..4 {
            both.union_forward(index(0, col), index(0, (col + 1) % 4), 1);
        }
        for row in 0..4 {
            both.union_forward(index(row, 0), index((row + 1) % 4, 0), 0);
        }
        assert_eq!(both.winding(), [true, true]);
    }

    #[test]
    fn bond_metrics_reuse_component_counts() {
        let lattice = lattice_4x4();
        let bonds = bond_set();
        let mut metrics = BondMetrics::new(&lattice);
        let mut uf = uf_bonds_with(
            &lattice,
            |site, dim| bonds.contains(&(site, dim)),
            |site, dim| metrics.record_bond(&lattice, site, dim),
        );
        let counts = uf_flatten_counts(&mut uf.parent);
        let summary = metrics.finish(counts.len(), counts.iter().copied().enumerate());

        assert_eq!(summary.active_bonds, bonds.len() as u32);
        assert_eq!(summary.top4, [4, 3, 1, 1]);
        assert!(summary.observed);
        let sizes: Vec<f64> = counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| c as f64 / 16.0)
            .collect();
        assert!((summary.s2 - sizes.iter().map(|s| s * s).sum::<f64>()).abs() < 1e-15);
        assert!((summary.s4 - sizes.iter().map(|s| s.powi(4)).sum::<f64>()).abs() < 1e-15);
    }

    /// A full row wraps along its own direction only; its cluster is the one that is
    /// counted and weighed. A line through a 3D lattice wraps along one of three axes.
    #[test]
    fn wrapping_clusters_are_counted_with_their_mass() {
        let wrap_of = |lattice: &Lattice, bonds: &[(usize, usize)]| {
            let mut metrics = BondMetrics::new(lattice);
            let mut uf = uf_bonds_with(
                lattice,
                |site, dim| bonds.contains(&(site, dim)),
                |site, dim| metrics.record_bond(lattice, site, dim),
            );
            let counts = uf_flatten_counts(&mut uf.parent);
            metrics.finish(counts.len(), counts.iter().copied().enumerate())
        };

        let square = lattice_4x4();
        // Row 1 closed along dimension 1, plus one stray bond elsewhere.
        let row: Vec<(usize, usize)> = (4..8).map(|site| (site, 1)).chain([(12, 1)]).collect();
        let slot = wrap_of(&square, &row);
        assert_eq!(slot.wraps, [false, true, false]);
        assert!(!slot.wraps_all);
        assert_eq!((slot.wrapping_clusters, slot.wrapping_sites), (1, 4));

        let cubic = Lattice::new(vec![3, 3, 3]);
        // The line of sites (x, 0, 0), closed along dimension 0.
        let line: Vec<(usize, usize)> = (0..3).map(|x| (9 * x, 0)).collect();
        let slot = wrap_of(&cubic, &line);
        assert_eq!(slot.wraps, [true, false, false]);
        assert_eq!((slot.wrapping_clusters, slot.wrapping_sites), (1, 3));

        // All bonds: one cluster wrapping along every axis.
        let all: Vec<(usize, usize)> = (0..27).flat_map(|i| (0..3).map(move |d| (i, d))).collect();
        let slot = wrap_of(&cubic, &all);
        assert!(slot.wraps_all);
        assert_eq!((slot.wrapping_clusters, slot.wrapping_sites), (1, 27));
        assert!((slot.s2 - 1.0).abs() < 1e-15);
    }
}
