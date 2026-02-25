use crate::geometry::Lattice;
use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;

// --- Union-Find ---

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
// `bfs_cluster` and `uf_bonds` factor out the two cluster-building patterns
// (BFS single-cluster and union-find global decomposition). Each takes a
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

/// Grow a BFS cluster from `seed`. `should_add(site, neighbor, dim, forward)`
/// decides whether to add each not-yet-visited neighbor.
/// Caller owns buffers: `in_cluster` must be all-false, `stack` must be empty.
#[inline]
pub(super) fn bfs_cluster(
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
/// and cluster sizes are histogrammed into the slice (`hist[s]` += 1 for each
/// cluster of size `s`).
#[inline]
pub(super) fn uf_bonds(
    lattice: &Lattice,
    mut should_bond: impl FnMut(usize, usize) -> bool,
    csd: Option<&mut [u64]>,
) -> (Vec<u32>, Vec<u8>) {
    let n_spins = lattice.n_spins;
    let mut parent: Vec<u32> = (0..n_spins as u32).collect();
    let mut rank = vec![0u8; n_spins];

    for i in 0..n_spins {
        for d in 0..lattice.n_neighbors {
            if should_bond(i, d) {
                let j = lattice.neighbor(i, d, true);
                union(&mut parent, &mut rank, i as u32, j as u32);
            }
        }
    }

    if let Some(hist) = csd {
        for i in 0..n_spins {
            parent[i] = find(&mut parent, i as u32);
        }
        let mut counts = vec![0u32; n_spins];
        for i in 0..n_spins {
            counts[parent[i] as usize] += 1;
        }
        for &c in &counts {
            if c > 0 {
                hist[c as usize] += 1;
            }
        }
    }

    (parent, rank)
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

    fn bfs_bond_closure(
        bonds: &HashSet<(usize, usize)>,
    ) -> impl FnMut(usize, usize, usize, bool) -> bool + '_ {
        |site, nb, d, fwd| {
            let src = if fwd { site } else { nb };
            bonds.contains(&(src, d))
        }
    }

    #[test]
    fn test_bfs_bond_based() {
        let lattice = lattice_4x4();
        let n = lattice.n_spins;
        let bonds = bond_set();

        // Seed at 0 — should grow {0, 1, 3, 4}
        let mut in_cluster = vec![false; n];
        let mut stack = Vec::new();
        bfs_cluster(
            &lattice,
            0,
            &mut in_cluster,
            &mut stack,
            bfs_bond_closure(&bonds),
        );
        let cluster: HashSet<usize> = (0..n).filter(|&i| in_cluster[i]).collect();
        assert_eq!(cluster, [0, 1, 3, 4].into_iter().collect());

        // Seed at 10 — should grow {10, 11, 14}
        in_cluster.fill(false);
        stack.clear();
        bfs_cluster(
            &lattice,
            10,
            &mut in_cluster,
            &mut stack,
            bfs_bond_closure(&bonds),
        );
        let cluster: HashSet<usize> = (0..n).filter(|&i| in_cluster[i]).collect();
        assert_eq!(cluster, [10, 11, 14].into_iter().collect());

        // Seed at 7 — isolated singleton
        in_cluster.fill(false);
        stack.clear();
        bfs_cluster(
            &lattice,
            7,
            &mut in_cluster,
            &mut stack,
            bfs_bond_closure(&bonds),
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

    fn bfs_site_closure(
        sites: &HashSet<usize>,
    ) -> impl FnMut(usize, usize, usize, bool) -> bool + '_ {
        |site, nb, _d, fwd| {
            let src = if fwd { site } else { nb };
            sites.contains(&src)
        }
    }

    #[test]
    fn test_bfs_site_based() {
        let lattice = lattice_4x4();
        let n = lattice.n_spins;
        let sites = active_sites();

        // Seed at 0 — should grow {0, 1, 3, 4, 7} (3→7 adds site 7)
        let mut in_cluster = vec![false; n];
        let mut stack = Vec::new();
        bfs_cluster(
            &lattice,
            0,
            &mut in_cluster,
            &mut stack,
            bfs_site_closure(&sites),
        );
        let cluster: HashSet<usize> = (0..n).filter(|&i| in_cluster[i]).collect();
        assert_eq!(cluster, [0, 1, 3, 4, 7].into_iter().collect());

        // Seed at 10 — should grow {10, 11, 14}
        in_cluster.fill(false);
        stack.clear();
        bfs_cluster(
            &lattice,
            10,
            &mut in_cluster,
            &mut stack,
            bfs_site_closure(&sites),
        );
        let cluster: HashSet<usize> = (0..n).filter(|&i| in_cluster[i]).collect();
        assert_eq!(cluster, [10, 11, 14].into_iter().collect());
    }

    #[test]
    fn test_uf_bond_based() {
        let lattice = lattice_4x4();
        let n = lattice.n_spins;
        let bonds = bond_set();

        let (mut parent, _) = uf_bonds(&lattice, |i, d| bonds.contains(&(i, d)), None);

        // Flatten parents for easy inspection
        for i in 0..n {
            parent[i] = find(&mut parent, i as u32);
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
        for i in 0..n {
            if !clustered.contains(&i) {
                assert_eq!(parent[i], i as u32, "site {i} should be a singleton");
            }
        }
    }

    #[test]
    fn test_uf_site_based() {
        let lattice = lattice_4x4();
        let n = lattice.n_spins;
        let sites = active_sites();

        let (mut parent, _) = uf_bonds(&lattice, |i, _d| sites.contains(&i), None);

        for i in 0..n {
            parent[i] = find(&mut parent, i as u32);
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
        for i in 0..n {
            if !clustered.contains(&i) {
                assert_eq!(parent[i], i as u32, "site {i} should be a singleton");
            }
        }
    }

    #[test]
    fn test_uf_csd_bond_based() {
        let lattice = lattice_4x4();
        let n = lattice.n_spins;
        let bonds = bond_set();

        let mut hist = vec![0u64; n + 1];
        let _ = uf_bonds(&lattice, |i, d| bonds.contains(&(i, d)), Some(&mut hist));

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

        let mut hist = vec![0u64; n + 1];
        let _ = uf_bonds(&lattice, |i, _d| sites.contains(&i), Some(&mut hist));

        // Clusters: size 5 ×1, size 3 ×1, size 1 ×8
        assert_eq!(hist[1], 8);
        assert_eq!(hist[3], 1);
        assert_eq!(hist[5], 1);
        assert_eq!(hist.iter().sum::<u64>(), 10);
    }
}
