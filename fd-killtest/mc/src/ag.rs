//! Arboreal gas Markov chain: weight prod_{e in F} y on spanning forests F (y = sinh^2 beta).
//!
//! Moves:
//! * Sweeny heat-bath on one edge given the rest: present with probability p = y/(1+y) if its
//!   endpoints are in different trees of F \ {e}, absent otherwise (link-cut tree connectivity).
//!   A sweep visits every edge once in a fixed order; each step leaves the measure invariant.
//! * Tree refresh (exact Gibbs step for the trees given the vertex partition): conditioned on the
//!   partition into trees, the trees are independent uniform spanning trees of the induced blocks
//!   G[B], resampled by Wilson's algorithm. It does not change the partition but moves the cut
//!   structure that the bond moves act on.

use crate::graph::Graph;
use crate::lct::Lct;
use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub struct AgChain<'g> {
    pub g: &'g Graph,
    pub p: f64,
    pub present: Vec<bool>,
    lct: Lct,
    pub rng: Xoshiro256PlusPlus,
    comp: Vec<u32>,
    in_tree: Vec<bool>,
    next: Vec<(u32, u32)>,
}

impl<'g> AgChain<'g> {
    pub fn new(g: &'g Graph, beta: f64, seed: u64) -> Self {
        let y = beta.sinh().powi(2);
        Self::with_y(g, y, seed)
    }

    pub fn with_y(g: &'g Graph, y: f64, seed: u64) -> Self {
        AgChain {
            g,
            p: y / (1.0 + y),
            present: vec![false; g.edges.len()],
            lct: Lct::new(g.n),
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
            comp: vec![0; g.n],
            in_tree: vec![false; g.n],
            next: vec![(u32::MAX, u32::MAX); g.n],
        }
    }

    /// One Sweeny sweep over all edges in index order.
    pub fn sweep(&mut self) {
        let p = self.p;
        for (e, &(a, b)) in self.g.edges.iter().enumerate() {
            let r: f64 = self.rng.gen();
            if self.present[e] {
                if r >= p {
                    self.lct.cut(a, b);
                    self.present[e] = false;
                }
            } else if r < p && !self.lct.connected(a, b) {
                self.lct.link(a, b);
                self.present[e] = true;
            }
        }
    }

    /// Single heat-bath step on edge e (used by the exact kernel tests).
    pub fn heat_bath_edge(&mut self, e: usize, r: f64) {
        let (a, b) = self.g.edges[e];
        if self.present[e] {
            if r >= self.p {
                self.lct.cut(a, b);
                self.present[e] = false;
            }
        } else if r < self.p && !self.lct.connected(a, b) {
            self.lct.link(a, b);
            self.present[e] = true;
        }
    }

    /// Resample every tree as a uniform spanning tree of its block (Wilson), keeping the partition.
    pub fn refresh_trees(&mut self) {
        self.wilson(false);
    }

    /// Start from a uniform spanning tree of each component of G.
    pub fn init_spanning_tree(&mut self) {
        self.wilson(true);
    }

    fn wilson(&mut self, whole_graph: bool) {
        let g = self.g;
        let n = g.n;
        // partition from the current forest
        const UNSET: u32 = u32::MAX;
        for c in self.comp.iter_mut() {
            *c = UNSET;
        }
        let mut blocks: Vec<(u32, u32)> = Vec::new(); // (start, len) into order
        let mut order: Vec<u32> = Vec::with_capacity(n);
        let mut nc = 0u32;
        for s in 0..n {
            if self.comp[s] != UNSET {
                continue;
            }
            let start = order.len() as u32;
            self.comp[s] = nc;
            order.push(s as u32);
            let mut i = start as usize;
            while i < order.len() {
                let x = order[i];
                i += 1;
                for &(w, e) in &g.adj[x as usize] {
                    if (whole_graph || self.present[e as usize]) && self.comp[w as usize] == UNSET {
                        self.comp[w as usize] = nc;
                        order.push(w);
                    }
                }
            }
            blocks.push((start, order.len() as u32 - start));
            nc += 1;
        }
        let mut parent = vec![u32::MAX; n];
        for x in self.present.iter_mut() {
            *x = false;
        }
        for &(start, len) in &blocks {
            let vs = &order[start as usize..(start + len) as usize];
            if len == 1 {
                continue;
            }
            for &v in vs {
                self.in_tree[v as usize] = false;
            }
            let root = vs[self.rng.gen_range(0..len as usize)];
            self.in_tree[root as usize] = true;
            for &v in vs {
                let mut u = v;
                while !self.in_tree[u as usize] {
                    let nb = &g.adj[u as usize];
                    let cu = self.comp[u as usize];
                    let (w, e) = loop {
                        let (w, e) = nb[self.rng.gen_range(0..nb.len())];
                        if self.comp[w as usize] == cu {
                            break (w, e);
                        }
                    };
                    self.next[u as usize] = (w, e);
                    u = w;
                }
                let mut u = v;
                while !self.in_tree[u as usize] {
                    self.in_tree[u as usize] = true;
                    let (w, e) = self.next[u as usize];
                    parent[u as usize] = w;
                    self.present[e as usize] = true;
                    u = w;
                }
            }
        }
        self.lct.load_parents(&parent);
    }

    #[inline]
    pub fn connected(&mut self, a: u32, b: u32) -> bool {
        self.lct.connected(a, b)
    }

    /// Estimator of P(u <-> v): for a pair joined by edge e, the conditional expectation given
    /// F \ {e} (p if e in F or if u, v are disconnected without e; 1 otherwise).
    #[inline]
    pub fn observe(&mut self, u: u32, v: u32, e: Option<u32>) -> f64 {
        match e {
            Some(e) => {
                if self.present[e as usize] {
                    self.p
                } else if self.lct.connected(u, v) {
                    1.0
                } else {
                    self.p
                }
            }
            None => self.lct.connected(u, v) as u8 as f64,
        }
    }

    /// Star Rao-Blackwell estimator of P(u <-> v): conditional expectation given F minus the
    /// edges at u. With u isolated, the edges from u into each tree of F' are chosen as "none or
    /// exactly one" (two would close a cycle through u), so P(u <-> v | F') = c y / (1 + c y) with
    /// c = number of neighbours of u in the tree of v. Averaged over the roles of u and v.
    pub fn observe_star(&mut self, u: u32, v: u32) -> f64 {
        0.5 * (self.star_one(u, v) + self.star_one(v, u))
    }

    fn star_one(&mut self, u: u32, v: u32) -> f64 {
        let g = self.g;
        let y = self.p / (1.0 - self.p);
        let mut cut: [u32; 16] = [0; 16];
        let mut nc = 0;
        for &(w, e) in &g.adj[u as usize] {
            if self.present[e as usize] {
                self.lct.cut(u, w);
                cut[nc] = w;
                nc += 1;
            }
        }
        let mut c = 0u32;
        for &(w, _) in &g.adj[u as usize] {
            if self.lct.connected(w, v) {
                c += 1;
            }
        }
        for &w in &cut[..nc] {
            self.lct.link(u, w);
        }
        let cy = c as f64 * y;
        cy / (1.0 + cy)
    }

    /// Double-star Rao-Blackwell estimator: conditional expectation of 1[u <-> v] given F minus
    /// every edge at u or v. For each tree T of the remainder with a_T edges to u and b_T edges to v,
    /// u and v can be joined through T (one edge each side) with weight a_T b_T y^2, and at most one
    /// such tree (or the direct edge u-v) may be used, so
    /// P = (y_uv + S) / (1 + y_uv + S), S = sum_T a_T b_T y^2 / (1 + (a_T + b_T) y).
    pub fn observe_star2(&mut self, u: u32, v: u32) -> f64 {
        let g = self.g;
        let y = self.p / (1.0 - self.p);
        let mut cut: [(u32, u32); 32] = [(0, 0); 32];
        let mut nc = 0;
        for x in [u, v] {
            for &(w, e) in &g.adj[x as usize] {
                if self.present[e as usize] && !cut[..nc].iter().any(|&(a, b)| (a, b) == (w, x)) {
                    self.lct.cut(x, w);
                    cut[nc] = (x, w);
                    nc += 1;
                }
            }
        }
        let mut ru: [u32; 16] = [0; 16];
        let mut nu = 0;
        let mut y_uv = 0.0;
        for &(w, _) in &g.adj[u as usize] {
            if w == v {
                y_uv = y;
            } else {
                ru[nu] = self.lct.find_root(w);
                nu += 1;
            }
        }
        let mut rv: [u32; 16] = [0; 16];
        let mut nv = 0;
        for &(x, _) in &g.adj[v as usize] {
            if x != u {
                rv[nv] = self.lct.find_root(x);
                nv += 1;
            }
        }
        let mut s = 0.0;
        let mut seen: [u32; 16] = [u32::MAX; 16];
        let mut ns = 0;
        for i in 0..nu {
            let t = ru[i];
            if seen[..ns].contains(&t) {
                continue;
            }
            seen[ns] = t;
            ns += 1;
            let a = ru[..nu].iter().filter(|&&r| r == t).count() as f64;
            let b = rv[..nv].iter().filter(|&&r| r == t).count() as f64;
            if b > 0.0 {
                s += a * b * y * y / (1.0 + (a + b) * y);
            }
        }
        for &(x, w) in &cut[..nc] {
            self.lct.link(x, w);
        }
        (y_uv + s) / (1.0 + y_uv + s)
    }

    /// Debug: compare LCT connectivity with BFS over `present` for `k` random pairs and check that
    /// `present` is a forest. Returns the number of disagreements (cycles count as one).
    pub fn check_consistency(&mut self, k: usize) -> usize {
        let g = self.g;
        let mut comp = vec![u32::MAX; g.n];
        let mut nc = 0u32;
        let mut nedges = 0usize;
        for s0 in 0..g.n {
            if comp[s0] != u32::MAX {
                continue;
            }
            comp[s0] = nc;
            let mut st = vec![s0 as u32];
            while let Some(x) = st.pop() {
                for &(w, e) in &g.adj[x as usize] {
                    if self.present[e as usize] && comp[w as usize] == u32::MAX {
                        comp[w as usize] = nc;
                        st.push(w);
                    }
                }
            }
            nc += 1;
        }
        for e in 0..g.edges.len() {
            if self.present[e] {
                nedges += 1;
            }
        }
        let mut bad = (nedges != g.n - nc as usize) as usize;
        for _ in 0..k {
            let a = self.rng.gen_range(0..g.n as u32);
            let b = if self.rng.gen::<bool>() {
                g.adj[a as usize].get(0).map(|x| x.0).unwrap_or(a)
            } else {
                self.rng.gen_range(0..g.n as u32)
            };
            if self.lct.connected(a, b) != (comp[a as usize] == comp[b as usize]) {
                bad += 1;
            }
        }
        bad
    }

    /// Load a forest given as an edge bitmask (tiny graphs, tests).
    pub fn load_forest(&mut self, mask: u32) {
        let mut parent = vec![u32::MAX; self.g.n];
        self.lct.load_parents(&parent);
        for (e, &(a, b)) in self.g.edges.iter().enumerate() {
            self.present[e] = mask >> e & 1 == 1;
            if self.present[e] {
                self.lct.link(a, b);
            }
        }
        parent.clear();
    }

    pub fn mask(&self) -> u32 {
        self.present.iter().enumerate().map(|(e, &x)| (x as u32) << e).sum()
    }
}
