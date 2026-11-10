//! Strip lattices w x L (square, triangular, honeycomb as a brick wall), free or periodic in the short
//! direction, with per-edge magnitudes. Vertex (x, y), x in [0, L) along the strip and y in [0, w)
//! across, has index x * w + y. Every edge (a, b), a < b, satisfies b - a <= win, so a sliding window
//! of `win` consecutive vertices is a valid frontier for both transfer computations.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Square,
    Tri,
    Honey,
}

impl Kind {
    pub fn parse(s: &str) -> Option<Kind> {
        match s {
            "square" | "sq" => Some(Kind::Square),
            "tri" | "triangular" => Some(Kind::Tri),
            "honey" | "honeycomb" | "hex" => Some(Kind::Honey),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Kind::Square => "square",
            Kind::Tri => "tri",
            Kind::Honey => "honey",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Graph {
    pub n: usize,
    pub win: usize,
    /// (a, b) with a < b.
    pub edges: Vec<(usize, usize)>,
    /// Coupling magnitude multiplier: K_e = beta * mag_e. Zero-magnitude edges are removed.
    pub mag: Vec<f64>,
    /// For each vertex b: (edge index, neighbour a < b).
    pub back: Vec<Vec<(usize, usize)>>,
}

impl Graph {
    pub fn new(n: usize, raw: &[(usize, usize, f64)]) -> Graph {
        let mut edges = Vec::new();
        let mut mag = Vec::new();
        for &(a, b, m) in raw {
            assert!(a != b && a < n && b < n, "bad edge ({a},{b})");
            if m == 0.0 {
                continue;
            }
            edges.push((a.min(b), a.max(b)));
            mag.push(m);
        }
        let win = edges.iter().map(|&(a, b)| b - a).max().unwrap_or(1).max(1);
        let mut back = vec![Vec::new(); n];
        for (i, &(a, b)) in edges.iter().enumerate() {
            back[b].push((i, a));
        }
        for bl in back.iter_mut() {
            bl.sort_by_key(|&(_, a)| a);
        }
        Graph { n, win, edges, mag, back }
    }

    /// Relabel vertices by v -> n - 1 - v (180-degree rotation of a strip). Edge offsets are preserved.
    pub fn rotated(&self) -> Graph {
        let n = self.n;
        let raw: Vec<(usize, usize, f64)> =
            self.edges.iter().zip(&self.mag).map(|(&(a, b), &m)| (n - 1 - b, n - 1 - a, m)).collect();
        let mut g = Graph::new(n, &raw);
        g.win = g.win.max(self.win);
        g
    }

    pub fn components(&self) -> Vec<usize> {
        let mut uf = UnionFind::new(self.n);
        for &(a, b) in &self.edges {
            uf.union(a, b);
        }
        (0..self.n).map(|v| uf.find(v)).collect()
    }

    pub fn n_components(&self) -> usize {
        let c = self.components();
        let mut seen = vec![false; self.n];
        let mut k = 0;
        for &r in &c {
            if !seen[r] {
                seen[r] = true;
                k += 1;
            }
        }
        k
    }

    pub fn cyclomatic(&self) -> usize {
        self.edges.len() + self.n_components() - self.n
    }

    /// Spanning forest (BFS order by edge index): true for tree edges.
    pub fn spanning_forest(&self) -> Vec<bool> {
        let mut uf = UnionFind::new(self.n);
        self.edges.iter().map(|&(a, b)| uf.union(a, b)).collect()
    }

    /// Graph distance by BFS (usize::MAX if disconnected).
    pub fn distance(&self, u: usize, v: usize) -> usize {
        let mut adj = vec![Vec::new(); self.n];
        for &(a, b) in &self.edges {
            adj[a].push(b);
            adj[b].push(a);
        }
        let mut d = vec![usize::MAX; self.n];
        let mut q = std::collections::VecDeque::new();
        d[u] = 0;
        q.push_back(u);
        while let Some(x) = q.pop_front() {
            for &y in &adj[x] {
                if d[y] == usize::MAX {
                    d[y] = d[x] + 1;
                    q.push_back(y);
                }
            }
        }
        d[v]
    }
}

pub struct UnionFind {
    p: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        UnionFind { p: (0..n).collect() }
    }
    pub fn find(&mut self, mut x: usize) -> usize {
        while self.p[x] != x {
            self.p[x] = self.p[self.p[x]];
            x = self.p[x];
        }
        x
    }
    /// Returns true if a and b were in different sets.
    pub fn union(&mut self, a: usize, b: usize) -> bool {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return false;
        }
        self.p[ra] = rb;
        true
    }
}

#[derive(Clone, Debug)]
pub struct Strip {
    pub kind: Kind,
    pub w: usize,
    pub l: usize,
    pub periodic: bool,
}

impl Strip {
    pub fn vid(&self, x: usize, y: usize) -> usize {
        x * self.w + y
    }
    pub fn n(&self) -> usize {
        self.w * self.l
    }
    pub fn bc(&self) -> &'static str {
        if self.periodic {
            "cyl"
        } else {
            "free"
        }
    }

    /// Edge list with unit magnitudes.
    pub fn raw_edges(&self) -> Vec<(usize, usize, f64)> {
        let (w, l) = (self.w, self.l);
        let mut e = Vec::new();
        let across = |x: usize, y: usize| -> bool {
            match self.kind {
                Kind::Honey => (x + y) % 2 == 0,
                _ => true,
            }
        };
        for x in 0..l {
            for y in 0..w {
                let v = self.vid(x, y);
                if y + 1 < w && across(x, y) {
                    e.push((v, self.vid(x, y + 1), 1.0));
                }
                if self.periodic && y + 1 == w && w > 2 && across(x, y) {
                    e.push((self.vid(x, 0), v, 1.0));
                }
                if x + 1 < l {
                    e.push((v, self.vid(x + 1, y), 1.0));
                }
                if self.kind == Kind::Tri && x + 1 < l {
                    if self.periodic {
                        // diagonals (x, y)-(x+1, y+1 mod w); window w + 1
                        if w > 2 || y + 1 < w {
                            e.push((v, self.vid(x + 1, (y + 1) % w), 1.0));
                        }
                    } else if y + 1 < w {
                        // diagonals (x, y+1)-(x+1, y); window w
                        e.push((self.vid(x, y + 1), self.vid(x + 1, y), 1.0));
                    }
                }
            }
        }
        if self.periodic && self.kind == Kind::Honey {
            assert!(w % 2 == 0, "honeycomb cylinder needs even width");
        }
        e
    }

    pub fn graph(&self) -> Graph {
        Graph::new(self.n(), &self.raw_edges())
    }

    /// Bond-diluted graph with random magnitudes: each edge kept with probability `keep`, magnitude
    /// drawn uniformly from [mlo, mhi].
    pub fn diluted(&self, keep: f64, mlo: f64, mhi: f64, seed: u64) -> Graph {
        let mut rng = crate::rng::Rng::new(seed ^ 0xD11u64);
        let raw: Vec<(usize, usize, f64)> = self
            .raw_edges()
            .into_iter()
            .map(|(a, b, _)| {
                let k = rng.uniform() < keep;
                let m = mlo + (mhi - mlo) * rng.uniform();
                (a, b, if k { m } else { 0.0 })
            })
            .collect();
        let mut g = Graph::new(self.n(), &raw);
        g.win = g.win.max(self.graph().win);
        g
    }

    /// Standard pair set: (label, u, v). Pairs that fall outside the strip are omitted.
    pub fn pairs(&self) -> Vec<(String, usize, usize)> {
        let (w, l) = (self.w, self.l);
        let xm = if l >= 2 { (l - 1) / 2 } else { 0 };
        let ym = (w - 1) / 2;
        let mut p: Vec<(String, usize, usize)> = Vec::new();
        let mut add = |name: &str, x1: usize, y1: usize, x2: usize, y2: usize| {
            if x1 < l && x2 < l && y1 < w && y2 < w && (x1, y1) != (x2, y2) {
                p.push((name.to_string(), self.vid(x1, y1), self.vid(x2, y2)));
            }
        };
        add("corner-adj-a", 0, 0, 0, 1);
        add("corner-adj-l", 0, 0, 1, 0);
        if w >= 3 {
            add("bdry-adj-a", 0, 1, 0, 2);
        }
        if l >= 3 {
            add("bdry-adj-l", 1, 0, 2, 0);
        }
        if l >= 4 {
            add("bdry-adj-lmid", xm, 0, xm + 1, 0);
            add("bdry-d2-l", xm.saturating_sub(1), 0, xm.saturating_sub(1) + 2, 0);
        }
        if l >= 5 {
            add("bdry-d3-l", xm.saturating_sub(1), 0, xm.saturating_sub(1) + 3, 0);
        }
        if w >= 4 {
            add("bdry-d2-a", 0, 0, 0, 2);
            add("bdry-d3-a", 0, 0, 0, 3);
        }
        add("short-end-to-end", 0, 0, 0, w - 1);
        add("end-to-end-l", 0, 0, l - 1, 0);
        add("opp-corners", 0, 0, l - 1, w - 1);
        if w >= 3 && l >= 3 {
            add("centre-adj-l", xm, ym, xm + 1, ym);
            add("centre-adj-a", xm, ym, xm, ym + 1);
            for d in 2..=4usize {
                if xm + d < l {
                    add(&format!("centre-d{d}-l"), xm.saturating_sub(d / 2), ym, xm.saturating_sub(d / 2) + d, ym);
                }
            }
        }
        if w >= 5 && l >= 3 {
            add("centre-d2-a", xm, ym - 1, xm, ym + 1);
        }
        p
    }
}
