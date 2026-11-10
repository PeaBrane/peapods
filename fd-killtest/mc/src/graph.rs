//! Finite graphs with free boundaries: square, triangular (rhombus), honeycomb (brick wall),
//! diluted square and cubic boxes, plus the pair classes probed by the kill test.
//!
//! 2D lattices carry a straight-line planar embedding (`pos`) and, for vertices on the box
//! boundary, a perimeter coordinate and an outward direction; the planar engine uses these to
//! draw auxiliary outer-face edges u-v.

use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Clone)]
pub struct Graph {
    pub n: usize,
    pub edges: Vec<(u32, u32)>,
    pub pos: Vec<[f64; 2]>,
    pub planar: bool,
    /// Perimeter coordinate (CCW) of box-boundary vertices, NaN in the interior.
    pub perim: Vec<f64>,
    pub perim_len: f64,
    /// Outward direction (radians) at box-boundary vertices, NaN in the interior.
    pub out_angle: Vec<f64>,
    /// Lattice coordinates, for labels.
    pub coord: Vec<[i32; 3]>,
    pub adj: Vec<Vec<(u32, u32)>>,
}

impl Graph {
    pub fn build_adj(&mut self) {
        let mut adj = vec![Vec::new(); self.n];
        for (e, &(a, b)) in self.edges.iter().enumerate() {
            adj[a as usize].push((b, e as u32));
            adj[b as usize].push((a, e as u32));
        }
        self.adj = adj;
    }

    pub fn edge_between(&self, u: u32, v: u32) -> Option<u32> {
        self.adj[u as usize].iter().find(|&&(w, _)| w == v).map(|&(_, e)| e)
    }

    /// Components as labels.
    pub fn components(&self) -> (Vec<u32>, usize) {
        let mut comp = vec![u32::MAX; self.n];
        let mut nc = 0u32;
        let mut q = VecDeque::new();
        for s in 0..self.n {
            if comp[s] != u32::MAX {
                continue;
            }
            comp[s] = nc;
            q.push_back(s as u32);
            while let Some(x) = q.pop_front() {
                for &(w, _) in &self.adj[x as usize] {
                    if comp[w as usize] == u32::MAX {
                        comp[w as usize] = nc;
                        q.push_back(w);
                    }
                }
            }
            nc += 1;
        }
        (comp, nc as usize)
    }

    /// Bridges (Tarjan, iterative).
    pub fn bridges(&self) -> Vec<bool> {
        let n = self.n;
        let mut is_bridge = vec![false; self.edges.len()];
        let mut disc = vec![u32::MAX; n];
        let mut low = vec![0u32; n];
        let mut t = 0u32;
        for s in 0..n {
            if disc[s] != u32::MAX {
                continue;
            }
            // stack of (vertex, parent edge, next adj index)
            let mut st: Vec<(u32, u32, usize)> = vec![(s as u32, u32::MAX, 0)];
            disc[s] = t;
            low[s] = t;
            t += 1;
            while let Some(&mut (x, pe, ref mut i)) = st.last_mut() {
                let xa = &self.adj[x as usize];
                if *i < xa.len() {
                    let (w, e) = xa[*i];
                    *i += 1;
                    if e == pe {
                        continue;
                    }
                    if disc[w as usize] == u32::MAX {
                        disc[w as usize] = t;
                        low[w as usize] = t;
                        t += 1;
                        st.push((w, e, 0));
                    } else {
                        low[x as usize] = low[x as usize].min(disc[w as usize]);
                    }
                } else {
                    st.pop();
                    if let Some(&(px, _, _)) = st.last() {
                        low[px as usize] = low[px as usize].min(low[x as usize]);
                        if low[x as usize] > disc[px as usize] {
                            is_bridge[pe as usize] = true;
                        }
                    }
                }
            }
        }
        is_bridge
    }

    /// Restrict to the vertex subset `keep` (re-indexed in order); returns the old->new map.
    pub fn induced(&self, keep: &[bool]) -> (Graph, Vec<u32>) {
        let mut map = vec![u32::MAX; self.n];
        let mut k = 0u32;
        for v in 0..self.n {
            if keep[v] {
                map[v] = k;
                k += 1;
            }
        }
        let pick = |src: &[f64]| -> Vec<f64> { (0..self.n).filter(|&v| keep[v]).map(|v| src[v]).collect() };
        let mut g = Graph {
            n: k as usize,
            edges: self
                .edges
                .iter()
                .filter(|&&(a, b)| keep[a as usize] && keep[b as usize])
                .map(|&(a, b)| (map[a as usize], map[b as usize]))
                .collect(),
            pos: (0..self.n).filter(|&v| keep[v]).map(|v| self.pos[v]).collect(),
            planar: self.planar,
            perim: pick(&self.perim),
            perim_len: self.perim_len,
            out_angle: pick(&self.out_angle),
            coord: (0..self.n).filter(|&v| keep[v]).map(|v| self.coord[v]).collect(),
            adj: Vec::new(),
        };
        g.build_adj();
        (g, map)
    }
}

#[derive(Clone, Debug)]
pub struct PairClass {
    pub name: String,
    pub images: Vec<(u32, u32)>,
}

pub struct Instance {
    pub lattice: String,
    pub size: String,
    pub graph: Graph,
    pub classes: Vec<PairClass>,
}

type Map2 = Box<dyn Fn(i64, i64) -> (i64, i64)>;

/// 2D lattice on index box [0,lx) x [0,ly); vertex id = j*lx + i; real position = i*a1 + j*a2.
fn lattice2d(
    lx: usize,
    ly: usize,
    offsets: &[(i64, i64)],
    keep: &dyn Fn(i64, i64, i64, i64) -> bool,
    a1: [f64; 2],
    a2: [f64; 2],
) -> Graph {
    let n = lx * ly;
    let id = |i: i64, j: i64| (j as usize * lx + i as usize) as u32;
    let mut edges = Vec::new();
    for j in 0..ly as i64 {
        for i in 0..lx as i64 {
            for &(di, dj) in offsets {
                let (ii, jj) = (i + di, j + dj);
                if ii < 0 || jj < 0 || ii >= lx as i64 || jj >= ly as i64 {
                    continue;
                }
                if keep(i, j, di, dj) {
                    edges.push((id(i, j), id(ii, jj)));
                }
            }
        }
    }
    let (lxf, lyf) = ((lx - 1) as f64, (ly - 1) as f64);
    let plen = 2.0 * lxf + 2.0 * lyf;
    let mut pos = vec![[0.0; 2]; n];
    let mut perim = vec![f64::NAN; n];
    let mut out_angle = vec![f64::NAN; n];
    let mut coord = vec![[0i32; 3]; n];
    for j in 0..ly {
        for i in 0..lx {
            let v = j * lx + i;
            let (fi, fj) = (i as f64, j as f64);
            pos[v] = [fi * a1[0] + fj * a2[0], fi * a1[1] + fj * a2[1]];
            coord[v] = [i as i32, j as i32, 0];
            let on_b = i == 0 || j == 0 || i == lx - 1 || j == ly - 1;
            if !on_b {
                continue;
            }
            perim[v] = if j == 0 {
                fi
            } else if i == lx - 1 {
                lxf + fj
            } else if j == ly - 1 {
                lxf + lyf + (lxf - fi)
            } else {
                2.0 * lxf + lyf + (lyf - fj)
            };
            let dx = if i == 0 { -1.0 } else if i == lx - 1 { 1.0 } else { 0.0 };
            let dy = if j == 0 { -1.0 } else if j == ly - 1 { 1.0 } else { 0.0 };
            let ox = dx * a1[0] + dy * a2[0];
            let oy = dx * a1[1] + dy * a2[1];
            out_angle[v] = oy.atan2(ox);
        }
    }
    let mut g = Graph {
        n,
        edges,
        pos,
        planar: true,
        perim,
        perim_len: plen,
        out_angle,
        coord,
        adj: Vec::new(),
    };
    g.build_adj();
    g
}

fn edge_set(g: &Graph) -> HashSet<(u32, u32)> {
    g.edges.iter().map(|&(a, b)| (a.min(b), a.max(b))).collect()
}

/// Images of base pairs under symmetry maps (each map is checked to be a graph automorphism).
fn images2d(g: &Graph, lx: usize, maps: &[Map2], base: &[(i64, i64, i64, i64)]) -> Vec<(u32, u32)> {
    let es = edge_set(g);
    let id = |i: i64, j: i64| (j as usize * lx + i as usize) as u32;
    for m in maps {
        for &(a, b) in &g.edges {
            let (ai, aj) = ((a as usize % lx) as i64, (a as usize / lx) as i64);
            let (bi, bj) = ((b as usize % lx) as i64, (b as usize / lx) as i64);
            let (pa, qa) = m(ai, aj);
            let (pb, qb) = m(bi, bj);
            let (x, y) = (id(pa, qa), id(pb, qb));
            assert!(es.contains(&(x.min(y), x.max(y))), "symmetry map is not an automorphism");
        }
    }
    let mut out: Vec<(u32, u32)> = Vec::new();
    let mut seen = HashSet::new();
    for &(i1, j1, i2, j2) in base {
        for m in maps {
            let (a, b) = (m(i1, j1), m(i2, j2));
            let (u, v) = (id(a.0, a.1), id(b.0, b.1));
            if seen.insert((u.min(v), u.max(v))) {
                out.push((u, v));
            }
        }
    }
    out
}

/// Keep images whose outer-face chords do not interleave with already accepted ones.
pub fn planar_compatible(g: &Graph, accepted: &mut Vec<(u32, u32)>, cand: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let p = g.perim_len;
    let inside = |x: f64, a: f64, b: f64| {
        // strictly inside the CCW arc a->b
        let d = (x - a).rem_euclid(p);
        let l = (b - a).rem_euclid(p);
        d > 1e-9 && d < l - 1e-9
    };
    let mut out = Vec::new();
    for &(u, v) in cand {
        let (su, sv) = (g.perim[u as usize], g.perim[v as usize]);
        if su.is_nan() || sv.is_nan() {
            continue;
        }
        if accepted.iter().any(|&(a, b)| (a, b) == (u, v) || (a, b) == (v, u)) {
            out.push((u, v));
            continue;
        }
        let ok = accepted.iter().all(|&(a, b)| {
            let (sa, sb) = (g.perim[a as usize], g.perim[b as usize]);
            let ia = inside(sa, su, sv) && (sa - su).abs() > 1e-9 && (sa - sv).abs() > 1e-9;
            let ib = inside(sb, su, sv) && (sb - su).abs() > 1e-9 && (sb - sv).abs() > 1e-9;
            let oa = !inside(sa, su, sv) && (sa - su).abs() > 1e-9 && (sa - sv).abs() > 1e-9;
            let ob = !inside(sb, su, sv) && (sb - su).abs() > 1e-9 && (sb - sv).abs() > 1e-9;
            !((ia && ob) || (ib && oa))
        });
        if ok {
            accepted.push((u, v));
            out.push((u, v));
        }
    }
    out
}

fn d4_maps(l: i64) -> Vec<Map2> {
    let m = l - 1;
    vec![
        Box::new(|i, j| (i, j)),
        Box::new(move |i, j| (m - i, j)),
        Box::new(move |i, j| (i, m - j)),
        Box::new(move |i, j| (m - i, m - j)),
        Box::new(|i, j| (j, i)),
        Box::new(move |i, j| (m - j, i)),
        Box::new(move |i, j| (j, m - i)),
        Box::new(move |i, j| (m - j, m - i)),
    ]
}

fn finish_planar_classes(g: &Graph, raw: Vec<(String, Vec<(u32, u32)>)>) -> Vec<PairClass> {
    let mut accepted = Vec::new();
    raw.into_iter()
        .filter_map(|(name, cand)| {
            let images = planar_compatible(g, &mut accepted, &cand);
            (!images.is_empty()).then_some(PairClass { name, images })
        })
        .collect()
}

pub fn square(l: usize) -> Instance {
    let g = lattice2d(l, l, &[(1, 0), (0, 1)], &|_, _, _, _| true, [1.0, 0.0], [0.0, 1.0]);
    let li = l as i64;
    let maps = d4_maps(li);
    let m = (li / 2 - 2).max(0);
    let mut raw = vec![("corner-adj".to_string(), images2d(&g, l, &maps, &[(0, 0, 1, 0)]))];
    for d in 1..=4 {
        if m + d <= li - 1 {
            raw.push((format!("bnd-d{d}"), images2d(&g, l, &maps[..1], &[(m, 0, m + d, 0)])));
        }
    }
    // the four side images, generated by rotations only so that they stay nested
    for (k, (name, imgs)) in raw.iter_mut().enumerate() {
        if k == 0 || !name.starts_with("bnd") {
            continue;
        }
        let (a, b) = imgs[0];
        let rots: Vec<Map2> = vec![
            Box::new(move |i, j| (li - 1 - j, i)),
            Box::new(move |i, j| (li - 1 - i, li - 1 - j)),
            Box::new(move |i, j| (j, li - 1 - i)),
        ];
        let (ai, aj) = ((a as usize % l) as i64, (a as usize / l) as i64);
        let (bi, bj) = ((b as usize % l) as i64, (b as usize / l) as i64);
        for r in &rots {
            let (p, q) = (r(ai, aj), r(bi, bj));
            imgs.push(((p.1 as usize * l + p.0 as usize) as u32, (q.1 as usize * l + q.0 as usize) as u32));
        }
    }
    raw.push(("opp-corner".to_string(), vec![(0, (l * l - 1) as u32)]));
    let classes = finish_planar_classes(&g, raw);
    Instance { lattice: "square".into(), size: format!("{l}x{l}"), graph: g, classes }
}

pub fn triangular(l: usize) -> Instance {
    let s3 = 3f64.sqrt() / 2.0;
    let g = lattice2d(l, l, &[(1, 0), (0, 1), (1, 1)], &|_, _, _, _| true, [1.0, 0.0], [-0.5, s3]);
    let li = l as i64;
    let mm = li - 1;
    let maps: Vec<Map2> = vec![
        Box::new(|i, j| (i, j)),
        Box::new(move |i, j| (mm - i, mm - j)),
        Box::new(move |i, j| (mm - j, mm - i)),
        Box::new(|i, j| (j, i)),
    ];
    let m = (li / 2 - 2).max(0);
    let mut raw = vec![
        ("acute-corner-adj".to_string(), images2d(&g, l, &maps, &[(mm, 0, mm - 1, 0)])),
        ("obtuse-corner-adj".to_string(), images2d(&g, l, &maps, &[(0, 0, 1, 0)])),
    ];
    for d in 1..=4 {
        if m + d <= li - 1 {
            raw.push((format!("bnd-d{d}"), images2d(&g, l, &maps, &[(m, 0, m + d, 0)])));
        }
    }
    raw.push(("opp-corner".to_string(), vec![((mm) as u32, (mm as usize * l) as u32)]));
    let classes = finish_planar_classes(&g, raw);
    Instance { lattice: "triangular".into(), size: format!("{l}x{l}"), graph: g, classes }
}

pub fn honeycomb(l: usize) -> Instance {
    assert!(l % 2 == 0, "honeycomb brick wall needs even L");
    let g = lattice2d(
        l,
        l,
        &[(1, 0), (0, 1)],
        &|i, j, di, _| di == 1 || (i + j) % 2 == 0,
        [1.0, 0.0],
        [0.0, 1.0],
    );
    let li = l as i64;
    let mm = li - 1;
    let maps: Vec<Map2> = vec![Box::new(|i, j| (i, j)), Box::new(move |i, j| (i, mm - j))];
    let m = (li / 2 - 2).max(0);
    let mut raw = vec![
        ("corner-adj-h".to_string(), images2d(&g, l, &maps, &[(0, 0, 1, 0)])),
        ("corner-adj-v".to_string(), images2d(&g, l, &maps, &[(0, 0, 0, 1)])),
    ];
    for d in 1..=4 {
        if m + d <= li - 1 {
            raw.push((format!("bnd-d{d}"), images2d(&g, l, &maps, &[(m, 0, m + d, 0)])));
        }
    }
    raw.push(("opp-corner".to_string(), vec![(0, (l * l - 1) as u32)]));
    let classes = finish_planar_classes(&g, raw);
    Instance { lattice: "honeycomb".into(), size: format!("{l}x{l}"), graph: g, classes }
}

/// Bond-diluted square box restricted to its largest component; every pair image is its own
/// class (no symmetry), bridges are skipped (Q = AG exactly there).
pub fn diluted_square(l: usize, occ: f64, real: u64) -> Instance {
    let full = square(l);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xD11u64 ^ (real.wrapping_mul(0x9E3779B97F4A7C15)) ^ ((occ * 1e6) as u64));
    let mut g = full.graph.clone();
    g.edges.retain(|_| rng.gen::<f64>() < occ);
    g.build_adj();
    let (comp, nc) = g.components();
    let mut sizes = vec![0usize; nc];
    for &c in &comp {
        sizes[c as usize] += 1;
    }
    let big = (0..nc).max_by_key(|&c| sizes[c]).unwrap() as u32;
    let keep: Vec<bool> = comp.iter().map(|&c| c == big).collect();
    let (h, map) = g.induced(&keep);
    let br = h.bridges();
    let mut raw = Vec::new();
    for c in &full.classes {
        for (k, &(u, v)) in c.images.iter().enumerate() {
            let (a, b) = (map[u as usize], map[v as usize]);
            if a == u32::MAX || b == u32::MAX {
                continue;
            }
            if let Some(e) = full.graph.edge_between(u, v) {
                let _ = e;
                match h.edge_between(a, b) {
                    Some(he) if !br[he as usize] => {}
                    _ => continue,
                }
            }
            raw.push((format!("{}#{k}", c.name), vec![(a, b)]));
        }
    }
    let classes = finish_planar_classes(&h, raw);
    Instance {
        lattice: format!("dil{:.2}", occ),
        size: format!("{l}x{l}r{real}"),
        graph: h,
        classes,
    }
}

pub fn cubic(lx: usize, ly: usize, lz: usize) -> Instance {
    let n = lx * ly * lz;
    let id = |i: i64, j: i64, k: i64| ((k as usize * ly + j as usize) * lx + i as usize) as u32;
    let mut edges = Vec::new();
    let mut coord = vec![[0i32; 3]; n];
    for k in 0..lz as i64 {
        for j in 0..ly as i64 {
            for i in 0..lx as i64 {
                coord[id(i, j, k) as usize] = [i as i32, j as i32, k as i32];
                if i + 1 < lx as i64 {
                    edges.push((id(i, j, k), id(i + 1, j, k)));
                }
                if j + 1 < ly as i64 {
                    edges.push((id(i, j, k), id(i, j + 1, k)));
                }
                if k + 1 < lz as i64 {
                    edges.push((id(i, j, k), id(i, j, k + 1)));
                }
            }
        }
    }
    let mut g = Graph {
        n,
        edges,
        pos: vec![[0.0; 2]; n],
        planar: false,
        perim: vec![f64::NAN; n],
        perim_len: 0.0,
        out_angle: vec![f64::NAN; n],
        coord,
        adj: Vec::new(),
    };
    g.build_adj();
    let mut classes = Vec::new();
    if lx == ly && ly == lz {
        let l = lx as i64;
        let mm = l - 1;
        let es = edge_set(&g);
        let perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
        let mut syms: Vec<([usize; 3], [bool; 3])> = Vec::new();
        for p in perms {
            for f in 0..8 {
                syms.push((p, [f & 1 != 0, f & 2 != 0, f & 4 != 0]));
            }
        }
        let apply = |s: &([usize; 3], [bool; 3]), c: [i64; 3]| -> u32 {
            let mut o = [0i64; 3];
            for d in 0..3 {
                let x = c[s.0[d]];
                o[d] = if s.1[d] { mm - x } else { x };
            }
            id(o[0], o[1], o[2])
        };
        for s in &syms {
            for &(a, b) in &g.edges {
                let ca = g.coord[a as usize];
                let cb = g.coord[b as usize];
                let x = apply(s, [ca[0] as i64, ca[1] as i64, ca[2] as i64]);
                let y = apply(s, [cb[0] as i64, cb[1] as i64, cb[2] as i64]);
                assert!(es.contains(&(x.min(y), x.max(y))));
            }
        }
        let m = l / 2 - 1;
        let base: Vec<(&str, [i64; 3], [i64; 3])> = vec![
            ("corner-adj", [0, 0, 0], [1, 0, 0]),
            ("edge-adj", [m, 0, 0], [m + 1, 0, 0]),
            ("edge-d2", [m, 0, 0], [m + 2, 0, 0]),
            ("face-adj", [m, m, 0], [m + 1, m, 0]),
            ("opp-corner", [0, 0, 0], [mm, mm, mm]),
        ];
        for (name, a, b) in base {
            if b.iter().any(|&x| x > mm) {
                continue;
            }
            let mut seen = HashSet::new();
            let mut imgs = Vec::new();
            for s in &syms {
                let (u, v) = (apply(s, a), apply(s, b));
                if seen.insert((u.min(v), u.max(v))) {
                    imgs.push((u, v));
                }
            }
            classes.push(PairClass { name: name.into(), images: imgs });
        }
    }
    Instance { lattice: "cubic".into(), size: format!("{lx}x{ly}x{lz}"), graph: g, classes }
}

/// Free 2D grid W x H with vertex id r*W + c (Z1 convention), pairs given explicitly.
pub fn grid_with_pairs(w: usize, h: usize, pairs: &[(u32, u32)]) -> Instance {
    let g = lattice2d(w, h, &[(1, 0), (0, 1)], &|_, _, _, _| true, [1.0, 0.0], [0.0, 1.0]);
    let raw = pairs.iter().map(|&(u, v)| (format!("{u}-{v}"), vec![(u, v)])).collect();
    let classes = finish_planar_classes(&g, raw);
    Instance { lattice: "square".into(), size: format!("{w}x{h}"), graph: g, classes }
}

/// Decay classes along the boundary: corner u and the vertex at distance r along a side, for
/// each r in `rs`, with all symmetric images whose outer-face chords stay non-crossing.
pub fn decay(lattice: &str, l: usize, rs: &[usize]) -> Instance {
    let li = l as i64;
    let mm = li - 1;
    let (mut inst, maps, base): (Instance, Vec<Map2>, Box<dyn Fn(i64) -> (i64, i64, i64, i64)>) = match lattice {
        // rotations only: each corner paired along its CCW side, so chords never cross
        "square" => (
            square(l),
            vec![
                Box::new(|i, j| (i, j)),
                Box::new(move |i, j| (mm - j, i)),
                Box::new(move |i, j| (mm - i, mm - j)),
                Box::new(move |i, j| (j, mm - i)),
            ],
            Box::new(|r| (0, 0, r, 0)),
        ),
        "triangular" => (
            triangular(l),
            vec![
                Box::new(|i, j| (i, j)),
                Box::new(move |i, j| (mm - i, mm - j)),
                Box::new(move |i, j| (mm - j, mm - i)),
                Box::new(|i, j| (j, i)),
            ],
            Box::new(move |r| (mm, 0, mm - r, 0)),
        ),
        _ => panic!("decay classes only for square and triangular"),
    };
    let raw = rs
        .iter()
        .filter(|&&r| r >= 1 && (r as i64) <= mm)
        .map(|&r| (format!("corner-r{r}"), images2d(&inst.graph, l, &maps, &[base(r as i64)])))
        .collect();
    inst.classes = finish_planar_classes(&inst.graph, raw);
    inst
}

pub fn by_name(lattice: &str, l: usize, occ: f64, real: u64) -> Instance {
    match lattice {
        "square" => square(l),
        "triangular" => triangular(l),
        "honeycomb" => honeycomb(l),
        "diluted" => diluted_square(l, occ, real),
        "cubic" => cubic(l, l, l),
        _ => panic!("unknown lattice {lattice}"),
    }
}

pub fn label(g: &Graph, v: u32) -> String {
    let c = g.coord[v as usize];
    if g.planar {
        format!("({};{})", c[0], c[1])
    } else {
        format!("({};{};{})", c[0], c[1], c[2])
    }
}

#[allow(dead_code)]
pub fn describe(inst: &Instance) -> String {
    let mut s = format!(
        "{} {}: n={} m={} classes:",
        inst.lattice,
        inst.size,
        inst.graph.n,
        inst.graph.edges.len()
    );
    let mut h: HashMap<&str, usize> = HashMap::new();
    for c in &inst.classes {
        *h.entry(&c.name).or_default() += c.images.len();
        s += &format!(" {}[{}]", c.name, c.images.len());
    }
    s
}
