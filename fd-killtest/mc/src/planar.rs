//! Exact <s_u s_v> on planar graphs for outer-face pairs, one sign sample at a time.
//!
//! Construction: every vertex of degree >= 4 is split into a chain of degree-3 vertices joined by
//! weight-1 edges (infinite coupling); each vertex of degree d <= 3 becomes a gadget whose ports are
//! matched internally in exactly one way for every even set of externally matched ports
//! (d=1: port-centre edge; d=2: port-port edge; d=3: star from a centre plus one port-port edge).
//! Perfect matchings of the resulting Fisher graph H are then in bijection with even subgraphs of G,
//! with weight prod tanh(K_e). A Kasteleyn orientation (spanning tree + dual-tree peeling on the
//! rotation system) gives Pf(K) = +- sum over even subgraphs.
//!
//! An auxiliary outer-face edge u-v of weight w adds w * sum_{boundary {u,v}}; with K0 the matrix at
//! w = 0, Pf(K(w))/Pf(K0) = 1 + w (K0^{-1})_{qp} for the aux long edge oriented p -> q, hence
//! <s_u s_v> = (K0^{-1})_{qp}. All aux edges are present (at weight 0) in one embedding.

use crate::graph::Graph;
use faer::linalg::solvers::Solve;
use faer::sparse::linalg::solvers::{Lu, SymbolicLu};
use faer::sparse::{Argsort, Pair, SparseColMat, SymbolicSparseColMat};
use faer::Mat;
use std::collections::VecDeque;
use std::f64::consts::PI;

#[derive(Clone, Copy, PartialEq, Debug)]
enum GE {
    Orig(u32),
    Aux(u32),
    X,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum HVal {
    Orig(u32),
    One,
    Aux(u32),
}

pub struct Fisher {
    pub nn: usize,
    n_hedges: usize,
    /// (row, col, sign, value source) for each stored entry pair; entry (a,b) = s*val, (b,a) = -s*val.
    entries: Vec<(u32, u32, f64, HVal)>,
    /// per aux pair: (p, q) with <s_u s_v> = (K^{-1})_{q p}
    pub aux: Vec<(u32, u32)>,
    symbolic: SymbolicSparseColMat<usize>,
    argsort: Argsort<usize>,
    symlu: SymbolicLu<usize>,
}

impl Fisher {
    /// `pairs` must be outer-face pairs whose chords outside the box do not interleave.
    pub fn new(g: &Graph, pairs: &[(u32, u32)]) -> Fisher {
        assert!(g.planar);
        // ---- half-edges of G + aux, with angles
        // half-edge h: (vertex, angle, gedge index); gedges: kind
        let mut gedges: Vec<GE> = Vec::new();
        let mut hv: Vec<u32> = Vec::new();
        let mut hang: Vec<f64> = Vec::new();
        for (e, &(a, b)) in g.edges.iter().enumerate() {
            gedges.push(GE::Orig(e as u32));
            let (pa, pb) = (g.pos[a as usize], g.pos[b as usize]);
            hv.push(a);
            hang.push((pb[1] - pa[1]).atan2(pb[0] - pa[0]));
            hv.push(b);
            hang.push((pa[1] - pb[1]).atan2(pa[0] - pb[0]));
        }
        let p = g.perim_len;
        for (k, &(u, v)) in pairs.iter().enumerate() {
            gedges.push(GE::Aux(k as u32));
            for (x, y) in [(u, v), (v, u)] {
                let (sx, sy) = (g.perim[x as usize], g.perim[y as usize]);
                assert!(!sx.is_nan() && !sy.is_nan(), "aux pair not on the box boundary");
                let delta = (sy - sx).rem_euclid(p);
                hv.push(x);
                hang.push(g.out_angle[x as usize] + (PI / 3.0) * (0.5 - delta / p));
            }
        }
        let nh = hv.len();
        let mut rot_g: Vec<Vec<u32>> = vec![Vec::new(); g.n];
        for h in 0..nh {
            rot_g[hv[h] as usize].push(h as u32);
        }
        for r in rot_g.iter_mut() {
            r.sort_by(|&x, &y| {
                let ax = hang[x as usize].rem_euclid(2.0 * PI);
                let ay = hang[y as usize].rem_euclid(2.0 * PI);
                ax.partial_cmp(&ay).unwrap()
            });
            for w in r.windows(2) {
                let d = (hang[w[0] as usize].rem_euclid(2.0 * PI) - hang[w[1] as usize].rem_euclid(2.0 * PI)).abs();
                assert!(d > 1e-9, "coincident half-edge angles");
            }
        }
        // ---- split to max degree 3: G'-half-edges are the original ones plus x half-edges
        // gh_edge[h] = gedge index; halves of gedge k are 2k, 2k+1 for orig/aux (by construction).
        let mut gh_edge: Vec<u32> = (0..nh as u32).map(|h| h / 2).collect();
        let mut gverts: Vec<Vec<u32>> = Vec::new();
        for r in &rot_g {
            let d = r.len();
            if d <= 3 {
                if d > 0 {
                    gverts.push(r.clone());
                }
                continue;
            }
            let base = gh_edge.len() as u32; // x_k halves: base + 2k (minus side), base + 2k + 1 (plus side)
            for _ in 0..d - 3 {
                let k = gedges.len() as u32;
                gedges.push(GE::X);
                gh_edge.push(k);
                gh_edge.push(k);
            }
            let xm = |k: usize| base + 2 * k as u32;
            let xp = |k: usize| base + 2 * k as u32 + 1;
            gverts.push(vec![r[0], r[1], xm(0)]);
            for k in 1..d - 3 {
                gverts.push(vec![xp(k - 1), r[k + 1], xm(k)]);
            }
            gverts.push(vec![xp(d - 4), r[d - 2], r[d - 1]]);
        }
        let ngh = gh_edge.len();
        // ---- Fisher graph H: nodes, edges, rotation
        let mut port = vec![u32::MAX; ngh];
        let mut nn = 0u32;
        let mut hedges: Vec<(u32, u32, HVal)> = Vec::new();
        let mut rot: Vec<Vec<u32>> = Vec::new();
        let mut new_node = |rot: &mut Vec<Vec<u32>>| {
            rot.push(Vec::new());
            nn += 1;
            nn - 1
        };
        for gv in &gverts {
            for &h in gv {
                port[h as usize] = new_node(&mut rot);
            }
        }
        // long edges first: one per G'-edge, between the ports of its two halves
        let mut halves: Vec<[u32; 2]> = vec![[u32::MAX; 2]; gedges.len()];
        for h in 0..ngh {
            let k = gh_edge[h] as usize;
            if halves[k][0] == u32::MAX {
                halves[k][0] = h as u32;
            } else {
                halves[k][1] = h as u32;
            }
        }
        let mut long_of = vec![u32::MAX; gedges.len()];
        for (k, ge) in gedges.iter().enumerate() {
            let [h0, h1] = halves[k];
            let val = match *ge {
                GE::Orig(e) => HVal::Orig(e),
                GE::Aux(a) => HVal::Aux(a),
                GE::X => HVal::One,
            };
            long_of[k] = hedges.len() as u32;
            hedges.push((port[h0 as usize], port[h1 as usize], val));
        }
        // half-edge id of an H-edge k from node x: 2k if x == a_k else 2k+1
        let he = |hedges: &Vec<(u32, u32, HVal)>, k: u32, x: u32| -> u32 {
            if hedges[k as usize].0 == x {
                2 * k
            } else {
                2 * k + 1
            }
        };
        for gv in &gverts {
            let ports: Vec<u32> = gv.iter().map(|&h| port[h as usize]).collect();
            let longs: Vec<u32> = gv.iter().map(|&h| long_of[gh_edge[h as usize] as usize]).collect();
            match gv.len() {
                1 => {
                    let c = new_node(&mut rot);
                    let i1 = hedges.len() as u32;
                    hedges.push((ports[0], c, HVal::One));
                    rot[ports[0] as usize] = vec![he(&hedges, longs[0], ports[0]), 2 * i1];
                    rot[c as usize] = vec![2 * i1 + 1];
                }
                2 => {
                    let i = hedges.len() as u32;
                    hedges.push((ports[0], ports[1], HVal::One));
                    rot[ports[0] as usize] = vec![he(&hedges, longs[0], ports[0]), 2 * i];
                    rot[ports[1] as usize] = vec![he(&hedges, longs[1], ports[1]), 2 * i + 1];
                }
                3 => {
                    let c = new_node(&mut rot);
                    let i0 = hedges.len() as u32;
                    hedges.push((c, ports[0], HVal::One));
                    hedges.push((c, ports[1], HVal::One));
                    hedges.push((c, ports[2], HVal::One));
                    hedges.push((ports[1], ports[2], HVal::One));
                    let (i1, i2, i3) = (i0 + 1, i0 + 2, i0 + 3);
                    rot[c as usize] = vec![2 * i0, 2 * i1, 2 * i2];
                    rot[ports[0] as usize] = vec![he(&hedges, longs[0], ports[0]), 2 * i0 + 1];
                    rot[ports[1] as usize] = vec![he(&hedges, longs[1], ports[1]), 2 * i3, 2 * i1 + 1];
                    rot[ports[2] as usize] = vec![he(&hedges, longs[2], ports[2]), 2 * i2 + 1, 2 * i3 + 1];
                }
                _ => unreachable!(),
            }
        }
        let nn = nn as usize;
        let ne = hedges.len();
        // ---- faces (face on the left): next(h: x->y) = CW-next of twin in rot[y]
        let mut pos_in_rot = vec![0u32; 2 * ne];
        for r in &rot {
            for (i, &h) in r.iter().enumerate() {
                pos_in_rot[h as usize] = i as u32;
            }
        }
        let head = |h: u32| -> u32 {
            let (a, b, _) = hedges[(h / 2) as usize];
            if h % 2 == 0 {
                b
            } else {
                a
            }
        };
        let mut face_of = vec![u32::MAX; 2 * ne];
        let mut faces: Vec<Vec<u32>> = Vec::new();
        for h0 in 0..2 * ne as u32 {
            if face_of[h0 as usize] != u32::MAX {
                continue;
            }
            let f = faces.len() as u32;
            let mut cyc = Vec::new();
            let mut h = h0;
            loop {
                face_of[h as usize] = f;
                cyc.push(h);
                let y = head(h);
                let t = h ^ 1;
                let r = &rot[y as usize];
                let i = pos_in_rot[t as usize] as usize;
                h = r[(i + r.len() - 1) % r.len()];
                if h == h0 {
                    break;
                }
            }
            faces.push(cyc);
        }
        let nf = faces.len();
        // connectivity + Euler check
        let mut seen = vec![false; nn];
        let mut parent_edge = vec![u32::MAX; nn];
        let mut q = VecDeque::new();
        seen[0] = true;
        q.push_back(0u32);
        let mut nseen = 1;
        while let Some(x) = q.pop_front() {
            for &h in &rot[x as usize] {
                let y = head(h);
                if !seen[y as usize] {
                    seen[y as usize] = true;
                    nseen += 1;
                    parent_edge[y as usize] = h / 2;
                    q.push_back(y);
                }
            }
        }
        assert_eq!(nseen, nn, "Fisher graph is disconnected (restrict G to one component)");
        assert_eq!(nn as i64 - ne as i64 + nf as i64, 2, "rotation system is not planar (Euler)");
        // ---- Kasteleyn orientation: orient[k] = +1 if a->b else -1
        let mut orient = vec![0i8; ne];
        for &pe in parent_edge.iter() {
            if pe != u32::MAX {
                orient[pe as usize] = 1;
            }
        }
        let mut pending = vec![0u32; nf];
        for k in 0..ne {
            if orient[k] == 0 {
                pending[face_of[2 * k] as usize] += 1;
                pending[face_of[2 * k + 1] as usize] += 1;
            }
        }
        let root = 0usize;
        let mut fq: VecDeque<usize> = (0..nf).filter(|&f| f != root && pending[f] == 1).collect();
        // clockwise w.r.t. face-on-left traversal: edge oriented against the half-edge direction
        let is_cw = |orient: &Vec<i8>, h: u32| -> bool {
            let o = orient[(h / 2) as usize];
            (h % 2 == 0 && o == -1) || (h % 2 == 1 && o == 1)
        };
        while let Some(f) = fq.pop_front() {
            if pending[f] != 1 {
                continue;
            }
            let mut cw = 0;
            let mut hp = u32::MAX;
            for &h in &faces[f] {
                if orient[(h / 2) as usize] == 0 {
                    hp = h;
                } else if is_cw(&orient, h) {
                    cw += 1;
                }
            }
            let k = (hp / 2) as usize;
            // want total cw odd: if cw even, make hp clockwise (oriented against hp)
            let along = if cw % 2 == 0 { -1 } else { 1 };
            orient[k] = if hp % 2 == 0 { along } else { -along };
            for f2 in [face_of[2 * k] as usize, face_of[2 * k + 1] as usize] {
                pending[f2] -= 1;
                if f2 != f && f2 != root && pending[f2] == 1 {
                    fq.push_back(f2);
                }
            }
        }
        assert!(orient.iter().all(|&o| o != 0), "Kasteleyn peeling left edges unoriented");
        for (f, cyc) in faces.iter().enumerate() {
            if f == root {
                continue;
            }
            let cw = cyc.iter().filter(|&&h| is_cw(&orient, h)).count();
            assert!(cw % 2 == 1, "face {f} violates the Kasteleyn condition");
        }
        // ---- matrix pattern
        let mut entries = Vec::new();
        let mut aux = vec![(0u32, 0u32); pairs.len()];
        for (k, &(a, b, val)) in hedges.iter().enumerate() {
            let s = orient[k] as f64;
            match val {
                HVal::Aux(i) => {
                    aux[i as usize] = if orient[k] == 1 { (a, b) } else { (b, a) };
                }
                _ => entries.push((a, b, s, val)),
            }
        }
        let mut idx = Vec::with_capacity(2 * entries.len());
        for &(a, b, _, _) in &entries {
            idx.push(Pair::new(a as usize, b as usize));
            idx.push(Pair::new(b as usize, a as usize));
        }
        let (symbolic, argsort) = SymbolicSparseColMat::try_new_from_indices(nn, nn, &idx).unwrap();
        let symlu = SymbolicLu::try_new(symbolic.as_ref()).unwrap();
        Fisher { nn, n_hedges: ne, entries, aux, symbolic, argsort, symlu }
    }

    pub fn n_hedges(&self) -> usize {
        self.n_hedges
    }

    /// <s_u s_v> for every aux pair, given w_e = tanh(K_e) on the edges of G.
    pub fn correlations(&self, w: &[f64]) -> Vec<f64> {
        let mut vals = Vec::with_capacity(2 * self.entries.len());
        for &(_, _, s, val) in &self.entries {
            let x = match val {
                HVal::Orig(e) => w[e as usize],
                HVal::One => 1.0,
                HVal::Aux(_) => unreachable!(),
            };
            vals.push(s * x);
            vals.push(-s * x);
        }
        let mat = SparseColMat::<usize, f64>::new_from_argsort(self.symbolic.clone(), &self.argsort, &vals).unwrap();
        let lu = Lu::try_new_with_symbolic(self.symlu.clone(), mat.as_ref()).unwrap();
        let na = self.aux.len();
        let mut rhs = Mat::<f64>::zeros(self.nn, na);
        for (k, &(p, _)) in self.aux.iter().enumerate() {
            rhs[(p as usize, k)] = 1.0;
        }
        lu.solve_in_place(rhs.as_mut());
        self.aux.iter().enumerate().map(|(k, &(_, q))| rhs[(q as usize, k)]).collect()
    }
}
