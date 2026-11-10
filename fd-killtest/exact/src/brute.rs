//! Brute force on tiny graphs: all edge subsets for the arboreal gas, all signs and all spins for Q.

use crate::lattice::{Graph, UnionFind};

/// P^AG(u <-> v) by enumerating every edge subset.
pub fn ag(g: &Graph, u: usize, v: usize, beta: f64) -> f64 {
    let m = g.edges.len();
    assert!(m <= 26);
    let y: Vec<f64> = g.mag.iter().map(|&k| (beta * k).sinh().powi(2)).collect();
    let (mut z, mut zc) = (0.0, 0.0);
    'sub: for s in 0u64..(1u64 << m) {
        let mut uf = UnionFind::new(g.n);
        let mut w = 1.0;
        for e in 0..m {
            if s >> e & 1 == 1 {
                let (a, b) = g.edges[e];
                if !uf.union(a, b) {
                    continue 'sub;
                }
                w *= y[e];
            }
        }
        z += w;
        if uf.find(u) == uf.find(v) {
            zc += w;
        }
    }
    zc / z
}

/// Q = E_eps <s_u s_v>^2 for every pair, by enumerating every sign and every spin configuration.
pub fn q_all(g: &Graph, pairs: &[(usize, usize)], beta: f64) -> Vec<f64> {
    let (n, m) = (g.n, g.edges.len());
    assert!(n + m <= 30);
    let np = pairs.len();
    let mut tot = vec![0.0; np];
    let mut num = vec![0.0; np];
    for s in 0u64..(1u64 << m) {
        let k: Vec<f64> = (0..m).map(|e| if s >> e & 1 == 0 { beta * g.mag[e] } else { -beta * g.mag[e] }).collect();
        let mut z = 0.0;
        num.iter_mut().for_each(|x| *x = 0.0);
        for sp in 0u64..(1u64 << n) {
            let mut h = 0.0;
            for (e, &(a, b)) in g.edges.iter().enumerate() {
                if ((sp >> a) ^ (sp >> b)) & 1 == 0 {
                    h += k[e];
                } else {
                    h -= k[e];
                }
            }
            let w = h.exp();
            z += w;
            for (p, &(u, v)) in pairs.iter().enumerate() {
                if ((sp >> u) ^ (sp >> v)) & 1 == 0 {
                    num[p] += w;
                } else {
                    num[p] -= w;
                }
            }
        }
        for p in 0..np {
            let c = num[p] / z;
            tot[p] += c * c;
        }
    }
    tot.iter().map(|x| x / (1u64 << m) as f64).collect()
}
