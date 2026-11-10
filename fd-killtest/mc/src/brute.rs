//! Brute force on tiny graphs: forests, spins and signs enumerated exactly.

use crate::graph::Graph;

fn find(p: &mut [u32], x: u32) -> u32 {
    let mut r = x;
    while p[r as usize] != r {
        r = p[r as usize];
    }
    let mut y = x;
    while p[y as usize] != r {
        let nx = p[y as usize];
        p[y as usize] = r;
        y = nx;
    }
    r
}

/// All forests as edge bitmasks (m <= 30).
pub fn forests(g: &Graph) -> Vec<u32> {
    let m = g.edges.len();
    assert!(m <= 26);
    let mut out = Vec::new();
    let mut p = vec![0u32; g.n];
    'outer: for mask in 0u32..(1u32 << m) {
        for (i, x) in p.iter_mut().enumerate() {
            *x = i as u32;
        }
        for e in 0..m {
            if mask >> e & 1 == 1 {
                let (a, b) = g.edges[e];
                let (ra, rb) = (find(&mut p, a), find(&mut p, b));
                if ra == rb {
                    continue 'outer;
                }
                p[ra as usize] = rb;
            }
        }
        out.push(mask);
    }
    out
}

pub fn connected_in(g: &Graph, mask: u32, u: u32, v: u32) -> bool {
    let mut p: Vec<u32> = (0..g.n as u32).collect();
    for e in 0..g.edges.len() {
        if mask >> e & 1 == 1 {
            let (a, b) = g.edges[e];
            let (ra, rb) = (find(&mut p, a), find(&mut p, b));
            if ra != rb {
                p[ra as usize] = rb;
            }
        }
    }
    find(&mut p, u) == find(&mut p, v)
}

/// Streaming exact P^AG(u <-> v) (no forest list in memory; m <= 30).
pub fn ag_exact_stream(g: &Graph, y: f64, u: u32, v: u32) -> f64 {
    let m = g.edges.len();
    assert!(m <= 30);
    let (mut z, mut num) = (0.0f64, 0.0f64);
    let mut p = vec![0u32; g.n];
    let yp: Vec<f64> = (0..=m).map(|k| y.powi(k as i32)).collect();
    'outer: for mask in 0u32..(1u32 << m) {
        for (i, x) in p.iter_mut().enumerate() {
            *x = i as u32;
        }
        for e in 0..m {
            if mask >> e & 1 == 1 {
                let (a, b) = g.edges[e];
                let (ra, rb) = (find(&mut p, a), find(&mut p, b));
                if ra == rb {
                    continue 'outer;
                }
                p[ra as usize] = rb;
            }
        }
        let w = yp[mask.count_ones() as usize];
        z += w;
        if find(&mut p, u) == find(&mut p, v) {
            num += w;
        }
    }
    num / z
}

/// Exact P^AG(u <-> v) with uniform activity y.
pub fn ag_exact(g: &Graph, y: f64, pairs: &[(u32, u32)]) -> Vec<f64> {
    let fs = forests(g);
    let mut z = 0.0;
    let mut num = vec![0.0; pairs.len()];
    for &f in &fs {
        let w = y.powi(f.count_ones() as i32);
        z += w;
        for (k, &(u, v)) in pairs.iter().enumerate() {
            if connected_in(g, f, u, v) {
                num[k] += w;
            }
        }
    }
    num.iter().map(|x| x / z).collect()
}

/// <s_u s_v> for real couplings J_e by spin enumeration (n <= 24).
pub fn corr_exact(g: &Graph, j: &[f64], pairs: &[(u32, u32)]) -> Vec<f64> {
    let n = g.n;
    assert!(n <= 24);
    // energies with spin 0 fixed to +1 (global flip symmetry)
    let mut z = 0.0f64;
    let mut num = vec![0.0f64; pairs.len()];
    let mut emax = f64::NEG_INFINITY;
    let mut es = Vec::with_capacity(1 << (n - 1));
    for s in 0u32..(1u32 << (n - 1)) {
        let spin = |v: u32| -> f64 {
            if v == 0 {
                1.0
            } else if s >> (v - 1) & 1 == 1 {
                -1.0
            } else {
                1.0
            }
        };
        let mut e = 0.0;
        for (k, &(a, b)) in g.edges.iter().enumerate() {
            e += j[k] * spin(a) * spin(b);
        }
        emax = emax.max(e);
        es.push(e);
    }
    for (s, &e) in es.iter().enumerate() {
        let s = s as u32;
        let spin = |v: u32| -> f64 {
            if v == 0 {
                1.0
            } else if s >> (v - 1) & 1 == 1 {
                -1.0
            } else {
                1.0
            }
        };
        let w = (e - emax).exp();
        z += w;
        for (k, &(u, v)) in pairs.iter().enumerate() {
            num[k] += w * spin(u) * spin(v);
        }
    }
    num.iter().map(|x| x / z).collect()
}

/// Exact Q = E_eps <s_u s_v>^2 over all 2^m sign configurations (+-beta couplings).
pub fn q_exact(g: &Graph, beta: f64, pairs: &[(u32, u32)]) -> Vec<f64> {
    let m = g.edges.len();
    assert!(m <= 20);
    let mut q = vec![0.0; pairs.len()];
    let mut j = vec![0.0; m];
    for mask in 0u32..(1u32 << m) {
        for (e, x) in j.iter_mut().enumerate() {
            *x = if mask >> e & 1 == 1 { -beta } else { beta };
        }
        let c = corr_exact(g, &j, pairs);
        for k in 0..pairs.len() {
            q[k] += c[k] * c[k];
        }
    }
    q.iter().map(|x| x / (1u64 << m) as f64).collect()
}
