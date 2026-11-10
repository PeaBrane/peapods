//! Two-replica Monte Carlo for E_eps <s_u s_v>^2 on general graphs with +-J couplings:
//! Metropolis sweeps plus parallel tempering, two independent replicas per temperature.
//! Per disorder sample the estimator is m1 * m2, the product of the time averages of s_u s_v
//! in the two replicas (unbiased for <s_u s_v>^2 once equilibrated).

use crate::graph::Graph;
use rand::Rng;
use rand::RngCore;
use rand_xoshiro::Xoshiro256PlusPlus;

pub struct GlassSystem {
    pub n: usize,
    off: Vec<u32>,
    nbr: Vec<u32>,
    eid: Vec<u32>,
    dmax: usize,
}

impl GlassSystem {
    pub fn new(g: &Graph) -> Self {
        let mut off = vec![0u32];
        let mut nbr = Vec::new();
        let mut eid = Vec::new();
        let mut dmax = 0;
        for v in 0..g.n {
            for &(w, e) in &g.adj[v] {
                nbr.push(w);
                eid.push(e);
            }
            dmax = dmax.max(g.adj[v].len());
            off.push(nbr.len() as u32);
        }
        GlassSystem { n: g.n, off, nbr, eid, dmax }
    }
}

pub struct SampleOut {
    /// [target][image] estimates of <s_u s_v>^2 from the whole measurement window
    pub full: Vec<Vec<f64>>,
    /// same from the second half only (equilibration check)
    pub late: Vec<Vec<f64>>,
    pub swap_acc: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
pub fn run_sample(
    sys: &GlassSystem,
    signs: &[i8],
    betas: &[f64],
    targets: &[usize],
    warm: usize,
    meas: usize,
    images: &[(u32, u32)],
    rng: &mut Xoshiro256PlusPlus,
) -> SampleOut {
    let n = sys.n;
    let nt = betas.len();
    let j: Vec<i8> = sys.eid.iter().map(|&e| signs[e as usize]).collect();
    let thr: Vec<Vec<u32>> = betas
        .iter()
        .map(|&b| {
            (0..=sys.dmax)
                .map(|k| {
                    let a = (-2.0 * b * k as f64).exp();
                    if a >= 1.0 {
                        u32::MAX
                    } else {
                        (a * 4294967296.0) as u32
                    }
                })
                .collect()
        })
        .collect();
    // configs: replica r, slot c in 0..nt; perm[r][t] = slot at temperature t
    let mut spins: Vec<Vec<i8>> = (0..2 * nt).map(|_| (0..n).map(|_| if rng.gen::<bool>() { 1 } else { -1 }).collect()).collect();
    let energy = |s: &[i8]| -> i64 {
        let mut e = 0i64;
        for v in 0..n {
            for k in sys.off[v] as usize..sys.off[v + 1] as usize {
                let w = sys.nbr[k] as usize;
                if w > v {
                    e -= (j[k] as i64) * (s[v] as i64) * (s[w] as i64);
                }
            }
        }
        e
    };
    let mut en: Vec<i64> = spins.iter().map(|s| energy(s)).collect();
    let mut perm: Vec<Vec<usize>> = (0..2).map(|r| (0..nt).map(|t| r * nt + t).collect()).collect();
    let nimg = images.len();
    let ntg = targets.len();
    let mut acc = vec![vec![0i64; ntg * nimg]; 2];
    let mut acc_late = vec![vec![0i64; ntg * nimg]; 2];
    let mut swaps = vec![0u64; nt.saturating_sub(1)];
    let mut tries = vec![0u64; nt.saturating_sub(1)];
    let total = warm + meas;
    for it in 0..total {
        for r in 0..2 {
            for t in 0..nt {
                let c = perm[r][t];
                let s = &mut spins[c];
                let th = &thr[t];
                let mut de_tot = 0i64;
                for v in 0..n {
                    let mut h = 0i32;
                    for k in sys.off[v] as usize..sys.off[v + 1] as usize {
                        h += (j[k] as i32) * (s[sys.nbr[k] as usize] as i32);
                    }
                    let sh = (s[v] as i32) * h;
                    if sh <= 0 || rng.next_u32() < th[sh as usize] {
                        s[v] = -s[v];
                        de_tot += 2 * sh as i64;
                    }
                }
                en[c] += de_tot;
            }
            // tempering: alternate even/odd neighbour pairs
            let start = it % 2;
            let mut t = start;
            while t + 1 < nt {
                let (ca, cb) = (perm[r][t], perm[r][t + 1]);
                let d = (betas[t] - betas[t + 1]) * ((en[ca] - en[cb]) as f64);
                tries[t] += 1;
                if d >= 0.0 || rng.gen::<f64>() < d.exp() {
                    perm[r].swap(t, t + 1);
                    swaps[t] += 1;
                }
                t += 2;
            }
        }
        if it >= warm {
            let late = it >= warm + meas / 2;
            for r in 0..2 {
                for (ti, &t) in targets.iter().enumerate() {
                    let s = &spins[perm[r][t]];
                    for (i, &(u, v)) in images.iter().enumerate() {
                        let x = (s[u as usize] * s[v as usize]) as i64;
                        acc[r][ti * nimg + i] += x;
                        if late {
                            acc_late[r][ti * nimg + i] += x;
                        }
                    }
                }
            }
        }
    }
    let m = meas as f64;
    let ml = (meas - meas / 2) as f64;
    let mut full = vec![vec![0.0; nimg]; ntg];
    let mut late = vec![vec![0.0; nimg]; ntg];
    for ti in 0..ntg {
        for i in 0..nimg {
            let k = ti * nimg + i;
            full[ti][i] = (acc[0][k] as f64 / m) * (acc[1][k] as f64 / m);
            late[ti][i] = (acc_late[0][k] as f64 / ml) * (acc_late[1][k] as f64 / ml);
        }
    }
    let swap_acc = swaps.iter().zip(&tries).map(|(&a, &b)| a as f64 / b.max(1) as f64).collect();
    SampleOut { full, late, swap_acc }
}

/// PT ladder containing every target beta, refined geometrically so that consecutive betas differ
/// by at most the ratio `rmax`.
pub fn ladder(targets: &[f64], rmax: f64) -> (Vec<f64>, Vec<usize>) {
    let mut t: Vec<f64> = targets.to_vec();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t.dedup();
    let mut out = vec![t[0]];
    for w in t.windows(2) {
        let (a, b) = (w[0], w[1]);
        let k = (((b / a).ln() / rmax.ln()).ceil() as usize).max(1);
        for i in 1..k {
            out.push(a * (b / a).powf(i as f64 / k as f64));
        }
        out.push(b);
    }
    let idx = targets
        .iter()
        .map(|&b| out.iter().position(|&x| (x - b).abs() < 1e-12).unwrap())
        .collect();
    (out, idx)
}
