//! Exact <s_u s_v> for one sign configuration by a site-by-site transfer matrix over the 2^win spins
//! of the sliding window, and Q = E_eps <s_u s_v>^2 by sign sampling or gauge-reduced enumeration.
//!
//! Layout after inserting vertex t: bit p of the index holds the spin of vertex t - p (bit set means
//! spin -1). Inserting t drops vertex t - win (the old top bit) and pushes t into bit 0, so with
//! k = j >> 1 the new index is j = 2k + s_t and the old one is k + s_old * half.
//! F_t: weight of the edges among vertices <= t, summed over the spins that left the window.
//! B_t: weight of the edges with an endpoint > t, summed over the spins of vertices > t.
//! Z = sum_j F_t[j] B_t[j] for every t. Both sweeps are rescaled by their max at every step.

use crate::lattice::Graph;
use crate::rng::Rng;

#[derive(Clone, Debug)]
struct StepG {
    /// edge to the departing vertex t - win
    left: Option<usize>,
    /// (edge, bit position in k) for the other back edges
    others: Vec<(usize, u32)>,
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub n: usize,
    pub win: usize,
    half: usize,
    size: usize,
    steps: Vec<StepG>,
    pub pairs: Vec<(usize, usize)>,
    /// for each step v: list of (pair index, distance v - u) evaluated directly from F_v, B_v
    direct: Vec<Vec<(usize, usize)>>,
    /// propagated groups: (u, [(pair index, v)] sorted by v)
    groups: Vec<(usize, Vec<(usize, usize)>)>,
    f_slot: Vec<Option<usize>>,
    b_slot: Vec<Option<usize>>,
    nf: usize,
    nb: usize,
    m: usize,
    /// FDK_SLOW=1 selects the reference (per-element lookup) kernels
    slow: bool,
}

impl Plan {
    pub fn new(g: &Graph, pairs: &[(usize, usize)]) -> Plan {
        let w = g.win;
        assert!(w <= 24, "window too large for the spin transfer matrix");
        let size = 1usize << w;
        let half = size >> 1;
        let mut steps = Vec::with_capacity(g.n);
        for t in 0..g.n {
            let mut left = None;
            let mut others = Vec::new();
            for &(e, a) in &g.back[t] {
                let d = t - a;
                assert!(d >= 1 && d <= w);
                if d == w {
                    left = Some(e);
                } else {
                    others.push((e, (d - 1) as u32));
                }
            }
            assert!(others.len() <= 4);
            others.sort_by_key(|&(_, b)| b);
            steps.push(StepG { left, others });
        }
        let pairs: Vec<(usize, usize)> = pairs.iter().map(|&(a, b)| (a.min(b), a.max(b))).collect();
        let mut direct = vec![Vec::new(); g.n];
        let mut groups: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
        let mut need_f = vec![false; g.n];
        let mut need_b = vec![false; g.n];
        for (i, &(u, v)) in pairs.iter().enumerate() {
            assert!(u < v);
            need_b[v] = true;
            need_f[v] = true;
            if v - u < w {
                direct[v].push((i, v - u));
            } else {
                need_f[u] = true;
                match groups.iter_mut().find(|(gu, _)| *gu == u) {
                    Some((_, l)) => l.push((i, v)),
                    None => groups.push((u, vec![(i, v)])),
                }
            }
        }
        for (_, l) in groups.iter_mut() {
            l.sort_by_key(|&(_, v)| v);
        }
        let mut nf = 0;
        let f_slot = need_f
            .iter()
            .map(|&x| {
                if x {
                    nf += 1;
                    Some(nf - 1)
                } else {
                    None
                }
            })
            .collect();
        let mut nbk = 0;
        let b_slot = need_b
            .iter()
            .map(|&x| {
                if x {
                    nbk += 1;
                    Some(nbk - 1)
                } else {
                    None
                }
            })
            .collect();
        let slow = std::env::var("FDK_SLOW").map(|v| v == "1").unwrap_or(false);
        Plan { n: g.n, win: w, half, size, steps, pairs, direct, groups, f_slot, b_slot, nf, nb: nbk, m: g.edges.len(), slow }
    }

    pub fn work(&self) -> Work {
        Work {
            a: vec![0.0; self.size],
            b: vec![0.0; self.size],
            fs: vec![vec![0.0; self.size]; self.nf],
            bs: vec![vec![0.0; self.size]; self.nb],
            scale: vec![1.0; self.n],
            ex: vec![0.0; self.m],
        }
    }
}

pub struct Work {
    a: Vec<f64>,
    b: Vec<f64>,
    fs: Vec<Vec<f64>>,
    bs: Vec<Vec<f64>>,
    scale: Vec<f64>,
    ex: Vec<f64>,
}

#[inline(always)]
fn mbits(k: usize, bits: &[u32; 4], nbits: usize) -> usize {
    let mut m = 0;
    for i in 0..nbits {
        m |= ((k >> bits[i]) & 1) << i;
    }
    m
}

/// Diagonal factor tables for step t: dp[m] for s_t = +1, dm[m] for s_t = -1.
#[inline]
fn tables(st: &StepG, ex: &[f64], sc: f64, bits: &mut [u32; 4], dp: &mut [f64; 16], dm: &mut [f64; 16]) -> (usize, f64, f64) {
    let nbits = st.others.len();
    for (i, &(_, bpos)) in st.others.iter().enumerate() {
        bits[i] = bpos;
    }
    let full = (1usize << nbits) - 1;
    for m in 0..(1usize << nbits) {
        let mut f = 1.0;
        for (i, &(e, _)) in st.others.iter().enumerate() {
            // neighbour spin +1 if bit clear
            if m >> i & 1 == 0 {
                f *= ex[e];
            } else {
                f /= ex[e];
            }
        }
        dp[m] = f * sc;
    }
    // D(-, s) = D(+, flip s) exactly: keeps F and B bit-exactly flip-symmetric and the inserted
    // vectors exactly antisymmetric, so no symmetric component leaks into tiny correlations.
    for m in 0..(1usize << nbits) {
        dm[m] = dp[m ^ full];
    }
    let (ep, em) = match st.left {
        Some(e) => (ex[e], 1.0 / ex[e]),
        None => (1.0, 1.0),
    };
    (nbits, ep, em)
}

macro_rules! fwd_loop {
    ($inp:expr, $out:expr, $half:expr, $ep:expr, $em:expr, $dp:expr, $dm:expr, $mfn:expr) => {{
        let (lo, hi) = $inp.split_at($half);
        let mut mx = 0.0f64;
        for (k, ((o, &x0), &x1)) in $out.chunks_exact_mut(2).zip(lo.iter()).zip(hi.iter()).enumerate() {
            let m = $mfn(k);
            let p = ($ep * x0 + $em * x1) * $dp[m];
            let q = ($em * x0 + $ep * x1) * $dm[m];
            o[0] = p;
            o[1] = q;
            mx = mx.max(p.abs()).max(q.abs());
        }
        mx
    }};
}

macro_rules! bwd_loop {
    ($inp:expr, $out:expr, $half:expr, $ep:expr, $em:expr, $dp:expr, $dm:expr, $mfn:expr) => {{
        let (lo, hi) = $out.split_at_mut($half);
        let mut mx = 0.0f64;
        for (k, ((i, o0), o1)) in $inp.chunks_exact(2).zip(lo.iter_mut()).zip(hi.iter_mut()).enumerate() {
            let m = $mfn(k);
            let x = i[0] * $dp[m];
            let y = i[1] * $dm[m];
            let p = $ep * x + $em * y;
            let q = $em * x + $ep * y;
            *o0 = p;
            *o1 = q;
            mx = mx.max(p).max(q);
        }
        mx
    }};
}

fn fwd(inp: &[f64], out: &mut [f64], half: usize, ep: f64, em: f64, bits: &[u32; 4], nbits: usize, dp: &[f64; 16], dm: &[f64; 16]) -> f64 {
    match nbits {
        0 => fwd_loop!(inp, out, half, ep, em, dp, dm, |_k: usize| 0usize),
        1 => {
            let b0 = bits[0];
            fwd_loop!(inp, out, half, ep, em, dp, dm, |k: usize| (k >> b0) & 1)
        }
        2 => {
            let (b0, b1) = (bits[0], bits[1]);
            fwd_loop!(inp, out, half, ep, em, dp, dm, |k: usize| ((k >> b0) & 1) | (((k >> b1) & 1) << 1))
        }
        _ => fwd_loop!(inp, out, half, ep, em, dp, dm, |k: usize| mbits(k, bits, nbits)),
    }
}

fn bwd(inp: &[f64], out: &mut [f64], half: usize, ep: f64, em: f64, bits: &[u32; 4], nbits: usize, dp: &[f64; 16], dm: &[f64; 16]) -> f64 {
    match nbits {
        0 => bwd_loop!(inp, out, half, ep, em, dp, dm, |_k: usize| 0usize),
        1 => {
            let b0 = bits[0];
            bwd_loop!(inp, out, half, ep, em, dp, dm, |k: usize| (k >> b0) & 1)
        }
        2 => {
            let (b0, b1) = (bits[0], bits[1]);
            bwd_loop!(inp, out, half, ep, em, dp, dm, |k: usize| ((k >> b0) & 1) | (((k >> b1) & 1) << 1))
        }
        _ => bwd_loop!(inp, out, half, ep, em, dp, dm, |k: usize| mbits(k, bits, nbits)),
    }
}

/// Block kernels: with the neighbour bits sorted, the factor index is constant on blocks of
/// 2^(lowest bit >= 1) consecutive k except for bit 0, which alternates and is unrolled by two.
fn fwd_blk(inp: &[f64], out: &mut [f64], half: usize, ep: f64, em: f64, bits: &[u32; 4], nbits: usize, dp: &[f64; 16], dm: &[f64; 16]) {
    let has0 = nbits > 0 && bits[0] == 0;
    let fh = has0 as usize;
    let bs = if nbits > fh { 1usize << bits[fh] } else { half };
    let (lo, hi) = inp.split_at(half);
    for k0 in (0..half).step_by(bs) {
        let mut mh = 0usize;
        for i in fh..nbits {
            mh |= ((k0 >> bits[i]) & 1) << i;
        }
        let lb = &lo[k0..k0 + bs];
        let hb = &hi[k0..k0 + bs];
        let ob = &mut out[2 * k0..2 * (k0 + bs)];
        if has0 {
            let (p0, q0, p1, q1) = (dp[mh], dm[mh], dp[mh | 1], dm[mh | 1]);
            for ((o, a), b) in ob.chunks_exact_mut(4).zip(lb.chunks_exact(2)).zip(hb.chunks_exact(2)) {
                o[0] = (ep * a[0] + em * b[0]) * p0;
                o[1] = (em * a[0] + ep * b[0]) * q0;
                o[2] = (ep * a[1] + em * b[1]) * p1;
                o[3] = (em * a[1] + ep * b[1]) * q1;
            }
        } else {
            let (p, q) = (dp[mh], dm[mh]);
            for ((o, &a), &b) in ob.chunks_exact_mut(2).zip(lb).zip(hb) {
                o[0] = (ep * a + em * b) * p;
                o[1] = (em * a + ep * b) * q;
            }
        }
    }
}

fn bwd_blk(inp: &[f64], out: &mut [f64], half: usize, ep: f64, em: f64, bits: &[u32; 4], nbits: usize, dp: &[f64; 16], dm: &[f64; 16]) {
    let has0 = nbits > 0 && bits[0] == 0;
    let fh = has0 as usize;
    let bs = if nbits > fh { 1usize << bits[fh] } else { half };
    let (lo, hi) = out.split_at_mut(half);
    for k0 in (0..half).step_by(bs) {
        let mut mh = 0usize;
        for i in fh..nbits {
            mh |= ((k0 >> bits[i]) & 1) << i;
        }
        let ib = &inp[2 * k0..2 * (k0 + bs)];
        let lb = &mut lo[k0..k0 + bs];
        let hb = &mut hi[k0..k0 + bs];
        if has0 {
            let (p0, q0, p1, q1) = (dp[mh], dm[mh], dp[mh | 1], dm[mh | 1]);
            for ((i4, l2), h2) in ib.chunks_exact(4).zip(lb.chunks_exact_mut(2)).zip(hb.chunks_exact_mut(2)) {
                let (x0, y0, x1, y1) = (i4[0] * p0, i4[1] * q0, i4[2] * p1, i4[3] * q1);
                l2[0] = ep * x0 + em * y0;
                h2[0] = em * x0 + ep * y0;
                l2[1] = ep * x1 + em * y1;
                h2[1] = em * x1 + ep * y1;
            }
        } else {
            let (p, q) = (dp[mh], dm[mh]);
            for ((i2, l), h) in ib.chunks_exact(2).zip(lb.iter_mut()).zip(hb.iter_mut()) {
                let (x, y) = (i2[0] * p, i2[1] * q);
                *l = ep * x + em * y;
                *h = em * x + ep * y;
            }
        }
    }
}

#[inline]
fn vmax(v: &[f64]) -> f64 {
    let mut m = [0.0f64; 4];
    for c in v.chunks_exact(4) {
        for i in 0..4 {
            m[i] = if c[i] > m[i] { c[i] } else { m[i] };
        }
    }
    let mut r = m[0].max(m[1]).max(m[2]).max(m[3]);
    for &x in v.chunks_exact(4).remainder() {
        r = r.max(x);
    }
    r
}

/// Rescale the sweeps by their max every RESCALE steps (growth per step is at most ~e^{12}).
const RESCALE: usize = 4;

/// <s_u s_v> for every planned pair, for couplings J_e (signed, already multiplied by beta).
pub fn correlations(plan: &Plan, j: &[f64], wk: &mut Work, out: &mut [f64]) {
    let (n, half, size) = (plan.n, plan.half, plan.size);
    for (e, &x) in wk.ex.iter_mut().zip(j) {
        *e = x.exp();
    }
    let mut bits = [0u32; 4];
    let mut dp = [0f64; 16];
    let mut dm = [0f64; 16];
    // forward
    wk.a.iter_mut().for_each(|x| *x = 1.0);
    let mut prev_mx = 1.0;
    for t in 0..n {
        let sc = 1.0 / prev_mx;
        wk.scale[t] = sc;
        let (nbits, ep, em) = tables(&plan.steps[t], &wk.ex, sc, &mut bits, &mut dp, &mut dm);
        if plan.slow {
            prev_mx = fwd(&wk.a, &mut wk.b, half, ep, em, &bits, nbits, &dp, &dm);
        } else {
            fwd_blk(&wk.a, &mut wk.b, half, ep, em, &bits, nbits, &dp, &dm);
            prev_mx = if t % RESCALE == RESCALE - 1 { vmax(&wk.b) } else { 1.0 };
        }
        std::mem::swap(&mut wk.a, &mut wk.b);
        if let Some(s) = plan.f_slot[t] {
            wk.fs[s].copy_from_slice(&wk.a);
        }
    }
    // backward
    wk.a.iter_mut().for_each(|x| *x = 1.0);
    if let Some(s) = plan.b_slot[n - 1] {
        wk.bs[s].copy_from_slice(&wk.a);
    }
    let mut prev_mx = 1.0;
    for t in (1..n).rev() {
        let sc = 1.0 / prev_mx;
        let (nbits, ep, em) = tables(&plan.steps[t], &wk.ex, sc, &mut bits, &mut dp, &mut dm);
        if plan.slow {
            prev_mx = bwd(&wk.a, &mut wk.b, half, ep, em, &bits, nbits, &dp, &dm);
        } else {
            bwd_blk(&wk.a, &mut wk.b, half, ep, em, &bits, nbits, &dp, &dm);
            prev_mx = if t % RESCALE == 0 { vmax(&wk.b) } else { 1.0 };
        }
        std::mem::swap(&mut wk.a, &mut wk.b);
        if let Some(s) = plan.b_slot[t - 1] {
            wk.bs[s].copy_from_slice(&wk.a);
        }
    }
    // direct pairs
    for v in 0..n {
        if plan.direct[v].is_empty() {
            continue;
        }
        let f = &wk.fs[plan.f_slot[v].unwrap()];
        let b = &wk.bs[plan.b_slot[v].unwrap()];
        let z: f64 = f.iter().zip(b).map(|(x, y)| x * y).sum();
        for &(pi, d) in &plan.direct[v] {
            let mut num = 0.0;
            for jx in 0..size {
                let fb = f[jx] * b[jx];
                if ((jx ^ (jx >> d)) & 1) == 0 {
                    num += fb;
                } else {
                    num -= fb;
                }
            }
            out[pi] = num / z;
        }
    }
    // propagated groups
    for (u, list) in &plan.groups {
        let u = *u;
        let fu = &wk.fs[plan.f_slot[u].unwrap()];
        for jx in 0..size {
            wk.a[jx] = if jx & 1 == 0 { fu[jx] } else { -fu[jx] };
        }
        let mut li = 0;
        for t in (u + 1)..n {
            if li >= list.len() {
                break;
            }
            let (nbits, ep, em) = tables(&plan.steps[t], &wk.ex, wk.scale[t], &mut bits, &mut dp, &mut dm);
            if plan.slow {
                fwd(&wk.a, &mut wk.b, half, ep, em, &bits, nbits, &dp, &dm);
            } else {
                fwd_blk(&wk.a, &mut wk.b, half, ep, em, &bits, nbits, &dp, &dm);
            }
            std::mem::swap(&mut wk.a, &mut wk.b);
            while li < list.len() && list[li].1 == t {
                let pi = list[li].0;
                let f = &wk.fs[plan.f_slot[t].unwrap()];
                let b = &wk.bs[plan.b_slot[t].unwrap()];
                let mut z = 0.0;
                let mut num = 0.0;
                for jx in 0..size {
                    z += f[jx] * b[jx];
                    let gb = wk.a[jx] * b[jx];
                    if jx & 1 == 0 {
                        num += gb;
                    } else {
                        num -= gb;
                    }
                }
                out[pi] = num / z;
                li += 1;
            }
        }
    }
}

/// Accumulated sums of c^2 and c^4 per (beta, pair).
#[derive(Clone, Debug)]
pub struct Acc {
    pub nb: usize,
    pub np: usize,
    pub s1: Vec<f64>,
    pub s2: Vec<f64>,
    pub count: f64,
}

impl Acc {
    pub fn new(nb: usize, np: usize) -> Acc {
        Acc { nb, np, s1: vec![0.0; nb * np], s2: vec![0.0; nb * np], count: 0.0 }
    }
    pub fn add(&mut self, o: &Acc) {
        for i in 0..self.s1.len() {
            self.s1[i] += o.s1[i];
            self.s2[i] += o.s2[i];
        }
        self.count += o.count;
    }
    pub fn mean(&self, b: usize, p: usize) -> f64 {
        self.s1[b * self.np + p] / self.count
    }
    pub fn se(&self, b: usize, p: usize) -> f64 {
        let nn = self.count;
        if nn < 2.0 {
            return f64::NAN;
        }
        let m = self.mean(b, p);
        let var = ((self.s2[b * self.np + p] / nn - m * m) * nn / (nn - 1.0)).max(0.0);
        (var / nn).sqrt()
    }
}

fn eval_config(plan: &Plan, g: &Graph, betas: &[f64], eps: &[f64], wk: &mut Work, j: &mut [f64], c: &mut [f64], acc: &mut Acc) {
    let np = plan.pairs.len();
    for (bi, &beta) in betas.iter().enumerate() {
        for e in 0..j.len() {
            j[e] = beta * g.mag[e] * eps[e];
        }
        correlations(plan, j, wk, c);
        for p in 0..np {
            let x = c[p] * c[p];
            acc.s1[bi * np + p] += x;
            acc.s2[bi * np + p] += x * x;
        }
    }
    acc.count += 1.0;
}

/// Sign sampling over chunks [c0, c1) of `chunk` samples; chunk c uses stream (seed, c).
pub fn sample(g: &Graph, plan: &Plan, betas: &[f64], seed: u64, c0: u64, c1: u64, chunk: usize, threads: usize) -> Acc {
    let np = plan.pairs.len();
    let nb = betas.len();
    let next = std::sync::atomic::AtomicU64::new(c0);
    let results = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| {
                let mut wk = plan.work();
                let mut j = vec![0.0; g.edges.len()];
                let mut eps = vec![0.0; g.edges.len()];
                let mut c = vec![0.0; np];
                loop {
                    let ci = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if ci >= c1 {
                        break;
                    }
                    let mut rng = Rng::stream(seed, ci);
                    let mut acc = Acc::new(nb, np);
                    for _ in 0..chunk {
                        let mut bitsrc = 0u64;
                        for (e, x) in eps.iter_mut().enumerate() {
                            if e % 64 == 0 {
                                bitsrc = rng.next_u64();
                            }
                            *x = if (bitsrc >> (e % 64)) & 1 == 0 { 1.0 } else { -1.0 };
                        }
                        eval_config(plan, g, betas, &eps, &mut wk, &mut j, &mut c, &mut acc);
                    }
                    results.lock().unwrap().push((ci, acc));
                }
            });
        }
    });
    let mut r = results.into_inner().unwrap();
    r.sort_by_key(|x| x.0);
    let mut tot = Acc::new(nb, np);
    for (_, a) in &r {
        tot.add(a);
    }
    tot
}

/// Exact Q by enumerating all signs of the non-tree edges (tree edges gauge-fixed to +1).
pub fn exhaustive(g: &Graph, plan: &Plan, betas: &[f64], threads: usize) -> Acc {
    let np = plan.pairs.len();
    let nb = betas.len();
    let tree = g.spanning_forest();
    let free: Vec<usize> = (0..g.edges.len()).filter(|&e| !tree[e]).collect();
    let cyc = free.len();
    assert!(cyc <= 34, "cyclomatic number {cyc} too large for enumeration");
    let total: u64 = 1u64 << cyc;
    let chunk: u64 = 1024.min(total);
    let nchunks = total / chunk;
    let next = std::sync::atomic::AtomicU64::new(0);
    let results = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| {
                let mut wk = plan.work();
                let mut j = vec![0.0; g.edges.len()];
                let mut eps = vec![1.0; g.edges.len()];
                let mut c = vec![0.0; np];
                loop {
                    let ci = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if ci >= nchunks {
                        break;
                    }
                    let mut acc = Acc::new(nb, np);
                    for x in (ci * chunk)..((ci + 1) * chunk) {
                        for (i, &e) in free.iter().enumerate() {
                            eps[e] = if (x >> i) & 1 == 0 { 1.0 } else { -1.0 };
                        }
                        eval_config(plan, g, betas, &eps, &mut wk, &mut j, &mut c, &mut acc);
                    }
                    results.lock().unwrap().push((ci, acc));
                }
            });
        }
    });
    let mut r = results.into_inner().unwrap();
    r.sort_by_key(|x| x.0);
    let mut tot = Acc::new(nb, np);
    for (_, a) in &r {
        tot.add(a);
    }
    tot
}

/// Exact enumeration returning every correlation: out[beta][pair][config] over the 2^cyc
/// gauge-fixed sign configurations (each gauge class once, all equally likely).
pub fn exhaustive_c(g: &Graph, plan: &Plan, betas: &[f64], threads: usize) -> Vec<Vec<Vec<f64>>> {
    let np = plan.pairs.len();
    let nb = betas.len();
    let tree = g.spanning_forest();
    let free: Vec<usize> = (0..g.edges.len()).filter(|&e| !tree[e]).collect();
    let cyc = free.len();
    assert!(cyc <= 24, "cyclomatic number {cyc} too large to store all correlations");
    let total: u64 = 1u64 << cyc;
    let chunk: u64 = 256.min(total);
    let nchunks = total / chunk;
    let next = std::sync::atomic::AtomicU64::new(0);
    let results = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| {
                let mut wk = plan.work();
                let mut j = vec![0.0; g.edges.len()];
                let mut eps = vec![1.0; g.edges.len()];
                let mut c = vec![0.0; np];
                loop {
                    let ci = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if ci >= nchunks {
                        break;
                    }
                    let mut buf = vec![0.0; (chunk as usize) * nb * np];
                    for (xi, x) in ((ci * chunk)..((ci + 1) * chunk)).enumerate() {
                        for (i, &e) in free.iter().enumerate() {
                            eps[e] = if (x >> i) & 1 == 0 { 1.0 } else { -1.0 };
                        }
                        for (bi, &beta) in betas.iter().enumerate() {
                            for e in 0..j.len() {
                                j[e] = beta * g.mag[e] * eps[e];
                            }
                            correlations(plan, &j, &mut wk, &mut c);
                            for p in 0..np {
                                buf[(bi * np + p) * chunk as usize + xi] = c[p];
                            }
                        }
                    }
                    results.lock().unwrap().push((ci, buf));
                }
            });
        }
    });
    let mut r = results.into_inner().unwrap();
    r.sort_by_key(|x| x.0);
    let mut out = vec![vec![Vec::with_capacity(total as usize); np]; nb];
    for (_, buf) in &r {
        for b in 0..nb {
            for p in 0..np {
                out[b][p].extend_from_slice(&buf[(b * np + p) * chunk as usize..(b * np + p + 1) * chunk as usize]);
            }
        }
    }
    out
}
