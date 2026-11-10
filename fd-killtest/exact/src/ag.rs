//! Exact arboreal-gas connection probabilities P^AG(u <-> v) by a frontier transfer DP.
//!
//! The frontier is the sliding window of the last `win` inserted vertices. A state is the partition
//! of the window into blocks (connectivity through the forest chosen so far), encoded as 4-bit
//! canonical labels (first-occurrence order) packed into a u64. Labels 14 and 15 mark v's and u's
//! block. Each state carries one weight per beta; layers are rescaled by their max every step and
//! the log scale is accumulated.
//!
//! Two ways to get P(u <-> v):
//! * `Contract`: forests of G with u, v in different trees are exactly the forests of G/uv, so
//!   P = 1 - Z(G/uv) / Z(G). Only u's block is marked, and only between the insertions of u and v.
//!   Cheap, but loses relative accuracy when P is small.
//! * `TwoMark`: track u's and v's blocks until they merge; drop states where either dies first.
//!   Direct and accurate for small P, but the state space is larger.

use crate::lattice::Graph;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Mutex;

#[derive(Default)]
pub struct MixHasher(u64);

impl Hasher for MixHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    #[inline]
    fn write_u64(&mut self, x: u64) {
        let mut z = x ^ self.0;
        z = (z ^ (z >> 33)).wrapping_mul(0xff51_afd7_ed55_8ccd);
        z = (z ^ (z >> 33)).wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        self.0 = z ^ (z >> 33);
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

type Map = HashMap<u64, u32, BuildHasherDefault<MixHasher>>;

const UMARK: u8 = 15;
const VMARK: u8 = 14;
const NEWLAB: u8 = 16;
const CONN: u8 = 17;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    Base,
    Contract { u: usize, v: usize },
    TwoMark { u: usize, v: usize },
}

#[derive(Clone)]
pub struct Dp {
    pub mode: Mode,
    pub tag: usize,
    pub keys: Vec<u64>,
    pub vals: Vec<f64>,
    pub logs: Vec<f64>,
    pub max_states: usize,
}

pub struct StepInfo {
    pub t: usize,
    pub pos: Vec<usize>,
    /// prod[mask * nb + b] = product of y_e(beta_b) over back edges in mask.
    pub prod: Vec<f64>,
}

pub fn step_info(g: &Graph, t: usize, ys: &[Vec<f64>], nb: usize) -> StepInfo {
    let w = g.win;
    let mut pos = Vec::new();
    let mut ye = Vec::new();
    for &(e, a) in &g.back[t] {
        assert!(a + w >= t, "edge outside window");
        pos.push(a + w - t);
        ye.push(e);
    }
    let k = pos.len();
    let mut prod = vec![1.0; (1 << k) * nb];
    for mask in 0..(1usize << k) {
        for i in 0..k {
            if mask >> i & 1 == 1 {
                for b in 0..nb {
                    prod[mask * nb + b] *= ys[ye[i]][b];
                }
            }
        }
    }
    StepInfo { t, pos, prod }
}

#[inline]
fn has_marker(key: u64) -> bool {
    (key >> 1) & (key >> 2) & (key >> 3) & 0x1111_1111_1111_1111 != 0
}

#[inline]
fn merge(a: &mut [u8], x: u8, y: u8) {
    let xm = x == UMARK || x == VMARK;
    let ym = y == UMARK || y == VMARK;
    let r = if xm && ym {
        CONN
    } else if xm {
        x
    } else if ym {
        y
    } else {
        x
    };
    for e in a.iter_mut() {
        if *e == x || *e == y {
            *e = r;
        }
    }
}

impl Dp {
    pub fn initial(g: &Graph, nb: usize) -> Dp {
        let w = g.win;
        assert!(w <= 14, "window {w} > 14 not supported by the u64 encoding");
        let mut key = 0u64;
        for p in 0..w {
            key |= (p as u64) << (4 * p);
        }
        Dp { mode: Mode::Base, tag: usize::MAX, keys: vec![key], vals: vec![1.0; nb], logs: vec![0.0; nb], max_states: 1 }
    }

    fn params(&self, t: usize) -> (usize, usize, bool, bool) {
        let (mu, mv, two) = match self.mode {
            Mode::Base => (usize::MAX, usize::MAX, false),
            Mode::Contract { u, v } => (u, v, false),
            Mode::TwoMark { u, v } => (u, v, true),
        };
        // Marker nibbles in the encoded output: Contract keeps u's mark only between u and v.
        let out_marked = if two { t >= mu } else { t >= mu && t < mv };
        (mu, mv, two, out_marked)
    }

    /// Call f(child key, edge mask) for every valid transition of `key` at step info.t.
    #[inline]
    fn children(key: u64, w: usize, info: &StepInfo, prm: (usize, usize, bool, bool), mut f: impl FnMut(u64, usize)) {
        let (mu, mv, two, out_marked) = prm;
        let t = info.t;
        let k = info.pos.len();
        let mut arr = [0u8; 16];
        let mut a = [0u8; 16];
        for p in 0..w {
            arr[p] = ((key >> (4 * p)) & 0xF) as u8;
        }
        let newlab = if t == mu {
            UMARK
        } else if t == mv {
            if two {
                VMARK
            } else if arr[..w].contains(&UMARK) {
                UMARK
            } else {
                NEWLAB
            }
        } else {
            NEWLAB
        };
        arr[w] = newlab;
        'mask: for mask in 0..(1usize << k) {
            a[..=w].copy_from_slice(&arr[..=w]);
            for j in 0..k {
                if mask >> j & 1 == 1 {
                    let lv = a[w];
                    let lb = a[info.pos[j]];
                    if lv == lb {
                        continue 'mask;
                    }
                    merge(&mut a[..=w], lv, lb);
                }
            }
            let rest = &a[1..=w];
            if two && t >= mu {
                let h15 = rest.contains(&UMARK);
                let h14 = rest.contains(&VMARK);
                let ok = if t < mv { h15 } else { h14 == h15 };
                if !ok {
                    continue;
                }
            }
            let mut map_lab = [0xFFu8; 18];
            let mut next = 0u8;
            let mut nk = 0u64;
            for (p, &lab) in rest.iter().enumerate() {
                let nib = if out_marked && (lab == UMARK || lab == VMARK) {
                    lab
                } else {
                    let ml = &mut map_lab[lab as usize];
                    if *ml == 0xFF {
                        *ml = next;
                        next += 1;
                    }
                    *ml
                };
                nk |= (nib as u64) << (4 * p);
            }
            debug_assert!(next <= 14);
            f(nk, mask);
        }
    }

    fn rescale(&mut self, nkeys: Vec<u64>, mut nvals: Vec<f64>, nb: usize) {
        let ns = nkeys.len();
        for b in 0..nb {
            let mut mx = 0.0f64;
            for s in 0..ns {
                mx = mx.max(nvals[s * nb + b]);
            }
            if mx > 0.0 {
                let inv = 1.0 / mx;
                for s in 0..ns {
                    nvals[s * nb + b] *= inv;
                }
                self.logs[b] += mx.ln();
            }
        }
        self.keys = nkeys;
        self.vals = nvals;
        self.max_states = self.max_states.max(ns);
    }

    pub fn advance(&mut self, w: usize, info: &StepInfo, nb: usize) {
        let prm = self.params(info.t);
        let cap = self.keys.len() + self.keys.len() / 4 + 16;
        let mut map: Map = HashMap::with_capacity_and_hasher(cap, Default::default());
        let mut nkeys: Vec<u64> = Vec::with_capacity(cap);
        let mut nvals: Vec<f64> = Vec::with_capacity(cap * nb);
        for (i, &key) in self.keys.iter().enumerate() {
            let src = &self.vals[i * nb..(i + 1) * nb];
            Dp::children(key, w, info, prm, |nk, mask| {
                let idx = *map.entry(nk).or_insert_with(|| {
                    nkeys.push(nk);
                    nvals.extend(std::iter::repeat(0.0).take(nb));
                    (nkeys.len() - 1) as u32
                }) as usize;
                let pr = &info.prod[mask * nb..(mask + 1) * nb];
                let dst = &mut nvals[idx * nb..(idx + 1) * nb];
                for b in 0..nb {
                    dst[b] += src[b] * pr[b];
                }
            });
        }
        drop(map);
        self.rescale(nkeys, nvals, nb);
    }

    /// Parallel step: threads emit (child, parent << 4 | mask) into hash-sharded buckets, then each
    /// shard is aggregated by one thread. Per child key the summation order equals the serial one.
    pub fn advance_par(&mut self, w: usize, info: &StepInfo, nb: usize, threads: usize) {
        let ns = self.keys.len();
        let th = threads.max(1);
        let par_min: usize = std::env::var("FDK_PARMIN").ok().and_then(|v| v.parse().ok()).unwrap_or(20_000);
        if th == 1 || ns < par_min {
            return self.advance(w, info, nb);
        }
        assert!(ns < (1 << 28));
        let prm = self.params(info.t);
        let nsh = th;
        let chunk = ns.div_ceil(th);
        let keys = &self.keys;
        let shard_of = |nk: u64| -> usize {
            let mut h = MixHasher::default();
            h.write_u64(nk);
            ((h.finish() >> 32) as usize * nsh) >> 32
        };
        let buckets: Vec<Vec<Vec<(u64, u32)>>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..th)
                .map(|ti| {
                    s.spawn(move || {
                        let lo = (ti * chunk).min(ns);
                        let hi = ((ti + 1) * chunk).min(ns);
                        let mut b: Vec<Vec<(u64, u32)>> = (0..nsh).map(|_| Vec::with_capacity((hi - lo) * 3 / nsh + 16)).collect();
                        for i in lo..hi {
                            Dp::children(keys[i], w, info, prm, |nk, mask| {
                                b[shard_of(nk)].push((nk, ((i as u32) << 4) | mask as u32));
                            });
                        }
                        b
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let vals = &self.vals;
        let bref = &buckets;
        let parts: Vec<(Vec<u64>, Vec<f64>)> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..nsh)
                .map(|sh| {
                    s.spawn(move || {
                        let tot: usize = bref.iter().map(|b| b[sh].len()).sum();
                        let cap = tot / 2 + 16;
                        let mut map: Map = HashMap::with_capacity_and_hasher(cap, Default::default());
                        let mut nkeys: Vec<u64> = Vec::with_capacity(cap);
                        let mut nvals: Vec<f64> = Vec::with_capacity(cap * nb);
                        for b in bref.iter() {
                            for &(nk, sm) in &b[sh] {
                                let i = (sm >> 4) as usize;
                                let mask = (sm & 15) as usize;
                                let idx = *map.entry(nk).or_insert_with(|| {
                                    nkeys.push(nk);
                                    nvals.extend(std::iter::repeat(0.0).take(nb));
                                    (nkeys.len() - 1) as u32
                                }) as usize;
                                let src = &vals[i * nb..(i + 1) * nb];
                                let pr = &info.prod[mask * nb..(mask + 1) * nb];
                                let dst = &mut nvals[idx * nb..(idx + 1) * nb];
                                for bb in 0..nb {
                                    dst[bb] += src[bb] * pr[bb];
                                }
                            }
                        }
                        (nkeys, nvals)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        drop(buckets);
        let total: usize = parts.iter().map(|p| p.0.len()).sum();
        let mut nkeys = Vec::with_capacity(total);
        let mut nvals = Vec::with_capacity(total * nb);
        for (k, v) in parts {
            nkeys.extend_from_slice(&k);
            nvals.extend_from_slice(&v);
        }
        self.rescale(nkeys, nvals, nb);
    }

    /// log of the final weight sum (only marker-free states count for TwoMark).
    pub fn finish(&self, nb: usize) -> Vec<f64> {
        let two = matches!(self.mode, Mode::TwoMark { .. });
        let mut out = vec![0.0; nb];
        for b in 0..nb {
            let mut s = 0.0;
            for (i, &key) in self.keys.iter().enumerate() {
                if two && has_marker(key) {
                    continue;
                }
                s += self.vals[i * nb + b];
            }
            out[b] = if s > 0.0 { s.ln() + self.logs[b] } else { f64::NEG_INFINITY };
        }
        out
    }
}

#[derive(Clone, Debug)]
pub struct PairAg {
    /// 1 - Z(G/uv)/Z(G)
    pub p_sub: Vec<f64>,
    /// direct two-marker value, if requested
    pub p_two: Option<Vec<f64>>,
    pub states_sub: usize,
    pub states_two: usize,
}

#[derive(Clone, Debug)]
pub struct AgOut {
    pub log_z: Vec<f64>,
    pub states_base: usize,
    pub pairs: Vec<PairAg>,
    pub secs: f64,
}

pub fn y_of(g: &Graph, betas: &[f64]) -> Vec<Vec<f64>> {
    g.mag.iter().map(|&m| betas.iter().map(|&b| (b * m).sinh().powi(2)).collect()).collect()
}

/// Run the base DP and, for every pair, a Contract branch (and optionally a TwoMark branch) born from
/// the base layer at the insertion of min(u, v). All DPs advance in lockstep; live DPs are advanced
/// in parallel on `threads` threads.
pub fn ag_pairs(g: &Graph, betas: &[f64], pairs: &[(usize, usize)], two: &[bool], threads: usize) -> AgOut {
    ag_pairs_opt(g, betas, pairs, true, two, threads)
}

/// As `ag_pairs`, optionally without the Contract branches (then `p_sub` is empty).
pub fn ag_pairs_opt(g: &Graph, betas: &[f64], pairs: &[(usize, usize)], sub: bool, two: &[bool], threads: usize) -> AgOut {
    let t0 = std::time::Instant::now();
    let nb = betas.len();
    let ys = y_of(g, betas);
    let w = g.win;
    let mut births: Vec<Vec<(Mode, usize)>> = vec![Vec::new(); g.n];
    for (i, &(a, b)) in pairs.iter().enumerate() {
        let (u, v) = (a.min(b), a.max(b));
        if sub {
            births[u].push((Mode::Contract { u, v }, 2 * i));
        }
        if two[i] {
            births[u].push((Mode::TwoMark { u, v }, 2 * i + 1));
        }
    }
    let mut dps: Vec<Mutex<Dp>> = vec![Mutex::new(Dp::initial(g, nb))];
    for t in 0..g.n {
        if !births[t].is_empty() {
            let base = dps[0].lock().unwrap().clone();
            for &(mode, tag) in &births[t] {
                let mut d = base.clone();
                d.mode = mode;
                d.tag = tag;
                d.max_states = 0;
                dps.push(Mutex::new(d));
            }
        }
        let info = step_info(g, t, &ys, nb);
        let nt = threads.max(1).min(dps.len());
        if dps.len() < threads && threads > 1 {
            for d in &dps {
                d.lock().unwrap().advance_par(w, &info, nb, threads);
            }
        } else if nt <= 1 {
            for d in &dps {
                d.lock().unwrap().advance(w, &info, nb);
            }
        } else {
            let next = std::sync::atomic::AtomicUsize::new(0);
            // largest DPs first
            let mut order: Vec<usize> = (0..dps.len()).collect();
            order.sort_by_key(|&i| std::cmp::Reverse(dps[i].lock().unwrap().keys.len()));
            std::thread::scope(|s| {
                for _ in 0..nt {
                    s.spawn(|| loop {
                        let j = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if j >= order.len() {
                            break;
                        }
                        dps[order[j]].lock().unwrap().advance(w, &info, nb);
                    });
                }
            });
        }
    }
    let dps: Vec<Dp> = dps.into_iter().map(|m| m.into_inner().unwrap()).collect();
    let log_z = dps[0].finish(nb);
    let mut out: Vec<PairAg> =
        pairs.iter().map(|_| PairAg { p_sub: vec![], p_two: None, states_sub: 0, states_two: 0 }).collect();
    for d in &dps[1..] {
        let i = d.tag / 2;
        let lz = d.finish(nb);
        if d.tag % 2 == 0 {
            out[i].p_sub = (0..nb).map(|b| -(lz[b] - log_z[b]).exp_m1()).collect();
            out[i].states_sub = d.max_states;
        } else {
            out[i].p_two = Some((0..nb).map(|b| (lz[b] - log_z[b]).exp()).collect());
            out[i].states_two = d.max_states;
        }
    }
    AgOut { log_z, states_base: dps[0].max_states, pairs: out, secs: t0.elapsed().as_secs_f64() }
}

/// Survival of u's tree across the strip: for each u, the per-step log of (prefix weight with u's
/// block still on the frontier) / (total prefix weight), per beta. Its slope per column is the
/// arboreal-gas decay rate m_AG (ratio of the two leading transfer-matrix eigenvalues).
pub fn ag_survival(g: &Graph, betas: &[f64], us: &[usize], threads: usize) -> Vec<Vec<Vec<f64>>> {
    let nb = betas.len();
    let ys = y_of(g, betas);
    let w = g.win;
    let mut base = Dp::initial(g, nb);
    let mut marked: Vec<Option<Dp>> = us.iter().map(|_| None).collect();
    let mut out: Vec<Vec<Vec<f64>>> = us.iter().map(|_| vec![vec![f64::NAN; nb]; g.n]).collect();
    for t in 0..g.n {
        for (k, &u) in us.iter().enumerate() {
            if u == t {
                let mut d = base.clone();
                d.mode = Mode::Contract { u, v: usize::MAX };
                d.max_states = 0;
                marked[k] = Some(d);
            }
        }
        let info = step_info(g, t, &ys, nb);
        base.advance_par(w, &info, nb, threads);
        for d in marked.iter_mut().flatten() {
            d.advance_par(w, &info, nb, threads);
        }
        let tot: Vec<f64> = (0..nb)
            .map(|b| base.logs[b] + (0..base.keys.len()).map(|i| base.vals[i * nb + b]).sum::<f64>().ln())
            .collect();
        for (k, d) in marked.iter().enumerate() {
            let Some(d) = d else { continue };
            for b in 0..nb {
                let m: f64 = d.keys.iter().enumerate().filter(|(_, &key)| has_marker(key)).map(|(i, _)| d.vals[i * nb + b]).sum();
                out[k][t][b] = if m > 0.0 { d.logs[b] + m.ln() - tot[b] } else { f64::NEG_INFINITY };
            }
        }
    }
    out
}
