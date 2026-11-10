//! Targeted measurements: decay rates of P^AG(r) and Q(r) along strips, and distributional tests
//! of the pendant/series step on two-terminal blocks.

use crate::ag;
use crate::glass::{self, Acc, Plan};
use crate::lattice::{Graph, Kind, Strip};
use std::io::Write;

fn lsq_slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx) * (a - mx)).sum();
    sxy / sxx
}

/// Slope of -ln f over the window and a systematic error from the two halves of the window.
fn rate(r: &[f64], f: &[f64]) -> (f64, f64) {
    let y: Vec<f64> = f.iter().map(|v| -v.ln()).collect();
    let m = lsq_slope(r, &y);
    if r.len() < 6 {
        return (m, f64::NAN);
    }
    let h = r.len() / 2;
    let m1 = lsq_slope(&r[..h + 1], &y[..h + 1]);
    let m2 = lsq_slope(&r[h..], &y[h..]);
    (m, (m1 - m2).abs() / 2.0)
}

pub struct DecayCfg {
    pub kind: Kind,
    pub w: usize,
    pub l: usize,
    pub periodic: bool,
    pub betas: Vec<f64>,
    pub ag_two: bool,
    pub ag_surv: bool,
    pub sparse: bool,
    pub nsamp: u64,
    pub batches: u64,
    pub chunk: u64,
    pub seed: u64,
    pub threads: usize,
    pub out: String,
    pub fit_out: String,
}

const SPARSE: [usize; 24] = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 127];

pub fn decay(cfg: &DecayCfg) {
    let t0 = std::time::Instant::now();
    let strip = Strip { kind: cfg.kind, w: cfg.w, l: cfg.l, periodic: cfg.periodic };
    let g = strip.graph();
    let (w, l) = (cfg.w, cfg.l);
    let mut rows: Vec<(&str, usize)> = vec![("bdry", 0)];
    if (w - 1) / 2 != 0 && !cfg.periodic {
        rows.push(("centre", (w - 1) / 2));
    }
    let rs: Vec<usize> = if cfg.sparse { SPARSE.iter().cloned().filter(|&r| r < l).collect() } else { (1..l).collect() };
    let mut pairs = Vec::new();
    let mut meta = Vec::new();
    for (ri, &(_, y)) in rows.iter().enumerate() {
        for &r in &rs {
            pairs.push((strip.vid(0, y), strip.vid(r, y)));
            meta.push((ri, r));
        }
    }
    let nb = cfg.betas.len();
    // exact AG(r) by two-marker DPs
    let agv: Option<Vec<Vec<f64>>> = if cfg.ag_two {
        let two = vec![true; pairs.len()];
        let o = ag::ag_pairs_opt(&g, &cfg.betas, &pairs, false, &two, cfg.threads);
        eprintln!("  AG two-mark {:.1}s states base={}", o.secs, o.states_base);
        Some(o.pairs.iter().map(|p| p.p_two.clone().unwrap()).collect())
    } else {
        None
    };
    // survival rates
    let surv: Option<Vec<Vec<Vec<f64>>>> = if cfg.ag_surv {
        let us: Vec<usize> = rows.iter().map(|&(_, y)| strip.vid(0, y)).collect();
        let t1 = std::time::Instant::now();
        let s = ag::ag_survival(&g, &cfg.betas, &us, cfg.threads);
        eprintln!("  AG survival {:.1}s", t1.elapsed().as_secs_f64());
        Some(s)
    } else {
        None
    };
    // sampled Q(r) in independent batches
    let plan = Plan::new(&g, &pairs);
    let per = cfg.nsamp.div_ceil(cfg.batches * cfg.chunk);
    let t2 = std::time::Instant::now();
    let accs: Vec<Acc> = (0..cfg.batches)
        .map(|bt| glass::sample(&g, &plan, &cfg.betas, cfg.seed, bt * per, (bt + 1) * per, cfg.chunk as usize, cfg.threads))
        .collect();
    let mut pooled = Acc::new(nb, pairs.len());
    for a in &accs {
        pooled.add(a);
    }
    eprintln!("  glass {:.1}s n={}", t2.elapsed().as_secs_f64(), pooled.count);
    let new = !std::path::Path::new(&cfg.out).exists();
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&cfg.out).unwrap();
    if new {
        writeln!(f, "lattice,w,L,bc,row,beta,r,u,v,AG,Q,SE,nsamp").unwrap();
    }
    for b in 0..nb {
        for (p, &(ri, r)) in meta.iter().enumerate() {
            let a = agv.as_ref().map(|v| format!("{:.12e}", v[p][b])).unwrap_or_default();
            writeln!(
                f,
                "{},{},{},{},{},{},{},{},{},{},{:.12e},{:.6e},{}",
                cfg.kind.name(),
                w,
                l,
                strip.bc(),
                rows[ri].0,
                cfg.betas[b],
                r,
                pairs[p].0,
                pairs[p].1,
                a,
                pooled.mean(b, p),
                pooled.se(b, p),
                pooled.count
            )
            .unwrap();
        }
    }
    // fits
    let new = !std::path::Path::new(&cfg.fit_out).exists();
    let mut ff = std::fs::OpenOptions::new().create(true).append(true).open(&cfg.fit_out).unwrap();
    if new {
        writeln!(ff, "lattice,w,L,bc,row,beta,r_lo,r_hi,npts,m_AG,err_AG,m_AG_surv,err_AG_surv,m_Q,err_Q,gap,err_gap,r_at_max_ratio,max_ratio,z_at_max,nsamp").unwrap();
    }
    for (ri, &(rname, _)) in rows.iter().enumerate() {
        for b in 0..nb {
            let idx: Vec<usize> = (0..pairs.len()).filter(|&p| meta[p].0 == ri).collect();
            let ok = |p: usize| -> bool {
                let q = pooled.mean(b, p);
                let se = pooled.se(b, p);
                let a_ok = agv.as_ref().map(|v| v[p][b] >= 1e-12).unwrap_or(true);
                q >= 1e-12 && se <= 0.3 * q && a_ok
            };
            // window: the leading run of valid r, fitted over its upper two thirds
            let mut run = Vec::new();
            for &p in &idx {
                if !ok(p) {
                    break;
                }
                run.push(p);
            }
            let r_hi = run.last().map(|&p| meta[p].1).unwrap_or(0);
            let r_lo = ((r_hi as f64) / 3.0).ceil().max(2.0) as usize;
            let win: Vec<usize> = run.iter().cloned().filter(|&p| meta[p].1 >= r_lo).collect();
            let rr: Vec<f64> = win.iter().map(|&p| meta[p].1 as f64).collect();
            let (m_ag, e_ag) = match &agv {
                Some(v) if win.len() >= 3 => rate(&rr, &win.iter().map(|&p| v[p][b]).collect::<Vec<_>>()),
                _ => (f64::NAN, f64::NAN),
            };
            let (m_q, mut e_q) = if win.len() >= 3 {
                rate(&rr, &win.iter().map(|&p| pooled.mean(b, p)).collect::<Vec<_>>())
            } else {
                (f64::NAN, f64::NAN)
            };
            // batch (jackknife-free) statistical error of the Q slope
            if win.len() >= 3 {
                let ms: Vec<f64> = accs
                    .iter()
                    .filter_map(|a| {
                        let v: Vec<f64> = win.iter().map(|&p| a.mean(b, p)).collect();
                        if v.iter().all(|&x| x > 0.0) {
                            Some(lsq_slope(&rr, &v.iter().map(|x| -x.ln()).collect::<Vec<_>>()))
                        } else {
                            None
                        }
                    })
                    .collect();
                if ms.len() >= 3 {
                    let mean = ms.iter().sum::<f64>() / ms.len() as f64;
                    let var = ms.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (ms.len() as f64 - 1.0);
                    let stat = (var / ms.len() as f64).sqrt();
                    e_q = (stat * stat + if e_q.is_finite() { e_q * e_q } else { 0.0 }).sqrt();
                }
            }
            // survival rate per column over the last two thirds of the strip
            let (m_s, e_s) = match &surv {
                Some(s) => {
                    let xs: Vec<usize> = ((l / 3).max(1)..l).collect();
                    let vals: Vec<f64> = xs.iter().map(|&x| s[ri][x * w + w - 1][b]).collect();
                    if vals.iter().all(|v| v.is_finite()) && xs.len() >= 3 {
                        let xr: Vec<f64> = xs.iter().map(|&x| x as f64).collect();
                        rate(&xr, &vals.iter().map(|v| v.exp()).collect::<Vec<_>>())
                    } else {
                        (f64::NAN, f64::NAN)
                    }
                }
                None => (f64::NAN, f64::NAN),
            };
            let m_ag_use = if m_ag.is_finite() { m_ag } else { m_s };
            let e_ag_use = if m_ag.is_finite() { e_ag } else { e_s };
            let gap = m_q - m_ag_use;
            let e_gap = (e_q.powi(2) + e_ag_use.powi(2)).sqrt();
            // where Q/AG peaks (needs exact AG)
            let (mut rmax, mut best, mut zmax) = (0usize, f64::NAN, f64::NAN);
            if let Some(v) = &agv {
                for &p in &idx {
                    let q = pooled.mean(b, p);
                    let se = pooled.se(b, p);
                    if v[p][b] < 1e-12 || q < 1e-12 || se > 0.3 * q {
                        continue;
                    }
                    let ratio = q / v[p][b];
                    if !(best >= ratio) {
                        best = ratio;
                        rmax = meta[p].1;
                        zmax = (q - v[p][b]) / se;
                    }
                }
            }
            writeln!(
                ff,
                "{},{},{},{},{},{},{},{},{},{:.6},{:.2e},{:.6},{:.2e},{:.6},{:.2e},{:.6},{:.2e},{},{:.6},{:.3},{}",
                cfg.kind.name(),
                w,
                l,
                strip.bc(),
                rname,
                cfg.betas[b],
                r_lo,
                r_hi,
                win.len(),
                m_ag,
                e_ag,
                m_s,
                e_s,
                m_q,
                e_q,
                gap,
                e_gap,
                rmax,
                best,
                zmax,
                pooled.count
            )
            .unwrap();
        }
    }
    eprintln!("  decay {} {}x{} done in {:.1}s", cfg.kind.name(), w, l, t0.elapsed().as_secs_f64());
}

// ------------------------------------------------------------------ series / pendant step

/// phi(g) of the pendant identity: 1 - Q_G(p,b) = (1-a^2) E phi(c^2).
fn phi(g: f64, a: f64, s: f64) -> f64 {
    let (a2, s2) = (a * a, s * s);
    (1.0 - s2 * g) * (1.0 + a2 * s2 * g) / (1.0 - a2 * s2 * g).powi(2)
}

/// 1 - P^AG_G(p<->b) = (1-a^2) * rhs(P_R).
fn rhs(p: f64, a: f64, s: f64) -> f64 {
    let a2 = a * a;
    let w = s * s / (1.0 - s * s);
    1.0 / (1.0 + (1.0 - a2) * w * p / (1.0 + w * (1.0 - p)))
}

/// Lower convex envelope of phi(., a, s) on [0, 1] evaluated at q: the minimum of E phi(X) over
/// all laws of X in [0, 1] with mean q (attained by a two-point law).
fn envelope(q: f64, a: f64, s: f64) -> f64 {
    let n = 4000;
    let xs: Vec<f64> = (0..=n).map(|i| i as f64 / n as f64).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| phi(x, a, s)).collect();
    let mut hull: Vec<usize> = Vec::new();
    for i in 0..=n {
        while hull.len() >= 2 {
            let (i1, i2) = (hull[hull.len() - 2], hull[hull.len() - 1]);
            let cross = (xs[i2] - xs[i1]) * (ys[i] - ys[i1]) - (ys[i2] - ys[i1]) * (xs[i] - xs[i1]);
            if cross <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(i);
    }
    for k in 1..hull.len() {
        let (i1, i2) = (hull[k - 1], hull[k]);
        if q <= xs[i2] {
            let t = (q - xs[i1]) / (xs[i2] - xs[i1]);
            return ys[i1] + t * (ys[i2] - ys[i1]);
        }
    }
    ys[n]
}

const GRID: [f64; 12] = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99];

pub struct Law {
    pub label: String,
    pub beta: f64,
    pub x: Vec<f64>, // c^2 per equally likely configuration
    pub c: Vec<f64>, // signed c (empty for composites built from c^2 only)
    pub p: f64,      // P^AG between the terminals
}

/// Write per-(a, s) tests and return a summary line.
fn analyse(law: &Law, tests: &mut std::fs::File) -> String {
    let n = law.x.len() as f64;
    let q = law.x.iter().sum::<f64>() / n;
    let p = law.p;
    let mut srt = law.x.clone();
    srt.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let qu = |f: f64| srt[((srt.len() - 1) as f64 * f).round() as usize];
    let frac0 = law.x.iter().filter(|&&v| v < 1e-6).count() as f64 / n;
    let mut hist = [0usize; 20];
    for &v in &law.x {
        hist[((v * 20.0) as usize).min(19)] += 1;
    }
    // icx comparison with Bernoulli(P): E (c^2 - theta)_+ <= P (1 - theta)
    let mut icx = f64::NEG_INFINITY;
    let mut icx_at = 0.0;
    for k in 0..=200 {
        let th = k as f64 / 200.0;
        let lhs = law.x.iter().map(|&v| (v - th).max(0.0)).sum::<f64>() / n;
        let d = lhs - p * (1.0 - th);
        if d > icx {
            icx = d;
            icx_at = th;
        }
    }
    let (mut min_pend, mut arg_pend) = (f64::INFINITY, (0.0, 0.0));
    let (mut min_fd, mut arg_fd) = (f64::INFINITY, (0.0, 0.0));
    let (mut n_jen, mut n_fd, mut n_pend) = (0, 0, 0);
    let mut min_jen = f64::INFINITY;
    let (mut min_s5, mut arg_s5, mut n_s5) = (f64::INFINITY, (0.0, 0.0), 0usize);
    let mut min_tri = f64::INFINITY;
    // S3 sufficiency: (i) the Bernoulli(P) law itself satisfies S3; (ii) two-point laws with the
    // observed mean q that satisfy S3 (checked at theta = 0 and theta = g1).
    let (mut min_bern, mut n_bern) = (f64::INFINITY, 0usize);
    let (mut min_s3two, mut n_s3two, mut arg_s3two) = (f64::INFINITY, 0usize, (0.0, 0.0, 0.0, 0.0));
    let gpts: Vec<f64> = (0..=100).map(|k| k as f64 / 100.0).collect();
    for &a in &GRID {
        for &s in &GRID {
            let r = rhs(p, a, s);
            let mb = ((1.0 - p) + p * phi(1.0, a, s) - r) / r;
            min_bern = min_bern.min(mb);
            n_bern += (mb < 0.0) as usize;
            let mut worst = f64::INFINITY;
            let mut warg = (0.0, 0.0);
            if q <= p {
                for &g1 in gpts.iter().filter(|&&g| g <= q) {
                    for &g2 in gpts.iter().filter(|&&g| g > q) {
                        let p2 = (q - g1) / (g2 - g1);
                        if p2 * (g2 - g1) > p * (1.0 - g1) + 1e-15 {
                            continue;
                        }
                        let e = (1.0 - p2) * phi(g1, a, s) + p2 * phi(g2, a, s);
                        if e < worst {
                            worst = e;
                            warg = (g1, g2);
                        }
                    }
                }
            }
            let m = (worst - r) / r;
            if m < min_s3two {
                min_s3two = m;
                arg_s3two = (a, s, warg.0, warg.1);
            }
            n_s3two += (m < 0.0) as usize;
        }
    }
    for &a in &GRID {
        for &s in &GRID {
            let ephi = law.x.iter().map(|&v| phi(v, a, s)).sum::<f64>() / n;
            let r = rhs(p, a, s);
            let pq = phi(q, a, s);
            let env = envelope(q, a, s);
            let m_pend = (ephi - r) / r;
            let m_jen = (ephi - pq) / r;
            let m_fd = (env - r) / r;
            if m_pend < min_pend {
                min_pend = m_pend;
                arg_pend = (a, s);
            }
            if m_fd < min_fd {
                min_fd = m_fd;
                arg_fd = (a, s);
            }
            min_jen = min_jen.min(m_jen);
            // S5: E phi(c^2) >= phi(P^AG); triangle: phi(P) >= rhs(P)
            let pp = phi(p, a, s);
            let m_s5 = (ephi - pp) / r;
            if m_s5 < min_s5 {
                min_s5 = m_s5;
                arg_s5 = (a, s);
            }
            n_s5 += (m_s5 < 0.0) as usize;
            min_tri = min_tri.min((pp - r) / r);
            n_jen += (m_jen < 0.0) as usize;
            n_fd += (m_fd < 0.0) as usize;
            n_pend += (m_pend < 0.0) as usize;
            writeln!(
                tests,
                "{},{},{:.12e},{:.12e},{},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.4e},{:.4e},{:.4e},{:.12e},{:.4e}",
                law.label, law.beta, p, q, a, s, ephi, r, pq, env, m_pend, m_jen, m_fd, pp, m_s5
            )
            .unwrap();
        }
    }
    format!(
        "{},{},{},{:.12e},{:.12e},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},{:.4e},{},{:.4e},{:.2},{:.4e},{},{:.3},{:.3},{},{:.4e},{},{:.3},{:.3},{},{:.4e},{},{:.3},{:.3},{:.4e},{:.4e},{},{:.4e},{},{:.3},{:.3},{:.3},{:.3}",
        law.label,
        law.beta,
        law.x.len(),
        p,
        q,
        p - q,
        frac0,
        qu(0.05),
        qu(0.25),
        qu(0.5),
        qu(0.75),
        qu(0.95),
        qu(1.0),
        hist.iter().map(|h| format!("{:.4}", *h as f64 / n)).collect::<Vec<_>>().join(";"),
        icx,
        icx_at,
        min_pend,
        n_pend,
        arg_pend.0,
        arg_pend.1,
        n_jen,
        min_fd,
        n_fd,
        arg_fd.0,
        arg_fd.1,
        if min_jen < 0.0 { "jensen-fails" } else { "jensen-ok" },
        min_s5,
        n_s5,
        arg_s5.0,
        arg_s5.1,
        min_tri,
        min_bern,
        n_bern,
        min_s3two,
        n_s3two,
        arg_s3two.0,
        arg_s3two.1,
        arg_s3two.2,
        arg_s3two.3
    )
}

pub struct SeriesCfg {
    pub betas: Vec<f64>,
    pub threads: usize,
    pub max_cyc: usize,
    pub out_prefix: String,
}

fn block_laws(name: &str, g: &Graph, u: usize, v: usize, betas: &[f64], threads: usize) -> Vec<Law> {
    let plan = Plan::new(g, &[(u, v)]);
    let cs = glass::exhaustive_c(g, &plan, betas, threads);
    let o = ag::ag_pairs_opt(g, betas, &[(u.min(v), u.max(v))], false, &[true], threads);
    let pt = o.pairs[0].p_two.clone().unwrap();
    betas
        .iter()
        .enumerate()
        .map(|(b, &beta)| Law { label: name.to_string(), beta, x: cs[b][0].iter().map(|c| c * c).collect(), c: cs[b][0].clone(), p: pt[b] })
        .collect()
}

pub fn series(cfg: &SeriesCfg) {
    let mut tests = std::fs::File::create(format!("{}_tests.csv", cfg.out_prefix)).unwrap();
    writeln!(tests, "block,beta,P,Q,a,s,Ephi,rhs,phiQ,envQ,m_pend,m_jensen,m_fdonly,phiP,m_s5").unwrap();
    let mut blocks = std::fs::File::create(format!("{}_blocks.csv", cfg.out_prefix)).unwrap();
    let head = "block,beta,nconf,P,Q,P_minus_Q,frac_c2_lt_1e-6,q05,q25,q50,q75,q95,qmax,hist20,icx_max,icx_theta,min_pend,n_pend_neg,pend_a,pend_s,n_jensen_neg,min_fdonly,n_fdonly_neg,fdonly_a,fdonly_s,jensen,min_s5,n_s5_neg,s5_a,s5_s,min_triangle,min_bern,n_bern_neg,min_s3two,n_s3two_neg,s3two_a,s3two_s,s3two_g1,s3two_g2";
    writeln!(blocks, "{head}").unwrap();
    // two-terminal blocks cut from strips
    let mut small: Vec<(String, Vec<Law>)> = Vec::new();
    let shapes = [(2usize, 2usize), (2, 3), (2, 4), (2, 6), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4), (3, 6)];
    let mut list: Vec<(String, Graph, usize, usize)> = Vec::new();
    for kind in [Kind::Square, Kind::Tri, Kind::Honey] {
        for &(w, l) in &shapes {
            let st = Strip { kind, w, l, periodic: false };
            let g = st.graph();
            for (tname, u, v) in [("opp", st.vid(0, 0), st.vid(l - 1, w - 1)), ("row", st.vid(0, 0), st.vid(l - 1, 0))] {
                list.push((format!("{}-{}x{}-{}", kind.name(), w, l, tname), g.clone(), u, v));
            }
        }
    }
    // inhomogeneous magnitudes on 3x3 blocks
    for kind in [Kind::Square, Kind::Tri] {
        for seed in 1..=3u64 {
            let st = Strip { kind, w: 3, l: 3, periodic: false };
            let g = st.diluted(1.0, 0.2, 2.0, 100 + seed);
            list.push((format!("{}-3x3-opp-mag{}", kind.name(), seed), g, st.vid(0, 0), st.vid(2, 2)));
        }
    }
    for (name, g, u, v) in &list {
        if g.cyclomatic() > cfg.max_cyc {
            continue;
        }
        let laws = block_laws(name, g, *u, *v, &cfg.betas, cfg.threads);
        for law in &laws {
            let line = analyse(law, &mut tests);
            writeln!(blocks, "{line}").unwrap();
        }
        if g.cyclomatic() <= 9 {
            small.push((name.clone(), laws));
        }
        eprintln!("  block {name} cyc={} done", g.cyclomatic());
    }
    // series compositions of two blocks at a cut vertex: c = c1 c2 with independent signs
    let picks: Vec<usize> = (0..small.len()).filter(|&i| small[i].1[0].x.len() >= 2).collect();
    let mut ncomp = 0;
    for (ii, &i) in picks.iter().enumerate() {
        for &j in picks.iter().skip(ii) {
            if ncomp >= 60 {
                break;
            }
            if small[i].1[0].x.len() * small[j].1[0].x.len() > 1 << 18 {
                continue;
            }
            if (i * 7 + j * 3) % 3 != 0 && i != j {
                continue;
            }
            ncomp += 1;
            for b in 0..cfg.betas.len() {
                let (l1, l2) = (&small[i].1[b], &small[j].1[b]);
                let mut x = Vec::with_capacity(l1.x.len() * l2.x.len());
                for &a in &l1.x {
                    for &c in &l2.x {
                        x.push(a * c);
                    }
                }
                let law = Law { label: format!("{}+{}", small[i].0, small[j].0), beta: cfg.betas[b], x, c: vec![], p: l1.p * l2.p };
                let line = analyse(&law, &mut tests);
                writeln!(blocks, "{line}").unwrap();
            }
        }
    }
    eprintln!("  series compositions: {ncomp}");
    // parallel compositions (both terminals identified): K = K1 + K2, AG odds add
    let mut npar = 0;
    for (ii, &i) in picks.iter().enumerate() {
        for &j in picks.iter().skip(ii) {
            if npar >= 60 {
                break;
            }
            if small[i].1[0].c.len() * small[j].1[0].c.len() > 1 << 18 {
                continue;
            }
            if (i * 5 + j * 11) % 3 != 0 && i != j {
                continue;
            }
            npar += 1;
            for b in 0..cfg.betas.len() {
                let (l1, l2) = (&small[i].1[b], &small[j].1[b]);
                // The enumeration is gauge-fixed, so the stored sign of c is arbitrary; the fair law
                // of c is symmetric (flip a u-v cut), so compose c1 with both signs of c2.
                let mut x = Vec::with_capacity(2 * l1.c.len() * l2.c.len());
                for &c1 in &l1.c {
                    for &c2 in &l2.c {
                        // tanh(K1 +- K2) = (c1 +- c2) / (1 +- c1 c2)
                        let cp = (c1 + c2) / (1.0 + c1 * c2);
                        let cm = (c1 - c2) / (1.0 - c1 * c2);
                        x.push(cp * cp);
                        x.push(cm * cm);
                    }
                }
                let (o1, o2) = (l1.p / (1.0 - l1.p), l2.p / (1.0 - l2.p));
                let p = (o1 + o2) / (1.0 + o1 + o2);
                let law = Law { label: format!("{}|{}", small[i].0, small[j].0), beta: cfg.betas[b], x, c: vec![], p };
                let line = analyse(&law, &mut tests);
                writeln!(blocks, "{line}").unwrap();
            }
        }
    }
    eprintln!("  parallel compositions: {npar}");
}

/// Check the pendant identities against brute force on tiny blocks: returns the max abs errors of
/// 1 - Q_G = (1-a^2) E phi(c^2) and 1 - P^AG_G = (1-a^2) rhs(P_R).
pub fn pendant_selfcheck() -> (f64, f64) {
    let (mut eq, mut ep) = (0.0f64, 0.0f64);
    for (kind, w, l) in [(Kind::Square, 2usize, 2usize), (Kind::Tri, 2, 3), (Kind::Square, 2, 3)] {
        let st = Strip { kind, w, l, periodic: false };
        let gb = st.graph();
        let (x, bt) = (st.vid(0, 0), st.vid(l - 1, w - 1));
        for &beta in &[0.4, 1.1] {
            for &(a, s) in &[(0.3, 0.6), (0.8, 0.2), (0.5, 0.95)] {
                let n = gb.n;
                let p = n;
                let mut raw: Vec<(usize, usize, f64)> = gb.edges.iter().zip(&gb.mag).map(|(&(u, v), &m)| (u, v, beta * m)).collect();
                raw.push((p, bt, f64::atanh(a)));
                raw.push((p, x, f64::atanh(s)));
                let g = Graph::new(n + 1, &raw);
                let qg = crate::brute::q_all(&g, &[(p, bt)], 1.0)[0];
                let pg = crate::brute::ag(&g, p, bt, 1.0);
                let rawr: Vec<(usize, usize, f64)> = gb.edges.iter().zip(&gb.mag).map(|(&(u, v), &m)| (u, v, beta * m)).collect();
                let gr = Graph::new(n, &rawr);
                let pr = crate::brute::ag(&gr, x, bt, 1.0);
                let m = gr.edges.len();
                let mut ephi = 0.0;
                for sg in 0u64..(1u64 << m) {
                    let (mut z, mut num) = (0.0, 0.0);
                    for sp in 0u64..(1u64 << n) {
                        let mut h = 0.0;
                        for (e, &(u, v)) in gr.edges.iter().enumerate() {
                            let k = if sg >> e & 1 == 0 { gr.mag[e] } else { -gr.mag[e] };
                            h += if ((sp >> u) ^ (sp >> v)) & 1 == 0 { k } else { -k };
                        }
                        let wgt = h.exp();
                        z += wgt;
                        num += if ((sp >> x) ^ (sp >> bt)) & 1 == 0 { wgt } else { -wgt };
                    }
                    let c = num / z;
                    ephi += phi(c * c, a, s);
                }
                ephi /= (1u64 << m) as f64;
                eq = eq.max(((1.0 - qg) - (1.0 - a * a) * ephi).abs());
                ep = ep.max(((1.0 - pg) - (1.0 - a * a) * rhs(pr, a, s)).abs());
            }
        }
    }
    (eq, ep)
}
