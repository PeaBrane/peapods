//! fd-mc: kill test of forest domination, Q_G(u,v) = E_eps <s_u s_v>^2 <= P^AG(u <-> v).
//!
//! Subcommands (all take `--key value` options):
//!   selftest-planar | selftest-ag | selftest-glass   validation against brute force
//!   q-planar   exact planar correlations per sign sample (Kasteleyn), Q with SE
//!   q-glass    two-replica PT Monte Carlo, Q with SE (3D, or any graph)
//!   ag         arboreal-gas Monte Carlo, P^AG with binned SE and tau_int
//!   combine    join Q and AG rows into the final CSV with ratio and z

mod ag;
mod brute;
mod glass;
mod graph;
mod lct;
mod planar;
mod stats;

use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::time::Instant;

use graph::{Graph, Instance};

struct Args(HashMap<String, String>);

impl Args {
    fn parse(v: &[String]) -> Args {
        let mut m = HashMap::new();
        let mut i = 0;
        while i < v.len() {
            let k = v[i].trim_start_matches("--").to_string();
            let val = v.get(i + 1).cloned().unwrap_or_default();
            m.insert(k, val);
            i += 2;
        }
        Args(m)
    }
    fn s(&self, k: &str, d: &str) -> String {
        self.0.get(k).cloned().unwrap_or_else(|| d.to_string())
    }
    fn f(&self, k: &str, d: f64) -> f64 {
        self.0.get(k).map(|x| x.parse().unwrap()).unwrap_or(d)
    }
    fn u(&self, k: &str, d: usize) -> usize {
        self.0.get(k).map(|x| x.parse().unwrap()).unwrap_or(d)
    }
    fn fl(&self, k: &str, d: &str) -> Vec<f64> {
        self.s(k, d).split(',').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect()
    }
}

fn mix(a: u64, b: u64, c: u64) -> u64 {
    let mut z = a ^ b.wrapping_mul(0x9E3779B97F4A7C15) ^ c.wrapping_mul(0xC2B2AE3D27D4EB4F);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Instance from CLI: --lattice square|triangular|honeycomb|diluted|cubic|grid|box
fn instance(a: &Args) -> Instance {
    let lat = a.s("lattice", "square");
    match lat.as_str() {
        "grid" => {
            let (w, h) = (a.u("W", 6), a.u("H", 6));
            let pairs = parse_pairs(&a.s("pairs", "0-1"));
            graph::grid_with_pairs(w, h, &pairs)
        }
        "box" => {
            let d: Vec<usize> = a.s("dims", "3,3,3").split(',').map(|x| x.parse().unwrap()).collect();
            let mut inst = graph::cubic(d[0], d[1], d[2]);
            if a.0.contains_key("pairs") {
                inst.classes = parse_pairs(&a.s("pairs", ""))
                    .into_iter()
                    .map(|(u, v)| graph::PairClass { name: format!("{u}-{v}"), images: vec![(u, v)] })
                    .collect();
            }
            inst
        }
        _ if a.0.contains_key("decay") => {
            let rs: Vec<usize> = a.s("decay", "").split(',').map(|x| x.parse().unwrap()).collect();
            graph::decay(&lat, a.u("L", 16), &rs)
        }
        _ => graph::by_name(&lat, a.u("L", 16), a.f("occ", 0.75), a.u("real", 0) as u64),
    }
}

/// Effective conductance between u and v with unit edge conductances (Jacobi-preconditioned CG on
/// the Laplacian grounded at v).
fn ceff_unit(g: &Graph, u: u32, v: u32) -> f64 {
    let n = g.n;
    let (u, v) = (u as usize, v as usize);
    let deg: Vec<f64> = g.adj.iter().map(|x| x.len() as f64).collect();
    let matvec = |x: &[f64], out: &mut [f64]| {
        for i in 0..n {
            if i == v {
                out[i] = 0.0;
                continue;
            }
            let mut s = deg[i] * x[i];
            for &(w, _) in &g.adj[i] {
                if w as usize != v {
                    s -= x[w as usize];
                }
            }
            out[i] = s;
        }
    };
    let mut x = vec![0.0; n];
    let mut r = vec![0.0; n];
    r[u] = 1.0;
    let mut z: Vec<f64> = (0..n).map(|i| if i == v { 0.0 } else { r[i] / deg[i] }).collect();
    let mut p = z.clone();
    let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
    let mut ap = vec![0.0; n];
    for _ in 0..20 * n + 100 {
        matvec(&p, &mut ap);
        let alpha = rz / p.iter().zip(&ap).map(|(a, b)| a * b).sum::<f64>();
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rn = r.iter().map(|a| a * a).sum::<f64>().sqrt();
        if rn < 1e-13 {
            break;
        }
        for i in 0..n {
            z[i] = if i == v { 0.0 } else { r[i] / deg[i] };
        }
        let rz2: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let beta = rz2 / rz;
        rz = rz2;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }
    1.0 / x[u]
}

/// Graph distance and Nash-Williams bound from the BFS shells around u: the edge sets between
/// shells j and j+1 (j < d(u,v)) are disjoint cutsets separating u from v; NW = 1 / sum_j 1/|Pi_j|.
fn nash_williams_unit(g: &Graph, u: u32, v: u32) -> (u32, f64, Vec<usize>) {
    let mut dist = vec![u32::MAX; g.n];
    let mut q = std::collections::VecDeque::new();
    dist[u as usize] = 0;
    q.push_back(u);
    while let Some(x) = q.pop_front() {
        for &(w, _) in &g.adj[x as usize] {
            if dist[w as usize] == u32::MAX {
                dist[w as usize] = dist[x as usize] + 1;
                q.push_back(w);
            }
        }
    }
    let d = dist[v as usize];
    let mut cuts = vec![0usize; d as usize];
    for &(a, b) in &g.edges {
        let (da, db) = (dist[a as usize], dist[b as usize]);
        let lo = da.min(db);
        if da.abs_diff(db) == 1 && lo < d {
            cuts[lo as usize] += 1;
        }
    }
    let nw = 1.0 / cuts.iter().map(|&c| 1.0 / c as f64).sum::<f64>();
    (d, nw, cuts)
}

fn cmd_conductance(a: &Args) {
    let inst = instance(a);
    let g = &inst.graph;
    let mut w = std::io::BufWriter::new(std::fs::OpenOptions::new().create(true).append(true).open(a.s("out", "conductance.csv")).unwrap());
    for c in &inst.classes {
        let (u, v) = c.images[0];
        let ce = ceff_unit(g, u, v);
        let (d, nw, cuts) = nash_williams_unit(g, u, v);
        // all images must agree (symmetric) unless diluted; report the spread
        let spread = c.images.iter().map(|&(x, y)| ceff_unit(g, x, y)).fold(0.0f64, |m, z| m.max((z - ce).abs()));
        let check = if g.n <= 16 && g.edges.len() <= 24 {
            // matrix-tree: R_eff = #(2-forests separating u,v) / #(spanning trees)
            let fs = brute::forests(g);
            let t = fs.iter().filter(|f| f.count_ones() as usize == g.n - 1).count() as f64;
            let f2 = fs
                .iter()
                .filter(|&&f| f.count_ones() as usize == g.n - 2 && !brute::connected_in(g, f, u, v))
                .count() as f64;
            format!("{:.3e}", (t / f2 - ce).abs())
        } else {
            String::new()
        };
        writeln!(
            w,
            "{},{},{},{},{},{d},{ce:.12e},{nw:.12e},{},{spread:.1e},{check}",
            inst.lattice,
            inst.size,
            c.name,
            graph::label(g, u),
            graph::label(g, v),
            cuts.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(";")
        )
        .unwrap();
    }
}

fn parse_pairs(s: &str) -> Vec<(u32, u32)> {
    s.split(',')
        .filter(|x| !x.is_empty())
        .map(|p| {
            let mut it = p.split('-');
            (it.next().unwrap().parse().unwrap(), it.next().unwrap().parse().unwrap())
        })
        .collect()
}

fn csv_header() -> &'static str {
    "side,lattice,size,pair,u,v,beta,value,se,tau,n,nimages,extra"
}

fn out_file(a: &Args) -> Box<dyn Write> {
    let p = a.s("out", "-");
    if p == "-" {
        Box::new(std::io::stdout())
    } else {
        let exists = std::path::Path::new(&p).exists();
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&p).unwrap();
        if !exists {
            writeln!(f, "{}", csv_header()).unwrap();
        }
        Box::new(f)
    }
}

fn row(
    w: &mut dyn Write,
    side: &str,
    inst: &Instance,
    c: &graph::PairClass,
    beta: f64,
    est: (f64, f64, f64),
    n: u64,
    extra: &str,
) {
    let g = &inst.graph;
    let (u, v) = c.images[0];
    writeln!(
        w,
        "{side},{},{},{},{},{},{beta},{:.10e},{:.4e},{:.3},{n},{},{extra}",
        inst.lattice,
        inst.size,
        c.name,
        graph::label(g, u),
        graph::label(g, v),
        est.0,
        est.1,
        est.2,
        c.images.len()
    )
    .unwrap();
    w.flush().unwrap();
}

/// Unique aux pairs over all class images, and for each class the indices of its images.
fn aux_list(inst: &Instance) -> (Vec<(u32, u32)>, Vec<Vec<usize>>) {
    let mut uniq: Vec<(u32, u32)> = Vec::new();
    let mut map: HashMap<(u32, u32), usize> = HashMap::new();
    let idx = inst
        .classes
        .iter()
        .map(|c| {
            c.images
                .iter()
                .map(|&(u, v)| {
                    let k = (u.min(v), u.max(v));
                    *map.entry(k).or_insert_with(|| {
                        uniq.push((u, v));
                        uniq.len() - 1
                    })
                })
                .collect()
        })
        .collect();
    (uniq, idx)
}

/// Adjacent classes get the tight target; distance >= 2 and opposite-corner classes the far one.
fn class_target(name: &str, near: f64, far: f64) -> f64 {
    if name.contains("adj") || name.contains("d1") {
        near
    } else {
        far
    }
}

fn signs(rng: &mut Xoshiro256PlusPlus, m: usize) -> Vec<i8> {
    (0..m).map(|_| if rng.gen::<bool>() { 1 } else { -1 }).collect()
}

// ------------------------------------------------------------------ Q: planar exact
fn cmd_q_planar(a: &Args) {
    let inst = instance(a);
    let g = &inst.graph;
    let betas = a.fl("betas", "0.5");
    let target = a.f("target-rse", 0.003);
    let target_far = a.f("target-rse-far", target);
    let qmin = a.f("qmin", 0.01);
    let nmin = a.u("min-samples", 200);
    let nmax = a.u("max-samples", 1_000_000);
    let tlim = a.f("time-limit", 600.0);
    let seed = a.u("seed", 1) as u64;
    let batch = a.u("batch", 4 * rayon::current_num_threads());
    let t0 = Instant::now();
    let (all, cidx) = aux_list(&inst);
    let fisher = planar::Fisher::new(g, &all);
    eprintln!(
        "{}: fisher nodes={} hedges={} aux={} build {:.2}s",
        graph::describe(&inst),
        fisher.nn,
        fisher.n_hedges(),
        all.len(),
        t0.elapsed().as_secs_f64()
    );
    let mut w = out_file(a);
    for (bi, &beta) in betas.iter().enumerate() {
        let t = beta.tanh();
        let tb = Instant::now();
        let mut acc = vec![stats::Acc::default(); inst.classes.len()];
        let mut n = 0usize;
        loop {
            let res: Vec<Vec<f64>> = (0..batch)
                .into_par_iter()
                .map(|k| {
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(mix(seed, bi as u64 + 1, (n + k) as u64));
                    let s = signs(&mut rng, g.edges.len());
                    let wv: Vec<f64> = s.iter().map(|&x| x as f64 * t).collect();
                    let c = fisher.correlations(&wv);
                    cidx.iter().map(|ix| ix.iter().map(|&i| c[i] * c[i]).sum::<f64>() / ix.len() as f64).collect()
                })
                .collect();
            for r in &res {
                for (i, &x) in r.iter().enumerate() {
                    acc[i].add(x);
                }
            }
            n += batch;
            let el = tb.elapsed().as_secs_f64();
            let done = acc
                .iter()
                .zip(&inst.classes)
                .all(|(x, c)| x.mean() < qmin || x.se() <= class_target(&c.name, target, target_far) * x.mean());
            if (done && n >= nmin) || n >= nmax || el >= tlim {
                break;
            }
        }
        let el = tb.elapsed().as_secs_f64();
        for (i, c) in inst.classes.iter().enumerate() {
            row(
                &mut *w,
                "Q",
                &inst,
                c,
                beta,
                (acc[i].mean(), acc[i].se(), 0.0),
                acc[i].n,
                &format!("exact-planar;secs={el:.1}"),
            );
        }
        eprintln!("beta {beta}: {n} samples in {el:.1}s ({:.4}s/sample/thread)", el * rayon::current_num_threads() as f64 / n as f64);
    }
}

// ------------------------------------------------------------------ Q: replica MC
fn cmd_q_glass(a: &Args) {
    let inst = instance(a);
    let g = &inst.graph;
    let targets = a.fl("betas", "0.5");
    let (ladder, tidx) = if a.0.contains_key("ladder") {
        let l = a.fl("ladder", "");
        let idx = targets.iter().map(|&b| l.iter().position(|&x| (x - b).abs() < 1e-12).unwrap()).collect();
        (l, idx)
    } else {
        glass::ladder(&targets, a.f("pt-ratio", 1.08))
    };
    let warm = a.u("warm", 2000);
    let meas = a.u("meas", 2000);
    let nmax = a.u("nsamp", 1000);
    let nmin = a.u("min-samples", 100);
    let tlim = a.f("time-limit", 600.0);
    let seed = a.u("seed", 1) as u64;
    let batch = a.u("batch", 2 * rayon::current_num_threads());
    let target = a.f("target-rse", 0.0);
    let target_far = a.f("target-rse-far", target);
    let qmin = a.f("qmin", 0.01);
    let sys = glass::GlassSystem::new(g);
    let all: Vec<(u32, u32)> = inst.classes.iter().flat_map(|c| c.images.iter().copied()).collect();
    eprintln!("{}: ladder {:?}", graph::describe(&inst), ladder.iter().map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>());
    let nc = inst.classes.len();
    let nt = targets.len();
    let mut acc = vec![stats::Acc::default(); nc * nt];
    let mut acc_late = vec![stats::Acc::default(); nc * nt];
    let mut swap_min = 1.0f64;
    let t0 = Instant::now();
    let mut n = 0usize;
    while n < nmax && t0.elapsed().as_secs_f64() < tlim {
        let res: Vec<glass::SampleOut> = (0..batch.min(nmax - n))
            .into_par_iter()
            .map(|k| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(mix(seed, 7, (n + k) as u64));
                let s = signs(&mut rng, g.edges.len());
                glass::run_sample(&sys, &s, &ladder, &tidx, warm, meas, &all, &mut rng)
            })
            .collect();
        for r in &res {
            for ti in 0..nt {
                let mut off = 0;
                for (ci, c) in inst.classes.iter().enumerate() {
                    let k = c.images.len();
                    let f = r.full[ti][off..off + k].iter().sum::<f64>() / k as f64;
                    let l = r.late[ti][off..off + k].iter().sum::<f64>() / k as f64;
                    acc[ti * nc + ci].add(f);
                    acc_late[ti * nc + ci].add(l);
                    off += k;
                }
            }
            for &x in &r.swap_acc {
                swap_min = swap_min.min(x);
            }
        }
        n += res.len();
        let done = (0..nt).all(|ti| {
            inst.classes.iter().enumerate().all(|(ci, c)| {
                let x = &acc[ti * nc + ci];
                x.mean() < qmin || x.se() <= class_target(&c.name, target, target_far) * x.mean()
            })
        });
        if target > 0.0 && done && n >= nmin {
            break;
        }
    }
    let el = t0.elapsed().as_secs_f64();
    let mut w = out_file(a);
    for (ti, &beta) in targets.iter().enumerate() {
        for (ci, c) in inst.classes.iter().enumerate() {
            let x = &acc[ti * nc + ci];
            let l = &acc_late[ti * nc + ci];
            row(
                &mut *w,
                "Q",
                &inst,
                c,
                beta,
                (x.mean(), x.se(), 0.0),
                x.n,
                &format!(
                    "replica-pt;warm={warm};meas={meas};late={:.6};se_late={:.2e};swapmin={swap_min:.3};secs={el:.1}",
                    l.mean(),
                    l.se()
                ),
            );
        }
    }
    eprintln!("{n} samples in {el:.1}s ({:.3}s/sample/thread), min swap acceptance {swap_min:.3}", el * rayon::current_num_threads() as f64 / n as f64);
}

// ------------------------------------------------------------------ AG
fn cmd_ag(a: &Args) {
    let inst = instance(a);
    let g = &inst.graph;
    let betas = a.fl("betas", "0.5");
    let chains = a.u("chains", rayon::current_num_threads());
    let warm = a.u("warm", 1000);
    let round = a.u("round", 2000);
    let max_sweeps = a.u("max-sweeps", 200_000);
    let refresh = a.u("refresh", 1);
    let est = a.s("estimator", "star");
    let star = est == "star";
    let star2 = est == "star2";
    let every = a.u("measure-every", 1).max(1);
    let target = a.f("target-rse", 0.003);
    let target_far = a.f("target-rse-far", target);
    let qmin = a.f("qmin", 0.01);
    let tlim = a.f("time-limit", 600.0);
    let seed = a.u("seed", 1) as u64;
    let edges_of: Vec<Vec<Option<u32>>> =
        inst.classes.iter().map(|c| c.images.iter().map(|&(u, v)| g.edge_between(u, v)).collect()).collect();
    eprintln!("{}", graph::describe(&inst));
    let mut w = out_file(a);
    for (bi, &beta) in betas.iter().enumerate() {
        let tb = Instant::now();
        // chains keep running in rounds; each round continues every chain from its state
        let mut states: Vec<Option<ag::AgChain>> = (0..chains).map(|_| None).collect();
        let mut series: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); inst.classes.len()]; chains];
        let mut done_sweeps = 0usize;
        loop {
            let first = states[0].is_none();
            let outs: Vec<(ag::AgChain, Vec<Vec<f64>>)> = states
                .par_iter_mut()
                .enumerate()
                .map(|(ci, st)| {
                    let mut ch = st.take().unwrap_or_else(|| {
                        let mut c = ag::AgChain::new(g, beta, mix(seed, 1000 + bi as u64, ci as u64));
                        if c.p > 0.5 {
                            c.init_spanning_tree();
                        }
                        c
                    });
                    if first {
                        for it in 0..warm {
                            ch.sweep();
                            if refresh > 0 && it % refresh == 0 {
                                ch.refresh_trees();
                            }
                        }
                    }
                    let mut ser = vec![Vec::with_capacity(round); inst.classes.len()];
                    for it in 0..round {
                        ch.sweep();
                        if refresh > 0 && it % refresh == 0 {
                            ch.refresh_trees();
                        }
                        if it % every != 0 {
                            continue;
                        }
                        for (k, c) in inst.classes.iter().enumerate() {
                            let mut s = 0.0;
                            for (i, &(u, v)) in c.images.iter().enumerate() {
                                s += if star2 {
                                    ch.observe_star2(u, v)
                                } else if star {
                                    ch.observe_star(u, v)
                                } else {
                                    ch.observe(u, v, edges_of[k][i])
                                };
                            }
                            ser[k].push(s / c.images.len() as f64);
                        }
                    }
                    (ch, ser)
                })
                .collect();
            for (ci, (ch, ser)) in outs.into_iter().enumerate() {
                states[ci] = Some(ch);
                for (k, s) in ser.into_iter().enumerate() {
                    series[ci][k].extend(s);
                }
            }
            done_sweeps += round;
            let ests: Vec<stats::Est> = (0..inst.classes.len())
                .map(|k| stats::combine_chains(&series.iter().map(|s| stats::binning(&s[k])).collect::<Vec<_>>()))
                .collect();
            let ok = ests
                .iter()
                .zip(&inst.classes)
                .all(|(e, c)| e.mean < qmin || e.se <= class_target(&c.name, target, target_far) * e.mean);
            let el = tb.elapsed().as_secs_f64();
            if ok || done_sweeps >= max_sweeps || el >= tlim {
                for (k, c) in inst.classes.iter().enumerate() {
                    let e = &ests[k];
                    row(
                        &mut *w,
                        "AG",
                        &inst,
                        c,
                        beta,
                        (e.mean, e.se, e.tau),
                        (done_sweeps * chains / every) as u64,
                        &format!("sweeny+wilson(refresh={refresh});est={est};chains={chains};warm={warm};secs={el:.1}"),
                    );
                }
                eprintln!(
                    "beta {beta}: {done_sweeps} sweeps x {chains} chains in {el:.1}s ({:.2}ms/sweep/chain), max tau {:.2}",
                    1000.0 * el * rayon::current_num_threads().min(chains) as f64 / (done_sweeps * chains) as f64,
                    ests.iter().map(|e| e.tau).fold(0.0, f64::max)
                );
                break;
            }
        }
    }
}

// ------------------------------------------------------------------ combine
fn cmd_combine(a: &Args) {
    let files: Vec<String> = a.s("in", "").split(',').map(String::from).collect();
    let mut q: BTreeMap<(String, String, String, String), Vec<String>> = BTreeMap::new();
    let mut ag: BTreeMap<(String, String, String, String), Vec<String>> = BTreeMap::new();
    for f in &files {
        let s = std::fs::read_to_string(f).unwrap();
        for line in s.lines().skip(1) {
            let c: Vec<String> = line.split(',').map(String::from).collect();
            if c.len() < 13 || c[0] == "side" {
                continue;
            }
            let key = (c[1].clone(), c[2].clone(), c[3].clone(), format!("{:.4}", c[6].parse::<f64>().unwrap()));
            if c[0] == "Q" {
                q.insert(key, c);
            } else {
                // duplicate AG rows: prefer the stronger estimator (star2 > star > edge), not the
                // smaller SE, so the choice does not depend on the estimates
                let rank = |r: &Vec<String>| {
                    if r[12].contains("est=star2") {
                        2
                    } else if r[12].contains("est=star") {
                        1
                    } else {
                        0
                    }
                };
                if ag.get(&key).map_or(true, |old| rank(&c) >= rank(old)) {
                    ag.insert(key, c);
                }
            }
        }
    }
    let mut w = std::fs::File::create(a.s("out", "fd_mc.csv")).unwrap();
    writeln!(w, "lattice,size,pair,u,v,beta,Q,SE_Q,AG,SE_AG,ratio,z,tau_int,n_Q,n_AG,images,method_Q,note").unwrap();
    for (key, qc) in &q {
        let Some(gc) = ag.get(key) else { continue };
        let (qv, qs): (f64, f64) = (qc[7].parse().unwrap(), qc[8].parse().unwrap());
        let (av, as_raw): (f64, f64) = (gc[7].parse().unwrap(), gc[8].parse().unwrap());
        // rare-event floor for the AG side: one connection event among all measurements
        let n_ag: f64 = gc[10].parse::<f64>().unwrap() * gc[11].parse::<f64>().unwrap();
        let as_ = as_raw.max(1.0 / n_ag);
        let qs = qs.max(1e-12);
        let ratio = if av > 0.0 { qv / av } else { f64::NAN };
        let z = (qv - av) / (qs * qs + as_ * as_).sqrt();
        let method = qc[12].split(';').next().unwrap_or("");
        let note = if av * n_ag < 100.0 || qv.abs() < 1e-10 { "unresolved" } else { "" };
        writeln!(
            w,
            "{},{},{},{},{},{},{:.8e},{:.3e},{:.8e},{:.3e},{:.6},{:.3},{},{},{},{},{},{}",
            key.0, key.1, key.2, qc[4], qc[5], qc[6], qv, qs, av, as_, ratio, z, gc[9], qc[10], gc[10], qc[11], method, note
        )
        .unwrap();
    }
}

// ------------------------------------------------------------------ self tests
fn cmd_selftest_planar(a: &Args) {
    let trials = a.u("trials", 20);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(a.u("seed", 5) as u64);
    let mut insts: Vec<Instance> = vec![graph::square(3), graph::square(4), graph::triangular(3), graph::triangular(4), graph::honeycomb(4)];
    for r in 0..6 {
        insts.push(graph::diluted_square(5, 0.75, r));
        insts.push(graph::diluted_square(4, 0.9, r));
    }
    insts.push(graph::grid_with_pairs(4, 5, &[(0, 1), (0, 3), (1, 2), (0, 19), (4, 9), (3, 16)]));
    let mut worst = 0.0f64;
    for inst in &insts {
        let g = &inst.graph;
        if g.n > 24 || inst.classes.is_empty() {
            eprintln!("skip {} (n={})", graph::describe(inst), g.n);
            continue;
        }
        let (all, _) = aux_list(inst);
        let f = planar::Fisher::new(g, &all);
        let mut err = 0.0f64;
        for tr in 0..trials {
            // +-J at several beta, and Gaussian magnitudes (frustrated, inhomogeneous)
            let j: Vec<f64> = (0..g.edges.len())
                .map(|_| {
                    let s = if rng.gen::<bool>() { 1.0 } else { -1.0 };
                    match tr % 4 {
                        0 => 0.3 * s,
                        1 => 1.0 * s,
                        2 => 2.5 * s,
                        _ => s * rng.gen_range(0.05..2.0),
                    }
                })
                .collect();
            let wv: Vec<f64> = j.iter().map(|x| x.tanh()).collect();
            let c1 = f.correlations(&wv);
            let c2 = brute::corr_exact(g, &j, &all);
            for (x, y) in c1.iter().zip(&c2) {
                err = err.max((x - y).abs());
            }
        }
        worst = worst.max(err);
        println!("planar-check {:<60} pairs={:<3} max|err|={:.2e}", graph::describe(inst), all.len(), err);
    }
    println!("planar-check worst max|err| = {worst:.2e}");
}

fn forest_index(fs: &[u32]) -> HashMap<u32, usize> {
    fs.iter().enumerate().map(|(i, &f)| (f, i)).collect()
}

fn cmd_selftest_ag(a: &Args) {
    let iters = a.u("iters", 400_000);
    let graphs: Vec<(String, Graph)> = vec![
        ("square3x3".into(), graph::square(3).graph),
        ("triangular3x3".into(), graph::triangular(3).graph),
        ("cubic2x2x2".into(), graph::cubic(2, 2, 2).graph),
    ];
    for (name, g) in &graphs {
        let fs = brute::forests(g);
        let idx = forest_index(&fs);
        let m = g.edges.len();
        for &y in &[0.3f64, 2.0, 13.15] {
            let pi: Vec<f64> = {
                let w: Vec<f64> = fs.iter().map(|f| y.powi(f.count_ones() as i32)).collect();
                let z: f64 = w.iter().sum();
                w.iter().map(|x| x / z).collect()
            };
            let p = y / (1.0 + y);
            // (a) exact kernel of the implemented single-edge step, from every forest
            let mut ch = ag::AgChain::with_y(g, y, 1);
            let mut db = 0.0f64;
            let mut stat = 0.0f64;
            let mut vec = pi.clone();
            for e in 0..m {
                let mut next = vec![0.0; fs.len()];
                let mut kern: Vec<[(usize, f64); 2]> = Vec::with_capacity(fs.len());
                for (i, &f) in fs.iter().enumerate() {
                    let mut out = [(0usize, 0.0); 2];
                    for (b, (r, pr)) in [(0.5 * p, p), (0.5 * (1.0 + p), 1.0 - p)].into_iter().enumerate() {
                        ch.load_forest(f);
                        ch.heat_bath_edge(e, r);
                        let j = idx[&ch.mask()];
                        out[b] = (j, pr);
                        next[j] += vec[i] * pr;
                    }
                    kern.push(out);
                }
                for (i, k) in kern.iter().enumerate() {
                    for &(j, _) in k {
                        let fwd: f64 = k.iter().filter(|x| x.0 == j).map(|x| x.1).sum();
                        let back: f64 = kern[j].iter().filter(|x| x.0 == i).map(|x| x.1).sum();
                        db = db.max((pi[i] * fwd - pi[j] * back).abs());
                    }
                }
                vec = next;
            }
            for i in 0..fs.len() {
                stat = stat.max((vec[i] - pi[i]).abs());
            }
            // (b) tree refresh from fixed forests vs uniform-given-partition
            let mut tv_ref = 0.0f64;
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(9);
            for _ in 0..3 {
                let f0 = fs[rng.gen_range(0..fs.len())];
                let part = |f: u32| -> Vec<u32> {
                    let mut lab = vec![0u32; g.n];
                    for v in 0..g.n {
                        lab[v] = (0..g.n).find(|&w| brute::connected_in(g, f, v as u32, w as u32)).unwrap() as u32;
                    }
                    lab
                };
                let p0 = part(f0);
                let same: Vec<u32> = fs.iter().copied().filter(|&f| f.count_ones() == f0.count_ones() && part(f) == p0).collect();
                let mut cnt: HashMap<u32, u64> = HashMap::new();
                let nrep = a.u("nrep", 100_000);
                for _ in 0..nrep {
                    ch.load_forest(f0);
                    ch.refresh_trees();
                    *cnt.entry(ch.mask()).or_default() += 1;
                }
                let mut chi = 0.0;
                let ex = nrep as f64 / same.len() as f64;
                for &f in &same {
                    chi += (cnt.get(&f).copied().unwrap_or(0) as f64 - ex).powi(2) / ex;
                }
                let tv = chi / (same.len().max(2) - 1) as f64;
                let outside: u64 = cnt.iter().filter(|(f, _)| !same.contains(f)).map(|(_, c)| *c).sum();
                assert_eq!(outside, 0, "refresh changed the partition");
                if same.len() > 1 {
                    eprintln!("  refresh start {f0:#x}: {} trees-in-partition, chi2/dof={tv:.3} (dof {})", same.len(), same.len() - 1);
                    tv_ref = tv_ref.max(tv);
                }
            }
            // (c) full chain (sweep + refresh) histogram vs pi, and P(u<->v)
            let mut ch2 = ag::AgChain::with_y(g, y, 77);
            let mut hist = vec![0u64; fs.len()];
            let (u, v) = (0u32, (g.n - 1) as u32);
            let ex = brute::ag_exact(g, y, &[(0, 1), (u, v)]);
            let mut s1 = [0.0f64; 2];
            for it in 0..iters {
                ch2.sweep();
                if it % 2 == 0 {
                    ch2.refresh_trees();
                }
                hist[idx[&ch2.mask()]] += 1;
                s1[0] += ch2.observe(0, 1, g.edge_between(0, 1));
                s1[1] += ch2.observe(u, v, g.edge_between(u, v));
            }
            let tv: f64 = 0.5 * hist.iter().zip(&pi).map(|(&h, &q)| (h as f64 / iters as f64 - q).abs()).sum::<f64>();
            let chi2: f64 = hist
                .iter()
                .zip(&pi)
                .filter(|(_, &q)| q * iters as f64 > 5.0)
                .map(|(&h, &q)| (h as f64 - q * iters as f64).powi(2) / (q * iters as f64))
                .sum();
            let dof = pi.iter().filter(|&&q| q * iters as f64 > 5.0).count().max(2) - 1;
            println!(
                "ag-check {name:<14} y={y:<6} forests={:<5} detailed-balance max={db:.1e} sweep-stationarity max={stat:.1e} refresh chi2/dof(max of 3 starts)={tv_ref:.2} chain TV={tv:.4} chi2/dof={:.2} P01 mc={:.5} ex={:.5} Pfar mc={:.5} ex={:.5}",
                fs.len(),
                chi2 / dof as f64,
                s1[0] / iters as f64,
                ex[0],
                s1[1] / iters as f64,
                ex[1]
            );
        }
    }
}

fn cmd_selftest_glass(a: &Args) {
    // (1) all-signs exact on 2x2x2 vs replica MC; (2) paired comparison vs planar exact on 2D boxes
    let nsamp = a.u("nsamp", 4000);
    let inst = graph::cubic(2, 2, 2);
    let g = &inst.graph;
    let pairs: Vec<(u32, u32)> = vec![(0, 1), (0, 3), (0, 7)];
    let sys = glass::GlassSystem::new(g);
    for &beta in &[0.5f64, 1.5] {
        let ex = brute::q_exact(g, beta, &pairs);
        let (lad, tid) = glass::ladder(&[0.3, beta], 1.15);
        let res: Vec<Vec<f64>> = (0..nsamp)
            .into_par_iter()
            .map(|k| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(mix(3, beta.to_bits(), k as u64));
                let s = signs(&mut rng, g.edges.len());
                let o = glass::run_sample(&sys, &s, &lad, &tid, 500, 1000, &pairs, &mut rng);
                o.full[1].clone()
            })
            .collect();
        for (k, p) in pairs.iter().enumerate() {
            let mut acc = stats::Acc::default();
            for r in &res {
                acc.add(r[k]);
            }
            println!(
                "glass-check cubic2x2x2 beta={beta} pair={p:?} exact={:.5} mc={:.5}+-{:.5} z={:.2}",
                ex[k],
                acc.mean(),
                acc.se(),
                (acc.mean() - ex[k]) / acc.se()
            );
        }
    }
    for &(l, beta, warm, meas) in &[(8usize, 0.5f64, 1000usize, 2000usize), (8, 1.0, 4000, 4000), (12, 2.0, 20000, 20000)] {
        let inst = graph::square(l);
        let g = &inst.graph;
        let (all, cidx) = aux_list(&inst);
        let f = planar::Fisher::new(g, &all);
        let sys = glass::GlassSystem::new(g);
        let (lad, tid) = glass::ladder(&[0.3, beta], 1.1);
        let ns = (nsamp / 4).max(50);
        let res: Vec<(Vec<f64>, Vec<f64>)> = (0..ns)
            .into_par_iter()
            .map(|k| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(mix(4, beta.to_bits() ^ l as u64, k as u64));
                let s = signs(&mut rng, g.edges.len());
                let wv: Vec<f64> = s.iter().map(|&x| x as f64 * beta.tanh()).collect();
                let c = f.correlations(&wv);
                let o = glass::run_sample(&sys, &s, &lad, &tid, warm, meas, &all, &mut rng);
                (c.iter().map(|x| x * x).collect(), o.full[1].clone())
            })
            .collect();
        for (ci, c) in inst.classes.iter().enumerate() {
            let ix = &cidx[ci];
            let k = ix.len() as f64;
            let mut d = stats::Acc::default();
            let mut e = stats::Acc::default();
            for (ex, mc) in &res {
                let a1 = ix.iter().map(|&i| ex[i]).sum::<f64>() / k;
                let a2 = ix.iter().map(|&i| mc[i]).sum::<f64>() / k;
                d.add(a2 - a1);
                e.add(a1);
            }
            println!(
                "glass-vs-planar square{l} beta={beta} {:<11} Q_exact={:.5} mean(mc-exact)={:+.5}+-{:.5} z={:.2}",
                c.name,
                e.mean(),
                d.mean(),
                d.se(),
                d.mean() / d.se()
            );
        }
    }
}

fn main() {
    faer::set_global_parallelism(faer::Par::Seq);
    let argv: Vec<String> = std::env::args().collect();
    let cmd = argv.get(1).cloned().unwrap_or_default();
    let a = Args::parse(&argv[2.min(argv.len())..]);
    if let Some(t) = a.0.get("threads") {
        rayon::ThreadPoolBuilder::new().num_threads(t.parse().unwrap()).build_global().unwrap();
    }
    match cmd.as_str() {
        "q-planar" => cmd_q_planar(&a),
        "q-glass" => cmd_q_glass(&a),
        "ag" => cmd_ag(&a),
        "combine" => cmd_combine(&a),
        "conductance" => cmd_conductance(&a),
        "selftest-planar" => cmd_selftest_planar(&a),
        "selftest-ag" => cmd_selftest_ag(&a),
        "selftest-glass" => cmd_selftest_glass(&a),
        "exact-ag" => {
            // brute-force P^AG for the instance's class pairs (tiny graphs only)
            let inst = instance(&a);
            let pairs: Vec<(u32, u32)> = inst.classes.iter().map(|c| c.images[0]).collect();
            for beta in a.fl("betas", "0.5") {
                let ex = brute::ag_exact(&inst.graph, beta.sinh().powi(2), &pairs);
                for (c, v) in inst.classes.iter().zip(ex) {
                    println!("exact-ag {} {} beta={beta} {} = {v:.10}", inst.lattice, inst.size, c.name);
                }
            }
        }
        "local" => {
            // ball of graph radius R around the first image of --class; brute force on it if small
            let inst = instance(&a);
            let g = &inst.graph;
            let c = inst.classes.iter().find(|c| c.name == a.s("class", "")).expect("class");
            let (u, v) = c.images[0];
            let r = a.u("R", 3) as u32;
            let mut dist = vec![u32::MAX; g.n];
            let mut q = std::collections::VecDeque::new();
            for s in [u, v] {
                dist[s as usize] = 0;
                q.push_back(s);
            }
            while let Some(x) = q.pop_front() {
                for &(w, _) in &g.adj[x as usize] {
                    if dist[w as usize] == u32::MAX {
                        dist[w as usize] = dist[x as usize] + 1;
                        q.push_back(w);
                    }
                }
            }
            let mut keep: Vec<bool> = dist.iter().map(|&d| d <= r).collect();
            // prune dangling trees (exact for both sides): repeatedly drop degree-1 vertices != u, v
            loop {
                let mut changed = false;
                for x in 0..g.n {
                    if !keep[x] || x as u32 == u || x as u32 == v {
                        continue;
                    }
                    let d = g.adj[x].iter().filter(|&&(w, _)| keep[w as usize]).count();
                    if d <= 1 {
                        keep[x] = false;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
            let (h, map) = g.induced(&keep);
            println!("ball R={r} (dangling trees pruned) around {}-{}: n={} m={}", graph::label(g, u), graph::label(g, v), h.n, h.edges.len());
            for &(x, y) in &h.edges {
                print!("{}-{} ", graph::label(&h, x), graph::label(&h, y));
            }
            println!();
            let (hu, hv) = (map[u as usize], map[v as usize]);
            if h.edges.len() > 26 && h.edges.len() <= 28 {
                for beta in a.fl("betas", "0.5") {
                    let ag = brute::ag_exact_stream(&h, beta.sinh().powi(2), hu, hv);
                    println!("ball exact (streamed) beta={beta}: AG={ag:.8}");
                }
            }
            if h.edges.len() <= 26 && h.n <= 24 {
                for beta in a.fl("betas", "0.5") {
                    let ag = brute::ag_exact(&h, beta.sinh().powi(2), &[(hu, hv)])[0];
                    let qx = if h.edges.len() <= 16 { brute::q_exact(&h, beta, &[(hu, hv)])[0] } else { f64::NAN };
                    println!("ball exact beta={beta}: Q={qx:.8} AG={ag:.8} ratio={:.6}", qx / ag);
                }
            }
            // MC on the ball with each estimator (single chain), to test estimators in this geometry
            let sweeps = a.u("mc-sweeps", 0);
            for beta in a.fl("betas", "0.5") {
                if sweeps == 0 {
                    break;
                }
                for est in ["edge", "star", "star2"] {
                    let mut ch = ag::AgChain::new(&h, beta, 77);
                    let e = h.edge_between(hu, hv);
                    let mut xs = Vec::with_capacity(sweeps);
                    for it in 0..sweeps + 1000 {
                        ch.sweep();
                        ch.refresh_trees();
                        if it >= 1000 {
                            xs.push(match est {
                                "edge" => ch.observe(hu, hv, e),
                                "star" => ch.observe_star(hu, hv),
                                _ => ch.observe_star2(hu, hv),
                            });
                        }
                    }
                    let b = stats::binning(&xs);
                    println!("ball MC beta={beta} est={est:<5}: AG={:.8} +- {:.2e}", b.mean, b.se);
                }
            }
        }
        "lct-check" => {
            // stress test: chain + star2 measurements on a big graph, LCT vs BFS after each step
            let inst = instance(&a);
            let g = &inst.graph;
            let beta = a.f("beta", 0.5);
            let mut ch = ag::AgChain::new(g, beta, a.u("seed", 3) as u64);
            if ch.p > 0.5 {
                ch.init_spanning_tree();
            }
            let pairs: Vec<(u32, u32)> = inst.classes.iter().flat_map(|c| c.images.iter().copied()).collect();
            let (mut bad, mut checks) = (0usize, 0usize);
            for it in 0..a.u("sweeps", 200) {
                ch.sweep();
                bad += ch.check_consistency(200);
                if it % 2 == 0 {
                    ch.refresh_trees();
                    bad += ch.check_consistency(200);
                }
                for &(u, v) in &pairs {
                    ch.observe_star2(u, v);
                    ch.observe_star(u, v);
                }
                bad += ch.check_consistency(200);
                checks += 3;
            }
            println!("lct-check {}: {checks} consistency checks x 200 queries, disagreements = {bad}", graph::describe(&inst));
        }
        "describe" => {
            let inst = instance(&a);
            println!("{}", graph::describe(&inst));
            for c in &inst.classes {
                let im: Vec<String> = c.images.iter().map(|&(u, v)| format!("{}-{}", graph::label(&inst.graph, u), graph::label(&inst.graph, v))).collect();
                println!("  {}: {}", c.name, im.join(" "));
            }
        }
        _ => {
            eprintln!("usage: fd-mc <q-planar|q-glass|ag|combine|selftest-planar|selftest-ag|selftest-glass|describe> [--key value ...]");
            std::process::exit(2);
        }
    }
}
