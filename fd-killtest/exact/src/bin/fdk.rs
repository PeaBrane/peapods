//! fdk: validation and sweep driver for the forest-domination kill test.
//!
//!   fdk validate-brute [--threads T]
//!   fdk validate-ref --refs refs.csv [--heavy] [--nsamp N] [--threads T] [--seed S]
//!   fdk sweep --kind square|tri|honey --w W --l L [--cyl] [--dilute keep,mlo,mhi,seed]
//!             [--betas b1,b2,..] [--seed S] [--rse 0.002] [--nmin N] [--nmax N] [--nfixed N]
//!             [--chunk C] [--threads T] [--exh-budget X] [--no-exh] [--two-max-w W] [--two-far-max-w W]
//!             [--pairs all|near|LABEL,..] --out rows.csv [--meta meta.txt]
//!   fdk agbench --kind K --w W --l L [--cyl] [--betas ..] [--two] [--threads T]

use fd_exact::ag::{self, PairAg};
use fd_exact::glass::{self, Acc, Plan};
use fd_exact::lattice::{Graph, Kind, Strip};
use fd_exact::brute;
use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

const BETAS: &str = "0.2,0.3,0.44,0.5,0.6,0.8,1.0,1.2,1.5,2.0,2.5,3.0";

struct Args {
    kv: HashMap<String, String>,
}

impl Args {
    fn parse(v: &[String]) -> Args {
        let mut kv = HashMap::new();
        let mut i = 0;
        while i < v.len() {
            let k = v[i].trim_start_matches("--").to_string();
            if i + 1 < v.len() && !v[i + 1].starts_with("--") {
                kv.insert(k, v[i + 1].clone());
                i += 2;
            } else {
                kv.insert(k, String::new());
                i += 1;
            }
        }
        Args { kv }
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.kv.get(k).map(|s| s.as_str())
    }
    fn has(&self, k: &str) -> bool {
        self.kv.contains_key(k)
    }
    fn num<T: std::str::FromStr>(&self, k: &str, d: T) -> T {
        match self.get(k) {
            Some(s) => s.parse().unwrap_or_else(|_| panic!("bad --{k} {s}")),
            None => d,
        }
    }
    fn list(&self, k: &str, d: &str) -> Vec<f64> {
        self.get(k).unwrap_or(d).split(',').map(|x| x.parse().expect("bad float list")).collect()
    }
}

fn max_rss_mb() -> f64 {
    // ru_maxrss: KiB on Linux, bytes on macOS
    #[repr(C)]
    struct Rusage {
        utime: [i64; 2],
        stime: [i64; 2],
        maxrss: i64,
        rest: [i64; 13],
    }
    extern "C" {
        fn getrusage(who: i32, usage: *mut Rusage) -> i32;
    }
    let mut r = Rusage { utime: [0; 2], stime: [0; 2], maxrss: 0, rest: [0; 13] };
    unsafe { getrusage(0, &mut r) };
    if cfg!(target_os = "macos") {
        r.maxrss as f64 / 1048576.0
    } else {
        r.maxrss as f64 / 1024.0
    }
}

/// AG for every pair, choosing per pair the orientation (identity or 180-degree rotation) that
/// inserts min(u, v) as late as possible. Both orientations run concurrently.
fn ag_oriented(g: &Graph, pairs: &[(usize, usize)], betas: &[f64], two: &[bool], threads: usize) -> (Vec<PairAg>, [usize; 3], f64) {
    let t0 = Instant::now();
    let n = g.n;
    let rot = g.rotated();
    let mut fw = (Vec::new(), Vec::new(), Vec::new());
    let mut rw = (Vec::new(), Vec::new(), Vec::new());
    for (i, &(a, b)) in pairs.iter().enumerate() {
        let (u, v) = (a.min(b), a.max(b));
        if n - 1 - v > u {
            rw.0.push(i);
            rw.1.push((n - 1 - v, n - 1 - u));
            rw.2.push(two[i]);
        } else {
            fw.0.push(i);
            fw.1.push((u, v));
            fw.2.push(two[i]);
        }
    }
    let both = !fw.0.is_empty() && !rw.0.is_empty();
    let th = if both { (threads / 2).max(1) } else { threads.max(1) };
    let (of, or) = std::thread::scope(|s| {
        let hf = if fw.0.is_empty() { None } else { Some(s.spawn(|| ag::ag_pairs(g, betas, &fw.1, &fw.2, th))) };
        let hr = if rw.0.is_empty() { None } else { Some(s.spawn(|| ag::ag_pairs(&rot, betas, &rw.1, &rw.2, th))) };
        (hf.map(|h| h.join().unwrap()), hr.map(|h| h.join().unwrap()))
    });
    let mut out: Vec<Option<PairAg>> = vec![None; pairs.len()];
    let mut st = [0usize; 3];
    for (o, idx) in [(of, &fw.0), (or, &rw.0)] {
        if let Some(o) = o {
            st[0] = st[0].max(o.states_base);
            for (k, &i) in idx.iter().enumerate() {
                st[1] = st[1].max(o.pairs[k].states_sub);
                st[2] = st[2].max(o.pairs[k].states_two);
                out[i] = Some(o.pairs[k].clone());
            }
        }
    }
    (out.into_iter().map(|x| x.unwrap()).collect(), st, t0.elapsed().as_secs_f64())
}

fn best_ag(p: &PairAg, b: usize) -> (f64, &'static str) {
    match &p.p_two {
        Some(t) => (t[b], "two"),
        None => (p.p_sub[b], "sub"),
    }
}

// ---------------------------------------------------------------- validate-brute

fn validate_brute(a: &Args) {
    let threads = a.num("threads", 2usize);
    let betas = [0.3, 0.7, 1.5];
    let mut cases: Vec<(String, Graph)> = Vec::new();
    for kind in [Kind::Square, Kind::Tri, Kind::Honey] {
        for &(w, l) in &[(2, 2), (2, 3), (3, 2), (2, 4), (3, 3)] {
            let s = Strip { kind, w, l, periodic: false };
            cases.push((format!("{}-{}x{}-free", kind.name(), w, l), s.graph()));
        }
    }
    cases.push(("square-3x3-cyl".into(), Strip { kind: Kind::Square, w: 3, l: 3, periodic: true }.graph()));
    cases.push(("tri-3x2-cyl".into(), Strip { kind: Kind::Tri, w: 3, l: 2, periodic: true }.graph()));
    cases.push(("honey-4x2-cyl".into(), Strip { kind: Kind::Honey, w: 4, l: 2, periodic: true }.graph()));
    cases.push(("honey-4x3-cyl".into(), Strip { kind: Kind::Honey, w: 4, l: 3, periodic: true }.graph()));
    cases.push(("square-3x4-dil".into(), Strip { kind: Kind::Square, w: 3, l: 4, periodic: false }.diluted(0.8, 0.4, 1.6, 7)));
    cases.push(("tri-3x3-dil".into(), Strip { kind: Kind::Tri, w: 3, l: 3, periodic: false }.diluted(0.75, 0.3, 1.8, 11)));
    cases.push(("square-4x3-dil".into(), Strip { kind: Kind::Square, w: 4, l: 3, periodic: false }.diluted(0.7, 0.5, 1.5, 3)));
    let (mut e_sub, mut e_two, mut e_q, mut e_rot) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut ncmp = 0;
    let mut min_slack = f64::INFINITY;
    for (name, g) in &cases {
        let comp = g.components();
        let mut pairs = Vec::new();
        for u in 0..g.n {
            for v in (u + 1)..g.n {
                pairs.push((u, v));
            }
        }
        let two = vec![true; pairs.len()];
        let o = ag::ag_pairs(g, &betas, &pairs, &two, threads);
        // rotated graph must give the same values
        let rot = g.rotated();
        let rpairs: Vec<(usize, usize)> = pairs.iter().map(|&(u, v)| (g.n - 1 - v, g.n - 1 - u)).collect();
        let orot = ag::ag_pairs(&rot, &betas, &rpairs, &two, threads);
        let plan = Plan::new(g, &pairs);
        let acc = glass::exhaustive(g, &plan, &betas, threads);
        let brute_q_ok = g.edges.len() + g.n <= 26;
        let bq: Vec<Vec<f64>> =
            if brute_q_ok { betas.iter().map(|&b| brute::q_all(g, &pairs, b)).collect() } else { vec![] };
        for (pi, &(u, v)) in pairs.iter().enumerate() {
            for (bi, &beta) in betas.iter().enumerate() {
                let agb = brute::ag(g, u, v, beta);
                let ps = o.pairs[pi].p_sub[bi];
                let pt = o.pairs[pi].p_two.as_ref().unwrap()[bi];
                let pr = orot.pairs[pi].p_two.as_ref().unwrap()[bi];
                e_sub = e_sub.max((ps - agb).abs());
                e_two = e_two.max(if agb > 0.0 { ((pt - agb) / agb).abs() } else { pt.abs() });
                e_rot = e_rot.max((pr - pt).abs());
                let qt = acc.mean(bi, pi);
                if brute_q_ok {
                    e_q = e_q.max((qt - bq[bi][pi]).abs());
                }
                if comp[u] == comp[v] {
                    min_slack = min_slack.min(agb - qt);
                }
                ncmp += 1;
            }
        }
        println!(
            "{name}: n={} m={} win={} cyc={} pairs={} states base={} brute_q={}",
            g.n,
            g.edges.len(),
            g.win,
            g.cyclomatic(),
            pairs.len(),
            o.states_base,
            brute_q_ok
        );
    }
    println!("compared {ncmp} (pair, beta) values");
    println!("max |P_sub - P_brute|        = {e_sub:.3e}");
    println!("max rel |P_two - P_brute|    = {e_two:.3e}");
    println!("max |P_two(rot) - P_two|     = {e_rot:.3e}");
    println!("max |Q_TM_exh - Q_brute|     = {e_q:.3e}");
    println!("min (AG - Q) over connected  = {min_slack:.3e}");
}

// ---------------------------------------------------------------- validate-ref

struct RefRow {
    source: String,
    kind: String,
    w: usize,
    l: usize,
    beta: f64,
    u: usize,
    v: usize,
    q: f64,
    q_se: f64,
    ag: f64,
}

fn read_refs(path: &str) -> Vec<RefRow> {
    let s = std::fs::read_to_string(path).expect("cannot read refs");
    s.lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            RefRow {
                source: f[0].into(),
                kind: f[1].into(),
                w: f[2].parse().unwrap(),
                l: f[3].parse().unwrap(),
                beta: f[4].parse().unwrap(),
                u: f[5].parse().unwrap(),
                v: f[6].parse().unwrap(),
                q: f[7].parse().unwrap(),
                q_se: f[8].parse().unwrap(),
                ag: f[9].parse().unwrap(),
            }
        })
        .collect()
}

fn validate_ref(a: &Args) {
    let refs = read_refs(a.get("refs").unwrap_or("refs.csv"));
    let heavy = a.has("heavy");
    let threads = a.num("threads", 2usize);
    let nsamp = a.num("nsamp", 20000u64);
    let seed = a.num("seed", 12345u64);
    let only = a.get("only").map(|s| s.to_string());
    // group by (w, L, kind class)
    let mut groups: Vec<((usize, usize, String), Vec<usize>)> = Vec::new();
    for (i, r) in refs.iter().enumerate() {
        if r.kind == "exact-heavy" && !heavy {
            continue;
        }
        if r.kind == "mcQ" && r.w >= 10 && !heavy {
            continue;
        }
        if let Some(o) = &only {
            if !r.source.contains(o.as_str()) {
                continue;
            }
        }
        let cls = if r.kind.starts_with("exact") && r.kind != "exactQ-sample" { "exh" } else { "mc" };
        let key = (r.w, r.l, cls.to_string());
        match groups.iter_mut().find(|(k, _)| *k == key) {
            Some((_, l)) => l.push(i),
            None => groups.push((key, vec![i])),
        }
    }
    let mut worst_ag_rel = 0.0f64;
    let mut worst_q_abs = 0.0f64;
    let mut worst_z = 0.0f64;
    for ((w, l, cls), idx) in &groups {
        let s = Strip { kind: Kind::Square, w: *w, l: *l, periodic: false };
        let g = s.graph();
        let mut betas: Vec<f64> = Vec::new();
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for &i in idx {
            let r = &refs[i];
            if !betas.iter().any(|&b| (b - r.beta).abs() < 1e-15) {
                betas.push(r.beta);
            }
            if !pairs.contains(&(r.u, r.v)) {
                pairs.push((r.u, r.v));
            }
        }
        let t0 = Instant::now();
        let two: Vec<bool> = pairs.iter().map(|_| *w <= 12).collect();
        let (agp, st, ag_secs) = ag_oriented(&g, &pairs, &betas, &two, threads);
        let plan = Plan::new(&g, &pairs);
        let t1 = Instant::now();
        let acc: Acc = if cls == "exh" {
            glass::exhaustive(&g, &plan, &betas, threads)
        } else {
            let chunk = 50u64;
            glass::sample(&g, &plan, &betas, seed, 0, nsamp / chunk, chunk as usize, threads)
        };
        let gl_secs = t1.elapsed().as_secs_f64();
        println!(
            "== square {w}x{l} [{cls}] cyc={} betas={} pairs={} ag {:.2}s states(base,sub,two)={:?} glass {:.2}s n={}",
            g.cyclomatic(),
            betas.len(),
            pairs.len(),
            ag_secs,
            st,
            gl_secs,
            acc.count
        );
        for &i in idx {
            let r = &refs[i];
            let bi = betas.iter().position(|&b| (b - r.beta).abs() < 1e-15).unwrap();
            let pi = pairs.iter().position(|&p| p == (r.u, r.v)).unwrap();
            let (agv, how) = best_ag(&agp[pi], bi);
            let ag_rel = ((agv - r.ag) / r.ag).abs();
            worst_ag_rel = worst_ag_rel.max(ag_rel);
            let q = acc.mean(bi, pi);
            let se = acc.se(bi, pi);
            let line = if cls == "exh" {
                let d = (q - r.q).abs();
                worst_q_abs = worst_q_abs.max(d / r.q.max(1e-300));
                format!("Q {q:.15e} ref {:.15e} relerr {:.1e}", r.q, d / r.q)
            } else {
                let z = (q - r.q) / (se * se + r.q_se * r.q_se).sqrt();
                worst_z = worst_z.max(z.abs());
                format!("Q {q:.6} +- {se:.2e}  ref {:.6} +- {:.2e}  z {z:+.2}", r.q, r.q_se)
            };
            println!(
                "  {} b={:.4} ({},{}): AG[{how}] {agv:.15e} ref {:.15e} relerr {ag_rel:.1e} | sub-two {:.1e} | {line} | Q/AG {:.5}",
                r.source,
                r.beta,
                r.u,
                r.v,
                r.ag,
                agp[pi].p_two.as_ref().map(|t| (t[bi] - agp[pi].p_sub[bi]).abs()).unwrap_or(f64::NAN),
                q / agv
            );
        }
        let _ = t0;
    }
    println!("worst AG relerr {worst_ag_rel:.2e}; worst exact-Q relerr {worst_q_abs:.2e}; worst |z| vs MC refs {worst_z:.2}");
    println!("max RSS {:.1} MB", max_rss_mb());
}

// ---------------------------------------------------------------- sweep

fn sweep(a: &Args) {
    let kind = Kind::parse(a.get("kind").expect("--kind")).expect("bad kind");
    let w: usize = a.num("w", 0);
    let l: usize = a.num("l", 0);
    let periodic = a.has("cyl");
    let strip = Strip { kind, w, l, periodic };
    let (g, inst) = match a.get("dilute") {
        Some(d) => {
            let f: Vec<f64> = d.split(',').map(|x| x.parse().unwrap()).collect();
            (strip.diluted(f[0], f[1], f[2], f[3] as u64), format!("dil{}_{}-{}_s{}", f[0], f[1], f[2], f[3] as u64))
        }
        None => (strip.graph(), "pure".to_string()),
    };
    let betas = a.list("betas", BETAS);
    let nb = betas.len();
    let seed = a.num("seed", 1u64);
    let threads = a.num("threads", 2usize);
    let rse = a.num("rse", 0.002f64);
    let chunk = a.num("chunk", 32u64);
    let nmin = a.num("nmin", 4096u64);
    let nmax = a.num("nmax", 262144u64);
    let nfixed: Option<u64> = a.get("nfixed").map(|s| s.parse().unwrap());
    let exh_budget = a.num("exh-budget", 3e11f64);
    let two_max_w = a.num("two-max-w", 9usize);
    let two_far_max_w = a.num("two-far-max-w", 10usize);
    let out_path = a.get("out").expect("--out");
    let t_start = Instant::now();

    let comp = g.components();
    let sel = a.get("pairs").unwrap_or("all");
    let mut labels = Vec::new();
    let mut pairs = Vec::new();
    let mut dists = Vec::new();
    let mut cand_pairs = strip.pairs();
    if let Some(uv) = a.get("uv") {
        // custom pairs "x1:y1-x2:y2,..." in strip coordinates
        cand_pairs.clear();
        for item in uv.split(',') {
            let pts: Vec<Vec<usize>> = item.split('-').map(|p| p.split(':').map(|x| x.parse().unwrap()).collect()).collect();
            cand_pairs.push((format!("custom-{}", item.replace(':', ".")), strip.vid(pts[0][0], pts[0][1]), strip.vid(pts[1][0], pts[1][1])));
        }
    }
    for (lab, u, v) in cand_pairs {
        if comp[u] != comp[v] {
            continue;
        }
        let d = g.distance(u, v);
        let long = lab == "end-to-end-l" || lab == "opp-corners";
        if long && g.win >= a.num("long-max-w", 10usize) + 1 {
            continue;
        }
        let keep = match sel {
            "all" => true,
            "near" => d <= 3,
            s => s.split(',').any(|x| x == lab) || lab.starts_with("custom-"),
        };
        if !keep || pairs.contains(&(u.min(v), u.max(v))) {
            continue;
        }
        labels.push(lab);
        pairs.push((u.min(v), u.max(v)));
        dists.push(d);
    }
    eprintln!(
        "[{} {}x{} {} {}] n={} m={} win={} cyc={} pairs={}",
        kind.name(),
        w,
        l,
        strip.bc(),
        inst,
        g.n,
        g.edges.len(),
        g.win,
        g.cyclomatic(),
        pairs.len()
    );
    // AG
    let two: Vec<bool> = dists.iter().map(|&d| g.win <= two_max_w || (d >= 3 && g.win <= two_far_max_w)).collect();
    let (agp, st, ag_secs) = ag_oriented(&g, &pairs, &betas, &two, threads);
    let rss_ag = max_rss_mb();
    eprintln!("  AG {:.2}s states base={} sub={} two={} rss={:.0}MB", ag_secs, st[0], st[1], st[2], rss_ag);
    // glass
    let plan = Plan::new(&g, &pairs);
    let cyc = g.cyclomatic();
    let exh_cost = (2f64).powi(cyc as i32) * g.n as f64 * (1u64 << g.win) as f64 * nb as f64;
    let use_exh = !a.has("no-exh") && cyc <= 34 && exh_cost <= exh_budget;
    let t1 = Instant::now();
    let mut accs: Vec<Acc> = Vec::new();
    let qmethod;
    if use_exh {
        let acc = glass::exhaustive(&g, &plan, &betas, threads);
        for b in 0..nb {
            let mut x = Acc::new(1, pairs.len());
            x.count = acc.count;
            for p in 0..pairs.len() {
                x.s1[p] = acc.s1[b * pairs.len() + p];
                x.s2[p] = acc.s2[b * pairs.len() + p];
            }
            accs.push(x);
        }
        qmethod = "exact";
    } else {
        qmethod = "sample";
        accs = (0..nb).map(|_| Acc::new(1, pairs.len())).collect();
        let target_pairs: Vec<usize> = (0..pairs.len()).filter(|&p| dists[p] <= 1).collect();
        let mut active: Vec<usize> = (0..nb).collect();
        let mut done = 0u64;
        let mut target = match nfixed {
            Some(n) => n.div_ceil(chunk),
            None => nmin.div_ceil(chunk),
        };
        let cmax = match nfixed {
            Some(n) => n.div_ceil(chunk),
            None => nmax.div_ceil(chunk),
        };
        loop {
            let bsub: Vec<f64> = active.iter().map(|&b| betas[b]).collect();
            let acc = glass::sample(&g, &plan, &bsub, seed, done, target, chunk as usize, threads);
            for (k, &b) in active.iter().enumerate() {
                for p in 0..pairs.len() {
                    accs[b].s1[p] += acc.s1[k * pairs.len() + p];
                    accs[b].s2[p] += acc.s2[k * pairs.len() + p];
                }
                accs[b].count += acc.count;
            }
            done = target;
            if done >= cmax {
                break;
            }
            let mut need = 0u64;
            active.retain(|&b| {
                let worst = target_pairs
                    .iter()
                    .map(|&p| {
                        let (agv, _) = best_ag(&agp[p], b);
                        if agv < 1e-3 {
                            0.0
                        } else {
                            accs[b].se(0, p) / agv
                        }
                    })
                    .fold(0.0f64, f64::max);
                if worst > rse {
                    // chunks needed for SE/AG = rse, with 15% margin
                    let req = (done as f64 * (worst / rse).powi(2) * 1.15).ceil() as u64;
                    need = need.max(req);
                    true
                } else {
                    false
                }
            });
            if active.is_empty() {
                break;
            }
            target = need.max(done + done / 4).min(cmax);
        }
    }
    let gl_secs = t1.elapsed().as_secs_f64();
    // rows
    let new_file = !std::path::Path::new(out_path).exists();
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(out_path).unwrap();
    if new_file {
        writeln!(f, "lattice,w,L,bc,inst,pair,u,v,dist,beta,Q,SE,AG,ratio,z,nsamp,q_method,ag_method,ag_sub,ag_two,seed,flag").unwrap();
    }
    let mut worst_ratio = 0.0f64;
    let mut worst_z = f64::NEG_INFINITY;
    for b in 0..nb {
        for p in 0..pairs.len() {
            let (agv, how) = best_ag(&agp[p], b);
            let q = accs[b].mean(0, p);
            let se = if use_exh { 0.0 } else { accs[b].se(0, p) };
            let ratio = q / agv;
            let z = if use_exh { f64::NAN } else { (q - agv) / (se * se + (1e-8 * agv).powi(2)).sqrt() };
            // tinyAG: f64 spin sums are not trusted; subprec: AG from 1 - Z(G/uv)/Z(G) below its resolution
            let flag = if agv < 1e-30 {
                "tinyAG"
            } else if how == "sub" && agv < 1e-9 {
                "subprec"
            } else if (q - agv).abs() <= 1e-9 * agv && se <= 1e-9 * agv {
                "equal"
            } else {
                ""
            };
            if flag.is_empty() {
                worst_ratio = worst_ratio.max(ratio);
                if z.is_finite() {
                    worst_z = worst_z.max(z);
                }
            }
            writeln!(
                f,
                "{},{},{},{},{},{},{},{},{},{},{:.12e},{:.6e},{:.12e},{:.9},{:.4},{},{},{},{:.12e},{},{},{}",
                kind.name(),
                w,
                l,
                strip.bc(),
                inst,
                labels[p],
                pairs[p].0,
                pairs[p].1,
                dists[p],
                betas[b],
                q,
                se,
                agv,
                ratio,
                z,
                accs[b].count,
                qmethod,
                how,
                agp[p].p_sub[b],
                agp[p].p_two.as_ref().map(|t| format!("{:.12e}", t[b])).unwrap_or_default(),
                seed,
                flag
            )
            .unwrap();
        }
    }
    let ns: Vec<String> = accs.iter().map(|x| format!("{}", x.count)).collect();
    let meta = format!(
        "{} w={} L={} bc={} inst={} n={} m={} win={} cyc={} pairs={} ag_secs={:.2} states_base={} states_sub={} states_two={} glass={} glass_secs={:.2} nsamp=[{}] worst_ratio={:.6} worst_z={:.3} total_secs={:.2} max_rss_mb={:.0} threads={}",
        kind.name(),
        w,
        l,
        strip.bc(),
        inst,
        g.n,
        g.edges.len(),
        g.win,
        cyc,
        pairs.len(),
        ag_secs,
        st[0],
        st[1],
        st[2],
        qmethod,
        gl_secs,
        ns.join(" "),
        worst_ratio,
        worst_z,
        t_start.elapsed().as_secs_f64(),
        max_rss_mb(),
        threads
    );
    eprintln!("  {meta}");
    if let Some(mp) = a.get("meta") {
        let mut m = std::fs::OpenOptions::new().create(true).append(true).open(mp).unwrap();
        writeln!(m, "{meta}").unwrap();
    }
}

// ---------------------------------------------------------------- agbench

fn agbench(a: &Args) {
    let kind = Kind::parse(a.get("kind").expect("--kind")).expect("bad kind");
    let strip = Strip { kind, w: a.num("w", 4), l: a.num("l", 8), periodic: a.has("cyl") };
    let g = strip.graph();
    let betas = a.list("betas", BETAS);
    let threads = a.num("threads", 1usize);
    let pairs: Vec<(usize, usize)> = strip
        .pairs()
        .into_iter()
        .filter(|(lab, _, _)| a.get("pairs").map(|s| s.split(',').any(|x| x == lab)).unwrap_or(true))
        .map(|(_, u, v)| (u.min(v), u.max(v)))
        .collect();
    let two = vec![a.has("two"); pairs.len()];
    let (agp, st, secs) = ag_oriented(&g, &pairs, &betas, &two, threads);
    println!("{} {}x{} win={} pairs={} secs={:.2} states={:?} rss={:.0}MB", kind.name(), strip.w, strip.l, g.win, pairs.len(), secs, st, max_rss_mb());
    for (i, p) in pairs.iter().enumerate() {
        println!("  {:?} sub={:.10e} two={:?}", p, agp[i].p_sub[0], agp[i].p_two.as_ref().map(|t| t[0]));
    }
}

fn glassbench(a: &Args) {
    let kind = Kind::parse(a.get("kind").expect("--kind")).expect("bad kind");
    let strip = Strip { kind, w: a.num("w", 4), l: a.num("l", 8), periodic: a.has("cyl") };
    let g = strip.graph();
    let betas = a.list("betas", "0.5");
    let threads = a.num("threads", 1usize);
    let n = a.num("n", 64u64);
    let pairs: Vec<(usize, usize)> = strip.pairs().into_iter().map(|(_, u, v)| (u.min(v), u.max(v))).collect();
    let plan = Plan::new(&g, &pairs);
    let t0 = Instant::now();
    let acc = glass::sample(&g, &plan, &betas, 1, 0, n, 1, threads);
    let s = t0.elapsed().as_secs_f64();
    println!(
        "{} {}x{} win={} pairs={} samples={} betas={} secs={:.3} ms/(sample*beta)={:.3} ns/(site*state*sample*beta)={:.3}",
        kind.name(),
        strip.w,
        strip.l,
        g.win,
        pairs.len(),
        acc.count,
        betas.len(),
        s,
        1e3 * s * threads as f64 / (acc.count * betas.len() as f64),
        1e9 * s * threads as f64 / (acc.count * betas.len() as f64 * g.n as f64 * (1u64 << g.win) as f64)
    );
}

// ---------------------------------------------------------------- summarize

fn summarize(a: &Args) {
    let files: Vec<&str> = a.get("csv").expect("--csv a.csv,b.csv").split(',').collect();
    let top = a.num("top", 10usize);
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut header: Vec<String> = Vec::new();
    for f in &files {
        let s = std::fs::read_to_string(f).unwrap_or_else(|_| panic!("cannot read {f}"));
        let mut lines = s.lines();
        header = lines.next().unwrap().split(',').map(String::from).collect();
        for l in lines {
            if l.starts_with("lattice,") || l.trim().is_empty() {
                continue;
            }
            rows.push(l.split(',').map(String::from).collect());
        }
    }
    let col = |n: &str| header.iter().position(|h| h == n).unwrap_or_else(|| panic!("no column {n}"));
    let (ci, cq, cag, cz, cr, cm, cfl) = (col("inst"), col("Q"), col("AG"), col("z"), col("ratio"), col("q_method"), col("flag"));
    let f = |r: &Vec<String>, c: usize| r[c].parse::<f64>().unwrap_or(f64::NAN);
    let class = |r: &Vec<String>| format!("{}-{}{}", r[0], r[3], if r[ci] == "pure" { "" } else { "-dil" });
    let mut classes: Vec<String> = rows.iter().map(|r| class(r)).collect();
    classes.sort();
    classes.dedup();
    println!("rows {} from {} files", rows.len(), files.len());
    let mut cand = Vec::new();
    for c in &classes {
        let rs: Vec<&Vec<String>> = rows.iter().filter(|r| class(r) == *c).collect();
        let good: Vec<&&Vec<String>> = rs.iter().filter(|r| r[cfl].is_empty()).collect();
        let nflag = |fl: &str| rs.iter().filter(|r| r[cfl] == fl).count();
        let maxr = good.iter().map(|r| f(r, cr)).fold(f64::NEG_INFINITY, f64::max);
        let maxz = good.iter().map(|r| f(r, cz)).filter(|z| z.is_finite()).fold(f64::NEG_INFINITY, f64::max);
        let nexact = good.iter().filter(|r| r[cm] == "exact").count();
        let res: Vec<&&&Vec<String>> = good.iter().filter(|r| f(r, col("SE")) <= 0.002 * f(r, cag)).collect();
        let maxr_res = res.iter().map(|r| f(r, cr)).fold(f64::NEG_INFINITY, f64::max);
        let maxr_exact = good.iter().filter(|r| r[cm] == "exact").map(|r| f(r, cr)).fold(f64::NEG_INFINITY, f64::max);
        let ws: std::collections::BTreeSet<usize> = rs.iter().map(|r| r[1].parse().unwrap()).collect();
        println!(
            "{c}: rows {} (scored {}, exact {}, resolved {}, equal {}, tinyAG {}, subprec {}) widths {:?} max Q/AG all {:.6} resolved {:.6} exact {:.6} | max z {:.2}",
            rs.len(),
            good.len(),
            nexact,
            res.len(),
            nflag("equal"),
            nflag("tinyAG"),
            nflag("subprec"),
            ws,
            maxr,
            maxr_res,
            maxr_exact,
            maxz
        );
        for r in &good {
            let z = f(r, cz);
            if (r[cm] == "exact" && f(r, cq) > f(r, cag) * (1.0 + 1e-9)) || (z.is_finite() && z > 5.0) {
                cand.push((*r).clone());
            }
        }
    }
    let show = |r: &Vec<String>| {
        format!(
            "{} {}x{} {} {} {}({},{}) d={} b={} Q={:.6e} SE={:.2e} AG={:.6e} Q/AG={:.6} z={} n={} {}",
            r[0], r[1], r[2], r[3], r[ci], r[5], r[6], r[7], r[8], r[9], f(r, cq), f(r, col("SE")), f(r, cag), f(r, cr), r[cz], r[col("nsamp")], r[cm]
        )
    };
    let mut good: Vec<&Vec<String>> = rows.iter().filter(|r| r[cfl].is_empty()).collect();
    good.sort_by(|x, y| f(y, cr).partial_cmp(&f(x, cr)).unwrap());
    println!("top {top} by Q/AG:");
    for r in good.iter().take(top) {
        println!("  {}", show(r));
    }
    let mut gb: Vec<&&Vec<String>> = good.iter().filter(|r| f(r, col("SE")) <= 0.002 * f(r, cag)).collect();
    gb.sort_by(|x, y| f(y, cr).partial_cmp(&f(x, cr)).unwrap());
    println!("top {top} by Q/AG among resolved rows (exact, or SE <= 0.2% of AG):");
    for r in gb.iter().take(top) {
        println!("  {}", show(r));
    }
    let mut gb: Vec<&&Vec<String>> =
        good.iter().filter(|r| f(r, 9) >= 0.5 && f(r, col("SE")) <= 0.002 * f(r, cag)).collect();
    gb.sort_by(|x, y| f(y, cr).partial_cmp(&f(x, cr)).unwrap());
    println!("top {top} by Q/AG among resolved rows with beta >= 0.5:");
    for r in gb.iter().take(top) {
        println!("  {}", show(r));
    }
    let mut gz: Vec<&&Vec<String>> = good.iter().filter(|r| f(r, cz).is_finite()).collect();
    gz.sort_by(|x, y| f(y, cz).partial_cmp(&f(x, cz)).unwrap());
    println!("top {top} by z:");
    for r in gz.iter().take(top) {
        println!("  {}", show(r));
    }
    println!("candidates (exact Q > AG(1+1e-9), or sampled z > 5): {}", cand.len());
    for r in &cand {
        println!("  CANDIDATE {}", show(r));
    }
}

fn decay_cmd(a: &Args) {
    let kind = Kind::parse(a.get("kind").expect("--kind")).expect("bad kind");
    let w: usize = a.num("w", 4);
    let ag = a.get("ag").unwrap_or("two");
    let cfg = fd_exact::targeted::DecayCfg {
        kind,
        w,
        l: a.num("l", 128),
        periodic: a.has("cyl"),
        betas: a.list("betas", BETAS),
        ag_two: ag == "two" || ag == "both",
        ag_surv: ag == "survival" || ag == "both",
        sparse: a.has("sparse"),
        nsamp: a.num("nsamp", 16384),
        batches: a.num("batches", 8),
        chunk: a.num("chunk", 32),
        seed: a.num("seed", 1),
        threads: a.num("threads", 2),
        out: a.get("out").expect("--out").to_string(),
        fit_out: a.get("fit").expect("--fit").to_string(),
    };
    eprintln!("[decay {} {}x{} ag={}]", kind.name(), w, cfg.l, ag);
    fd_exact::targeted::decay(&cfg);
    eprintln!("  max_rss_mb={:.0}", max_rss_mb());
}

fn series_cmd(a: &Args) {
    let (eq, ep) = fd_exact::targeted::pendant_selfcheck();
    println!("pendant identity self-check vs brute force: max|err| Q-side {eq:.2e}, AG-side {ep:.2e}");
    let cfg = fd_exact::targeted::SeriesCfg {
        betas: a.list("betas", "0.3,0.5,0.8,1.2,2.0"),
        threads: a.num("threads", 2),
        max_cyc: a.num("max-cyc", 20),
        out_prefix: a.get("out").unwrap_or("series").to_string(),
    };
    fd_exact::targeted::series(&cfg);
    println!("max_rss_mb={:.0}", max_rss_mb());
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let a = Args::parse(&argv[2.min(argv.len())..]);
    match argv.get(1).map(|s| s.as_str()) {
        Some("validate-brute") => validate_brute(&a),
        Some("validate-ref") => validate_ref(&a),
        Some("sweep") => sweep(&a),
        Some("agbench") => agbench(&a),
        Some("glassbench") => glassbench(&a),
        Some("summarize") => summarize(&a),
        Some("decay") => decay_cmd(&a),
        Some("series") => series_cmd(&a),
        _ => eprintln!("usage: fdk validate-brute|validate-ref|sweep|agbench|glassbench [--opts]"),
    }
}
