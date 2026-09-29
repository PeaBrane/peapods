//! XY sweep timing through the public `XySimulation` API.
//!
//! `cargo run --release -p spin-sim --example bench_xy`, configured by
//! environment variables (defaults first):
//! `PEAPODS_L` (64), `PEAPODS_DIMS` (2), `PEAPODS_N_TEMPS` (16),
//! `PEAPODS_N_REPLICAS` (2), `PEAPODS_N_SWEEPS` (50), `PEAPODS_N_REALIZATIONS` (128),
//! `PEAPODS_SWEEP_MODE` (metropolis | gibbs | none), `PEAPODS_CLUSTER_MODE`
//! (none | sw | wolff), `PEAPODS_OVERRELAXATION` (0), `PEAPODS_PT_INTERVAL` (unset),
//! `PEAPODS_WARMUP_RATIO` (0.25), `PEAPODS_COUPLINGS` (ferro | bimodal | gaussian),
//! `PEAPODS_DILUTION` (vacancy probability, 0), `PEAPODS_SEQUENTIAL` (flag).
//! Temperatures are geometric on [0.5, 2], spanning the 2D KT point.
use std::env;
use std::str::FromStr;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use spin_sim::config::*;
use spin_sim::statistics::physics::PhysicsOptions;
use spin_sim::{XyConfig, XySimulation};

fn var<T: FromStr>(name: &str, default: T) -> T {
    let Ok(value) = env::var(name) else {
        return default;
    };
    value
        .parse()
        .unwrap_or_else(|_| panic!("{name}={value} is not a valid value"))
}

fn main() {
    let l = var("PEAPODS_L", 64usize);
    let dims = var("PEAPODS_DIMS", 2usize);
    let n_temps = var("PEAPODS_N_TEMPS", 16usize);
    let n_replicas = var("PEAPODS_N_REPLICAS", 2usize);
    let n_sweeps = var("PEAPODS_N_SWEEPS", 50usize);
    let n_realizations = var("PEAPODS_N_REALIZATIONS", 128usize);
    let sweep_mode = var("PEAPODS_SWEEP_MODE", "metropolis".to_string());
    let cluster_mode = var("PEAPODS_CLUSTER_MODE", "none".to_string());
    let overrelaxation_sweeps = var("PEAPODS_OVERRELAXATION", 0usize);
    let pt_interval = env::var("PEAPODS_PT_INTERVAL")
        .ok()
        .map(|v| v.parse::<usize>().expect("PEAPODS_PT_INTERVAL"));
    let warmup_ratio = var("PEAPODS_WARMUP_RATIO", 0.25f64);
    let couplings_mode = var("PEAPODS_COUPLINGS", "ferro".to_string());
    let dilution = var("PEAPODS_DILUTION", 0.0f64);
    let sequential = env::var_os("PEAPODS_SEQUENTIAL").is_some();

    let shape = vec![l; dims];
    let n_spins = l.pow(dims as u32);
    let temperatures: Vec<f64> = (0..n_temps)
        .map(|i| 0.5 * 4.0f64.powf(i as f64 / (n_temps.max(2) - 1) as f64))
        .collect();
    let mut rng = Xoshiro256StarStar::seed_from_u64(0x5eed);
    let couplings: Vec<Vec<f64>> = (0..n_realizations)
        .map(|_| {
            (0..n_spins * dims)
                .map(|_| match couplings_mode.as_str() {
                    "ferro" => 1.0,
                    "bimodal" => {
                        if rng.gen::<bool>() {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                    "gaussian" => {
                        // Box-Muller; only the distribution shape matters here.
                        let (u, v) = (1.0 - rng.gen::<f64>(), rng.gen::<f64>());
                        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
                    }
                    other => panic!("unknown PEAPODS_COUPLINGS '{other}'"),
                })
                .collect()
        })
        .collect();
    let occupation = (dilution > 0.0).then(|| {
        (0..n_realizations)
            .map(|_| (0..n_spins).map(|_| rng.gen::<f64>() >= dilution).collect())
            .collect()
    });
    let mut simulation =
        XySimulation::new(shape, couplings, &temperatures, n_replicas, occupation, 42)
            .expect("invalid XY benchmark configuration");

    let cluster_update = match cluster_mode.as_str() {
        "none" => None,
        mode => Some(ClusterConfig {
            interval: 1,
            mode: ClusterMode::try_from(mode).expect("PEAPODS_CLUSTER_MODE"),
            action: ClusterAction::Update,
            collect_stats: false,
        }),
    };
    let warmup_sweeps = ((n_sweeps as f64 * warmup_ratio).round() as usize).min(n_sweeps - 1);
    let config = XyConfig {
        simulation: SimConfig {
            n_sweeps,
            warmup_sweeps,
            sweep_mode: SweepMode::try_from(sweep_mode.as_str()).expect("PEAPODS_SWEEP_MODE"),
            cluster_update,
            pt_interval,
            pt_schedule: PtSchedule::SingleRandomEdge,
            overlap_cluster: None,
            autocorrelation_max_lag: None,
            autocorrelation_backend: AutocorrelationBackend::Ring,
            sequential,
            equilibration_diagnostic: false,
        },
        cluster_updates: 1,
        overrelaxation_sweeps,
        physics: PhysicsOptions::default(),
    };

    let interrupted = AtomicBool::new(false);
    let start = Instant::now();
    let result = match simulation.sample(&config, &interrupted, &|| {}) {
        Ok(result) => result,
        Err(error) => {
            eprintln!("unsupported configuration: {error}");
            std::process::exit(2);
        }
    };
    let elapsed = start.elapsed();
    let energy: f64 = result.values["energies"].iter().map(|row| row[0]).sum();
    println!(
        "L={l} dims={dims} temps={n_temps} replicas={n_replicas} realizations={n_realizations} \
         sweeps={n_sweeps} warmup={warmup_sweeps} sweep={sweep_mode} cluster={cluster_mode} \
         overrelaxation={overrelaxation_sweeps} pt={pt_interval:?} couplings={couplings_mode} \
         dilution={dilution} sequential={sequential} threads={}",
        rayon::current_num_threads()
    );
    println!("mean energy checksum: {:.6}", energy / n_temps as f64);
    println!(
        "{:.3} ms/sweep ({:.3} s total)",
        elapsed.as_secs_f64() * 1e3 / n_sweeps as f64,
        elapsed.as_secs_f64()
    );
}
