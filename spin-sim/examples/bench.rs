use std::env;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use rand::{Rng, RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use spin_sim::config::*;
use spin_sim::geometry::hypercubic;
use spin_sim::{run_sweep_parallel, Lattice, Realization};

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name).map_or(default, |value| {
        value
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be an integer"))
    })
}

fn main() {
    let l = env_usize("PEAPODS_L", 128);
    let n_dims = env_usize("PEAPODS_DIM", 2);
    let n_temps = env_usize("PEAPODS_TEMPS", 16);
    let n_replicas = env_usize("PEAPODS_REPLICAS", 2);
    let n_sweeps = env_usize("PEAPODS_SWEEPS", 50);
    let n_realizations = env_usize("PEAPODS_NREAL", 100);
    let sequential = env::var_os("PEAPODS_SEQUENTIAL").is_some();
    let overlap_cluster_mode = if env::var_os("PEAPODS_OVERLAP_WOLFF").is_some() {
        ClusterMode::Wolff
    } else {
        ClusterMode::Sw
    };
    let generic_lattice = env::var_os("PEAPODS_GENERIC_LATTICE").is_some();
    let mode = env::var("PEAPODS_MODE").unwrap_or_else(|_| "cmr".to_string());
    let couplings_kind = env::var("PEAPODS_COUPLINGS").unwrap_or_else(|_| "bimodal".to_string());
    let sweep_mode = match env::var("PEAPODS_SWEEP").as_deref().unwrap_or("metropolis") {
        "metropolis" => SweepMode::Metropolis,
        "gibbs" => SweepMode::Gibbs,
        other => panic!("unknown PEAPODS_SWEEP '{other}'"),
    };
    let shape = vec![l; n_dims];
    let lattice = if generic_lattice {
        Lattice::with_offsets(shape, hypercubic(n_dims))
    } else {
        Lattice::new(shape)
    };
    let n_spins = lattice.n_spins;
    let n_neighbors = lattice.n_neighbors;

    let t_min: f32 = env::var("PEAPODS_TMIN").map_or(0.1, |v| v.parse().unwrap());
    let t_max: f32 = env::var("PEAPODS_TMAX").map_or(5.0, |v| v.parse().unwrap());
    let temps: Vec<f32> = (0..n_temps)
        .map(|i| t_min * (t_max / t_min).powf(i as f32 / (n_temps.max(2) - 1) as f32))
        .collect();

    let n_pairs = n_replicas / 2;
    let n_systems = n_replicas * n_temps;
    let rngs_per_real = n_systems + n_temps * n_pairs;

    let mut rng = Xoshiro256StarStar::seed_from_u64(0x5eed);
    let mut realizations = Vec::with_capacity(n_realizations);
    for r in 0..n_realizations {
        let couplings: Vec<f32> = (0..n_spins * n_neighbors)
            .map(|_| match couplings_kind.as_str() {
                "bimodal" => {
                    if rng.gen::<bool>() {
                        1.0
                    } else {
                        -1.0
                    }
                }
                "ferro" => 1.0,
                // Box-Muller keeps the bench free of extra distribution crates.
                "gaussian" => {
                    let u1: f64 = 1.0 - rng.gen::<f64>();
                    let u2: f64 = rng.gen();
                    ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
                }
                other => panic!("unknown PEAPODS_COUPLINGS '{other}'"),
            })
            .collect();
        let base_seed = 42 + (r * rngs_per_real) as u64;
        realizations.push(Realization::new(
            &lattice, couplings, &temps, n_replicas, base_seed,
        ));
    }

    let interrupted = AtomicBool::new(false);
    let (cluster_update, pt_interval, overlap_cluster) = match mode.as_str() {
        "metropolis" => (None, None, None),
        "pt" => (None, Some(1), None),
        "sw" => (
            Some(ClusterConfig {
                interval: 1,
                mode: ClusterMode::Sw,
                action: ClusterAction::Update,
                collect_stats: false,
            }),
            None,
            None,
        ),
        "wolff" => (
            Some(ClusterConfig {
                interval: 1,
                mode: ClusterMode::Wolff,
                action: ClusterAction::Update,
                collect_stats: false,
            }),
            None,
            None,
        ),
        "cmr" => (
            None,
            Some(1),
            Some(OverlapClusterConfig {
                interval: 1,
                modes: vec![OverlapClusterBuildMode::Cmr],
                cluster_mode: overlap_cluster_mode,
                action: ClusterAction::Update,
                collect_stats: false,
                snapshot_interval: None,
            }),
        ),
        "sw_pt" => (
            Some(ClusterConfig {
                interval: 1,
                mode: ClusterMode::Sw,
                action: ClusterAction::Update,
                collect_stats: false,
            }),
            Some(1),
            None,
        ),
        "houdayer" | "jorg" => (
            None,
            Some(1),
            Some(OverlapClusterConfig {
                interval: 1,
                modes: vec![if mode == "jorg" {
                    OverlapClusterBuildMode::Jorg(2)
                } else {
                    OverlapClusterBuildMode::Houdayer(2)
                }],
                cluster_mode: overlap_cluster_mode,
                action: ClusterAction::Update,
                collect_stats: false,
                snapshot_interval: None,
            }),
        ),
        // Any other overlap build mode string, e.g. "pair4" or "jorg4", with PT.
        other => (
            None,
            Some(1),
            Some(OverlapClusterConfig {
                interval: 1,
                modes: parse_overlap_modes(other)
                    .unwrap_or_else(|err| panic!("unknown PEAPODS_MODE '{other}': {err}")),
                cluster_mode: overlap_cluster_mode,
                action: ClusterAction::Update,
                collect_stats: false,
                snapshot_interval: None,
            }),
        ),
    };

    let config = SimConfig {
        n_sweeps,
        warmup_sweeps: 0,
        sweep_mode,
        cluster_update,
        pt_interval,
        pt_schedule: PtSchedule::SingleRandomEdge,
        overlap_cluster,
        autocorrelation_max_lag: None,
        autocorrelation_backend: AutocorrelationBackend::Ring,
        sequential,
        equilibration_diagnostic: false,
    };

    println!(
        "Lattice: {l}^{n_dims}  |  Temps: {n_temps}  |  Replicas: {n_replicas}  |  Sweeps: {n_sweeps}  |  Realizations: {n_realizations}"
    );
    println!(
        "Config: {couplings_kind}, mode={mode}, sequential={sequential}, generic_lattice={generic_lattice}"
    );
    println!("{}", "-".repeat(70));

    let t0 = Instant::now();
    let result = run_sweep_parallel(
        &lattice,
        &mut realizations,
        n_replicas,
        n_temps,
        &config,
        &interrupted,
        &|| {},
    )
    .unwrap();
    let elapsed = t0.elapsed().as_secs_f64();

    let mut state_hash = DefaultHasher::new();
    for realization in &realizations {
        realization.spins.hash(&mut state_hash);
        realization.system_ids.hash(&mut state_hash);
        for rng in realization.rngs.iter().chain(&realization.pair_rngs) {
            let mut rng = rng.clone();
            rng.next_u64().hash(&mut state_hash);
        }
    }
    for values in [
        &result.mags,
        &result.mags2,
        &result.mags4,
        &result.energies,
        &result.energies2,
    ] {
        for value in values {
            value.to_bits().hash(&mut state_hash);
        }
    }
    for values in [
        &result.overlap_stats.overlap,
        &result.overlap_stats.overlap2,
        &result.overlap_stats.overlap4,
        &result.overlap_stats.link_overlap,
        &result.overlap_stats.link_overlap2,
        &result.overlap_stats.link_overlap4,
    ] {
        for value in values {
            value.to_bits().hash(&mut state_hash);
        }
    }
    result.overlap_stats.histogram.hash(&mut state_hash);
    result
        .overlap_stats
        .per_sample_histogram
        .hash(&mut state_hash);
    for samples in [
        &result.overlap_stats.ql_at_q_sum,
        &result.overlap_stats.ql2_at_q_sum,
    ] {
        for bins in samples {
            for value in bins {
                value.to_bits().hash(&mut state_hash);
            }
        }
    }
    for samples in [
        &result.overlap_stats.per_sample_ql_at_q_sum,
        &result.overlap_stats.per_sample_ql2_at_q_sum,
    ] {
        for temperatures in samples {
            for bins in temperatures {
                for value in bins {
                    value.to_bits().hash(&mut state_hash);
                }
            }
        }
    }

    let per_sweep = elapsed / n_sweeps as f64 * 1000.0;
    println!("Total: {:.3} s  |  {:.3} ms/sweep", elapsed, per_sweep);
    println!("State checksum: {:016x}", state_hash.finish());
}
