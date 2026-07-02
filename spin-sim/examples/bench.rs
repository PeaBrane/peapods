use std::env;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use rand::{Rng, RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use spin_sim::config::*;
use spin_sim::geometry::hypercubic;
use spin_sim::{run_sweep_parallel, Lattice, Realization};

const L: usize = 128;
const N_TEMPS: usize = 16;
const N_REPLICAS: usize = 2;
const N_SWEEPS: usize = 50;
const N_REALIZATIONS: usize = 100;

fn main() {
    let sequential = env::var_os("PEAPODS_SEQUENTIAL").is_some();
    let generic_lattice = env::var_os("PEAPODS_GENERIC_LATTICE").is_some();
    let mode = env::var("PEAPODS_MODE").unwrap_or_else(|_| "cmr".to_string());
    let lattice = if generic_lattice {
        Lattice::with_offsets(vec![L, L], hypercubic(2))
    } else {
        Lattice::new(vec![L, L])
    };
    let n_spins = lattice.n_spins;
    let n_neighbors = lattice.n_neighbors;

    let temps: Vec<f32> = (0..N_TEMPS)
        .map(|i| 0.1 * (50.0f32).powf(i as f32 / (N_TEMPS - 1) as f32))
        .collect();

    let n_pairs = N_REPLICAS / 2;
    let n_systems = N_REPLICAS * N_TEMPS;
    let rngs_per_real = n_systems + N_TEMPS * n_pairs;

    let mut rng = Xoshiro256StarStar::seed_from_u64(0x5eed);
    let mut realizations = Vec::with_capacity(N_REALIZATIONS);
    for r in 0..N_REALIZATIONS {
        let couplings: Vec<f32> = (0..n_spins * n_neighbors)
            .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
            .collect();
        let base_seed = 42 + (r * rngs_per_real) as u64;
        realizations.push(Realization::new(
            &lattice, couplings, &temps, N_REPLICAS, base_seed,
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
                cluster_mode: ClusterMode::Sw,
                action: ClusterAction::Update,
                collect_stats: false,
                snapshot_interval: None,
            }),
        ),
        _ => panic!("unknown PEAPODS_MODE '{mode}'"),
    };

    let config = SimConfig {
        n_sweeps: N_SWEEPS,
        warmup_sweeps: 0,
        sweep_mode: SweepMode::Metropolis,
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
        "Lattice: {}x{}  |  Temps: {}  |  Replicas: {}  |  Sweeps: {}  |  Realizations: {}",
        L, L, N_TEMPS, N_REPLICAS, N_SWEEPS, N_REALIZATIONS
    );
    println!(
        "Config: bimodal, mode={mode}, sequential={sequential}, generic_lattice={generic_lattice}"
    );
    println!("{}", "-".repeat(70));

    let t0 = Instant::now();
    let result = run_sweep_parallel(
        &lattice,
        &mut realizations,
        N_REPLICAS,
        N_TEMPS,
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

    let per_sweep = elapsed / N_SWEEPS as f64 * 1000.0;
    println!("Total: {:.3} s  |  {:.3} ms/sweep", elapsed, per_sweep);
    println!("State checksum: {:016x}", state_hash.finish());
}
