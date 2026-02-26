use std::sync::atomic::AtomicBool;
use std::time::Instant;

use rand::Rng;
use spin_sim::config::*;
use spin_sim::{run_sweep_parallel, Lattice, Realization};

const L: usize = 128;
const N_TEMPS: usize = 16;
const N_REPLICAS: usize = 2;
const N_SWEEPS: usize = 50;
const N_REALIZATIONS: usize = 100;

fn main() {
    let lattice = Lattice::new(vec![L, L]);
    let n_spins = lattice.n_spins;
    let n_neighbors = lattice.n_neighbors;

    let temps: Vec<f32> = (0..N_TEMPS)
        .map(|i| 0.1 * (50.0f32).powf(i as f32 / (N_TEMPS - 1) as f32))
        .collect();

    let n_pairs = N_REPLICAS / 2;
    let n_systems = N_REPLICAS * N_TEMPS;
    let rngs_per_real = n_systems + N_TEMPS * n_pairs;

    let mut rng = rand::thread_rng();
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

    let config = SimConfig {
        n_sweeps: N_SWEEPS,
        warmup_sweeps: 0,
        sweep_mode: SweepMode::Metropolis,
        cluster_update: None,
        pt_interval: Some(1),
        overlap_cluster: Some(OverlapClusterConfig {
            interval: 1,
            mode: OverlapClusterBuildMode::Cmr,
            cluster_mode: ClusterMode::Sw,
            update_mode: OverlapUpdateMode::Free,
            collect_csd: false,
            collect_top_clusters: false,
        }),
    };

    println!(
        "Lattice: {}x{}  |  Temps: {}  |  Replicas: {}  |  Sweeps: {}  |  Realizations: {}",
        L, L, N_TEMPS, N_REPLICAS, N_SWEEPS, N_REALIZATIONS
    );
    println!("Config: bimodal, CMR free, SW overlap, PT every sweep");
    println!("{}", "-".repeat(70));

    let t0 = Instant::now();
    run_sweep_parallel(
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

    let per_sweep = elapsed / N_SWEEPS as f64 * 1000.0;
    println!("Total: {:.3} s  |  {:.3} ms/sweep", elapsed, per_sweep);
}
