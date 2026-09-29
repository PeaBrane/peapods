use spin_sim::config::*;
use spin_sim::{run_sweep_parallel, Lattice, Realization};
use std::sync::atomic::AtomicBool;

fn config(n_sweeps: usize, warmup_sweeps: usize) -> SimConfig {
    SimConfig {
        n_sweeps,
        warmup_sweeps,
        sweep_mode: SweepMode::Metropolis,
        cluster_update: None,
        pt_interval: None,
        pt_schedule: PtSchedule::SingleRandomEdge,
        overlap_cluster: None,
        autocorrelation_max_lag: None,
        autocorrelation_backend: AutocorrelationBackend::Ring,
        sequential: true,
        equilibration_diagnostic: false,
    }
}

fn realizations(lattice: &Lattice, temps: &[f32], n_replicas: usize, n: usize) -> Vec<Realization> {
    (0..n)
        .map(|r| {
            let couplings = (0..lattice.n_spins * lattice.n_neighbors)
                .map(|b| if (b * 7 + r) % 3 == 0 { -1.0 } else { 1.0 })
                .collect();
            Realization::new(lattice, couplings, temps, n_replicas, 100 + r as u64)
        })
        .collect()
}

#[test]
fn every_overlap_mode_keeps_top_cluster_shape_when_one_never_measures() {
    // Calls alternate cmr, houdayer; the only measured call uses houdayer.
    let lattice = Lattice::new(vec![4, 4]);
    let temps = [1.0, 2.0, 3.0];
    let mut config = config(2, 1);
    config.overlap_cluster = Some(OverlapClusterConfig {
        interval: 1,
        modes: parse_overlap_modes("cmr+houdayer").unwrap(),
        cluster_mode: ClusterMode::Sw,
        action: ClusterAction::Update,
        collect_stats: true,
        snapshot_interval: None,
        max_temperature: None,
    });
    let mut reals = realizations(&lattice, &temps, 2, 2);
    let result = run_sweep_parallel(
        &lattice,
        &mut reals,
        2,
        temps.len(),
        &config,
        &AtomicBool::new(false),
        &|| {},
    )
    .unwrap();

    let tops = &result.cluster_stats.top_cluster_sizes;
    assert_eq!(tops.len(), 2);
    assert!(tops.iter().all(|mode| mode.len() == temps.len()));
    assert!(tops[0].iter().flatten().all(|&v| v == 0.0));
    assert!(tops[1].iter().all(|sizes| sizes[0] > 0.0));
}

#[test]
fn energy_variance_is_the_disorder_mean_of_thermal_variances() {
    let lattice = Lattice::new(vec![4, 4]);
    let temps = [1.5, 3.0];
    let config = config(64, 16);
    let run = |reals: &mut [Realization]| {
        run_sweep_parallel(
            &lattice,
            reals,
            2,
            temps.len(),
            &config,
            &AtomicBool::new(false),
            &|| {},
        )
        .unwrap()
    };
    let singles: Vec<_> = (0..3)
        .map(|r| run(&mut realizations(&lattice, &temps, 2, 3)[r..r + 1]))
        .collect();
    let batch = run(&mut realizations(&lattice, &temps, 2, 3));

    for t in 0..temps.len() {
        let expected = singles.iter().map(|s| s.energy_variance[t]).sum::<f64>() / 3.0;
        assert!((batch.energy_variance[t] - expected).abs() < 1e-12);
        for single in &singles {
            let thermal = single.energies2[t] - single.energies[t].powi(2);
            assert!((single.energy_variance[t] - thermal).abs() < 1e-12);
        }
    }
}
