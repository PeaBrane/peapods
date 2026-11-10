//! Shared scheduling and ordered disorder dispatch. Model hooks are monomorphized.
use super::realization::ModelRealization;
use crate::{
    config::{OverlapClusterBuildMode, SimConfig},
    geometry::Lattice,
    spins::model::Spin,
};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) struct SweepStep {
    pub index: usize,
    pub record: bool,
    pub cluster: bool,
    pub temper: bool,
}

pub(crate) fn run_sweeps(
    config: &SimConfig,
    interrupted: &AtomicBool,
    on_sweep: &(dyn Fn() + Sync),
    mut step: impl FnMut(SweepStep),
) -> Result<(), String> {
    for index in 0..config.n_sweeps {
        if interrupted.load(Ordering::Relaxed) {
            return Err("interrupted".into());
        }
        on_sweep();
        step(SweepStep {
            index,
            record: index >= config.warmup_sweeps,
            cluster: config
                .cluster_update
                .as_ref()
                .is_some_and(|c| index % c.interval == 0),
            temper: config
                .pt_interval
                .is_some_and(|interval| index % interval == 0),
        });
    }
    if interrupted.load(Ordering::Relaxed) {
        return Err("interrupted".into());
    }
    Ok(())
}

/// Every rule that needs the replica and temperature counts as well as the config.
pub(crate) fn validate_layout(
    config: &SimConfig,
    n_replicas: usize,
    n_temps: usize,
) -> Result<(), String> {
    config.validate()?;
    if config.equilibration_diagnostic && n_replicas < 2 {
        return Err("equilibration_diagnostic needs n_replicas >= 2 for the link overlap".into());
    }
    let Some(overlap) = &config.overlap_cluster else {
        return Ok(());
    };
    let max_group = overlap.max_group_size();
    if n_replicas < max_group {
        return Err(format!(
            "overlap cluster moves need n_replicas >= {max_group} (the largest mode's group size), got {n_replicas}"
        ));
    }
    if n_temps < 2 && overlap.modes.contains(&OverlapClusterBuildMode::Rmc) {
        return Err("rmc requires at least two temperatures".into());
    }
    Ok(())
}

/// Whether replicas run on the calling thread. Once disorder realizations alone
/// occupy every thread, nested replica parallelism only adds scheduling overhead;
/// results do not depend on the choice.
pub(crate) fn effective_sequential(config: &SimConfig, n_realizations: usize) -> bool {
    config.sequential || n_realizations >= rayon::current_num_threads()
}

pub(crate) fn validate_batch<S: Spin>(
    lattice: &Lattice,
    realizations: &[ModelRealization<S>],
    n_replicas: usize,
    n_temps: usize,
    config: &SimConfig,
) -> Result<(), String> {
    validate_layout(config, n_replicas, n_temps)?;
    if realizations.is_empty() {
        return Err("at least one disorder realization is required".into());
    }
    for real in realizations {
        real.validate(lattice, n_replicas, n_temps)?;
        if real
            .temperatures
            .chunks_exact(n_temps)
            .any(|temperatures| temperatures != &realizations[0].temperatures[..n_temps])
        {
            return Err(
                "temperature ladders must agree across replicas and disorder realizations".into(),
            );
        }
    }
    Ok(())
}

pub(crate) fn map_realizations<S: Spin, R: Send>(
    realizations: &mut [ModelRealization<S>],
    body: impl Fn(usize, &mut ModelRealization<S>) -> Result<R, String> + Sync + Send,
) -> Result<Vec<R>, String> {
    if realizations.len() == 1 {
        return Ok(vec![body(0, &mut realizations[0])?]);
    }
    let results: Vec<_> = realizations
        .par_iter_mut()
        .enumerate()
        .map(|(i, r)| body(i, r))
        .collect();
    results.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::XyConfig;
    #[test]
    fn cancellation_during_final_sweep_is_reported() {
        let mut config = XyConfig::default().simulation;
        config.n_sweeps = 1;
        let interrupted = AtomicBool::new(false);
        let result = run_sweeps(&config, &interrupted, &|| {}, |_| {
            interrupted.store(true, Ordering::Relaxed)
        });
        assert_eq!(result.unwrap_err(), "interrupted");
    }
    #[test]
    fn equilibration_diagnostic_needs_a_replica_pair() {
        let mut config = XyConfig::default().simulation;
        config.equilibration_diagnostic = true;
        let error = validate_layout(&config, 1, 2).unwrap_err();
        assert!(error.contains("n_replicas >= 2"), "{error}");
        assert!(validate_layout(&config, 2, 2).is_ok());
    }

    #[test]
    fn validates_entire_batch_and_equal_ladders_before_sampling() {
        let mut simulation =
            crate::XySimulation::new(vec![3, 3], vec![vec![1.0; 18]; 2], &[1.0, 2.0], 2, None, 15)
                .unwrap();
        let initial = simulation.realizations[0].spins.clone();
        let config = XyConfig::default();
        simulation.realizations[1].system_ids[1] = 0;
        assert!(simulation
            .sample(&config, &AtomicBool::new(false), &|| {})
            .err()
            .unwrap()
            .contains("permutation"));
        assert_eq!(simulation.realizations[0].spins, initial);
        simulation.realizations[1].system_ids[1] = 1;
        simulation.realizations[1].temperatures[0] = 1.1;
        let error = simulation
            .sample(&config, &AtomicBool::new(false), &|| {})
            .err()
            .unwrap();
        assert!(error.contains("ladders"));
        assert_eq!(simulation.realizations[0].spins, initial);
    }
}
