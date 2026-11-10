//! Shared scheduling and ordered disorder dispatch. Model hooks are monomorphized.
use super::realization::ModelRealization;
use crate::{
    config::{OverlapClusterBuildMode, SimConfig},
    geometry::Lattice,
    spins::model::{Real, Spin},
    statistics::autocorrelation::PrecisionAutocorr,
};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) struct SweepStep {
    /// Sweep within this call.
    pub index: usize,
    /// Sweep counted from the realization's creation or last reset.
    pub clock: usize,
    pub record: bool,
    pub cluster: bool,
    pub overlap: bool,
    pub temper: bool,
}

/// Runs `config.n_sweeps` sweeps. `start` is the number of sweeps the realization has
/// already done, so interval events keep their phase across calls; warmup counts from
/// the start of this call.
pub(crate) fn run_sweeps(
    config: &SimConfig,
    interrupted: &AtomicBool,
    on_sweep: &(dyn Fn() + Sync),
    start: usize,
    mut step: impl FnMut(SweepStep),
) -> Result<(), String> {
    let due = |interval: usize, clock: usize| clock.is_multiple_of(interval);
    for index in 0..config.n_sweeps {
        if interrupted.load(Ordering::Relaxed) {
            return Err("interrupted".into());
        }
        on_sweep();
        let clock = start + index;
        step(SweepStep {
            index,
            clock,
            record: index >= config.warmup_sweeps,
            cluster: config
                .cluster_update
                .as_ref()
                .is_some_and(|c| due(c.interval, clock)),
            overlap: config
                .overlap_cluster
                .as_ref()
                .is_some_and(|o| due(o.interval, clock)),
            temper: config
                .pt_interval
                .is_some_and(|interval| due(interval, clock)),
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

/// τ_int per temperature with its convergence flags; empty when autocorrelation is off.
pub(crate) fn tau_estimates<T: Real>(
    accum: Option<&PrecisionAutocorr<T>>,
) -> (Vec<f64>, Vec<bool>) {
    accum.map_or_else(Default::default, |accum| {
        accum
            .estimates()
            .iter()
            .map(|estimate| (estimate.tau, estimate.converged))
            .unzip()
    })
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
        let result = run_sweeps(&config, &interrupted, &|| {}, 0, |_| {
            interrupted.store(true, Ordering::Relaxed)
        });
        assert_eq!(result.unwrap_err(), "interrupted");
    }
    #[test]
    fn interval_phase_continues_from_the_starting_sweep() {
        let mut config = XyConfig::default().simulation;
        config.n_sweeps = 3;
        config.warmup_sweeps = 1;
        config.pt_interval = Some(4);
        let mut steps = Vec::new();
        run_sweeps(&config, &AtomicBool::new(false), &|| {}, 6, |step| {
            steps.push((step.index, step.record, step.temper));
        })
        .unwrap();
        // Sweeps 6, 7 and 8 of the realization: tempering is due at 8.
        assert_eq!(
            steps,
            vec![(0, false, false), (1, true, false), (2, true, true)]
        );
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
