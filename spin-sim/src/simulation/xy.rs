//! XY simulation using the shared realization, scheduling, clusters and tempering.
use super::{driver, realization::XyRealization};
use crate::{
    clusters::fk::embedded_update,
    config::{
        AutocorrelationBackend, ClusterAction, ClusterConfig, ClusterMode, PtSchedule, SimConfig,
        SweepMode,
    },
    geometry::Lattice,
    spins::{
        model::Spin,
        xy::{local_sweep, LocalMove, XyEmbedding},
    },
    statistics::{
        autocorrelation::PrecisionAutocorr,
        physics::{PhysicsCollector, PhysicsOptions, PhysicsResult, PhysicsValues},
        sokal_tau,
    },
};
use std::sync::atomic::AtomicBool;
use validator::Validate;

#[derive(Debug)]
pub struct XyConfig {
    pub simulation: SimConfig,
    /// Fixed number of cluster updates at each scheduled cluster event.
    pub cluster_updates: usize,
    pub overrelaxation_sweeps: usize,
    pub physics: PhysicsOptions,
}
impl Default for XyConfig {
    fn default() -> Self {
        Self {
            simulation: SimConfig {
                n_sweeps: 1000,
                warmup_sweeps: 250,
                sweep_mode: SweepMode::Metropolis,
                cluster_update: Some(ClusterConfig {
                    interval: 1,
                    mode: ClusterMode::Sw,
                    action: ClusterAction::Update,
                    collect_stats: false,
                }),
                pt_interval: None,
                pt_schedule: PtSchedule::SingleRandomEdge,
                overlap_cluster: None,
                autocorrelation_max_lag: None,
                autocorrelation_backend: AutocorrelationBackend::Ring,
                sequential: false,
                equilibration_diagnostic: false,
            },
            cluster_updates: 1,
            overrelaxation_sweeps: 0,
            physics: PhysicsOptions::default(),
        }
    }
}
impl XyConfig {
    fn validate(&self) -> Result<(), String> {
        let cfg = &self.simulation;
        cfg.validate().map_err(|e| e.to_string())?;
        if cfg.n_sweeps <= cfg.warmup_sweeps {
            return Err("XY sampling requires at least one measurement sweep".into());
        }
        if cfg.overlap_cluster.is_some() || cfg.equilibration_diagnostic {
            return Err(
                "overlap updates and Ising equilibration diagnostics are not supported for XY"
                    .into(),
            );
        }
        if self.cluster_updates == 0 {
            return Err("cluster_updates must be positive".into());
        }
        if cfg
            .cluster_update
            .as_ref()
            .is_some_and(|c| c.action != ClusterAction::Update || c.collect_stats)
        {
            return Err("XY clusters support updates and visited-spin counts; Ising graph collectors are unsupported".into());
        }
        if cfg.sweep_mode == SweepMode::None && cfg.cluster_update.is_none() {
            return Err("XY sampling requires Metropolis, Gibbs or cluster updates".into());
        }
        Ok(())
    }
}

pub struct XyDisorderResult {
    pub physics: PhysicsResult,
    pub energy_tau: Vec<f64>,
    pub mags2_tau: Vec<f64>,
    /// Counts per temperature, summed over replicas, including warmup.
    pub cluster_updates: Vec<u64>,
    /// SW visits N spins; Wolff visits its seed cluster, including vacant spins.
    pub visited_spins: Vec<u64>,
}
pub struct XyResult {
    pub values: PhysicsValues,
    pub per_disorder: Vec<XyDisorderResult>,
}

/// A batch of XY disorder realizations on one periodic hypercubic lattice.
///
/// ```
/// use spin_sim::{XyConfig, XySimulation};
/// use std::sync::atomic::AtomicBool;
/// let mut xy = XySimulation::new(
///     vec![4, 4], vec![vec![1.0; 32]], &[0.8, 1.2], 1, None, 42,
/// ).unwrap();
/// let result = xy.sample(&XyConfig::default(), &AtomicBool::new(false), &|| {}).unwrap();
/// assert_eq!(result.values["energies"].len(), 2);
/// ```
pub struct XySimulation {
    pub lattice: Lattice,
    pub realizations: Vec<XyRealization>,
    pub occupation: Vec<Vec<bool>>,
    pub n_replicas: usize,
    pub n_temps: usize,
    constructor_seed: u64,
}
impl XySimulation {
    pub fn new(
        shape: Vec<usize>,
        couplings: Vec<Vec<f64>>,
        temperatures: &[f64],
        n_replicas: usize,
        occupation: Option<Vec<Vec<bool>>>,
        seed: u64,
    ) -> Result<Self, String> {
        if shape.is_empty() || shape.iter().any(|&l| l < 3) {
            return Err("XY hypercubic extents must be at least three".into());
        }
        let n = shape
            .iter()
            .try_fold(1usize, |n, &l| n.checked_mul(l))
            .ok_or("lattice size overflow")?;
        if n > u32::MAX as usize || n.checked_mul(shape.len()).is_none() {
            return Err("lattice is too large".into());
        }
        if n_replicas == 0
            || temperatures.is_empty()
            || temperatures.iter().any(|t| !t.is_finite() || *t <= 0.0)
        {
            return Err(
                "temperatures must be nonempty, positive and finite; n_replicas must be positive"
                    .into(),
            );
        }
        n_replicas
            .checked_mul(temperatures.len())
            .and_then(|systems| systems.checked_mul(n))
            .ok_or("spin count overflow")?;
        if couplings.is_empty()
            || couplings
                .iter()
                .any(|j| j.len() != n * shape.len() || j.iter().any(|x| !x.is_finite()))
        {
            return Err("couplings must have the lattice shape and be finite".into());
        }
        let occupation = occupation.unwrap_or_else(|| vec![vec![true; n]; couplings.len()]);
        if occupation.len() != couplings.len() || occupation.iter().any(|m| m.len() != n) {
            return Err("occupation must match the disorder and lattice dimensions".into());
        }
        let lattice = Lattice::new(shape);
        let realizations = couplings
            .into_iter()
            .enumerate()
            .map(|(d, mut j)| {
                for i in 0..n {
                    for axis in 0..lattice.n_dims {
                        if !occupation[d][i] || !occupation[d][lattice.neighbor_fwd(i, axis)] {
                            j[i * lattice.n_dims + axis] = 0.0;
                        }
                    }
                }
                XyRealization::new(
                    &lattice,
                    j,
                    temperatures,
                    n_replicas,
                    super::realization::realization_seed(seed, d),
                )
            })
            .collect();
        Ok(Self {
            lattice,
            realizations,
            occupation,
            n_replicas,
            n_temps: temperatures.len(),
            constructor_seed: seed,
        })
    }
    pub fn reset(&mut self, seed: Option<u64>) {
        for (d, real) in self.realizations.iter_mut().enumerate() {
            real.reset(
                &self.lattice,
                self.n_replicas,
                self.n_temps,
                super::realization::realization_seed(seed.unwrap_or(self.constructor_seed), d),
            );
        }
    }
    pub fn sample(
        &mut self,
        config: &XyConfig,
        interrupted: &AtomicBool,
        on_sweep: &(dyn Fn() + Sync),
    ) -> Result<XyResult, String> {
        config.validate()?;
        driver::validate_batch(
            &self.lattice,
            &self.realizations,
            self.n_replicas,
            self.n_temps,
            &config.simulation,
        )?;
        // Preflight every optional collector and mask before any realization mutates.
        PhysicsCollector::new(&self.lattice, self.n_temps, &config.physics, 2)?;
        if self.occupation.len() != self.realizations.len()
            || self
                .occupation
                .iter()
                .any(|m| m.len() != self.lattice.n_spins)
        {
            return Err("occupation dimensions changed".into());
        }
        for (real, mask) in self.realizations.iter().zip(&self.occupation) {
            if real.spins.iter().any(|s| {
                !s[0].is_finite() || !s[1].is_finite() || (s[0].hypot(s[1]) - 1.0).abs() > 1e-8
            }) {
                return Err("XY spins must be finite unit vectors".into());
            }
            for i in 0..self.lattice.n_spins {
                for d in 0..self.lattice.n_dims {
                    if (!mask[i] || !mask[self.lattice.neighbor_fwd(i, d)])
                        && real.couplings[i * self.lattice.n_dims + d] != 0.0
                    {
                        return Err("vacant sites must have zero incident couplings".into());
                    }
                }
            }
        }
        let per_disorder = driver::map_realizations(&mut self.realizations, |d, real| {
            run_xy(
                &self.lattice,
                real,
                &self.occupation[d],
                self.n_replicas,
                self.n_temps,
                config,
                interrupted,
                on_sweep,
            )
        })?;
        let values = PhysicsResult::aggregate(
            per_disorder.iter().map(|r| &r.physics),
            &self.lattice.shape,
            &self.realizations[0].temperatures[..self.n_temps],
            2,
        );
        Ok(XyResult {
            values,
            per_disorder,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn run_xy(
    lattice: &Lattice,
    real: &mut XyRealization,
    occupation: &[bool],
    n_replicas: usize,
    n_temps: usize,
    config: &XyConfig,
    interrupted: &AtomicBool,
    on_sweep: &(dyn Fn() + Sync),
) -> Result<XyDisorderResult, String> {
    let cfg = &config.simulation;
    let mut physics = PhysicsCollector::new(lattice, n_temps, &config.physics, 2)?;
    let n_measures = cfg.n_sweeps - cfg.warmup_sweeps;
    let lag = cfg
        .autocorrelation_max_lag
        .map(|k| k.min(n_measures / 4).max(1));
    let mut energy_ac = lag.map(|k| {
        PrecisionAutocorr::<f64>::with_backend(k, n_temps, cfg.autocorrelation_backend, n_measures)
    });
    let mut m2_ac = lag.map(|k| {
        PrecisionAutocorr::<f64>::with_backend(k, n_temps, cfg.autocorrelation_backend, n_measures)
    });
    let mut energy_buf = vec![0.0; n_temps];
    let mut m2_buf = vec![0.0; n_temps];
    let mut visits = vec![0; n_temps * n_replicas];
    let mut updates = vec![0; n_temps];
    let occupied = (!occupation.iter().all(|&o| o)).then_some(occupation);
    let local_move = match cfg.sweep_mode {
        SweepMode::Metropolis => Some(LocalMove::Metropolis),
        SweepMode::Gibbs => Some(LocalMove::HeatBath),
        SweepMode::None => None,
    };
    driver::run_sweeps(cfg, interrupted, on_sweep, |step| {
        let moves = local_move.into_iter().chain(std::iter::repeat_n(
            LocalMove::Overrelaxation,
            config.overrelaxation_sweeps,
        ));
        for kind in moves {
            local_sweep(
                lattice,
                &mut real.spins,
                &real.couplings,
                &real.temperatures,
                &real.system_ids,
                &mut real.rngs,
                occupied,
                cfg.sequential,
                kind,
            );
        }
        if step.cluster {
            let cluster = cfg.cluster_update.as_ref().unwrap();
            for _ in 0..config.cluster_updates {
                embedded_update::<XyEmbedding>(
                    lattice,
                    &mut real.spins,
                    &real.couplings,
                    &real.temperatures,
                    &real.system_ids,
                    &mut real.rngs,
                    cluster.mode == ClusterMode::Wolff,
                    ClusterAction::Update,
                    None,
                    None,
                    cfg.sequential,
                    Some(&mut visits),
                );
                for count in &mut updates {
                    *count += n_replicas as u64;
                }
            }
        }
        if step.record {
            energy_buf.fill(0.0);
            m2_buf.fill(0.0);
            for (slot, &id) in real.system_ids.iter().enumerate() {
                let t = slot % n_temps;
                let measured = physics.measure(
                    lattice,
                    &real.spins[id * lattice.n_spins..(id + 1) * lattice.n_spins],
                    &real.couplings,
                    occupation,
                    t,
                );
                real.energies[id] = -measured[0];
                energy_buf[t] += measured[0] / n_replicas as f64;
                m2_buf[t] += measured[1] / n_replicas as f64;
            }
            physics.end_sweep(n_replicas);
            if let Some(ac) = &mut energy_ac {
                ac.push(&energy_buf);
            }
            if let Some(ac) = &mut m2_ac {
                ac.push(&m2_buf);
            }
        } else if step.temper {
            <[f64; 2]>::interactions(lattice, &real.spins, &real.couplings, &mut real.energies);
        }
        if step.temper {
            real.temper(lattice.n_spins, n_replicas, n_temps, cfg.pt_schedule);
        }
    })?;
    let tau = |ac: Option<PrecisionAutocorr<f64>>| {
        ac.map(|a| a.finish().iter().map(|g| sokal_tau(g)).collect())
            .unwrap_or_default()
    };
    let mut visited_spins = vec![0; n_temps];
    for (slot, visited) in visits.into_iter().enumerate() {
        visited_spins[slot % n_temps] += visited;
    }
    Ok(XyDisorderResult {
        physics: physics.finish(),
        energy_tau: tau(energy_ac),
        mags2_tau: tau(m2_ac),
        cluster_updates: updates,
        visited_spins,
    })
}
