//! XY simulation using the shared realization, scheduling, clusters and tempering.
use super::{driver, realization::XyRealization};
use crate::{
    clusters::fk::embedded_update,
    config::{
        AutocorrelationBackend, ClusterAction, ClusterConfig, ClusterMode, PtSchedule, SimConfig,
        SweepMode,
    },
    geometry::Lattice,
    spins::xy::{interaction, local_sweep, LocalMove, XyEmbedding},
    statistics::{
        autocorrelation::PrecisionAutocorr,
        physics::{MomentBlock, PhysicsCollector, PhysicsOptions, PhysicsResult, PhysicsValues},
    },
};
use rayon::prelude::*;
use std::sync::atomic::AtomicBool;

#[derive(Debug)]
pub struct XyConfig {
    pub simulation: SimConfig,
    /// Fixed number of cluster updates at each scheduled cluster event.
    pub cluster_updates: usize,
    pub overrelaxation_sweeps: usize,
    pub physics: PhysicsOptions,
    /// Measure every this many sweeps after warmup. Near a transition consecutive
    /// sweeps are strongly correlated, so a stride saves measurement work at little
    /// statistical cost; block sizes and autocorrelation times are then in units of
    /// measurements.
    pub measure_interval: usize,
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
                conditional_link_overlap: true,
            },
            cluster_updates: 1,
            overrelaxation_sweeps: 0,
            physics: PhysicsOptions::default(),
            measure_interval: 1,
        }
    }
}
impl XyConfig {
    fn validate(&self) -> Result<(), String> {
        let cfg = &self.simulation;
        cfg.validate()?;
        if cfg.overlap_cluster.is_some() || cfg.equilibration_diagnostic {
            return Err(
                "overlap updates and Ising equilibration diagnostics are not supported for XY"
                    .into(),
            );
        }
        if self.cluster_updates == 0 {
            return Err("cluster_updates must be positive".into());
        }
        if self.measure_interval == 0 {
            return Err("measure_interval must be positive".into());
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
    /// τ_int per temperature in measured sweeps; NaN for a series without variance.
    pub energy_tau: Vec<f64>,
    pub mags2_tau: Vec<f64>,
    /// Whether each τ's Sokal window closed; if not, τ is a truncated lower bound.
    pub energy_tau_converged: Vec<bool>,
    pub mags2_tau_converged: Vec<bool>,
    /// Counts per temperature, summed over replicas, including warmup.
    pub cluster_updates: Vec<u64>,
    /// SW visits N spins; Wolff visits its seed cluster, including vacant spins.
    pub visited_spins: Vec<u64>,
    /// Accepted Metropolis proposals per temperature over the measured sweeps, summed
    /// over replicas, and the proposals made: one per occupied site and sweep. The
    /// ratio shows where uniform proposals freeze and the heat bath pays off.
    pub metropolis_accepted: Vec<u64>,
    pub metropolis_attempts: Vec<u64>,
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
        // Preflight the physics options and masks before any realization mutates.
        config.physics.validate(&self.lattice, 2)?;
        if self.occupation.len() != self.realizations.len()
            || self
                .occupation
                .iter()
                .any(|m| m.len() != self.lattice.n_spins)
        {
            return Err("occupation dimensions changed".into());
        }
        for (real, mask) in self.realizations.iter().zip(&self.occupation) {
            // |r² - 1| <= 2e-8 bounds |r - 1| by 1e-8; NaN components fail as well.
            if real.spins.iter().any(|s| {
                let excess = (s[0] * s[0] + s[1] * s[1] - 1.0).abs();
                excess.is_nan() || excess > 2e-8
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
        let sequential = driver::effective_sequential(&config.simulation, self.realizations.len());
        // Disorder batches already measure in parallel; split temperatures only for idle threads.
        let measurement_groups = if sequential {
            1
        } else {
            (rayon::current_num_threads() / self.realizations.len()).max(1)
        };
        let per_disorder = driver::map_realizations(&mut self.realizations, |d, real| {
            run_xy(
                &self.lattice,
                real,
                &self.occupation[d],
                self.n_replicas,
                self.n_temps,
                config,
                sequential,
                measurement_groups,
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

/// Physics collectors, each owning a fixed residue class of temperatures.
///
/// A temperature's moments are accumulated by one collector in replica order,
/// exactly as with a single collector, so results do not depend on the grouping.
struct Measurement {
    groups: Vec<PhysicsCollector>,
}
impl Measurement {
    fn new(
        lattice: &Lattice,
        n_temps: usize,
        options: &PhysicsOptions,
        max_groups: usize,
    ) -> Result<Self, String> {
        let n_groups = n_temps.min(max_groups).max(1);
        let groups = (0..n_groups)
            .map(|group| {
                let mut collector = PhysicsCollector::new(lattice, n_temps, options, 2)?;
                if n_groups > 1 {
                    collector.own_block_temperatures(|t| t % n_groups == group);
                }
                Ok(collector)
            })
            .collect::<Result<_, String>>()?;
        Ok(Self { groups })
    }

    /// Measure every system; returns per-slot physical energy/site and m².
    fn measure(
        &mut self,
        lattice: &Lattice,
        real: &XyRealization,
        occupation: &[bool],
        n_temps: usize,
        out: &mut [[f64; 2]],
    ) {
        let n_groups = self.groups.len();
        let n = lattice.n_spins;
        let measure_group = |(group, collector): (usize, &mut PhysicsCollector)| {
            let mut measured = Vec::new();
            for t in (group..n_temps).step_by(n_groups) {
                for slot in (t..real.system_ids.len()).step_by(n_temps) {
                    let id = real.system_ids[slot];
                    let spins = &real.spins[id * n..(id + 1) * n];
                    let values = collector.measure(lattice, spins, &real.couplings, occupation, t);
                    measured.push((slot, values));
                }
            }
            measured
        };
        let measured: Vec<_> = if n_groups == 1 {
            vec![measure_group((0, &mut self.groups[0]))]
        } else {
            self.groups
                .par_iter_mut()
                .enumerate()
                .map(measure_group)
                .collect()
        };
        for (slot, values) in measured.into_iter().flatten() {
            out[slot] = values;
        }
    }

    fn end_sweep(&mut self, n_replicas: usize) {
        for collector in &mut self.groups {
            collector.end_sweep(n_replicas);
        }
    }

    fn finish(self, n_temps: usize) -> PhysicsResult {
        let mut groups: Vec<_> = self.groups.into_iter().map(|c| c.finish()).collect();
        if groups.len() == 1 {
            return groups.pop().unwrap();
        }
        let n_groups = groups.len();
        let owner = |t: usize| &groups[t % n_groups];
        let blocks = (0..groups[0].blocks.len())
            .map(|b| MomentBlock {
                sums: (0..n_temps)
                    .map(|t| owner(t).blocks[b].sums[t].clone())
                    .collect(),
                count: groups[0].blocks[b].count,
                sweeps: groups[0].blocks[b].sweeps,
            })
            .collect();
        PhysicsResult {
            fields: groups[0].fields.clone(),
            moments: (0..n_temps).map(|t| owner(t).moments[t].clone()).collect(),
            count: groups[0].count,
            blocks,
        }
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
    sequential: bool,
    measurement_groups: usize,
    interrupted: &AtomicBool,
    on_sweep: &(dyn Fn() + Sync),
) -> Result<XyDisorderResult, String> {
    let cfg = &config.simulation;
    let mut physics = Measurement::new(lattice, n_temps, &config.physics, measurement_groups)?;
    let n_measures = cfg.n_measurement_sweeps().div_ceil(config.measure_interval);
    let lag = cfg
        .autocorrelation_max_lag
        .map(|lag| lag.min(n_measures / 4).max(1));
    let mut energy_ac = lag.map(|k| {
        PrecisionAutocorr::<f64>::with_backend(k, n_temps, cfg.autocorrelation_backend, n_measures)
    });
    let mut m2_ac = lag.map(|k| {
        PrecisionAutocorr::<f64>::with_backend(k, n_temps, cfg.autocorrelation_backend, n_measures)
    });
    let mut energy_buf = vec![0.0; n_temps];
    let mut m2_buf = vec![0.0; n_temps];
    let mut measured = vec![[0.0; 2]; real.system_ids.len()];
    let mut visits = vec![0; n_temps * n_replicas];
    let mut updates = vec![0; n_temps];
    let mut metropolis_accepted = vec![0u64; n_temps * n_replicas];
    let mut metropolis_passes = 0u64;
    let occupied = (!occupation.iter().all(|&o| o)).then_some(occupation);
    let local_move = match cfg.sweep_mode {
        SweepMode::Metropolis => Some(LocalMove::Metropolis),
        SweepMode::Gibbs => Some(LocalMove::HeatBath),
        SweepMode::None => None,
    };
    driver::run_sweeps(cfg, interrupted, on_sweep, real.schedule.sweeps, |step| {
        let moves = local_move.into_iter().chain(std::iter::repeat_n(
            LocalMove::Overrelaxation,
            config.overrelaxation_sweeps,
        ));
        for kind in moves {
            let count = step.record && kind == LocalMove::Metropolis;
            metropolis_passes += u64::from(count);
            local_sweep(
                lattice,
                &mut real.spins,
                &real.couplings,
                &real.temperatures,
                &real.system_ids,
                &mut real.rngs,
                occupied,
                sequential,
                kind,
                count.then_some(metropolis_accepted.as_mut_slice()),
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
                    sequential,
                    Some(&mut visits),
                );
                for count in &mut updates {
                    *count += n_replicas as u64;
                }
            }
        }
        let measure =
            step.record && (step.index - cfg.warmup_sweeps).is_multiple_of(config.measure_interval);
        if measure {
            physics.measure(lattice, real, occupation, n_temps, &mut measured);
            energy_buf.fill(0.0);
            m2_buf.fill(0.0);
            for (slot, &id) in real.system_ids.iter().enumerate() {
                let t = slot % n_temps;
                real.energies[id] = -measured[slot][0];
                energy_buf[t] += measured[slot][0] / n_replicas as f64;
                m2_buf[t] += measured[slot][1] / n_replicas as f64;
            }
            physics.end_sweep(n_replicas);
            if let Some(ac) = &mut energy_ac {
                ac.push(&energy_buf);
            }
            if let Some(ac) = &mut m2_ac {
                ac.push(&m2_buf);
            }
        } else if step.temper {
            let n = lattice.n_spins;
            let refresh = |(energy, spins): (&mut f64, &[[f64; 2]])| {
                *energy = interaction(lattice, spins, &real.couplings) / n as f64;
            };
            if sequential {
                real.energies
                    .iter_mut()
                    .zip(real.spins.chunks_exact(n))
                    .for_each(refresh);
            } else {
                real.energies
                    .par_iter_mut()
                    .zip(real.spins.par_chunks_exact(n))
                    .for_each(refresh);
            }
        }
        if step.temper {
            real.temper(lattice.n_spins, n_replicas, n_temps, cfg.pt_schedule);
        }
        real.schedule.sweeps += 1;
    })?;
    let (energy_tau, energy_tau_converged) = driver::tau_estimates(energy_ac.as_ref());
    let (mags2_tau, mags2_tau_converged) = driver::tau_estimates(m2_ac.as_ref());
    let per_temperature = |slots: Vec<u64>| {
        let mut totals = vec![0; n_temps];
        for (slot, count) in slots.into_iter().enumerate() {
            totals[slot % n_temps] += count;
        }
        totals
    };
    let n_occupied = occupation.iter().filter(|&&o| o).count() as u64;
    Ok(XyDisorderResult {
        physics: physics.finish(n_temps),
        energy_tau,
        mags2_tau,
        energy_tau_converged,
        mags2_tau_converged,
        cluster_updates: updates,
        visited_spins: per_temperature(visits),
        metropolis_accepted: per_temperature(metropolis_accepted),
        metropolis_attempts: vec![metropolis_passes * n_replicas as u64 * n_occupied; n_temps],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_with_threads(sequential: bool, threads: usize) -> (XyResult, Vec<u64>) {
        let couplings = (0..40).map(|i| ((i * 7) % 11) as f64 / 5.0 - 0.6).collect();
        let occupation = (0..20).map(|i| i != 6).collect();
        let temperatures = [0.5, 0.8, 1.1, 1.6, 2.4];
        let mut simulation = XySimulation::new(
            vec![5, 4],
            vec![couplings],
            &temperatures,
            2,
            Some(vec![occupation]),
            21,
        )
        .unwrap();
        let mut config = XyConfig::default();
        let cfg = &mut config.simulation;
        (cfg.n_sweeps, cfg.warmup_sweeps, cfg.sequential) = (64, 21, sequential);
        (cfg.pt_interval, cfg.autocorrelation_max_lag) = (Some(1), Some(4));
        config.overrelaxation_sweeps = 1;
        config.physics = PhysicsOptions {
            displacements: vec![vec![1, 1]],
            vortices: true,
            block_size: Some(7),
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let result = pool
            .install(|| simulation.sample(&config, &AtomicBool::new(false), &|| {}))
            .unwrap();
        let spins = simulation.realizations[0]
            .spins
            .iter()
            .flatten()
            .map(|v| v.to_bits())
            .collect();
        (result, spins)
    }

    /// A stride measures every k-th sweep after warmup; Metropolis counts cover the
    /// measured passes over every occupied site.
    #[test]
    fn measurement_stride_and_metropolis_counts() {
        let mut xy =
            XySimulation::new(vec![4, 4], vec![vec![1.0; 32]], &[0.4, 2.0], 2, None, 7).unwrap();
        let mut config = XyConfig::default();
        config.simulation.n_sweeps = 25;
        config.simulation.warmup_sweeps = 5;
        config.simulation.cluster_update = None;
        config.measure_interval = 3;
        let result = xy.sample(&config, &AtomicBool::new(false), &|| {}).unwrap();
        let disorder = &result.per_disorder[0];
        // Measured sweeps 5, 8, ..., 23: seven, each over two replicas.
        assert_eq!(disorder.physics.count, 14);
        assert_eq!(disorder.metropolis_attempts, vec![20 * 2 * 16; 2]);
        let rate: Vec<f64> = disorder
            .metropolis_accepted
            .iter()
            .zip(&disorder.metropolis_attempts)
            .map(|(&accepted, &attempts)| accepted as f64 / attempts as f64)
            .collect();
        assert!(
            rate.iter().all(|&rate| rate > 0.0 && rate <= 1.0),
            "{rate:?}"
        );
        assert!(
            rate[0] < rate[1],
            "colder temperatures accept fewer proposals: {rate:?}"
        );

        config.measure_interval = 0;
        assert!(xy.sample(&config, &AtomicBool::new(false), &|| {}).is_err());
    }

    #[test]
    fn parallel_measurement_matches_sequential_for_any_thread_count() {
        let bits = |rows: &[Vec<f64>]| -> Vec<Vec<u64>> {
            rows.iter()
                .map(|row| row.iter().map(|v| v.to_bits()).collect())
                .collect()
        };
        let (reference, reference_spins) = sample_with_threads(true, 1);
        let expected = &reference.per_disorder[0];
        // Two, three and five temperature groups.
        for threads in [2, 3, 8] {
            let (result, spins) = sample_with_threads(false, threads);
            assert_eq!(spins, reference_spins);
            for (name, rows) in &reference.values {
                assert_eq!(bits(rows), bits(&result.values[name]), "{name}");
            }
            let actual = &result.per_disorder[0];
            assert_eq!(
                bits(&actual.physics.moments),
                bits(&expected.physics.moments)
            );
            assert_eq!(actual.physics.blocks.len(), expected.physics.blocks.len());
            for (a, b) in actual.physics.blocks.iter().zip(&expected.physics.blocks) {
                assert_eq!((a.count, a.sweeps), (b.count, b.sweeps));
                assert_eq!(bits(&a.sums), bits(&b.sums));
            }
            assert_eq!(
                bits(&[actual.energy_tau.clone(), actual.mags2_tau.clone()]),
                bits(&[expected.energy_tau.clone(), expected.mags2_tau.clone()])
            );
            assert_eq!(actual.visited_spins, expected.visited_spins);
            assert_eq!(actual.metropolis_accepted, expected.metropolis_accepted);
        }
    }
}
