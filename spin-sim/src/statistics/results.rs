use super::equilibration::EquilCheckpoint;
use super::overlap::OverlapStats;

pub struct ClusterSnapshot {
    pub sweep_id: usize,
    pub mode_idx: usize,
    pub cluster_ids: Vec<Vec<u32>>,
    pub blue_ids: Option<Vec<Vec<u32>>>,
    pub spins: Vec<[Vec<i8>; 2]>,
    pub system_ids: Vec<[usize; 2]>,
}

pub struct ClusterStats {
    /// FK cluster size histogram per temperature: `hist[s]` = count of size-`s` clusters.
    pub fk_csd: Vec<Vec<u64>>,
    /// Per-mode overlap cluster size histogram: `[n_modes][n_temps][n_spins+1]`.
    /// Counts clusters of the sites a move acts on; the sites it leaves alone are
    /// not size-1 clusters. CMR acts on every site.
    pub overlap_csd: Vec<Vec<Vec<u64>>>,
    /// Per-mode average relative size of k-th largest overlap cluster: `[n_modes][n_temps][4]`.
    /// Zero for a mode that never ran during measurement, and where fewer than `k`
    /// clusters exist.
    pub top_cluster_sizes: Vec<Vec<[f64; 4]>>,
    /// Recorded FK cluster updates per temperature, summed over replicas. Empty
    /// unless FK clusters update spins.
    pub fk_updates: Vec<u64>,
    /// Spins those updates visited: the flipped cluster for Wolff, every spin for
    /// Swendsen-Wang. For a ferromagnet the mean Wolff cluster size
    /// `fk_visited_spins / fk_updates` estimates `N ⟨m²⟩`, and for any model it
    /// converts autocorrelation times from cluster updates to sweeps.
    pub fk_visited_spins: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct GraphObservationSummary {
    pub observation_count: Vec<u64>,
    pub cluster_size_counts: Vec<Vec<u64>>,
    pub top_four_component_fractions: Vec<[f64; 4]>,
    pub active_bond_density: Vec<f64>,
    pub large_component_count: Vec<f64>,
    pub winding: Option<Vec<[f64; 4]>>,
}

#[derive(Clone, Debug, Default)]
pub struct ClusterObservations {
    pub fk: Option<GraphObservationSummary>,
    pub houdayer: Option<GraphObservationSummary>,
    pub jorg: Option<GraphObservationSummary>,
    pub cmr_blue: Option<GraphObservationSummary>,
}

pub struct Diagnostics {
    /// Integrated autocorrelation time τ_int(m²) per temperature, in measured sweeps.
    /// Empty if autocorrelation_max_lag is None. A series without variance (frozen,
    /// or never measured) has no autocorrelation time and reports NaN; after
    /// aggregation this is the mean over the disorder samples with a defined τ.
    pub mags2_tau: Vec<f64>,
    /// τ_int(q²) per temperature, with q² averaged over all replica pairs.
    /// Empty if autocorrelation_max_lag is None or n_replicas < 2.
    pub overlap2_tau: Vec<f64>,
    /// τ_int of the replica-averaged energy per temperature.
    /// Empty if autocorrelation_max_lag is None.
    pub energy_tau: Vec<f64>,
    /// Whether the Sokal window of each τ closed within the estimated lags. A τ
    /// whose window did not close is truncated and is a lower bound. After
    /// aggregation: whether it closed in every disorder sample.
    pub mags2_tau_converged: Vec<bool>,
    pub overlap2_tau_converged: Vec<bool>,
    pub energy_tau_converged: Vec<bool>,
    /// Largest lag that was estimated: the requested lag capped at a quarter of the
    /// measured sweeps. `None` if autocorrelation_max_lag is None.
    pub autocorrelation_lag: Option<usize>,
    /// Equilibration diagnostic checkpoints: energy and link overlap averaged over
    /// the log-binned window of each checkpoint (see [`EquilCheckpoint`]).
    /// Empty if equilibration_diagnostic is false.
    pub equil_checkpoints: Vec<EquilCheckpoint>,
}

impl Diagnostics {
    fn empty() -> Self {
        Self {
            mags2_tau: vec![],
            overlap2_tau: vec![],
            energy_tau: vec![],
            mags2_tau_converged: vec![],
            overlap2_tau_converged: vec![],
            energy_tau_converged: vec![],
            autocorrelation_lag: None,
            equil_checkpoints: vec![],
        }
    }
}

/// Thermal averages of one disorder realization, per temperature.
///
/// The disorder mean alone carries no error bar; these per-sample values give the
/// standard error of any disorder average and feed jackknife or bootstrap errors of
/// ratios such as the Binder parameter. Overlap vectors are empty when
/// `n_replicas < 2`, and τ vectors when autocorrelation is off.
#[derive(Clone, Debug, Default)]
pub struct DisorderSample {
    pub mags2: Vec<f64>,
    pub mags4: Vec<f64>,
    pub energies: Vec<f64>,
    pub energy_variance: Vec<f64>,
    pub overlap2: Vec<f64>,
    pub overlap4: Vec<f64>,
    pub link_overlap: Vec<f64>,
    pub link_overlap2: Vec<f64>,
    pub link_overlap4: Vec<f64>,
    pub mags2_tau: Vec<f64>,
    pub overlap2_tau: Vec<f64>,
    pub energy_tau: Vec<f64>,
    pub mags2_tau_converged: Vec<bool>,
    pub overlap2_tau_converged: Vec<bool>,
    pub energy_tau_converged: Vec<bool>,
}

/// Per-temperature observables averaged over measurement sweeps and replicas.
///
/// All vectors are indexed by temperature index and have length `n_temps`.
/// Overlap vectors are empty when `n_replicas < 2`. After
/// [`SweepResult::aggregate`] every moment is additionally averaged over disorder.
pub struct SweepResult {
    /// ⟨m⟩ — mean magnetization per spin.
    pub mags: Vec<f64>,
    /// ⟨m²⟩.
    pub mags2: Vec<f64>,
    /// ⟨m⁴⟩.
    pub mags4: Vec<f64>,
    /// ⟨e⟩ with `e = -H/N = Σ_⟨ij⟩ J_ij s_i s_j / N`, the interaction sum per spin.
    /// This is minus the physical energy per spin; the opt-in physics collector
    /// ([`super::physics::PhysicsResult`]) reports `+H/N` instead.
    pub energies: Vec<f64>,
    /// ⟨e²⟩.
    pub energies2: Vec<f64>,
    /// Thermal variance `⟨e²⟩ - ⟨e⟩²` of one disorder sample; after aggregation, its
    /// disorder average. The heat capacity per spin is `N · energy_variance / T²`.
    /// Unlike `energies2 - energies²` after aggregation, it excludes the
    /// disorder variance of `⟨e⟩`.
    pub energy_variance: Vec<f64>,
    pub overlap_stats: OverlapStats,
    pub cluster_stats: ClusterStats,
    pub per_disorder_physics: Vec<super::physics::PhysicsResult>,
    pub per_disorder_cluster_observations: Vec<ClusterObservations>,
    /// One entry per disorder realization, in order.
    pub per_disorder: Vec<DisorderSample>,
    pub diagnostics: Diagnostics,
    pub cluster_snapshots: Vec<ClusterSnapshot>,
}

/// Element-wise mean over `results` of the vector that `field` selects, with the
/// length of the first.
fn mean_over<'a, R>(results: &'a [R], field: impl Fn(&'a R) -> &'a [f64]) -> Vec<f64> {
    let mut mean = vec![0.0; results.first().map_or(0, |first| field(first).len())];
    for result in results {
        for (total, &value) in mean.iter_mut().zip(field(result)) {
            *total += value;
        }
    }
    let n = results.len() as f64;
    for value in &mut mean {
        *value /= n;
    }
    mean
}

/// [`mean_over`] that skips NaN entries; NaN where every entry is NaN.
fn defined_mean_over<'a, R>(results: &'a [R], field: impl Fn(&'a R) -> &'a [f64]) -> Vec<f64> {
    let len = results.first().map_or(0, |first| field(first).len());
    (0..len)
        .map(|index| {
            let defined = results
                .iter()
                .filter_map(|result| field(result).get(index).copied())
                .filter(|value| !value.is_nan());
            let (sum, count) = defined.fold((0.0, 0usize), |(sum, count), value| {
                (sum + value, count + 1)
            });
            sum / count as f64
        })
        .collect()
}

/// Element-wise sum over `results` of the counters that `field` selects.
fn sum_over<'a, R>(results: &'a [R], field: impl Fn(&'a R) -> &'a [u64]) -> Vec<u64> {
    let mut total = vec![0; results.first().map_or(0, |first| field(first).len())];
    for result in results {
        for (total, &count) in total.iter_mut().zip(field(result)) {
            *total += count;
        }
    }
    total
}

/// Element-wise conjunction over `results`.
fn all_over<'a, R>(results: &'a [R], field: impl Fn(&'a R) -> &'a [bool]) -> Vec<bool> {
    let len = results.first().map_or(0, |first| field(first).len());
    (0..len)
        .map(|index| results.iter().all(|result| field(result)[index]))
        .collect()
}

impl SweepResult {
    /// A result with no temperatures, returned when aggregating nothing.
    pub fn empty() -> Self {
        Self {
            mags: vec![],
            mags2: vec![],
            mags4: vec![],
            energies: vec![],
            energies2: vec![],
            energy_variance: vec![],
            overlap_stats: OverlapStats::empty(),
            cluster_stats: ClusterStats {
                fk_csd: vec![],
                overlap_csd: vec![],
                top_cluster_sizes: vec![],
                fk_updates: vec![],
                fk_visited_spins: vec![],
            },
            per_disorder_physics: vec![],
            per_disorder_cluster_observations: vec![],
            per_disorder: vec![],
            diagnostics: Diagnostics::empty(),
            cluster_snapshots: vec![],
        }
    }

    /// Average [`SweepResult`]s across disorder realizations with equal weights.
    ///
    /// Thermal variances (`energy_variance`) are averaged per sample, so they do not
    /// pick up disorder fluctuations. An empty slice yields [`SweepResult::empty`].
    pub fn aggregate(results: &[Self]) -> Self {
        Self::aggregate_impl(results, true)
    }

    pub(crate) fn aggregate_without_overlap_samples(results: &[Self]) -> Self {
        Self::aggregate_impl(results, false)
    }

    fn aggregate_impl(results: &[Self], retain_overlap_samples: bool) -> Self {
        if results.is_empty() {
            return Self::empty();
        }
        let n = results.len() as f64;
        let n_temps = results[0].mags.len();
        let first = &results[0].cluster_stats;

        let mut fk_csd = first.fk_csd.clone();
        for histogram in &mut fk_csd {
            histogram.fill(0);
        }
        let mut overlap_csd = first.overlap_csd.clone();
        for histogram in overlap_csd.iter_mut().flatten() {
            histogram.fill(0);
        }
        // A mode that never measured in one sample reports no temperatures there.
        let mut top_cluster_sizes: Vec<Vec<[f64; 4]>> = (0..first.top_cluster_sizes.len())
            .map(|mode| {
                let n_top_temps = results
                    .iter()
                    .filter_map(|r| r.cluster_stats.top_cluster_sizes.get(mode))
                    .map(Vec::len)
                    .max()
                    .unwrap_or(0);
                vec![[0.0; 4]; n_top_temps]
            })
            .collect();
        for stats in results.iter().map(|r| &r.cluster_stats) {
            for (total, part) in fk_csd.iter_mut().zip(&stats.fk_csd) {
                for (total, &count) in total.iter_mut().zip(part) {
                    *total += count;
                }
            }
            for (total, part) in overlap_csd
                .iter_mut()
                .flatten()
                .zip(stats.overlap_csd.iter().flatten())
            {
                for (total, &count) in total.iter_mut().zip(part) {
                    *total += count;
                }
            }
            for (total, part) in top_cluster_sizes.iter_mut().zip(&stats.top_cluster_sizes) {
                for (total, sizes) in total.iter_mut().zip(part) {
                    for k in 0..4 {
                        total[k] += sizes[k];
                    }
                }
            }
        }
        for size in top_cluster_sizes.iter_mut().flatten().flatten() {
            *size /= n;
        }

        let overlap_results: Vec<_> = results.iter().map(|r| &r.overlap_stats).collect();
        let overlap_stats = if retain_overlap_samples {
            OverlapStats::aggregate(&overlap_results)
        } else {
            OverlapStats::aggregate_without_samples(&overlap_results)
        };

        let checkpoints = &results[0].diagnostics.equil_checkpoints;
        let equil_checkpoints = (0..checkpoints.len())
            .map(|i| EquilCheckpoint {
                sweep: checkpoints[i].sweep,
                energy_avg: mean_over(results, |r| &r.diagnostics.equil_checkpoints[i].energy_avg),
                link_overlap_avg: mean_over(results, |r| {
                    &r.diagnostics.equil_checkpoints[i].link_overlap_avg
                }),
                overlap2_avg: mean_over(results, |r| {
                    &r.diagnostics.equil_checkpoints[i].overlap2_avg
                }),
            })
            .collect();
        debug_assert!(results.iter().all(|r| r.mags.len() == n_temps));

        SweepResult {
            mags: mean_over(results, |r| &r.mags),
            mags2: mean_over(results, |r| &r.mags2),
            mags4: mean_over(results, |r| &r.mags4),
            energies: mean_over(results, |r| &r.energies),
            energies2: mean_over(results, |r| &r.energies2),
            energy_variance: mean_over(results, |r| &r.energy_variance),
            overlap_stats,
            cluster_stats: ClusterStats {
                fk_csd,
                overlap_csd,
                top_cluster_sizes,
                fk_updates: sum_over(results, |r| &r.cluster_stats.fk_updates),
                fk_visited_spins: sum_over(results, |r| &r.cluster_stats.fk_visited_spins),
            },
            per_disorder_cluster_observations: results
                .iter()
                .flat_map(|r| r.per_disorder_cluster_observations.iter().cloned())
                .collect(),
            per_disorder_physics: results
                .iter()
                .flat_map(|r| r.per_disorder_physics.iter().cloned())
                .collect(),
            per_disorder: results
                .iter()
                .flat_map(|r| r.per_disorder.iter().cloned())
                .collect(),
            diagnostics: Diagnostics {
                mags2_tau: defined_mean_over(results, |r| &r.diagnostics.mags2_tau),
                overlap2_tau: defined_mean_over(results, |r| &r.diagnostics.overlap2_tau),
                energy_tau: defined_mean_over(results, |r| &r.diagnostics.energy_tau),
                mags2_tau_converged: all_over(results, |r| &r.diagnostics.mags2_tau_converged),
                overlap2_tau_converged: all_over(results, |r| {
                    &r.diagnostics.overlap2_tau_converged
                }),
                energy_tau_converged: all_over(results, |r| &r.diagnostics.energy_tau_converged),
                autocorrelation_lag: results[0].diagnostics.autocorrelation_lag,
                equil_checkpoints,
            },
            cluster_snapshots: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A per-sample result whose energies have the given thermal mean and variance.
    fn sample(mean: f64, variance: f64, top: Vec<Vec<[f64; 4]>>) -> SweepResult {
        let mut result = SweepResult::empty();
        result.mags = vec![0.0];
        result.mags2 = vec![0.0];
        result.mags4 = vec![0.0];
        result.energies = vec![mean];
        result.energies2 = vec![variance + mean * mean];
        result.energy_variance = vec![variance];
        result.cluster_stats.top_cluster_sizes = top;
        result
    }

    #[test]
    fn aggregate_averages_per_sample_thermal_variance() {
        let agg = SweepResult::aggregate(&[sample(-1.0, 0.25, vec![]), sample(-2.0, 0.75, vec![])]);
        assert_eq!(agg.energies, vec![-1.5]);
        assert_eq!(agg.energy_variance, vec![0.5]);
        // Pooling across disorder would add Var_J(⟨e⟩) = 0.25.
        assert_eq!(agg.energies2[0] - agg.energies[0].powi(2), 0.75);
    }

    #[test]
    fn empty_aggregate_is_empty() {
        let agg = SweepResult::aggregate(&[]);
        assert!(agg.energies.is_empty() && agg.cluster_stats.top_cluster_sizes.is_empty());
    }

    #[test]
    fn top_cluster_modes_keep_their_own_lengths() {
        let tops = |first: Vec<[f64; 4]>| vec![first, vec![[0.5, 0.25, 0.0, 0.0]; 2]];
        let agg = SweepResult::aggregate(&[
            sample(0.0, 0.0, tops(vec![])),
            sample(0.0, 0.0, tops(vec![[1.0; 4]; 2])),
        ]);
        let sizes = &agg.cluster_stats.top_cluster_sizes;
        assert_eq!(sizes[0], vec![[0.5; 4]; 2]);
        assert_eq!(sizes[1], vec![[0.5, 0.25, 0.0, 0.0]; 2]);
    }
}
