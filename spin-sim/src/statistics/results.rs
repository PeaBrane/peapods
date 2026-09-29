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
    pub overlap_csd: Vec<Vec<Vec<u64>>>,
    /// Per-mode average relative size of k-th largest overlap cluster: `[n_modes][n_temps][4]`.
    /// Zero for a mode that never ran during measurement.
    pub top_cluster_sizes: Vec<Vec<[f64; 4]>>,
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
    /// Integrated autocorrelation time τ_int(m²) per temperature.
    /// Empty if autocorrelation_max_lag is None.
    pub mags2_tau: Vec<f64>,
    /// Integrated autocorrelation time τ_int(q²) per temperature.
    /// Empty if autocorrelation_max_lag is None or n_replicas < 2.
    pub overlap2_tau: Vec<f64>,
    /// Equilibration diagnostic checkpoints: energy and link overlap averaged over
    /// the log-binned window of each checkpoint (see [`EquilCheckpoint`]).
    /// Empty if equilibration_diagnostic is false.
    pub equil_checkpoints: Vec<EquilCheckpoint>,
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
    pub diagnostics: Diagnostics,
    pub cluster_snapshots: Vec<ClusterSnapshot>,
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
            },
            per_disorder_physics: vec![],
            per_disorder_cluster_observations: vec![],
            diagnostics: Diagnostics {
                mags2_tau: vec![],
                overlap2_tau: vec![],
                equil_checkpoints: vec![],
            },
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
        let n_fk_csd = results[0].cluster_stats.fk_csd.len();

        let fk_len = results[0]
            .cluster_stats
            .fk_csd
            .first()
            .map_or(0, |v| v.len());

        let n_modes = results[0].cluster_stats.overlap_csd.len();
        let ov_inner_len = results[0]
            .cluster_stats
            .overlap_csd
            .first()
            .and_then(|v| v.first())
            .map_or(0, |v| v.len());
        let ov_inner_temps = results[0]
            .cluster_stats
            .overlap_csd
            .first()
            .map_or(0, |v| v.len());

        let n_top_modes = results[0].cluster_stats.top_cluster_sizes.len();
        let n_top_temps: Vec<usize> = (0..n_top_modes)
            .map(|mode| {
                results
                    .iter()
                    .filter_map(|r| r.cluster_stats.top_cluster_sizes.get(mode))
                    .map(Vec::len)
                    .max()
                    .unwrap_or(0)
            })
            .collect();

        let m2_tau_len = results[0].diagnostics.mags2_tau.len();
        let q2_tau_len = results[0].diagnostics.overlap2_tau.len();
        let n_ckpts = results[0].diagnostics.equil_checkpoints.len();

        let overlap_results: Vec<_> = results.iter().map(|r| &r.overlap_stats).collect();
        let overlap_stats = if retain_overlap_samples {
            OverlapStats::aggregate(&overlap_results)
        } else {
            OverlapStats::aggregate_without_samples(&overlap_results)
        };
        let per_disorder_cluster_observations = results
            .iter()
            .flat_map(|result| result.per_disorder_cluster_observations.iter().cloned())
            .collect();

        let mut agg = SweepResult {
            mags: vec![0.0; n_temps],
            mags2: vec![0.0; n_temps],
            mags4: vec![0.0; n_temps],
            energies: vec![0.0; n_temps],
            energies2: vec![0.0; n_temps],
            energy_variance: vec![0.0; n_temps],
            overlap_stats,
            cluster_stats: ClusterStats {
                fk_csd: (0..n_fk_csd).map(|_| vec![0u64; fk_len]).collect(),
                overlap_csd: (0..n_modes)
                    .map(|_| {
                        (0..ov_inner_temps)
                            .map(|_| vec![0u64; ov_inner_len])
                            .collect()
                    })
                    .collect(),
                top_cluster_sizes: n_top_temps.iter().map(|&len| vec![[0.0; 4]; len]).collect(),
            },
            per_disorder_cluster_observations,
            per_disorder_physics: results
                .iter()
                .flat_map(|r| r.per_disorder_physics.iter().cloned())
                .collect(),
            diagnostics: Diagnostics {
                mags2_tau: vec![0.0; m2_tau_len],
                overlap2_tau: vec![0.0; q2_tau_len],
                equil_checkpoints: (0..n_ckpts)
                    .map(|i| EquilCheckpoint {
                        sweep: results[0].diagnostics.equil_checkpoints[i].sweep,
                        energy_avg: vec![0.0; n_temps],
                        link_overlap_avg: vec![0.0; n_temps],
                    })
                    .collect(),
            },
            cluster_snapshots: Vec::new(),
        };

        for r in results {
            for (a, &v) in agg.mags.iter_mut().zip(r.mags.iter()) {
                *a += v;
            }
            for (a, &v) in agg.mags2.iter_mut().zip(r.mags2.iter()) {
                *a += v;
            }
            for (a, &v) in agg.mags4.iter_mut().zip(r.mags4.iter()) {
                *a += v;
            }
            for (a, &v) in agg.energies.iter_mut().zip(r.energies.iter()) {
                *a += v;
            }
            for (a, &v) in agg.energies2.iter_mut().zip(r.energies2.iter()) {
                *a += v;
            }
            for (a, &v) in agg.energy_variance.iter_mut().zip(r.energy_variance.iter()) {
                *a += v;
            }
            for (a, s) in agg
                .cluster_stats
                .fk_csd
                .iter_mut()
                .zip(r.cluster_stats.fk_csd.iter())
            {
                for (ah, &sh) in a.iter_mut().zip(s.iter()) {
                    *ah += sh;
                }
            }
            for (am, sm) in agg
                .cluster_stats
                .overlap_csd
                .iter_mut()
                .zip(r.cluster_stats.overlap_csd.iter())
            {
                for (a, s) in am.iter_mut().zip(sm.iter()) {
                    for (ah, &sh) in a.iter_mut().zip(s.iter()) {
                        *ah += sh;
                    }
                }
            }
            for (am, sm) in agg
                .cluster_stats
                .top_cluster_sizes
                .iter_mut()
                .zip(r.cluster_stats.top_cluster_sizes.iter())
            {
                for (a, &s) in am.iter_mut().zip(sm.iter()) {
                    for k in 0..4 {
                        a[k] += s[k];
                    }
                }
            }
            for (a, &v) in agg
                .diagnostics
                .mags2_tau
                .iter_mut()
                .zip(r.diagnostics.mags2_tau.iter())
            {
                *a += v;
            }
            for (a, &v) in agg
                .diagnostics
                .overlap2_tau
                .iter_mut()
                .zip(r.diagnostics.overlap2_tau.iter())
            {
                *a += v;
            }
            for (ac, rc) in agg
                .diagnostics
                .equil_checkpoints
                .iter_mut()
                .zip(r.diagnostics.equil_checkpoints.iter())
            {
                for (a, &v) in ac.energy_avg.iter_mut().zip(rc.energy_avg.iter()) {
                    *a += v;
                }
                for (a, &v) in ac
                    .link_overlap_avg
                    .iter_mut()
                    .zip(rc.link_overlap_avg.iter())
                {
                    *a += v;
                }
            }
        }

        for v in agg
            .mags
            .iter_mut()
            .chain(agg.mags2.iter_mut())
            .chain(agg.mags4.iter_mut())
            .chain(agg.energies.iter_mut())
            .chain(agg.energies2.iter_mut())
            .chain(agg.energy_variance.iter_mut())
        {
            *v /= n;
        }

        for mode_tops in agg.cluster_stats.top_cluster_sizes.iter_mut() {
            for arr in mode_tops.iter_mut() {
                for v in arr.iter_mut() {
                    *v /= n;
                }
            }
        }

        for v in agg.diagnostics.mags2_tau.iter_mut() {
            *v /= n;
        }
        for v in agg.diagnostics.overlap2_tau.iter_mut() {
            *v /= n;
        }
        for ckpt in agg.diagnostics.equil_checkpoints.iter_mut() {
            for v in ckpt.energy_avg.iter_mut() {
                *v /= n;
            }
            for v in ckpt.link_overlap_avg.iter_mut() {
                *v /= n;
            }
        }

        agg
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
