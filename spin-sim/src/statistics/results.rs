use super::equilibration::EquilCheckpoint;

pub struct ClusterStats {
    /// FK cluster size histogram per temperature: `hist[s]` = count of size-`s` clusters.
    pub fk_csd: Vec<Vec<u64>>,
    /// Overlap cluster size histogram per temperature: `hist[s]` = count of size-`s` clusters.
    pub overlap_csd: Vec<Vec<u64>>,
    /// Average relative size of k-th largest blue cluster per temperature.
    /// Shape: [n_temps][4]. Empty if collect_top_clusters=false.
    pub top_cluster_sizes: Vec<[f64; 4]>,
}

pub struct Diagnostics {
    /// Integrated autocorrelation time τ_int(m²) per temperature.
    /// Empty if autocorrelation_max_lag is None.
    pub mags2_tau: Vec<f64>,
    /// Integrated autocorrelation time τ_int(q²) per temperature.
    /// Empty if autocorrelation_max_lag is None or n_replicas < 2.
    pub overlap2_tau: Vec<f64>,
    /// Equilibration diagnostic checkpoints (energy + link overlap running averages).
    /// Empty if equilibration_diagnostic is false.
    pub equil_checkpoints: Vec<EquilCheckpoint>,
}

/// Per-temperature observables averaged over measurement sweeps and replicas.
///
/// All vectors are indexed by temperature index and have length `n_temps`.
/// Overlap vectors are empty when `n_replicas < 2`.
pub struct SweepResult {
    /// ⟨m⟩ — mean magnetization per spin.
    pub mags: Vec<f64>,
    /// ⟨m²⟩.
    pub mags2: Vec<f64>,
    /// ⟨m⁴⟩.
    pub mags4: Vec<f64>,
    /// ⟨E⟩ — mean energy per spin.
    pub energies: Vec<f64>,
    /// ⟨E²⟩.
    pub energies2: Vec<f64>,
    /// ⟨q⟩ — mean replica overlap.
    pub overlap: Vec<f64>,
    /// ⟨q²⟩.
    pub overlap2: Vec<f64>,
    /// ⟨q⁴⟩.
    pub overlap4: Vec<f64>,
    pub cluster_stats: ClusterStats,
    pub diagnostics: Diagnostics,
}

impl SweepResult {
    /// Average [`SweepResult`]s across disorder realizations.
    pub fn aggregate(results: &[Self]) -> Self {
        let n = results.len() as f64;
        let n_temps = results[0].mags.len();
        let n_overlap = results[0].overlap.len();
        let n_fk_csd = results[0].cluster_stats.fk_csd.len();
        let n_ov_csd = results[0].cluster_stats.overlap_csd.len();

        let fk_len = results[0]
            .cluster_stats
            .fk_csd
            .first()
            .map_or(0, |v| v.len());
        let ov_len = results[0]
            .cluster_stats
            .overlap_csd
            .first()
            .map_or(0, |v| v.len());

        let n_top = results[0].cluster_stats.top_cluster_sizes.len();

        let m2_tau_len = results[0].diagnostics.mags2_tau.len();
        let q2_tau_len = results[0].diagnostics.overlap2_tau.len();
        let n_ckpts = results[0].diagnostics.equil_checkpoints.len();

        let mut agg = SweepResult {
            mags: vec![0.0; n_temps],
            mags2: vec![0.0; n_temps],
            mags4: vec![0.0; n_temps],
            energies: vec![0.0; n_temps],
            energies2: vec![0.0; n_temps],
            overlap: vec![0.0; n_overlap],
            overlap2: vec![0.0; n_overlap],
            overlap4: vec![0.0; n_overlap],
            cluster_stats: ClusterStats {
                fk_csd: (0..n_fk_csd).map(|_| vec![0u64; fk_len]).collect(),
                overlap_csd: (0..n_ov_csd).map(|_| vec![0u64; ov_len]).collect(),
                top_cluster_sizes: vec![[0.0; 4]; n_top],
            },
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
            for (a, &v) in agg.overlap.iter_mut().zip(r.overlap.iter()) {
                *a += v;
            }
            for (a, &v) in agg.overlap2.iter_mut().zip(r.overlap2.iter()) {
                *a += v;
            }
            for (a, &v) in agg.overlap4.iter_mut().zip(r.overlap4.iter()) {
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
            for (a, s) in agg
                .cluster_stats
                .overlap_csd
                .iter_mut()
                .zip(r.cluster_stats.overlap_csd.iter())
            {
                for (ah, &sh) in a.iter_mut().zip(s.iter()) {
                    *ah += sh;
                }
            }
            for (a, &s) in agg
                .cluster_stats
                .top_cluster_sizes
                .iter_mut()
                .zip(r.cluster_stats.top_cluster_sizes.iter())
            {
                for k in 0..4 {
                    a[k] += s[k];
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
            .chain(agg.overlap.iter_mut())
            .chain(agg.overlap2.iter_mut())
            .chain(agg.overlap4.iter_mut())
        {
            *v /= n;
        }

        for arr in agg.cluster_stats.top_cluster_sizes.iter_mut() {
            for v in arr.iter_mut() {
                *v /= n;
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
