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
    /// FK cluster size histogram per temperature: `hist[s]` = count of size-`s` clusters.
    pub fk_csd: Vec<Vec<u64>>,
    /// Overlap cluster size histogram per temperature: `hist[s]` = count of size-`s` clusters.
    pub overlap_csd: Vec<Vec<u64>>,
    /// Average relative size of k-th largest blue cluster per temperature.
    /// Shape: [n_temps][4]. Empty if collect_top_clusters=false.
    pub top_cluster_sizes: Vec<[f64; 4]>,
    /// Normalized autocorrelation Γ(Δt) of m², shape [n_temps][max_lag+1].
    /// Empty if autocorrelation_max_lag is None.
    pub mags2_autocorrelation: Vec<Vec<f64>>,
    /// Normalized autocorrelation Γ(Δt) of q², shape [n_temps][max_lag+1].
    /// Empty if autocorrelation_max_lag is None or n_replicas < 2.
    pub overlap2_autocorrelation: Vec<Vec<f64>>,
}

impl SweepResult {
    /// Average [`SweepResult`]s across disorder realizations.
    pub fn aggregate(results: &[Self]) -> Self {
        let n = results.len() as f64;
        let n_temps = results[0].mags.len();
        let n_overlap = results[0].overlap.len();
        let n_fk_csd = results[0].fk_csd.len();
        let n_ov_csd = results[0].overlap_csd.len();

        let fk_len = results[0].fk_csd.first().map_or(0, |v| v.len());
        let ov_len = results[0].overlap_csd.first().map_or(0, |v| v.len());

        let n_top = results[0].top_cluster_sizes.len();

        let m2_ac_ntemps = results[0].mags2_autocorrelation.len();
        let m2_ac_len = results[0]
            .mags2_autocorrelation
            .first()
            .map_or(0, |v| v.len());
        let q2_ac_ntemps = results[0].overlap2_autocorrelation.len();
        let q2_ac_len = results[0]
            .overlap2_autocorrelation
            .first()
            .map_or(0, |v| v.len());

        let mut agg = SweepResult {
            mags: vec![0.0; n_temps],
            mags2: vec![0.0; n_temps],
            mags4: vec![0.0; n_temps],
            energies: vec![0.0; n_temps],
            energies2: vec![0.0; n_temps],
            overlap: vec![0.0; n_overlap],
            overlap2: vec![0.0; n_overlap],
            overlap4: vec![0.0; n_overlap],
            fk_csd: (0..n_fk_csd).map(|_| vec![0u64; fk_len]).collect(),
            overlap_csd: (0..n_ov_csd).map(|_| vec![0u64; ov_len]).collect(),
            top_cluster_sizes: vec![[0.0; 4]; n_top],
            mags2_autocorrelation: (0..m2_ac_ntemps).map(|_| vec![0.0; m2_ac_len]).collect(),
            overlap2_autocorrelation: (0..q2_ac_ntemps).map(|_| vec![0.0; q2_ac_len]).collect(),
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
            for (a, s) in agg.fk_csd.iter_mut().zip(r.fk_csd.iter()) {
                for (ah, &sh) in a.iter_mut().zip(s.iter()) {
                    *ah += sh;
                }
            }
            for (a, s) in agg.overlap_csd.iter_mut().zip(r.overlap_csd.iter()) {
                for (ah, &sh) in a.iter_mut().zip(s.iter()) {
                    *ah += sh;
                }
            }
            for (a, &s) in agg
                .top_cluster_sizes
                .iter_mut()
                .zip(r.top_cluster_sizes.iter())
            {
                for k in 0..4 {
                    a[k] += s[k];
                }
            }
            for (a_row, r_row) in agg
                .mags2_autocorrelation
                .iter_mut()
                .zip(r.mags2_autocorrelation.iter())
            {
                for (a, &v) in a_row.iter_mut().zip(r_row.iter()) {
                    *a += v;
                }
            }
            for (a_row, r_row) in agg
                .overlap2_autocorrelation
                .iter_mut()
                .zip(r.overlap2_autocorrelation.iter())
            {
                for (a, &v) in a_row.iter_mut().zip(r_row.iter()) {
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

        for arr in agg.top_cluster_sizes.iter_mut() {
            for v in arr.iter_mut() {
                *v /= n;
            }
        }

        for row in agg.mags2_autocorrelation.iter_mut() {
            for v in row.iter_mut() {
                *v /= n;
            }
        }
        for row in agg.overlap2_autocorrelation.iter_mut() {
            for v in row.iter_mut() {
                *v /= n;
            }
        }

        agg
    }
}
