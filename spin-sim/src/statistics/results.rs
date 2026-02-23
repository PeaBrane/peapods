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

        agg
    }
}
