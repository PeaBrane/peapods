use crate::geometry::Lattice;

use super::Statistics;

/// Overlap statistics from replica-pair measurements.
///
/// All vectors are indexed by temperature. Empty when `n_replicas < 2`.
pub struct OverlapStats {
    /// ⟨q⟩ — mean replica overlap.
    pub overlap: Vec<f64>,
    /// ⟨q²⟩.
    pub overlap2: Vec<f64>,
    /// ⟨q⁴⟩.
    pub overlap4: Vec<f64>,
    /// ⟨q_l⟩ — mean link overlap.
    pub link_overlap: Vec<f64>,
    /// ⟨q_l²⟩.
    pub link_overlap2: Vec<f64>,
    /// ⟨q_l⁴⟩.
    pub link_overlap4: Vec<f64>,
    /// Overlap histogram P(q) per temperature: `[n_temps][n_spins + 1]`.
    /// Bins correspond to dot-product values −N, −N+2, …, N where `idx = (dot + N) / 2`.
    pub histogram: Vec<Vec<u64>>,
    /// Conditional sum of q_l at each q bin: `[n_temps][n_spins + 1]`.
    /// Divide by `histogram` counts to get ⟨q_l | q⟩.
    pub ql_at_q_sum: Vec<Vec<f64>>,
    /// Conditional sum of q_l² at each q bin: `[n_temps][n_spins + 1]`.
    /// Use with `ql_at_q_sum` and `histogram` to compute A(q) = Var(q_l | q).
    pub ql2_at_q_sum: Vec<Vec<f64>>,
    /// Per-disorder-sample overlap histograms: `[n_disorder][n_temps][n_spins + 1]`.
    /// Only populated by `aggregate()`; empty for single-realization results.
    pub per_sample_histogram: Vec<Vec<Vec<u64>>>,
    /// Per-disorder-sample conditional sums: `[n_disorder][n_temps][n_spins + 1]`.
    /// Only populated by `aggregate()`; empty for single-realization results.
    pub per_sample_ql_at_q_sum: Vec<Vec<Vec<f64>>>,
    /// Per-disorder-sample conditional sums of squares: `[n_disorder][n_temps][n_spins + 1]`.
    /// Only populated by `aggregate()`; empty for single-realization results.
    pub per_sample_ql2_at_q_sum: Vec<Vec<Vec<f64>>>,
}

impl OverlapStats {
    pub fn empty() -> Self {
        Self {
            overlap: vec![],
            overlap2: vec![],
            overlap4: vec![],
            link_overlap: vec![],
            link_overlap2: vec![],
            link_overlap4: vec![],
            histogram: vec![],
            ql_at_q_sum: vec![],
            ql2_at_q_sum: vec![],
            per_sample_histogram: vec![],
            per_sample_ql_at_q_sum: vec![],
            per_sample_ql2_at_q_sum: vec![],
        }
    }

    pub fn aggregate(results: &[&Self]) -> Self {
        if results[0].overlap.is_empty() {
            return Self::empty();
        }

        let n = results.len() as f64;
        let n_temps = results[0].overlap.len();
        let n_hist = results[0].histogram.len();
        let hist_bins = results[0].histogram[0].len();
        let n_ql = results[0].ql_at_q_sum.len();
        let ql_bins = results[0].ql_at_q_sum.first().map_or(0, |v| v.len());

        let mut agg = Self {
            overlap: vec![0.0; n_temps],
            overlap2: vec![0.0; n_temps],
            overlap4: vec![0.0; n_temps],
            link_overlap: vec![0.0; n_temps],
            link_overlap2: vec![0.0; n_temps],
            link_overlap4: vec![0.0; n_temps],
            histogram: (0..n_hist).map(|_| vec![0u64; hist_bins]).collect(),
            ql_at_q_sum: (0..n_ql).map(|_| vec![0.0; ql_bins]).collect(),
            ql2_at_q_sum: (0..n_ql).map(|_| vec![0.0; ql_bins]).collect(),
            per_sample_histogram: results.iter().map(|r| r.histogram.clone()).collect(),
            per_sample_ql_at_q_sum: results.iter().map(|r| r.ql_at_q_sum.clone()).collect(),
            per_sample_ql2_at_q_sum: results.iter().map(|r| r.ql2_at_q_sum.clone()).collect(),
        };

        for r in results {
            for (a, &v) in agg.overlap.iter_mut().zip(r.overlap.iter()) {
                *a += v;
            }
            for (a, &v) in agg.overlap2.iter_mut().zip(r.overlap2.iter()) {
                *a += v;
            }
            for (a, &v) in agg.overlap4.iter_mut().zip(r.overlap4.iter()) {
                *a += v;
            }
            for (a, &v) in agg.link_overlap.iter_mut().zip(r.link_overlap.iter()) {
                *a += v;
            }
            for (a, &v) in agg.link_overlap2.iter_mut().zip(r.link_overlap2.iter()) {
                *a += v;
            }
            for (a, &v) in agg.link_overlap4.iter_mut().zip(r.link_overlap4.iter()) {
                *a += v;
            }
            for (a, s) in agg.histogram.iter_mut().zip(r.histogram.iter()) {
                for (ah, &sh) in a.iter_mut().zip(s.iter()) {
                    *ah += sh;
                }
            }
            for (a, s) in agg.ql_at_q_sum.iter_mut().zip(r.ql_at_q_sum.iter()) {
                for (ah, &sh) in a.iter_mut().zip(s.iter()) {
                    *ah += sh;
                }
            }
            for (a, s) in agg.ql2_at_q_sum.iter_mut().zip(r.ql2_at_q_sum.iter()) {
                for (ah, &sh) in a.iter_mut().zip(s.iter()) {
                    *ah += sh;
                }
            }
        }

        for v in agg
            .overlap
            .iter_mut()
            .chain(agg.overlap2.iter_mut())
            .chain(agg.overlap4.iter_mut())
            .chain(agg.link_overlap.iter_mut())
            .chain(agg.link_overlap2.iter_mut())
            .chain(agg.link_overlap4.iter_mut())
        {
            *v /= n;
        }

        agg
    }
}

/// Accumulator for overlap measurements during the sweep loop.
pub struct OverlapAccum {
    n_pairs: usize,
    n_temps: usize,
    n_spins: usize,
    n_bonds: usize,
    equil_diag: bool,
    collect_q2_ac: bool,

    overlap_stat: Statistics,
    overlap2_stat: Statistics,
    overlap4_stat: Statistics,
    link_overlap_stat: Statistics,
    link_overlap2_stat: Statistics,
    link_overlap4_stat: Statistics,

    histogram: Vec<Vec<u64>>,
    ql_at_q_sum: Vec<Vec<f64>>,
    ql2_at_q_sum: Vec<Vec<f64>>,

    overlaps_buf: Vec<f32>,
    overlaps2_buf: Vec<f32>,
    overlaps4_buf: Vec<f32>,
    link_overlaps_buf: Vec<f32>,
    link_overlaps2_buf: Vec<f32>,
    link_overlaps4_buf: Vec<f32>,

    pub diag_ql_buf: Vec<f32>,
    pub q2_ac_buf: Vec<f64>,
}

impl OverlapAccum {
    pub fn new(
        n_temps: usize,
        n_spins: usize,
        n_pairs: usize,
        n_neighbors: usize,
        equil_diag: bool,
        collect_q2_ac: bool,
    ) -> Self {
        let has_pairs = n_pairs > 0;
        Self {
            n_pairs,
            n_temps,
            n_spins,
            n_bonds: n_spins * n_neighbors,
            equil_diag,
            collect_q2_ac,

            overlap_stat: Statistics::new(n_temps, 1),
            overlap2_stat: Statistics::new(n_temps, 1),
            overlap4_stat: Statistics::new(n_temps, 1),
            link_overlap_stat: Statistics::new(n_temps, 1),
            link_overlap2_stat: Statistics::new(n_temps, 1),
            link_overlap4_stat: Statistics::new(n_temps, 1),

            histogram: if has_pairs {
                (0..n_temps).map(|_| vec![0u64; n_spins + 1]).collect()
            } else {
                vec![]
            },
            ql_at_q_sum: if has_pairs {
                (0..n_temps).map(|_| vec![0.0f64; n_spins + 1]).collect()
            } else {
                vec![]
            },
            ql2_at_q_sum: if has_pairs {
                (0..n_temps).map(|_| vec![0.0f64; n_spins + 1]).collect()
            } else {
                vec![]
            },

            overlaps_buf: vec![0.0f32; n_temps],
            overlaps2_buf: vec![0.0f32; n_temps],
            overlaps4_buf: vec![0.0f32; n_temps],
            link_overlaps_buf: vec![0.0f32; n_temps],
            link_overlaps2_buf: vec![0.0f32; n_temps],
            link_overlaps4_buf: vec![0.0f32; n_temps],

            diag_ql_buf: if equil_diag {
                vec![0.0f32; n_temps]
            } else {
                vec![]
            },
            q2_ac_buf: if collect_q2_ac {
                vec![0.0f64; n_temps]
            } else {
                vec![]
            },
        }
    }

    #[inline]
    pub fn collect(&mut self, lattice: &Lattice, spins: &[i8], system_ids: &[usize], record: bool) {
        if self.equil_diag {
            self.diag_ql_buf.fill(0.0);
        }
        if self.collect_q2_ac && record {
            self.q2_ac_buf.fill(0.0);
        }

        for pair_idx in 0..self.n_pairs {
            let r_a = 2 * pair_idx;
            let r_b = 2 * pair_idx + 1;

            for t in 0..self.n_temps {
                let sys_a = system_ids[r_a * self.n_temps + t];
                let sys_b = system_ids[r_b * self.n_temps + t];
                let base_a = sys_a * self.n_spins;
                let base_b = sys_b * self.n_spins;

                let mut dot_spin = 0i64;
                let mut dot_link = 0i64;
                for j in 0..self.n_spins {
                    let sa = spins[base_a + j] as i64;
                    let sb = spins[base_b + j] as i64;
                    dot_spin += sa * sb;
                    for d in 0..lattice.n_neighbors {
                        let k = lattice.neighbor_fwd(j, d);
                        dot_link +=
                            (sa * spins[base_a + k] as i64) * (sb * spins[base_b + k] as i64);
                    }
                }

                let ql = dot_link as f32 / self.n_bonds as f32;

                if self.equil_diag {
                    self.diag_ql_buf[t] += ql;
                }
                if !record {
                    continue;
                }

                let q = dot_spin as f32 / self.n_spins as f32;
                let q2 = q * q;
                self.overlaps_buf[t] = q;
                self.overlaps2_buf[t] = q2;
                self.overlaps4_buf[t] = q2 * q2;

                let ql2 = ql * ql;
                self.link_overlaps_buf[t] = ql;
                self.link_overlaps2_buf[t] = ql2;
                self.link_overlaps4_buf[t] = ql2 * ql2;

                let idx = ((dot_spin + self.n_spins as i64) / 2) as usize;
                self.histogram[t][idx] += 1;
                self.ql_at_q_sum[t][idx] += ql as f64;
                self.ql2_at_q_sum[t][idx] += (ql * ql) as f64;
            }

            if !record {
                continue;
            }

            if self.collect_q2_ac {
                for t in 0..self.n_temps {
                    self.q2_ac_buf[t] += self.overlaps2_buf[t] as f64;
                }
            }

            self.overlap_stat.update(&self.overlaps_buf);
            self.overlap2_stat.update(&self.overlaps2_buf);
            self.overlap4_stat.update(&self.overlaps4_buf);
            self.link_overlap_stat.update(&self.link_overlaps_buf);
            self.link_overlap2_stat.update(&self.link_overlaps2_buf);
            self.link_overlap4_stat.update(&self.link_overlaps4_buf);
        }

        if self.equil_diag {
            let inv = 1.0 / self.n_pairs as f32;
            for v in self.diag_ql_buf.iter_mut() {
                *v *= inv;
            }
        }
    }

    pub fn finish(self) -> OverlapStats {
        if self.n_pairs == 0 {
            return OverlapStats::empty();
        }
        OverlapStats {
            overlap: self.overlap_stat.average(),
            overlap2: self.overlap2_stat.average(),
            overlap4: self.overlap4_stat.average(),
            link_overlap: self.link_overlap_stat.average(),
            link_overlap2: self.link_overlap2_stat.average(),
            link_overlap4: self.link_overlap4_stat.average(),
            histogram: self.histogram,
            ql_at_q_sum: self.ql_at_q_sum,
            ql2_at_q_sum: self.ql2_at_q_sum,
            per_sample_histogram: vec![],
            per_sample_ql_at_q_sum: vec![],
            per_sample_ql2_at_q_sum: vec![],
        }
    }
}
