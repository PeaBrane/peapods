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

    /// Average across disorder realizations; an empty slice yields [`OverlapStats::empty`].
    pub fn aggregate(results: &[&Self]) -> Self {
        Self::aggregate_impl(results, true)
    }

    pub(crate) fn aggregate_without_samples(results: &[&Self]) -> Self {
        Self::aggregate_impl(results, false)
    }

    fn aggregate_impl(results: &[&Self], retain_samples: bool) -> Self {
        if results.first().is_none_or(|r| r.overlap.is_empty()) {
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
            per_sample_histogram: if retain_samples {
                results.iter().map(|r| r.histogram.clone()).collect()
            } else {
                vec![]
            },
            per_sample_ql_at_q_sum: if retain_samples {
                results.iter().map(|r| r.ql_at_q_sum.clone()).collect()
            } else {
                vec![]
            },
            per_sample_ql2_at_q_sum: if retain_samples {
                results.iter().map(|r| r.ql2_at_q_sum.clone()).collect()
            } else {
                vec![]
            },
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
    /// Replicas entering the all-pairs q² average; odd counts have one unpaired.
    n_replicas: usize,
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
    site_overlap: Vec<i8>,
}

/// Sum of `q²` over every pair among `n_replicas` equal-length configurations, with
/// the number of pairs. Unlike fixed pairs, the sum ignores how replicas are labeled.
pub(crate) fn pair_q2_sum<'a>(
    n_replicas: usize,
    replica: impl Fn(usize) -> &'a [i8],
) -> (f64, usize) {
    let mut sum = 0.0;
    let mut pairs = 0;
    for a in 0..n_replicas {
        let spins_a = replica(a);
        let n = spins_a.len() as f64;
        for b in a + 1..n_replicas {
            let dot: i64 = spins_a
                .iter()
                .zip(replica(b))
                .map(|(&x, &y)| i64::from(x * y))
                .sum();
            let q = dot as f64 / n;
            sum += q * q;
            pairs += 1;
        }
    }
    (sum, pairs)
}

/// Returns `(Σ_i q_i, Σ_i Σ_d q_i q_{fwd(i,d)})` for site overlaps `q_i = ±1`.
///
/// Canonical square and cubic lattices use contiguous shifted products; other
/// lattices walk the neighbor table. Both give the same integers.
fn overlap_sums(lattice: &Lattice, q: &[i8]) -> (i64, i64) {
    if let Some((l0, l1)) = lattice.square_shape() {
        return overlap_sums_strided(&[l0, l1], q);
    }
    if let Some(shape) = lattice.cubic_shape() {
        return overlap_sums_strided(&shape, q);
    }
    overlap_sums_chunked(lattice, q, i32::MAX as usize / (lattice.n_neighbors + 1))
}

/// Exact dot product of two ±1 vectors.
fn dot_i8(a: &[i8], b: &[i8]) -> i64 {
    // Short chunks keep each i32 partial sum far from overflow and vectorizable.
    const CHUNK: usize = 1 << 15;
    a.chunks(CHUNK)
        .zip(b.chunks(CHUNK))
        .map(|(a, b)| {
            i64::from(
                a.iter()
                    .zip(b)
                    .map(|(&x, &y)| i32::from(x * y))
                    .sum::<i32>(),
            )
        })
        .sum()
}

/// [`overlap_sums`] on a row-major hypercubic lattice with the given extents.
///
/// Along a direction of stride `s` and extent `L` the lattice splits into blocks of
/// `s * L` sites; within a block site `j` pairs with `j + s`, and the last `s` sites
/// wrap to the first `s`.
fn overlap_sums_strided(extents: &[usize], q: &[i8]) -> (i64, i64) {
    if q.is_empty() {
        return (0, 0);
    }
    let dot_spin = q
        .chunks(1 << 15)
        .map(|chunk| i64::from(chunk.iter().map(|&v| i32::from(v)).sum::<i32>()))
        .sum();
    let mut dot_link = 0i64;
    let mut stride = q.len();
    for &extent in extents {
        let period = stride;
        stride /= extent;
        for block in q.chunks_exact(period) {
            dot_link += dot_i8(&block[..period - stride], &block[stride..]);
            dot_link += dot_i8(&block[period - stride..], &block[..stride]);
        }
    }
    (dot_spin, dot_link)
}

fn overlap_sums_chunked(lattice: &Lattice, q: &[i8], chunk: usize) -> (i64, i64) {
    let n_neighbors = lattice.n_neighbors;
    let chunk = chunk.max(1);
    let mut dot_spin = 0i64;
    let mut dot_link = 0i64;
    for start in (0..q.len()).step_by(chunk) {
        let mut spin_acc = 0i32;
        let mut link_acc = 0i32;
        for j in start..q.len().min(start + chunk) {
            let qj = q[j] as i32;
            let mut neighbors = 0i32;
            for d in 0..n_neighbors {
                neighbors += q[lattice.neighbor_fwd(j, d)] as i32;
            }
            spin_acc += qj;
            link_acc += qj * neighbors;
        }
        dot_spin += spin_acc as i64;
        dot_link += link_acc as i64;
    }
    (dot_spin, dot_link)
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
            n_replicas: 2 * n_pairs,
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
            site_overlap: if has_pairs { vec![0; n_spins] } else { vec![] },
        }
    }

    /// Drops the link-overlap sums conditioned on the overlap bin, which are two of
    /// the three histogram-sized arrays.
    pub fn without_conditional_link_overlap(mut self) -> Self {
        self.ql_at_q_sum = vec![];
        self.ql2_at_q_sum = vec![];
        self
    }

    /// Sets the replica count for the all-pairs q² average, which also covers the
    /// unpaired replica of an odd count.
    pub fn with_replicas(mut self, n_replicas: usize) -> Self {
        self.n_replicas = n_replicas;
        self
    }

    #[inline]
    pub fn collect(&mut self, lattice: &Lattice, spins: &[i8], system_ids: &[usize], record: bool) {
        if self.equil_diag {
            self.diag_ql_buf.fill(0.0);
        }
        for pair_idx in 0..self.n_pairs {
            let r_a = 2 * pair_idx;
            let r_b = 2 * pair_idx + 1;

            for t in 0..self.n_temps {
                let sys_a = system_ids[r_a * self.n_temps + t];
                let sys_b = system_ids[r_b * self.n_temps + t];
                let spins_a = &spins[sys_a * self.n_spins..(sys_a + 1) * self.n_spins];
                let spins_b = &spins[sys_b * self.n_spins..(sys_b + 1) * self.n_spins];
                for ((q, &sa), &sb) in self.site_overlap.iter_mut().zip(spins_a).zip(spins_b) {
                    *q = sa * sb;
                }
                let (dot_spin, dot_link) = overlap_sums(lattice, &self.site_overlap);

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
                if let Some(sum) = self.ql_at_q_sum.get_mut(t) {
                    sum[idx] += ql as f64;
                    self.ql2_at_q_sum[t][idx] += (ql * ql) as f64;
                }
            }

            if !record {
                continue;
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
        if self.collect_q2_ac && record {
            self.average_q2_over_all_pairs(spins, system_ids);
        }
    }

    /// Mean q^2 over every replica pair at each temperature, for the autocorrelation.
    ///
    /// Fixed pairs (2p, 2p + 1) would let moves that merely relabel replicas at one
    /// temperature (e.g. a spanning Houdayer cluster) look like decorrelation; the
    /// all-pairs mean is invariant under such relabelings.
    fn average_q2_over_all_pairs(&mut self, spins: &[i8], system_ids: &[usize]) {
        let n = self.n_spins;
        for t in 0..self.n_temps {
            let (sum, pairs) = pair_q2_sum(self.n_replicas, |replica| {
                let system = system_ids[replica * self.n_temps + t];
                &spins[system * n..(system + 1) * n]
            });
            self.q2_ac_buf[t] = sum / pairs as f64;
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

#[cfg(test)]
mod tests {

    /// The autocorrelation observable must not change when replicas at one temperature
    /// swap labels, and must reduce to the single pair's q^2 for two replicas.
    #[test]
    fn q2_autocorrelation_observable_is_relabel_invariant() {
        let lattice = Lattice::new(vec![4, 4]);
        let n = lattice.n_spins;
        let n_temps = 2;
        let spins: Vec<i8> = (0..4 * n_temps * n)
            .map(|k| if (k * 37 + k / 5) % 7 < 3 { -1 } else { 1 })
            .collect();
        let observable = |system_ids: &[usize], n_pairs: usize| {
            let mut accum =
                OverlapAccum::new(n_temps, n, n_pairs, lattice.n_neighbors, false, true);
            accum.collect(&lattice, &spins, system_ids, true);
            accum.q2_ac_buf.clone()
        };
        let identity: Vec<usize> = (0..4 * n_temps).collect();
        // Swap replicas 1 and 2 at temperature 0 only.
        let mut relabelled = identity.clone();
        relabelled.swap(n_temps, 2 * n_temps);
        assert_eq!(observable(&identity, 2), observable(&relabelled, 2));

        // An odd count keeps its unpaired replica in the average.
        let three = |system_ids: &[usize]| {
            let mut accum =
                OverlapAccum::new(n_temps, n, 1, lattice.n_neighbors, false, true).with_replicas(3);
            accum.collect(&lattice, &spins, system_ids, true);
            accum.q2_ac_buf.clone()
        };
        let mut rotated = identity[..3 * n_temps].to_vec();
        rotated.swap(0, 2 * n_temps);
        assert_eq!(three(&identity[..3 * n_temps]), three(&rotated));
        assert_ne!(
            three(&identity[..3 * n_temps]),
            observable(&identity[..2 * n_temps], 1)
        );

        let two = observable(&identity[..2 * n_temps], 1);
        for (t, value) in two.iter().enumerate() {
            let dot: i64 = (0..n)
                .map(|i| i64::from(spins[t * n + i] * spins[(n_temps + t) * n + i]))
                .sum();
            let q = dot as f64 / n as f64;
            assert!((value - q * q).abs() < 1e-12);
        }
    }

    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoshiro256StarStar;

    /// The strided path must reproduce the neighbor-table walk, including double
    /// bonds (extent 2) and unequal extents.
    #[test]
    fn strided_overlap_sums_match_the_table_walk() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(21);
        for shape in [
            vec![2, 2],
            vec![3, 7],
            vec![8, 5],
            vec![2, 3, 4],
            vec![5, 2, 3],
            vec![4, 4, 4],
        ] {
            let lattice = Lattice::new(shape.clone());
            let q: Vec<i8> = (0..lattice.n_spins)
                .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
                .collect();
            assert_eq!(
                overlap_sums(&lattice, &q),
                overlap_sums_chunked(&lattice, &q, 7),
                "{shape:?}"
            );
        }
    }

    #[test]
    fn chunked_overlap_sums_match_direct_definition() {
        let lattice = Lattice::with_offsets(vec![5, 6], vec![vec![1, 0], vec![0, 1], vec![1, -1]]);
        let mut rng = Xoshiro256StarStar::seed_from_u64(9);
        let q: Vec<i8> = (0..lattice.n_spins)
            .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
            .collect();
        let spin: i64 = q.iter().map(|&v| v as i64).sum();
        let link: i64 = (0..lattice.n_spins)
            .flat_map(|i| (0..lattice.n_neighbors).map(move |d| (i, d)))
            .map(|(i, d)| q[i] as i64 * q[lattice.neighbor_fwd(i, d)] as i64)
            .sum();
        assert_eq!(overlap_sums(&lattice, &q), (spin, link));
        for chunk in [1, 7, 29, 30, 1000] {
            assert_eq!(overlap_sums_chunked(&lattice, &q, chunk), (spin, link));
        }
    }

    #[test]
    fn empty_aggregate_is_empty() {
        let empty = OverlapStats::aggregate(&[]);
        assert!(empty.overlap.is_empty() && empty.per_sample_histogram.is_empty());
    }
}
