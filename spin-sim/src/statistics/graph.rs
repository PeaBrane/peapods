//! Accumulators for cluster-size histograms and observed cluster graphs.

use super::results::{ClusterObservations, GraphObservationSummary};
use crate::clusters::{GraphObservationSlot, MAX_WINDING_DIMS};
use crate::config::OverlapClusterBuildMode;

/// `n` zeroed cluster-size histograms with one bin per size `0..=n_spins`.
pub(crate) fn histograms(n: usize, n_spins: usize) -> Vec<Vec<u64>> {
    (0..n).map(|_| vec![0; n_spins + 1]).collect()
}

/// Sums per-slot histograms into one per temperature; `temp_of` maps a slot to its
/// temperature. No slots gives zeroed histograms of the full shape.
pub(crate) fn fold_histograms(
    slots: &[Vec<u64>],
    n_temps: usize,
    n_spins: usize,
    temp_of: impl Fn(usize) -> usize,
) -> Vec<Vec<u64>> {
    let mut folded = histograms(n_temps, n_spins);
    for (slot, histogram) in slots.iter().enumerate() {
        add_histograms(&mut folded[temp_of(slot)], histogram);
    }
    folded
}

fn add_histograms(total: &mut [u64], part: &[u64]) {
    for (total, &count) in total.iter_mut().zip(part) {
        *total += count;
    }
}

pub(crate) struct GraphObservationAccum {
    observation_count: Vec<u64>,
    top4_sum: Vec<[u64; 4]>,
    active_bonds: Vec<u64>,
    wraps: Vec<[u64; MAX_WINDING_DIMS]>,
    wraps_any: Vec<u64>,
    wraps_all: Vec<u64>,
    wrapping_clusters: Vec<u64>,
    wrapping_sites: Vec<u64>,
    large_components: Vec<u64>,
    s2: Vec<f64>,
    s4: Vec<f64>,
    s2_squared: Vec<f64>,
}

impl GraphObservationAccum {
    fn new(n_temps: usize) -> Self {
        Self {
            observation_count: vec![0; n_temps],
            top4_sum: vec![[0; 4]; n_temps],
            active_bonds: vec![0; n_temps],
            wraps: vec![[0; MAX_WINDING_DIMS]; n_temps],
            wraps_any: vec![0; n_temps],
            wraps_all: vec![0; n_temps],
            wrapping_clusters: vec![0; n_temps],
            wrapping_sites: vec![0; n_temps],
            large_components: vec![0; n_temps],
            s2: vec![0.0; n_temps],
            s4: vec![0.0; n_temps],
            s2_squared: vec![0.0; n_temps],
        }
    }

    pub(crate) fn push(&mut self, temp: usize, slot: GraphObservationSlot) {
        if !slot.observed {
            return;
        }
        self.observation_count[temp] += 1;
        for (total, value) in self.top4_sum[temp].iter_mut().zip(slot.top4) {
            *total += u64::from(value);
        }
        self.active_bonds[temp] += u64::from(slot.active_bonds);
        for (total, wraps) in self.wraps[temp].iter_mut().zip(slot.wraps) {
            *total += u64::from(wraps);
        }
        self.wraps_any[temp] += u64::from(slot.wraps.contains(&true));
        self.wraps_all[temp] += u64::from(slot.wraps_all);
        self.wrapping_clusters[temp] += u64::from(slot.wrapping_clusters);
        self.wrapping_sites[temp] += u64::from(slot.wrapping_sites);
        self.large_components[temp] += u64::from(slot.large_components);
        self.s2[temp] += slot.s2;
        self.s4[temp] += slot.s4;
        self.s2_squared[temp] += slot.s2 * slot.s2;
    }

    /// `cluster_size_counts` are the histograms of the observed graphs, per
    /// temperature; `winding_dims` is the number of axes wrapping was tracked along
    /// (0 on lattices where it is not).
    fn finish(
        self,
        n_spins: usize,
        n_neighbors: usize,
        winding_dims: usize,
        cluster_size_counts: Vec<Vec<u64>>,
    ) -> Option<GraphObservationSummary> {
        if self.observation_count.iter().all(|&count| count == 0) {
            return None;
        }

        // Means over the observations of each temperature; zero where none was made.
        let mean = |total: f64, temp: usize| match self.observation_count[temp] {
            0 => 0.0,
            count => total / count as f64,
        };
        let means = |totals: &[u64]| -> Vec<f64> {
            totals
                .iter()
                .enumerate()
                .map(|(temp, &total)| mean(total as f64, temp))
                .collect()
        };
        let float_means = |totals: &[f64]| -> Vec<f64> {
            totals
                .iter()
                .enumerate()
                .map(|(temp, &total)| mean(total, temp))
                .collect()
        };
        let n_temps = self.observation_count.len();
        let n_sites = n_spins as f64;
        let tracked = winding_dims > 0;

        let wrapping_any = means(&self.wraps_any);
        let wrapping_all = means(&self.wraps_all);
        let wrapping_axes: Vec<Vec<f64>> = (0..n_temps)
            .map(|temp| {
                self.wraps[temp][..winding_dims]
                    .iter()
                    .map(|&total| mean(total as f64, temp))
                    .collect()
            })
            .collect();
        // The 2D layout predates wrapping along three axes.
        let winding = (winding_dims == 2).then(|| {
            (0..n_temps)
                .map(|temp| {
                    [
                        wrapping_axes[temp][0],
                        wrapping_axes[temp][1],
                        wrapping_any[temp],
                        wrapping_all[temp],
                    ]
                })
                .collect()
        });

        Some(GraphObservationSummary {
            top_four_component_fractions: (0..n_temps)
                .map(|temp| self.top4_sum[temp].map(|total| mean(total as f64, temp) / n_sites))
                .collect(),
            active_bond_density: means(&self.active_bonds)
                .iter()
                .map(|bonds| bonds / (n_sites * n_neighbors as f64))
                .collect(),
            large_component_count: means(&self.large_components),
            winding,
            wrapping_cluster_count: tracked.then(|| means(&self.wrapping_clusters)),
            wrapping_fraction: tracked.then(|| {
                means(&self.wrapping_sites)
                    .iter()
                    .map(|sites| sites / n_sites)
                    .collect()
            }),
            wrapping_axes: tracked.then_some(wrapping_axes),
            wrapping_any: tracked.then_some(wrapping_any),
            wrapping_all: tracked.then_some(wrapping_all),
            cluster_s2: float_means(&self.s2),
            cluster_s4: float_means(&self.s4),
            cluster_s2_squared: float_means(&self.s2_squared),
            observation_count: self.observation_count,
            cluster_size_counts,
        })
    }
}

/// The families that observed overlap graphs are reported under.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Family {
    Houdayer,
    Jorg,
    CmrBlue,
}

impl Family {
    /// Replica Monte Carlo publishes no graph.
    fn of(mode: &OverlapClusterBuildMode) -> Option<Self> {
        match mode {
            OverlapClusterBuildMode::Houdayer(_)
            | OverlapClusterBuildMode::Pair(_)
            | OverlapClusterBuildMode::Perm(_)
            | OverlapClusterBuildMode::GaugedPerm(_) => Some(Self::Houdayer),
            OverlapClusterBuildMode::Jorg(_) => Some(Self::Jorg),
            OverlapClusterBuildMode::Cmr => Some(Self::CmrBlue),
            OverlapClusterBuildMode::Rmc => None,
        }
    }
}

/// One accumulator per family of observed graphs.
#[derive(Default)]
pub(crate) struct ClusterObservationAccums {
    pub(crate) fk: Option<GraphObservationAccum>,
    houdayer: Option<GraphObservationAccum>,
    jorg: Option<GraphObservationAccum>,
    cmr_blue: Option<GraphObservationAccum>,
}

impl ClusterObservationAccums {
    fn family(&mut self, family: Family) -> &mut Option<GraphObservationAccum> {
        match family {
            Family::Houdayer => &mut self.houdayer,
            Family::Jorg => &mut self.jorg,
            Family::CmrBlue => &mut self.cmr_blue,
        }
    }

    pub(crate) fn enable_fk(&mut self, n_temps: usize) {
        self.fk = Some(GraphObservationAccum::new(n_temps));
    }

    pub(crate) fn enable_overlap(&mut self, mode: &OverlapClusterBuildMode, n_temps: usize) {
        let Some(family) = Family::of(mode) else {
            return;
        };
        self.family(family)
            .get_or_insert_with(|| GraphObservationAccum::new(n_temps));
    }

    pub(crate) fn overlap_mut(
        &mut self,
        mode: &OverlapClusterBuildMode,
    ) -> &mut GraphObservationAccum {
        Family::of(mode)
            .and_then(|family| self.family(family).as_mut())
            .expect("observed graph accumulator must be allocated")
    }

    /// `fk_csd` and `overlap_csd` (one entry per mode of `modes`) are the folded
    /// cluster-size histograms of the run, which in observe mode are those of the
    /// observed graphs.
    pub(crate) fn finish(
        self,
        n_spins: usize,
        n_neighbors: usize,
        winding_dims: usize,
        fk_csd: &[Vec<u64>],
        modes: &[OverlapClusterBuildMode],
        overlap_csd: &[Vec<Vec<u64>>],
    ) -> ClusterObservations {
        // Several modes can report under one family; their histograms add up.
        let family_counts = |family: Family| {
            let mut total: Vec<Vec<u64>> = Vec::new();
            let of_family = modes
                .iter()
                .zip(overlap_csd)
                .filter(|(mode, _)| Family::of(mode) == Some(family));
            for (_, csd) in of_family {
                if total.is_empty() {
                    total = csd.clone();
                    continue;
                }
                for (total, part) in total.iter_mut().zip(csd) {
                    add_histograms(total, part);
                }
            }
            total
        };
        let finish = |accum: Option<GraphObservationAccum>, counts: Vec<Vec<u64>>| {
            accum.and_then(|accum| accum.finish(n_spins, n_neighbors, winding_dims, counts))
        };
        ClusterObservations {
            fk: finish(self.fk, fk_csd.to_vec()),
            houdayer: finish(self.houdayer, family_counts(Family::Houdayer)),
            jorg: finish(self.jorg, family_counts(Family::Jorg)),
            cmr_blue: finish(self.cmr_blue, family_counts(Family::CmrBlue)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folding_keeps_the_shape_without_slots() {
        assert_eq!(fold_histograms(&[], 2, 3, |slot| slot), vec![vec![0; 4]; 2]);
        let slots = vec![vec![1, 0, 2], vec![0, 4, 0], vec![3, 0, 0]];
        assert_eq!(
            fold_histograms(&slots, 2, 2, |slot| slot % 2),
            vec![vec![4, 0, 2], vec![0, 4, 0]]
        );
    }
}
