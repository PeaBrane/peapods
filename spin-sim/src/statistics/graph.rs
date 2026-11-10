//! Accumulators for cluster-size histograms and observed cluster graphs.

use super::results::{ClusterObservations, GraphObservationSummary};
use crate::clusters::GraphObservationSlot;
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
    winding: Vec<[u64; 4]>,
    large_components: Vec<u64>,
}

impl GraphObservationAccum {
    fn new(n_temps: usize) -> Self {
        Self {
            observation_count: vec![0; n_temps],
            top4_sum: vec![[0; 4]; n_temps],
            active_bonds: vec![0; n_temps],
            winding: vec![[0; 4]; n_temps],
            large_components: vec![0; n_temps],
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
        self.winding[temp][0] += u64::from(slot.winding_x);
        self.winding[temp][1] += u64::from(slot.winding_y);
        self.winding[temp][2] += u64::from(slot.winding_x || slot.winding_y);
        self.winding[temp][3] += u64::from(slot.winding_x && slot.winding_y);
        self.large_components[temp] += u64::from(slot.large_components);
    }

    /// `cluster_size_counts` are the histograms of the observed graphs, per temperature.
    fn finish(
        self,
        n_spins: usize,
        n_neighbors: usize,
        has_winding: bool,
        cluster_size_counts: Vec<Vec<u64>>,
    ) -> Option<GraphObservationSummary> {
        if self.observation_count.iter().all(|&count| count == 0) {
            return None;
        }

        let n_temps = self.observation_count.len();
        let mut top_four_component_fractions = vec![[0.0; 4]; n_temps];
        let mut active_bond_density = vec![0.0; n_temps];
        let mut large_component_count = vec![0.0; n_temps];
        let mut winding = has_winding.then(|| vec![[0.0; 4]; n_temps]);
        for temp in 0..n_temps {
            let count = self.observation_count[temp];
            if count == 0 {
                continue;
            }
            let count_f64 = count as f64;
            for k in 0..4 {
                top_four_component_fractions[temp][k] =
                    self.top4_sum[temp][k] as f64 / (count_f64 * n_spins as f64);
                if let Some(ref mut winding) = winding {
                    winding[temp][k] = self.winding[temp][k] as f64 / count_f64;
                }
            }
            active_bond_density[temp] =
                self.active_bonds[temp] as f64 / (count_f64 * n_spins as f64 * n_neighbors as f64);
            large_component_count[temp] = self.large_components[temp] as f64 / count_f64;
        }

        Some(GraphObservationSummary {
            observation_count: self.observation_count,
            cluster_size_counts,
            top_four_component_fractions,
            active_bond_density,
            large_component_count,
            winding,
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
        has_winding: bool,
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
            accum.and_then(|accum| accum.finish(n_spins, n_neighbors, has_winding, counts))
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
