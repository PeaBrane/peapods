/// Replica-averaged energy and link overlap over the log-binned window ending at `sweep`.
///
/// The window is the last `⌈sweep / 2⌉` sweeps, `(⌊sweep / 2⌋, sweep]` counting from
/// one, i.e. `[2^(k-1), 2^k)` at the power-of-two checkpoints (Katzgraber, Palassini
/// & Young, PRB 63, 184422 (2001)). Early non-equilibrium sweeps therefore drop out
/// of later checkpoints instead of biasing a cumulative average.
pub struct EquilCheckpoint {
    pub sweep: usize,
    pub energy_avg: Vec<f64>,
    pub link_overlap_avg: Vec<f64>,
}

pub struct EquilDiagnosticAccum {
    n_temps: usize,
    checkpoints: Vec<usize>,
    next_ckpt_idx: usize,
    next_start_idx: usize,
    count: usize,
    sum_energy: Vec<f64>,
    sum_link_overlap: Vec<f64>,
    /// Cumulative sums at the start of each opened window, indexed like `checkpoints`.
    window_starts: Vec<[Vec<f64>; 2]>,
    snapshots: Vec<EquilCheckpoint>,
}

impl EquilDiagnosticAccum {
    pub fn new(n_temps: usize, n_sweeps: usize) -> Self {
        let mut checkpoints = Vec::new();
        let mut p = 128usize;
        while p < n_sweeps {
            checkpoints.push(p);
            p *= 2;
        }
        if checkpoints.last() != Some(&n_sweeps) {
            checkpoints.push(n_sweeps);
        }

        let mut accum = Self {
            n_temps,
            checkpoints,
            next_ckpt_idx: 0,
            next_start_idx: 0,
            count: 0,
            sum_energy: vec![0.0; n_temps],
            sum_link_overlap: vec![0.0; n_temps],
            window_starts: Vec::new(),
            snapshots: Vec::new(),
        };
        accum.open_windows();
        accum
    }

    fn window_start(checkpoint: usize) -> usize {
        checkpoint / 2
    }

    fn open_windows(&mut self) {
        while self
            .checkpoints
            .get(self.next_start_idx)
            .is_some_and(|&c| Self::window_start(c) == self.count)
        {
            self.window_starts
                .push([self.sum_energy.clone(), self.sum_link_overlap.clone()]);
            self.next_start_idx += 1;
        }
    }

    pub fn push(&mut self, energies: &[f32], link_overlaps: &[f32]) {
        self.count += 1;
        for t in 0..self.n_temps {
            self.sum_energy[t] += energies[t] as f64;
            self.sum_link_overlap[t] += link_overlaps[t] as f64;
        }

        while self.checkpoints.get(self.next_ckpt_idx) == Some(&self.count) {
            let [start_energy, start_link_overlap] = &self.window_starts[self.next_ckpt_idx];
            let width = (self.count - Self::window_start(self.count)) as f64;
            let window_avg = |sums: &[f64], starts: &[f64]| -> Vec<f64> {
                sums.iter()
                    .zip(starts)
                    .map(|(sum, start)| (sum - start) / width)
                    .collect()
            };
            self.snapshots.push(EquilCheckpoint {
                sweep: self.count,
                energy_avg: window_avg(&self.sum_energy, start_energy),
                link_overlap_avg: window_avg(&self.sum_link_overlap, start_link_overlap),
            });
            self.next_ckpt_idx += 1;
        }
        self.open_windows();
    }

    pub fn finish(self) -> Vec<EquilCheckpoint> {
        self.snapshots
    }
}

#[cfg(test)]
mod tests {
    use super::EquilDiagnosticAccum;

    /// Energy equal to the 1-based sweep index; link overlap equal to its square.
    fn run(n_sweeps: usize) -> Vec<(usize, f64, f64)> {
        let mut accum = EquilDiagnosticAccum::new(1, n_sweeps);
        for sweep in 1..=n_sweeps {
            let value = sweep as f32;
            accum.push(&[value], &[value * value]);
        }
        accum
            .finish()
            .into_iter()
            .map(|c| (c.sweep, c.energy_avg[0], c.link_overlap_avg[0]))
            .collect()
    }

    fn window_mean(lo: usize, hi: usize, f: impl Fn(f64) -> f64) -> f64 {
        (lo..=hi).map(|s| f(s as f64)).sum::<f64>() / (hi - lo + 1) as f64
    }

    #[test]
    fn checkpoints_average_the_last_half_window() {
        let got = run(600);
        let sweeps: Vec<usize> = got.iter().map(|c| c.0).collect();
        assert_eq!(sweeps, vec![128, 256, 512, 600]);
        for (sweep, energy, link) in got {
            let lo = sweep / 2 + 1;
            assert_eq!(energy, window_mean(lo, sweep, |s| s), "sweep {sweep}");
            assert_eq!(link, window_mean(lo, sweep, |s| s * s), "sweep {sweep}");
        }
        // The window [65, 128] excludes early sweeps: mean 96.5, not the cumulative 64.5.
        assert_eq!(run(128)[0].1, 96.5);
    }

    #[test]
    fn short_and_overlapping_windows() {
        assert_eq!(run(1), vec![(1, 1.0, 1.0)]);
        assert_eq!(run(3), vec![(3, 2.5, 6.5)]);
        // Windows (64, 128] and (64, 129] open at the same sweep.
        let got = run(129);
        assert_eq!(got[0], (128, 96.5, window_mean(65, 128, |s| s * s)));
        assert_eq!(got[1], (129, 97.0, window_mean(65, 129, |s| s * s)));
        assert!(run(0).is_empty());
    }
}
