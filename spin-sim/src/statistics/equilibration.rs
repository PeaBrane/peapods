pub struct EquilCheckpoint {
    pub sweep: usize,
    pub energy_avg: Vec<f64>,
    pub link_overlap_avg: Vec<f64>,
}

pub struct EquilDiagnosticAccum {
    n_temps: usize,
    checkpoints: Vec<usize>,
    next_ckpt_idx: usize,
    count: usize,
    sum_energy: Vec<f64>,
    sum_link_overlap: Vec<f64>,
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

        Self {
            n_temps,
            checkpoints,
            next_ckpt_idx: 0,
            count: 0,
            sum_energy: vec![0.0; n_temps],
            sum_link_overlap: vec![0.0; n_temps],
            snapshots: Vec::new(),
        }
    }

    pub fn push(&mut self, energies: &[f32], link_overlaps: &[f32]) {
        self.count += 1;
        for t in 0..self.n_temps {
            self.sum_energy[t] += energies[t] as f64;
            self.sum_link_overlap[t] += link_overlaps[t] as f64;
        }

        if self.next_ckpt_idx < self.checkpoints.len()
            && self.count == self.checkpoints[self.next_ckpt_idx]
        {
            let c = self.count as f64;
            self.snapshots.push(EquilCheckpoint {
                sweep: self.count,
                energy_avg: self.sum_energy.iter().map(|&s| s / c).collect(),
                link_overlap_avg: self.sum_link_overlap.iter().map(|&s| s / c).collect(),
            });
            self.next_ckpt_idx += 1;
        }
    }

    pub fn finish(self) -> Vec<EquilCheckpoint> {
        self.snapshots
    }
}
