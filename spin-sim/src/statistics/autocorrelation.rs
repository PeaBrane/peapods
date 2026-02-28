/// Streaming autocorrelation accumulator using a ring buffer.
///
/// Computes the normalized autocorrelation function Γ(δ) for a per-temperature
/// time series without storing the full history. Memory is O(max_lag × n_temps).
pub struct AutocorrAccum {
    max_lag: usize,
    n_temps: usize,
    /// Ring buffer of recent values, shape [n_temps][max_lag].
    ring: Vec<Vec<f32>>,
    /// Running sum of o, shape [n_temps].
    sum_o: Vec<f64>,
    /// Running sum of o², shape [n_temps].
    sum_o2: Vec<f64>,
    /// Running sum of o(t)·o(t−δ), shape [n_temps][max_lag+1].
    sum_prod: Vec<Vec<f64>>,
    /// Total number of values pushed so far.
    n_recorded: usize,
    /// Current position in the ring buffer.
    ring_pos: usize,
}

impl AutocorrAccum {
    pub fn new(max_lag: usize, n_temps: usize) -> Self {
        Self {
            max_lag,
            n_temps,
            ring: (0..n_temps).map(|_| vec![0.0f32; max_lag]).collect(),
            sum_o: vec![0.0; n_temps],
            sum_o2: vec![0.0; n_temps],
            sum_prod: (0..n_temps).map(|_| vec![0.0; max_lag + 1]).collect(),
            n_recorded: 0,
            ring_pos: 0,
        }
    }

    #[allow(clippy::needless_range_loop)]
    pub fn push(&mut self, values: &[f64]) {
        let pos = self.ring_pos;
        let ml = self.max_lag;
        for t in 0..self.n_temps {
            let o = values[t] as f32;
            self.ring[t][pos % ml] = o;
            self.sum_o[t] += o as f64;
            self.sum_o2[t] += (o as f64) * (o as f64);

            let n_back = self.n_recorded.min(ml);
            for delta in 0..=n_back {
                let idx = if pos >= delta {
                    pos - delta
                } else {
                    pos + ml - delta
                } % ml;
                self.sum_prod[t][delta] += o as f64 * self.ring[t][idx] as f64;
            }
        }
        self.n_recorded += 1;
        self.ring_pos = (pos + 1) % ml;
    }

    pub fn finish(&self) -> Vec<Vec<f64>> {
        let m = self.n_recorded as f64;
        (0..self.n_temps)
            .map(|t| {
                let mean = self.sum_o[t] / m;
                let var = self.sum_o2[t] / m - mean * mean;
                (0..=self.max_lag)
                    .map(|delta| {
                        let count = self.n_recorded.saturating_sub(delta) as f64;
                        if count <= 0.0 || var <= 0.0 {
                            return if delta == 0 { 1.0 } else { 0.0 };
                        }
                        (self.sum_prod[t][delta] / count - mean * mean) / var
                    })
                    .collect()
            })
            .collect()
    }
}

pub fn sokal_tau(gamma: &[f64]) -> f64 {
    let mut tau = 0.5;
    for (w, &g) in gamma.iter().enumerate().skip(1) {
        tau += g;
        if w as f64 >= 5.0 * tau {
            return tau;
        }
    }
    tau
}
