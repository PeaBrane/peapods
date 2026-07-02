use crate::config::AutocorrelationBackend;
use rustfft::num_complex::Complex64;
use rustfft::FftPlanner;

enum AutocorrStorage {
    Ring {
        ring_len: usize,
        ring: Vec<Vec<f32>>,
        sum_prod: Vec<Vec<f64>>,
        ring_pos: usize,
    },
    Fft {
        series: Vec<Vec<f32>>,
    },
}

/// Streaming autocorrelation accumulator.
///
/// [`AutocorrAccum::new`] uses the exact bounded-memory ring backend. The FFT
/// backend is available through simulation configuration and retains the full
/// measurement history, using O(n_recorded * n_temps) memory.
pub struct AutocorrAccum {
    max_lag: usize,
    n_temps: usize,
    sum_o: Vec<f64>,
    sum_o2: Vec<f64>,
    n_recorded: usize,
    storage: AutocorrStorage,
}

impl AutocorrAccum {
    pub fn new(max_lag: usize, n_temps: usize) -> Self {
        Self::with_backend(max_lag, n_temps, AutocorrelationBackend::Ring, 0)
    }

    pub(crate) fn with_backend(
        max_lag: usize,
        n_temps: usize,
        backend: AutocorrelationBackend,
        expected_samples: usize,
    ) -> Self {
        let storage = match backend {
            AutocorrelationBackend::Ring => {
                let ring_len = max_lag + 1;
                AutocorrStorage::Ring {
                    ring_len,
                    ring: (0..n_temps).map(|_| vec![0.0; ring_len]).collect(),
                    sum_prod: (0..n_temps).map(|_| vec![0.0; max_lag + 1]).collect(),
                    ring_pos: 0,
                }
            }
            AutocorrelationBackend::Fft => AutocorrStorage::Fft {
                series: (0..n_temps)
                    .map(|_| Vec::with_capacity(expected_samples))
                    .collect(),
            },
        };

        Self {
            max_lag,
            n_temps,
            sum_o: vec![0.0; n_temps],
            sum_o2: vec![0.0; n_temps],
            n_recorded: 0,
            storage,
        }
    }

    #[allow(clippy::needless_range_loop)]
    pub fn push(&mut self, values: &[f64]) {
        for t in 0..self.n_temps {
            let o = values[t] as f32;
            self.sum_o[t] += o as f64;
            self.sum_o2[t] += (o as f64) * (o as f64);
        }

        match &mut self.storage {
            AutocorrStorage::Ring {
                ring_len,
                ring,
                sum_prod,
                ring_pos,
            } => {
                let pos = *ring_pos;
                let n_back = self.n_recorded.min(self.max_lag);
                for t in 0..self.n_temps {
                    let o = values[t] as f32;
                    let temp_ring = &mut ring[t];
                    let temp_sum_prod = &mut sum_prod[t];
                    temp_ring[pos] = o;

                    let no_wrap = pos.min(n_back);
                    for delta in 0..=no_wrap {
                        temp_sum_prod[delta] += o as f64 * temp_ring[pos - delta] as f64;
                    }
                    for delta in pos + 1..=n_back {
                        temp_sum_prod[delta] +=
                            o as f64 * temp_ring[pos + *ring_len - delta] as f64;
                    }
                }
                *ring_pos = (pos + 1) % *ring_len;
            }
            AutocorrStorage::Fft { series } => {
                for t in 0..self.n_temps {
                    series[t].push(values[t] as f32);
                }
            }
        }

        self.n_recorded += 1;
    }

    pub fn finish(&self) -> Vec<Vec<f64>> {
        if self.n_recorded == 0 {
            return self.degenerate_gamma();
        }

        match &self.storage {
            AutocorrStorage::Ring { sum_prod, .. } => (0..self.n_temps)
                .map(|t| self.normalize_products(t, |delta| sum_prod[t][delta]))
                .collect(),
            AutocorrStorage::Fft { series } => self.finish_fft(series),
        }
    }

    fn finish_fft(&self, series: &[Vec<f32>]) -> Vec<Vec<f64>> {
        let fft_len = self
            .n_recorded
            .checked_mul(2)
            .and_then(usize::checked_next_power_of_two)
            .expect("autocorrelation series is too large for FFT padding");
        let mut planner = FftPlanner::<f64>::new();
        let forward = planner.plan_fft_forward(fft_len);
        let inverse = planner.plan_fft_inverse(fft_len);
        let scratch_len = forward
            .get_inplace_scratch_len()
            .max(inverse.get_inplace_scratch_len());
        let mut scratch = vec![Complex64::default(); scratch_len];
        let mut spectrum = vec![Complex64::default(); fft_len];

        (0..self.n_temps)
            .map(|t| {
                let m = self.n_recorded as f64;
                let mean = self.sum_o[t] / m;
                let var = self.sum_o2[t] / m - mean * mean;
                if var <= 0.0 {
                    return self.degenerate_row();
                }

                spectrum.fill(Complex64::default());
                for (value, &sample) in spectrum.iter_mut().zip(&series[t]) {
                    value.re = sample as f64;
                }
                forward.process_with_scratch(&mut spectrum, &mut scratch);
                for value in &mut spectrum {
                    *value = Complex64::new(value.norm_sqr(), 0.0);
                }
                inverse.process_with_scratch(&mut spectrum, &mut scratch);

                self.normalize_products(t, |delta| spectrum[delta].re / fft_len as f64)
            })
            .collect()
    }

    fn normalize_products(
        &self,
        temp: usize,
        mut sum_product: impl FnMut(usize) -> f64,
    ) -> Vec<f64> {
        let m = self.n_recorded as f64;
        let mean = self.sum_o[temp] / m;
        let var = self.sum_o2[temp] / m - mean * mean;
        if var <= 0.0 {
            return self.degenerate_row();
        }

        (0..=self.max_lag)
            .map(|delta| {
                let count = self.n_recorded.saturating_sub(delta) as f64;
                if count <= 0.0 {
                    return if delta == 0 { 1.0 } else { 0.0 };
                }
                (sum_product(delta) / count - mean * mean) / var
            })
            .collect()
    }

    fn degenerate_gamma(&self) -> Vec<Vec<f64>> {
        (0..self.n_temps).map(|_| self.degenerate_row()).collect()
    }

    fn degenerate_row(&self) -> Vec<f64> {
        (0..=self.max_lag)
            .map(|delta| if delta == 0 { 1.0 } else { 0.0 })
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

#[cfg(test)]
mod tests {
    use super::{sokal_tau, AutocorrAccum};
    use crate::config::AutocorrelationBackend;

    struct LegacyRing {
        max_lag: usize,
        ring_len: usize,
        ring: Vec<Vec<f32>>,
        sum_o: Vec<f64>,
        sum_o2: Vec<f64>,
        sum_prod: Vec<Vec<f64>>,
        n_recorded: usize,
        ring_pos: usize,
    }

    impl LegacyRing {
        fn new(max_lag: usize, n_temps: usize) -> Self {
            let ring_len = max_lag + 1;
            Self {
                max_lag,
                ring_len,
                ring: (0..n_temps).map(|_| vec![0.0; ring_len]).collect(),
                sum_o: vec![0.0; n_temps],
                sum_o2: vec![0.0; n_temps],
                sum_prod: (0..n_temps).map(|_| vec![0.0; max_lag + 1]).collect(),
                n_recorded: 0,
                ring_pos: 0,
            }
        }

        fn push(&mut self, values: &[f64]) {
            let pos = self.ring_pos;
            for (t, &value) in values.iter().enumerate() {
                let o = value as f32;
                self.ring[t][pos] = o;
                self.sum_o[t] += o as f64;
                self.sum_o2[t] += (o as f64) * (o as f64);
                for delta in 0..=self.n_recorded.min(self.max_lag) {
                    let idx = if pos >= delta {
                        pos - delta
                    } else {
                        pos + self.ring_len - delta
                    } % self.ring_len;
                    self.sum_prod[t][delta] += o as f64 * self.ring[t][idx] as f64;
                }
            }
            self.n_recorded += 1;
            self.ring_pos = (pos + 1) % self.ring_len;
        }

        fn finish(&self) -> Vec<Vec<f64>> {
            let m = self.n_recorded as f64;
            self.sum_prod
                .iter()
                .enumerate()
                .map(|(t, products)| {
                    let mean = self.sum_o[t] / m;
                    let var = self.sum_o2[t] / m - mean * mean;
                    products
                        .iter()
                        .enumerate()
                        .map(|(delta, &product)| {
                            let count = self.n_recorded.saturating_sub(delta) as f64;
                            if count <= 0.0 || var <= 0.0 {
                                return if delta == 0 { 1.0 } else { 0.0 };
                            }
                            (product / count - mean * mean) / var
                        })
                        .collect()
                })
                .collect()
        }
    }

    fn deterministic_values(sample: usize) -> [f64; 2] {
        [
            ((sample * 13 % 31) as f32 / 8.0 - 2.0) as f64,
            ((sample * 7 % 23) as f32 / 4.0 - 1.5) as f64,
        ]
    }

    fn brute_force_gamma(series: &[f64], max_lag: usize) -> Vec<f64> {
        let count = series.len() as f64;
        let mean = series.iter().sum::<f64>() / count;
        let variance = series.iter().map(|value| value * value).sum::<f64>() / count - mean * mean;
        (0..=max_lag)
            .map(|delta| {
                let pairs = series.len().saturating_sub(delta);
                if pairs == 0 || variance <= 0.0 {
                    return if delta == 0 { 1.0 } else { 0.0 };
                }
                let product_sum = (delta..series.len())
                    .map(|index| series[index] * series[index - delta])
                    .sum::<f64>();
                (product_sum / pairs as f64 - mean * mean) / variance
            })
            .collect()
    }

    #[test]
    fn streamlined_ring_is_bitwise_equal_across_wraps() {
        let mut current = AutocorrAccum::new(7, 2);
        let mut legacy = LegacyRing::new(7, 2);
        for sample in 0..41 {
            let values = deterministic_values(sample);
            current.push(&values);
            legacy.push(&values);
        }

        for (got, want) in current
            .finish()
            .iter()
            .flatten()
            .zip(legacy.finish().iter().flatten())
        {
            assert_eq!(got.to_bits(), want.to_bits());
        }
    }

    #[test]
    fn empty_and_constant_series_are_degenerate() {
        for backend in [AutocorrelationBackend::Ring, AutocorrelationBackend::Fft] {
            let empty = AutocorrAccum::with_backend(4, 1, backend, 0);
            assert_eq!(empty.finish(), vec![vec![1.0, 0.0, 0.0, 0.0, 0.0]]);

            let mut constant = AutocorrAccum::with_backend(4, 1, backend, 8);
            for _ in 0..8 {
                constant.push(&[3.5]);
            }
            assert_eq!(constant.finish(), vec![vec![1.0, 0.0, 0.0, 0.0, 0.0]]);
        }
    }

    #[test]
    fn fft_matches_ring_gamma_and_tau() {
        let mut ring = AutocorrAccum::with_backend(40, 2, AutocorrelationBackend::Ring, 0);
        let mut fft = AutocorrAccum::with_backend(40, 2, AutocorrelationBackend::Fft, 128);
        for sample in 0..128 {
            let values = deterministic_values(sample);
            ring.push(&values);
            fft.push(&values);
        }

        let ring_gamma = ring.finish();
        let fft_gamma = fft.finish();
        let first_series: Vec<f64> = (0..128)
            .map(|sample| deterministic_values(sample)[0])
            .collect();
        let brute_gamma = brute_force_gamma(&first_series, 40);
        for (got, want) in fft_gamma.iter().flatten().zip(ring_gamma.iter().flatten()) {
            assert!((got - want).abs() < 1e-10, "got {got}, want {want}");
        }
        for (got, want) in fft_gamma[0].iter().zip(brute_gamma) {
            assert!((got - want).abs() < 1e-10, "got {got}, want {want}");
        }
        for (got, want) in fft_gamma
            .iter()
            .map(|gamma| sokal_tau(gamma))
            .zip(ring_gamma.iter().map(|gamma| sokal_tau(gamma)))
        {
            assert!((got - want).abs() < 1e-10, "got {got}, want {want}");
        }
    }
}
