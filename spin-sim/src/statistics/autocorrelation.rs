use crate::config::AutocorrelationBackend;
use crate::spins::model::Real;
use rustfft::num_complex::Complex64;
use rustfft::FftPlanner;

enum AutocorrStorage<T: Real> {
    Ring {
        /// Last `max_lag + 1` shifted samples per temperature.
        ring: Vec<Vec<f64>>,
        /// First `max_lag` shifted samples per temperature.
        head: Vec<Vec<f64>>,
        /// `sum_prod[t][δ] = Σ_i x_i x_{i+δ}` over shifted samples.
        sum_prod: Vec<Vec<f64>>,
        ring_pos: usize,
    },
    Fft {
        series: Vec<Vec<T>>,
    },
}

/// Streaming autocorrelation accumulator.
///
/// Estimates the normalized autocovariance
/// `Γ(δ) = [Σ_{i<n-δ} (o_i - ō)(o_{i+δ} - ō) / (n - δ)] / [Σ_i (o_i - ō)² / n]`.
///
/// [`AutocorrAccum::new`] uses the exact bounded-memory ring backend. The FFT
/// backend is available through simulation configuration and retains the full
/// measurement history, using O(n_recorded * n_temps) memory.
///
/// Samples are rounded to `T` and then shifted by the first sample of each
/// temperature before any moment is accumulated. The autocovariance is
/// shift-invariant, and the shift keeps the accumulated moments of order σ
/// rather than ō, so the centring below does not cancel catastrophically when
/// `|ō| ≫ σ` (e.g. m² below T_c or the XY energy).
pub struct PrecisionAutocorr<T: Real> {
    max_lag: usize,
    n_temps: usize,
    reference: Vec<f64>,
    sum_x: Vec<f64>,
    sum_x2: Vec<f64>,
    n_recorded: usize,
    storage: AutocorrStorage<T>,
}

pub type AutocorrAccum = PrecisionAutocorr<f32>;

impl<T: Real> PrecisionAutocorr<T> {
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
            AutocorrelationBackend::Ring => AutocorrStorage::Ring {
                ring: (0..n_temps).map(|_| vec![0.0; max_lag + 1]).collect(),
                head: (0..n_temps).map(|_| Vec::with_capacity(max_lag)).collect(),
                sum_prod: (0..n_temps).map(|_| vec![0.0; max_lag + 1]).collect(),
                ring_pos: 0,
            },
            AutocorrelationBackend::Fft => AutocorrStorage::Fft {
                series: (0..n_temps)
                    .map(|_| Vec::with_capacity(expected_samples))
                    .collect(),
            },
        };

        Self {
            max_lag,
            n_temps,
            reference: vec![0.0; n_temps],
            sum_x: vec![0.0; n_temps],
            sum_x2: vec![0.0; n_temps],
            n_recorded: 0,
            storage,
        }
    }

    #[allow(clippy::needless_range_loop)]
    pub fn push(&mut self, values: &[f64]) {
        if self.n_recorded == 0 {
            for (reference, &value) in self.reference.iter_mut().zip(values) {
                *reference = T::from_f64(value).to_f64();
            }
        }

        let ring_len = self.max_lag + 1;
        let n_back = self.n_recorded.min(self.max_lag);
        for t in 0..self.n_temps {
            let o = T::from_f64(values[t]);
            let x = o.to_f64() - self.reference[t];
            self.sum_x[t] += x;
            self.sum_x2[t] += x * x;

            match &mut self.storage {
                AutocorrStorage::Ring {
                    ring,
                    head,
                    sum_prod,
                    ring_pos,
                } => {
                    let pos = *ring_pos;
                    let temp_ring = &mut ring[t];
                    let temp_sum_prod = &mut sum_prod[t];
                    temp_ring[pos] = x;
                    if self.n_recorded < self.max_lag {
                        head[t].push(x);
                    }

                    let no_wrap = pos.min(n_back);
                    for delta in 0..=no_wrap {
                        temp_sum_prod[delta] += x * temp_ring[pos - delta];
                    }
                    for delta in pos + 1..=n_back {
                        temp_sum_prod[delta] += x * temp_ring[pos + ring_len - delta];
                    }
                }
                AutocorrStorage::Fft { series } => series[t].push(o),
            }
        }

        if let AutocorrStorage::Ring { ring_pos, .. } = &mut self.storage {
            *ring_pos = (*ring_pos + 1) % ring_len;
        }
        self.n_recorded += 1;
    }

    pub fn finish(&self) -> Vec<Vec<f64>> {
        if self.n_recorded == 0 {
            return self.degenerate_gamma();
        }

        match &self.storage {
            AutocorrStorage::Ring {
                ring,
                head,
                sum_prod,
                ring_pos,
            } => (0..self.n_temps)
                .map(|t| self.finish_ring(t, &ring[t], &head[t], &sum_prod[t], *ring_pos))
                .collect(),
            AutocorrStorage::Fft { series } => self.finish_fft(series),
        }
    }

    /// Population mean and variance of the shifted samples of one temperature.
    fn moments(&self, temp: usize) -> (f64, f64) {
        let n = self.n_recorded as f64;
        let mean = self.sum_x[temp] / n;
        (mean, self.sum_x2[temp] / n - mean * mean)
    }

    /// Centres the raw lag products exactly:
    /// `Σ_{i<n-δ} (x_i - x̄)(x_{i+δ} - x̄) = S(δ) - x̄ (A(δ) + B(δ)) + (n - δ) x̄²`,
    /// where `A(δ)` omits the last δ samples and `B(δ)` omits the first δ.
    fn finish_ring(
        &self,
        temp: usize,
        ring: &[f64],
        head: &[f64],
        sum_prod: &[f64],
        ring_pos: usize,
    ) -> Vec<f64> {
        let (mean, var) = self.moments(temp);
        if var <= 0.0 {
            return self.degenerate_row();
        }
        let n = self.n_recorded;
        let ring_len = ring.len();
        let newest = (ring_pos + ring_len - 1) % ring_len;
        let total = self.sum_x[temp];
        let mut last_sum = 0.0;
        let mut first_sum = 0.0;
        self.normalize(var, |delta| {
            if delta > 0 {
                last_sum += ring[(newest + ring_len - (delta - 1)) % ring_len];
                first_sum += head[delta - 1];
            }
            let pairs = (n - delta) as f64;
            sum_prod[delta] - mean * ((total - last_sum) + (total - first_sum))
                + pairs * mean * mean
        })
    }

    fn finish_fft(&self, series: &[Vec<T>]) -> Vec<Vec<f64>> {
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
                let (mean, var) = self.moments(t);
                if var <= 0.0 {
                    return self.degenerate_row();
                }

                spectrum.fill(Complex64::default());
                for (value, &sample) in spectrum.iter_mut().zip(&series[t]) {
                    value.re = (sample.to_f64() - self.reference[t]) - mean;
                }
                forward.process_with_scratch(&mut spectrum, &mut scratch);
                for value in &mut spectrum {
                    *value = Complex64::new(value.norm_sqr(), 0.0);
                }
                inverse.process_with_scratch(&mut spectrum, &mut scratch);

                self.normalize(var, |delta| spectrum[delta].re / fft_len as f64)
            })
            .collect()
    }

    /// Maps centred lag sums to Γ(δ); lags without any pair are zero.
    fn normalize(&self, var: f64, mut centred_sum: impl FnMut(usize) -> f64) -> Vec<f64> {
        (0..=self.max_lag)
            .map(|delta| {
                if delta >= self.n_recorded {
                    return if delta == 0 { 1.0 } else { 0.0 };
                }
                let pairs = (self.n_recorded - delta) as f64;
                centred_sum(delta) / pairs / var
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

/// Sokal automatic-window estimate of the integrated autocorrelation time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SokalEstimate {
    /// `τ_int = 1/2 + Σ_{δ=1}^{W} Γ(δ)`.
    pub tau: f64,
    /// Window `W` at which the sum stopped.
    pub window: usize,
    /// Whether `W ≥ 5 τ_int` was reached within the available lags. When false,
    /// `tau` is truncated at `max_lag` and is a lower bound in practice; increase
    /// `autocorrelation_max_lag` or the run length.
    pub converged: bool,
}

/// Sokal's self-consistent window `W ≥ 5 τ_int(W)` applied to `Γ`.
pub fn sokal_estimate(gamma: &[f64]) -> SokalEstimate {
    let mut tau = 0.5;
    for (w, &g) in gamma.iter().enumerate().skip(1) {
        tau += g;
        if w as f64 >= 5.0 * tau {
            return SokalEstimate {
                tau,
                window: w,
                converged: true,
            };
        }
    }
    SokalEstimate {
        tau,
        window: gamma.len().saturating_sub(1),
        converged: false,
    }
}

/// τ_int from [`sokal_estimate`]. If the window never closes within the
/// available lags this silently returns the truncated sum; use
/// [`sokal_estimate`] to detect that case.
pub fn sokal_tau(gamma: &[f64]) -> f64 {
    sokal_estimate(gamma).tau
}

#[cfg(test)]
mod tests {
    use super::{sokal_estimate, sokal_tau, AutocorrAccum, PrecisionAutocorr};
    use crate::config::AutocorrelationBackend;
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoshiro256StarStar;

    const BACKENDS: [AutocorrelationBackend; 2] =
        [AutocorrelationBackend::Ring, AutocorrelationBackend::Fft];

    fn deterministic_values(sample: usize) -> [f64; 2] {
        [
            ((sample * 13 % 31) as f32 / 8.0 - 2.0) as f64,
            ((sample * 7 % 23) as f32 / 4.0 - 1.5) as f64,
        ]
    }

    /// Stationary AR(1) `x_{i+1} = φ x_i + ε_i` with unit marginal variance.
    fn ar1(phi: f64, n: usize, offset: f64, seed: u64) -> Vec<f64> {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        let mut gaussian = || {
            let u: f64 = 1.0 - rng.gen::<f64>();
            let v: f64 = rng.gen();
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        };
        let innovation = (1.0 - phi * phi).sqrt();
        let mut x = gaussian();
        (0..n)
            .map(|_| {
                x = phi * x + innovation * gaussian();
                offset + x
            })
            .collect()
    }

    /// Two-pass centred reference estimator.
    fn brute_force_gamma(series: &[f64], max_lag: usize) -> Vec<f64> {
        let count = series.len() as f64;
        let mean = series.iter().sum::<f64>() / count;
        let variance = series.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count;
        (0..=max_lag)
            .map(|delta| {
                let pairs = series.len().saturating_sub(delta);
                if pairs == 0 || variance <= 0.0 {
                    return if delta == 0 { 1.0 } else { 0.0 };
                }
                let product_sum = (delta..series.len())
                    .map(|index| (series[index] - mean) * (series[index - delta] - mean))
                    .sum::<f64>();
                product_sum / pairs as f64 / variance
            })
            .collect()
    }

    fn gamma_of<T: crate::spins::model::Real>(
        series: &[f64],
        max_lag: usize,
        backend: AutocorrelationBackend,
    ) -> Vec<f64> {
        let mut accum = PrecisionAutocorr::<T>::with_backend(max_lag, 1, backend, series.len());
        for &value in series {
            accum.push(&[value]);
        }
        accum.finish().remove(0)
    }

    #[test]
    fn ring_and_fft_match_centred_brute_force_across_wraps() {
        // Large offset relative to σ: the uncentred estimator misses by ~ ō·δ/(nσ).
        let series: Vec<f64> = ar1(0.7, 257, 1.0e3, 3)
            .iter()
            .map(|&v| v as f32 as f64)
            .collect();
        for max_lag in [7, 64, 256] {
            let want = brute_force_gamma(&series, max_lag);
            for backend in BACKENDS {
                let got = gamma_of::<f32>(&series, max_lag, backend);
                for (delta, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (g - w).abs() < 1e-9,
                        "{backend:?} max_lag={max_lag} delta={delta}: got {g}, want {w}"
                    );
                }
            }
        }
    }

    #[test]
    fn ar1_tau_is_recovered_with_large_mean_offset() {
        // τ_int = (1 + φ) / (2 (1 - φ)) = 4.5 with ō/σ = 100; the uncentred
        // estimator averaged τ ≈ 7.6 on these series.
        let phi = 0.8;
        let expected = (1.0 + phi) / (2.0 * (1.0 - phi));
        let n_series = 16;
        for backend in BACKENDS {
            let mut mean_tau = 0.0;
            for seed in 0..n_series {
                let series = ar1(phi, 1 << 13, 100.0, seed);
                let estimate = sokal_estimate(&gamma_of::<f32>(&series, 100, backend));
                assert!(estimate.converged, "{backend:?}: {estimate:?}");
                mean_tau += estimate.tau / n_series as f64;
            }
            assert!(
                (mean_tau - expected).abs() < 0.1 * expected,
                "{backend:?}: mean τ = {mean_tau}, expected {expected}"
            );
        }
    }

    #[test]
    fn gamma_is_shift_invariant_for_huge_offsets() {
        let series = ar1(0.5, 4096, 0.0, 5);
        let shifted: Vec<f64> = series.iter().map(|v| v + 1.0e6).collect();
        for backend in BACKENDS {
            let base = gamma_of::<f64>(&series, 32, backend);
            let moved = gamma_of::<f64>(&shifted, 32, backend);
            for (a, b) in base.iter().zip(&moved) {
                assert!((a - b).abs() < 1e-6, "{backend:?}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn empty_and_constant_series_are_degenerate() {
        for backend in BACKENDS {
            let empty = AutocorrAccum::with_backend(4, 1, backend, 0);
            assert_eq!(empty.finish(), vec![vec![1.0, 0.0, 0.0, 0.0, 0.0]]);

            for value in [3.5, 0.1] {
                let mut constant = AutocorrAccum::with_backend(4, 1, backend, 8);
                for _ in 0..8 {
                    constant.push(&[value]);
                }
                assert_eq!(constant.finish(), vec![vec![1.0, 0.0, 0.0, 0.0, 0.0]]);
            }
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

    #[test]
    fn sokal_estimate_flags_unclosed_windows() {
        let fast = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.0, 0.0, 0.0, 0.0, 0.0];
        let closed = sokal_estimate(&fast);
        assert!(closed.converged);
        assert_eq!(closed.tau, sokal_tau(&fast));

        let slow = [1.0; 10];
        let open = sokal_estimate(&slow);
        assert!(!open.converged);
        assert_eq!(open.window, 9);
        assert_eq!(open.tau, 9.5);
    }
}
