//! Exact Boltzmann laws on tiny tori for stationarity tests.
use crate::geometry::Lattice;
use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;

pub(crate) struct ExactLaw {
    pub states: Vec<Vec<i8>>,
    pub probabilities: Vec<f64>,
    cdf: Vec<f64>,
}

pub(crate) fn bond_sum(lattice: &Lattice, couplings: &[f32], spins: &[i8]) -> f64 {
    let mut total = 0.0;
    for i in 0..lattice.n_spins {
        for d in 0..lattice.n_neighbors {
            let j = lattice.neighbor_fwd(i, d);
            total +=
                f64::from(couplings[i * lattice.n_neighbors + d] * f32::from(spins[i] * spins[j]));
        }
    }
    total
}

impl ExactLaw {
    pub(crate) fn new(lattice: &Lattice, couplings: &[f32], temperature: f32) -> Self {
        let n = lattice.n_spins;
        let states: Vec<Vec<i8>> = (0..1usize << n)
            .map(|bits| {
                (0..n)
                    .map(|i| if (bits >> i) & 1 == 1 { 1 } else { -1 })
                    .collect()
            })
            .collect();
        let weights: Vec<f64> = states
            .iter()
            .map(|s| (bond_sum(lattice, couplings, s) / f64::from(temperature)).exp())
            .collect();
        let z: f64 = weights.iter().sum();
        let probabilities: Vec<f64> = weights.iter().map(|w| w / z).collect();
        let cdf = probabilities
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x;
                Some(*acc)
            })
            .collect();
        Self {
            states,
            probabilities,
            cdf,
        }
    }

    pub(crate) fn sample(&self, rng: &mut Xoshiro256StarStar) -> &[i8] {
        let u: f64 = rng.gen();
        let index = self
            .cdf
            .partition_point(|&c| c < u)
            .min(self.states.len() - 1);
        &self.states[index]
    }

    pub(crate) fn index(spins: &[i8]) -> usize {
        spins
            .iter()
            .enumerate()
            .map(|(i, &s)| usize::from(s == 1) << i)
            .sum()
    }

    /// Pearson chi-square of observed state counts against this law.
    pub(crate) fn chi2(&self, counts: &[usize]) -> f64 {
        let trials: usize = counts.iter().sum();
        counts
            .iter()
            .zip(&self.probabilities)
            .map(|(&observed, &p)| {
                let expected = p * trials as f64;
                (observed as f64 - expected).powi(2) / expected
            })
            .sum()
    }
}

/// Chi-square cutoff ~5 standard deviations above the mean for `dof` degrees of freedom.
pub(crate) fn chi2_cutoff(dof: usize) -> f64 {
    dof as f64 + 5.0 * (2.0 * dof as f64).sqrt()
}
