//! Error bars for correlated time series (binning) and iid samples.

pub struct Est {
    pub mean: f64,
    pub se: f64,
    pub tau: f64,
}

/// Blocking analysis: SE is the maximum over block sizes 2^k with at least 32 blocks;
/// tau_int = N SE^2 / (2 Var).
pub fn binning(x: &[f64]) -> Est {
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n.max(2) - 1) as f64;
    let mut best = (var / n as f64).sqrt();
    let mut b = 2usize;
    while n / b >= 32 {
        let nb = n / b;
        let means: Vec<f64> = (0..nb).map(|i| x[i * b..(i + 1) * b].iter().sum::<f64>() / b as f64).collect();
        let m = means.iter().sum::<f64>() / nb as f64;
        let v = means.iter().map(|y| (y - m).powi(2)).sum::<f64>() / (nb - 1) as f64;
        best = best.max((v / nb as f64).sqrt());
        b *= 2;
    }
    let tau = if var > 0.0 { n as f64 * best * best / (2.0 * var) } else { 0.5 };
    Est { mean, se: best, tau }
}

/// Combine independent chains: mean of chain means; SE = max(binning-combined, between-chain).
pub fn combine_chains(chains: &[Est]) -> Est {
    let c = chains.len() as f64;
    let mean = chains.iter().map(|e| e.mean).sum::<f64>() / c;
    let se_in = chains.iter().map(|e| e.se * e.se).sum::<f64>().sqrt() / c;
    let se_between = if chains.len() > 1 {
        (chains.iter().map(|e| (e.mean - mean).powi(2)).sum::<f64>() / (c - 1.0) / c).sqrt()
    } else {
        0.0
    };
    let tau = chains.iter().map(|e| e.tau).sum::<f64>() / c;
    Est { mean, se: se_in.max(se_between), tau }
}

#[derive(Clone, Default)]
pub struct Acc {
    pub n: u64,
    pub s1: f64,
    pub s2: f64,
}

impl Acc {
    pub fn add(&mut self, x: f64) {
        self.n += 1;
        self.s1 += x;
        self.s2 += x * x;
    }
    pub fn merge(&mut self, o: &Acc) {
        self.n += o.n;
        self.s1 += o.s1;
        self.s2 += o.s2;
    }
    pub fn mean(&self) -> f64 {
        self.s1 / self.n as f64
    }
    pub fn se(&self) -> f64 {
        let n = self.n as f64;
        let m = self.mean();
        ((self.s2 / n - m * m).max(0.0) / (n - 1.0).max(1.0)).sqrt()
    }
}
