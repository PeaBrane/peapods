//! Exact one-step stationarity checks for replica cluster moves.
//!
//! Every temperature slot is drawn exactly from its Boltzmann law on a tiny torus, one
//! move is applied, and the result is compared with the product law it must preserve:
//! per-slot state histograms and energies, per-temperature energy sums, and the overlap
//! histogram and energy product of every pair of slots.

use super::{overlap_update, rmc_update};
use crate::config::{ClusterAction, ClusterMode, OverlapClusterBuildMode};
use crate::geometry::Lattice;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;

// At T = 3 one step of a red-only CMR grey closure, or of one that re-samples blue
// bonds at the flipped blue cluster, misses the exact energy by more than 10 standard
// errors on the 2x2 torus.
const TEMP: f32 = 3.0;
const TRIALS: usize = 40_000;
/// Tolerance, in standard deviations, of every mean and chi-square check.
const MAX_Z: f64 = 5.0;

#[derive(Clone, Copy, Debug)]
enum Move {
    Overlap(OverlapClusterBuildMode, ClusterMode, bool),
    Rmc(ClusterMode),
}

fn couplings(lattice: &Lattice) -> Vec<f32> {
    const VALUES: [f32; 8] = [0.9, -1.3, 0.4, -0.7, 1.1, -0.2, 0.6, -1.5];
    (0..lattice.n_spins * lattice.n_neighbors)
        .map(|k| VALUES[k % VALUES.len()] * (1.0 + 0.1 * (k / VALUES.len()) as f32))
        .collect()
}

fn bond_sum(lattice: &Lattice, couplings: &[f32], s: &[i8]) -> f64 {
    let mut total = 0.0;
    for i in 0..lattice.n_spins {
        for d in 0..lattice.n_neighbors {
            let j = lattice.neighbor_fwd(i, d);
            total += (couplings[i * lattice.n_neighbors + d] * s[i] as f32 * s[j] as f32) as f64;
        }
    }
    total
}

fn state(n: usize, bits: usize) -> Vec<i8> {
    (0..n)
        .map(|i| if (bits >> i) & 1 == 1 { 1 } else { -1 })
        .collect()
}

fn bits(s: &[i8]) -> usize {
    s.iter()
        .enumerate()
        .filter(|&(_, &x)| x == 1)
        .map(|(i, _)| 1 << i)
        .sum()
}

/// Boltzmann law of one temperature over all `2^n` states.
struct Law {
    p: Vec<f64>,
    cdf: Vec<f64>,
    /// -H / N per state.
    energy: Vec<f64>,
}

impl Law {
    fn new(lattice: &Lattice, couplings: &[f32], temp: f32) -> Self {
        let n = lattice.n_spins;
        let sums: Vec<f64> = (0..1usize << n)
            .map(|b| bond_sum(lattice, couplings, &state(n, b)))
            .collect();
        let weights: Vec<f64> = sums.iter().map(|s| (s / temp as f64).exp()).collect();
        let z: f64 = weights.iter().sum();
        let p: Vec<f64> = weights.iter().map(|w| w / z).collect();
        let cdf = p
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x;
                Some(*acc)
            })
            .collect();
        let energy = sums.iter().map(|s| s / n as f64).collect();
        Self { p, cdf, energy }
    }

    fn draw(&self, rng: &mut Xoshiro256StarStar) -> usize {
        let u: f64 = rng.gen();
        self.cdf.partition_point(|&c| c < u).min(self.p.len() - 1)
    }

    /// Mean and variance of `f(state)`.
    fn moments(&self, f: impl Fn(usize) -> f64) -> (f64, f64) {
        let mean: f64 = self.p.iter().enumerate().map(|(b, p)| p * f(b)).sum();
        let second: f64 = self
            .p
            .iter()
            .enumerate()
            .map(|(b, p)| p * f(b) * f(b))
            .sum();
        (mean, second - mean * mean)
    }
}

/// Wilson-Hilferty normal score of Pearson's statistic; infinite if a bin with zero
/// probability is hit.
fn chi_square_z(counts: &[u64], p: &[f64]) -> f64 {
    let trials = counts.iter().sum::<u64>() as f64;
    let (mut x, mut bins) = (0.0, 0usize);
    for (&observed, &prob) in counts.iter().zip(p) {
        if prob == 0.0 {
            if observed > 0 {
                return f64::INFINITY;
            }
            continue;
        }
        let expected = trials * prob;
        x += (observed as f64 - expected).powi(2) / expected;
        bins += 1;
    }
    let k = (bins - 1) as f64;
    let spread = 2.0 / (9.0 * k);
    ((x / k).cbrt() - (1.0 - spread)) / spread.sqrt()
}

/// Applies one `mv` to exact Boltzmann draws `TRIALS` times and returns every check
/// that misses the product law by more than `MAX_Z` standard deviations.
///
/// Slot `k * temps.len() + t` holds system `system_ids[slot]` at `temps[t]`.
fn stationarity_failures(
    shape: &[usize],
    temps: &[f32],
    n_replicas: usize,
    system_ids: &[usize],
    mv: Move,
) -> Vec<String> {
    let lattice = Lattice::new(shape.to_vec());
    let n = lattice.n_spins;
    let n_temps = temps.len();
    let n_slots = n_replicas * n_temps;
    let couplings = couplings(&lattice);
    let laws: Vec<Law> = temps
        .iter()
        .map(|&t| Law::new(&lattice, &couplings, t))
        .collect();
    let slot_temps = temps.repeat(n_replicas);
    let law = |slot: usize| &laws[slot % n_temps];
    let pairs: Vec<(usize, usize)> = (0..n_slots)
        .flat_map(|a| (a + 1..n_slots).map(move |b| (a, b)))
        .collect();

    let mut draw_rng = Xoshiro256StarStar::seed_from_u64(7);
    let mut rngs: Vec<_> = (0..n_slots)
        .map(|i| Xoshiro256StarStar::seed_from_u64(11 + i as u64))
        .collect();
    let mut top4 = vec![[0u32; 4]; n_temps * (n_replicas / 2)];
    let mut spins = vec![0i8; n_slots * n];
    let mut state_counts = vec![vec![0u64; 1 << n]; n_slots];
    let mut overlap_counts = vec![vec![0u64; n + 1]; pairs.len()];
    let mut energy_products = vec![0.0; pairs.len()];
    let mut temp_energy = vec![0.0; n_temps];
    let mut slot_bits = vec![0usize; n_slots];
    for _ in 0..TRIALS {
        for (slot, &system) in system_ids.iter().enumerate() {
            let b = law(slot).draw(&mut draw_rng);
            spins[system * n..(system + 1) * n].copy_from_slice(&state(n, b));
        }
        match mv {
            Move::Overlap(mode, cluster_mode, with_stats) => overlap_update(
                &lattice,
                &mut spins,
                &couplings,
                &slot_temps,
                system_ids,
                n_replicas,
                n_temps,
                &mut rngs,
                &mode,
                cluster_mode,
                ClusterAction::Update,
                None,
                with_stats.then_some(top4.as_mut_slice()),
                None,
                true,
                None,
                None,
                None,
                None,
            ),
            Move::Rmc(cluster_mode) => rmc_update(
                &lattice,
                &mut spins,
                &couplings,
                &slot_temps,
                system_ids,
                n_replicas,
                n_temps,
                &mut rngs,
                cluster_mode,
                true,
            ),
        }
        for (slot, &system) in system_ids.iter().enumerate() {
            let b = bits(&spins[system * n..(system + 1) * n]);
            slot_bits[slot] = b;
            state_counts[slot][b] += 1;
            temp_energy[slot % n_temps] += law(slot).energy[b];
        }
        for (k, &(a, b)) in pairs.iter().enumerate() {
            let agree = n - (slot_bits[a] ^ slot_bits[b]).count_ones() as usize;
            overlap_counts[k][agree] += 1;
            energy_products[k] += law(a).energy[slot_bits[a]] * law(b).energy[slot_bits[b]];
        }
    }

    let trials = TRIALS as f64;
    let mut failures = Vec::new();
    let mut check = |name: String, z: f64| {
        if z.abs() >= MAX_Z || z.is_nan() {
            failures.push(format!("{mv:?} {shape:?} {name}: z = {z:.1}"));
        }
    };
    for (t, law) in laws.iter().enumerate() {
        let (mean, var) = law.moments(|b| law.energy[b]);
        let (mean, var) = (mean * n_replicas as f64, var * n_replicas as f64);
        let z = (temp_energy[t] / trials - mean) / (var / trials).sqrt();
        check(format!("energy sum at T={}", temps[t]), z);
    }
    for (slot, counts) in state_counts.iter().enumerate() {
        let law = law(slot);
        let (mean, var) = law.moments(|b| law.energy[b]);
        let observed: f64 = counts
            .iter()
            .enumerate()
            .map(|(b, &c)| c as f64 * law.energy[b])
            .sum::<f64>()
            / trials;
        check(
            format!("slot {slot} energy"),
            (observed - mean) / (var / trials).sqrt(),
        );
        check(format!("slot {slot} states"), chi_square_z(counts, &law.p));
    }
    for (k, &(a, b)) in pairs.iter().enumerate() {
        let (la, lb) = (law(a), law(b));
        let mut p_overlap = vec![0.0; n + 1];
        for (x, px) in la.p.iter().enumerate() {
            for (y, py) in lb.p.iter().enumerate() {
                p_overlap[n - (x ^ y).count_ones() as usize] += px * py;
            }
        }
        check(
            format!("slots ({a}, {b}) overlap"),
            chi_square_z(&overlap_counts[k], &p_overlap),
        );
        let (ma, va) = la.moments(|x| la.energy[x]);
        let (mb, vb) = lb.moments(|y| lb.energy[y]);
        let mean = ma * mb;
        let var = (va + ma * ma) * (vb + mb * mb) - mean * mean;
        check(
            format!("slots ({a}, {b}) energy product"),
            (energy_products[k] / trials - mean) / (var / trials).sqrt(),
        );
    }
    failures
}

fn assert_stationary(
    shape: &[usize],
    temps: &[f32],
    n_replicas: usize,
    system_ids: &[usize],
    mv: Move,
) {
    let failures = stationarity_failures(shape, temps, n_replicas, system_ids, mv);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn identity(n_slots: usize) -> Vec<usize> {
    (0..n_slots).collect()
}

fn same_temperature_moves(modes: &[OverlapClusterBuildMode], shape: &[usize], n_replicas: usize) {
    for &mode in modes {
        for cluster_mode in [ClusterMode::Wolff, ClusterMode::Sw] {
            for with_stats in [false, true] {
                assert_stationary(
                    shape,
                    &[TEMP],
                    n_replicas,
                    &identity(n_replicas),
                    Move::Overlap(mode, cluster_mode, with_stats),
                );
            }
        }
    }
}

#[test]
fn overlap_moves_preserve_boltzmann() {
    let modes = [
        OverlapClusterBuildMode::Houdayer(2),
        OverlapClusterBuildMode::Jorg(2),
        OverlapClusterBuildMode::Cmr,
    ];
    for shape in [[2, 2], [3, 2]] {
        same_temperature_moves(&modes, &shape, 2);
    }
}

#[test]
fn four_replica_pair_moves_preserve_boltzmann() {
    let modes = [
        OverlapClusterBuildMode::Pair(4),
        OverlapClusterBuildMode::Jorg(4),
        // Two independent groups at one temperature.
        OverlapClusterBuildMode::Pair(2),
        OverlapClusterBuildMode::Jorg(2),
    ];
    same_temperature_moves(&modes, &[2, 2], 4);
}

#[test]
fn six_replica_pair_moves_preserve_boltzmann() {
    for mode in [
        OverlapClusterBuildMode::Pair(6),
        OverlapClusterBuildMode::Jorg(6),
    ] {
        for cluster_mode in [ClusterMode::Wolff, ClusterMode::Sw] {
            assert_stationary(
                &[3, 2],
                &[TEMP],
                6,
                &identity(6),
                Move::Overlap(mode, cluster_mode, false),
            );
        }
    }
}

#[test]
fn replica_monte_carlo_preserves_boltzmann() {
    // A shuffled permutation checks that temperatures follow slots, not systems.
    let ladders: [(&[f32], usize, &[usize]); 2] = [
        (&[2.0, 3.0], 1, &[0, 1]),
        (&[2.0, 2.6, 3.4], 2, &[4, 0, 2, 1, 5, 3]),
    ];
    for (temps, n_replicas, system_ids) in ladders {
        for cluster_mode in [ClusterMode::Wolff, ClusterMode::Sw] {
            assert_stationary(
                &[2, 2],
                temps,
                n_replicas,
                system_ids,
                Move::Rmc(cluster_mode),
            );
        }
    }
}

/// Positive control for the harness: the experimental balanced-site houd4 move negates
/// all four replicas at a site with two up and two down spins, which does not conserve
/// the summed energy, so it cannot preserve the product law.
#[test]
fn harness_detects_non_stationary_houd4() {
    let failures = stationarity_failures(
        &[2, 2],
        &[TEMP],
        4,
        &identity(4),
        Move::Overlap(OverlapClusterBuildMode::Houdayer(4), ClusterMode::Sw, false),
    );
    assert!(!failures.is_empty());
}
