//! O(2) kernels for finite signed, zero-field nearest-neighbor interactions.
use super::model::Spin;
use crate::{clusters::fk::Embedding, geometry::Lattice, parallel::par_over_replicas};
use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;

pub type XySpin = [f64; 2];
#[inline]
pub fn dot(a: XySpin, b: XySpin) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
#[inline]
pub fn cross(a: XySpin, b: XySpin) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

pub fn interaction(lattice: &Lattice, spins: &[XySpin], couplings: &[f64]) -> f64 {
    let mut energy = 0.0;
    for i in 0..lattice.n_spins {
        for d in 0..lattice.n_neighbors {
            energy += couplings[i * lattice.n_neighbors + d]
                * dot(spins[i], spins[lattice.neighbor_fwd(i, d)]);
        }
    }
    energy
}

pub fn local_field(lattice: &Lattice, spins: &[XySpin], couplings: &[f64], site: usize) -> XySpin {
    let mut field = [0.0; 2];
    for d in 0..lattice.n_neighbors {
        let f = lattice.neighbor_fwd(site, d);
        let b = lattice.neighbor_bwd(site, d);
        let jf = couplings[site * lattice.n_neighbors + d];
        let jb = couplings[b * lattice.n_neighbors + d];
        for (k, component) in field.iter_mut().enumerate() {
            *component += jf * spins[f][k] + jb * spins[b][k];
        }
    }
    field
}

/// `local_field` for a non-boundary site of a canonical 2D lattice, whose
/// directions have strides `width` and 1, with the same summation order.
#[inline]
fn square_interior_field(spins: &[XySpin], couplings: &[f64], site: usize, width: usize) -> XySpin {
    let (up, down, right, left) = (site + width, site - width, site + 1, site - 1);
    let (j_up, j_down) = (couplings[site * 2], couplings[down * 2]);
    let (j_right, j_left) = (couplings[site * 2 + 1], couplings[left * 2 + 1]);
    let mut field = [0.0; 2];
    for (k, component) in field.iter_mut().enumerate() {
        *component += j_up * spins[up][k] + j_down * spins[down][k];
        *component += j_right * spins[right][k] + j_left * spins[left][k];
    }
    field
}

/// Physical H(new)-H(old) for a local proposal.
pub fn energy_change(
    lattice: &Lattice,
    spins: &[XySpin],
    couplings: &[f64],
    site: usize,
    proposal: XySpin,
) -> f64 {
    let h = local_field(lattice, spins, couplings, site);
    dot(spins[site], h) - dot(proposal, h)
}

/// Uniform point `(x, y, x²+y²)` in the punctured open unit disk.
#[inline]
fn disk_point(rng: &mut Xoshiro256StarStar) -> (f64, f64, f64) {
    loop {
        let x = 2.0 * rng.gen::<f64>() - 1.0;
        let y = 2.0 * rng.gen::<f64>() - 1.0;
        let r2 = x * x + y * y;
        if r2 < 1.0 && r2 > 0.0 {
            return (x, y, r2);
        }
    }
}

/// Trig-free uniform unit vector: a uniform disk point has a uniform polar
/// angle, hence so does its doubled angle `(x²-y², 2xy)/r²`.
#[inline]
pub(crate) fn random_unit(rng: &mut Xoshiro256StarStar) -> XySpin {
    let (x, y, r2) = disk_point(rng);
    let inverse = 1.0 / r2;
    [(x * x - y * y) * inverse, 2.0 * x * y * inverse]
}

/// Visit every site in index order with its local field, skipping vacancies.
#[inline(always)]
fn sweep_sites(
    lattice: &Lattice,
    spins: &mut [XySpin],
    couplings: &[f64],
    occupied: Option<&[bool]>,
    mut update: impl FnMut(&mut [XySpin], usize, XySpin),
) {
    let square = lattice
        .square_shape()
        .filter(|&(height, width)| height >= 3 && width >= 3);
    let width = square.map_or(0, |(_, width)| width);
    let mut visit = |spins: &mut [XySpin], i: usize, interior: bool| {
        if occupied.is_some_and(|mask| !mask[i]) {
            return;
        }
        let h = if interior {
            square_interior_field(spins, couplings, i, width)
        } else {
            local_field(lattice, spins, couplings, i)
        };
        update(spins, i, h);
    };
    let Some((height, _)) = square else {
        for i in 0..lattice.n_spins {
            visit(spins, i, false);
        }
        return;
    };
    for i in 0..width {
        visit(spins, i, false);
    }
    for row in 1..height - 1 {
        let start = row * width;
        visit(spins, start, false);
        for i in start + 1..start + width - 1 {
            visit(spins, i, true);
        }
        visit(spins, start + width - 1, false);
    }
    for i in (height - 1) * width..height * width {
        visit(spins, i, false);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LocalMove {
    Metropolis,
    Overrelaxation,
}

/// One pass of `kind` over every system. `occupied: None` means no vacancies;
/// vacant sites are skipped since they have no bonds and no measured weight.
#[allow(clippy::too_many_arguments)]
pub(crate) fn local_sweep(
    lattice: &Lattice,
    spins: &mut [XySpin],
    couplings: &[f64],
    temperatures: &[f64],
    ids: &[usize],
    rngs: &mut [Xoshiro256StarStar],
    occupied: Option<&[bool]>,
    sequential: bool,
    kind: LocalMove,
) {
    par_over_replicas(
        spins,
        rngs,
        temperatures,
        ids,
        lattice.n_spins,
        sequential,
        |s, rng, t, _, _| {
            let beta = 1.0 / t;
            match kind {
                LocalMove::Metropolis => sweep_sites(lattice, s, couplings, occupied, |s, i, h| {
                    let proposal = random_unit(rng);
                    let delta = dot(s[i], h) - dot(proposal, h);
                    if delta <= 0.0 || rng.gen::<f64>() < (-delta * beta).exp() {
                        s[i] = proposal;
                    }
                }),
                LocalMove::Overrelaxation => {
                    sweep_sites(lattice, s, couplings, occupied, |s, i, h| {
                        let h2 = dot(h, h);
                        if h2.is_normal() {
                            let scale = 2.0 * dot(s[i], h) / h2;
                            s[i] = [scale * h[0] - s[i][0], scale * h[1] - s[i][1]];
                            return;
                        }
                        // hypot avoids squaring very large/small fields before normalization.
                        let norm = h[0].hypot(h[1]);
                        if norm == 0.0 {
                            return;
                        }
                        let axis = [h[0] / norm, h[1] / norm];
                        let projection = dot(s[i], axis);
                        s[i] = [
                            2.0 * projection * axis[0] - s[i][0],
                            2.0 * projection * axis[1] - s[i][1],
                        ];
                    })
                }
            }
        },
    );
}

pub(crate) struct XyEmbedding;
impl Embedding for XyEmbedding {
    type Spin = XySpin;
    type Axis = XySpin;
    fn axis(rng: &mut Xoshiro256StarStar) -> XySpin {
        XySpin::random(rng)
    }
    fn bond(
        a: XySpin,
        b: XySpin,
        j: f64,
        t: f64,
        axis: &XySpin,
        rng: &mut Xoshiro256StarStar,
    ) -> bool {
        let inter = j * dot(a, *axis) * dot(b, *axis);
        inter > 0.0 && rng.gen::<f64>() < -(-2.0 * inter / t).exp_m1()
    }
    fn reflect(spin: &mut XySpin, axis: &XySpin) {
        let projection = dot(*spin, *axis);
        spin[0] -= 2.0 * projection * axis[0];
        spin[1] -= 2.0 * projection * axis[1];
    }
    fn coin(rng: &mut Xoshiro256StarStar) -> bool {
        rng.gen::<f64>() < 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        clusters::fk::embedded_update, config::ClusterAction,
        simulation::realization::XyRealization,
    };
    use rand::SeedableRng;

    fn sweep(lattice: &Lattice, real: &mut XyRealization, kind: LocalMove) {
        local_sweep(
            lattice,
            &mut real.spins,
            &real.couplings,
            &real.temperatures,
            &real.system_ids,
            &mut real.rngs,
            None,
            true,
            kind,
        );
    }

    /// Kernels: 0 Metropolis, 1 SW, 2 Wolff.
    fn apply(lattice: &Lattice, real: &mut XyRealization, kernel: usize) {
        match kernel {
            0 => sweep(lattice, real, LocalMove::Metropolis),
            _ => embedded_update::<XyEmbedding>(
                lattice,
                &mut real.spins,
                &real.couplings,
                &real.temperatures,
                &real.system_ids,
                &mut real.rngs,
                kernel == 2,
                ClusterAction::Update,
                None,
                None,
                true,
                None,
            ),
        }
    }

    #[test]
    fn local_delta_rotation_norms_and_overrelaxation() {
        let lattice = Lattice::new(vec![4, 4]);
        let couplings = (0..32).map(|i| (i as f64 - 13.0) / 17.0).collect();
        let mut real = XyRealization::new(&lattice, couplings, &[0.8], 1, 13);
        let initial = interaction(&lattice, &real.spins, &real.couplings);
        let proposal = [0.6, 0.8];
        let before = real.spins[3];
        let delta = energy_change(&lattice, &real.spins, &real.couplings, 3, proposal);
        real.spins[3] = proposal;
        assert!(
            (initial - interaction(&lattice, &real.spins, &real.couplings) - delta).abs() < 1e-12
        );
        real.spins[3] = before;
        for s in &mut real.spins {
            *s = [0.6 * s[0] - 0.8 * s[1], 0.8 * s[0] + 0.6 * s[1]];
        }
        assert!((interaction(&lattice, &real.spins, &real.couplings) - initial).abs() < 1e-12);
        for _ in 0..100 {
            sweep(&lattice, &mut real, LocalMove::Overrelaxation);
        }
        assert!((interaction(&lattice, &real.spins, &real.couplings) - initial).abs() < 1e-10);
        assert!(real.spins.iter().all(|s| (dot(*s, *s) - 1.0).abs() < 1e-12));
        let original = real.spins.clone();
        real.couplings.fill(0.0);
        sweep(&lattice, &mut real, LocalMove::Overrelaxation);
        assert_eq!(original, real.spins);
    }

    fn pair_quadrature(beta_j: f64, n: usize) -> f64 {
        let mut z = 0.0;
        let mut c = 0.0;
        for i in 0..n {
            let cosine = (std::f64::consts::TAU * i as f64 / n as f64).cos();
            let weight = (beta_j * cosine).exp();
            z += weight;
            c += weight * cosine;
        }
        c / z
    }
    #[test]
    fn signed_pairs_match_bessel_integral_for_each_kernel() {
        let lattice = Lattice::new(vec![3, 3]);
        for j in [-1.0, 1.0] {
            let exact = pair_quadrature(j / 0.8, 2048);
            assert!((exact - pair_quadrature(j / 0.8, 1024)).abs() < 1e-13);
            for kernel in 0..3 {
                let mut couplings = vec![0.0; 18];
                couplings[0] = j;
                let mut real =
                    XyRealization::new(&lattice, couplings, &[0.8], 1, 712 + kernel as u64);
                let mut sum = 0.0;
                for sweep in 0..302000 {
                    apply(&lattice, &mut real, kernel);
                    if sweep >= 2000 {
                        sum += dot(real.spins[0], real.spins[3]);
                    }
                }
                let measured = sum / 300000.0;
                assert!(
                    (measured - exact).abs() < 0.018,
                    "J={j} kernel={kernel}: {measured} vs {exact}"
                );
            }
        }
    }

    fn frustrated_quadrature(n: usize) -> [f64; 3] {
        let vectors: Vec<_> = (0..n)
            .map(|i| {
                let (s, c) = (std::f64::consts::TAU * i as f64 / n as f64).sin_cos();
                [c, s]
            })
            .collect();
        let mut z = 0.0;
        let mut sums = [0.0; 3];
        for &a in &vectors {
            for &b in &vectors {
                for &c in &vectors {
                    let inter = a[0] + dot(a, b) + dot(b, c) - c[0];
                    let weight = (inter / 1.2).exp();
                    z += weight;
                    for (total, value) in sums.iter_mut().zip([-inter, a[0], c[0]]) {
                        *total += weight * value;
                    }
                }
            }
        }
        sums.map(|v| v / z)
    }
    #[test]
    fn frustrated_plaquette_matches_converged_angular_quadrature() {
        let exact = frustrated_quadrature(32);
        let finer = frustrated_quadrature(48);
        assert!(exact.iter().zip(finer).all(|(a, b)| (a - b).abs() < 1e-11));
        let lattice = Lattice::new(vec![3, 3]);
        for kernel in 0..3 {
            let mut couplings = vec![0.0; 18];
            for (i, j) in [(0, 1.0), (7, 1.0), (2, 1.0), (1, -1.0)] {
                couplings[i] = j;
            }
            let mut real = XyRealization::new(&lattice, couplings, &[1.2], 1, 951 + kernel as u64);
            let mut sums = [0.0; 3];
            for sweep in 0..82000 {
                apply(&lattice, &mut real, kernel);
                if sweep < 2000 {
                    continue;
                }
                let values = [
                    -interaction(&lattice, &real.spins, &real.couplings),
                    dot(real.spins[0], real.spins[3]),
                    dot(real.spins[0], real.spins[1]),
                ];
                for (sum, value) in sums.iter_mut().zip(values) {
                    *sum += value;
                }
            }
            for (measured, expected) in sums.map(|v| v / 80000.0).iter().zip(exact) {
                assert!(
                    (measured - expected).abs() < 0.027,
                    "kernel={kernel}: {measured} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn random_unit_is_uniform_on_the_circle() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(2024);
        let (n, bins) = (256_000, 64);
        let mut counts = vec![0u32; bins];
        for _ in 0..n {
            let s = random_unit(&mut rng);
            assert!((dot(s, s) - 1.0).abs() < 1e-15);
            let fraction = s[1].atan2(s[0]) / std::f64::consts::TAU + 0.5;
            counts[((fraction * bins as f64) as usize).min(bins - 1)] += 1;
        }
        let expected = n as f64 / bins as f64;
        let chi2: f64 = counts
            .iter()
            .map(|&c| (c as f64 - expected).powi(2) / expected)
            .sum();
        // 63 degrees of freedom: the 0.9999 quantile is about 110.
        assert!(chi2 < 110.0, "chi2 = {chi2}");
    }

    #[test]
    fn square_specialization_matches_generic_trajectory() {
        let square = Lattice::new(vec![5, 7]);
        let generic = Lattice::with_offsets(vec![5, 7], crate::geometry::hypercubic(2));
        assert!(square.square_shape().is_some() && generic.square_shape().is_none());
        let mut rng = Xoshiro256StarStar::seed_from_u64(3);
        let couplings: Vec<f64> = (0..70).map(|_| 2.0 * rng.gen::<f64>() - 0.5).collect();
        let mask: Vec<bool> = (0..35).map(|i| i % 6 != 4).collect();
        let bits = |real: &XyRealization| -> Vec<u64> {
            real.spins.iter().flatten().map(|v| v.to_bits()).collect()
        };
        for occupied in [None, Some(&mask[..])] {
            let mut a = XyRealization::new(&square, couplings.clone(), &[0.4, 1.3], 2, 5);
            let mut b = XyRealization::new(&generic, couplings.clone(), &[0.4, 1.3], 2, 5);
            let initial = a.spins.clone();
            for kind in [
                LocalMove::Metropolis,
                LocalMove::Overrelaxation,
                LocalMove::Metropolis,
            ] {
                for (lattice, real) in [(&square, &mut a), (&generic, &mut b)] {
                    local_sweep(
                        lattice,
                        &mut real.spins,
                        &real.couplings,
                        &real.temperatures,
                        &real.system_ids,
                        &mut real.rngs,
                        occupied,
                        true,
                        kind,
                    );
                }
                assert_eq!(
                    bits(&a),
                    bits(&b),
                    "{kind:?}, masked={}",
                    occupied.is_some()
                );
            }
            for (i, (now, before)) in a.spins.iter().zip(&initial).enumerate() {
                let vacant = occupied.is_some_and(|m| !m[i % 35]);
                assert_eq!(now == before, vacant, "site {i}");
            }
        }
    }

    #[test]
    fn overrelaxation_conserves_energy_at_extreme_field_scales() {
        let lattice = Lattice::new(vec![4, 4]);
        // Fields overflow (1e200) or underflow (1e-200) when squared; both use hypot.
        for scale in [1e-200, 1.0, 1e200] {
            let couplings = (0..32).map(|i| scale * (i as f64 - 13.0) / 17.0).collect();
            let mut real = XyRealization::new(&lattice, couplings, &[0.8], 1, 13);
            let initial = interaction(&lattice, &real.spins, &real.couplings);
            let before = real.spins.clone();
            for _ in 0..50 {
                sweep(&lattice, &mut real, LocalMove::Overrelaxation);
            }
            let energy = interaction(&lattice, &real.spins, &real.couplings);
            assert!((energy - initial).abs() < 1e-10 * scale, "scale={scale}");
            assert!(real.spins.iter().all(|s| (dot(*s, *s) - 1.0).abs() < 1e-12));
            assert_ne!(real.spins, before);
        }
    }

    #[test]
    fn bipartite_gauge_mapping_preserves_energy_and_bonds() {
        let lattice = Lattice::new(vec![4, 6]);
        let ferro = vec![1.0; 48];
        let anti = vec![-1.0; 48];
        let mut rng = Xoshiro256StarStar::seed_from_u64(65);
        let original: Vec<_> = (0..24).map(|_| XySpin::random(&mut rng)).collect();
        let transformed: Vec<_> = original
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                if (i / 6 + i % 6) % 2 == 0 {
                    s
                } else {
                    [-s[0], -s[1]]
                }
            })
            .collect();
        assert!(
            (interaction(&lattice, &original, &ferro) - interaction(&lattice, &transformed, &anti))
                .abs()
                < 1e-12
        );
        let axis = XySpin::random(&mut rng);
        for i in 0..24 {
            for d in 0..2 {
                let j = lattice.neighbor_fwd(i, d);
                assert!(
                    (dot(original[i], original[j]) + dot(transformed[i], transformed[j])).abs()
                        < 1e-12
                );
                let mut a = rng.clone();
                let mut b = rng.clone();
                assert_eq!(
                    XyEmbedding::bond(original[i], original[j], 1.0, 0.7, &axis, &mut a),
                    XyEmbedding::bond(transformed[i], transformed[j], -1.0, 0.7, &axis, &mut b)
                );
            }
        }
    }
}
