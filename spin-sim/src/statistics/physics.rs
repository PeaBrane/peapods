//! Physical moments with thermal variances formed before disorder averaging.
use crate::{
    geometry::Lattice,
    spins::{
        model::{Real, Spin},
        xy::{cross, dot},
    },
};
use std::{
    collections::BTreeMap,
    f64::consts::{PI, TAU},
};

#[derive(Clone, Debug, Default)]
pub struct PhysicsOptions {
    pub displacements: Vec<Vec<isize>>,
    pub vortices: bool,
    /// Measured sweeps per block, including all replicas. None disables storage.
    pub block_size: Option<usize>,
}

impl PhysicsOptions {
    /// Checks these options against a lattice without allocating a collector.
    pub fn validate(&self, lattice: &Lattice, components: usize) -> Result<(), String> {
        if self.block_size == Some(0) {
            return Err("block_size must be positive".into());
        }
        if self.vortices && (lattice.n_dims != 2 || components != 2) {
            return Err("angle vortices require a two-dimensional XY lattice".into());
        }
        if self.displacements.iter().any(|r| r.len() != lattice.n_dims) {
            return Err("correlation displacements must match lattice dimensionality".into());
        }
        Ok(())
    }
}

/// Energy and magnetization of one configuration, already known to the caller.
#[derive(Clone, Copy, Debug)]
pub struct CachedTotals {
    /// Physical energy per site, `H/N`.
    pub energy: f64,
    /// Magnetization summed over occupied sites.
    pub magnetization: [f64; 2],
}

#[derive(Clone, Debug)]
pub struct MomentField {
    pub name: &'static str,
    pub width: usize,
    pub offset: usize,
}

#[derive(Clone, Debug)]
pub struct MomentBlock {
    /// Sum over configurations, indexed by temperature then flattened moment.
    pub sums: Vec<Vec<f64>>,
    /// Configurations per temperature, including replicas and any partial block.
    pub count: u64,
    pub sweeps: usize,
}

/// Per-disorder moments averaged over measured configurations.
///
/// `energies` is the physical energy per site `+H/N`, the opposite sign of
/// [`super::SweepResult::energies`] (`-H/N`).
#[derive(Clone, Debug)]
pub struct PhysicsResult {
    pub fields: Vec<MomentField>,
    pub moments: Vec<Vec<f64>>,
    pub count: u64,
    pub blocks: Vec<MomentBlock>,
}

pub type PhysicsValues = BTreeMap<&'static str, Vec<Vec<f64>>>;
impl PhysicsResult {
    pub fn values(
        &self,
        shape: &[usize],
        temperatures: &[f64],
        components: usize,
    ) -> PhysicsValues {
        let n: f64 = shape.iter().product::<usize>() as f64;
        let mut out: PhysicsValues = self
            .fields
            .iter()
            .map(|f| {
                (
                    f.name,
                    self.moments
                        .iter()
                        .map(|row| row[f.offset..f.offset + f.width].to_vec())
                        .collect(),
                )
            })
            .collect();
        let mut heat = Vec::new();
        let mut binder = Vec::new();
        let mut s0 = Vec::new();
        let mut chi = Vec::new();
        let mut helicity = Vec::new();
        let mut xi = Vec::new();
        let mut xi_ratio = Vec::new();
        for (t, &temperature) in temperatures.iter().enumerate() {
            let beta = 1.0 / temperature;
            let m2 = out["mags2"][t][0];
            heat.push(vec![
                beta * beta * n * (out["energies2"][t][0] - out["energies"][t][0].powi(2)),
            ]);
            let denominator = (components as f64 + 2.0) / components as f64 * m2 * m2;
            binder.push(vec![if denominator > 0.0 {
                1.0 - out["mags4"][t][0] / denominator
            } else {
                f64::NAN
            }]);
            s0.push(vec![n * m2]);
            chi.push(vec![beta * n * m2 / components as f64]);
            let mut length = Vec::new();
            let mut ratio = Vec::new();
            let mut stiffness = Vec::new();
            for (d, &extent) in shape.iter().enumerate() {
                let sk = out["structure_factor_min"][t][d];
                let radicand = n * m2 / sk - 1.0;
                let value = if sk > 0.0 && radicand >= 0.0 {
                    radicand.sqrt() / (2.0 * (PI / extent as f64).sin())
                } else {
                    f64::NAN
                };
                length.push(value);
                ratio.push(value / extent as f64);
                if components == 2 {
                    stiffness.push((out["helicity_d"][t][d] - beta * out["helicity_i2"][t][d]) / n);
                }
            }
            helicity.push(stiffness);
            xi.push(length);
            xi_ratio.push(ratio);
        }
        for (name, values) in [
            ("heat_capacity", heat),
            ("binder_cumulant", binder),
            ("structure_factor_0", s0),
            ("susceptibility", chi),
            ("correlation_length", xi),
            ("correlation_length_ratio", xi_ratio),
        ] {
            out.insert(name, values);
        }
        if components == 2 {
            out.insert("helicity_modulus", helicity);
        }
        out
    }

    /// Equal disorder weights for raw moments; heat capacities retain thermal centering.
    pub fn aggregate<'a>(
        results: impl IntoIterator<Item = &'a Self>,
        shape: &[usize],
        temperatures: &[f64],
        components: usize,
    ) -> PhysicsValues {
        let results: Vec<_> = results.into_iter().collect();
        assert!(!results.is_empty());
        let mut pooled = Self {
            fields: results[0].fields.clone(),
            moments: vec![vec![0.0; results[0].moments[0].len()]; temperatures.len()],
            count: 0,
            blocks: vec![],
        };
        for result in &results {
            for (dst, src) in pooled.moments.iter_mut().zip(&result.moments) {
                for (a, b) in dst.iter_mut().zip(src) {
                    *a += b / results.len() as f64;
                }
            }
        }
        let mut out = pooled.values(shape, temperatures, components);
        let mut heat = vec![vec![0.0]; temperatures.len()];
        for result in &results {
            let values = result.values(shape, temperatures, components);
            for (dst, src) in heat.iter_mut().zip(&values["heat_capacity"]) {
                dst[0] += src[0] / results.len() as f64;
            }
        }
        out.insert("heat_capacity", heat);
        out
    }
}

/// Geometry and Fourier phases shared by every measurement of a run.
pub struct PhysicsCollector {
    fields: Vec<MomentField>,
    /// `phases[d][x] = (cos, sin)(2π x / L_d)`.
    phases: Vec<Vec<[f64; 2]>>,
    /// Per-dimension spin sums over the hyperplanes `x_d = x`, `[n_dims][L_d]`.
    planes: Vec<Vec<[f64; 2]>>,
    /// Per-site spin angles for vortex detection; empty unless vortices are enabled.
    angles: Vec<f64>,
    displaced: Vec<Vec<usize>>,
    options: PhysicsOptions,
    components: usize,
    sums: Vec<Vec<f64>>,
    block_sums: Vec<Vec<f64>>,
    /// Temperatures whose block rows are stored; empty means all.
    block_temperatures: Vec<bool>,
    block_count: u64,
    block_sweeps: usize,
    blocks: Vec<MomentBlock>,
    count: u64,
    scratch: Vec<f64>,
}

impl PhysicsCollector {
    pub fn new(
        lattice: &Lattice,
        n_temps: usize,
        options: &PhysicsOptions,
        components: usize,
    ) -> Result<Self, String> {
        options.validate(lattice, components)?;
        let mut fields = Vec::new();
        let mut width = 0;
        let mut add = |name, n| {
            fields.push(MomentField {
                name,
                width: n,
                offset: width,
            });
            width += n;
        };
        for name in ["energies", "energies2", "mags", "mags2", "mags4"] {
            add(name, 1);
        }
        if components == 2 {
            for name in ["helicity_d", "helicity_i", "helicity_i2"] {
                add(name, lattice.n_dims);
            }
        }
        add("structure_factor_min", lattice.n_dims);
        if !options.displacements.is_empty() {
            add("correlations", options.displacements.len());
        }
        if options.vortices {
            add("angle_vortex_density", 1);
            add("intact_plaquette_fraction", 1);
        }
        let phases = lattice
            .shape
            .iter()
            .map(|&extent| {
                (0..extent)
                    .map(|x| {
                        let (s, c) = (TAU * x as f64 / extent as f64).sin_cos();
                        [c, s]
                    })
                    .collect()
            })
            .collect();
        let planes = lattice
            .shape
            .iter()
            .map(|&extent| vec![[0.0; 2]; extent])
            .collect();
        let displaced = options
            .displacements
            .iter()
            .map(|r| {
                (0..lattice.n_spins)
                    .map(|i| {
                        r.iter()
                            .enumerate()
                            .map(|(d, &delta)| {
                                let extent = lattice.shape[d] as isize;
                                let coordinate =
                                    ((i / lattice.strides[d]) % lattice.shape[d]) as isize;
                                ((coordinate + delta.rem_euclid(extent)).rem_euclid(extent)
                                    as usize)
                                    * lattice.strides[d]
                            })
                            .sum()
                    })
                    .collect()
            })
            .collect();
        Ok(Self {
            fields,
            phases,
            planes,
            angles: if options.vortices {
                vec![0.0; lattice.n_spins]
            } else {
                vec![]
            },
            displaced,
            options: options.clone(),
            components,
            sums: vec![vec![0.0; width]; n_temps],
            block_sums: if options.block_size.is_some() {
                vec![vec![0.0; width]; n_temps]
            } else {
                vec![]
            },
            block_temperatures: vec![],
            block_count: 0,
            block_sweeps: 0,
            blocks: vec![],
            count: 0,
            scratch: vec![0.0; width],
        })
    }

    /// Returns (physical energy/site, magnetization squared/site²) for diagnostics.
    pub fn measure<S: Spin>(
        &mut self,
        lattice: &Lattice,
        spins: &[S],
        couplings: &[S::Value],
        occupied: &[bool],
        temperature: usize,
    ) -> [f64; 2] {
        self.measure_impl(lattice, spins, couplings, occupied, temperature, None)
    }

    /// Like [`Self::measure`], but takes energy and magnetization from `cached`
    /// instead of repeating the bond pass. One-component collectors only; XY
    /// helicity moments need the bond pass, so two-component collectors ignore it.
    pub fn measure_cached<S: Spin>(
        &mut self,
        lattice: &Lattice,
        spins: &[S],
        couplings: &[S::Value],
        occupied: &[bool],
        temperature: usize,
        cached: CachedTotals,
    ) -> [f64; 2] {
        debug_assert_eq!(self.components, 1);
        self.measure_impl(
            lattice,
            spins,
            couplings,
            occupied,
            temperature,
            Some(cached),
        )
    }

    fn measure_impl<S: Spin>(
        &mut self,
        lattice: &Lattice,
        spins: &[S],
        couplings: &[S::Value],
        occupied: &[bool],
        temperature: usize,
        cached: Option<CachedTotals>,
    ) -> [f64; 2] {
        let n = lattice.n_spins as f64;
        self.scratch.fill(0.0);
        let i2_offset = 5 + 2 * lattice.n_dims;
        let sk_offset = if self.components == 2 {
            5 + 3 * lattice.n_dims
        } else {
            5
        };
        let (energy, magnetization) = match cached {
            Some(totals) if self.components == 1 => (totals.energy, totals.magnetization),
            _ => self.bond_pass(lattice, spins, couplings, occupied),
        };
        let m2 = dot(magnetization, magnetization) / (n * n);
        self.scratch[..5].copy_from_slice(&[energy, energy * energy, m2.sqrt(), m2, m2 * m2]);
        self.accumulate_planes(lattice, spins, occupied);
        for d in 0..lattice.n_dims {
            if self.components == 2 {
                self.scratch[i2_offset + d] = self.scratch[5 + lattice.n_dims + d].powi(2);
            }
            let mut fourier = [[0.0; 2]; 2];
            for (plane, phase) in self.planes[d].iter().zip(&self.phases[d]) {
                for k in 0..2 {
                    for a in 0..2 {
                        fourier[k][a] += plane[k] * phase[a];
                    }
                }
            }
            self.scratch[sk_offset + d] = fourier.iter().flatten().map(|v| v * v).sum::<f64>() / n;
        }
        let correlation_offset = sk_offset + lattice.n_dims;
        for (r, neighbors) in self.displaced.iter().enumerate() {
            self.scratch[correlation_offset + r] = neighbors
                .iter()
                .enumerate()
                .filter(|&(i, &j)| occupied[i] && occupied[j])
                .map(|(i, &j)| dot(spins[i].components(), spins[j].components()))
                .sum::<f64>()
                / n;
        }
        if self.options.vortices {
            let offset = correlation_offset + self.displaced.len();
            let [density, intact] = self.vortices(lattice, spins, couplings, occupied);
            self.scratch[offset] = density;
            self.scratch[offset + 1] = intact;
        }
        for (dst, &value) in self.sums[temperature].iter_mut().zip(&self.scratch) {
            *dst += value;
        }
        if self.options.block_size.is_some() {
            for (dst, &value) in self.block_sums[temperature].iter_mut().zip(&self.scratch) {
                *dst += value;
            }
        }
        [energy, m2]
    }

    /// Energy per site and magnetization, plus the XY helicity sums in `scratch`.
    fn bond_pass<S: Spin>(
        &mut self,
        lattice: &Lattice,
        spins: &[S],
        couplings: &[S::Value],
        occupied: &[bool],
    ) -> (f64, [f64; 2]) {
        let d_offset = 5;
        let i_offset = 5 + lattice.n_dims;
        let mut magnetization = [0.0; 2];
        let mut interaction = 0.0;
        for i in 0..lattice.n_spins {
            let spin = spins[i].components();
            if occupied[i] {
                magnetization[0] += spin[0];
                magnetization[1] += spin[1];
            }
            for d in 0..lattice.n_neighbors {
                let j = lattice.neighbor_fwd(i, d);
                let coupling = couplings[i * lattice.n_neighbors + d].to_f64();
                let neighbor = spins[j].components();
                let cosine = coupling * dot(spin, neighbor);
                interaction += cosine;
                if self.components == 2 {
                    self.scratch[d_offset + d] += cosine;
                    self.scratch[i_offset + d] += coupling * cross(spin, neighbor);
                }
            }
        }
        (-interaction / lattice.n_spins as f64, magnetization)
    }

    /// Sums occupied spins over each hyperplane `x_d = x`. In row-major order a
    /// period of `stride_d · L_d` sites holds `L_d` contiguous slabs of `stride_d`.
    fn accumulate_planes<S: Spin>(&mut self, lattice: &Lattice, spins: &[S], occupied: &[bool]) {
        for (d, plane) in self.planes.iter_mut().enumerate() {
            plane.fill([0.0; 2]);
            let stride = lattice.strides[d];
            let period = stride * lattice.shape[d];
            for (spin_period, mask_period) in spins
                .chunks_exact(period)
                .zip(occupied.chunks_exact(period))
            {
                if stride == 1 {
                    for (sum, (spin, &occ)) in
                        plane.iter_mut().zip(spin_period.iter().zip(mask_period))
                    {
                        let [a, b] = masked(*spin, occ);
                        sum[0] += a;
                        sum[1] += b;
                    }
                    continue;
                }
                let slabs = spin_period
                    .chunks_exact(stride)
                    .zip(mask_period.chunks_exact(stride));
                for (sum, (slab, mask)) in plane.iter_mut().zip(slabs) {
                    let [a, b] = masked_sum(slab, mask);
                    sum[0] += a;
                    sum[1] += b;
                }
            }
        }
    }

    /// Geometric angle winding per intact plaquette, and intact plaquettes per site.
    fn vortices<S: Spin>(
        &mut self,
        lattice: &Lattice,
        spins: &[S],
        couplings: &[S::Value],
        occupied: &[bool],
    ) -> [f64; 2] {
        for (angle, spin) in self.angles.iter_mut().zip(spins) {
            let [c, s] = spin.components();
            *angle = s.atan2(c);
        }
        let angles = &self.angles;
        // Angle from site a to site b, wrapped to [-π, π].
        let edge = |a: usize, b: usize| {
            let delta = angles[b] - angles[a];
            delta - TAU * (delta / TAU).round()
        };
        let mut intact = 0;
        let mut winding = 0.0;
        for i in 0..lattice.n_spins {
            let x = lattice.neighbor_fwd(i, 0);
            let y = lattice.neighbor_fwd(i, 1);
            let xy = lattice.neighbor_fwd(x, 1);
            if ![i, x, xy, y].iter().all(|&j| occupied[j])
                || [2 * i, 2 * i + 1, 2 * x + 1, 2 * y]
                    .iter()
                    .any(|&b| couplings[b].to_f64() == 0.0)
            {
                continue;
            }
            intact += 1;
            let circulation = edge(i, x) + edge(x, xy) + edge(xy, y) + edge(y, i);
            winding += (circulation / TAU).round().abs();
        }
        let density = if intact > 0 {
            winding / intact as f64
        } else {
            f64::NAN
        };
        [density, intact as f64 / lattice.n_spins as f64]
    }

    pub fn end_sweep(&mut self, n_replicas: usize) {
        self.count += n_replicas as u64;
        if self.options.block_size.is_none() {
            return;
        }
        self.block_count += n_replicas as u64;
        self.block_sweeps += 1;
        if self.options.block_size == Some(self.block_sweeps) {
            self.flush_block();
        }
    }
    /// Stores block rows only for temperatures where `owned` holds; the rest stay
    /// empty. Collectors that each measure a subset of temperatures use this so
    /// their blocks do not repeat everyone else's rows.
    pub(crate) fn own_block_temperatures(&mut self, owned: impl Fn(usize) -> bool) {
        self.block_temperatures = (0..self.sums.len()).map(owned).collect();
    }

    fn flush_block(&mut self) {
        if self.block_count == 0 {
            return;
        }
        let sums = self
            .block_sums
            .iter()
            .enumerate()
            .map(|(t, row)| {
                let stored = self.block_temperatures.get(t).copied().unwrap_or(true);
                if stored {
                    row.clone()
                } else {
                    Vec::new()
                }
            })
            .collect();
        self.blocks.push(MomentBlock {
            sums,
            count: self.block_count,
            sweeps: self.block_sweeps,
        });
        for row in &mut self.block_sums {
            row.fill(0.0);
        }
        self.block_count = 0;
        self.block_sweeps = 0;
    }
    /// Moments are zero, not NaN, when no sweep was measured.
    pub fn finish(mut self) -> PhysicsResult {
        self.flush_block();
        if self.count > 0 {
            for row in &mut self.sums {
                for value in row {
                    *value /= self.count as f64;
                }
            }
        }
        PhysicsResult {
            fields: self.fields,
            moments: self.sums,
            count: self.count,
            blocks: self.blocks,
        }
    }
}

#[inline]
fn masked<S: Spin>(spin: S, occupied: bool) -> [f64; 2] {
    let weight = f64::from(u8::from(occupied));
    let [a, b] = spin.components();
    [weight * a, weight * b]
}

/// Sum of occupied spin components, with four accumulators to break the
/// floating-point add dependency chain.
fn masked_sum<S: Spin>(slab: &[S], mask: &[bool]) -> [f64; 2] {
    let mut lanes = [[0.0; 2]; 4];
    let spins4 = slab.chunks_exact(4);
    let masks4 = mask.chunks_exact(4);
    let mut tail = [0.0; 2];
    for (&spin, &occ) in spins4.remainder().iter().zip(masks4.remainder()) {
        let [a, b] = masked(spin, occ);
        tail[0] += a;
        tail[1] += b;
    }
    for (spins, masks) in spins4.zip(masks4) {
        for ((lane, &spin), &occ) in lanes.iter_mut().zip(spins).zip(masks) {
            let [a, b] = masked(spin, occ);
            lane[0] += a;
            lane[1] += b;
        }
    }
    let [l0, l1, l2, l3] = lanes;
    [
        (l0[0] + l1[0]) + (l2[0] + l3[0]) + tail[0],
        (l0[1] + l1[1]) + (l2[1] + l3[1]) + tail[1],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoshiro256StarStar;
    #[test]
    fn signed_winding_mask_coverage_and_rotation_invariance() {
        let lattice = Lattice::new(vec![3, 3]);
        let mut spins = vec![[1.0, 0.0]; 9];
        spins[3] = [0.0, 1.0];
        spins[4] = [-1.0, 0.0];
        spins[1] = [0.0, -1.0];
        let mut j = vec![0.0; 18];
        for (bond, value) in [(0, 1.0), (7, -1.0), (2, 1.0), (1, 1.0)] {
            j[bond] = value;
        }
        let mut occupied = vec![false; 9];
        for i in [0, 3, 4, 1, 8] {
            occupied[i] = true;
        }
        let options = PhysicsOptions {
            vortices: true,
            displacements: vec![vec![1, 0], vec![0, 0]],
            block_size: Some(2),
        };
        let measure = |s: &[[f64; 2]], j: &[f64]| {
            let mut collector = PhysicsCollector::new(&lattice, 1, &options, 2).unwrap();
            collector.measure(&lattice, s, j, &occupied, 0);
            collector.end_sweep(1);
            collector.finish().values(&lattice.shape, &[1.0], 2)
        };
        let original = measure(&spins, &j);
        assert_eq!(original["angle_vortex_density"][0][0], 1.0);
        assert_eq!(original["intact_plaquette_fraction"][0][0], 1.0 / 9.0);
        assert_eq!(original["correlations"][0][1], 5.0 / 9.0);
        for s in &mut spins {
            *s = [0.6 * s[0] - 0.8 * s[1], 0.8 * s[0] + 0.6 * s[1]];
        }
        let rotated = measure(&spins, &j);
        for (name, rows) in &original {
            for (a, b) in rows.iter().flatten().zip(rotated[name].iter().flatten()) {
                assert!(
                    (a - b).abs() < 1e-12 || (a.is_nan() && b.is_nan()),
                    "{name}: {a} vs {b}"
                );
            }
        }
        j[0] = 0.0;
        let broken = measure(&spins, &j);
        assert!(broken["angle_vortex_density"][0][0].is_nan());
        assert_eq!(broken["intact_plaquette_fraction"][0][0], 0.0);
    }
    #[test]
    fn disorder_variance_is_excluded_from_heat_capacity() {
        let lattice = Lattice::new(vec![3, 3]);
        let spins = vec![[1.0, 0.0]; 9];
        let occupied = vec![true; 9];
        let results: Vec<_> = [1.0, 2.0]
            .iter()
            .map(|&j| {
                let mut c =
                    PhysicsCollector::new(&lattice, 1, &PhysicsOptions::default(), 2).unwrap();
                for _ in 0..3 {
                    c.measure(&lattice, &spins, &[j; 18], &occupied, 0);
                    c.end_sweep(1);
                }
                c.finish()
            })
            .collect();
        let aggregate = PhysicsResult::aggregate(&results, &[3, 3], &[1.0], 2);
        assert_eq!(aggregate["energies"][0][0], -3.0);
        assert_eq!(aggregate["energies2"][0][0], 10.0);
        assert_eq!(aggregate["heat_capacity"][0][0], 0.0);
        assert_eq!(aggregate["binder_cumulant"][0][0], 0.5);
    }
    fn random_xy(n: usize, seed: u64) -> Vec<[f64; 2]> {
        let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
        (0..n).map(|_| <[f64; 2]>::random(&mut rng)).collect()
    }

    /// Direct per-site Fourier sum Σ_d |Σ_i s_i e^{2πi x_d/L_d}|² / N.
    fn direct_structure_factor(
        lattice: &Lattice,
        spins: &[[f64; 2]],
        occupied: &[bool],
    ) -> Vec<f64> {
        (0..lattice.n_dims)
            .map(|d| {
                let mut f = [[0.0; 2]; 2];
                for (i, spin) in spins.iter().enumerate().filter(|&(i, _)| occupied[i]) {
                    let x = (i / lattice.strides[d]) % lattice.shape[d];
                    let (sin, cos) = (TAU * x as f64 / lattice.shape[d] as f64).sin_cos();
                    for k in 0..2 {
                        f[k][0] += spin[k] * cos;
                        f[k][1] += spin[k] * sin;
                    }
                }
                f.iter().flatten().map(|v| v * v).sum::<f64>() / lattice.n_spins as f64
            })
            .collect()
    }

    #[test]
    fn plane_sums_match_direct_fourier_sums() {
        let lattice = Lattice::new(vec![3, 4, 5]);
        let spins = random_xy(lattice.n_spins, 3);
        let occupied: Vec<bool> = (0..lattice.n_spins).map(|i| i % 7 != 3).collect();
        let mut c = PhysicsCollector::new(&lattice, 1, &PhysicsOptions::default(), 2).unwrap();
        c.measure(
            &lattice,
            &spins,
            &vec![0.0; 3 * lattice.n_spins],
            &occupied,
            0,
        );
        c.end_sweep(1);
        let got = &c.finish().values(&lattice.shape, &[1.0], 2)["structure_factor_min"][0];
        for (a, b) in got
            .iter()
            .zip(direct_structure_factor(&lattice, &spins, &occupied))
        {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    #[test]
    fn angle_differences_match_per_edge_atan2_winding() {
        let lattice = Lattice::new(vec![6, 5]);
        let couplings = vec![1.0; 2 * lattice.n_spins];
        let occupied = vec![true; lattice.n_spins];
        let options = PhysicsOptions {
            vortices: true,
            ..Default::default()
        };
        for seed in 0..8 {
            let spins = random_xy(lattice.n_spins, seed);
            let mut c = PhysicsCollector::new(&lattice, 1, &options, 2).unwrap();
            c.measure(&lattice, &spins, &couplings, &occupied, 0);
            c.end_sweep(1);
            let got = c.finish().values(&lattice.shape, &[1.0], 2)["angle_vortex_density"][0][0];
            let vortices: f64 = (0..lattice.n_spins)
                .map(|i| {
                    let x = lattice.neighbor_fwd(i, 0);
                    let y = lattice.neighbor_fwd(i, 1);
                    let xy = lattice.neighbor_fwd(x, 1);
                    let circulation: f64 = [i, x, xy, y, i]
                        .windows(2)
                        .map(|e| {
                            let (a, b) = (spins[e[0]], spins[e[1]]);
                            cross(a, b).atan2(dot(a, b))
                        })
                        .sum();
                    (circulation / TAU).round().abs()
                })
                .sum();
            assert_eq!(got, vortices / lattice.n_spins as f64, "seed {seed}");
        }
    }

    #[test]
    fn cached_totals_reproduce_the_bond_pass() {
        // N = 32 keeps the f32 energy cache exact for ±J couplings.
        let lattice = Lattice::new(vec![4, 8]);
        let mut rng = Xoshiro256StarStar::seed_from_u64(17);
        let spins: Vec<i8> = (0..lattice.n_spins).map(|_| i8::random(&mut rng)).collect();
        let couplings: Vec<f32> = (0..2 * lattice.n_spins)
            .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
            .collect();
        let occupied = vec![true; lattice.n_spins];
        let options = PhysicsOptions {
            displacements: vec![vec![1, 2]],
            ..Default::default()
        };
        let mut energies = [0.0f32];
        let mut magnetization = [0i64];
        crate::spins::energy::compute_energies_and_magnetizations_into(
            &lattice,
            &spins,
            &couplings,
            &mut energies,
            &mut magnetization,
        );
        let mut full = PhysicsCollector::new(&lattice, 1, &options, 1).unwrap();
        let mut cached = PhysicsCollector::new(&lattice, 1, &options, 1).unwrap();
        let a = full.measure(&lattice, &spins, &couplings, &occupied, 0);
        let b = cached.measure_cached(
            &lattice,
            &spins,
            &couplings,
            &occupied,
            0,
            CachedTotals {
                energy: -(energies[0] as f64),
                magnetization: [magnetization[0] as f64, 0.0],
            },
        );
        assert_eq!(a, b);
        assert_eq!(full.sums, cached.sums);
    }

    #[test]
    fn finish_without_measurements_is_zero_not_nan() {
        let lattice = Lattice::new(vec![3, 3]);
        let c = PhysicsCollector::new(&lattice, 2, &PhysicsOptions::default(), 1).unwrap();
        let result = c.finish();
        assert_eq!(result.count, 0);
        assert!(result.moments.iter().flatten().all(|&v| v == 0.0));
    }

    #[test]
    fn antiferromagnetic_uniform_length_is_undefined_without_clamping() {
        let lattice = Lattice::new(vec![4, 4]);
        let spins: Vec<_> = (0..16)
            .map(|i| {
                if (i / 4 + i % 4) % 2 == 0 {
                    [1.0, 0.0]
                } else {
                    [-1.0, 0.0]
                }
            })
            .collect();
        let mut c = PhysicsCollector::new(
            &lattice,
            1,
            &PhysicsOptions {
                displacements: vec![vec![1, 0]],
                ..Default::default()
            },
            2,
        )
        .unwrap();
        c.measure(&lattice, &spins, &vec![-1.0; 32], &[true; 16], 0);
        c.end_sweep(1);
        let result = c.finish().values(&[4, 4], &[1.0], 2);
        assert_eq!(result["correlations"][0][0], -1.0);
        assert!(result["correlation_length"][0].iter().all(|v| v.is_nan()));
    }
}
