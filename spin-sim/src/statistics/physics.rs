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
    phases: Vec<Vec<[f64; 2]>>,
    displaced: Vec<Vec<usize>>,
    options: PhysicsOptions,
    components: usize,
    sums: Vec<Vec<f64>>,
    block_sums: Vec<Vec<f64>>,
    block_count: u64,
    block_sweeps: usize,
    blocks: Vec<MomentBlock>,
    count: u64,
    scratch: Vec<f64>,
    fourier: Vec<[[f64; 2]; 2]>,
}

impl PhysicsCollector {
    pub fn new(
        lattice: &Lattice,
        n_temps: usize,
        options: &PhysicsOptions,
        components: usize,
    ) -> Result<Self, String> {
        if options.block_size == Some(0) {
            return Err("block_size must be positive".into());
        }
        if options.vortices && (lattice.n_dims != 2 || components != 2) {
            return Err("angle vortices require a two-dimensional XY lattice".into());
        }
        if options
            .displacements
            .iter()
            .any(|r| r.len() != lattice.n_dims)
        {
            return Err("correlation displacements must match lattice dimensionality".into());
        }
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
        let phases = (0..lattice.n_dims)
            .map(|d| {
                (0..lattice.n_spins)
                    .map(|i| {
                        let phase = TAU * ((i / lattice.strides[d]) % lattice.shape[d]) as f64
                            / lattice.shape[d] as f64;
                        let (s, c) = phase.sin_cos();
                        [c, s]
                    })
                    .collect()
            })
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
            displaced,
            options: options.clone(),
            components,
            sums: vec![vec![0.0; width]; n_temps],
            block_sums: if options.block_size.is_some() {
                vec![vec![0.0; width]; n_temps]
            } else {
                vec![]
            },
            block_count: 0,
            block_sweeps: 0,
            blocks: vec![],
            count: 0,
            scratch: vec![0.0; width],
            fourier: vec![[[0.0; 2]; 2]; lattice.n_dims],
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
        let n = lattice.n_spins as f64;
        self.scratch.fill(0.0);
        self.fourier.fill([[0.0; 2]; 2]);
        let mut magnetization = [0.0; 2];
        let mut interaction = 0.0;
        let d_offset = 5;
        let i_offset = 5 + lattice.n_dims;
        let i2_offset = 5 + 2 * lattice.n_dims;
        let sk_offset = if self.components == 2 {
            5 + 3 * lattice.n_dims
        } else {
            5
        };
        for i in 0..lattice.n_spins {
            let spin = spins[i].components();
            if occupied[i] {
                for k in 0..2 {
                    magnetization[k] += spin[k];
                }
                for d in 0..lattice.n_dims {
                    for (k, &component) in spin.iter().enumerate() {
                        for a in 0..2 {
                            self.fourier[d][k][a] += component * self.phases[d][i][a];
                        }
                    }
                }
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
        let energy = -interaction / n;
        let m2 = dot(magnetization, magnetization) / (n * n);
        self.scratch[..5].copy_from_slice(&[energy, energy * energy, m2.sqrt(), m2, m2 * m2]);
        for d in 0..lattice.n_dims {
            if self.components == 2 {
                self.scratch[i2_offset + d] = self.scratch[i_offset + d].powi(2);
            }
            self.scratch[sk_offset + d] =
                self.fourier[d].iter().flatten().map(|v| v * v).sum::<f64>() / n;
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
                let loop_sites = [i, x, xy, y, i];
                let circulation: f64 = loop_sites
                    .windows(2)
                    .map(|edge| {
                        let a = spins[edge[0]].components();
                        let b = spins[edge[1]].components();
                        cross(a, b).atan2(dot(a, b))
                    })
                    .sum();
                winding += (circulation / TAU).round().abs();
            }
            let offset = correlation_offset + self.displaced.len();
            self.scratch[offset] = if intact > 0 {
                winding / intact as f64
            } else {
                f64::NAN
            };
            self.scratch[offset + 1] = intact as f64 / n;
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
    fn flush_block(&mut self) {
        if self.block_count == 0 {
            return;
        }
        self.blocks.push(MomentBlock {
            sums: self.block_sums.clone(),
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

#[cfg(test)]
mod tests {
    use super::*;
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
