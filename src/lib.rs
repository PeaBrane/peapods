use indicatif::{ProgressBar, ProgressStyle};
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;

mod clusters;
mod energy;
mod lattice;
mod parallel;
mod stats;
mod sweep;
mod tempering;

use lattice::Lattice;
use stats::Statistics;

struct Realization {
    couplings: Vec<f32>,
    spins: Vec<i8>,
    temperatures: Vec<f32>,
    system_ids: Vec<usize>,
    rngs: Vec<Xoshiro256StarStar>,
    energies: Vec<f32>,
    interactions: Vec<f32>,
}

impl Realization {
    fn new(
        lattice: &Lattice,
        couplings: Vec<f32>,
        temps: &[f32],
        n_replicas: usize,
        base_seed: u64,
    ) -> Self {
        let n_spins = lattice.n_spins;
        let n_temps = temps.len();
        let n_systems = n_replicas * n_temps;

        let temperatures = temps.repeat(n_replicas);

        let mut rngs = Vec::with_capacity(n_systems);
        for i in 0..n_systems {
            rngs.push(Xoshiro256StarStar::seed_from_u64(base_seed + i as u64));
        }

        let mut spins = vec![0i8; n_systems * n_spins];
        for (i, rng) in rngs.iter_mut().enumerate() {
            for j in 0..n_spins {
                spins[i * n_spins + j] = if rng.gen::<f32>() < 0.5 { -1 } else { 1 };
            }
        }

        let system_ids: Vec<usize> = (0..n_systems).collect();

        let (energies, interactions) =
            energy::compute_energies(lattice, &spins, &couplings, n_systems, true);
        let interactions = interactions.unwrap();

        Self {
            couplings,
            spins,
            temperatures,
            system_ids,
            rngs,
            energies,
            interactions,
        }
    }

    fn reset(&mut self, lattice: &Lattice, n_replicas: usize, n_temps: usize, base_seed: u64) {
        let n_spins = lattice.n_spins;
        let n_systems = n_replicas * n_temps;

        for i in 0..n_systems {
            self.rngs[i] = Xoshiro256StarStar::seed_from_u64(base_seed + i as u64);
            for j in 0..n_spins {
                self.spins[i * n_spins + j] = if self.rngs[i].gen::<f32>() < 0.5 {
                    -1
                } else {
                    1
                };
            }
        }

        self.system_ids = (0..n_systems).collect();

        let (energies, interactions) =
            energy::compute_energies(lattice, &self.spins, &self.couplings, n_systems, true);
        self.energies = energies;
        self.interactions = interactions.unwrap();
    }
}

struct SweepResult {
    mags: Vec<f64>,
    mags2: Vec<f64>,
    mags4: Vec<f64>,
    energies: Vec<f64>,
    energies2: Vec<f64>,
    overlap: Vec<f64>,
    overlap2: Vec<f64>,
    overlap4: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
fn run_sweep_loop(
    lattice: &Lattice,
    real: &mut Realization,
    n_replicas: usize,
    n_temps: usize,
    n_sweeps: usize,
    warmup_sweeps: usize,
    sweep_mode: &str,
    cluster_update_interval: Option<usize>,
    cluster_mode: &str,
    pt_interval: Option<usize>,
    houdayer_interval: Option<usize>,
    pb: &ProgressBar,
) -> SweepResult {
    let n_spins = lattice.n_spins;
    let n_systems = n_replicas * n_temps;

    let mut mags_stat = Statistics::new(n_temps, 1);
    let mut mags2_stat = Statistics::new(n_temps, 1);
    let mut mags4_stat = Statistics::new(n_temps, 1);
    let mut energies_stat = Statistics::new(n_temps, 1);
    let mut energies2_stat = Statistics::new(n_temps, 2);

    let n_pairs = n_replicas / 2;
    let mut overlap_stat = Statistics::new(n_temps, 1);
    let mut overlap2_stat = Statistics::new(n_temps, 1);
    let mut overlap4_stat = Statistics::new(n_temps, 1);

    for sweep_id in 0..n_sweeps {
        pb.inc(1);
        let record = sweep_id >= warmup_sweeps;

        match sweep_mode {
            "metropolis" => sweep::metropolis_sweep(
                lattice,
                &mut real.spins,
                &real.couplings,
                &real.temperatures,
                &real.system_ids,
                &mut real.rngs,
            ),
            "gibbs" => sweep::gibbs_sweep(
                lattice,
                &mut real.spins,
                &real.couplings,
                &real.temperatures,
                &real.system_ids,
                &mut real.rngs,
            ),
            _ => unreachable!(),
        }

        let do_cluster = cluster_update_interval.is_some_and(|interval| sweep_id % interval == 0);

        if do_cluster {
            match cluster_mode {
                "wolff" => {
                    clusters::wolff_update(
                        lattice,
                        &mut real.spins,
                        &real.couplings,
                        &real.temperatures,
                        &real.system_ids,
                        &mut real.rngs,
                    );
                    (real.energies, _) = energy::compute_energies(
                        lattice,
                        &real.spins,
                        &real.couplings,
                        n_systems,
                        false,
                    );
                }
                "sw" => {
                    let (energies, interactions) = energy::compute_energies(
                        lattice,
                        &real.spins,
                        &real.couplings,
                        n_systems,
                        true,
                    );
                    real.energies = energies;
                    real.interactions = interactions.unwrap();

                    clusters::sw_update(
                        lattice,
                        &mut real.spins,
                        &real.interactions,
                        &real.temperatures,
                        &real.system_ids,
                        &mut real.rngs,
                    );

                    (real.energies, _) = energy::compute_energies(
                        lattice,
                        &real.spins,
                        &real.couplings,
                        n_systems,
                        false,
                    );
                }
                _ => unreachable!(),
            }
        } else {
            (real.energies, _) =
                energy::compute_energies(lattice, &real.spins, &real.couplings, n_systems, false);
        }

        if record {
            let mut mags = vec![0.0f32; n_temps];
            let mut mags2 = vec![0.0f32; n_temps];
            let mut mags4 = vec![0.0f32; n_temps];
            let mut energies_ordered = vec![0.0f32; n_temps];

            for r in 0..n_replicas {
                let offset = r * n_temps;
                for t in 0..n_temps {
                    let system_id = real.system_ids[offset + t];
                    let spin_base = system_id * n_spins;
                    let mut sum = 0i64;
                    for j in 0..n_spins {
                        sum += real.spins[spin_base + j] as i64;
                    }
                    let mag = sum as f32 / n_spins as f32;
                    let m2 = mag * mag;
                    mags[t] = mag;
                    mags2[t] = m2;
                    mags4[t] = m2 * m2;
                    energies_ordered[t] = real.energies[system_id];
                }

                mags_stat.update(&mags);
                mags2_stat.update(&mags2);
                mags4_stat.update(&mags4);
                energies_stat.update(&energies_ordered);
                energies2_stat.update(&energies_ordered);
            }

            for pair_idx in 0..n_pairs {
                let r_a = 2 * pair_idx;
                let r_b = 2 * pair_idx + 1;
                let mut overlaps = vec![0.0f32; n_temps];
                let mut overlaps2 = vec![0.0f32; n_temps];
                let mut overlaps4 = vec![0.0f32; n_temps];

                for t in 0..n_temps {
                    let sys_a = real.system_ids[r_a * n_temps + t];
                    let sys_b = real.system_ids[r_b * n_temps + t];
                    let base_a = sys_a * n_spins;
                    let base_b = sys_b * n_spins;
                    let mut dot = 0i64;
                    for j in 0..n_spins {
                        dot += (real.spins[base_a + j] as i64) * (real.spins[base_b + j] as i64);
                    }
                    let q = dot as f32 / n_spins as f32;
                    let q2 = q * q;
                    overlaps[t] = q;
                    overlaps2[t] = q2;
                    overlaps4[t] = q2 * q2;
                }

                overlap_stat.update(&overlaps);
                overlap2_stat.update(&overlaps2);
                overlap4_stat.update(&overlaps4);
            }
        }

        if let Some(interval) = pt_interval {
            if sweep_id % interval == 0 {
                for r in 0..n_replicas {
                    let offset = r * n_temps;
                    let sid_slice = &mut real.system_ids[offset..offset + n_temps];
                    let temp_slice = &real.temperatures[offset..offset + n_temps];
                    tempering::parallel_tempering(
                        &real.energies,
                        temp_slice,
                        sid_slice,
                        n_spins,
                        &mut real.rngs[offset],
                    );
                }
            }
        }

        if let Some(interval) = houdayer_interval {
            if sweep_id % interval == 0 && n_replicas >= 2 {
                clusters::houdayer_update(
                    lattice,
                    &mut real.spins,
                    &real.system_ids,
                    n_replicas,
                    n_temps,
                    &mut real.rngs[0],
                );
                (real.energies, _) = energy::compute_energies(
                    lattice,
                    &real.spins,
                    &real.couplings,
                    n_systems,
                    false,
                );
            }
        }
    }

    SweepResult {
        mags: mags_stat.average(),
        mags2: mags2_stat.average(),
        mags4: mags4_stat.average(),
        energies: energies_stat.average(),
        energies2: energies2_stat.average(),
        overlap: if n_pairs > 0 {
            overlap_stat.average()
        } else {
            vec![]
        },
        overlap2: if n_pairs > 0 {
            overlap2_stat.average()
        } else {
            vec![]
        },
        overlap4: if n_pairs > 0 {
            overlap4_stat.average()
        } else {
            vec![]
        },
    }
}

fn aggregate_results(results: &[SweepResult]) -> SweepResult {
    let n = results.len() as f64;
    let n_temps = results[0].mags.len();
    let n_overlap = results[0].overlap.len();

    let mut agg = SweepResult {
        mags: vec![0.0; n_temps],
        mags2: vec![0.0; n_temps],
        mags4: vec![0.0; n_temps],
        energies: vec![0.0; n_temps],
        energies2: vec![0.0; n_temps],
        overlap: vec![0.0; n_overlap],
        overlap2: vec![0.0; n_overlap],
        overlap4: vec![0.0; n_overlap],
    };

    for r in results {
        for (a, &v) in agg.mags.iter_mut().zip(r.mags.iter()) {
            *a += v;
        }
        for (a, &v) in agg.mags2.iter_mut().zip(r.mags2.iter()) {
            *a += v;
        }
        for (a, &v) in agg.mags4.iter_mut().zip(r.mags4.iter()) {
            *a += v;
        }
        for (a, &v) in agg.energies.iter_mut().zip(r.energies.iter()) {
            *a += v;
        }
        for (a, &v) in agg.energies2.iter_mut().zip(r.energies2.iter()) {
            *a += v;
        }
        for (a, &v) in agg.overlap.iter_mut().zip(r.overlap.iter()) {
            *a += v;
        }
        for (a, &v) in agg.overlap2.iter_mut().zip(r.overlap2.iter()) {
            *a += v;
        }
        for (a, &v) in agg.overlap4.iter_mut().zip(r.overlap4.iter()) {
            *a += v;
        }
    }

    for v in agg
        .mags
        .iter_mut()
        .chain(agg.mags2.iter_mut())
        .chain(agg.mags4.iter_mut())
        .chain(agg.energies.iter_mut())
        .chain(agg.energies2.iter_mut())
        .chain(agg.overlap.iter_mut())
        .chain(agg.overlap2.iter_mut())
        .chain(agg.overlap4.iter_mut())
    {
        *v /= n;
    }

    agg
}

#[pyclass]
struct IsingSimulation {
    lattice: Lattice,
    n_replicas: usize,
    n_temps: usize,
    n_realizations: usize,
    realizations: Vec<Realization>,
}

#[pymethods]
impl IsingSimulation {
    #[new]
    #[pyo3(signature = (lattice_shape, couplings, temperatures, n_replicas=None))]
    fn new(
        lattice_shape: Vec<usize>,
        couplings: PyReadonlyArrayDyn<f32>,
        temperatures: PyReadonlyArray1<f32>,
        n_replicas: Option<usize>,
    ) -> PyResult<Self> {
        let lattice = Lattice::new(lattice_shape);
        let n_spins = lattice.n_spins;
        let n_dims = lattice.n_dims;
        let n_replicas = n_replicas.unwrap_or(1);

        let temps_raw = temperatures.as_slice()?;
        let n_temps = temps_raw.len();
        let n_systems = n_replicas * n_temps;

        let coup_shape = couplings.shape();
        let expected_single: Vec<usize> = lattice
            .shape
            .iter()
            .copied()
            .chain(std::iter::once(n_dims))
            .collect();

        let n_realizations = if coup_shape == expected_single.as_slice() {
            1
        } else if coup_shape.len() == expected_single.len() + 1
            && coup_shape[1..] == *expected_single.as_slice()
        {
            coup_shape[0]
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "couplings shape {:?} does not match lattice {:?}",
                coup_shape, expected_single
            )));
        };

        let couplings_raw = couplings.as_slice()?;
        let chunk_size = n_spins * n_dims;

        let mut realizations = Vec::with_capacity(n_realizations);
        for r in 0..n_realizations {
            let coup_chunk = couplings_raw[r * chunk_size..(r + 1) * chunk_size].to_vec();
            let base_seed = 42 + (r * n_systems) as u64;
            realizations.push(Realization::new(
                &lattice, coup_chunk, temps_raw, n_replicas, base_seed,
            ));
        }

        Ok(Self {
            lattice,
            n_replicas,
            n_temps,
            n_realizations,
            realizations,
        })
    }

    #[pyo3(signature = (n_sweeps, sweep_mode, cluster_update_interval=None, cluster_mode=None, pt_interval=None, houdayer_interval=None, warmup_ratio=None))]
    #[allow(clippy::too_many_arguments)]
    fn sample<'py>(
        &mut self,
        py: Python<'py>,
        n_sweeps: usize,
        sweep_mode: &str,
        cluster_update_interval: Option<usize>,
        cluster_mode: Option<&str>,
        pt_interval: Option<usize>,
        houdayer_interval: Option<usize>,
        warmup_ratio: Option<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let warmup = warmup_ratio.unwrap_or(0.25);
        let warmup_sweeps = (n_sweeps as f64 * warmup).round() as usize;
        let cluster_mode = cluster_mode.unwrap_or("sw");

        match sweep_mode {
            "metropolis" | "gibbs" => {}
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Invalid sweep mode. Use 'metropolis' or 'gibbs'.",
                ))
            }
        }
        if cluster_update_interval.is_some() {
            match cluster_mode {
                "sw" | "wolff" => {}
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Invalid cluster mode. Use 'sw' or 'wolff'.",
                    ))
                }
            }
        }

        let n_replicas = self.n_replicas;
        let n_temps = self.n_temps;
        let sweep_mode = sweep_mode.to_string();
        let cluster_mode = cluster_mode.to_string();

        let pb = ProgressBar::new((self.n_realizations * n_sweeps) as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{msg} [{bar:40}] {pos}/{len} [{elapsed_precise} < {eta_precise}, {per_sec}]",
            )
            .unwrap()
            .progress_chars("=> "),
        );
        pb.set_message("sweeps");

        let lattice = &self.lattice;
        let realizations = &mut self.realizations;

        let results: Vec<SweepResult> = py.allow_threads(|| {
            realizations
                .par_iter_mut()
                .map(|real| {
                    run_sweep_loop(
                        lattice,
                        real,
                        n_replicas,
                        n_temps,
                        n_sweeps,
                        warmup_sweeps,
                        &sweep_mode,
                        cluster_update_interval,
                        &cluster_mode,
                        pt_interval,
                        houdayer_interval,
                        &pb,
                    )
                })
                .collect()
        });

        pb.finish();
        let agg = aggregate_results(&results);

        let dict = PyDict::new(py);
        dict.set_item("mags", Array1::from(agg.mags).into_pyarray(py))?;
        dict.set_item("mags2", Array1::from(agg.mags2).into_pyarray(py))?;
        dict.set_item("mags4", Array1::from(agg.mags4).into_pyarray(py))?;
        dict.set_item("energies", Array1::from(agg.energies).into_pyarray(py))?;
        dict.set_item("energies2", Array1::from(agg.energies2).into_pyarray(py))?;

        if !agg.overlap.is_empty() {
            dict.set_item("overlap", Array1::from(agg.overlap).into_pyarray(py))?;
            dict.set_item("overlap2", Array1::from(agg.overlap2).into_pyarray(py))?;
            dict.set_item("overlap4", Array1::from(agg.overlap4).into_pyarray(py))?;
        }

        Ok(dict)
    }

    fn get_spins<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i8>>> {
        Ok(Array1::from(self.realizations[0].spins.clone()).into_pyarray(py))
    }

    #[pyo3(signature = (seed=None))]
    fn reset(&mut self, seed: Option<u64>) {
        let base_seed = seed.unwrap_or(42);
        let n_replicas = self.n_replicas;
        let n_temps = self.n_temps;
        let n_systems = n_replicas * n_temps;
        let lattice = &self.lattice;
        for (r, real) in self.realizations.iter_mut().enumerate() {
            real.reset(
                lattice,
                n_replicas,
                n_temps,
                base_seed + (r * n_systems) as u64,
            );
        }
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IsingSimulation>()?;
    Ok(())
}
