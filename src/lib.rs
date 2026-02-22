use indicatif::{ProgressBar, ProgressStyle};
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;

mod clusters;
mod energy;
mod lattice;
mod parallel;
mod stats;
mod sweep;
mod tempering;

use lattice::Lattice;
use stats::Statistics;

#[pyclass]
struct IsingSimulation {
    lattice: Lattice,
    n_replicas: usize,
    n_temps: usize,
    spins: Vec<i8>,
    couplings: Vec<f32>,
    temperatures: Vec<f32>,
    system_ids: Vec<usize>,
    rngs: Vec<Xoshiro256StarStar>,
    // cached
    energies: Vec<f32>,
    interactions: Vec<f32>,
}

#[pymethods]
impl IsingSimulation {
    /// Create a new Ising simulation.
    ///
    /// Arguments:
    ///   lattice_shape: list of lattice dimensions, e.g. [32, 32]
    ///   couplings: numpy array of shape (*lattice_shape, n_dims), float32
    ///   temperatures: numpy array of shape (n_temps,), float32
    ///   n_replicas: independent copies of the PT ladder (default 1)
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

        // Copy couplings to flat Vec (n_spins * n_dims)
        let couplings_raw = couplings.as_slice()?;
        let couplings_vec: Vec<f32> = couplings_raw.to_vec();

        // Copy temperatures and tile for replicas
        let temps_raw = temperatures.as_slice()?;
        let n_temps = temps_raw.len();
        let n_replicas = n_replicas.unwrap_or(1);
        let n_systems = n_replicas * n_temps;
        let temps_vec: Vec<f32> = temps_raw.repeat(n_replicas);

        // Initialize RNGs — one per system
        let mut rngs = Vec::with_capacity(n_systems);
        for i in 0..n_systems {
            rngs.push(Xoshiro256StarStar::seed_from_u64(i as u64 + 42));
        }

        // Initialize random spins
        let mut spins = vec![0i8; n_systems * n_spins];
        for (i, rng) in rngs.iter_mut().enumerate() {
            for j in 0..n_spins {
                spins[i * n_spins + j] = if rng.gen::<f32>() < 0.5 { -1 } else { 1 };
            }
        }

        // system_ids: identity mapping initially
        let system_ids: Vec<usize> = (0..n_systems).collect();

        // Compute initial energies and interactions
        let (energies, interactions) =
            energy::compute_energies(&lattice, &spins, &couplings_vec, n_systems, true);
        let interactions = interactions.unwrap();

        Ok(Self {
            lattice,
            n_replicas,
            n_temps,
            spins,
            couplings: couplings_vec,
            temperatures: temps_vec,
            system_ids,
            rngs,
            energies,
            interactions,
        })
    }

    /// Run the full sampling loop.
    ///
    /// Arguments:
    ///   n_sweeps: total number of sweeps
    ///   sweep_mode: "metropolis" or "gibbs"
    ///   cluster_update_interval: if set, do cluster update every N sweeps
    ///   cluster_mode: "sw" or "wolff"
    ///   pt_interval: if set, do parallel tempering every N sweeps
    ///   warmup_ratio: fraction of sweeps to discard (default 0.25)
    ///
    /// Returns: dict with keys "mags", "mags2", "mags4", "energies", "energies2"
    ///   Each is a numpy array of shape (n_temps,) with the running averages.
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

        let n_spins = self.lattice.n_spins;
        let n_temps = self.n_temps;
        let n_systems = self.n_replicas * n_temps;

        let mut mags_stat = Statistics::new(n_temps, 1);
        let mut mags2_stat = Statistics::new(n_temps, 1);
        let mut mags4_stat = Statistics::new(n_temps, 1);
        let mut energies_stat = Statistics::new(n_temps, 1);
        let mut energies2_stat = Statistics::new(n_temps, 2);

        let n_pairs = self.n_replicas / 2;
        let mut overlap_stat = Statistics::new(n_temps, 1);
        let mut overlap2_stat = Statistics::new(n_temps, 1);
        let mut overlap4_stat = Statistics::new(n_temps, 1);

        let pb = ProgressBar::new(n_sweeps as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{msg} [{bar:40}] {pos}/{len} [{elapsed_precise} < {eta_precise}, {per_sec}]",
            )
            .unwrap()
            .progress_chars("=> "),
        );
        pb.set_message("sweeps");

        for sweep_id in 0..n_sweeps {
            pb.inc(1);
            let record = sweep_id >= warmup_sweeps;

            // Single-spin-flip sweep
            match sweep_mode {
                "metropolis" => sweep::metropolis_sweep(
                    &self.lattice,
                    &mut self.spins,
                    &self.couplings,
                    &self.temperatures,
                    &self.system_ids,
                    &mut self.rngs,
                ),
                "gibbs" => sweep::gibbs_sweep(
                    &self.lattice,
                    &mut self.spins,
                    &self.couplings,
                    &self.temperatures,
                    &self.system_ids,
                    &mut self.rngs,
                ),
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Invalid sweep mode. Use 'metropolis' or 'gibbs'.",
                    ))
                }
            }

            // Cluster updates
            let do_cluster =
                cluster_update_interval.is_some_and(|interval| sweep_id % interval == 0);

            if do_cluster {
                match cluster_mode {
                    "wolff" => {
                        clusters::wolff_update(
                            &self.lattice,
                            &mut self.spins,
                            &self.couplings,
                            &self.temperatures,
                            &self.system_ids,
                            &mut self.rngs,
                        );
                        // Recompute energies after wolff
                        (self.energies, _) = energy::compute_energies(
                            &self.lattice,
                            &self.spins,
                            &self.couplings,
                            n_systems,
                            false,
                        );
                    }
                    "sw" => {
                        // Need interactions for SW
                        let (energies, interactions) = energy::compute_energies(
                            &self.lattice,
                            &self.spins,
                            &self.couplings,
                            n_systems,
                            true,
                        );
                        self.energies = energies;
                        self.interactions = interactions.unwrap();

                        clusters::sw_update(
                            &self.lattice,
                            &mut self.spins,
                            &self.interactions,
                            &self.temperatures,
                            &self.system_ids,
                            &mut self.rngs,
                        );

                        // Recompute energies after SW
                        (self.energies, _) = energy::compute_energies(
                            &self.lattice,
                            &self.spins,
                            &self.couplings,
                            n_systems,
                            false,
                        );
                    }
                    _ => {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "Invalid cluster mode. Use 'sw' or 'wolff'.",
                        ))
                    }
                }
            } else {
                // Just recompute energies
                (self.energies, _) = energy::compute_energies(
                    &self.lattice,
                    &self.spins,
                    &self.couplings,
                    n_systems,
                    false,
                );
            }

            // Record statistics — each replica contributes independently
            if record {
                let mut mags = vec![0.0f32; n_temps];
                let mut mags2 = vec![0.0f32; n_temps];
                let mut mags4 = vec![0.0f32; n_temps];
                let mut energies_ordered = vec![0.0f32; n_temps];

                for r in 0..self.n_replicas {
                    let offset = r * n_temps;
                    for t in 0..n_temps {
                        let system_id = self.system_ids[offset + t];
                        let spin_base = system_id * n_spins;
                        let mut sum = 0i64;
                        for j in 0..n_spins {
                            sum += self.spins[spin_base + j] as i64;
                        }
                        let mag = sum as f32 / n_spins as f32;
                        let m2 = mag * mag;
                        mags[t] = mag;
                        mags2[t] = m2;
                        mags4[t] = m2 * m2;
                        energies_ordered[t] = self.energies[system_id];
                    }

                    mags_stat.update(&mags);
                    mags2_stat.update(&mags2);
                    mags4_stat.update(&mags4);
                    energies_stat.update(&energies_ordered);
                    energies2_stat.update(&energies_ordered);
                }

                // Overlap statistics: pair consecutive replicas
                for pair_idx in 0..n_pairs {
                    let r_a = 2 * pair_idx;
                    let r_b = 2 * pair_idx + 1;
                    let mut overlaps = vec![0.0f32; n_temps];
                    let mut overlaps2 = vec![0.0f32; n_temps];
                    let mut overlaps4 = vec![0.0f32; n_temps];

                    for t in 0..n_temps {
                        let sys_a = self.system_ids[r_a * n_temps + t];
                        let sys_b = self.system_ids[r_b * n_temps + t];
                        let base_a = sys_a * n_spins;
                        let base_b = sys_b * n_spins;
                        let mut dot = 0i64;
                        for j in 0..n_spins {
                            dot +=
                                (self.spins[base_a + j] as i64) * (self.spins[base_b + j] as i64);
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

            // Parallel tempering — per replica
            if let Some(interval) = pt_interval {
                if sweep_id % interval == 0 {
                    for r in 0..self.n_replicas {
                        let offset = r * n_temps;
                        let sid_slice = &mut self.system_ids[offset..offset + n_temps];
                        let temp_slice = &self.temperatures[offset..offset + n_temps];
                        tempering::parallel_tempering(
                            &self.energies,
                            temp_slice,
                            sid_slice,
                            n_spins,
                            &mut self.rngs[offset],
                        );
                    }
                }
            }

            // Houdayer isoenergetic cluster move
            if let Some(interval) = houdayer_interval {
                if sweep_id % interval == 0 && self.n_replicas >= 2 {
                    clusters::houdayer_update(
                        &self.lattice,
                        &mut self.spins,
                        &self.system_ids,
                        self.n_replicas,
                        self.n_temps,
                        &mut self.rngs[0],
                    );
                    (self.energies, _) = energy::compute_energies(
                        &self.lattice,
                        &self.spins,
                        &self.couplings,
                        n_systems,
                        false,
                    );
                }
            }
        }
        pb.finish();

        // Build result dict
        let dict = PyDict::new(py);
        dict.set_item("mags", Array1::from(mags_stat.average()).into_pyarray(py))?;
        dict.set_item("mags2", Array1::from(mags2_stat.average()).into_pyarray(py))?;
        dict.set_item("mags4", Array1::from(mags4_stat.average()).into_pyarray(py))?;
        dict.set_item(
            "energies",
            Array1::from(energies_stat.average()).into_pyarray(py),
        )?;
        dict.set_item(
            "energies2",
            Array1::from(energies2_stat.average()).into_pyarray(py),
        )?;

        if n_pairs > 0 {
            dict.set_item(
                "overlap",
                Array1::from(overlap_stat.average()).into_pyarray(py),
            )?;
            dict.set_item(
                "overlap2",
                Array1::from(overlap2_stat.average()).into_pyarray(py),
            )?;
            dict.set_item(
                "overlap4",
                Array1::from(overlap4_stat.average()).into_pyarray(py),
            )?;
        }

        Ok(dict)
    }

    /// Return current spins as a numpy array of shape (n_replicas, *lattice_shape).
    fn get_spins<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i8>>> {
        Ok(Array1::from(self.spins.clone()).into_pyarray(py))
    }

    /// Reset spins and RNGs.
    #[pyo3(signature = (seed=None))]
    fn reset(&mut self, seed: Option<u64>) {
        let base_seed = seed.unwrap_or(42);
        let n_spins = self.lattice.n_spins;
        let n_systems = self.n_replicas * self.n_temps;

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
            energy::compute_energies(&self.lattice, &self.spins, &self.couplings, n_systems, true);
        self.energies = energies;
        self.interactions = interactions.unwrap();
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IsingSimulation>()?;
    Ok(())
}
