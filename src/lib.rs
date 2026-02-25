use indicatif::{ProgressBar, ProgressStyle};
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use spin_sim::{run_sweep_loop, Lattice, Realization, SweepResult};

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
    #[pyo3(signature = (lattice_shape, couplings, temperatures, n_replicas=None, neighbor_offsets=None))]
    fn new(
        lattice_shape: Vec<usize>,
        couplings: PyReadonlyArrayDyn<f32>,
        temperatures: PyReadonlyArray1<f32>,
        n_replicas: Option<usize>,
        neighbor_offsets: Option<Vec<Vec<i64>>>,
    ) -> PyResult<Self> {
        let lattice = if let Some(offsets) = neighbor_offsets {
            let offsets: Vec<Vec<isize>> = offsets
                .into_iter()
                .map(|v| v.into_iter().map(|x| x as isize).collect())
                .collect();
            Lattice::with_offsets(lattice_shape, offsets)
        } else {
            Lattice::new(lattice_shape)
        };
        let n_spins = lattice.n_spins;
        let n_neighbors = lattice.n_neighbors;
        let n_replicas = n_replicas.unwrap_or(1);

        let temps_raw = temperatures.as_slice()?;
        let n_temps = temps_raw.len();
        let n_systems = n_replicas * n_temps;

        let coup_shape = couplings.shape();
        let expected_single: Vec<usize> = lattice
            .shape
            .iter()
            .copied()
            .chain(std::iter::once(n_neighbors))
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
        let chunk_size = n_spins * n_neighbors;

        let n_pairs = n_replicas / 2;
        let rngs_per_real = n_systems + n_temps * n_pairs;

        let mut realizations = Vec::with_capacity(n_realizations);
        for r in 0..n_realizations {
            let coup_chunk = couplings_raw[r * chunk_size..(r + 1) * chunk_size].to_vec();
            let base_seed = 42 + (r * rngs_per_real) as u64;
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

    #[pyo3(signature = (n_sweeps, sweep_mode, cluster_update_interval=None, cluster_mode=None, pt_interval=None, houdayer_interval=None, houdayer_mode=None, overlap_cluster_mode=None, warmup_ratio=None, collect_csd=None))]
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
        houdayer_mode: Option<&str>,
        overlap_cluster_mode: Option<&str>,
        warmup_ratio: Option<f64>,
        collect_csd: Option<bool>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let warmup = warmup_ratio.unwrap_or(0.25);
        let warmup_sweeps = (n_sweeps as f64 * warmup).round() as usize;
        let cluster_mode = cluster_mode.unwrap_or("sw");
        let houdayer_mode = houdayer_mode.unwrap_or("houdayer");
        let overlap_cluster_mode = overlap_cluster_mode.unwrap_or("wolff");

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
        if houdayer_interval.is_some() {
            match houdayer_mode {
                "houdayer" | "jorg" | "cmr" => {}
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Invalid houdayer mode. Use 'houdayer', 'jorg', or 'cmr'.",
                    ))
                }
            }
            match overlap_cluster_mode {
                "sw" | "wolff" => {}
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Invalid overlap cluster mode. Use 'sw' or 'wolff'.",
                    ))
                }
            }
        }

        let n_replicas = self.n_replicas;
        let n_temps = self.n_temps;
        let sweep_mode = sweep_mode.to_string();
        let cluster_mode = cluster_mode.to_string();
        let houdayer_mode = houdayer_mode.to_string();
        let overlap_cluster_mode = overlap_cluster_mode.to_string();
        let collect_csd = collect_csd.unwrap_or(false);

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
                        &houdayer_mode,
                        &overlap_cluster_mode,
                        collect_csd,
                        &|| pb.inc(1),
                    )
                })
                .collect()
        });

        pb.finish();
        let agg = SweepResult::aggregate(&results);

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

        if agg.fk_csd.iter().any(|h| h.iter().any(|&c| c > 0)) {
            let csd_py: Vec<_> = agg
                .fk_csd
                .into_iter()
                .map(|hist| Array1::from(hist).into_pyarray(py))
                .collect();
            dict.set_item("fk_csd", csd_py)?;
        }

        if agg.overlap_csd.iter().any(|h| h.iter().any(|&c| c > 0)) {
            let csd_py: Vec<_> = agg
                .overlap_csd
                .into_iter()
                .map(|hist| Array1::from(hist).into_pyarray(py))
                .collect();
            dict.set_item("overlap_csd", csd_py)?;
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
        let n_pairs = n_replicas / 2;
        let rngs_per_real = n_replicas * n_temps + n_temps * n_pairs;
        let lattice = &self.lattice;
        for (r, real) in self.realizations.iter_mut().enumerate() {
            real.reset(
                lattice,
                n_replicas,
                n_temps,
                base_seed + (r * rngs_per_real) as u64,
            );
        }
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IsingSimulation>()?;
    Ok(())
}
