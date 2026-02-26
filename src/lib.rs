use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use indicatif::{ProgressBar, ProgressStyle};
use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use spin_sim::config::*;
use spin_sim::{run_sweep_parallel, Lattice, Realization};

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

    #[pyo3(signature = (
        n_sweeps,
        sweep_mode,
        cluster_update_interval=None,
        cluster_mode=None,
        pt_interval=None,
        overlap_cluster_update_interval=None,
        overlap_cluster_build_mode=None,
        overlap_cluster_mode=None,
        warmup_ratio=None,
        collect_csd=None,
        overlap_update_mode=None,
        collect_top_clusters=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn sample<'py>(
        &mut self,
        py: Python<'py>,
        n_sweeps: usize,
        sweep_mode: &str,
        cluster_update_interval: Option<usize>,
        cluster_mode: Option<&str>,
        pt_interval: Option<usize>,
        overlap_cluster_update_interval: Option<usize>,
        overlap_cluster_build_mode: Option<&str>,
        overlap_cluster_mode: Option<&str>,
        warmup_ratio: Option<f64>,
        collect_csd: Option<bool>,
        overlap_update_mode: Option<&str>,
        collect_top_clusters: Option<bool>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let warmup = warmup_ratio.unwrap_or(0.25);
        let warmup_sweeps = (n_sweeps as f64 * warmup).round() as usize;
        let collect_csd = collect_csd.unwrap_or(false);
        let collect_top_clusters = collect_top_clusters.unwrap_or(false);

        let sweep_mode_enum =
            SweepMode::try_from(sweep_mode).map_err(pyo3::exceptions::PyValueError::new_err)?;

        let cluster_update = cluster_update_interval
            .map(|interval| {
                let mode_str = cluster_mode.unwrap_or("sw");
                let mode = ClusterMode::try_from(mode_str)
                    .map_err(pyo3::exceptions::PyValueError::new_err);
                mode.map(|m| ClusterConfig {
                    interval,
                    mode: m,
                    collect_csd,
                })
            })
            .transpose()?;

        let overlap_cluster = overlap_cluster_update_interval
            .map(|interval| {
                let build_mode_str = overlap_cluster_build_mode.unwrap_or("houdayer");
                let oc_mode_str = overlap_cluster_mode.unwrap_or("wolff");
                let ou_mode_str = overlap_update_mode.unwrap_or("swap");

                let build_mode = OverlapClusterBuildMode::try_from(build_mode_str)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                let oc_mode = ClusterMode::try_from(oc_mode_str)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                let ou_mode = OverlapUpdateMode::try_from(ou_mode_str)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;

                Ok::<_, PyErr>(OverlapClusterConfig {
                    interval,
                    mode: build_mode,
                    cluster_mode: oc_mode,
                    update_mode: ou_mode,
                    collect_csd,
                    collect_top_clusters,
                })
            })
            .transpose()?;

        let config = SimConfig {
            n_sweeps,
            warmup_sweeps,
            sweep_mode: sweep_mode_enum,
            cluster_update,
            pt_interval,
            overlap_cluster,
        };

        let n_replicas = self.n_replicas;
        let n_temps = self.n_temps;

        let pb = ProgressBar::new(n_sweeps as u64);
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
        let n_real = self.n_realizations as u64;
        let counter = AtomicU64::new(0);

        let interrupted = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&interrupted);
        let _ = ctrlc::set_handler(move || {
            flag.store(true, Ordering::Relaxed);
        });

        let agg = py
            .allow_threads(|| {
                run_sweep_parallel(
                    lattice,
                    realizations,
                    n_replicas,
                    n_temps,
                    &config,
                    &interrupted,
                    &|| {
                        let prev = counter.fetch_add(1, Ordering::Relaxed);
                        if (prev + 1).is_multiple_of(n_real) {
                            pb.inc(1);
                        }
                    },
                )
            })
            .map_err(|e| {
                if e == "interrupted" {
                    pyo3::exceptions::PyKeyboardInterrupt::new_err(e)
                } else {
                    pyo3::exceptions::PyValueError::new_err(e)
                }
            })?;

        pb.finish();

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

        if !agg.top_cluster_sizes.is_empty() {
            let arr = Array2::from_shape_fn((n_temps, 4), |(t, k)| agg.top_cluster_sizes[t][k])
                .into_pyarray(py);
            dict.set_item("top_cluster_sizes", arr)?;
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
