use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use indicatif::{ProgressBar, ProgressStyle};
use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use spin_sim::config::*;
use spin_sim::{run_sweep_parallel, GraphObservationSummary, Lattice, Realization};

#[pyclass]
struct IsingSimulation {
    lattice: Lattice,
    n_replicas: usize,
    n_temps: usize,
    n_realizations: usize,
    constructor_seed: u64,
    realizations: Vec<Realization>,
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}

fn realization_seed(root: u64, realization: usize) -> u64 {
    splitmix64(root ^ splitmix64(realization as u64))
}

fn set_graph_observation<'py>(
    py: Python<'py>,
    output: &Bound<'py, PyDict>,
    name: &str,
    summaries: &[&GraphObservationSummary],
) -> PyResult<()> {
    if summaries.is_empty() {
        return Ok(());
    }

    let n_disorder = summaries.len();
    let n_temps = summaries[0].observation_count.len();
    let n_sizes = summaries[0].cluster_size_counts.first().map_or(0, Vec::len);
    let graph = PyDict::new(py);
    graph.set_item(
        "observation_count",
        Array2::from_shape_fn((n_disorder, n_temps), |(d, t)| {
            summaries[d].observation_count[t]
        })
        .into_pyarray(py),
    )?;
    graph.set_item(
        "cluster_size_counts",
        Array3::from_shape_fn((n_disorder, n_temps, n_sizes), |(d, t, size)| {
            summaries[d].cluster_size_counts[t][size]
        })
        .into_pyarray(py),
    )?;
    graph.set_item(
        "top_four_component_fractions",
        Array3::from_shape_fn((n_disorder, n_temps, 4), |(d, t, k)| {
            summaries[d].top_four_component_fractions[t][k]
        })
        .into_pyarray(py),
    )?;
    graph.set_item(
        "active_bond_density",
        Array2::from_shape_fn((n_disorder, n_temps), |(d, t)| {
            summaries[d].active_bond_density[t]
        })
        .into_pyarray(py),
    )?;
    graph.set_item(
        "large_component_count",
        Array2::from_shape_fn((n_disorder, n_temps), |(d, t)| {
            summaries[d].large_component_count[t]
        })
        .into_pyarray(py),
    )?;

    if summaries[0].winding.is_some() {
        for (name, index) in [
            ("winding_x", 0),
            ("winding_y", 1),
            ("winding_either", 2),
            ("winding_both", 3),
        ] {
            graph.set_item(
                name,
                Array2::from_shape_fn((n_disorder, n_temps), |(d, t)| {
                    summaries[d].winding.as_ref().unwrap()[t][index]
                })
                .into_pyarray(py),
            )?;
        }
    }

    output.set_item(name, graph)
}

#[pymethods]
impl IsingSimulation {
    #[new]
    #[pyo3(signature = (lattice_shape, couplings, temperatures, n_replicas=None, neighbor_offsets=None, seed=None))]
    fn new(
        lattice_shape: Vec<usize>,
        couplings: PyReadonlyArrayDyn<f32>,
        temperatures: PyReadonlyArray1<f32>,
        n_replicas: Option<usize>,
        neighbor_offsets: Option<Vec<Vec<i64>>>,
        seed: Option<u64>,
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

        let constructor_seed = seed.unwrap_or(42);

        let mut realizations = Vec::with_capacity(n_realizations);
        for r in 0..n_realizations {
            let coup_chunk = couplings_raw[r * chunk_size..(r + 1) * chunk_size].to_vec();
            let base_seed = realization_seed(constructor_seed, r);
            realizations.push(Realization::new(
                &lattice, coup_chunk, temps_raw, n_replicas, base_seed,
            ));
        }

        Ok(Self {
            lattice,
            n_replicas,
            n_temps,
            n_realizations,
            constructor_seed,
            realizations,
        })
    }

    #[pyo3(signature = (
        n_sweeps,
        sweep_mode,
        cluster_update_interval=None,
        cluster_mode=None,
        cluster_action=None,
        pt_interval=None,
        pt_schedule=None,
        overlap_cluster_update_interval=None,
        overlap_cluster_build_mode=None,
        overlap_cluster_mode=None,
        overlap_cluster_action=None,
        warmup_ratio=None,
        collect_cluster_stats=None,
        autocorrelation_max_lag=None,
        sequential=None,
        equilibration_diagnostic=None,
        snapshot_interval=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn sample<'py>(
        &mut self,
        py: Python<'py>,
        n_sweeps: usize,
        sweep_mode: &str,
        cluster_update_interval: Option<usize>,
        cluster_mode: Option<&str>,
        cluster_action: Option<&str>,
        pt_interval: Option<usize>,
        pt_schedule: Option<&str>,
        overlap_cluster_update_interval: Option<usize>,
        overlap_cluster_build_mode: Option<&str>,
        overlap_cluster_mode: Option<&str>,
        overlap_cluster_action: Option<&str>,
        warmup_ratio: Option<f64>,
        collect_cluster_stats: Option<bool>,
        autocorrelation_max_lag: Option<usize>,
        sequential: Option<bool>,
        equilibration_diagnostic: Option<bool>,
        snapshot_interval: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let warmup = warmup_ratio.unwrap_or(0.25);
        let warmup_sweeps = (n_sweeps as f64 * warmup).round() as usize;
        let collect_cluster_stats = collect_cluster_stats.unwrap_or(false);

        let sweep_mode_enum =
            SweepMode::try_from(sweep_mode).map_err(pyo3::exceptions::PyValueError::new_err)?;
        let pt_schedule = PtSchedule::try_from(pt_schedule.unwrap_or("single_random_edge"))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        let cluster_update = cluster_update_interval
            .map(|interval| {
                let mode_str = cluster_mode.unwrap_or("sw");
                let mode = ClusterMode::try_from(mode_str)
                    .map_err(pyo3::exceptions::PyValueError::new_err);
                let action = ClusterAction::try_from(cluster_action.unwrap_or("update"))
                    .map_err(pyo3::exceptions::PyValueError::new_err);
                mode.and_then(|m| {
                    action.map(|action| ClusterConfig {
                        interval,
                        mode: m,
                        action,
                        collect_stats: collect_cluster_stats || action == ClusterAction::Observe,
                    })
                })
            })
            .transpose()?;

        let overlap_cluster = overlap_cluster_update_interval
            .map(|interval| {
                let build_mode_str = overlap_cluster_build_mode.unwrap_or("houdayer");
                let oc_mode_str = overlap_cluster_mode.unwrap_or("wolff");

                let modes = spin_sim::config::parse_overlap_modes(build_mode_str)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                let oc_mode = ClusterMode::try_from(oc_mode_str)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                let action = ClusterAction::try_from(overlap_cluster_action.unwrap_or("update"))
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;

                Ok::<_, PyErr>(OverlapClusterConfig {
                    interval,
                    modes,
                    cluster_mode: oc_mode,
                    action,
                    collect_stats: collect_cluster_stats || action == ClusterAction::Observe,
                    snapshot_interval,
                })
            })
            .transpose()?;

        let config = SimConfig {
            n_sweeps,
            warmup_sweeps,
            sweep_mode: sweep_mode_enum,
            cluster_update,
            pt_interval,
            pt_schedule,
            overlap_cluster,
            autocorrelation_max_lag,
            sequential: sequential.unwrap_or(false),
            equilibration_diagnostic: equilibration_diagnostic.unwrap_or(false),
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

        let ov = agg.overlap_stats;
        if !ov.overlap.is_empty() {
            dict.set_item("overlap", Array1::from(ov.overlap).into_pyarray(py))?;
            dict.set_item("overlap2", Array1::from(ov.overlap2).into_pyarray(py))?;
            dict.set_item("overlap4", Array1::from(ov.overlap4).into_pyarray(py))?;
            dict.set_item(
                "link_overlap",
                Array1::from(ov.link_overlap).into_pyarray(py),
            )?;
            dict.set_item(
                "link_overlap2",
                Array1::from(ov.link_overlap2).into_pyarray(py),
            )?;
            dict.set_item(
                "link_overlap4",
                Array1::from(ov.link_overlap4).into_pyarray(py),
            )?;

            if !ov.histogram.is_empty() {
                let hist_py: Vec<_> = ov
                    .histogram
                    .into_iter()
                    .map(|hist| Array1::from(hist).into_pyarray(py))
                    .collect();
                dict.set_item("overlap_histogram", hist_py)?;
            }

            if !ov.ql_at_q_sum.is_empty() {
                let n_ql_temps = ov.ql_at_q_sum.len();
                let n_ql_bins = ov.ql_at_q_sum[0].len();
                let arr =
                    Array2::from_shape_fn((n_ql_temps, n_ql_bins), |(t, b)| ov.ql_at_q_sum[t][b])
                        .into_pyarray(py);
                dict.set_item("ql_at_q_sum", arr)?;

                let arr2 =
                    Array2::from_shape_fn((n_ql_temps, n_ql_bins), |(t, b)| ov.ql2_at_q_sum[t][b])
                        .into_pyarray(py);
                dict.set_item("ql2_at_q_sum", arr2)?;
            }

            if !ov.per_sample_histogram.is_empty() {
                let n_real = ov.per_sample_histogram.len();
                let n_hist_temps = ov.per_sample_histogram[0].len();
                let n_bins = ov.per_sample_histogram[0].first().map_or(0, |v| v.len());
                let arr = Array3::from_shape_fn((n_real, n_hist_temps, n_bins), |(r, t, b)| {
                    ov.per_sample_histogram[r][t][b]
                })
                .into_pyarray(py);
                dict.set_item("per_sample_overlap_histogram", arr)?;
            }

            if !ov.per_sample_ql_at_q_sum.is_empty() {
                let n_real = ov.per_sample_ql_at_q_sum.len();
                let n_ps_temps = ov.per_sample_ql_at_q_sum[0].len();
                let n_ps_bins = ov.per_sample_ql_at_q_sum[0].first().map_or(0, |v| v.len());
                let arr = Array3::from_shape_fn((n_real, n_ps_temps, n_ps_bins), |(r, t, b)| {
                    ov.per_sample_ql_at_q_sum[r][t][b]
                })
                .into_pyarray(py);
                dict.set_item("per_sample_ql_at_q_sum", arr)?;

                let arr2 = Array3::from_shape_fn((n_real, n_ps_temps, n_ps_bins), |(r, t, b)| {
                    ov.per_sample_ql2_at_q_sum[r][t][b]
                })
                .into_pyarray(py);
                dict.set_item("per_sample_ql2_at_q_sum", arr2)?;
            }
        }

        let per_disorder = PyDict::new(py);
        let cluster_observations = PyDict::new(py);
        let observation_sets = [
            (
                "fk",
                agg.per_disorder_cluster_observations
                    .iter()
                    .filter_map(|observations| observations.fk.as_ref())
                    .collect::<Vec<_>>(),
            ),
            (
                "houdayer",
                agg.per_disorder_cluster_observations
                    .iter()
                    .filter_map(|observations| observations.houdayer.as_ref())
                    .collect::<Vec<_>>(),
            ),
            (
                "jorg",
                agg.per_disorder_cluster_observations
                    .iter()
                    .filter_map(|observations| observations.jorg.as_ref())
                    .collect::<Vec<_>>(),
            ),
            (
                "cmr_blue",
                agg.per_disorder_cluster_observations
                    .iter()
                    .filter_map(|observations| observations.cmr_blue.as_ref())
                    .collect::<Vec<_>>(),
            ),
        ];
        let mut has_per_disorder = false;
        for (name, summaries) in observation_sets {
            if summaries.len() != self.n_realizations {
                continue;
            }
            set_graph_observation(py, &cluster_observations, name, &summaries)?;
            has_per_disorder = true;
        }
        if has_per_disorder {
            per_disorder.set_item("cluster_observations", cluster_observations)?;
        }

        if pt_interval.is_some() {
            let n_edges = self.n_temps.saturating_sub(1);
            let pt = PyDict::new(py);
            pt.set_item(
                "edge_attempts",
                Array2::from_shape_fn((self.n_realizations, n_edges), |(d, edge)| {
                    self.realizations[d].pt_edge_attempts()[edge]
                })
                .into_pyarray(py),
            )?;
            pt.set_item(
                "edge_acceptances",
                Array2::from_shape_fn((self.n_realizations, n_edges), |(d, edge)| {
                    self.realizations[d].pt_edge_acceptances()[edge]
                })
                .into_pyarray(py),
            )?;
            pt.set_item(
                "round_trips",
                Array3::from_shape_fn(
                    (self.n_realizations, self.n_replicas, self.n_temps),
                    |(d, replica, temp)| {
                        self.realizations[d].pt_round_trips()[replica * self.n_temps + temp]
                    },
                )
                .into_pyarray(py),
            )?;
            per_disorder.set_item("parallel_tempering", pt)?;
            has_per_disorder = true;
        }
        if has_per_disorder {
            dict.set_item("per_disorder", per_disorder)?;
        }

        if agg
            .cluster_stats
            .fk_csd
            .iter()
            .any(|h| h.iter().any(|&c| c > 0))
        {
            let csd_py: Vec<_> = agg
                .cluster_stats
                .fk_csd
                .into_iter()
                .map(|hist| Array1::from(hist).into_pyarray(py))
                .collect();
            dict.set_item("fk_csd", csd_py)?;
        }

        let has_any_ov_csd = agg
            .cluster_stats
            .overlap_csd
            .iter()
            .any(|mode| mode.iter().any(|h| h.iter().any(|&c| c > 0)));
        if has_any_ov_csd {
            let csd_py: Vec<Vec<_>> = agg
                .cluster_stats
                .overlap_csd
                .into_iter()
                .map(|mode_csd| {
                    mode_csd
                        .into_iter()
                        .map(|hist| Array1::from(hist).into_pyarray(py))
                        .collect()
                })
                .collect();
            dict.set_item("overlap_csd", csd_py)?;
        }

        let has_any_top = agg
            .cluster_stats
            .top_cluster_sizes
            .iter()
            .any(|mode| !mode.is_empty());
        if has_any_top {
            let top_py: Vec<_> = agg
                .cluster_stats
                .top_cluster_sizes
                .iter()
                .map(|mode_tops| {
                    let n_t = mode_tops.len();
                    Array2::from_shape_fn((n_t, 4), |(t, k)| mode_tops[t][k]).into_pyarray(py)
                })
                .collect();
            dict.set_item("top_cluster_sizes", top_py)?;
        }

        if !agg.diagnostics.mags2_tau.is_empty() {
            dict.set_item(
                "mags2_tau",
                Array1::from(agg.diagnostics.mags2_tau).into_pyarray(py),
            )?;
        }

        if !agg.diagnostics.overlap2_tau.is_empty() {
            dict.set_item(
                "overlap2_tau",
                Array1::from(agg.diagnostics.overlap2_tau).into_pyarray(py),
            )?;
        }

        let ckpts = &agg.diagnostics.equil_checkpoints;
        if !ckpts.is_empty() {
            let n_ckpts = ckpts.len();
            let equil_sweeps: Vec<u64> = ckpts.iter().map(|c| c.sweep as u64).collect();
            dict.set_item("equil_sweeps", Array1::from(equil_sweeps).into_pyarray(py))?;

            let equil_energy_avg =
                Array2::from_shape_fn((n_ckpts, n_temps), |(i, t)| ckpts[i].energy_avg[t])
                    .into_pyarray(py);
            dict.set_item("equil_energy_avg", equil_energy_avg)?;

            let equil_link_overlap_avg =
                Array2::from_shape_fn((n_ckpts, n_temps), |(i, t)| ckpts[i].link_overlap_avg[t])
                    .into_pyarray(py);
            dict.set_item("equil_link_overlap_avg", equil_link_overlap_avg)?;
        }

        if !agg.cluster_snapshots.is_empty() {
            let n_spins = self.lattice.n_spins;
            let snaps: Vec<Bound<'py, PyDict>> = agg
                .cluster_snapshots
                .iter()
                .map(|s| {
                    let d = PyDict::new(py);
                    d.set_item("sweep_id", s.sweep_id)?;
                    d.set_item("mode_idx", s.mode_idx)?;

                    let n_st = s.cluster_ids.len();
                    let ids = Array2::from_shape_fn((n_st, n_spins), |(t, i)| {
                        s.cluster_ids[t].get(i).copied().unwrap_or(0)
                    })
                    .into_pyarray(py);
                    d.set_item("cluster_ids", ids)?;

                    if let Some(ref blue) = s.blue_ids {
                        let blue_arr = Array2::from_shape_fn((n_st, n_spins), |(t, i)| {
                            blue[t].get(i).copied().unwrap_or(0)
                        })
                        .into_pyarray(py);
                        d.set_item("blue_ids", blue_arr)?;
                    }

                    let sp = Array3::from_shape_fn((n_st, 2, n_spins), |(t, r, i)| {
                        s.spins[t][r].get(i).copied().unwrap_or(0)
                    })
                    .into_pyarray(py);
                    d.set_item("spins", sp)?;

                    let sid = Array2::from_shape_fn((n_st, 2), |(t, r)| s.system_ids[t][r] as u64)
                        .into_pyarray(py);
                    d.set_item("system_ids", sid)?;

                    Ok(d)
                })
                .collect::<PyResult<Vec<_>>>()?;
            dict.set_item("cluster_snapshots", snaps)?;
        }

        Ok(dict)
    }

    fn get_spins<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i8>>> {
        Ok(Array1::from(self.realizations[0].spins.clone()).into_pyarray(py))
    }

    #[pyo3(signature = (seed=None))]
    fn reset(&mut self, seed: Option<u64>) {
        let base_seed = seed.unwrap_or(self.constructor_seed);
        let n_replicas = self.n_replicas;
        let n_temps = self.n_temps;
        let lattice = &self.lattice;
        for (r, real) in self.realizations.iter_mut().enumerate() {
            real.reset(lattice, n_replicas, n_temps, realization_seed(base_seed, r));
        }
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IsingSimulation>()?;
    Ok(())
}
