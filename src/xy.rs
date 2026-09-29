use crate::execution::{
    coupling_count, execute, physics_dict, pt_delta, pt_snapshot, set_array, warmup_sweeps,
};
use numpy::{PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::{prelude::*, types::PyDict};
use spin_sim::{config::*, statistics::physics::PhysicsOptions, XyConfig, XySimulation};

#[pyclass(name = "XYSimulation")]
pub(crate) struct PyXySimulation {
    model: XySimulation,
}
#[pymethods]
impl PyXySimulation {
    #[new]
    #[pyo3(signature=(lattice_shape,couplings,temperatures,n_replicas=1,occupation=None,seed=42))]
    fn new(
        lattice_shape: Vec<usize>,
        couplings: PyReadonlyArrayDyn<f64>,
        temperatures: PyReadonlyArray1<f64>,
        n_replicas: usize,
        occupation: Option<PyReadonlyArrayDyn<bool>>,
        seed: u64,
    ) -> PyResult<Self> {
        let n_disorder = coupling_count(couplings.shape(), &lattice_shape, lattice_shape.len())?;
        let values = couplings.as_slice()?;
        if values.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "couplings cannot be empty",
            ));
        }
        let n_sites = values.len() / n_disorder / lattice_shape.len().max(1);
        let masks = occupation
            .map(|m| {
                let expected: Vec<_> = std::iter::once(n_disorder)
                    .chain(lattice_shape.iter().copied())
                    .collect();
                if m.shape() != expected {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "occupation must have shape (n_disorder, *lattice_shape)",
                    ));
                }
                Ok(m.as_slice()?
                    .chunks_exact(n_sites)
                    .map(|c| c.to_vec())
                    .collect())
            })
            .transpose()?;
        let model = XySimulation::new(
            lattice_shape,
            values
                .chunks_exact(values.len() / n_disorder)
                .map(|c| c.to_vec())
                .collect(),
            temperatures.as_slice()?,
            n_replicas,
            masks,
            seed,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { model })
    }
    #[pyo3(signature=(n_sweeps,sweep_mode="metropolis",cluster_update_interval=Some(1),cluster_mode="sw",cluster_updates=1,overrelaxation_sweeps=0,pt_interval=None,pt_schedule="single_random_edge",warmup_ratio=0.25,displacements=None,vortices=false,block_size=None,autocorrelation_max_lag=None,autocorrelation_backend="ring",sequential=false))]
    #[allow(clippy::too_many_arguments)]
    fn sample<'py>(
        &mut self,
        py: Python<'py>,
        n_sweeps: usize,
        sweep_mode: &str,
        cluster_update_interval: Option<usize>,
        cluster_mode: &str,
        cluster_updates: usize,
        overrelaxation_sweeps: usize,
        pt_interval: Option<usize>,
        pt_schedule: &str,
        warmup_ratio: f64,
        displacements: Option<Vec<Vec<isize>>>,
        vortices: bool,
        block_size: Option<usize>,
        autocorrelation_max_lag: Option<usize>,
        autocorrelation_backend: &str,
        sequential: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let error = pyo3::exceptions::PyValueError::new_err;
        let mode = ClusterMode::try_from(cluster_mode).map_err(error)?;
        let config = XyConfig {
            simulation: SimConfig {
                n_sweeps,
                warmup_sweeps: warmup_sweeps(n_sweeps, warmup_ratio)?,
                sweep_mode: SweepMode::try_from(sweep_mode).map_err(error)?,
                cluster_update: cluster_update_interval.map(|interval| ClusterConfig {
                    interval,
                    mode,
                    action: ClusterAction::Update,
                    collect_stats: false,
                }),
                pt_interval,
                pt_schedule: PtSchedule::try_from(pt_schedule).map_err(error)?,
                overlap_cluster: None,
                autocorrelation_max_lag,
                autocorrelation_backend: AutocorrelationBackend::try_from(autocorrelation_backend)
                    .map_err(error)?,
                sequential,
                equilibration_diagnostic: false,
            },
            cluster_updates,
            overrelaxation_sweeps,
            physics: PhysicsOptions {
                displacements: displacements.unwrap_or_default(),
                vortices,
                block_size,
            },
        };
        // PT counters persist in the realizations; report this call's increments.
        let pt_before = pt_snapshot(&self.model.realizations);
        let result = execute(
            py,
            n_sweeps,
            self.model.realizations.len(),
            |interrupted, progress| self.model.sample(&config, interrupted, progress),
        )?;
        let nt = self.model.n_temps;
        let nd = self.model.realizations.len();
        let nr = self.model.n_replicas;
        let physics: Vec<_> = result.per_disorder.iter().map(|r| &r.physics).collect();
        let dict = physics_dict(
            py,
            &self.model.lattice.shape,
            &self.model.realizations[0].temperatures[..nt],
            &physics,
            2,
            Some(&result.values),
        )?;
        let per = dict
            .get_item("per_disorder")?
            .unwrap()
            .downcast_into::<PyDict>()?;
        for (name, which) in [("cluster_updates", false), ("visited_spins", true)] {
            let data = result
                .per_disorder
                .iter()
                .flat_map(|r| {
                    if which {
                        r.visited_spins.clone()
                    } else {
                        r.cluster_updates.clone()
                    }
                })
                .collect();
            set_array(py, &per, name, &[nd, nt], data)?;
        }
        if autocorrelation_max_lag.is_some() {
            for (name, energy) in [("energy_tau", true), ("mags2_tau", false)] {
                set_array(
                    py,
                    &per,
                    name,
                    &[nd, nt],
                    result
                        .per_disorder
                        .iter()
                        .flat_map(|r| {
                            if energy {
                                r.energy_tau.clone()
                            } else {
                                r.mags2_tau.clone()
                            }
                        })
                        .collect(),
                )?;
            }
        }
        if pt_interval.is_some() {
            let pt = PyDict::new(py);
            let delta = |which| pt_delta(&self.model.realizations, &pt_before, which);
            for (which, name) in ["edge_attempts", "edge_acceptances"]
                .into_iter()
                .enumerate()
            {
                set_array(py, &pt, name, &[nd, nt.saturating_sub(1)], delta(which))?;
            }
            set_array(py, &pt, "round_trips", &[nd, nr, nt], delta(2))?;
            per.set_item("parallel_tempering", pt)?;
        }
        Ok(dict)
    }
    #[pyo3(signature=(seed=None))]
    fn reset(&mut self, seed: Option<u64>) {
        self.model.reset(seed);
    }
    /// Spins with shape (disorder, replica, temperature slot, site, 2); tempering
    /// permutations are resolved, so `[d, r, t]` is the system currently at slot `t`.
    fn get_spins<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let n = self.model.lattice.n_spins;
        let dict = PyDict::new(py);
        set_array(
            py,
            &dict,
            "spins",
            &[
                self.model.realizations.len(),
                self.model.n_replicas,
                self.model.n_temps,
                self.model.lattice.n_spins,
                2,
            ],
            self.model
                .realizations
                .iter()
                .flat_map(|r| {
                    r.system_ids
                        .iter()
                        .flat_map(|&id| r.spins[id * n..(id + 1) * n].iter().flatten().copied())
                })
                .collect(),
        )?;
        Ok(dict.get_item("spins")?.unwrap())
    }
    fn get_system_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let dict = PyDict::new(py);
        set_array(
            py,
            &dict,
            "ids",
            &[
                self.model.realizations.len(),
                self.model.n_replicas,
                self.model.n_temps,
            ],
            self.model
                .realizations
                .iter()
                .flat_map(|r| r.system_ids.iter().map(|&v| v as u64))
                .collect(),
        )?;
        Ok(dict.get_item("ids")?.unwrap())
    }
}
