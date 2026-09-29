//! Shared Python sampling lifetime, progress, cancellation and array conversion.
use indicatif::{ProgressBar, ProgressStyle};
use numpy::{
    ndarray::{ArrayD, IxDyn},
    IntoPyArray,
};
use pyo3::{prelude::*, types::PyDict};
use spin_sim::spins::model::Spin;
use spin_sim::statistics::physics::{PhysicsResult, PhysicsValues};
use spin_sim::ModelRealization;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex, OnceLock, Weak,
};

static ACTIVE: Mutex<Vec<Weak<AtomicBool>>> = Mutex::new(Vec::new());
static SIGNAL: OnceLock<Result<(), String>> = OnceLock::new();
struct SamplingGuard {
    flag: Arc<AtomicBool>,
    progress: ProgressBar,
}
impl Drop for SamplingGuard {
    fn drop(&mut self) {
        ACTIVE.lock().unwrap().retain(|weak| {
            weak.upgrade()
                .is_some_and(|flag| !Arc::ptr_eq(&flag, &self.flag))
        });
        self.progress.finish_and_clear();
    }
}

pub(crate) fn execute<R: Send>(
    py: Python<'_>,
    n_sweeps: usize,
    n_disorder: usize,
    body: impl FnOnce(&AtomicBool, &(dyn Fn() + Sync)) -> Result<R, String> + Send,
) -> PyResult<R> {
    let signal = SIGNAL.get_or_init(|| {
        ctrlc::set_handler(|| {
            let active = ACTIVE.lock().unwrap();
            let mut signalled = false;
            for weak in active.iter() {
                if let Some(flag) = weak.upgrade() {
                    flag.store(true, Ordering::Relaxed);
                    signalled = true;
                }
            }
            if !signalled {
                unsafe {
                    pyo3::ffi::PyErr_SetInterrupt();
                }
            }
        })
        .map_err(|e| e.to_string())
    });
    if let Err(error) = signal {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "cannot install sampling interrupt handler: {error}"
        )));
    }
    let progress = ProgressBar::new(n_sweeps as u64);
    progress.set_style(
        ProgressStyle::with_template(
            "{msg} [{bar:40}] {pos}/{len} [{elapsed_precise} < {eta_precise}, {per_sec}]",
        )
        .unwrap()
        .progress_chars("=> "),
    );
    progress.set_message("sweeps");
    let guard = SamplingGuard {
        flag: Arc::new(AtomicBool::new(false)),
        progress,
    };
    ACTIVE.lock().unwrap().push(Arc::downgrade(&guard.flag));
    let counter = AtomicU64::new(0);
    let result = py.allow_threads(|| {
        body(&guard.flag, &|| {
            let previous = counter.fetch_add(1, Ordering::Relaxed);
            if (previous + 1).is_multiple_of(n_disorder.max(1) as u64) {
                guard.progress.inc(1);
            }
        })
    });
    if result.is_ok() && guard.flag.load(Ordering::Relaxed) {
        return Err(pyo3::exceptions::PyKeyboardInterrupt::new_err(
            "interrupted",
        ));
    }
    result.map_err(|e| {
        if e == "interrupted" {
            pyo3::exceptions::PyKeyboardInterrupt::new_err(e)
        } else {
            pyo3::exceptions::PyValueError::new_err(e)
        }
    })
}

pub(crate) fn set_array<T: numpy::Element>(
    py: Python<'_>,
    dict: &Bound<'_, PyDict>,
    name: &str,
    shape: &[usize],
    values: Vec<T>,
) -> PyResult<()> {
    let array = ArrayD::from_shape_vec(IxDyn(shape), values)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    dict.set_item(name, array.into_pyarray(py))
}
fn set_values(py: Python<'_>, dict: &Bound<'_, PyDict>, values: &PhysicsValues) -> PyResult<()> {
    for (name, rows) in values {
        let width = rows[0].len();
        let mut shape = vec![rows.len()];
        if is_vector(name) {
            shape.push(width);
        }
        set_array(
            py,
            dict,
            name,
            &shape,
            rows.iter().flatten().copied().collect(),
        )?;
    }
    Ok(())
}
fn is_vector(name: &str) -> bool {
    matches!(
        name,
        "helicity_d"
            | "helicity_i"
            | "helicity_i2"
            | "helicity_modulus"
            | "structure_factor_min"
            | "correlations"
            | "correlation_length"
            | "correlation_length_ratio"
    )
}

pub(crate) fn physics_dict<'py>(
    py: Python<'py>,
    shape: &[usize],
    temperatures: &[f64],
    results: &[&PhysicsResult],
    components: usize,
    aggregate: Option<&PhysicsValues>,
) -> PyResult<Bound<'py, PyDict>> {
    let output = PyDict::new(py);
    let computed;
    let aggregate = match aggregate {
        Some(values) => values,
        None => {
            computed =
                PhysicsResult::aggregate(results.iter().copied(), shape, temperatures, components);
            &computed
        }
    };
    set_values(py, &output, aggregate)?;
    let per = PyDict::new(py);
    let individual: Vec<_> = results
        .iter()
        .map(|r| r.values(shape, temperatures, components))
        .collect();
    for (name, rows) in aggregate {
        let mut dims = vec![results.len(), temperatures.len()];
        if is_vector(name) {
            dims.push(rows[0].len());
        }
        let data = individual
            .iter()
            .flat_map(|d| d[name].iter().flatten().copied())
            .collect();
        set_array(py, &per, name, &dims, data)?;
    }
    set_array(
        py,
        &per,
        "measurement_count",
        &[results.len()],
        results.iter().map(|r| r.count).collect(),
    )?;
    if !results[0].blocks.is_empty() {
        let blocks = PyDict::new(py);
        let sums = PyDict::new(py);
        let count = results[0].blocks.len();
        for field in &results[0].fields {
            let mut dims = vec![results.len(), count, temperatures.len()];
            if is_vector(field.name) {
                dims.push(field.width);
            }
            let data = results
                .iter()
                .flat_map(|r| {
                    r.blocks.iter().flat_map(|b| {
                        b.sums.iter().flat_map(|row| {
                            row[field.offset..field.offset + field.width]
                                .iter()
                                .copied()
                        })
                    })
                })
                .collect();
            set_array(py, &sums, field.name, &dims, data)?;
        }
        set_array(
            py,
            &blocks,
            "counts",
            &[results.len(), count],
            results
                .iter()
                .flat_map(|r| r.blocks.iter().map(|b| b.count))
                .collect(),
        )?;
        set_array(
            py,
            &blocks,
            "sweeps",
            &[results.len(), count],
            results
                .iter()
                .flat_map(|r| r.blocks.iter().map(|b| b.sweeps as u64))
                .collect(),
        )?;
        blocks.set_item("sums", sums)?;
        per.set_item("blocks", blocks)?;
    }
    output.set_item("per_disorder", per)?;
    Ok(output)
}

pub(crate) fn coupling_count(
    shape: &[usize],
    lattice: &[usize],
    neighbors: usize,
) -> PyResult<usize> {
    let expected: Vec<_> = lattice
        .iter()
        .copied()
        .chain(std::iter::once(neighbors))
        .collect();
    if shape == expected {
        return Ok(1);
    }
    if shape.len() == expected.len() + 1 && shape[1..] == expected && shape[0] > 0 {
        return Ok(shape[0]);
    }
    Err(pyo3::exceptions::PyValueError::new_err(format!(
        "couplings shape {shape:?} does not match lattice {expected:?}"
    )))
}

/// Lifetime PT edge attempts, edge acceptances and round trips of one realization.
///
/// They persist across `sample()` calls; bindings report each call's increments.
pub(crate) fn pt_counters<S: Spin>(real: &ModelRealization<S>) -> [&[u64]; 3] {
    [
        real.pt_edge_attempts(),
        real.pt_edge_acceptances(),
        real.pt_round_trips(),
    ]
}

/// Per-realization snapshot of [`pt_counters`].
pub(crate) fn pt_snapshot<S: Spin>(reals: &[ModelRealization<S>]) -> Vec<[Vec<u64>; 3]> {
    reals
        .iter()
        .map(|real| pt_counters(real).map(<[u64]>::to_vec))
        .collect()
}

/// Increments of counter `which` since `before`, concatenated over realizations.
pub(crate) fn pt_delta<S: Spin>(
    reals: &[ModelRealization<S>],
    before: &[[Vec<u64>; 3]],
    which: usize,
) -> Vec<u64> {
    reals
        .iter()
        .zip(before)
        .flat_map(|(real, before)| {
            pt_counters(real)[which]
                .iter()
                .zip(&before[which])
                .map(|(now, then)| now - then)
        })
        .collect()
}

pub(crate) fn warmup_sweeps(n: usize, ratio: f64) -> PyResult<usize> {
    if !ratio.is_finite() || !(0.0..=1.0).contains(&ratio) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "warmup_ratio must be finite and between zero and one",
        ));
    }
    Ok((n as f64 * ratio).round() as usize)
}
