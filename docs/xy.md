# XY simulation

`peapods.XY` samples the zero-field classical XY Hamiltonian
\(H=-\sum_{i,\mu}J_{i\mu}s_i\cdot s_{i+\hat\mu}\), with unit two-component
spins. Couplings and physical reductions use `float64`. Periodic hypercubic
lattices in any dimension are supported, with every extent at least three.

```python
from peapods import XY

model = XY((16, 16), temperatures=[0.6, 1 / 1.1199, 2.0], seed=42)
result = model.sample(
    4096, warmup_ratio=0.25,
    collect_blocks=True,                # blocks of 128 measured sweeps
    displacements=[[1, 0], [4, 0]],
    vortices=True,                      # only for two-dimensional XY
)
print(result["energies"])               # physical H/N, normally negative
print(result["helicity_modulus"])       # temperature × direction
print(result["correlations"])           # temperature × requested displacement
model.reset()                          # replay constructor dynamics
```

## Couplings, dilution and random streams

`couplings` accepts `"ferro"`, `"bimodal"`, `"gaussian"`, or an explicit array
of shape `(*lattice_shape, n_dims)`. Element `[..., mu]` is the forward bond
along axis `mu`, including the periodic seam. An array with a leading disorder
axis supplies independent coupling realizations; its leading size determines
the batch size. `n_disorder` controls the number of generated realizations for
string coupling modes. Negative bonds are present bonds; zero bonds are absent.

`occupation` must be Boolean with shape `lattice_shape` or
`(n_disorder, *lattice_shape)`. A single mask is broadcast over disorder.
Vacant sites zero all incident bonds and contribute zero to magnetic and
spatial measurements. Local passes skip them; their internal spin vectors still
participate in the fixed-size simulation state, cluster moves and work counters.
Occupied isolated spins remain physical spins. Arrays supplied by callers are
copied at construction.

```python
import numpy as np
from peapods import XY

occupation = np.random.default_rng(12).random((8, 8)) > 0.2
model = XY((8, 8), couplings="bimodal", n_disorder=4,
           temperatures=[0.8, 1.2, 2.0], occupation=occupation, seed=17)
result = model.sample(1024, pt_interval=1, pt_schedule="full_ladder")
```

System and disorder streams have deterministic separate seeds. Increasing a
built-in disorder batch preserves its prefix. `reset(seed=...)` supplies a
one-off dynamics seed; bare `reset()` continues to use the constructor seed.
Reset clears the PT permutation and counters, retaining couplings and masks.
Sampling retains final configurations, so successive calls continue each chain;
returned moments and cluster-work counts cover the current call. PT diagnostics
are cumulative since construction/reset. Existing result arrays on the Python
object are snapshots of the last completed `sample()` call, even after reset.

## Updates and measurement clock

By default each sweep performs one uniform-angle Metropolis pass and one
embedded Swendsen–Wang update. Optional overrelaxation passes run after the local
pass and before clusters, reflecting spins about their signed local field;
zero fields are skipped. Measurements follow these updates; PT follows
measurement. `n_sweeps` includes warmup, rounded from `warmup_ratio`.

- Metropolis only: `cluster_update_interval=None`.
- Cluster only: `sweep_mode="none"`, with `cluster_mode="sw"` or `"wolff"`.
- `cluster_updates=k` performs a **fixed** number of updates per scheduled cluster
  event. `cluster_update_interval` controls events per sweep; it does not adapt
  the measurement clock to the random cluster size.
- `overrelaxation_sweeps=k` adds deterministic passes. It defaults to zero and
  must accompany Metropolis or cluster updates.
- `pt_interval` defaults to `None`. Existing `single_random_edge` and
  `full_ladder` schedules and edge/round-trip diagnostics are shared with Ising.

For a random reflection axis \(r\), let \(a_i=r\cdot s_i\). A satisfied
embedded bond has \(J_{ij}a_i a_j>0\), and is activated with probability
\(1-\exp(-2\beta J_{ij}a_i a_j)\). Thus negative bonds can connect opposite
projections. SW reflects each component independently with probability one half;
Wolff reflects its seed component. The common reflection preserves internal
relative projections. This is the symmetry-based cluster construction of
[Kent-Dobias and Sethna, Section II](https://arxiv.org/abs/1805.04019).
Frustration can make equilibration slow; support for signed couplings does not
promise efficient equilibration of large frustrated samples.

`per_disorder["cluster_updates"]` and `["visited_spins"]` have shape
`(n_disorder, n_temps)`, sum over replicas, and include warmup. A SW update visits
all `N` spins; Wolff visits its seed cluster. Neither counts only accepted spin
reflections. `sequential=True` processes replicas on each realization's current
thread. Disorder batches use the same global Rayon pool; no nested pools are
created. Set `RAYON_NUM_THREADS` before sampling to bound it. Python releases the
GIL during sampling. Unsupported options and inconsistent batches are rejected
before mutation; interruption can leave a partially advanced simulation and
returns `KeyboardInterrupt`, with no partial result published.

## Observable definitions

All normalizations use the original lattice size \(N\), including vacant
sites. Write \(M=\sum_i\eta_i s_i\), \(m=|M|/N\), \(e=H/N\), and
\(\beta=1/T\). Scalar arrays have shape `(n_temps,)`; direction and displacement
arrays retain a trailing axis, even when its length is one.

| Key | Definition |
|---|---|
| `energies`, `energies2` | \(\langle e\rangle,\langle e^2\rangle\) |
| `heat_capacity` | \(\beta^2 N(\langle e^2\rangle-\langle e\rangle^2)\) within each disorder realization, then averaged |
| `mags`, `mags2`, `mags4` | \(\langle m\rangle,\langle m^2\rangle,\langle m^4\rangle\) |
| `binder_cumulant` | \(1-\langle m^4\rangle/(2\langle m^2\rangle^2)\) |
| `structure_factor_0` | \(S(0)=N\langle m^2\rangle\) |
| `susceptibility` | Per Cartesian component: \(\beta S(0)/2\) |
| `helicity_d`, `helicity_i`, `helicity_i2` | Extensive \(\langle D_\mu\rangle,\langle I_\mu\rangle,\langle I_\mu^2\rangle\) |
| `helicity_modulus` | Physical \(Y_\mu=(\langle D_\mu\rangle-\beta\langle I_\mu^2\rangle)/N\) |
| `structure_factor_min` | \(S(k_\mu)\), \(k_\mu=2\pi/L_\mu\), summing squared Fourier amplitudes of both Cartesian spin components |
| `correlation_length` | \(\xi_\mu=\sqrt{S(0)/S(k_\mu)-1}/[2\sin(\pi/L_\mu)]\) |
| `correlation_length_ratio` | \(\xi_\mu/L_\mu\) |
| `correlations` (opt-in) | \(N^{-1}\sum_i\eta_i\eta_{i+r}\langle s_i\cdot s_{i+r}\rangle\), in requested displacement order |

Here \(D_\mu=\sum_iJ_{i\mu}s_i\cdot s_{i+\hat\mu}\) and
\(I_\mu=\sum_iJ_{i\mu}(s_i^xs_{i+\hat\mu}^y-s_i^ys_{i+\hat\mu}^x)\).
The current orientation is the forward bond orientation. Couplings retain their
signs in both expressions. The zero-field finite-system symmetry gives
\(\langle I_\mu\rangle=0\); the estimator uses \(\langle I_\mu^2\rangle\)
and retains the measured mean current as a diagnostic. Negative stiffnesses
and correlations are returned as measured.

Binder and correlation length are derived from accumulated raw moments, not
averages of instantaneous ratios. Aggregate Binder and correlation length use
equally weighted disorder-averaged moments. `per_disorder` preserves each raw
moment and derived observable, with a leading disorder axis. Undefined ratios
return `NaN`. In particular, a negative correlation-length radicand can mean
ordering away from zero momentum as well as sampling uncertainty: this is a
**uniform** correlation length, not a staggered or spin-glass correlation length.
Uniform magnetization and susceptibility are not spin-glass order parameters.

With `vortices=True`, `angle_vortex_density` averages the absolute integer
winding of principal spin-angle differences around intact elementary
plaquettes. Intact means four occupied sites and four nonzero bonds, regardless
of bond signs. `intact_plaquette_fraction` reports their fraction among the `N`
plaquettes; density is `NaN` when none are intact. This is geometric angle
winding, not frustration-adjusted vorticity or chirality, and omits circulation
around dilution holes. Both measurements have the same per-disorder retention.

The ordinary signed correlations and magnetic response can inform `xy-note`.
The note's comparison theorems retain their stated nonnegative-coupling
assumptions; the simulation's broader support does not extend those theorems.

## Blocks, errors and autocorrelation

`collect_blocks=True` enables `per_disorder["blocks"]`. `sums` maps raw moment
names to arrays `(disorder, block, temperature[, direction/displacement])`.
`counts` and `sweeps` have shape `(disorder, block)`. Counts include all replicas
per temperature; the last block can be partial. Divide summed moment sums by
summed counts, then derive nonlinear observables. Use a block jackknife for
nonlinear errors. Blocks grow in memory with the number of measured sweeps;
none are stored by default.

`autocorrelation_max_lag=k` adds per-disorder `energy_tau` and `mags2_tau`, using
the existing Sokal-window infrastructure with `float64` storage for XY.
Their time unit is one measured sweep and their input averages replicas at each
temperature. The lag is bounded by one quarter of production length (minimum
one). The default `ring` backend has bounded history; `fft` retains the series
and requires an explicit maximum lag. Neither block sizes nor these diagnostics
automatically certify equilibration, especially for frustrated systems.

## Rust and Ising compatibility

```rust
use spin_sim::{XyConfig, XySimulation};
use std::sync::atomic::AtomicBool;

let mut xy = XySimulation::new(
    vec![8, 8], vec![vec![1.0; 128]], &[0.8, 1.2], 1, None, 42,
).unwrap();
let result = xy.sample(&XyConfig::default(), &AtomicBool::new(false), &|| {}).unwrap();
assert_eq!(result.values["energies"].len(), 2);
```

`ModelRealization<S>` shares initialization, reset, replica permutations and PT
state. `Realization` remains the Ising alias; `XyRealization` uses `[f64; 2]`.
The internal cache is `-H/N` for both models. XY exposes physical `H/N`; legacy
Ising `energies` keep their positive interaction convention and `float32`
kernels. Scheduling and FK traversal are shared; Ising overlap moves and
collectors remain specialized hooks.

Ising can opt into the physical collector with `sample(collect_physics=True,
block_size=128, displacements=[[1, 0]])`, or Rust
`simulation::run_sweep_parallel_with_physics`. Results are under `physics` and
include physical energy, thermodynamics, magnetic moments, structure factors,
uniform lengths and requested correlations; they omit XY helicity and winding.
Ising Binder and susceptibility use one-component normalizations. Legacy output
keys and default measurement cost are preserved. The default Ising heat-capacity
attribute uses `energy_variance`, the disorder average of each realization's
thermal energy variance, as does `collect_physics`. Forming
`energies2 - energies**2` from the disorder-averaged legacy keys would add the
disorder variance of the thermal mean energy.

## Bounded verification

Build using the selected worktree's environment, then run:

```sh
VIRTUAL_ENV=.venv .venv/bin/maturin develop --release
RAYON_NUM_THREADS=4 .venv/bin/python validation/xy_finite_size.py
```

The script retains reference constants and enforces a five-minute subprocess
limit, excluding compilation and ordinary tests. It starts four independent
chains per size with 4,096 warmup and 16,384 measured sweeps at \(\beta=1.1199\).
It checks axis-zero \(\beta Y\), \(\xi/L\), and \(S(0)\) directly against
[Hasenbusch, Table 1](https://arxiv.org/pdf/cond-mat/0502556#page=13), with four
combined standard errors, relative errors below 1%, and uncertainty stability
from blocks of 128 and 256 sweeps. One production extension is permitted for
insufficient precision, on the same chains. Reaching the cap is inconclusive.
Low/high-temperature checks use the leading low-temperature expansion
([Maccari et al.](https://arxiv.org/abs/1811.08734)) and broad trend bounds.
Generated NPZ files, logs and JSON are inspected within an owned temporary
directory and removed on success, numerical failure, interruption or timeout;
the script verifies removal. Exit codes are 0 (pass), 1 (failure), 2 (inconclusive).

Development validation on 2026-09-12 (standard errors in parentheses):

| L | \(\beta Y_0\) | \(\xi_0/L\) | \(S(0)\) |
|---|---|---|---|
| 16 | 0.72602(131) | 0.80245(411) | 133.169(186) |
| 32 | 0.70753(137) | 0.79313(397) | 453.026(672) |

All six comparisons passed (maximum 1.36 combined standard errors); reblocking
changed standard errors by at most 3.6%. No extension was needed. At `T=0.2`,
`e=-1.89743` and `Y≈0.94777`; at `T=2`, `e=-0.54742`, geometric vortex density
was `0.19703`, and `S(0)` fell from `230.777` to `4.172`. Cubic and diluted
smokes passed. Six captured Ising baselines (Metropolis, Gibbs, SW, Wolff, full
PT and Jörg overlap) matched exactly in Rust and Python after refactoring.
Short three-repeat local/SW/PT timing medians showed no slowdown.

CLI and sweep-framework integration, external fields, custom XY offsets,
spin-glass/chirality observables and frustration-adjusted vortices are deferred.
No phase diagram, infinite-volume claim or large frustrated-system equilibration
claim is made by these finite-system checks.
