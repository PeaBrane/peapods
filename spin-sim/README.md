# spin-sim

Pure-Rust Ising and XY Monte Carlo on periodic lattices: hypercubic, or any Bravais
lattice given by forward neighbor offsets (`Lattice::with_offsets`).

## Algorithms

- Single-spin flips (Metropolis, Gibbs)
- Swendsen-Wang cluster updates
- Wolff single-cluster updates
- Parallel tempering (replica exchange)
- Overlap cluster moves (Houdayer / pairN / Jörg / CMR / permN / gpermN / replica Monte Carlo) for spin glasses
- Open and antiperiodic boundaries as a coupling transform (`Lattice::boundary_factors`)

Replicas are parallelized over threads with [rayon](https://crates.io/crates/rayon).

## Usage

```rust
use spin_sim::config::*;
use spin_sim::{Lattice, Realization, run_sweep_loop};

let lattice = Lattice::new(vec![16, 16]);
let temps = vec![2.0, 2.27, 2.5];
let n_replicas = 2;

// A fixed ±1 pattern (+1 along axis 0, -1 along axis 1) is unfrustrated;
// draw random signs per bond for a ±J spin glass.
let couplings: Vec<f32> = (0..lattice.n_spins * lattice.n_neighbors)
    .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
    .collect();

let mut real = Realization::new(&lattice, couplings, &temps, n_replicas, 42);

let config = SimConfig {
    n_sweeps: 10_000,
    warmup_sweeps: 1_000,
    sweep_mode: SweepMode::Metropolis,
    cluster_update: None,
    pt_interval: Some(1),
    pt_schedule: PtSchedule::FullLadder,
    overlap_cluster: None,
    autocorrelation_max_lag: None,
    autocorrelation_backend: AutocorrelationBackend::Ring,
    sequential: false,
    equilibration_diagnostic: false,
    conditional_link_overlap: true,
};

use std::sync::atomic::AtomicBool;
let interrupted = AtomicBool::new(false);
let result = run_sweep_loop(
    &lattice, &mut real, n_replicas, temps.len(), &config, &interrupted, &|| {}, 0,
).unwrap();

println!("energies: {:?}", result.energies);
```

## Upgrading from 0.2.0

The next release changes public struct literals and enums; no legacy wrapper types
are provided. To keep 0.2.0's move schedule:

- `ClusterConfig` and `OverlapClusterConfig`: add `action: ClusterAction::Update`
  (cluster graphs can now also be observed without flipping spins).
- `OverlapClusterConfig`: add `max_temperature: None`.
- `SimConfig`: add `pt_schedule: PtSchedule::SingleRandomEdge` and
  `autocorrelation_backend: AutocorrelationBackend::Ring`.
- `OverlapClusterBuildMode::Jorg` is now `Jorg(usize)`; use `Jorg(2)`. The new
  variants `Pair(usize)` and `Rmc`, and `SweepMode::None`, break exhaustive matches.

Realization seeds are now derived per system rather than `base_seed + i`, so a given
seed produces different trajectories than in 0.2.0.

Since 0.2.1:

- `SimConfig::validate` is an inherent method returning `Result<(), String>`; the
  `validator` dependency is gone. `SimConfig` has a new field
  `conditional_link_overlap` (set `true` to keep the link-overlap sums per overlap
  bin) and `XyConfig` a new field `measure_interval` (set `1`).
- Overlap modes `Perm(usize)` and `GaugedPerm(usize)` break exhaustive matches; their
  size must be between 2 and 8, as validation now enforces.
- Moments of a call that measured nothing are NaN instead of zero, and a τ of a series
  without variance is NaN instead of 0.5. Each τ has a `*_converged` flag.
- Interval events keep their phase across calls; `reset` restarts it.
- Lattices with an extent of 1 are rejected.
- `SweepResult`, `ClusterStats`, `Diagnostics` and `GraphObservationSummary` gained
  fields (per-disorder samples, energy τ, FK visit counts, overlap and replica Monte
  Carlo counters, 3D wrapping and cluster moments).
- Swendsen-Wang Houdayer, pair and Jörg moves now update single-site clusters, so
  their trajectories differ from 0.2.1.

## Python

For a batteries-included Python interface, see [`peapods`](https://pypi.org/project/peapods/).


## XY models

`XySimulation` and `XyConfig` expose periodic hypercubic XY sampling with finite
signed couplings, embedded SW/Wolff, Metropolis, Gibbs (heat bath), optional
overrelaxation and PT.
Spins and physical moments use `f64`; the driver and cluster traversal are shared
with Ising. See the [XY guide](../docs/xy.md) for conventions, dilution, block
statistics and finite-size validation, and the `XySimulation` Rust API docs.
