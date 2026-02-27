# spin-sim

Pure-Rust Ising model Monte Carlo on periodic hypercubic lattices.

## Algorithms

- Single-spin flips (Metropolis, Gibbs)
- Swendsen-Wang cluster updates
- Wolff single-cluster updates
- Parallel tempering (replica exchange)
- Overlap cluster moves (Houdayer / Jörg / CMR) for spin glasses

Replicas are parallelized over threads with [rayon](https://crates.io/crates/rayon).

## Usage

```rust
use spin_sim::config::*;
use spin_sim::{Lattice, Realization, run_sweep_loop};

let lattice = Lattice::new(vec![16, 16]);
let temps = vec![2.0, 2.27, 2.5];
let n_replicas = 2;

// Random ±1 couplings (bimodal spin glass)
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
    overlap_cluster: None,
    autocorrelation_max_lag: None,
    sequential: false,
};

use std::sync::atomic::AtomicBool;
let interrupted = AtomicBool::new(false);
let result = run_sweep_loop(
    &lattice, &mut real, n_replicas, temps.len(), &config, &interrupted, &|| {},
).unwrap();

println!("energies: {:?}", result.energies);
```

## Python

For a batteries-included Python interface, see [`peapods`](https://pypi.org/project/peapods/).
