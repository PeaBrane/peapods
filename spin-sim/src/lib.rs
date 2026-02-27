//! Pure-Rust Ising model Monte Carlo on periodic Bravais lattices.
//!
//! # Algorithms
//!
//! | Move | Function |
//! |------|----------|
//! | Metropolis / Gibbs sweep | [`run_sweep_loop`] (`sweep_mode`) |
//! | Wolff / Swendsen-Wang | [`run_sweep_loop`] (`cluster_mode`) |
//! | Parallel tempering | [`run_sweep_loop`] (`pt_interval`) |
//! | Houdayer / Jörg / CMR | [`run_sweep_loop`] (`overlap_cluster`) |
//!
//! Replicas are parallelized over threads with [`rayon`].
//! Multiple disorder realizations can be run in parallel with [`run_sweep_parallel`].
//!
//! # Quick start
//!
//! ```
//! use spin_sim::config::*;
//! use spin_sim::{Lattice, Realization, run_sweep_loop};
//!
//! let lattice = Lattice::new(vec![16, 16]);
//! let n_spins = lattice.n_spins;
//! let n_neighbors = lattice.n_neighbors;
//! let temps = vec![2.0, 2.27, 2.5];
//!
//! // Uniform ferromagnetic couplings
//! let couplings = vec![1.0f32; n_spins * n_neighbors];
//! let mut real = Realization::new(&lattice, couplings, &temps, 2, 42);
//!
//! let config = SimConfig {
//!     n_sweeps: 5000,
//!     warmup_sweeps: 1000,
//!     sweep_mode: SweepMode::Metropolis,
//!     cluster_update: Some(ClusterConfig {
//!         interval: 1,
//!         mode: ClusterMode::Wolff,
//!         collect_csd: false,
//!     }),
//!     pt_interval: Some(1),
//!     overlap_cluster: None,
//!     autocorrelation_max_lag: None,
//! };
//!
//! use std::sync::atomic::AtomicBool;
//! let interrupted = AtomicBool::new(false);
//! let result = run_sweep_loop(
//!     &lattice, &mut real, 2, temps.len(), &config, &interrupted, &|| {},
//! ).unwrap();
//! ```
//!
//! For a Python interface, see the [`peapods`](https://pypi.org/project/peapods/) package.

pub mod config;
pub mod geometry;
pub mod simulation;
pub mod spins;
pub mod statistics;

mod clusters;
mod mcmc;
mod parallel;

pub use geometry::Lattice;
pub use simulation::{run_sweep_loop, run_sweep_parallel, Realization};
pub use statistics::SweepResult;
