//! Pure-Rust Ising model Monte Carlo on periodic Bravais lattices.
//!
//! # Algorithms
//!
//! | Move | Function |
//! |------|----------|
//! | Metropolis / Gibbs sweep | [`run_sweep_loop`] (`sweep_mode`) |
//! | Wolff / Swendsen-Wang | [`run_sweep_loop`] (`cluster_mode`) |
//! | Parallel tempering | [`run_sweep_loop`] (`pt_interval`) |
//! | Houdayer / Jörg / CMR | [`run_sweep_loop`] (`houdayer_interval`, `houdayer_mode`) |
//!
//! Replicas are parallelized over threads with [`rayon`].
//!
//! # Quick start
//!
//! ```
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
//! let result = run_sweep_loop(
//!     &lattice, &mut real,
//!     2, temps.len(),
//!     5000, 1000,
//!     "metropolis",
//!     Some(1), "wolff",
//!     Some(1), None, "houdayer",
//!     false,
//!     &|| {},
//! );
//! ```
//!
//! For a Python interface, see the [`peapods`](https://pypi.org/project/peapods/) package.

pub mod geometry;
pub mod simulation;
pub mod spins;
pub mod statistics;

mod clusters;
mod mcmc;
mod parallel;

pub use geometry::Lattice;
pub use simulation::{run_sweep_loop, Realization};
pub use statistics::SweepResult;
