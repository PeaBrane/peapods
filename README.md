# PeaPods

The goal of this project is to build a Python library for simulating spin systems with modern efficient Monte-Carlo methods.
The core simulation loop is written in Rust (via PyO3) for performance, with a thin Python wrapper for ease of use.

<div style="text-align:center">
  <img src="./docs/csd.png" alt="CSD" style="width:70%"/>
</div>

## Description

Currently, this project is at the very early stages,
and only supports simulating Ising ferromagnets and spin glasses.
Development for other spin classes (Potts, clock, and O(N) models) are planned,
including their disordered and quantum counterparts.

The following algoritms are currently supported:

- Single-spin flips (Metropolis and Gibbs sampling)
- [Swendsen-Wang cluster updates](https://en.wikipedia.org/wiki/Swendsen%E2%80%93Wang_algorithm)
- [Wolff cluster updates](https://en.wikipedia.org/wiki/Wolff_algorithm)
- [Parallel tempering](https://en.wikipedia.org/wiki/Parallel_tempering)

The following algorithms are planned:

- Cluster updates for frustrated spin systems
(e.g. [KBD algorithm](https://en.wikipedia.org/wiki/KBD_algorithm#:~:text=The%20KBD%20algorithm%20is%20an,algorithm%20more%20efficient%20in%20comparison.))
- [Replica cluster moves](https://en.wikipedia.org/wiki/Replica_cluster_move#:~:text=Replica%20cluster%20move%20in%20condensed,replicas%20instead%20of%20just%20one.)
(e.g. [Houdayer move](https://arxiv.org/abs/cond-mat/0101116),
[Jorg move](https://arxiv.org/abs/cond-mat/0410328)
)

## Quickstart

It is very easy to get started with simulating an (ensemble of) spin models.
For example, if we want to simulate an ensemble of 16 independent Ising ferromagnets
shaped 20 x 20, we can do the following:

```python
from spin_models import IsingEnsemble

ising_ensemble = IsingEnsemble(lattice_shape=(20, 20), n_ensemble=16)
ising_ensemble.sample(n_sweeps=2**14)
```

Note that code will try to start 16 parallel workers for simulation,
where each worker will maintain its own set of Ising models at different temperatures,
and simulate them for 2^14 sweeps.

For a more complete example, check out this [tutorial](tutorial.ipynb).

## Building

The Rust backend must be compiled before use. You need [maturin](https://www.maturin.rs/) and a Rust toolchain:

```bash
cd peapods_core
maturin develop --release
```

## Dependencies

- Rust toolchain (for building the core)
- maturin
- numpy
- joblib (for `IsingEnsemble`)

## Contribution

This is an open-source project, so everyone is welcomed to contribute!

Please open an issue if you spotted a bug or suggest any feature enhancements.
Submit a pull request if appropriate.
Alternatively, contact me at yanrpei@gmail.com if you wish to be a contributor to this project.
