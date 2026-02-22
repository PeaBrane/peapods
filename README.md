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

The following algorithms are currently supported:

- Single-spin flips (Metropolis and Gibbs sampling)
- [Swendsen-Wang cluster updates](https://en.wikipedia.org/wiki/Swendsen%E2%80%93Wang_algorithm)
- [Wolff cluster updates](https://en.wikipedia.org/wiki/Wolff_algorithm)
- [Parallel tempering](https://en.wikipedia.org/wiki/Parallel_tempering)
- [Houdayer isoenergetic cluster move](https://arxiv.org/abs/cond-mat/0101116) (replica cluster move for spin glasses)

The following algorithms are planned:

- Cluster updates for frustrated spin systems
(e.g. [KBD algorithm](https://en.wikipedia.org/wiki/KBD_algorithm))
- [Jorg move](https://arxiv.org/abs/cond-mat/0410328)
(Houdayer + SW-style bond breaking within clusters for smaller sub-clusters)

## Quickstart

It is very easy to get started with simulating an (ensemble of) spin models.
For example, if we want to simulate an ensemble of 16 independent Ising ferromagnets
shaped 20 x 20, we can do the following:

```python
from peapods import Ising

model = Ising((20, 20), temperatures=np.linspace(1.5, 3.0, 32), n_replicas=2)
model.sample(n_sweeps=5000, sweep_mode="metropolis", cluster_update_interval=1, pt_interval=1)
```

For a more complete example, check out [example.py](example.py).

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
- matplotlib (for plotting)

## Contribution

This is an open-source project, so everyone is welcomed to contribute!

Please open an issue if you spotted a bug or suggest any feature enhancements.
Submit a pull request if appropriate.
Alternatively, contact me at yanrpei@gmail.com if you wish to be a contributor to this project.
